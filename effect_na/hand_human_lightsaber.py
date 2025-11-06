# -*- coding: utf-8 -*-
import cv2
import time
import math
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple
from openvino.runtime import Core, CompiledModel
from PIL import Image, ImageTk  # Tk용
from PySide6.QtGui import QImage, QPixmap  # PySide6용
import glob

# =========================
# 설정 (네 코드 그대로)
# =========================
MODEL_XML = "intel/human-pose-estimation-0005/FP32/human-pose-estimation-0005.xml"
DEVICE = "AUTO"
CONF_KPT = 0.2
CONF_WRIST = 0.3
MAX_HANDS = 4
MIRROR = True

BODY_WRIST_IDX = {'Left': 10, 'Right': 9}
BODY_SHOULDER_IDX = {'Left': 5, 'Right': 2}

POSE_PAIRS = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
)

# =========================
# 유틸 (네 코드 그대로)
# =========================
@dataclass
class Keypoint:
    x: float
    y: float
    conf: float

Keypoints = List[Keypoint]

def make_black_transparent(img_rgba, thr=30, soft_edge=5):
    assert img_rgba is not None and img_rgba.shape[2] in (3, 4)
    if img_rgba.shape[2] == 3:
        a = np.full(img_rgba.shape[:2] + (1,), 255, np.uint8)
        img_rgba = np.concatenate([img_rgba, a], axis=2)

    bgr = img_rgba[..., :3]
    alpha = img_rgba[..., 3].copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    black_mask = (gray <= thr).astype(np.uint8) * 255
    if soft_edge > 0:
        inv = 255 - black_mask
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dist = np.clip(dist / float(soft_edge), 0.0, 1.0)
        feather = (dist * 255).astype(np.uint8)
        new_alpha = np.minimum(alpha, feather)
    else:
        new_alpha = alpha
        new_alpha[black_mask == 255] = 0

    out = img_rgba.copy()
    out[..., 3] = new_alpha
    return out

def trim_transparent_area(img_rgba):
    if img_rgba is None or img_rgba.shape[2] < 4:
        return img_rgba
    alpha = img_rgba[..., 3]
    ys, xs = np.where(alpha > 10)
    if len(ys) == 0 or len(xs) == 0:
        return img_rgba
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    return img_rgba[y1:y2+1, x1:x2+1]

def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)

# =========================
# OpenVINO 포즈 (네 코드 그대로)
# =========================
class OpenVinoPose:
    def __init__(self, model_xml: str, device: str = "AUTO"):
        ie = Core()
        model = ie.read_model(model_xml)
        self.compiled: CompiledModel = ie.compile_model(model, device)
        self.input_port = self.compiled.input(0)
        self.output_port = self.compiled.output(0)
        _, _, self.in_h, self.in_w = self.input_port.shape
        print(f"[Pose] Input expects: {self.in_h}x{self.in_w}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(frame, (self.in_w, self.in_h))
        img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        return img_input

    def infer(self, frame: np.ndarray) -> np.ndarray:
        inp = self.preprocess(frame)
        res = self.compiled({self.input_port: inp})[self.output_port]
        return np.asarray(res)

    def extract_keypoints(self, output: np.ndarray, orig_w: int, orig_h: int) -> Keypoints:
        out = output.squeeze(0) if output.ndim == 4 else output  # [C,H,W]
        C, Hh, Wh = out.shape
        max_k = min(18, C)
        kpts: Keypoints = []
        for i in range(max_k):
            hm = out[i]
            _, conf, _, point = cv2.minMaxLoc(hm)
            x_hm, y_hm = point
            x = int(x_hm * orig_w / Wh)
            y = int(y_hm * orig_h / Hh)
            kpts.append(Keypoint(x, y, float(conf)))
        return kpts

# =========================
# MediaPipe Hands (네 코드 그대로)
# =========================
class MediaPipeHands:
    def __init__(self, max_hands: int = 2):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=max_hands
        )
    def process(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)
    def draw(self, frame_bgr: np.ndarray, results):
        if not results.multi_hand_landmarks:
            return []
        drawn = []
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            self.mp_drawing.draw_landmarks(
                frame_bgr, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(thickness=2)
            )
            drawn.append((label, handedness.classification[0].score, hand_lm))
        return drawn

# =========================
# 스켈레톤 & 손-손목 연결 (네 코드 그대로)
# =========================
class Visualizer:
    def __init__(self, pose_pairs=POSE_PAIRS, conf_thr=CONF_KPT):
        self.pose_pairs = pose_pairs
        self.conf_thr = conf_thr

    def draw_keypoints(self, img: np.ndarray, kpts: Keypoints, color=(0,255,0)):
        for kp in kpts:
            if kp.conf > self.conf_thr:
                cv2.circle(img, (int(kp.x), int(kp.y)), 3, color, -1)

    def draw_skeleton(self, img: np.ndarray, kpts: Keypoints, color=(0,255,0)):
        self.draw_keypoints(img, kpts, color)
        for a, b in self.pose_pairs:
            if a < len(kpts) and b < len(kpts):
                ka, kb = kpts[a], kpts[b]
                if ka.conf > self.conf_thr and kb.conf > self.conf_thr:
                    cv2.line(img, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)), color, 2)

    def draw_hand_labels_and_attach(self, img: np.ndarray, hands_drawn, img_w, img_h, body_kpts: Keypoints):
        if not body_kpts:
            return
        wrists = {}
        for side, idx in BODY_WRIST_IDX.items():
            if idx < len(body_kpts) and body_kpts[idx].conf > self.conf_thr:
                wrists[side] = (int(body_kpts[idx].x), int(body_kpts[idx].y))

        mid_x = None
        ls, rs = BODY_SHOULDER_IDX['Left'], BODY_SHOULDER_IDX['Right']
        if ls < len(body_kpts) and rs < len(body_kpts):
            if body_kpts[ls].conf > self.conf_thr and body_kpts[rs].conf > self.conf_thr:
                mid_x = 0.5 * (body_kpts[ls].x + body_kpts[rs].x)

        d_thr = max(img_w, img_h) * 0.2

        for label, score, hand_lm in hands_drawn:
            hlabel = ('Right' if label == 'Left' else 'Left') if MIRROR else label
            w0 = hand_lm.landmark[0]
            hx, hy = int(w0.x * img_w), int(w0.y * img_h)

            best_pt, best_d2 = None, 1e18
            if hlabel in wrists:
                wx, wy = wrists[hlabel]
                d2 = (wx - hx)**2 + (wy - hy)**2
                best_pt, best_d2 = (wx, wy), d2
            if best_pt is None and wrists:
                for _, (wx, wy) in wrists.items():
                    d2 = (wx - hx)**2 + (wy - hy)**2
                    if d2 < best_d2:
                        best_pt, best_d2 = (wx, wy), d2
            if best_pt is not None and best_d2 <= (d_thr**2):
                cv2.line(img, (hx, hy), best_pt, (255, 255, 255), 2)
                disp = (label if not MIRROR else ('Left' if label == 'Right' else 'Right'))
                cv2.putText(img, f"{disp}", (hx - 30, hy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if disp == "Left" else (255,0,0), 2)

# =========================
# PNG 오버레이 (네가 메인에서 쓰던 "하단 중앙 정렬" 버전 그대로)
# =========================
def overlay_png(img, png, center_x, center_y):
    """
    img: 배경 (BGR)
    png: BGRA
    center_x, center_y: PNG의 하단 중앙이 맞춰질 기준점
    """
    ph, pw = png.shape[:2]
    x1 = int(center_x - pw / 2)
    y2 = int(center_y)
    x2 = x1 + pw
    y1 = int(y2 - ph)

    h, w = img.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return img

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    roi = img[y1c:y2c, x1c:x2c]
    png_crop = png[(y1c - y1):(y2c - y1), (x1c - x1):(x2c - x1)]

    if png_crop.shape[2] == 4:
        b, g, r, a = cv2.split(png_crop)
        overlay = cv2.merge((b, g, r))
        mask = cv2.merge((a, a, a)) / 255.0
    else:
        overlay = png_crop
        mask = np.ones_like(overlay, dtype=np.float32)

    img[y1c:y2c, x1c:x2c] = (roi * (1 - mask) + overlay * mask).astype(np.uint8)
    return img

# =========================
# 공통 내부 로직 (한 프레임 처리) — 네 main() 로직 그대로
# =========================
def _process_one_frame(frame, pose, hands, viz, overlay_frames, frame_idx_ref, mirror=MIRROR):
    if mirror:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # 포즈 추론 & 스켈레톤
    out = pose.infer(frame)
    body_kpts = pose.extract_keypoints(out, w, h)
    viz.draw_skeleton(frame, body_kpts, (0,255,0))

    # 손 추정 & 오른손 선택 (MIRROR 고려)
    results = hands.process(frame)
    hands_drawn = hands.draw(frame, results)

    right_hand_kpts = None
    for label, score, hand_lm in hands_drawn:
        is_right = (label == "Right") if not mirror else (label == "Left")
        if is_right:
            right_hand_kpts = [Keypoint(lm.x * w, lm.y * h, 1.0) for lm in hand_lm.landmark]
            break

    # 라이트세이버 프레임 오버레이
    if right_hand_kpts and len(right_hand_kpts) > 5 and overlay_frames:
        x, y = int(right_hand_kpts[5].x), int(right_hand_kpts[5].y)
        base = overlay_frames[frame_idx_ref[0] % len(overlay_frames)]
        frame = overlay_png(frame, base, x, y)
        frame_idx_ref[0] = (frame_idx_ref[0] + 1) % len(overlay_frames)

    return frame

# =========================
# ✅ PySide6용: run_lightsaber
# =========================
def run_lightsaber(video_label, cam_index=0, mirror=MIRROR):
    pose = OpenVinoPose(MODEL_XML, DEVICE)
    hands = MediaPipeHands(MAX_HANDS)
    viz = Visualizer()

    # PNG 프레임 로드 (네 로직을 그대로)
    frame_files = sorted(glob.glob("datasets/LightsaberPNG/ezgif-frame-*.png"))
    if not frame_files:
        raise FileNotFoundError("⚠️ PNG 프레임 파일을 찾을 수 없습니다.")
    overlay_frames = []
    for f in frame_files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = make_black_transparent(img, thr=30, soft_edge=5)
        overlay_frames.append(img)

    cap = cv2.VideoCapture(cam_index)
    frame_idx_ref = [0]  # 리스트로 캡처(파이썬 클로저에서 변경 가능하도록)

    def update():
        ok, frame = cap.read()
        if not ok:
            return
        frame = _process_one_frame(frame, pose, hands, viz, overlay_frames, frame_idx_ref, mirror=mirror)

        # PySide6 라벨에 출력
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        video_label.setPixmap(QPixmap.fromImage(qimg))

    return update

# =========================
# ✅ Tkinter용: run_lightsaber_tk
# =========================
def run_lightsaber_tk(video_label, cam_index=0, mirror=MIRROR):
    pose = OpenVinoPose(MODEL_XML, DEVICE)
    hands = MediaPipeHands(MAX_HANDS)
    viz = Visualizer()

    frame_files = sorted(glob.glob("datasets/LightsaberPNG/ezgif-frame-*.png"))
    if not frame_files:
        raise FileNotFoundError("⚠️ PNG 프레임 파일을 찾을 수 없습니다.")
    overlay_frames = []
    for f in frame_files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = make_black_transparent(img, thr=30, soft_edge=5)
        overlay_frames.append(img)

    cap = cv2.VideoCapture(cam_index)
    frame_idx_ref = [0]

    def update():
        ok, frame = cap.read()
        if not ok:
            return
        frame = _process_one_frame(frame, pose, hands, viz, overlay_frames, frame_idx_ref, mirror=mirror)

        # Tk 라벨에 출력
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    return update

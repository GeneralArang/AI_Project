# hand_human_lightsaber.py
# -*- coding: utf-8 -*-
import cv2
import time
import math
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple
from openvino.runtime import Core, CompiledModel
from PIL import Image, ImageTk
import glob

# =========================
# 설정 (처음 코드 유지)
# =========================
MODEL_XML = "intel/human-pose-estimation-0005/FP32/human-pose-estimation-0005.xml"
DEVICE = "AUTO"          # "CPU", "GPU", "AUTO" 등
CONF_KPT = 0.2           # 전신 키포인트 표시 임계값
CONF_WRIST = 0.3         # 손목 박스 생성 임계값
MAX_HANDS = 4            # MediaPipe 최대 손 수
MIRROR_DEFAULT = True    # 기본값 (함수 인자로도 바꿀 수 있게)

BODY_WRIST_IDX = {'Left': 10, 'Right': 9}
BODY_SHOULDER_IDX = {'Left': 5, 'Right': 2}

POSE_PAIRS = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
)

# =========================
# 데이터 구조
# =========================
@dataclass
class Keypoint:
    x: float
    y: float
    conf: float

Keypoints = List[Keypoint]

# =========================
# 유틸 (처음 코드 유지)
# =========================
def make_black_transparent(img_rgba, thr=30, soft_edge=5):
    """
    검은 배경을 투명화 + Feathering
    """
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
# 전신 포즈 추정 모듈(OpenVINO)  (처음 코드 유지)
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
        max_k = min(18, C)  # 0~17 관절만 사용
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
# 손 추정(미디어파이프) 모듈 (처음 코드 유지)
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
# 시각화(스켈레톤 & 손-손목 연결)  (처음 코드 유지)
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

    def draw_hand_labels_and_attach(self, img: np.ndarray, hands_drawn, img_w, img_h, body_kpts: Keypoints, mirror: bool):
        if not body_kpts:
            return
        # 전신 손목 좌표 수집
        wrists = {}
        for side, idx in BODY_WRIST_IDX.items():
            if idx < len(body_kpts) and body_kpts[idx].conf > self.conf_thr:
                wrists[side] = (int(body_kpts[idx].x), int(body_kpts[idx].y))

        # 몸 중심선(어깨 평균 x)
        mid_x = None
        ls, rs = BODY_SHOULDER_IDX['Left'], BODY_SHOULDER_IDX['Right']
        if ls < len(body_kpts) and rs < len(body_kpts):
            if body_kpts[ls].conf > self.conf_thr and body_kpts[rs].conf > self.conf_thr:
                mid_x = 0.5 * (body_kpts[ls].x + body_kpts[rs].x)

        d_thr = max(img_w, img_h) * 0.2  # 최대 허용 거리

        for label, score, hand_lm in hands_drawn:
            hlabel = ('Right' if label == 'Left' else 'Left') if mirror else label
            w0 = hand_lm.landmark[0]
            hx, hy = int(w0.x * img_w), int(w0.y * img_h)

            best_pt, best_d2 = None, 1e18
            # 1) 라벨 일치 우선
            if hlabel in wrists:
                wx, wy = wrists[hlabel]
                d2 = (wx - hx)**2 + (wy - hy)**2
                best_pt, best_d2 = (wx, wy), d2
            # 2) 폴백: 다른쪽
            if best_pt is None and wrists:
                for _, (wx, wy) in wrists.items():
                    d2 = (wx - hx)**2 + (wy - hy)**2
                    if d2 < best_d2:
                        best_pt, best_d2 = (wx, wy), d2
            # 3) 거리 임계값
            if best_pt is not None and best_d2 <= (d_thr**2):
                cv2.line(img, (hx, hy), best_pt, (255, 255, 255), 2)
                disp = (label if not mirror else ('Left' if label == 'Right' else 'Right'))
                cv2.putText(img, f"{disp}", (hx - 30, hy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if disp == "Left" else (255,0,0), 2)

# =========================
# PNG 오버레이 (하단 중앙 정렬) — 처음 코드 유지
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
# PNG 오버레이 (회전, bottom-center = p5) — 처음 코드 유지
# =========================
def overlay_png_rotate_bottom_center(img, png, p5, p17):
    """
    회전 중심: 원본 PNG의 하단 중앙 (bottom-center)
    p5: 손의 5번 좌표와 이미지의 하단 중앙을 일치
    p17: 방향 계산 (5→17)
    - 회전 시 이미지가 잘리지 않도록 캔버스를 확장
    - 회전 후에도 bottom-center가 정확히 p5에 맞게 유지
    """
    ph0, pw0 = png.shape[:2]

    # ===== 방향 벡터 및 각도 계산 =====
    dx, dy = p17[0] - p5[0], p17[1] - p5[1]
    angle = math.degrees(math.atan2(dy, dx))
    rot_deg = -(angle - 90)

    # ===== 캔버스 확장 (잘림 방지용 여백 추가) =====
    margin = max(ph0, pw0)
    pad = cv2.copyMakeBorder(png, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
    ph, pw = pad.shape[:2]

    # ===== 새 앵커 (확장된 하단 중앙) =====
    anchor = (pw / 2, ph - margin)

    # ===== 회전 후 bounding box 크기 계산 =====
    rad = math.radians(rot_deg)
    c, s = abs(math.cos(rad)), abs(math.sin(rad))
    new_w = int(pw * c + ph * s)
    new_h = int(pw * s + ph * c)

    # ===== 회전 행렬 + 중심 보정 =====
    M = cv2.getRotationMatrix2D(anchor, rot_deg, 1.0)
    M[0, 2] += (new_w / 2) - anchor[0]
    M[1, 2] += (new_h / 2) - anchor[1]

    # ===== 회전 수행 =====
    rotated = cv2.warpAffine(pad, M, (new_w, new_h),
                             flags=cv2.INTER_AREA,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    # ===== 회전 후 앵커 좌표 계산 =====
    ax = M[0, 0] * anchor[0] + M[0, 1] * anchor[1] + M[0, 2]
    ay = M[1, 0] * anchor[0] + M[1, 1] * anchor[1] + M[1, 2]

    # ===== p5와 (ax, ay)를 일치시켜 오버레이 =====
    x1 = int(p5[0] - ax)
    y1 = int(p5[1] - ay)
    x2, y2 = x1 + new_w, y1 + new_h

    # ===== 화면 클램프 =====
    H, W = img.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return img

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    png_crop = rotated[(y1c - y1):(y2c - y1), (x1c - x1):(x2c - x1)]
    roi = img[y1c:y2c, x1c:x2c]

    # ===== 알파 합성 =====
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
# 내부 상태 컨테이너
# =========================
@dataclass
class SaberState:
    frame_idx: int = 0
    scale: float = 0.45
    min_scale: float = 0.25
    max_scale: float = 3.0
    scale_step: float = 0.1
    mirror: bool = MIRROR_DEFAULT

# =========================
# 한 프레임 처리 (처음 코드의 흐름 그대로)
# =========================
def _process_one_frame(frame, pose, hands, viz, overlay_frames, state: SaberState):
    if state.mirror:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # ---- 전신 포즈 추론 & 스켈레톤 ----
    out = pose.infer(frame)
    body_kpts = pose.extract_keypoints(out, w, h)
    viz.draw_skeleton(frame, body_kpts, (0, 255, 0))

    # ---- 손 추정 & 손목 라벨/연결선 ----
    results = hands.process(frame)
    hands_drawn = hands.draw(frame, results)
    viz.draw_hand_labels_and_attach(frame, hands_drawn, w, h, body_kpts, state.mirror)

    # ---- 오른손 선택 (MIRROR 고려) ----
    right_hand_kpts = None
    for label, score, hand_lm in hands_drawn:
        is_right = (label == "Right") if not state.mirror else (label == "Left")
        if is_right:
            right_hand_kpts = [Keypoint(lm.x * w, lm.y * h, 1.0) for lm in hand_lm.landmark]
            break

    # ---- 라이트세이버 오버레이 (p5→p17 이용 회전, 하단 중앙 정렬) ----
    if right_hand_kpts and len(right_hand_kpts) > 17 and overlay_frames:
        p5 = (int(right_hand_kpts[5].x), int(right_hand_kpts[5].y))
        p17 = (int(right_hand_kpts[17].x), int(right_hand_kpts[17].y))

        # 디버그: 방향선 및 포인트
        cv2.line(frame, p5, p17, (0, 255, 255), 2)
        cv2.circle(frame, p5, 5, (0, 0, 255), -1)
        cv2.circle(frame, p17, 5, (255, 0, 0), -1)

        # 연장선 시각화(처음 코드 유지)
        dx, dy = p17[0] - p5[0], p17[1] - p5[1]
        length = math.hypot(dx, dy) if (dx or dy) else 1.0
        dir_x, dir_y = dx / length, dy / length
        line_len = 200
        end_x = int(p5[0] - dir_x * line_len)
        end_y = int(p5[1] - dir_y * line_len)
        cv2.line(frame, p5, (end_x, end_y), (0, 255, 0), 2)

        # 스케일 적용
        base = overlay_frames[state.frame_idx]
        if state.scale != 1.0:
            new_w = max(1, int(base.shape[1] * state.scale))
            new_h = max(1, int(base.shape[0] * state.scale))
            overlay_img = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            overlay_img = base

        # 회전+하단중앙 앵커 버전 (처음 코드 함수)
        frame = overlay_png_rotate_bottom_center(frame, overlay_img, p5, p17)
        cv2.putText(frame, "bottom_center=p5", (p5[0] + 10, p5[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 프레임 인덱스 증가 (애니메이션)
        state.frame_idx = (state.frame_idx + 1) % len(overlay_frames)

    return frame

# =========================
# 초기화 (모델/PNG 로드)
# =========================
def init_lightsaber(png_glob="datasets/LightsaberPNG/ezgif-frame-*.png"):
    pose = OpenVinoPose(MODEL_XML, DEVICE)
    hands = MediaPipeHands(MAX_HANDS)
    viz = Visualizer()

    frame_files = sorted(glob.glob(png_glob))
    if not frame_files:
        raise FileNotFoundError("⚠️ PNG 프레임 파일을 찾을 수 없습니다.")
    overlay_frames = []
    for f in frame_files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = make_black_transparent(img, thr=30, soft_edge=5)
        overlay_frames.append(img)
    print(f"🔹 {len(overlay_frames)}개의 PNG 프레임 로드 완료.")
    return pose, hands, viz, overlay_frames

# =========================
# OpenCV 단독 실행 루프 (키로 스케일 조정/FPS 표기 유지)
# =========================
def run_lightsaber_opencv(cam_index=0, mirror=MIRROR_DEFAULT):
    pose, hands, viz, overlay_frames = init_lightsaber()
    cap = cv2.VideoCapture(cam_index)
    state = SaberState(mirror=mirror)
    prev_t = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = _process_one_frame(frame, pose, hands, viz, overlay_frames, state)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_t) if prev_t else 0.0
        prev_t = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Full Body + Hand Pose + Lightsaber Animation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('-'), ord('_')):  # 축소
            state.scale = max(state.min_scale, round(state.scale - state.scale_step, 2))
            print(f"🔻 이미지 축소: scale={state.scale}")
        elif key in (ord('='), ord('+')):  # 확대
            state.scale = min(state.max_scale, round(state.scale + state.scale_step, 2))
            print(f"🔺 이미지 확대: scale={state.scale}")

    cap.release()
    cv2.destroyAllWindows()

# =========================
# Tkinter용 업데이트 콜백 (GUI에서 after로 호출)
# =========================
def run_lightsaber_tk(video_label, cam_index=0, mirror=MIRROR_DEFAULT):
    pose, hands, viz, overlay_frames = init_lightsaber()
    cap = cv2.VideoCapture(cam_index)
    state = SaberState(mirror=mirror)
    prev_t = [0.0]  # 클로저로 FPS 사용

    def update():
        ok, frame = cap.read()
        if not ok:
            return
        frame = _process_one_frame(frame, pose, hands, viz, overlay_frames, state)

        # FPS on frame (선택)
        now = time.time()
        fps = 1.0 / (now - prev_t[0]) if prev_t[0] else 0.0
        prev_t[0] = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Tk 라벨에 출력
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    return update

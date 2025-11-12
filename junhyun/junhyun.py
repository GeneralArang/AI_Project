# -*- coding: utf-8 -*-
import cv2
import glob
import time
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from openvino.runtime import Core, CompiledModel
from PIL import Image, ImageTk

# =========================
# 설정
# =========================
MODEL_XML = "intel/human-pose-estimation-0005/FP32/human-pose-estimation-0005.xml"
DEVICE = "AUTO"
CONF_KPT = 0.2
MAX_HANDS = 4
MIRROR = False

ANIM_DIR_GLOB = "animation/*.png"
ANIM_SIZE = (160, 160)
ANIM_SPEED = 2

POSE_PAIRS = (
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6)
)

@dataclass
class Keypoint:
    x: float
    y: float
    conf: float


# =========================
# PNG 애니메이션 로드
# =========================
ANIM_FRAMES = []
for file in sorted(glob.glob(ANIM_DIR_GLOB)):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.resize(img, ANIM_SIZE)
        ANIM_FRAMES.append(img)
if not ANIM_FRAMES:
    raise RuntimeError("[ERROR] animation 폴더에 PNG가 없습니다.")


# =========================
# 안전한 오버레이 함수
# =========================
def overlay_energy(img, energy, cx, cy):
    ph, pw = energy.shape[:2]
    h, w = img.shape[:2]

    x1 = int(cx - pw // 2)
    y1 = int(cy - ph // 2)
    x2 = x1 + pw
    y2 = y1 + ph

    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(w, x2)
    y2_clip = min(h, y2)
    if x2_clip <= x1_clip or y2_clip <= y1_clip:
        return img

    crop_x1 = x1_clip - x1
    crop_y1 = y1_clip - y1
    crop_x2 = crop_x1 + (x2_clip - x1_clip)
    crop_y2 = crop_y1 + (y2_clip - y1_clip)

    overlay_crop = energy[crop_y1:crop_y2, crop_x1:crop_x2]
    roi = img[y1_clip:y2_clip, x1_clip:x2_clip]

    h_min = min(overlay_crop.shape[0], roi.shape[0])
    w_min = min(overlay_crop.shape[1], roi.shape[1])
    overlay_crop = overlay_crop[:h_min, :w_min]
    roi = roi[:h_min, :w_min]

    overlay = overlay_crop.astype(float) / 255.0
    gray = cv2.cvtColor(overlay_crop, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(gray / 255.0, (7, 7), 0)
    mask = np.dstack([mask, mask, mask])

    roi_float = roi.astype(float) / 255.0
    blended = roi_float * (1 - mask) + overlay * mask
    img[y1_clip:y1_clip + h_min, x1_clip:x1_clip + w_min] = (blended * 255).astype(np.uint8)
    return img


# =========================
# OpenVINO 포즈 추정
# =========================
class OpenVinoPose:
    def __init__(self):
        ie = Core()
        model = ie.read_model(MODEL_XML)
        self.compiled: CompiledModel = ie.compile_model(model, DEVICE)
        self.input_port = self.compiled.input(0)
        self.output_port = self.compiled.output(0)
        _, _, self.in_h, self.in_w = self.input_port.shape

    def infer(self, frame):
        img = cv2.resize(frame, (self.in_w, self.in_h))
        inp = img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        res = self.compiled({self.input_port: inp})[self.output_port]
        return np.asarray(res)

    def extract_keypoints(self, output, w, h):
        out = output.squeeze(0)
        C, Hh, Wh = out.shape
        kpts = []
        for i in range(min(18, C)):
            hm = out[i]
            _, conf, _, pt = cv2.minMaxLoc(hm)
            x = int(pt[0] * w / Wh)
            y = int(pt[1] * h / Hh)
            kpts.append(Keypoint(x, y, float(conf)))
        return kpts


# =========================
# MediaPipe Hands
# =========================
class MediaPipeHands:
    def __init__(self):
        self.mp = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils
        self.hands = self.mp.Hands(max_num_hands=MAX_HANDS,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

    def process(self, frame):
        return self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# =========================
# 제스처 이펙트 (빠른 fade-in + 짧은 유지 + 3초 쿨타임)
# =========================
class DualEyeGestureSkill:
    def __init__(self):
        self.active = False
        self.done = False
        self.frame_idx = 0
        self.cooldown = 0
        self.threshold = 60

        # 전체 타임라인
        self.expand_duration = 30   # 확산 (1초)
        self.hold_duration = 15     # 유지 (0.5초)
        self.fade_duration = 15     # fade-out (0.5초)
        self.total_duration = self.expand_duration + self.hold_duration + self.fade_duration
        self.center = None

        # fade-in (0.4초 정도)
        self.fade_in_duration = 12

    def check_trigger(self, kpts, hands_data, w, h):
        if len(kpts) < 3:
            return False, None

        right_eye = (kpts[1].x, kpts[1].y)
        left_eye = (kpts[2].x, kpts[2].y)
        center_x = int((right_eye[0] + left_eye[0]) / 2)
        center_y = int((right_eye[1] + left_eye[1]) / 2)

        right_middle, left_middle = None, None
        for lm, hd in hands_data:
            label = hd.classification[0].label.lower().strip()
            mx = int(lm.landmark[12].x * w)
            my = int(lm.landmark[12].y * h)
            if label == "right":
                right_middle = (mx, my)
            elif label == "left":
                left_middle = (mx, my)

        if right_middle and left_middle:
            d_right = ((right_eye[0] - right_middle[0]) ** 2 + (right_eye[1] - right_middle[1]) ** 2) ** 0.5
            d_left = ((left_eye[0] - left_middle[0]) ** 2 + (left_eye[1] - left_middle[1]) ** 2) ** 0.5
            if d_right < self.threshold and d_left < self.threshold:
                return True, (center_x, center_y)
        return False, None

    def update(self, frame, kpts, hands_data, w, h):
        if self.cooldown > 0:
            self.cooldown -= 1

        if self.done and self.cooldown > 0:
            return

        triggered, center = self.check_trigger(kpts, hands_data, w, h)
        if triggered and not self.active and self.cooldown == 0:
            print("🔥 이펙트 발동!")
            self.active = True
            self.done = False
            self.frame_idx = 0
            self.center = center
            self.cooldown = 90  # 3초 쿨타임

        if self.active and self.center:
            cx, cy = self.center
            progress = self.frame_idx

            # 1️⃣ 확산 단계
            if progress < self.expand_duration:
                ratio = progress / self.expand_duration
                scale = 1.0 + (ratio ** 2.0) * 25.0
                png = ANIM_FRAMES[min(self.frame_idx // ANIM_SPEED, len(ANIM_FRAMES) - 1)]
                new_w = int(png.shape[1] * scale)
                new_h = int(png.shape[0] * scale)
                scaled = cv2.resize(png, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                overlay_energy(frame, scaled, cx, cy)

                # 빠른 fade-in (0.4초)
                if progress > self.expand_duration - self.fade_in_duration:
                    fade_ratio = (progress - (self.expand_duration - self.fade_in_duration)) / self.fade_in_duration
                    fade_ratio = min(fade_ratio, 1.0)
                    white = np.ones_like(frame, dtype=np.uint8) * 255
                    frame[:] = cv2.addWeighted(frame, 1 - fade_ratio, white, fade_ratio, 0)

            # 2️⃣ 유지 (0.5초)
            elif progress < self.expand_duration + self.hold_duration:
                frame[:] = np.ones_like(frame, dtype=np.uint8) * 255

            # 3️⃣ fade-out (0.5초)
            elif progress < self.total_duration:
                fade_ratio = (progress - self.expand_duration - self.hold_duration) / self.fade_duration
                fade_ratio = min(fade_ratio, 1.0)
                white = np.ones_like(frame, dtype=np.uint8) * 255
                frame[:] = cv2.addWeighted(white, 1 - fade_ratio, frame, fade_ratio, 0)

            self.frame_idx += 1
            if self.frame_idx >= self.total_duration:
                print("💥 이펙트 종료")
                self.active = False
                self.done = True


# =========================
# Tkinter 실행 루프
# =========================
def run_energy_skill_tk(video_label, cam_index=0, mirror=False):
    pose = OpenVinoPose()
    hands = MediaPipeHands()
    skill = DualEyeGestureSkill()

    cap = cv2.VideoCapture(cam_index)
    print("[INFO] 카메라 초기화 중...")
    time.sleep(1.0)
    if not cap.isOpened():
        print(f"[ERROR] 카메라({cam_index}) 열기 실패")
        return lambda: None
    print(f"[INFO] 카메라({cam_index}) 연결 성공")

    def update():
        ok, frame = cap.read()
        if not ok:
            print("[WARNING] 카메라 프레임 읽기 실패")
            return

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        out = pose.infer(frame)
        kpts = pose.extract_keypoints(out, w, h)

        for a, b in POSE_PAIRS:
            if a < len(kpts) and b < len(kpts):
                ka, kb = kpts[a], kpts[b]
                if ka.conf > CONF_KPT and kb.conf > CONF_KPT:
                    cv2.line(frame, (ka.x, ka.y), (kb.x, kb.y), (0, 255, 0), 2)

        res = hands.process(frame)
        hands_data = []
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                hands.drawer.draw_landmarks(
                    frame, lm,
                    hands.mp.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
                hands_data.append((lm, hd))

        skill.update(frame, kpts, hands_data, w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    return update

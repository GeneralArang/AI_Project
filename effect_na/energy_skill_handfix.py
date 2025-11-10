# -*- coding: utf-8 -*-
import cv2
import glob
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from openvino.runtime import Core, CompiledModel
from PySide6.QtGui import QImage, QPixmap
from PIL import Image, ImageTk

# =========================
# 설정
# =========================
MODEL_XML = "intel/human-pose-estimation-0005/FP32/human-pose-estimation-0005.xml"
DEVICE = "AUTO"
CONF_KPT = 0.2
MAX_HANDS = 4

ANIM_DIR_GLOB = "hand_anim/eff_1PNG/*.png"
ANIM_SIZE = (160, 160)
ANIM_SPEED = 2
DEBOUNCE_N = 2

POSE_PAIRS = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
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
    raise RuntimeError("[ERROR] PNG 애니메이션 없음")

# =========================
# PNG 오버레이
# =========================
def overlay_energy(img, energy, cx, cy):
    ph, pw = energy.shape[:2]
    x1 = int(cx - pw//2)
    y1 = int(cy - ph//2)
    x2 = x1 + pw
    y2 = y1 + ph

    h, w = img.shape[:2]
    if x1<0 or y1<0 or x2>w or y2>h:
        return img

    overlay = energy.astype(float)/255.0
    gray = cv2.cvtColor(energy, cv2.COLOR_BGR2GRAY) / 255.0
    mask = cv2.GaussianBlur(gray, (7,7),0)
    mask = np.dstack([mask,mask,mask])

    roi = img[y1:y2, x1:x2].astype(float)/255.0
    blended = roi*(1-mask) + overlay*mask
    img[y1:y2, x1:x2] = (blended*255).astype(np.uint8)
    return img

# =========================
# OpenVINO Pose
# =========================
class OpenVinoPose:
    def __init__(self):
        ie = Core()
        model = ie.read_model(MODEL_XML)
        self.compiled:CompiledModel = ie.compile_model(model, DEVICE)
        self.input_port = self.compiled.input(0)
        self.output_port = self.compiled.output(0)
        _,_,self.in_h,self.in_w = self.input_port.shape

    def infer(self, frame):
        img = cv2.resize(frame,(self.in_w,self.in_h))
        inp = img.transpose(2,0,1)[np.newaxis,:].astype(np.float32)
        return np.asarray(self.compiled({self.input_port: inp})[self.output_port])

    def extract_keypoints(self, out, w, h):
        out = out.squeeze(0)
        C, Hh, Wh = out.shape
        kpts=[]
        for i in range(min(18,C)):
            hm = out[i]
            _, conf, _, pt = cv2.minMaxLoc(hm)
            x = int(pt[0]*w/Wh)
            y = int(pt[1]*h/Hh)
            kpts.append(Keypoint(x,y,float(conf)))
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

# 손바닥 중심
def get_palm_center(lm, w, h):
    idxs=[2,5,9,13,17]
    xs=[lm.landmark[i].x*w for i in idxs]
    ys=[lm.landmark[i].y*h for i in idxs]
    return sum(xs)/5.0, sum(ys)/5.0

# =========================
# Palm 방향 안정 판정
# =========================
def palm_direction_stable(lm, w, h, prev="unknown"):
    p0 = np.array([lm.landmark[0].x*w, lm.landmark[0].y*h, lm.landmark[0].z])
    p5 = np.array([lm.landmark[5].x*w, lm.landmark[5].y*h, lm.landmark[5].z])
    p17 = np.array([lm.landmark[17].x*w, lm.landmark[17].y*h, lm.landmark[17].z])
    normal = np.cross(p5-p0, p17-p0)

    score_up=0
    score_down=0

    if normal[2]<0: score_up+=1
    else: score_down+=1

    palm_z = (lm.landmark[0].z + lm.landmark[5].z + lm.landmark[9].z + lm.landmark[13].z + lm.landmark[17].z)/5
    back_z = (lm.landmark[1].z + lm.landmark[2].z + lm.landmark[3].z + lm.landmark[4].z)/4
    if palm_z < back_z: score_up+=1
    else: score_down+=1

    if lm.landmark[9].z < lm.landmark[0].z: score_up+=1
    else: score_down+=1

    if score_up>=2: return "palm_up"
    if score_down>=2: return "palm_down"
    return prev

# =========================
# 강화형 주먹 판정
# =========================
def is_thumb_closed(lm, w, h):
    tx, ty = lm.landmark[4].x*w, lm.landmark[4].y*h
    ix, iy = lm.landmark[5].x*w, lm.landmark[5].y*h
    return ((tx-ix)**2 + (ty-iy)**2)**0.5 < 40

def fingers_folded(lm, w, h):
    px, py = get_palm_center(lm, w, h)
    tips=[4,8,12,16,20]
    folded=0
    for i in tips:
        if lm.landmark[i].y*h > py+10:
            folded+=1
    return folded>=4

def detect_fist_strong(lm, w, h):
    if fingers_folded(lm,w,h): return "fist"
    if is_thumb_closed(lm,w,h): return "fist"
    return "open"

# =========================
# ✅ Skill Logic (요구사항 적용)
# =========================
class RightHandSkill:
    def __init__(self):
        self.ready=False       # 주먹 + palm_up 상태
        self.active=False      # 애니메이션 출력 중
        self.frame_idx=0
        self.prev_dir="unknown"
        self.last="unknown"
        self.streak=0

    def stable(self,label):
        if label==self.last:
            self.streak+=1
        else:
            self.last=label
            self.streak=1
        return label if self.streak>=DEBOUNCE_N else "unknown"

    def update(self,label,direction):
        # ✅ 준비상태: 손바닥 위 + 주먹
        if label=="fist" and direction=="palm_up":
            self.ready=True
            return

        # ✅ READY 상태 + 손바닥 위 + 손 펼침 → 발동
        if self.ready and label=="open" and direction=="palm_up":
            self.active=True
            self.frame_idx=0
            self.ready=False

    def draw(self,frame,lm,w,h):
        if not self.active:
            return

        px, py = get_palm_center(lm,w,h)
        py -= 60     # 손바닥 위로 띄우기

        img = ANIM_FRAMES[(self.frame_idx*ANIM_SPEED)%len(ANIM_FRAMES)]
        overlay_energy(frame, img, int(px), int(py))

        self.frame_idx+=1
        if (self.frame_idx*ANIM_SPEED)>=len(ANIM_FRAMES):
            self.active=False

# =========================
# PySide6 버전
# =========================
def run_energy_skill(video_label, cam_index=0, mirror=False):
    pose = OpenVinoPose()
    hands = MediaPipeHands()
    skill = RightHandSkill()
    cap = cv2.VideoCapture(cam_index)

    def update():
        ok,frame = cap.read()
        if not ok: return
        if mirror: frame=cv2.flip(frame,1)

        h,w = frame.shape[:2]

        # pose skeleton
        kpts = pose.extract_keypoints(pose.infer(frame), w, h)
        for a,b in POSE_PAIRS:
            if a<len(kpts) and b<len(kpts):
                ka,kb=kpts[a],kpts[b]
                if ka.conf>CONF_KPT and kb.conf>CONF_KPT:
                    cv2.line(frame,(ka.x,ka.y),(kb.x,kb.y),(0,255,0),2)

        res = hands.process(frame)
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks,res.multi_handedness):

                fist = detect_fist_strong(lm,w,h)
                fist = skill.stable(fist)

                direction = palm_direction_stable(lm,w,h,skill.prev_dir)
                skill.prev_dir = direction

                skill.update(fist,direction)
                skill.draw(frame,lm,w,h)

                # visual
                hands.drawer.draw_landmarks(frame, lm,
                    hands.mp.HAND_CONNECTIONS,
                    hands.drawer.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    hands.drawer.DrawingSpec(color=(0,255,0), thickness=2)
                )

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        qimg=QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        video_label.setPixmap(QPixmap.fromImage(qimg))

    return update

# =========================
# Tkinter 버전
# =========================
def run_energy_skill_tk(video_label, cam_index=0, mirror=False):
    pose = OpenVinoPose()
    hands = MediaPipeHands()
    skill = RightHandSkill()
    cap = cv2.VideoCapture(cam_index)

    def update():
        ok,frame = cap.read()
        if not ok:return
        if mirror: frame=cv2.flip(frame,1)

        h,w = frame.shape[:2]

        kpts = pose.extract_keypoints(pose.infer(frame), w, h)
        for a,b in POSE_PAIRS:
            if a<len(kpts) and b<len(kpts):
                ka,kb=kpts[a],kpts[b]
                if ka.conf>CONF_KPT and kb.conf>CONF_KPT:
                    cv2.line(frame,(ka.x,ka.y),(kb.x,kb.y),(0,255,0),2)

        res = hands.process(frame)
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks,res.multi_handedness):

                fist = detect_fist_strong(lm,w,h)
                fist = skill.stable(fist)

                direction = palm_direction_stable(lm,w,h,skill.prev_dir)
                skill.prev_dir = direction

                skill.update(fist,direction)
                skill.draw(frame,lm,w,h)

                hands.drawer.draw_landmarks(frame, lm,
                    hands.mp.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(rgb)
        imgtk=ImageTk.PhotoImage(image=img)
        video_label.imgtk=imgtk
        video_label.config(image=imgtk)

    return update

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rvm_cats_handsonly_pointdir_autoreset_rec.py
- RVM + 배경 + 크로마키 고양이(겹치지 않게 랜덤 배치)
- ★ 손만 보이게: RVM 알파(pha)에 '손 마스크(랜드마크 컨벡스헐+팽창+페더)'를 곱함
- 검지 TIP이 히트박스에 닿으면 "HIT!" + '검지 방향'(MCP→TIP)으로 임펄스
- 물리 모드: off / impulse(감쇠만) / gravity(중력+감쇠)  ← 기본: impulse
- allow_escape=True면 화면 밖으로 나가게 허용(벽 반사/클램프 비활성화)
- 모든 고양이가 화면 밖이면 일정 시간 후 자동 리셋(repack)
- 경계 충돌/클램프는 '히트박스(bx,by,bw,bh)' 기준
- ★ v 키로 MP4 녹화 시작/정지(토글), recordings/ 폴더에 저장
"""

import os, sys, time, random, argparse, datetime
import numpy as np
import cv2
import torch

# ======================= CONFIG =======================
CONFIG = {
    # 입출력
    "cam_index": 6,
    "frame_w": 1280,
    "frame_h": 720,
    "mirror": True,

    # RVM
    "rvm_variant": "mobilenetv3",
    "rvm_model": None,             # None이면 REPO_ROOT/model/rvm_mobilenetv3.pth
    "downsample_ratio": 0.40,

    # 합성 배경
    "bg_path": "back.png",
    "alpha_blur": 0,
    "alpha_boost": 0.0,
    "spill": 0.0,

    # 고양이 이펙트
    "effect_path": "A.mp4",
    "cats_count": 5,
    "cat_width": 180,
    "cat_width_step": 30,
    "cat_width_minmax": (60, 2000),
    "nonoverlap_margin": 10,

    # 크로마키 HSV (녹색)
    "ck_lo_hsv": (35, 60, 40),
    "ck_hi_hsv": (85, 255, 255),
    "ck_feather": 4,
    "ck_spill": 0.12,

    # 손/표시
    "hand_task": "hand_landmarker.task",
    "show_hands_default": True,
    "show_gun_dir_default": True,

    # ★ 손만 보이게
    "hands_only": True,
    "hand_mask_dilate": 18,   # 손 마스크 팽창(px)
    "hand_mask_feather": 12,  # 손 마스크 페더(블러, px; 짝수면 +1)
    "hands_min_area": 150,    # 너무 작은 잡영역 무시(px)

    # 임펄스/물리
    "impulse_gain": 0.12,   # 방향 단위벡터 * (finger_speed + impulse_base) * 이 값
    "impulse_base": 650.0,  # 손가락이 거의 안 움직여도 기본 임펄스
    "damping": 1.0,         # impulse/gravity 모드 감쇠(쭉 가려면 1.0)
    "gravity": 900.0,       # gravity 모드 중력가속도
    "vmax": 3000.0,         # 속도 상한
    "wall_bounce": 0.5,     # allow_escape=False일 때만 사용
    "hit_flash_time": 0.5,  # "HIT!" 표시 시간(초)

    # 화면 밖 허용
    "allow_escape": False,  # True면 가장자리 클램프/반사 끄고 그대로 밖으로 나감

    # 자동 리셋
    "auto_reset_if_all_gone": True,
    "auto_reset_delay": 1.5,

    # 히트 순간 추가 배율
    "hit_impulse_multiplier": 1.6,

    # ★ 녹화
    "rec_out_dir": "recordings",   # 저장 폴더
    "rec_basename": "handcats",    # 파일 접두사
    "rec_fps": 30.0,               # 파일에 기록할 fps(고정 권장)
    "rec_codec": "mp4v",           # mp4v / avc1 등
    # "rec_max_minutes": 0,        # (옵션) 0이면 무제한, >0이면 자동 분할 구현 가능
    "window_title": "RVM+Cats+HandsOnly (pointdir + autoreset + REC)",
}

# ======================= KEYMAP =======================
KEYMAP = {
    "quit":        ("q / ESC", "종료"),
    "toggle_box":  ("b", "히트박스 표시 토글"),
    "redistrib":   ("r", "고양이 위치 재배치(겹치지 않음)"),
    "size_up":     ("+", "고양이 크기 +"),
    "size_down":   ("-", "고양이 크기 -"),
    "cats_inc":    ("]", "마리 수 +1"),
    "cats_dec":    ("[", "마리 수 -1(최소 1)"),
    "mirror":      ("m", "미러 토글"),
    "hands":       ("h", "손 랜드마크 표시 토글"),
    "gun":         ("g", "검지 방향 화살표 토글"),
    "phys_mode":   ("k", "물리 모드 순환: off→impulse→gravity"),
    "vel_reset":   ("p", "모든 고양이 속도 리셋"),
    "record":      ("v", "녹화 시작/정지 (MP4 저장)"),
}

# ======================= 경로/모델 =======================
ABC_DIR   = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ABC_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MattingNetwork

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------- 유틸 ----------
def alpha_postprocess_and_spill_suppress(fgr_rgb, alpha01, blur_ks=0, boost=0.0, spill=0.0):
    a = alpha01.astype(np.float32)
    if boost != 0.0:
        a = np.clip(a + float(boost), 0.0, 1.0)
    if int(blur_ks) > 0:
        ks = int(blur_ks);  ks += (ks % 2 == 0)
        a = cv2.GaussianBlur(a, (ks, ks), 0)
    if spill > 0.0:
        edge = cv2.Canny((a * 255).astype(np.uint8), 32, 64).astype(np.float32) / 255.0
        edge3 = edge[..., None]
        fgr_bgr = cv2.cvtColor(fgr_rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(fgr_bgr, cv2.COLOR_BGR2HSV)
        hsv[..., 1] *= (1.0 - float(spill) * edge3)
        hsv = np.clip(hsv, 0.0, 1.0)
        fgr_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        fgr_rgb = cv2.cvtColor((fgr_bgr * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return fgr_rgb, a

def compose_over_bg(fgr_rgb, alpha01, bg_bgr):
    H, W = alpha01.shape[:2]
    if bg_bgr.shape[:2] != (H, W):
        bg_bgr = cv2.resize(bg_bgr, (W, H), interpolation=cv2.INTER_AREA)
    fg_bgr = cv2.cvtColor(fgr_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    bg_bgr = bg_bgr.astype(np.float32)
    a3 = alpha01[..., None].astype(np.float32)
    out = fg_bgr * a3 + bg_bgr * (1.0 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)

def colorkey_to_bgra(bgr, lo_hsv, hi_hsv, feather_px=4, spill=0.12):
    if bgr.ndim == 2: bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(lo_hsv, np.uint8); hi = np.array(hi_hsv, np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    inv = 255 - mask
    if feather_px > 0:
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dist = np.clip(dist/float(feather_px), 0.0, 1.0)
        alpha = (dist * 255).astype(np.uint8)
    else:
        alpha = inv
    if spill > 0:
        fg = alpha > 0
        hsv = hsv.astype(np.float32)
        hsv[...,1][fg] = np.clip(hsv[...,1][fg]*(1.0-spill), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.dstack([bgr, alpha])

def resize_keep_w(img, w_target):
    h, w = img.shape[:2]
    w_target = max(1, int(w_target))
    h_target = max(1, int(h * (w_target / float(w))))
    return cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_AREA)

class EffectLoop:
    def __init__(self, path, w0, lo_hsv, hi_hsv, feather, spill, fps_hint=24.0):
        self.frames_bgr = []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"[ERROR] effect open fail: {path}")
        while True:
            ok, f = cap.read()
            if not ok: break
            self.frames_bgr.append(f)
        cap.release()
        if not self.frames_bgr:
            raise SystemExit("[ERROR] effect preload failed (no frames)")
        self.n = len(self.frames_bgr)
        self.t0 = time.time()
        self.w = int(w0)
        self.fps = float(fps_hint)
        self.lo_hsv = tuple(map(int, lo_hsv))
        self.hi_hsv = tuple(map(int, hi_hsv))
        self.feather = int(feather)
        self.spill = float(spill)
    def set_width(self, w):  self.w = int(max(40, min(2000, w)))
    def frame_bgra(self):
        idx = int((time.time() - self.t0) * self.fps) % self.n
        bgra = colorkey_to_bgra(self.frames_bgr[idx],
                                self.lo_hsv, self.hi_hsv,
                                feather_px=self.feather, spill=self.spill)
        if bgra.shape[1] != self.w:
            bgra = resize_keep_w(bgra, self.w)
        return bgra

def overlay_bgra(dst_bgr, fg_bgra, x, y):
    H, W = dst_bgr.shape[:2]
    h, w = fg_bgra.shape[:2]
    if x >= W or y >= H or x + w <= 0 or y + h <= 0: return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    roi_bg = dst_bgr[y0:y1, x0:x1]
    roi_fg = fg_bgra[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
    alpha = roi_fg[..., 3:4].astype(np.float32) / 255.0
    out = roi_fg[..., :3].astype(np.float32) * alpha + roi_bg.astype(np.float32) * (1.0 - alpha)
    roi_bg[...] = out.astype(np.uint8)

def tight_bbox_from_alpha(rgba, thr=10):
    a = rgba[..., 3]
    mask = (a > int(thr)).astype(np.uint8)
    if mask.max() == 0:
        h, w = a.shape
        return 0,0,w,h
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, (x1-x0+1), (y1-y0+1)

def non_overlapping_positions(W, H, w, h, count, margin=8, max_trials=4000, rng=None):
    rng = rng or random.Random()
    rects = []
    def intersects(x, y):
        for (ax, ay, aw, ah) in rects:
            if not (x + w + margin <= ax or ax + aw + margin <= x or
                    y + h + margin <= ay or ay + ah + margin <= y):
                return True
        return False
    trials = 0
    while len(rects) < count and trials < max_trials:
        x = rng.randint(margin, max(margin, W - w - margin))
        y = rng.randint(margin, max(margin, H - h - margin))
        if not intersects(x, y): rects.append((x, y, w, h))
        trials += 1
    return [(x, y) for (x, y, _, _) in rects]

# ---------- 손 추적 ----------
class HandTracker:
    IDX_TIP = 8
    IDX_MCP = 5
    def __init__(self, task_path, max_hands=2):
        base = mp_python.BaseOptions(model_asset_path=task_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base, num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(opts)
        self.t0 = time.time()
    def detect(self, rgb):
        ts_ms = int((time.time() - self.t0)*1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect_for_video(mp_img, ts_ms)
    @staticmethod
    def collect_points_px(W, H, lm):
        return np.array([(int(p.x*W), int(p.y*H)) for p in lm], dtype=np.int32)
    @staticmethod
    def tip_mcp_and_draw(W, H, lm, canvas=None, draw_arrow=False):
        pts = [(int(p.x*W), int(p.y*H)) for p in lm]
        tip = pts[HandTracker.IDX_TIP]
        mcp = pts[HandTracker.IDX_MCP]
        if draw_arrow and canvas is not None:
            cv2.arrowedLine(canvas, mcp, tip, (0,0,255), 2, tipLength=0.25)
        return tip, mcp

# ============== 메인 ==============
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=int, default=CONFIG["frame_w"])
    p.add_argument("--h", type=int, default=CONFIG["frame_h"])
    return p.parse_args()

def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def open_recorder(path, w, h, fps, codec="mp4v"):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        # 실패 시 대체 코덱 재시도
        alt = "avc1" if codec.lower() != "avc1" else "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*alt)
        writer = cv2.VideoWriter(path, fourcc, float(fps), (int(w), int(h)))
    return writer

def main():
    args = parse_args()
    W, H = args.w, args.h

    # 장치/RVM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    rvm_model = CONFIG["rvm_model"] or os.path.join(REPO_ROOT, "model", "rvm_mobilenetv3.pth")
    model = MattingNetwork(variant=CONFIG["rvm_variant"]).eval().to(device)
    try:
        state = torch.load(rvm_model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(rvm_model, map_location=device)
    model.load_state_dict(state)
    use_half = (device.type == "cuda")
    if use_half: model.half()

    # 배경
    bg_path = os.path.join(ABC_DIR, CONFIG["bg_path"])
    if os.path.exists(bg_path):
        bg_full = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        bg = cv2.resize(bg_full, (W, H), interpolation=cv2.INTER_AREA) if bg_full is not None else np.zeros((H,W,3), np.uint8)
    else:
        bg = np.zeros((H,W,3), np.uint8)

    # 카메라
    cap = cv2.VideoCapture(CONFIG["cam_index"], cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] camera open failed: {CONFIG['cam_index']}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # 이펙트
    effect = EffectLoop(os.path.join(ABC_DIR, CONFIG["effect_path"]),
                        CONFIG["cat_width"],
                        CONFIG["ck_lo_hsv"], CONFIG["ck_hi_hsv"],
                        CONFIG["ck_feather"], CONFIG["ck_spill"], fps_hint=24.0)

    # 손
    hand_tracker = HandTracker(os.path.join(ABC_DIR, CONFIG["hand_task"]), max_hands=2)
    show_hands = CONFIG["show_hands_default"]
    show_gundir = CONFIG["show_gun_dir_default"]

    # 상태(고양이들)
    cats = []  # {pos:[x,y], vel:[vx,vy], off:[bx,by,bw,bh], hit_t:float}
    def repack(n):
        nonlocal cats
        effect.set_width(CONFIG["cat_width"])
        rgba = effect.frame_bgra()
        ch, cw = rgba.shape[:2]
        bx,by,bw,bh = tight_bbox_from_alpha(rgba, thr=10)
        pos = non_overlapping_positions(W, H, cw, ch, n, margin=CONFIG["nonoverlap_margin"])
        cats = [{"pos":[float(px), float(py)], "vel":[0.0,0.0], "off":[bx,by,bw,bh], "hit_t":0.0}
                for (px,py) in pos]

    repack(CONFIG["cats_count"])

    # 물리 모드: 0:off, 1:impulse(감쇠만), 2:gravity
    phys_mode = 1
    mode_name = ["off","impulse","gravity"]

    # 기타
    mirror = CONFIG["mirror"]
    show_boxes = True
    last_tip = None
    dt_hist, t_prev = [], time.time()

    # 자동 리셋 타이머
    last_any_visible_ts = time.time()

    # ★ 녹화 상태
    ensure_dir(os.path.join(ABC_DIR, CONFIG["rec_out_dir"]))
    rec_on = False
    rec_writer = None
    rec_path = None
    rec_fps = float(CONFIG.get("rec_fps", 30.0))
    rec_codec = str(CONFIG.get("rec_codec", "mp4v"))

    # 키 설명
    print("[KEYS] " + " | ".join([f"{v[0]}:{v[1]}" for v in KEYMAP.values()]))

    while True:
        ok, frame = cap.read()
        if not ok: continue
        if mirror: frame = cv2.flip(frame, 1)

        # RVM 입력
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)
        t = (t.half() if use_half else t.float()) / 255.0

        # RVM 추론
        with torch.no_grad():
            if use_half and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    fgr, pha, _r1,_r2,_r3,_r4 = model(t, None,None,None,None,
                                                      downsample_ratio=CONFIG["downsample_ratio"])
            else:
                fgr, pha, _r1,_r2,_r3,_r4 = model(t, None,None,None,None,
                                                  downsample_ratio=CONFIG["downsample_ratio"])
        fgr_np = (fgr[0].permute(1,2,0).clamp(0,1).mul(255).byte().cpu().numpy())
        a_np   =  (pha[0,0].clamp(0,1).float().cpu().numpy())

        # 손 검출 (RVM 후에 마스크 생성용)
        res = hand_tracker.detect(rgb)

        # === 손만 보이게: 손 마스크 만들고 RVM 알파에 곱하기 ===
        if CONFIG.get("hands_only", True):
            hand_mask = np.zeros((H, W), np.uint8)
            if res and res.hand_landmarks:
                for lm in res.hand_landmarks:
                    pts = HandTracker.collect_points_px(W, H, lm)
                    if pts.shape[0] >= 3:
                        hull = cv2.convexHull(pts)
                        if cv2.contourArea(hull) >= CONFIG.get("hands_min_area", 150):
                            cv2.fillConvexPoly(hand_mask, hull, 255)
                # 팽창 + 페더
                dil = max(0, int(CONFIG.get("hand_mask_dilate", 18)))
                if dil > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
                    hand_mask = cv2.dilate(hand_mask, k, iterations=1)
                feather = int(CONFIG.get("hand_mask_feather", 12))
                if feather > 0:
                    if feather % 2 == 0: feather += 1
                    hand_mask = cv2.GaussianBlur(hand_mask, (feather, feather), 0)
            a_np = a_np * (hand_mask.astype(np.float32) / 255.0)

        # 알파 후처리(가장자리/스필)
        fgr_np, a_np = alpha_postprocess_and_spill_suppress(
            fgr_np, a_np, blur_ks=CONFIG["alpha_blur"], boost=CONFIG["alpha_boost"], spill=CONFIG["spill"]
        )

        # 합성
        out = compose_over_bg(fgr_np, a_np, bg.copy())

        # 시간/FPS
        now = time.time(); dt = now - t_prev; t_prev = now
        dt_hist.append(dt);  dt_hist = dt_hist[-30:]
        fps = 1.0 / (sum(dt_hist)/len(dt_hist)) if dt_hist else 0.0

        # 이펙트 프레임
        effect.set_width(CONFIG["cat_width"])
        cat_rgba = effect.frame_bgra()
        ch, cw = cat_rgba.shape[:2]

        # 손 디버그 표시/검지 TIP,MCP(임펄스용)
        tip_px = None; mcp_px = None
        if res and res.hand_landmarks:
            lm0 = res.hand_landmarks[0]
            tip_px, mcp_px = HandTracker.tip_mcp_and_draw(W, H, lm0, canvas=out, draw_arrow=CONFIG["show_gun_dir_default"])
            if CONFIG["show_hands_default"]:
                for lm in res.hand_landmarks:
                    for p in lm:
                        cv2.circle(out, (int(p.x*W), int(p.y*H)), 2, (0,255,255), -1, cv2.LINE_AA)

        # 검지 이동 속도(px/s)
        finger_speed = 0.0
        if tip_px and last_tip:
            dx = (tip_px[0] - last_tip[0]) / max(dt, 1e-3)
            dy = (tip_px[1] - last_tip[1]) / max(dt, 1e-3)
            finger_speed = float((dx*dx + dy*dy) ** 0.5)
        if tip_px: last_tip = tip_px

        # 고양이 업데이트/그리기
        any_visible = False
        for c in cats:
            bx,by,bw,bh = c["off"]
            hitL = c["pos"][0] + bx
            hitT = c["pos"][1] + by
            hitR = hitL + bw
            hitB = hitT + bh

            # 히트: TIP이 히트박스 안이면 MCP→TIP 방향 임펄스
            if tip_px and mcp_px and (hitL <= tip_px[0] <= hitR) and (hitT <= tip_px[1] <= hitB):
                dirx = float(tip_px[0] - mcp_px[0])
                diry = float(tip_px[1] - mcp_px[1])
                norm = (dirx*dirx + diry*diry) ** 0.5
                if norm < 1e-3:
                    dirx, diry, norm = 1.0, 0.0, 1.0
                dirx /= norm; diry /= norm
                mag  = CONFIG["impulse_gain"] * (finger_speed + CONFIG["impulse_base"]) * CONFIG.get("hit_impulse_multiplier", 1.0)
                c["vel"][0] += dirx * mag
                c["vel"][1] += diry * mag
                c["hit_t"] = CONFIG["hit_flash_time"]

            # 물리
            if phys_mode == 1:     # impulse(감쇠만)
                c["vel"][0] *= CONFIG["damping"]
                c["vel"][1] *= CONFIG["damping"]
            elif phys_mode == 2:   # gravity
                c["vel"][1] += CONFIG["gravity"] * dt
                c["vel"][0] *= CONFIG["damping"]
                c["vel"][1] *= CONFIG["damping"]

            # 속도 상한
            sp = (c["vel"][0]**2 + c["vel"][1]**2) ** 0.5
            if sp > CONFIG["vmax"]:
                s = CONFIG["vmax"]/sp
                c["vel"][0] *= s; c["vel"][1] *= s

            # 위치
            c["pos"][0] += c["vel"][0] * dt
            c["pos"][1] += c["vel"][1] * dt

            # 경계(히트박스 기준)
            if not CONFIG.get("allow_escape", False):
                hitL = c["pos"][0] + bx
                hitT = c["pos"][1] + by
                hitR = hitL + bw
                hitB = hitT + bh
                if hitL < 0:          c["pos"][0] = -bx;          c["vel"][0] *= -CONFIG["wall_bounce"]
                if hitR > W:          c["pos"][0] = W-bw-bx;      c["vel"][0] *= -CONFIG["wall_bounce"]
                if hitT < 0:          c["pos"][1] = -by;          c["vel"][1] *= -CONFIG["wall_bounce"]
                if hitB > H:          c["pos"][1] = H-bh-by;      c["vel"][1] *= -CONFIG["wall_bounce"]

            # 렌더 + 디버그
            overlay_bgra(out, cat_rgba, int(c["pos"][0]), int(c["pos"][1]))
            if show_boxes:
                cv2.rectangle(out, (int(hitL), int(hitT)), (int(hitR), int(hitB)), (0,255,0), 2)
            if c["hit_t"] > 0.0:
                c["hit_t"] -= dt
                cv2.putText(out, "HIT!", (int(hitL)+4, max(18, int(hitT)-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

            # 화면과 교차?
            if (hitL < W and hitT < H and hitR > 0 and hitB > 0):
                any_visible = True

        # 모두 화면 밖이면 일정 시간 뒤 리셋
        if CONFIG.get("auto_reset_if_all_gone", True):
            if any_visible:
                last_any_visible_ts = now
            else:
                if (now - last_any_visible_ts) >= CONFIG.get("auto_reset_delay", 1.5):
                    repack(CONFIG["cats_count"])
                    last_any_visible_ts = now

        # ★ 녹화 쓰기
        if rec_on and rec_writer is not None:
            # out은 BGR(3채널)이어야 함
            if out.shape[1] != W or out.shape[0] != H:
                out_to_write = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
            else:
                out_to_write = out
            rec_writer.write(out_to_write)

            # 화면 좌상단 REC 표시
            cv2.putText(out, "REC", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            cv2.circle(out, (60, 22), 6, (0,0,255), -1, cv2.LINE_AA)

        # HUD
        hud = (f"fps:{fps:4.1f} mode:{mode_name[phys_mode]} "
               f"cats:{len(cats)} size:{CONFIG['cat_width']}px "
               f"boxes:{int(show_boxes)} escape:{int(CONFIG['allow_escape'])} "
               f"handsOnly:{int(CONFIG['hands_only'])} "
               f"REC:{int(rec_on)}")
        cv2.putText(out, hud, (8, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        # 창 표시
        cv2.imshow(CONFIG["window_title"], out)

        # 키
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break

        elif k == ord('b'):
            show_boxes = not show_boxes

        elif k == ord('r'):
            repack(CONFIG["cats_count"])

        elif k in (ord('+'), ord('=')):
            CONFIG["cat_width"] = min(CONFIG["cat_width_minmax"][1], CONFIG["cat_width"] + CONFIG["cat_width_step"])
            repack(len(cats))

        elif k in (ord('-'), ord('_')):
            CONFIG["cat_width"] = max(CONFIG["cat_width_minmax"][0], CONFIG["cat_width"] - CONFIG["cat_width_step"])
            repack(len(cats))

        elif k == ord(']'):
            CONFIG["cats_count"] = len(cats) + 1
            repack(CONFIG["cats_count"])

        elif k == ord('['):
            CONFIG["cats_count"] = max(1, len(cats) - 1)
            repack(CONFIG["cats_count"])

        elif k == ord('m'):
            CONFIG["mirror"] = not CONFIG["mirror"]; mirror = CONFIG["mirror"]

        elif k == ord('h'):
            CONFIG["show_hands_default"] = not CONFIG["show_hands_default"]

        elif k == ord('g'):
            CONFIG["show_gun_dir_default"] = not CONFIG["show_gun_dir_default"]

        elif k == ord('k'):
            phys_mode = (phys_mode + 1) % 3

        elif k == ord('p'):
            for c in cats: c["vel"] = [0.0, 0.0]

        # ★ 녹화 토글
        elif k == ord('v'):
            if not rec_on:
                # 시작
                fname = f"{CONFIG['rec_basename']}_{timestamp()}.mp4"
                rec_path = os.path.join(ABC_DIR, CONFIG["rec_out_dir"], fname)
                rec_writer = open_recorder(rec_path, W, H, rec_fps, CONFIG.get("rec_codec", "mp4v"))
                if not rec_writer or not rec_writer.isOpened():
                    print(f"[ERROR] VideoWriter open failed: {rec_path}")
                    rec_writer = None
                    rec_on = False
                else:
                    rec_on = True
                    print(f"[REC] start -> {rec_path} (codec={CONFIG.get('rec_codec','mp4v')}, fps={rec_fps})")
            else:
                # 정지/저장
                rec_on = False
                if rec_writer is not None:
                    rec_writer.release()
                    print(f"[REC] saved -> {rec_path}")
                rec_writer = None
                rec_path = None

    # 종료 처리
    cap.release()
    if rec_writer is not None:
        rec_writer.release()
        print(f"[REC] saved -> {rec_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

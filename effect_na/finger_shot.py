# py_hand_module.py
# - MediaPipe Finger-Gun 제스처 → 검지 방향 탄환 발사
# - Tkinter Label에 OpenCV+PIL로 렌더
# - (전제) 효과/임팩트 PNG는 이미 "알파 포함 RGBA" 상태
# - 히트박스 충돌(AABB) + 바깥쪽 랜덤 튕김 + 임팩트 애니
# - 히트박스 가시화(draw_hitboxes=True 시 반투명 박스/테두리/라벨 렌더)
# - run_hand_gun_skill_tk(video_label, cam_index, mirror, hitboxes, draw_hitboxes) 제공

import os, glob, time, math, random
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ==============================
# 설정
# ==============================
CONFIG = {
    "model": "hand_landmarker.task",
    "camera": 0,
    "effect_folder": "hand_anim/shot/*.png",     # 탄환 애니 PNG 글롭 (RGBA 권장)
    "impact_folder": "hand_anim/impact/*.png",   # 임팩트 애니 PNG 글롭(없으면 꺼짐, RGBA 권장)
    "effect_size": (60, 60),
    "impact_size": (180, 180),
    "shot_speed": 25,       # px/frame
    "cooldown": 0.8,        # 발사 간격(초)
    "life": 1.8,            # 탄환 수명(초)
    "bullet_radius": 10,    # 충돌 반경(px)
    "max_hands": 2,
    "mirror": False,
    "impact_life": 0.35,    # 임팩트 재생 시간(초). 0이면 꺼짐
    "draw_hitboxes": True,  # ✅ 히트박스 가시화
    # 히트박스 시각화 스타일
    "hb_fill_rgba": (40, 200, 120, 70),  # (R,G,B,A) 반투명 채움
    "hb_edge_bgr": (80, 255, 160),       # 테두리 색(B,G,R)
    "hb_edge_th": 2,                      # 테두리 두께
    "hb_label": True,                     # 레이블 표시
}

# ==============================
# 유틸: 폴백 스프라이트(프레임 없을 때)
# ==============================
def make_fallback_sprite(size=(140,140), n=10):
    w, h = size
    frames=[]
    for i in range(n):
        rgba = np.zeros((h, w, 4), np.uint8)
        r = int(min(w,h)*0.32 + i*min(w,h)*0.03)
        # RGBA (R,G,B,A)
        cv2.circle(rgba, (w//2, h//2), r, (255, 200, 50, 180), -1)
        rgba = cv2.GaussianBlur(rgba, (0,0), sigmaX=6, sigmaY=6)
        frames.append(rgba)
    return frames

# ==============================
# PNG 로더 (RGBA 가정, 4채널 유지)
# ==============================
def load_rgba_frames(path_glob, size, must_have=True, label="effect"):
    files = sorted(glob.glob(path_glob))
    frames = []
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # BGRA → RGBA / BGR → RGBA(알파 생성 X, 불투명 255)
        if img.ndim == 3 and img.shape[2] == 4:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.ndim == 3:
            # 3채널이면 알파 채널을 255로 추가(이미 알파 처리된 이미지만 쓴다는 가정)
            alpha = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgba = np.dstack([rgb, alpha])
        else:
            # 예외적으로 1채널 등
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, np.uint8)
            rgba = np.dstack([rgb, alpha])

        rgba = cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)
        frames.append(rgba)

    if not frames:
        if must_have:
            print(f"[WARN] {label} frames not found: {path_glob} (cwd={os.getcwd()}) → using fallback sprite.")
        frames = make_fallback_sprite(size=size, n=10)
    return frames

# ==============================
# Overlay RGBA
# ==============================
def overlay_rgba(bg_rgb, fg_rgba, x, y):
    """bg_rgb: RGB, fg_rgba: RGBA, 좌상단(x,y)에 합성"""
    h, w = bg_rgb.shape[:2]
    fh, fw = fg_rgba.shape[:2]
    x1, y1 = x, y
    x2, y2 = x1 + fw, y1 + fh
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return bg_rgb

    # 클리핑
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    fx1, fy1 = cx1 - x1, cy1 - y1
    fx2, fy2 = fx1 + (cx2 - cx1), fy1 + (cy2 - cy1)

    roi = bg_rgb[cy1:cy2, cx1:cx2]
    frag = fg_rgba[fy1:fy2, fx1:fx2]
    alpha = frag[..., 3:4].astype(np.float32) / 255.0
    roi[:] = (roi * (1 - alpha) + frag[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    return bg_rgb

# ==============================
# MediaPipe Hand Landmarker
# ==============================
def create_landmarker(model_path):
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(opts)

def dist(a, b): return math.hypot(a.x - b.x, a.y - b.y)
def palm_dist(lm): return max(1e-6, dist(lm[0], lm[9]))

def is_finger_gun(lm):
    p = palm_dist(lm)
    index_up = dist(lm[8], lm[5]) / p > 0.62
    middle_dn = dist(lm[12], lm[9]) / p < 0.55
    ring_dn   = dist(lm[16], lm[13]) / p < 0.55
    pinky_dn  = dist(lm[20], lm[17]) / p < 0.55
    thumb_up  = dist(lm[4],  lm[5]) / p > 0.40
    return index_up and middle_dn and ring_dn and pinky_dn and thumb_up

def finger_dir(lm):
    dx = lm[8].x - lm[5].x
    dy = lm[8].y - lm[5].y
    n = math.hypot(dx, dy)
    if n < 1e-6: return 1.0, 0.0
    return dx / n, dy / n

# ==============================
# 원-사각형 충돌 (AABB)
# ==============================
def circle_rect_hit(cx, cy, r, rx, ry, rw, rh):
    """원 중심(cx,cy), 반지름 r vs 사각형 좌상단(rx,ry), 너비 rw, 높이 rh"""
    if rw <= 0 or rh <= 0:
        return False, (cx, cy)
    nx = min(max(cx, rx), rx + rw)
    ny = min(max(cy, ry), ry + rh)
    dx, dy = cx - nx, cy - ny
    return (dx*dx + dy*dy) <= (r*r), (nx, ny)

def outward_bounce_velocity(cx, cy, rect, speed, jitter_deg=60):
    """충돌 지점(cx,cy) 기준, 히트박스 바깥쪽으로 튕기는 속도 벡터(px/frame)"""
    rx, ry, rw, rh = rect
    rcx, rcy = rx + rw / 2.0, ry + rh / 2.0
    vx, vy = cx - rcx, cy - rcy
    n = math.hypot(vx, vy)
    if n < 1e-6:
        ang = random.uniform(0, 2 * math.pi)
    else:
        ang = math.atan2(vy, vx) + math.radians(random.uniform(-jitter_deg, jitter_deg))
    return math.cos(ang) * speed, math.sin(ang) * speed

# ==============================
# 메인 클래스 (Tk에서 사용)
# ==============================
class PyHandEffect:
    def __init__(self,
                 camera=0,
                 mirror=False,
                 effect_glob=CONFIG["effect_folder"],
                 impact_glob=CONFIG["impact_folder"],
                 draw_hitboxes=CONFIG["draw_hitboxes"]):
        # 카메라
        self.cap = cv2.VideoCapture(camera)
        self.mirror = mirror

        # MediaPipe
        self.lm = create_landmarker(CONFIG["model"])
        self.t0 = time.time()

        # 이펙트 로드
        self.frames = load_rgba_frames(effect_glob, CONFIG["effect_size"], must_have=True, label="effect")
        self.fn = len(self.frames)

        # 임팩트 로드(옵션)
        self.impact_frames = load_rgba_frames(impact_glob, CONFIG["impact_size"], must_have=False, label="impact") if impact_glob else []
        self.ifn = len(self.impact_frames)

        # 상태
        self.bullets = []  # {"x","y","vx","vy","t0"}
        self.impacts = []  # {"x","y","t0"}
        self.last_shot = 0

        # 히트박스 (여러 개 가능)
        self.hitboxes = []  # [(x,y,w,h), ...]
        self.draw_hitboxes = bool(draw_hitboxes)

    # ---------- 외부 API ----------
    def set_hitboxes(self, rects):
        """rects = [(x,y,w,h), ...]"""
        self.hitboxes = list(rects or [])

    def add_hitbox(self, x, y, w, h):
        self.hitboxes.append((int(x), int(y), int(w), int(h)))

    def clear_hitboxes(self):
        self.hitboxes.clear()

    def set_draw_hitboxes(self, on: bool):
        self.draw_hitboxes = bool(on)

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")

    # ---------- 내부 로직 ----------
    def _detect_and_fire(self, rgb):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.lm.detect_for_video(mp_img, int((time.time() - self.t0) * 1000))
        if not result or not result.hand_landmarks:
            return

        H, W = rgb.shape[:2]
        for lm in result.hand_landmarks[:CONFIG["max_hands"]]:
            if is_finger_gun(lm):
                t = time.time()
                if t - self.last_shot >= CONFIG["cooldown"]:
                    tip = lm[8]
                    ux, uy = finger_dir(lm)
                    bx, by = int(tip.x * W), int(tip.y * H)
                    self.bullets.append({
                        "x": bx, "y": by,
                        "vx": ux * CONFIG["shot_speed"],
                        "vy": uy * CONFIG["shot_speed"],
                        "t0": t
                    })
                    self.last_shot = t

    def _render_hitboxes(self, rgb):
        """프레임 위에 히트박스 반투명 렌더"""
        if not (self.draw_hitboxes and self.hitboxes):
            return rgb

        # 채움
        fill_r, fill_g, fill_b, fill_a = CONFIG["hb_fill_rgba"]
        if fill_a > 0:
            overlay = np.zeros_like(rgb, dtype=np.uint8)
            for i, (x, y, w, h) in enumerate(self.hitboxes, start=1):
                x, y, w, h = int(x), int(y), int(w), int(h)
                if w <= 0 or h <= 0: continue
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (fill_r, fill_g, fill_b), -1)
            alpha = (fill_a / 255.0)
            rgb = cv2.addWeighted(overlay, alpha, rgb, 1.0 - alpha, 0.0)

        # 테두리
        edge_bgr = CONFIG["hb_edge_bgr"]
        th = CONFIG["hb_edge_th"]
        for i, (x, y, w, h) in enumerate(self.hitboxes, start=1):
            x, y, w, h = int(x), int(y), int(w), int(h)
            if w <= 0 or h <= 0: continue
            cv2.rectangle(rgb, (x, y), (x+w, y+h), edge_bgr, th)
            if CONFIG["hb_label"]:
                label = f"HB{i}"
                cv2.putText(rgb, label, (x+4, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, edge_bgr, 2, cv2.LINE_AA)
        return rgb

    def _update_bullets_and_collisions(self, frame):
        now = time.time()
        r = CONFIG["bullet_radius"]
        alive = []

        for b in self.bullets:
            age = now - b["t0"]
            if age > CONFIG["life"]:
                continue

            # 이동
            b["x"] += b["vx"]
            b["y"] += b["vy"]

            # 충돌 체크
            for rect in self.hitboxes:
                hit, pt = circle_rect_hit(b["x"], b["y"], r, *rect)
                if hit:
                    # 튕김
                    b["vx"], b["vy"] = outward_bounce_velocity(b["x"], b["y"], rect, CONFIG["shot_speed"])
                    b["x"] += b["vx"] * 0.2
                    b["y"] += b["vy"] * 0.2
                    # 임팩트 기록
                    if CONFIG["impact_life"] > 0 and self.ifn > 0:
                        self.impacts.append({"x": int(pt[0]), "y": int(pt[1]), "t0": now})
                    break

            # 탄환 렌더 (안전 인덱싱)
            idx = 0 if self.fn <= 1 else (int(age * 10) % self.fn)
            fx = int(b["x"] - self.frames[idx].shape[1] // 2)
            fy = int(b["y"] - self.frames[idx].shape[0] // 2)
            frame = overlay_rgba(frame, self.frames[idx], fx, fy)

            alive.append(b)

        self.bullets = alive

        # 임팩트 렌더/정리
        if CONFIG["impact_life"] > 0 and self.ifn > 0:
            keep = []
            for it in self.impacts:
                age = now - it["t0"]
                if age <= CONFIG["impact_life"]:
                    idx = min(self.ifn - 1, int(age / CONFIG["impact_life"] * self.ifn))
                    rgba = self.impact_frames[idx]
                    fx = int(it["x"] - rgba.shape[1] // 2)
                    fy = int(it["y"] - rgba.shape[0] // 2)
                    frame = overlay_rgba(frame, rgba, fx, fy)
                    keep.append(it)
            self.impacts = keep
        else:
            self.impacts.clear()

        # 히트박스 가시화
        frame = self._render_hitboxes(frame)
        return frame

    # ---------- 외부에서 매 프레임 호출 ----------
    def get_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._detect_and_fire(rgb)
        rgb = self._update_bullets_and_collisions(rgb)
        return rgb

# ==============================
# ✅ Tkinter용 실행 함수
# ==============================
def run_hand_gun_skill_tk(video_label, cam_index=0, mirror=False, hitboxes=None, draw_hitboxes=True):
    """
    hitboxes: [(x,y,w,h), ...] 전달 시 즉시 적용
    사용 중에 히트박스를 바꾸려면: update_fn.eff.set_hitboxes(new_rects)
    가시화 ON/OFF: update_fn.eff.set_draw_hitboxes(True/False)
    """
    eff = PyHandEffect(camera=cam_index, mirror=mirror, draw_hitboxes=draw_hitboxes)
    eff.start()
    if hitboxes:
        eff.set_hitboxes(hitboxes)

    def update():
        frame = eff.get_frame()
        if frame is None:
            return
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    update.eff = eff
    return update

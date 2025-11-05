# py_hand.py
# - MediaPipe Hand Landmarker (VIDEO)
# - "손가락 총" 제스처 → 검지 각도 방향으로 탄환 발사
# - pygame 화면 위에 OpenCV 카메라 프레임 렌더 + RGBA 임팩트 합성
# - Pymunk 물리 충돌: 히트박스에 부딪히면 탄환이 히트박스 바깥쪽으로 랜덤 튕김
# - 히트박스: 마우스로 드래그/리사이즈(편집 모드), A로 적용/숨김, E로 편집 복귀
# - + / − / (키패드 +/−, 대괄호, 화살표) 로 임팩트 크기 동적 변경
#
# 실행 예:
#   python py_hand.py --camera 6 --mirror --effect A.mp4
#
# 단축키:
#   A : 히트박스 적용 & 숨김(편집 종료)
#   E : 편집 모드 복귀(보이기+드래그/리사이즈)
#   S : 히트박스 좌표 출력
#   R : 히트박스 초기화
#   + / − : 임팩트 크기 키우기/줄이기 (키패드 +/−, ] / [, ↑ / ↓ 도 지원)
#   Q / Esc : 종료

import os, json, time, math, random
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # Wayland 경고 회피

import argparse
import numpy as np
import cv2
import pygame as pg
import pymunk
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from math import sqrt

# ===================== 0) 설정 =====================
CONFIG = {
    "camera": 6, "width": 1280, "height": 720, "mirror": False, "fps": 60,
    "model": "hand_landmarker.task",
    "effect": "A.mp4", "effect_width": 220, "effect_fps": 24.0,
    "shot_speed": 300.0, "shot_life": 3.0, "cooldown": 2.0,
    "max_hands": 2, "y_offset": 0,
    "target_rect": "300,280,180,90", "target_sensor": False,
    "impact_life": 0.0,  # 0.0 이면 잔상 OFF (요청대로 기본 꺼둠)
    "EXT_ON": 0.62, "EXT_OFF": 0.55, "THUMB_MIN": 0.40, "HOLD_GRACE_S": 0.20,
    "GREEN_LO_HSV": [35, 60, 40], "GREEN_HI_HSV": [85, 255, 255],
    "EDGE_FEATHER_PX": 4, "SPILL_REDUCE": 0.15,
    "PX_PER_M": 60.0, "FIXED_DT": 1.0/240.0
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def merge_dict(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst

def parse_cli(cfg):
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    for k, v in cfg.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true" if not v else "store_false")
        elif isinstance(v, (int, float, str)):
            p.add_argument(f"--{k}", type=type(v))
    args = p.parse_args()
    if args.config: merge_dict(cfg, load_json(args.config))
    for k, v in vars(args).items():
        if k != "config" and v is not None: cfg[k] = v
    return cfg

# ===================== MediaPipe / 제스처 =====================
LANDMARK_INDEX_TIP = 8
LANDMARK_INDEX_MCP = 5

def create_landmarker(model_path: str):
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options)

def _dist(a,b): return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
def _palm(lm):  return max(1e-6, _dist(lm[0], lm[9]))

def _is_extended(lm, mcp, tip, palm, prev, EXT_ON, EXT_OFF):
    ratio = _dist(lm[tip], lm[mcp]) / palm
    thr = EXT_OFF if prev else EXT_ON
    return ratio >= thr

def is_finger_gun(lm, prev, EXT_ON, EXT_OFF, THUMB_MIN):
    palm = _palm(lm)
    index_up  = _is_extended(lm, 5, 8,  palm, prev, EXT_ON, EXT_OFF)
    middle_dn = not _is_extended(lm, 9, 12, palm, prev, EXT_ON, EXT_OFF)
    ring_dn   = not _is_extended(lm,13, 16, palm, prev, EXT_ON, EXT_OFF)
    pinky_dn  = not _is_extended(lm,17, 20, palm, prev, EXT_ON, EXT_OFF)
    thumb_up  = (_dist(lm[4], lm[5]) / palm) >= THUMB_MIN
    return index_up and thumb_up and middle_dn and ring_dn and pinky_dn

def finger_dir_unit(lm):
    mcp, tip = lm[LANDMARK_INDEX_MCP], lm[LANDMARK_INDEX_TIP]
    dx, dy = (tip.x - mcp.x), (tip.y - mcp.y)
    n = (dx*dx + dy*dy) ** 0.5
    if n < 1e-6: return 1.0, 0.0
    return dx/n, dy/n

# ===================== Effect(프리로드) - 동적 리사이즈 지원 =====================
def colorkey_to_rgba(bgr, lo_hsv, hi_hsv, feather_px=4, spill=0.15):
    if bgr.ndim == 2: bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(lo_hsv, np.uint8), np.array(hi_hsv, np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    inv  = 255 - mask
    if feather_px > 0:
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dist = np.clip(dist/float(feather_px), 0.0, 1.0)
        alpha = (dist*255).astype(np.uint8)
    else:
        alpha = inv
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])
    if spill > 0:
        fg = alpha > 0
        g = rgba[...,1].astype(np.float32); g[fg] = np.clip(g[fg]*(1.0-spill), 0, 255)
        rgba[...,1] = g.astype(np.uint8)
    return rgba

def resize_to_width(rgba, target_w):
    h, w = rgba.shape[:2]
    tw = max(1, int(target_w)); th = max(1, int(h*(tw/float(w))))
    return cv2.resize(rgba, (tw, th), interpolation=cv2.INTER_AREA)

class VideoEffectPreload:
    """
    ★변경점: 원본 RGBA 프레임들을 그대로 저장(frames_rgba_orig)
    frame_rgba() 호출 시점에 self.target_width로 '동적' 리사이즈.
    → 실행 중 +/−로 크기 바꿔도 즉시 반영됨.
    """
    def __init__(self, path, target_width, fps_for_loop, key_cfg):
        self.loop_fps = float(fps_for_loop)
        self.target_width = int(target_width)
        self.frames_rgba_orig = []   # 원본 RGBA 프레임
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise SystemExit(f"[ERROR] effect open fail: {path}")
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgba = colorkey_to_rgba(
                frame, key_cfg["GREEN_LO_HSV"], key_cfg["GREEN_HI_HSV"],
                key_cfg["EDGE_FEATHER_PX"], key_cfg["SPILL_REDUCE"]
            )
            self.frames_rgba_orig.append(rgba)  # 리사이즈하지 않고 원본 저장
        cap.release()
        if not self.frames_rgba_orig: raise SystemExit("[ERROR] effect preload failed")
        self.n = len(self.frames_rgba_orig)

    def frame_rgba(self, t0):
        idx = int((time.time() - t0) * self.loop_fps) % self.n
        rgba = self.frames_rgba_orig[idx]
        if rgba.shape[1] != self.target_width:
            rgba = resize_to_width(rgba, self.target_width)
        return rgba

# ===================== pygame helper =====================
def np_rgb_to_surface(rgb):
    h, w = rgb.shape[:2]
    return pg.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

def overlay_rgba_onto_surface(dst_surface, rgba, x, y):
    h, w = rgba.shape[:2]
    surf = pg.image.frombuffer(rgba.tobytes(), (w, h), "RGBA")
    dst_surface.blit(surf, (x, y))

# ===================== 물리/좌표 =====================
def px2m(x, PX_PER_M): return x / PX_PER_M
def m2px(x, PX_PER_M): return x * PX_PER_M

def make_static_box(space, px, py, pw, ph, sensor, PX_PER_M):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (px2m(px + pw/2, PX_PER_M), px2m(py + ph/2, PX_PER_M))
    shape = pymunk.Poly.create_box(body, (px2m(pw, PX_PER_M), px2m(ph, PX_PER_M)))
    shape.collision_type = 200
    shape.sensor = bool(sensor)
    space.add(body, shape)
    return body, shape

def spawn_bullet(space, px, py, vx_px_s, vy_px_s, life_s, PX_PER_M):
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)  # 등속
    body.position = (px2m(px, PX_PER_M), px2m(py, PX_PER_M))
    body.velocity = (px2m(vx_px_s, PX_PER_M), px2m(vy_px_s, PX_PER_M))
    shape = pymunk.Circle(body, radius=px2m(10, PX_PER_M))
    shape.collision_type = 100
    space.add(body, shape)
    return {"body": body, "shape": shape, "birth": time.time(), "life": life_s}

# ===================== 메인 =====================
def main():
    cfg = parse_cli(CONFIG.copy())

    # pygame 윈도우
    pg.init()
    screen = pg.display.set_mode((cfg["width"], cfg["height"]))
    pg.display.set_caption("Finger-Gun + Pymunk + pygame Hitbox (A: Apply/Hide, E: Edit)")
    clock = pg.time.Clock()
    font  = pg.font.SysFont(None, 18)

    # 카메라
    cap = cv2.VideoCapture(cfg["camera"], cv2.CAP_V4L2)
    if not cap.isOpened(): raise SystemExit(f"[ERROR] camera open failed: {cfg['camera']}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

    # MediaPipe
    lm = create_landmarker(cfg["model"])

    # 임팩트 소스(동적 리사이즈)
    key_cfg = {
        "GREEN_LO_HSV": cfg["GREEN_LO_HSV"],
        "GREEN_HI_HSV": cfg["GREEN_HI_HSV"],
        "EDGE_FEATHER_PX": cfg["EDGE_FEATHER_PX"],
        "SPILL_REDUCE": cfg["SPILL_REDUCE"],
    }
    effect = VideoEffectPreload(cfg["effect"], cfg["effect_width"], cfg["effect_fps"], key_cfg)
    t0 = time.time()

    # 물리
    space = pymunk.Space()
    PX_PER_M, FIXED_DT = cfg["PX_PER_M"], cfg["FIXED_DT"]

    # 히트박스 초기값
    tx, ty, tw, th = map(int, cfg["target_rect"].split(","))
    target_rect = pg.Rect(tx, ty, tw, th)
    target_body, target_shape = make_static_box(space, tx, ty, tw, th, cfg["target_sensor"], PX_PER_M)

    # 편집/적용 토글
    edit_mode = True   # True: 보이기+드래그/리사이즈, False: 숨김(충돌만)

    # 상태
    impacts, bullets, last_shot = [], [], 0.0

    def outward_random_velocity(px, py, rect, speed_px):
        """히트박스 바깥쪽을 향하는 랜덤 튕김(±60°)"""
        cx, cy = rect.center
        vx, vy = (px - cx), (py - cy)
        n = math.hypot(vx, vy)
        if n < 1e-6:
            ang = random.uniform(0, 2*math.pi); ux, uy = math.cos(ang), math.sin(ang)
        else:
            ux, uy = vx/n, vy/n
            ang = math.atan2(uy, ux) + random.uniform(-math.pi/3, math.pi/3)
            ux, uy = math.cos(ang), math.sin(ang)
        return ux*speed_px, uy*speed_px

    # 충돌 콜백
    def on_hit(arbiter, space_, data):
        cps = arbiter.contact_point_set
        if cps.points:
            p = cps.points[0].point_a
            cx, cy = m2px(p.x, PX_PER_M), m2px(p.y, PX_PER_M)
        else:
            a = arbiter.shapes[0]
            cx, cy = m2px(a.body.position.x, PX_PER_M), m2px(a.body.position.y, PX_PER_M)

        # 잔상(impact) 옵션: 기본 0.0이라 표시 안 함
        if cfg["impact_life"] > 0.0:
            impacts.append({"x": cx, "y": cy, "t0": time.time(),
                            "life": float(cfg["impact_life"]), "w": effect.target_width})

        # 탄환 튕김(바깥 방향으로)
        a, b = arbiter.shapes
        bullet = a if a.collision_type == 100 else b
        bx, by = m2px(bullet.body.position.x, PX_PER_M), m2px(bullet.body.position.y, PX_PER_M)
        vx, vy = outward_random_velocity(bx, by, target_rect, cfg["shot_speed"])
        bullet.body.velocity = (px2m(vx, PX_PER_M), px2m(vy, PX_PER_M))
        bullet.body.position = (bullet.body.position.x + px2m(5*vx/cfg["shot_speed"], PX_PER_M),
                                bullet.body.position.y + px2m(5*vy/cfg["shot_speed"], PX_PER_M))
        return False  # 물리 기본 반응 무시

    # 충돌 핸들러(버전 호환)
    try:
        h = space.add_collision_handler(100, 200); h.begin = on_hit
    except AttributeError:
        if hasattr(space, "add_wildcard_collision_handler"):
            hw = space.add_wildcard_collision_handler(100)
            def _wild_begin(arb, sp, dat):
                a, b = arb.shapes
                if getattr(a,"collision_type",None)==200 or getattr(b,"collision_type",None)==200:
                    return on_hit(arb, sp, dat)
                return True
            hw.begin = _wild_begin
        else:
            hd = space.add_default_collision_handler()
            def _begin(arb, sp, dat):
                a, b = arb.shapes
                ca, cb = getattr(a,"collision_type",None), getattr(b,"collision_type",None)
                if (ca==100 and cb==200) or (ca==200 and cb==100): return on_hit(arb, sp, dat)
                return True
            hd.begin = _begin

    # 편집용 핸들
    def handle_rects(r: pg.Rect):
        cx, cy = r.center
        return {
            "tl": pg.Rect(r.left-5,  r.top-5, 10,10),
            "t":  pg.Rect(cx-5,      r.top-5, 10,10),
            "tr": pg.Rect(r.right-5, r.top-5, 10,10),
            "l":  pg.Rect(r.left-5,  cy-5,    10,10),
            "r":  pg.Rect(r.right-5, cy-5,    10,10),
            "bl": pg.Rect(r.left-5,  r.bottom-5, 10,10),
            "b":  pg.Rect(cx-5,      r.bottom-5, 10,10),
            "br": pg.Rect(r.right-5, r.bottom-5, 10,10),
        }
    def hit_handle_index(r: pg.Rect, pos):
        if r.w == 0 and r.h == 0: return -1
        hs = handle_rects(r); order = ["tl","t","tr","l","r","bl","b","br"]
        for i,k in enumerate(order):
            if hs[k].collidepoint(pos): return i
        return 8 if r.collidepoint(pos) else -1

    dragging, drag_mode, start_pt, start_rect = False, -1, (0,0), None

    # 타이밍
    accum, t_prev = 0.0, time.time()
    running = True
    while running:
        # 이벤트 처리
        for e in pg.event.get():
            if e.type == pg.QUIT: running = False
            elif e.type == pg.KEYDOWN:
                if e.key in (pg.K_q, pg.K_ESCAPE): running = False
                # 크기 키우기 (다양한 키 지원)
                elif e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_RIGHTBRACKET, pg.K_KP_PLUS, pg.K_UP):
                    effect.target_width = min(effect.target_width + 40, 2000)
                    print(f"[size] {effect.target_width}px")
                # 크기 줄이기
                elif e.key in (pg.K_MINUS, pg.K_UNDERSCORE, pg.K_LEFTBRACKET, pg.K_KP_MINUS, pg.K_DOWN):
                    effect.target_width = max(effect.target_width - 40,  80)
                    print(f"[size] {effect.target_width}px")
                elif e.key == pg.K_s:
                    print(f"[SAVE] hitbox x={target_rect.x}, y={target_rect.y}, w={target_rect.w}, h={target_rect.h}")
                elif e.key == pg.K_r:
                    target_rect = pg.Rect(0,0,0,0)
                    try: space.remove(target_shape, target_body)
                    except: pass
                elif e.key == pg.K_a:
                    # 적용 & 숨김(편집 종료)
                    try: space.remove(target_shape, target_body)
                    except: pass
                    if target_rect.w > 0 and target_rect.h > 0:
                        target_body, target_shape = make_static_box(
                            space, target_rect.x, target_rect.y, target_rect.w, target_rect.h,
                            cfg["target_sensor"], PX_PER_M
                        )
                    edit_mode = False
                    print("[hitbox] applied & hidden (press E to edit)")
                elif e.key == pg.K_e:
                    # 편집 모드 복귀
                    edit_mode = True
                    print("[hitbox] edit mode ON")

            # 마우스 조작(편집 모드에서만)
            elif edit_mode and e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                pos = e.pos; hidx = hit_handle_index(target_rect, pos)
                if (target_rect.w == 0 and target_rect.h == 0) or hidx == -1:
                    dragging = True; drag_mode = -1; start_pt = pos
                    target_rect = pg.Rect(pos[0], pos[1], 0, 0)
                else:
                    dragging = True; drag_mode = hidx; start_pt = pos; start_rect = target_rect.copy()
            elif edit_mode and e.type == pg.MOUSEBUTTONUP and e.button == 1:
                dragging = False; drag_mode = -1; start_rect = None
                # 편집 중에도 물리 타깃 동기화
                try: space.remove(target_shape, target_body)
                except: pass
                if target_rect.w > 0 and target_rect.h > 0:
                    target_body, target_shape = make_static_box(
                        space, target_rect.x, target_rect.y, target_rect.w, target_rect.h,
                        cfg["target_sensor"], PX_PER_M
                    )

        # 드래그 적용
        if edit_mode and dragging:
            mx, my = pg.mouse.get_pos()
            if drag_mode == -1:
                x1,y1 = start_pt; x2,y2 = mx,my
                x, y = min(x1,x2), min(y1,y2); w, h = abs(x2-x1), abs(y2-y1)
                target_rect = pg.Rect(x,y,w,h)
            elif drag_mode == 8 and start_rect:
                dx, dy = mx-start_pt[0], my-start_pt[1]
                target_rect.x = start_rect.x + dx; target_rect.y = start_rect.y + dy
            elif start_rect:
                x,y,w,h = start_rect; rx,ry = mx-start_pt[0], my-start_pt[1]
                L,T,R,B = x, y, x+w, y+h
                if   drag_mode==0: L=x+rx; T=y+ry
                elif drag_mode==1: T=y+ry
                elif drag_mode==2: R=x+w+rx; T=y+ry
                elif drag_mode==3: L=x+rx
                elif drag_mode==4: R=x+w+rx
                elif drag_mode==5: L=x+rx; B=y+h+ry
                elif drag_mode==6: B=y+h+ry
                elif drag_mode==7: R=x+w+rx; B=y+h+ry
                nx, ny = min(L,R), min(T,B); nw, nh = abs(R-L), abs(B-T)
                target_rect = pg.Rect(nx, ny, nw, nh)

        # 카메라 프레임 → pygame
        ok, frame = cap.read()
        if not ok: continue
        if cfg["mirror"]: frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        screen.blit(np_rgb_to_surface(rgb), (0,0))

        # 타임 스텝
        t_now = time.time()
        dt = max(1e-3, t_now - (getattr(main, "_tprev", t_now)))
        main._tprev = t_now
        accum = (getattr(main, "_accum", 0.0) + dt)
        main._accum = accum

        # MediaPipe 추론
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = lm.detect_for_video(mp_img, int((t_now - t0)*1000))

        # 제스처 → 발사
        if result and result.hand_landmarks:
            H, W = rgb.shape[:2]
            for lms in result.hand_landmarks[:cfg["max_hands"]]:
                on_now = is_finger_gun(lms, False, cfg["EXT_ON"], cfg["EXT_OFF"], cfg["THUMB_MIN"])
                if on_now and (t_now - last_shot >= cfg["cooldown"]):
                    tip = lms[LANDMARK_INDEX_TIP]
                    cx, cy = int(tip.x*W), int(tip.y*H) + int(cfg["y_offset"])
                    ux, uy = finger_dir_unit(lms)
                    vx, vy = ux*cfg["shot_speed"], uy*cfg["shot_speed"]
                    blt = spawn_bullet(space, cx, cy, vx, vy, cfg["shot_life"], PX_PER_M)
                    bullets.append(blt); last_shot = t_now

        # 물리 고정 스텝
        while main._accum >= FIXED_DT:
            space.step(FIXED_DT); main._accum -= FIXED_DT

        # 탄환 렌더 (임팩트 프레임은 동적 리사이즈 후 사용)
        rgba_fx = effect.frame_rgba(t0)
        alive = []
        for blt in bullets:
            if (t_now - blt["birth"]) > blt["life"]:
                try: space.remove(blt["shape"], blt["body"])
                except: pass
                continue
            bx, by = blt["body"].position
            px, py = int(m2px(bx, PX_PER_M) - rgba_fx.shape[1]//2), int(m2px(by, PX_PER_M) - rgba_fx.shape[0]//2)
            overlay_rgba_onto_surface(screen, rgba_fx, px, py)
            alive.append(blt)
        bullets = alive

        # 편집 모드일 때만 히트박스/핸들 시각화
        if edit_mode and target_rect.w > 0 and target_rect.h > 0:
            pg.draw.rect(screen, (80,255,160), target_rect, width=2)
            for r in handle_rects(target_rect).values():
                pg.draw.rect(screen, (240,240,240), r); pg.draw.rect(screen, (40,40,40), r, 1)

        # 잔상(impact) 관리: 기본 0.0이라 표시 안 함
        if cfg["impact_life"] > 0.0:
            keep = []
            for it in impacts:
                if (t_now - it["t0"]) <= it["life"]:
                    effect.target_width = it["w"]
                    rgba = effect.frame_rgba(t0)
                    ex = int(it["x"] - rgba.shape[1]//2); ey = int(it["y"] - rgba.shape[0]//2)
                    overlay_rgba_onto_surface(screen, rgba, ex, ey)
                    keep.append(it)
            impacts = keep
        else:
            impacts.clear()

        # HUD
        mode_txt = "EDIT" if edit_mode else "APPLIED(HIDDEN)"
        hud = f"[{mode_txt}] A:apply&hide  E:edit  +/−:size  S:save  R:reset  Q/Esc:quit"
        screen.blit(font.render(hud, True, (0,255,255)), (8, 6))
        pg.display.flip()
        clock.tick(cfg["fps"])

    cap.release()
    pg.quit()

if __name__ == "__main__":
    main()

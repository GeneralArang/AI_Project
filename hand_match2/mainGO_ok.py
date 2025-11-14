#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Match Game — main2.py (canonical + Sound Volumes Split + GameOver BGM Pause)

- 내부 설정(외부 YAML 없음)
- 엄지: k-of-n 투표 + OFF 히스테리시스 + '4' 보정(엄지 Strict-OFF 재검사)
- Tkinter UI: 사이즈/시간/라이프, 카운트다운 → 진행 → 성공/실패/게임오버
- 왼쪽 UI:
    - 마인 Size, Time, Lives
    - BGM Volume (배경 음악 볼륨)
    - 효과음 Volume (정답/실패/게임오버 효과음 볼륨)
- 배경음: 한 곡(BGM_FILE) 루프 재생
- 효과음: 정답/실패/게임오버 각각 1개씩 (SFX_*_FILE)
- GAME OVER 시: BGM 잠시 멈추고, 게임오버 효과음만 재생 후 BGM 재개
- PiP(우하단 등) 손 랜드마크 미리보기 + 숫자 오버레이
- B.mp4(마인 영상) 크로마키 합성, mine.jpg 배경, 겹치지 않는 배치
- Enter=전체화면, Space=시작, S=정지
"""

import os, sys, time, random, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont

# =====================================================
# -------------------- 자원 경로 -----------------------
# =====================================================
# ⚙ 여기서부터 실행 파일/스크립트가 있는 폴더 기준으로 경로 설정
if getattr(sys, "frozen", False):
    # PyInstaller 등으로 빌드된 EXE로 실행될 때
    ROOT = os.path.dirname(sys.executable)
else:
    # 일반 파이썬 스크립트로 실행될 때
    ROOT = os.path.dirname(os.path.abspath(__file__))

MAIN_VIDEO_PATH  = os.path.join(ROOT, "B.mp4")     # '마인' 합성용 영상 경로
BACK_IMAGE_PATH  = os.path.join(ROOT, "mine.jpg")  # 배경 이미지 경로

# --------------------- 사운드 파일 --------------------
BGM_FILE          = os.path.join(ROOT, "bgm1.mp3")          # 배경 음악 1곡
SFX_CORRECT_FILE  = os.path.join(ROOT, "sfx_correct.mp3")   # 정답 효과음
SFX_WRONG_FILE    = os.path.join(ROOT, "sfx_wrong.mp3")     # 틀렸을 때 효과음
SFX_GAMEOVER_FILE = os.path.join(ROOT, "sfx_gameover.mp3")  # 게임오버 효과음

# =====================================================
# ----------------- 환경/렌더 설정 --------------------
# =====================================================
APP_FPS = 40                                       # 전체 앱 렌더/틱 빈도
FRAME_W, FRAME_H = 1280, 720                       # 카메라 캡처/캔버스 기본 크기
CAMERA_INDEX = 0                                   # 기본 카메라 인덱스
FULLSCREEN_START = False                           # 시작 시 풀스크린 켜기
LEFT_PANEL_VISIBLE = True                          # 시작 시 좌측 패널 표시
CAP_BUFFER_SIZE = 1                                # OpenCV 버퍼(지연 최소화)

# =====================================================
# ------------------ 라운드/게임 설정 -----------------
# =====================================================
DEFAULT_ROUND_TIME_S = 5                           # 기본 제한시간(초)
TIME_MIN, TIME_MAX = 3, 60                         # GUI 허용 범위(초)
COUNT_MIN, COUNT_MAX = 1, 10                       # 라운드 목표 숫자 범위
TARGET_MODE = "random"                             # 'random' | 'fixed'
TARGET_FIXED_VALUE = 3                             # TARGET_MODE='fixed'일 때 값
COUNTDOWN_MS_DEFAULT = 3000                        # 카운트다운(ms)
SUCCESS_HOLD_MS = 800                              # 성공 후 다음 라운드 진입 지연(ms)

# =====================================================
# --------------- 스프라이트(마인) 설정 ---------------
# =====================================================
MAIN_SIZE_PX_DEFAULT = 200                         # 마인 타일 기본 크기(px)
MAIN_MARGIN_PX = 24                                # 화면 가장자리/서로 간 여백(px)
MAIN_SIZE_MINMAX = (64, 240)                       # 좌측 슬라이더 최소/최대

# =====================================================
# ----------------- 손가락 안정화/모드 ----------------
# =====================================================
MIRROR_INPUT = True                                # 셀피 카메라용 미러
EMA_ALPHA = 0.45                                   # 지수평활(0~1, 클수록 최근값 반영↑)
DEBOUNCE_FRAMES = 8                                # 동일값 연속 프레임 요구
SKIP_DETECT_EVERY = 1                              # N프레임마다 1회만 검출(부하↓)

# MediaPipe 내부 파라미터
MP_MAX_HANDS = 2                                   # 최대 손 개수
MP_MODEL_COMPLEXITY = 1                            # 0/1/2(정확도↑↔속도↓)
MP_DET_CONF = 0.70                                 # min_detection_confidence
MP_TRK_CONF = 0.70                                 # min_tracking_confidence

# 손 모드: 'single' / 'sum2'(양손 합산) / HAND_ONLY: 'Left'|'Right' 제한
HAND_MODE = 'sum2'
HAND_ONLY = None

# =====================================================
# --------------------- 크로마키 -----------------------
# =====================================================
CHROMA_TOL = 32                                    # 크로마키 허용폭(클수록 관대)
FEATHER = 7                                        # 경계 페더링(커널 크기)
SPILL_FIX = True                                   # 초록 번짐 보정 사용(함수 내 포함)

# =====================================================
# ----------------------- PiP -------------------------
# =====================================================
PIP_W, PIP_H = 320, 180                            # PiP 크기(px)
PIP_MARGIN = 12                                    # 화면 모서리 여백(px)
PIP_VISIBLE = True                                 # PiP 표시 on/off
PIP_POS = "br"                                     # 'br','bl','tr','tl'

# =====================================================
# ------------------- 하단 표시(UI) -------------------
# =====================================================
HEART_CHAR    = "♥"                                # 하트 문자
DIVIDER_CHAR  = "|"                                # 구분자
WHITE = (255, 255, 255, 255)                       # RGBA 흰색

SCORE_COLOR   = WHITE                              # 점수 색
HEART_COLOR   = (255, 0, 0, 255)                   # 하트 색(빨강)
LIVES_COLOR   = WHITE                              # 라이프 색
DIVIDER_COLOR = WHITE                              # 구분자 색

SCORE_FONT_SIZE   = 40                             # 점수 폰트 크기
HEART_FONT_SIZE   = 44                             # 하트 폰트 크기
LIVES_FONT_SIZE   = 40                             # 라이프 폰트 크기
DIVIDER_FONT_SIZE = 40                             # 구분자 폰트 크기

GAP_SCORE_DIVIDER = 18                             # "Score"와 구분자 간격
GAP_DIVIDER_HEART = 18                             # 구분자와 하트 간격
GAP_HEART_LIVES   = 14                             # 하트와 "x L" 간격
BOTTOM_PADDING    = 12                             # 하단 여백(px)

# =====================================================
# ----------------- 디버그/녹화(옵션) -----------------
# =====================================================
THUMB_DEBUG = False                                # 엄지 지표 텍스트 오버레이
SHOW_LANDMARKS = True                              # 손 랜드마크 그리기 on/off
SAVE_SESSION_VIDEO = False                         # 최종 렌더 녹화 on/off

# =====================================================
# -------------------- 사운드 설정 --------------------
# =====================================================
SOUND_ENABLED       = True     # 전체 사운드 on/off
BGM_VOLUME_DEFAULT  = 0.3      # 배경음 기본 볼륨(0.0~1.0)
SFX_VOLUME_DEFAULT  = 1.0      # 효과음 기본 볼륨(0.0~1.0)

# =====================================================
# -------------------- MediaPipe ----------------------
# =====================================================
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    MEDIAPIPE = True
except Exception as e:
    print("[WARN] mediapipe import 실패:", e, file=sys.stderr)
    MEDIAPIPE = False

# =====================================================
# -------------------- pygame (사운드) ----------------
# =====================================================
try:
    import pygame
    PYGAME_AUDIO = True
except Exception as e:
    print("[WARN] pygame import 실패, 사운드 비활성화:", e, file=sys.stderr)
    PYGAME_AUDIO = False

# -----------------------------------------------------
# 유틸: 값 범위 클램프
# -----------------------------------------------------
def clamp(v, lo, hi):
    """값 v를 [lo, hi] 범위로 제한."""
    return max(lo, min(hi, v))

# -----------------------------------------------------
# 유틸: 두 사각형 겹침 여부
# -----------------------------------------------------
def rect_overlap(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    """(ax,ay,aw,ah)와 (bx,by,bw,bh)가 겹치면 True."""
    return not (ax+aw <= bx or bx+bw <= ax or ay+ah <= by or by+bh <= ay)

# -----------------------------------------------------
# 유틸: keepout 영역과 겹치면 살짝 밀어내기
# -----------------------------------------------------
def keepout_adjust(x, y, w, h, ko_rect):
    """(x,y,w,h) 영역이 금지영역 ko_rect와 겹치면 좌상 방향으로 조정."""
    kx, ky, kw, kh = ko_rect
    if not rect_overlap(x, y, w, h, kx, ky, kw, kh): return x, y
    nx = min(x, kx - w - 1); ny = min(y, ky - h - 1)
    return nx, ny

# -----------------------------------------------------
# 크로마키: 가장자리 색 히스토그램으로 키 컬러 추정 + 알파 생성
# -----------------------------------------------------
def chroma_key_rgba_keep_aspect(bgr: np.ndarray, max_px: int, tol: int=CHROMA_TOL, feather: int=FEATHER) -> Optional[Image.Image]:
    """BGR 프레임을 비율 유지 리사이즈 후 크로마키로 RGBA 이미지 생성."""
    if bgr is None: return None
    h0, w0 = bgr.shape[:2]
    if h0 == 0 or w0 == 0: return None

    scale = max_px / max(w0, h0)
    new_w, new_h = max(1, int(round(w0*scale))), max(1, int(round(h0*scale)))
    bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[...,0].astype(np.int16)
    b = 8
    border_h = np.concatenate([H[:b, :].ravel(), H[-b:, :].ravel(), H[:, :b].ravel(), H[:, -b:].ravel()])
    hist = np.bincount(border_h, minlength=180)
    key_h = int(hist.argmax())

    h_delta = max(18, min(32, int(tol*0.7)))
    h_lo = (key_h - h_delta) % 180
    h_hi = (key_h + h_delta) % 180

    s_min, v_min = 100, 40
    if h_lo <= h_hi:
        mask = cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))
    else:
        mask = (cv2.inRange(hsv, (0, s_min, v_min), (h_hi, 255, 255)) |
                cv2.inRange(hsv, (h_lo, s_min, v_min), (179, 255, 255)))

    k = 3 if feather <= 3 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    alpha = 255 - mask
    if feather > 0:
        kk = feather if feather % 2 else feather+1
        alpha = cv2.GaussianBlur(alpha, (kk, kk), 0)

    # 간단 디스필 보정(초록 번짐)
    B, G, R = cv2.split(bgr.copy())
    spill = (G.astype(np.int16) - np.maximum(R, B).astype(np.int16)).clip(0, 255).astype(np.uint8)
    atten = (mask.astype(np.float32)/255.0)*0.7 + 0.2
    G = np.clip(G.astype(np.float32) - spill*atten, 0, 255).astype(np.uint8)
    R = np.clip(R.astype(np.float32) + spill*(atten*0.25), 0, 255).astype(np.uint8)
    B = np.clip(B.astype(np.float32) + spill*(atten*0.25), 0, 255).astype(np.uint8)
    bgr_corr = cv2.merge([B, G, R])

    rgb  = cv2.cvtColor(bgr_corr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")

# =====================================================
# --------------- 엄지/손가락 판정 파라미터 -----------
# =====================================================

# ON 쪽 임계 (엄지를 '펴짐'으로 인정할 조건을 다소 빡빡하게)
THUMB_ON_LENRATIO   = 1.12
THUMB_ON_OUTSIDE    = 0.050
THUMB_ON_ANGLE_DEG  = 40.0
THUMB_ON_TIPIP      = 0.30
THUMB_ON_SIDE_X     = 0.032
THUMB_ON_IP_ANGLE   = 34.0

# OFF 쪽 임계 (엄지를 '접힘'으로 인정할 조건은 ON보다 느슨하게)
THUMB_OFF_LENRATIO  = 1.02
THUMB_OFF_OUTSIDE   = 0.030
THUMB_OFF_ANGLE_DEG = 28.0
THUMB_OFF_TIPIP     = 0.26
THUMB_OFF_SIDE_X    = 0.024
THUMB_OFF_IP_ANGLE  = 40.0

# 투표/폴백
THUMB_VOTES_ON_K    = 3
THUMB_VOTES_OFF_K   = 5
THUMB_FALLBACK      = True
THUMB_FALLBACK_DY   = 0.020
THUMB_FALLBACK_NEED = 2
THUMB_USE_HANDED    = False

# =====================================================
# ----------------- 손가락 카운터 클래스 ---------------
# =====================================================
class FingerCounter:
    """MediaPipe로 손가락 개수를 안정적으로 추정하는 클래스."""

    def __init__(self, mirror=MIRROR_INPUT):
        self.mirror = mirror
        self.ema = 0.0
        self.hist = []
        self.frame_cnt = 0
        self.last_vis_bgr = None
        self._thumb_on = False
        self._finger_on = {8:False, 12:False, 16:False, 20:False}

        if MEDIAPIPE:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MP_MAX_HANDS,
                model_complexity=MP_MODEL_COMPLEXITY,
                min_detection_confidence=MP_DET_CONF,
                min_tracking_confidence=MP_TRK_CONF
            )
        else:
            self.hands = None

    def close(self):
        """리소스 정리."""
        if self.hands:
            self.hands.close()

    @staticmethod
    def _dist(a, b):
        """랜드마크 a, b 사이의 2D 거리."""
        return ((a.x-b.x)**2 + (a.y-b.y)**2) ** 0.5

    def _open_thumb(self, lm, handed_label):
        """엄지 펴짐/접힘 판별(이력 기반 히스테리시스 적용)."""
        tip, ip, mcp = lm[4], lm[3], lm[2]
        idx_mcp, pinky_mcp = lm[5], lm[17]

        dist = self._dist
        scale = max(dist(idx_mcp, pinky_mcp), 1e-6)

        len_ratio    = dist(tip, mcp) / max(dist(ip, mcp), 1e-6)
        tip_ip_ratio = dist(tip, ip)  / max(dist(ip, mcp), 1e-6)

        vx, vy = (pinky_mcp.x - idx_mcp.x), (pinky_mcp.y - idx_mcp.y)
        def signed_area(p): return vx*(p.y-idx_mcp.y) - vy*(p.x-idx_mcp.y)
        s_tip, s_ip = signed_area(tip), signed_area(ip)
        outside_gain = 0.0
        if s_tip * s_ip > 0:
            gain = abs(s_tip) - abs(s_ip)
            outside_gain = max(0.0, gain/scale)

        ux, uy = (tip.x - mcp.x), (tip.y - mcp.y)
        vx2, vy2 = (idx_mcp.x - mcp.x), (idx_mcp.y - mcp.y)
        nu = max((ux*ux+uy*uy)**0.5, 1e-6)
        nv = max((vx2*vx2+vy2*vy2)**0.5, 1e-6)
        cosv = max(-1.0, min(1.0, (ux*vx2 + uy*vy2)/(nu*nv)))
        ang_deg = math.degrees(math.acos(cosv))

        side_mag = abs(tip.x - idx_mcp.x) / scale if not THUMB_USE_HANDED else (tip.x - idx_mcp.x)/scale

        ux2, uy2 = (tip.x - ip.x), (tip.y - ip.y)
        vx3, vy3 = (mcp.x - ip.x), (mcp.y - ip.y)
        nu2 = max((ux2*ux2+uy2*uy2)**0.5, 1e-6)
        nv3 = max((vx3*vx3+vy3*vy3)**0.5, 1e-6)
        cosv2 = max(-1.0, min(1.0, (ux2*vx3 + uy2*vy3)/(nu2*nv3)))
        ip_deg = math.degrees(math.acos(cosv2))

        votes_on = 0
        votes_on += 1 if (len_ratio    > THUMB_ON_LENRATIO)   else 0
        votes_on += 1 if (outside_gain > THUMB_ON_OUTSIDE)    else 0
        votes_on += 1 if (ang_deg      > THUMB_ON_ANGLE_DEG)  else 0
        votes_on += 1 if (tip_ip_ratio > THUMB_ON_TIPIP)      else 0
        votes_on += 1 if (abs(side_mag) > THUMB_ON_SIDE_X)    else 0
        votes_on += 1 if (ip_deg       < THUMB_ON_IP_ANGLE)   else 0

        votes_off = 0
        votes_off += 1 if (len_ratio    < THUMB_OFF_LENRATIO)   else 0
        votes_off += 1 if (outside_gain < THUMB_OFF_OUTSIDE)    else 0
        votes_off += 1 if (ang_deg      < THUMB_OFF_ANGLE_DEG)  else 0
        votes_off += 1 if (tip_ip_ratio < THUMB_OFF_TIPIP)      else 0
        votes_off += 1 if (abs(side_mag) < THUMB_OFF_SIDE_X)    else 0
        votes_off += 1 if (ip_deg       > THUMB_OFF_IP_ANGLE)   else 0

        if self._thumb_on:
            if votes_off >= THUMB_VOTES_OFF_K:
                self._thumb_on = False
        else:
            if votes_on >= THUMB_VOTES_ON_K:
                self._thumb_on = True
            elif THUMB_FALLBACK:
                weak = 0
                weak += 1 if (len_ratio    > (THUMB_ON_LENRATIO-0.02))   else 0
                weak += 1 if (outside_gain > (THUMB_ON_OUTSIDE-0.01))    else 0
                weak += 1 if (ang_deg      > (THUMB_ON_ANGLE_DEG-5.0))   else 0
                weak += 1 if (tip_ip_ratio > (THUMB_ON_TIPIP-0.04))      else 0
                weak += 1 if (abs(side_mag) > (THUMB_ON_SIDE_X-0.010))   else 0
                weak += 1 if (ip_deg       < (THUMB_ON_IP_ANGLE+6.0))    else 0
                dy_norm = (ip.y - tip.y) / scale
                if (weak >= THUMB_FALLBACK_NEED) and (dy_norm > THUMB_FALLBACK_DY):
                    self._thumb_on = True

        if THUMB_DEBUG and self.last_vis_bgr is not None:
            txt = f"len {len_ratio:.2f} out {outside_gain:.3f} ang {ang_deg:.1f} tipip {tip_ip_ratio:.2f} side {abs(side_mag):.3f} ip {ip_deg:.1f}"
            cv2.putText(self.last_vis_bgr, txt, (10, 20 + 18*(self.frame_cnt%20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        return self._thumb_on

    def _thumb_strict_closed(self, lm) -> bool:
        """'4' 보정: 엄지가 애매한 ON일 때 충분히 닫힘인지 재검사."""
        tip, ip, mcp = lm[4], lm[3], lm[2]
        idx_mcp, pinky_mcp = lm[5], lm[17]
        dist = self._dist
        scale = max(dist(idx_mcp, pinky_mcp), 1e-6)

        len_ratio    = dist(tip, mcp) / max(dist(ip, mcp), 1e-6)
        tip_ip_ratio = dist(tip, ip)  / max(dist(ip, mcp), 1e-6)

        vx, vy = (pinky_mcp.x - idx_mcp.x), (pinky_mcp.y - idx_mcp.y)
        def signed_area(p): return vx*(p.y-idx_mcp.y) - vy*(p.x-idx_mcp.y)
        s_tip, s_ip = signed_area(tip), signed_area(ip)
        outside_gain = 0.0
        if s_tip * s_ip > 0:
            gain = abs(s_tip) - abs(s_ip)
            outside_gain = max(0.0, gain/scale)

        ux, uy = (tip.x - mcp.x), (tip.y - mcp.y)
        vx2, vy2 = (idx_mcp.x - mcp.x), (idx_mcp.y - mcp.y)
        nu = max((ux*ux+uy*uy)**0.5, 1e-6)
        nv = max((vx2*vx2+vy2*vy2)**0.5, 1e-6)
        cosv = max(-1.0, min(1.0, (ux*vx2 + uy*vy2)/(nu*nv)))
        ang_deg = math.degrees(math.acos(cosv))

        side_mag = abs(tip.x - idx_mcp.x) / scale

        ux2, uy2 = (tip.x - ip.x), (tip.y - ip.y)
        vx3, vy3 = (mcp.x - ip.x), (mcp.y - ip.y)
        nu2 = max((ux2*ux2+uy2*uy2)**0.5, 1e-6)
        nv3 = max((vx3*vx3+vy3*vy3)**0.5, 1e-6)
        cosv2 = max(-1.0, min(1.0, (ux2*vx3 + uy2*vy3)/(nu2*nv3)))
        ip_deg = math.degrees(math.acos(cosv2))

        strict_votes = 0
        strict_votes += 1 if (len_ratio    < THUMB_OFF_LENRATIO - 0.03) else 0
        strict_votes += 1 if (outside_gain < THUMB_OFF_OUTSIDE - 0.010) else 0
        strict_votes += 1 if (ang_deg      < THUMB_OFF_ANGLE_DEG - 4.0) else 0
        strict_votes += 1 if (tip_ip_ratio < THUMB_OFF_TIPIP - 0.03) else 0
        strict_votes += 1 if (side_mag     < THUMB_OFF_SIDE_X - 0.008) else 0
        strict_votes += 1 if (ip_deg       > THUMB_OFF_IP_ANGLE + 6.0) else 0

        dy_norm = (tip.y - ip.y) / scale
        strict_votes += 1 if (dy_norm > 0.0) else 0

        return strict_votes >= 4

    def _open_finger(self, lm, tip, pip, mcp):
        """일반 손가락의 펴짐 판정(위치차/관절각 기반)."""
        wrist = lm[0]; mid_mcp = lm[9]
        scale = ((wrist.x-mid_mcp.x)**2 + (wrist.y-mid_mcp.y)**2) ** 0.5
        scale = max(scale, 1e-6)
        DY_ON, DY_OFF = 0.070*scale, 0.050*scale
        tip_dy = (lm[pip].y - lm[tip].y)
        pip_dy = (lm[mcp].y - lm[pip].y)
        ang_on, ang_off = 40.0, 32.0
        ux, uy = (lm[tip].x - lm[pip].x), (lm[tip].y - lm[pip].y)
        vx, vy = (lm[mcp].x - lm[pip].x), (lm[mcp].y - lm[pip].y)
        nu = max((ux*ux+uy*uy)**0.5, 1e-6)
        nv = max((vx*vx+vy*vy)**0.5, 1e-6)
        cosv = max(-1.0, min(1.0, (ux*vx + uy*vy)/(nu*nv)))
        ang = math.degrees(math.acos(cosv))
        was_on = self._finger_on[tip]
        if was_on:
            on = (tip_dy > DY_OFF) and (pip_dy > 0.015*scale) and (ang > ang_off)
        else:
            on = (tip_dy > DY_ON)  and (pip_dy > 0.020*scale) and (ang > ang_on)
        self._finger_on[tip] = on
        return on

    def count(self, frame_bgr: np.ndarray) -> int:
        """현재 프레임을 입력받아 손가락 개수를 안정화하여 반환."""
        self.frame_cnt += 1
        if not MEDIAPIPE or self.hands is None:
            self.last_vis_bgr = frame_bgr.copy()
            return 0

        img = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        vis = img.copy()
        total = 0

        if res.multi_hand_landmarks:
            candidates = []
            for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                handed_label = hd.classification[0].label
                if HAND_ONLY is not None and handed_label != HAND_ONLY:
                    continue
                lm = lms.landmark

                b_thumb = self._open_thumb(lm, handed_label)
                b_i = self._open_finger(lm, 8, 6, 5)
                b_m = self._open_finger(lm,12,10, 9)
                b_r = self._open_finger(lm,16,14,13)
                b_p = self._open_finger(lm,20,18,17)

                non_thumb_open = (1 if b_i else 0) + (1 if b_m else 0) + (1 if b_r else 0) + (1 if b_p else 0)
                if non_thumb_open == 4 and b_thumb:
                    if self._thumb_strict_closed(lm):
                        b_thumb = False
                        self._thumb_on = False

                cnt = (1 if b_thumb else 0) + non_thumb_open

                idx_mcp, pinky_mcp = lm[5], lm[17]
                hand_score = ((idx_mcp.x - pinky_mcp.x)**2 + (idx_mcp.y - pinky_mcp.y)**2) ** 0.5
                candidates.append((hand_score, cnt, lms))

            candidates.sort(key=lambda x: x[0], reverse=True)
            if HAND_MODE == 'single':
                if candidates:
                    total = candidates[0][1]
                    if SHOW_LANDMARKS:
                        mp_draw.draw_landmarks(
                            vis, candidates[0][2], mp_hands.HAND_CONNECTIONS,
                            mp_style.get_default_hand_landmarks_style(),
                            mp_style.get_default_hand_connections_style()
                        )
            else:  # sum2
                picked = candidates[:2]
                total = sum(c[1] for c in picked)
                if SHOW_LANDMARKS:
                    for _, _, lms in picked:
                        mp_draw.draw_landmarks(
                            vis, lms, mp_hands.HAND_CONNECTIONS,
                            mp_style.get_default_hand_landmarks_style(),
                            mp_style.get_default_hand_connections_style()
                        )

        self.last_vis_bgr = vis
        self.ema = (1-EMA_ALPHA)*self.ema + EMA_ALPHA*total
        rounded = int(round(self.ema))
        self._push_hist(rounded)
        return self._stable()

    def _push_hist(self, v:int):
        """최근 N프레임 값을 큐에 저장(디바운스용)."""
        self.hist.append(v)
        if len(self.hist) > DEBOUNCE_FRAMES:
            self.hist.pop(0)

    def _stable(self):
        """큐가 모두 동일하면 그 값을, 아니면 최빈값을 반환."""
        return self.hist[0] if len(set(self.hist)) == 1 else max(set(self.hist), key=self.hist.count)

    def last_pip_rgba(self, w=PIP_W, h=PIP_H) -> Optional[Image.Image]:
        """최근 시각화 BGR을 PiP 크기로 변환해 RGBA로 반환."""
        if self.last_vis_bgr is None: return None
        bgr = cv2.resize(self.last_vis_bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGBA")

# =====================================================
# ------------------- 사운드 매니저 --------------------
# =====================================================
class SoundManager:
    """배경음 + 효과음 재생 관리 (볼륨 분리 + GameOver 시 BGM pause)."""

    def __init__(
        self,
        bgm_file: str,
        sfx_correct: str,
        sfx_wrong: str,
        sfx_gameover: str,
        bgm_volume: float = BGM_VOLUME_DEFAULT,
        sfx_volume: float = SFX_VOLUME_DEFAULT,
    ):
        self.enabled = SOUND_ENABLED and PYGAME_AUDIO
        self.bgm_file = bgm_file
        self.sfx_paths = {
            "correct": sfx_correct,
            "wrong": sfx_wrong,
            "gameover": sfx_gameover,
        }
        self.bgm_volume = clamp(bgm_volume, 0.0, 1.0)
        self.sfx_volume = clamp(sfx_volume, 0.0, 1.0)
        self.sfx = {"correct": None, "wrong": None, "gameover": None}
        self._bgm_started = False   # BGM이 실제로 시작되었는지

        if not self.enabled:
            return

        try:
            pygame.mixer.init()
        except Exception as e:
            print("[WARN] pygame.mixer 초기화 실패, 사운드 비활성화:", e, file=sys.stderr)
            self.enabled = False
            return

        # 배경음 존재 여부만 체크
        if not os.path.exists(self.bgm_file):
            print(f"[INFO] BGM 파일 없음: {self.bgm_file}", file=sys.stderr)

        # 효과음 로드
        for key, path in self.sfx_paths.items():
            if not path or not os.path.exists(path):
                print(f"[INFO] 효과음 없음({key}): {path}", file=sys.stderr)
                continue
            try:
                snd = pygame.mixer.Sound(path)
                snd.set_volume(self.sfx_volume)
                self.sfx[key] = snd
            except Exception as e:
                print(f"[WARN] 효과음 로드 실패 {key}: {path} ({e})", file=sys.stderr)

        pygame.mixer.music.set_volume(self.bgm_volume)

    # ---- 볼륨 조절 ----
    def set_bgm_volume(self, v: float):
        self.bgm_volume = clamp(v, 0.0, 1.0)
        if not self.enabled:
            return
        pygame.mixer.music.set_volume(self.bgm_volume)

    def set_sfx_volume(self, v: float):
        self.sfx_volume = clamp(v, 0.0, 1.0)
        if not self.enabled:
            return
        for snd in self.sfx.values():
            if snd:
                snd.set_volume(self.sfx_volume)

    # ---- BGM 제어 ----
    def start_bgm(self):
        """배경음 루프 재생."""
        if not self.enabled: return
        if not os.path.exists(self.bgm_file):
            return
        try:
            pygame.mixer.music.load(self.bgm_file)
            pygame.mixer.music.set_volume(self.bgm_volume)
            pygame.mixer.music.play(-1)
            self._bgm_started = True
        except Exception as e:
            print(f"[WARN] BGM 재생 실패: {self.bgm_file} ({e})", file=sys.stderr)

    def pause_bgm(self):
        if not self.enabled: return
        if not self._bgm_started: return
        try:
            pygame.mixer.music.pause()
        except Exception:
            pass

    def resume_bgm_after_gameover(self):
        """게임오버 효과음 이후 BGM 재개."""
        if not self.enabled: return
        if not self._bgm_started:
            # 혹시 초기 재생 실패/미시작이었다면 다시 시도
            self.start_bgm()
            return
        try:
            pygame.mixer.music.unpause()
        except Exception:
            # 혹시 실패하면 다시 로드/플레이 시도
            try:
                pygame.mixer.music.load(self.bgm_file)
                pygame.mixer.music.set_volume(self.bgm_volume)
                pygame.mixer.music.play(-1)
            except Exception:
                pass

    # ---- SFX 재생 ----
    def play_correct(self):
        if not self.enabled: return
        snd = self.sfx.get("correct")
        if snd: snd.play()

    def play_wrong(self):
        if not self.enabled: return
        snd = self.sfx.get("wrong")
        if snd: snd.play()

    def play_gameover(self) -> float:
        """
        게임오버 효과음 재생.
        - BGM은 일시 정지(pause)
        - 재생 길이(초)를 반환해서, 호출 측에서 그만큼 기다렸다 BGM 재개 가능.
        """
        if not self.enabled:
            return 0.0
        snd = self.sfx.get("gameover")
        if not snd:
            return 0.0
        # BGM 잠시 멈추고
        self.pause_bgm()
        # 효과음 재생
        snd.play()
        try:
            length = snd.get_length()
        except Exception:
            length = 0.0
        return length

    def close(self):
        if not self.enabled: return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass

# -----------------------------------------------------
# 포지션 생성: 서로 겹치지 않게 N개 좌표 생성
# -----------------------------------------------------
def non_overlapping_positions(n, w, h, approx_size_px, margin, keepouts: List[Tuple[int,int,int,int]]):
    """서로 겹치지 않는 스프라이트 중심 좌표 n개를 생성."""
    pos, tries, mind = [], 0, int(approx_size_px*0.9)
    while len(pos) < n and tries < 2000:
        x = random.randint(margin+approx_size_px//2, w-margin-approx_size_px//2)
        y = random.randint(margin+approx_size_px//2, h-margin+approx_size_px//2)
        ok = True
        for px,py in pos:
            if math.hypot(x-px, y-py) < mind:
                ok = False; break
        if ok:
            half = approx_size_px//2
            for kx,ky,kw,kh in keepouts:
                if rect_overlap(x-half, y-half, approx_size_px, approx_size_px, kx,ky,kw,kh):
                    ok = False; break
        if ok: pos.append((x,y))
        tries += 1
    while len(pos) < n:  # 실패 시 랜덤 채움
        pos.append((
            random.randint(margin+approx_size_px//2, w-margin+approx_size_px//2),
            random.randint(margin+approx_size_px//2, h-margin+approx_size_px//2)
        ))
    return pos

# -----------------------------------------------------
# 상태 구조체
# -----------------------------------------------------
@dataclass
class GameState:
    """라운드/게임 진행 상태를 담는 구조체."""
    target: int = 1
    left_ms: int = DEFAULT_ROUND_TIME_S*1000
    phase: str = "ROUND_INIT"     # ROUND_INIT, COUNTDOWN, RUNNING, SUCCESS, FAIL, GAMEOVER
    fingers: int = 0
    main_pos: List[Tuple[int,int]] = field(default_factory=list)
    last_ms: int = 0
    success_ms: int = 0
    success_hold: int = SUCCESS_HOLD_MS
    countdown_ms: int = 0
    lives: int = 3
    score: int = 0

# =====================================================
# ------------------------ 앱 -------------------------
# =====================================================
class App:
    """Tkinter 기반 메인 애플리케이션."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hand Match Game — 마인")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.bind("<Return>", self.toggle_fullscreen)   # Enter=전체화면

        # 단축키
        self.root.bind("<space>", self._on_space)
        self.root.bind("<s>", self._on_stop_key)
        self.root.bind("<S>", self._on_stop_key)

        # 입력칸 포커스아웃 처리
        self.root.bind_all("<Button-1>", self._maybe_apply_on_click, add="+")

        self.is_fullscreen = False
        self.left_visible = True
        self._reset_on_start = True

        # 레이아웃
        self.paned = ttk.Panedwindow(root, orient="horizontal")
        self.left  = ttk.Frame(self.paned, width=300)
        self.right = ttk.Frame(self.paned)
        self.paned.add(self.left, weight=0)
        self.paned.add(self.right, weight=1)
        self.paned.pack(fill="both", expand=True)

        if not LEFT_PANEL_VISIBLE:
            try: self.paned.forget(self.left)
            except Exception: pass
            self.left_visible = False

        self.canvas = tk.Canvas(self.right, bg="black", width=FRAME_W, height=FRAME_H, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # ---- 좌측 GUI ----
        self.lbl_info = ttk.Label(self.left, text="Start to play!", anchor="w")
        self.lbl_info.pack(fill="x", padx=10, pady=(10,4))

        self.btn_start = ttk.Button(self.left, text="Start Round (Space)", command=self.start_round)
        self.btn_start.pack(fill="x", padx=10, pady=4)

        self.btn_stop  = ttk.Button(self.left, text="Stop (S)", command=self.stop_round)
        self.btn_stop.pack(fill="x", padx=10, pady=(0,6))

        ttk.Separator(self.left).pack(fill="x", padx=10, pady=6)

        ttk.Label(self.left, text="마인 Size (px)").pack(fill="x", padx=10, pady=(8,2))
        self.var_size = tk.IntVar(value=MAIN_SIZE_PX_DEFAULT)
        self.scale_size = ttk.Scale(
            self.left, from_=MAIN_SIZE_MINMAX[0], to=MAIN_SIZE_MINMAX[1],
            orient="horizontal", variable=self.var_size,
            command=lambda v: self._on_size(int(float(v)))
        )
        self.scale_size.pack(fill="x", padx=10, pady=(0,8))

        ttk.Label(self.left, text="Time Limit (sec)").pack(fill="x", padx=10, pady=(8,2))
        self.var_time_limit = tk.IntVar(value=DEFAULT_ROUND_TIME_S)
        self.spin_time = ttk.Spinbox(self.left, from_=TIME_MIN, to=TIME_MAX, textvariable=self.var_time_limit, width=6)
        self.spin_time.pack(fill="x", padx=10, pady=(0,8))

        ttk.Label(self.left, text="Lives").pack(fill="x", padx=10, pady=(8,2))
        self.var_lives = tk.IntVar(value=3)
        self.spin_lives = ttk.Spinbox(self.left, from_=1, to=9, textvariable=self.var_lives, width=6)
        self.spin_lives.pack(fill="x", padx=10, pady=(0,8))

        self.spin_time.bind("<FocusOut>", lambda e: self._apply_gui_settings())
        self.spin_lives.bind("<FocusOut>", lambda e: self._apply_gui_settings())
        self.spin_time.bind("<Return>", lambda e: (self.apply_and_defocus(), "break"))
        self.spin_lives.bind("<Return>", lambda e: (self.apply_and_defocus(), "break"))

        # ---- 사운드 UI: BGM / 효과음 볼륨 분리 ----
        ttk.Separator(self.left).pack(fill="x", padx=10, pady=8)

        ttk.Label(self.left, text="BGM Volume").pack(fill="x", padx=10, pady=(4,2))
        self.var_bgm_volume = tk.IntVar(value=int(BGM_VOLUME_DEFAULT * 100))
        self.scale_bgm_volume = ttk.Scale(
            self.left, from_=0, to=100,
            orient="horizontal", variable=self.var_bgm_volume,
            command=lambda v: self._on_bgm_volume(float(v))
        )
        self.scale_bgm_volume.pack(fill="x", padx=10, pady=(0,8))

        ttk.Label(self.left, text="효과음 Volume").pack(fill="x", padx=10, pady=(4,2))
        self.var_sfx_volume = tk.IntVar(value=int(SFX_VOLUME_DEFAULT * 100))
        self.scale_sfx_volume = ttk.Scale(
            self.left, from_=0, to=100,
            orient="horizontal", variable=self.var_sfx_volume,
            command=lambda v: self._on_sfx_volume(float(v))
        )
        self.scale_sfx_volume.pack(fill="x", padx=10, pady=(0,8))

        # 카메라
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAP_BUFFER_SIZE)

        # 리소스
        if not os.path.exists(BACK_IMAGE_PATH):
            print(f"[ERROR] 배경 이미지 없음: {BACK_IMAGE_PATH}", file=sys.stderr); sys.exit(1)
        self.bg_base = Image.open(BACK_IMAGE_PATH).convert("RGBA")
        if not os.path.exists(MAIN_VIDEO_PATH):
            print(f"[ERROR] 마인 영상 없음: {MAIN_VIDEO_PATH}", file=sys.stderr); sys.exit(1)
        self.main = MainSprite(cv2.VideoCapture(MAIN_VIDEO_PATH), size_px=self.var_size.get())

        # 손가락 카운터
        self.counter = FingerCounter(mirror=MIRROR_INPUT)

        # 사운드 매니저
        self.snd = SoundManager(
            bgm_file=BGM_FILE,
            sfx_correct=SFX_CORRECT_FILE,
            sfx_wrong=SFX_WRONG_FILE,
            sfx_gameover=SFX_GAMEOVER_FILE,
            bgm_volume=BGM_VOLUME_DEFAULT,
            sfx_volume=SFX_VOLUME_DEFAULT,
        )
        # 슬라이더 값 → 실제 볼륨 반영
        self._on_bgm_volume(self.var_bgm_volume.get())
        self._on_sfx_volume(self.var_sfx_volume.get())
        # 배경음 재생 시작
        if self.snd.enabled:
            self.snd.start_bgm()
        else:
            self.lbl_info.config(text="사운드 비활성화(pygame or 파일 문제)", foreground="red")

        # 상태
        self.state = GameState()
        self.tkimg = None
        self.state.last_ms = self._now()

        # 녹화 준비
        self._writer = None

        if FULLSCREEN_START and not self.is_fullscreen:
            self.toggle_fullscreen()

        self.root.after(int(1000/APP_FPS), self.tick)

    # ---------- 사운드 UI 콜백 ----------
    def _on_bgm_volume(self, v):
        """BGM 볼륨 슬라이더 콜백 (0~100)."""
        try:
            val = float(v) / 100.0
        except Exception:
            val = BGM_VOLUME_DEFAULT
        val = clamp(val, 0.0, 1.0)
        if self.snd:
            self.snd.set_bgm_volume(val)

    def _on_sfx_volume(self, v):
        """효과음 볼륨 슬라이더 콜백 (0~100)."""
        try:
            val = float(v) / 100.0
        except Exception:
            val = SFX_VOLUME_DEFAULT
        val = clamp(val, 0.0, 1.0)
        if self.snd:
            self.snd.set_sfx_volume(val)

    # ---------- 기타 UI/로직 ----------
    def _maybe_apply_on_click(self, event):
        """클릭으로 입력창 포커스가 빠질 때 설정 적용."""
        if self._focused_is_text_input():
            self._apply_gui_settings()
            self.canvas.focus_set()

    def _focused_is_text_input(self) -> bool:
        """Entry/Spinbox 등 텍스트 입력 위젯에 포커스인지 확인."""
        fw = self.root.focus_get()
        if fw is None: return False
        cls = str(fw.winfo_class()).lower()
        return ('entry' in cls) or ('spinbox' in cls)

    def apply_and_defocus(self):
        """GUI 값을 적용하고 포커스를 캔버스로 이동."""
        self._apply_gui_settings()
        self.canvas.focus_set()
        self.lbl_info.config(text="설정 적용됨", foreground="blue")

    def _apply_gui_settings(self):
        """시간/라이프 입력값을 검증하고 상태에 반영."""
        try: lim = int(self.var_time_limit.get())
        except: lim = DEFAULT_ROUND_TIME_S
        lim = clamp(lim, TIME_MIN, TIME_MAX)
        self.var_time_limit.set(lim)

        try: lives = int(self.var_lives.get())
        except: lives = 3
        lives = clamp(lives, 1, 9)
        self.var_lives.set(lives)
        if self.state.phase in ("ROUND_INIT", "GAMEOVER"):
            self.state.lives = lives

    def _on_space(self, e):
        """스페이스 키로 라운드 시작."""
        if not self._focused_is_text_input():
            self.start_round()

    def _on_stop_key(self, e):
        """S 키로 중지."""
        if not self._focused_is_text_input():
            self.stop_round()

    def toggle_fullscreen(self, _=None):
        """Enter 키로 전체화면 토글 및 좌측 패널 표시/숨김."""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        if self.is_fullscreen and self.left_visible:
            try: self.paned.forget(self.left)
            except Exception: pass
            self.left_visible = False
        elif not self.is_fullscreen and not self.left_visible:
            try: self.paned.insert(0, self.left)
            except Exception: self.paned.add(self.left, weight=0)
            self.left_visible = True

    def _on_size(self, sz:int):
        """슬라이더로 마인 크기 변경."""
        self.main.size_px = int(sz)

    def _now(self):
        """현재 시간을 ms로 반환."""
        return int(time.time()*1000)

    def _canvas_size(self):
        """캔버스의 현재 크기를 반환."""
        w = int(self.canvas.winfo_width()  or FRAME_W)
        h = int(self.canvas.winfo_height() or FRAME_H)
        return max(64, w), max(64, h)

    def _pip_rect(self, cw, ch):
        """PiP 위치/크기를 반환 (x, y, w, h)."""
        if PIP_POS == "br":
            return (cw - PIP_MARGIN - PIP_W, ch - PIP_MARGIN - PIP_H, PIP_W, PIP_H)
        if PIP_POS == "bl":
            return (PIP_MARGIN, ch - PIP_MARGIN - PIP_H, PIP_W, PIP_H)
        if PIP_POS == "tr":
            return (cw - PIP_MARGIN - PIP_W, PIP_MARGIN, PIP_W, PIP_H)
        return (PIP_MARGIN, PIP_MARGIN, PIP_W, PIP_H)

    def start_round(self):
        """라운드를 초기화하고 카운트다운 시작."""
        self._apply_gui_settings()

        try: lim = int(self.var_time_limit.get())
        except: lim = DEFAULT_ROUND_TIME_S
        lim = clamp(lim, TIME_MIN, TIME_MAX)
        self.var_time_limit.set(lim)

        try: lives = int(self.var_lives.get())
        except: lives = 3
        lives = clamp(lives, 1, 9)

        if self._reset_on_start:
            self.state.score = 0
            self._reset_on_start = False

        if self.state.phase == "GAMEOVER":
            self.state.lives = lives
        elif self.state.phase in ("ROUND_INIT", "SUCCESS", "FAIL") and self.state.lives <= 0:
            self.state.lives = lives
        elif self.state.phase == "ROUND_INIT" and self.state.lives == 0:
            self.state.lives = lives

        if TARGET_MODE == "fixed":
            self.state.target = int(TARGET_FIXED_VALUE)
        else:
            self.state.target = random.randint(COUNT_MIN, COUNT_MAX)

        self.state.countdown_ms = COUNTDOWN_MS_DEFAULT
        self.state.phase = "COUNTDOWN"
        self.state.main_pos = []
        self.lbl_info.config(text="카운트다운 중…", foreground="blue")
        self._next_round_time_limit_ms = lim * 1000

    def _enter_running_after_countdown(self):
        """카운트다운 종료 시 라운드 시작 세팅."""
        self.state.left_ms = getattr(self, "_next_round_time_limit_ms", DEFAULT_ROUND_TIME_S*1000)
        cw, ch = self._canvas_size()
        keepouts = []
        if PIP_VISIBLE:
            kx, ky, kw, kh = self._pip_rect(cw, ch)
            keepouts.append((kx, ky, kw, kh))
        self.state.main_pos = non_overlapping_positions(
            self.state.target, cw, ch, self.main.size_px, MAIN_MARGIN_PX, keepouts
        )
        self.state.phase = "RUNNING"
        self.lbl_info.config(text="라운드 진행 중…", foreground="black")

    def stop_round(self):
        """라운드를 중지하고 초기 상태로 전환."""
        self._reset_on_start = True
        self.state.phase = "ROUND_INIT"
        self.state.main_pos = []
        self.lbl_info.config(text="정지됨. Space로 시작", foreground="gray")

    def on_success(self):
        """성공 시 스코어 증가 및 SUCCESS 상태 진입."""
        self.state.score += 1
        self.lbl_info.config(text="✅ 성공!", foreground="green")
        if self.snd:
            self.snd.play_correct()
        self.state.phase = "SUCCESS"
        self.state.success_ms = 0

    def _resume_bgm_after_gameover(self):
        """게임오버 효과음 재생 후 BGM 다시 켜기."""
        if self.snd:
            self.snd.resume_bgm_after_gameover()

    def on_fail(self):
        """실패 시 라이프 감소 또는 게임오버."""
        self.state.lives = max(0, self.state.lives - 1)
        if self.state.lives <= 0:
            # GAME OVER 처리
            self.state.phase = "GAMEOVER"
            self._reset_on_start = True
            self.lbl_info.config(text="💀 Game Over! Space로 새 게임", foreground="red")

            if self.snd:
                # 게임오버 효과음 재생 + 길이(sec) 받아오기
                duration = self.snd.play_gameover()
                # 효과음이 유효한 길이를 가지면, 끝난 뒤 BGM 재개 예약
                if duration > 0:
                    self.root.after(int(duration * 1000) + 100, self._resume_bgm_after_gameover)
            return

        # 일반 실패 (라이프 남아있음)
        self.lbl_info.config(text="❌ 실패! 다시 도전", foreground="red")
        if self.snd:
            self.snd.play_wrong()
        self.state.phase = "FAIL"
        self.root.after(800, self.start_round)

    def on_close(self):
        """종료 시 카메라/비디오/녹화/MP/사운드 리소스 정리."""
        try:
            if self.cap: self.cap.release()
            if self.main.cap: self.main.cap.release()
            if self._writer is not None:
                self._writer.release()
            self.counter.close()
            if self.snd:
                self.snd.close()
        finally:
            self.root.destroy()

    def _load_font(self, size):
        """지정 크기의 폰트를 OS에 맞는 폰트로 로딩, 실패 시 기본폰트."""
        if os.name == "nt":  # Windows
            candidates = [
                r"C:\Windows\Fonts\malgunbd.ttf",
                r"C:\Windows\Fonts\malgun.ttf",
            ]
        else:  # Linux / 우분투
            candidates = [
                "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "DejaVuSans-Bold.ttf",
            ]

        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

        print(f"[WARN] 지정 폰트 로드 실패, 기본 폰트 사용 (size={size})", file=sys.stderr)
        return ImageFont.load_default()

    def _draw_text_shadow(self, draw, xy, text, font, fill, anchor="mm"):
        """흑색 얇은 그림자를 4방향으로 깔고 본문 텍스트 렌더."""
        x, y = xy
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
            draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0,160), anchor=anchor)
        draw.text((x, y), text, font=font, fill=fill, anchor=anchor)

    def _draw_top_center_time(self, bg_img, seconds: int):
        """상단 중앙에 남은 시간(초) 표시."""
        draw = ImageDraw.Draw(bg_img)
        W, _ = bg_img.size
        font = self._load_font(48)
        self._draw_text_shadow(draw, (W//2, 34), f"{seconds}s", font, WHITE, anchor="mm")

    def _draw_center_banner(self, bg_img, text, fill=(50,120,255,230)):
        """가운데 라운드 상태 배너 렌더."""
        draw = ImageDraw.Draw(bg_img)
        W, H = bg_img.size
        pad_x, pad_y = 28, 14
        radius = 18
        font_big  = self._load_font(44)
        tw, th = draw.textbbox((0, 0), text, font=font_big)[2:]
        bw, bh = tw + pad_x*2, th + pad_y*2
        cx, cy = W//2, H//2
        x0, y0 = cx - bw//2, cy - bh//2
        x1, y1 = x0 + bw, y0 + bh
        draw.rounded_rectangle([x0+4, y0+6, x1+4, y1+6], radius=radius, fill=(0,0,0,130))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, outline=(255,255,255,180), width=2)
        self._draw_text_shadow(draw, (cx, cy), text, font=font_big, fill=WHITE, anchor="mm")

    def _draw_bottom_center_score(self, bg_img, score: int, lives: int):
        """하단 중앙에 Score | ♥ x L 형식으로 렌더."""
        draw = ImageDraw.Draw(bg_img)
        W, H = bg_img.size

        f_score   = self._load_font(SCORE_FONT_SIZE)
        f_divider = self._load_font(DIVIDER_FONT_SIZE)
        f_heart   = self._load_font(HEART_FONT_SIZE)
        f_lives   = self._load_font(LIVES_FONT_SIZE)

        t_score   = f"Score: {score}"
        t_divider = DIVIDER_CHAR
        t_heart   = HEART_CHAR
        t_lives   = f"x {lives}"

        sw, sh = draw.textbbox((0,0), t_score,   font=f_score)[2:]
        dw, dh = draw.textbbox((0,0), t_divider, font=f_divider)[2:]
        hw, hh = draw.textbbox((0,0), t_heart,   font=f_heart)[2:]
        lw, lh = draw.textbbox((0,0), t_lives,   font=f_lives)[2:]

        total_w = sw + GAP_SCORE_DIVIDER + dw + GAP_DIVIDER_HEART + hw + GAP_HEART_LIVES + lw
        base_y  = H - max(sh, dh, hh, lh)//2 - BOTTOM_PADDING
        start_x = W//2 - total_w//2
        x = start_x

        self._draw_text_shadow(draw, (x+sw, base_y), t_score, f_score, SCORE_COLOR, anchor="rm")
        x += sw + GAP_SCORE_DIVIDER
        self._draw_text_shadow(draw, (x, base_y), t_divider, f_divider, DIVIDER_COLOR, anchor="lm")
        x += dw + GAP_DIVIDER_HEART
        self._draw_text_shadow(draw, (x, base_y), t_heart, f_heart, HEART_COLOR, anchor="lm")
        x += hw + GAP_HEART_LIVES
        self._draw_text_shadow(draw, (x, base_y), t_lives, f_lives, LIVES_COLOR, anchor="lm")

    def tick(self):
        """주기적으로 호출되어 프레임 처리 및 렌더를 수행."""
        now = self._now()
        dt = now - self.state.last_ms
        self.state.last_ms = now

        ok, cam = self.cap.read()
        if not ok:
            self.root.after(int(1000/APP_FPS), self.tick)
            return

        self.main.tick(dt)
        fingers = self.counter.count(cam)
        self.state.fingers = fingers

        cw, ch = self._canvas_size()
        bg = self.bg_base.resize((cw, ch), resample=Image.BILINEAR).copy()
        draw = ImageDraw.Draw(bg)

        # 마인 합성
        if self.main.frame_bgr is not None and self.state.main_pos:
            main_rgba = chroma_key_rgba_keep_aspect(self.main.frame_bgr, self.main.size_px)
            if main_rgba:
                w, h = main_rgba.size
                if PIP_VISIBLE:
                    keepout = self._pip_rect(cw, ch)
                for (cx, cy) in self.state.main_pos:
                    x0 = clamp(int(cx - w//2), 0, max(0, cw - w))
                    y0 = clamp(int(cy - h//2), 0, max(0, ch - h))
                    if PIP_VISIBLE:
                        x0, y0 = keepout_adjust(x0, y0, w, h, keepout)
                    x0 = clamp(x0, 0, max(0, cw - w))
                    y0 = clamp(y0, 0, max(0, ch - h))
                    bg.alpha_composite(main_rgba, dest=(x0, y0))

        # PiP
        if PIP_VISIBLE:
            pip_img = self.counter.last_pip_rgba(PIP_W, PIP_H)
            if pip_img:
                kx, ky, kw, kh = self._pip_rect(cw, ch)
                draw.rectangle([kx-4, ky-4, kx+kw+4, ky+kh+4], fill=(0,0,0,140))
                bg.alpha_composite(pip_img, dest=(kx, ky))
                font = self._load_font(36)
                num_text = str(self.state.fingers)
                pad = 6
                tx = kx + kw - pad
                ty = ky + kh - pad
                self._draw_text_shadow(draw, (tx, ty), num_text, font, WHITE, anchor="rd")

        # 상태/시간/스코어
        if self.state.phase == "COUNTDOWN":
            self.state.countdown_ms -= dt
            sec = max(0, int(math.ceil(self.state.countdown_ms/1000)))
            self._draw_center_banner(bg, f"{sec}")
            self._draw_bottom_center_score(bg, self.state.score, self.state.lives)
            if self.state.countdown_ms <= 0:
                self._enter_running_after_countdown()

        elif self.state.phase == "RUNNING":
            self.state.left_ms -= dt
            left_sec = max(0, self.state.left_ms//1000)
            self._draw_top_center_time(bg, int(left_sec))
            self._draw_bottom_center_score(bg, self.state.score, self.state.lives)
            if fingers == self.state.target:
                self.on_success()
            elif self.state.left_ms <= 0:
                self.on_fail()

        elif self.state.phase == "SUCCESS":
            self._draw_center_banner(bg, "SUCCESS!", fill=(32,180,90,230))
            self._draw_bottom_center_score(bg, self.state.score, self.state.lives)
            self.state.success_ms += dt
            if self.state.success_ms >= self.state.success_hold:
                self.start_round()

        elif self.state.phase == "FAIL":
            self._draw_center_banner(bg, "FAIL!", fill=(220,50,50,230))
            self._draw_bottom_center_score(bg, self.state.score, self.state.lives)

        elif self.state.phase == "GAMEOVER":
            self._draw_center_banner(bg, "GAME OVER", fill=(180,30,30,230))
            self._draw_bottom_center_score(bg, self.state.score, self.state.lives)

        # 최종 렌더
        self.tkimg = ImageTk.PhotoImage(bg)
        self.canvas.delete("all")
        self.canvas.create_image(0,0, anchor="nw", image=self.tkimg)

        # 선택: 녹화
        if SAVE_SESSION_VIDEO:
            frame_bgr = cv2.cvtColor(np.array(bg)[..., :3], cv2.COLOR_RGB2BGR)
            if self._writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter("session.mp4", fourcc, APP_FPS, (cw, ch))
            self._writer.write(frame_bgr)

        self.root.after(int(1000/APP_FPS), self.tick)

# -----------------------------------------------------
# 비디오 스프라이트: 비디오에서 프레임을 목적 fps에 맞게 업데이트
# -----------------------------------------------------
class MainSprite:
    """비디오를 일정 속도로 갱신해 RGBA 합성에 쓰는 스프라이트."""
    def __init__(self, cap: cv2.VideoCapture, size_px: int = MAIN_SIZE_PX_DEFAULT):
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.size_px = size_px
        self.frame_bgr: Optional[np.ndarray] = None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.dt_ms_target = 1000.0 / self.fps
        self.acc_ms = 0.0

    def tick(self, dt_ms: float):
        """경과 시간에 맞춰 비디오 프레임을 갱신."""
        self.acc_ms += dt_ms
        updated = False
        while self.acc_ms >= self.dt_ms_target:
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = self.cap.read()
                if not ok: break
            self.frame_bgr = f
            self.acc_ms -= self.dt_ms_target
            updated = True
        return updated

# -----------------------------------------------------
# 엔트리: 리소스 체크 후 앱 실행
# -----------------------------------------------------
def main():
    """리소스 확인 후 Tk 루프를 시작."""
    if not os.path.exists(BACK_IMAGE_PATH):
        print(f"[ERROR] 배경 이미지 없음: {BACK_IMAGE_PATH}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(MAIN_VIDEO_PATH):
        print(f"[ERROR] 마인 영상 없음: {MAIN_VIDEO_PATH}", file=sys.stderr); sys.exit(1)
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()

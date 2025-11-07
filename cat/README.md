RVM + Cats + HandsOnly (REC)

웹캠 → RVM 배경제거 → 손만 보이게(마스크×알파) → 크로마키 고양이 이펙트 합성.
검지 TIP이 히트박스에 닿으면 MCP→TIP 방향 임펄스. v로 MP4 녹화.

필요 파일
model/rvm_mobilenetv3.pth
hand_landmarker.task
back.png
A.mp4

설치(요지)
pip install torch opencv-python mediapipe numpy

실행
python rvm_cats_handsonly_pointdir_autoreset_rec.py --w 1280 --h 720

주요 키

v: 녹화 시작/정지 (recordings/*.mp4)

k: 물리 모드 off → impulse → gravity

b/r: 히트박스 표시 토글 / 위치 재배치

+ / -: 크기 조절, ] / [: 마리 수 조절

m/h/g: 미러 / 손 랜드마크 / 검지 화살표 토글

p: 속도 리셋, q/ESC: 종료

핵심 옵션(CONFIG)

cam_index: 카메라 인덱스(기본 6)

downsample_ratio: RVM 속도/품질(기본 0.40)

hands_only: 손만 보이게 적용 on/off

ck_lo/hi_hsv: 크로마키 범위

impulse_gain, impulse_base, gravity, damping, vmax

allow_escape: 화면 밖 허용

auto_reset_*: 전부 사라지면 자동 리셋

rec_*: 녹화 폴더/코덱/프레임레이트

트러블슈팅(한 줄)

카메라 안 열림: cam_index 변경.

녹화 안 됨: rec_codec을 mp4v ↔ avc1로 바꿔 시도.

초록 누출: ck_feather / ck_spill / HSV 조정.
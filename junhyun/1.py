cd /junhyun/AI_Project/junhyun
python3 - <<'PY'
import os
print("현재 실행 경로:", os.getcwd())
print("모델 파일 존재 여부:", os.path.exists("intel/human-pose-estimation-0005/FP32/human-pose-estimation-0005.xml"))
PY

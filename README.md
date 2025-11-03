1️⃣ 팀명 - 슬로건  
VisionPulse — “몸의 움직임으로 세상을 반응시키다”

2️⃣ 프로젝트 주제 선정

프로젝트명  
“Gesture Impact Vision – 실시간 랜드마크 인식을 통한 제스처 기반 인터랙티브 이펙트 시스템”

요약  
사용자의 몸동작을 AI가 인식하여, 카메라 화면 상에 즉각적인 시각적 효과(빛, 파티클, 폭발 등)를 구현하는 Vision AI 인터랙션 프로젝트

핵심 기술  
Pose/Landmark Detection (Openvino, YOLO)  
Gesture Classification (CNN)  
Effect Rendering (Overlay)

3️⃣ 유즈케이스 시나리오 작성

1. 사용자가 카메라 앞에 선다  
시스템이 랜드마크 감지  
(Pose Detection, Gesture Classification)  
2.  특정 제스처(예: 양손을 모아 앞으로 내밀기)  
제스처를 인식하여 “에너지파 효과” 출력  
(Overlay layer)


4️⃣ High Level Design (HLD)  
🧩 시스템 구성도 (개요)

[Camera Input]  
     ↓  
[Pose Detection / Landmark Extraction]  
      ↓  
[Gesture Classification (AI Model)]  
      ↓  
[Event Trigger & Effect Renderer]  
      ↓  
[Output: Real-time Visual Effect Display]

🧱 기술 구조  
Front Layer: OpenCV 영상 캡처 + 실시간 랜드마크 시각화  
AI Layer: Pose Detection → Gesture 분류 (TensorFlow/Keras CNN)  
Effect Layer: Overlay 렌더링
Control Layer: Python

5️⃣ Project Milestone 정의  
1단계   
 문제 정의 : 아이디어 초점 맞추기, 기본도구모음, 연출 정의

2단계  
 제스처 정의 및 분류모델 학습, 제스처 데이터 수집 및 분류

3단계  
 실시간 카메라 입력 + 모델 연동, OpenCV 과 모델 연결

4단계  
 이펙트 렌더링 구현, 특정 제스처 → 특정 시각효과 발생

5단계  
 통합 테스트 및 발표자료 준비, 전체 시스템 통합 및 영상 시연
 
6️⃣ 팀원별 역할 결정

팀장 / PM  
박상수 : 일정 관리, milestone, 발표 자료 총괄  
AI 모델 담당  
조경원 : AI 도구 모델링, Python Code 작성  
Data, Effect 담당  
백다빈, 정경준, 김준현 : Gesture Data 수집 정리, Effect Searching, Effect 연출 구현

7️⃣ 프로젝트 Repo 생성  
[GeneralArang/AI_Project: DX-3 Vision AI Project](https://github.com/GeneralArang/AI_Project)

8️⃣ 프로젝트용 README.md 초안 Template
# Gesture Impact Vision 🎥

## 🧩 Project Overview
Real-time gesture detection using Vision AI that triggers visual effects on the camera feed.

## 🎯 Objective
To develop an AI-based gesture control system that uses body movements as input switches for interactive visual effects.

## 🔧 Key Technologies
- Openvino
- TensorFlow / Keras
- OpenCV (real-time video processing)
- Python (main control logic)
- 
## 🧠 How It Works
1. Capture live camera feed.
2. Detect body landmarks.
3. Classify gestures.
4. Trigger mapped visual effect.

## 🗓️ Project Timeline
| Phase | Goal | Duration |
|-------|------|----------|
| 1 | Model Select | 1 |
| 2 | Gesture Training and Effect System |  2 |
| 3 | Integration | 3 |
| 4 | Demo & Docs | 4 |

## 👥 Team Members
| Role | Name | Description |
|------|------|-------------|
| PM | Park Sangsu | Planning / Coordination |
| AI Engineer | Cho Kyungwon | Model Design |
| Vision Engineer | Kim JunHuen | Video Stream Processing |
| Effect Designer | Beak Dabin, Jung Kungjun | Visual Effects |
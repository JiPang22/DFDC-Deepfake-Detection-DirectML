# rPPG-based Deepfake Detection

This project utilizes a dual-branch deep learning model to detect deepfake videos by analyzing both spatial and temporal (frequency) information. The core idea is to leverage remote photoplethysmography (rPPG) signals, which capture subtle color changes in the skin caused by blood flow, as a liveness indicator.

## Key Features

- **Dual-Branch Architecture**:
  - **Spatial Branch (CNN)**: Analyzes each frame for visual artifacts, compression errors, and unnatural textures.
  - **Frequency Branch (FFT)**: Extracts the rPPG signal from the video, performs a Fast Fourier Transform (FFT), and analyzes the frequency spectrum for a periodic heartbeat signal.
- **Focal Loss**: Focuses the training on hard-to-classify examples, improving model accuracy on ambiguous videos.
- **Advanced Augmentation**: Simulates real-world conditions like poor lighting, low bit-rate, and camera noise to build a robust model.
- **Automated Pipeline**: Includes scripts for training, evaluation, and visualization, all connected in a seamless workflow.

## Performance

The model achieves an accuracy of **~68%** on the validation set. The performance is balanced across both 'Real' and 'Fake' classes, as shown in the ROC curve and confusion matrix below.

![Performance Report](results/performance_report.png)

## How to Use

### 1. Setup

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd Deepfake_project
pip install -r requirements.txt
```

### 2. Data Preprocessing

Place your video data (or image sequences) in the `train_data_85/real` and `train_data_85/fake` directories. Then, run the preprocessing script to convert them into `.npy` tensors.

```bash
python src/preprocess_from_images.py
```

### 3. Training

Run the training script. The best model will be saved as `src/rppg_best_real.pth`.

```bash
# For a quick 30-minute run
python src/train_quick.py

# For a full, in-depth training session
python src/train_final.py
```

### 4. Evaluation & Inference

After training, the script will automatically evaluate the model and run a random inference test. You can also run them manually:

```bash
# Evaluate the best model
python src/evaluate.py

# Test with a random sample
python src/inference.py
```
# 🛡️ Deepfake Detection with rPPG & Hybrid ViViT

이 프로젝트는 시각적 왜곡뿐만 아니라 생체 신호(rPPG)의 일관성을 분석하여 딥페이크 여부를 판별하는 고성능 AI 모델을 구현합니다. AMD ROCm 환경(RX 6600)에 최적화되어 있습니다.

## 📂 Project Structure
```text
Deepfake_project/
├── data/               # (Git 제외) 원본 및 전처리 데이터셋
├── models/             # (Git 제외) 학습된 모델 가중치 (.pth)
├── src/                # 소스 코드
│   ├── model.py        # ViViT + rPPG 하이브리드 모델 정의
│   ├── train.py        # 시각화(plt) 및 조기 종료 기능 포함 학습 스크립트
│   └── inference.py    # 비디오 전체 분석 스캐너
├── results/            # 학습 곡선 그래프 등 결과물
├── .gitignore          # 대용량 데이터 및 바이너리 제외 설정
└── README.md           # 프로젝트 가이드


## 🛠️ Execution Guide (실행 순서)

프로젝트를 처음 시작할 때 다음 순서대로 스크립트를 실행하십시오.

### 1. 데이터 준비 및 전처리

먼저 원본 영상에서 학습에 필요한 얼굴 프레임을 고속으로 추출합니다.

```bash
# DFDC 영상에서 10프레임씩 추출하여 data/processed/에 저장
python src/extract_frames.py

```

### 2. 모델 학습 (rPPG + ViViT)

생체 신호와 시간적 흐름을 분석하는 하이브리드 모델을 학습시킵니다.

```bash
# 학습 진행, ETA 출력, 0.5% 오차 시 조기 종료 및 결과 그래프 팝업
python src/train.py

```

### 3. 영상 판별 (Inference)

학습된 모델을 사용하여 실제 영상 파일(.mp4)의 위조 여부를 스캔합니다.

```bash
# 영상 내 30개 프레임을 종합 분석하여 최종 확률 출력
python src/inference.py

```

## 🚀 Key Technologies

* **rPPG (remote Photoplethysmography)**: 얼굴 피부의 미세한 색상 변화를 추적하여 심박 신호의 유무를 판별합니다.
* **Hybrid ViViT**: CNN(공간 분석) + LSTM(시간 분석) 구조로 비디오 데이터에 최적화된 추론을 수행합니다.
* **Smart Training**: Train/Val Accuracy 오차가 0.5% 이내 수렴 시 자동 저장 및 종료.

## 🛠️ Environment

* **GPU**: AMD Radeon RX 6600 (ROCm 10.3.0 Override 적용)
* **OS**: Linux (Ubuntu 22.04/24.04 권장)

```

---

### 📋 지식 매핑
* **Execution Flow**: AI 프로젝트에서 전처리(Pre-processing) → 학습(Training) → 추론(Inference) 순서를 명시하는 것은 코드 사용자의 가독성을 극대화합니다.
* **Reproducibility**: `README.md`_에 명시된 명령어를 그대로 따라 했을 때 같은 결과가 나오게 하는 것이 배포의 핵심입니다.


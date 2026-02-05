import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import cv2
import numpy as np
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# [지식 매핑] 같은 디렉토리의 core_model.py를 참조하기 위한 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core_model import RPPGDeepfakeModel

# AMD GPU(RX 6600) 환경을 위한 환경 변수 설정
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 및 가중치 로드
model = RPPGDeepfakeModel().to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../models/deepfake_rppg_best.pth")

# [ETA: 2초] 모델 파라미터 로드
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()


def preprocess_video(video_path, num_frames=30):
    """
    [데이터셋 리스트 구성(라벨링)]
    영상 전체에서 균등하게 프레임을 추출하고, rPPG 신호 포착을 위해 대비를 강화함.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 영상 전체 구간을 균등하게 30분할
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    # 대비 향상을 위한 CLAHE 객체 생성
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break

        # 1. 흑백 변환 (모델 규격: 1채널)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. CLAHE 적용 (미세한 혈류 변화 강조)
        frame = clahe.apply(frame)
        # 3. 리사이즈 (모델 규격: 56x56)
        frame = cv2.resize(frame, (56, 56))
        frames.append(frame)
    cap.release()

    # 프레임 부족 시 제로 패딩
    while len(frames) < num_frames:
        frames.append(np.zeros((56, 56)))

    # 텐서 변환 및 표준화 (0.5 평균 기준)
    video_tensor = np.array(frames, dtype=np.float32) / 255.0
    video_tensor = (video_tensor - 0.5) / 0.5

    # 모델 입력 차원: [Batch, Seq, Channel, H, W] -> [1, 30, 1, 56, 56]
    return torch.from_numpy(video_tensor).unsqueeze(0).unsqueeze(2).to(device)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 업로드 파일 임시 저장
    temp_path = os.path.join(current_dir, f"../data/temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # [ETA: 10초 내외] 영상 분석 수행
        input_tensor = preprocess_video(temp_path, num_frames=30)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            fake_prob = prob[0][1].item() * 100
            real_prob = prob[0][0].item() * 100

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "prediction": "FAKE" if fake_prob > 50 else "REAL",
            "confidence": f"{max(fake_prob, real_prob):.2f}%",
            "details": {
                "fake_score": f"{fake_prob:.2f}%",
                "real_score": f"{real_prob:.2f}%"
            }
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    # 서버 실행 (포트 8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
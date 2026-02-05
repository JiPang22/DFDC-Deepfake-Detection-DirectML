import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import cv2
import numpy as np
from datetime import datetime

# [지식 매핑] 모델 및 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core_model import RPPGDeepfakeModel

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'


def preprocess_video(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = clahe.apply(frame)
        frame = cv2.resize(frame, (56, 56))
        frames.append(frame)
    cap.release()
    if len(frames) == 0: return None
    video_tensor = np.array(frames, dtype=np.float32) / 255.0
    video_tensor = (video_tensor - 0.5) / 0.5
    return torch.from_numpy(video_tensor).unsqueeze(0).unsqueeze(2)


def run_batch_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_raw_dir = os.path.join(project_root, "data/raw")
    model_path = os.path.join(project_root, "models/deepfake_rppg_best.pth")
    result_dir = os.path.join(project_root, "results")

    if not os.path.exists(result_dir): os.makedirs(result_dir)

    model = RPPGDeepfakeModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    video_files = [f for f in os.listdir(data_raw_dir) if f.lower().endswith(('.mp4', '.avi'))]
    if not video_files: return

    report_path = os.path.join(result_dir, "prediction_report.txt")

    # [데이터셋 리스트 구성(라벨링)] - 결과 기록 시작
    with open(report_path, "a") as f:
        f.write(f"\n--- Analysis Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        for v_file in video_files:
            target_path = os.path.join(data_raw_dir, v_file)
            input_tensor = preprocess_video(target_path)
            if input_tensor is None: continue

            with torch.no_grad():
                output = model(input_tensor.to(device))
                prob = torch.softmax(output, dim=1)
                fake_prob = prob[0][1].item() * 100

            res_text = f"[{'FAKE' if fake_prob > 50 else 'REAL'}] {v_file} (Conf: {max(fake_prob, 100 - fake_prob):.2f}%)"
            print(res_text)
            f.write(res_text + "\n")

    print(f"\n✅ 리포트 저장 완료: {report_path}")


if __name__ == "__main__":
    # [ETA: 영상당 3초] 모든 영상 순차 분석
    run_batch_predict()
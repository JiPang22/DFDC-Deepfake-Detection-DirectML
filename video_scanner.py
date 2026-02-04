import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 모델 구조 (동일)
class LightweightViViT(nn.Module):
    def __init__(self):
        super(LightweightViViT, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 2)
        )

    def forward(self, x): return self.features(x)


model = LightweightViViT().to(device)
model.load_state_dict(torch.load("deepfake_model_final.pth", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def scan_video(video_path):
    if not os.path.exists(video_path):
        print(f"파일 없음: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"영상을 읽을 수 없습니다: {video_path}")
        return

    scores = []
    print(f"분석 대상: {video_path}")

    # 30프레임 분석 루프
    frame_indices = np.linspace(0, total_frames - 1, 30, dtype=int)
    for i in tqdm(frame_indices, desc="프레임 스캔 중"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            scores.append(probs[0][1].item())

    cap.release()

    if scores:
        avg_score = sum(scores) / len(scores)
        final_label = "FAKE" if avg_score > 0.5 else "REAL"
        print(f"\n[최종 분석 결과] {final_label}")
        print(f"평균 FAKE 확률: {avg_score * 100:.2f}%")
    else:
        print("분석된 프레임이 없습니다.")


if __name__ == "__main__":
    # 데이터셋 폴더에서 영상 하나 찾기
    root_dir = "/home/jplinux/Deepfake_project/dfdc_data"
    found_video = ""
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".mp4"):
                found_video = os.path.join(root, f)
                break
        if found_video: break

    if found_video:
        scan_video(found_video)
    else:
        print("분석할 mp4 영상을 찾을 수 없습니다.")
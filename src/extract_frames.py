import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import pandas as pd
from tqdm import tqdm

# 1. 경로 설정
csv_path = "/home/jplinux/Deepfake_project/balanced_video_list.csv"
save_base_path = "/home/jplinux/Deepfake_project/train_data_85"
os.makedirs(os.path.join(save_base_path, "real"), exist_ok=True)
os.makedirs(os.path.join(save_base_path, "fake"), exist_ok=True)

# 2. 데이터 리스트 로드
df = pd.read_csv(csv_path)

print("프레임 추출 시작 (영상당 10장)...")

# 3. 추출 루프 (데이터셋 리스트 구성(라벨링))
for idx, row in tqdm(df.iterrows(), total=len(df)):
    video_path = row['video_path']
    label = row['label'].lower()  # 'real' 또는 'fake'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        cap.release()
        continue

    # 간격을 두고 10장 추출
    interval = total_frames // 10
    for i in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            # 파일명 규칙: 라벨_영상인덱스_프레임번호.jpg (예: 0_123_5.jpg)
            label_num = 0 if label == 'real' else 1
            filename = f"{label_num}_{idx}_{i}.jpg"
            save_path = os.path.join(save_base_path, label, filename)

            # 이미지 크기 조정 (224x224) - ViViT 입력 최적화
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(save_path, frame)

    cap.release()

print(f"\n✅ 추출 완료! 저장 위치: {save_base_path}")
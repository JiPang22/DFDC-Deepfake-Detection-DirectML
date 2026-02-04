import cv2
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# 1. 경로 설정
CSV_PATH = "/home/jplinux/Deepfake_project/balanced_video_list.csv"
SAVE_BASE_PATH = "/home/jplinux/Deepfake_project/train_data_85"
os.makedirs(os.path.join(SAVE_BASE_PATH, "real"), exist_ok=True)
os.makedirs(os.path.join(SAVE_BASE_PATH, "fake"), exist_ok=True)


def extract_from_video(row_data):
    idx, video_path, label = row_data
    label = label.lower()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        cap.release()
        return

    interval = total_frames // 10
    label_num = 0 if label == 'real' else 1

    for i in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            filename = f"{label_num}_{idx}_{i}.jpg"
            save_path = os.path.join(SAVE_BASE_PATH, label, filename)
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(save_path, frame)
    cap.release()


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    # 병렬 처리를 위한 데이터 리스트 구성
    tasks = [(idx, row['video_path'], row['label']) for idx, row in df.iterrows()]

    print(f"멀티코어 가동: {os.cpu_count()}개의 스레드로 고속 추출 시작...")

    # R5 5600의 12스레드 활용
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(extract_from_video, tasks), total=len(tasks)))

    print(f"\n✅ 고속 추출 완료! 저장 위치: {SAVE_BASE_PATH}")
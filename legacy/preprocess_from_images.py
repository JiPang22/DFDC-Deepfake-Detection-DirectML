# [파일 경로: /home/jplinux/Deepfake_project/src/preprocess_from_images.py]
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def preprocess_images(input_root, output_root, seq_len=30, img_size=128):
    """
    이미지 파일들을 읽어 비디오 ID별로 그룹화하고, .npy 텐서로 저장.
    """
    categories = ['real', 'fake']
    
    for category in categories:
        input_dir = os.path.join(input_root, category)
        output_dir = os.path.join(output_root, category)
        
        if not os.path.exists(input_dir):
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 이미지 파일 스캔 및 그룹화
        print(f"[{category.upper()}] 파일 스캔 및 그룹화 중...")
        video_groups = defaultdict(list)
        
        files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
        
        for f in files:
            # 파일명 파싱: 0_2620_2.jpg -> parts=['0', '2620', '2']
            try:
                parts = f.replace('.jpg', '').split('_')
                if len(parts) >= 2:
                    video_id = parts[1] # 중간 숫자를 비디오 ID로 가정
                    video_groups[video_id].append(os.path.join(input_dir, f))
            except:
                continue # 파싱 실패 시 스킵

        print(f"[{category.upper()}] 총 {len(video_groups)}개의 비디오 시퀀스 발견.")
        
        # 2. 각 그룹(비디오)별로 텐서 생성
        for video_id, image_paths in tqdm(video_groups.items()):
            save_path = os.path.join(output_dir, f"{video_id}.npy")
            
            if os.path.exists(save_path):
                continue

            # 이미지 로드 및 전처리
            frames = []
            # 파일명 끝자리(프레임번호) 기준으로 정렬 시도, 실패하면 이름순
            try:
                image_paths.sort(key=lambda x: int(x.split('_')[-1].replace('.jpg', '')))
            except:
                image_paths.sort()

            # 시퀀스 길이 조절 (seq_len 개수만큼 샘플링)
            if len(image_paths) > seq_len:
                indices = np.linspace(0, len(image_paths) - 1, seq_len, dtype=int)
                selected_paths = [image_paths[i] for i in indices]
            else:
                selected_paths = image_paths # 부족하면 있는 거 다 씀

            for img_path in selected_paths:
                # Grayscale 로드
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Resize
                img = cv2.resize(img, (img_size, img_size))
                
                # Normalize (0~1)
                img = img / 255.0
                
                frames.append(img)

            if not frames:
                continue

            # 프레임 부족 시 패딩 (마지막 프레임 복사 or 0 채움)
            # 여기선 0으로 채움
            if len(frames) < seq_len:
                padding = [np.zeros((img_size, img_size)) for _ in range(seq_len - len(frames))]
                frames.extend(padding)

            # (T, H, W) -> (C, T, H, W)
            video_tensor = np.stack(frames, axis=0)
            video_tensor = np.expand_dims(video_tensor, axis=0)
            
            # 저장
            np.save(save_path, video_tensor.astype(np.float32))

if __name__ == "__main__":
    INPUT_ROOT = '/home/jplinux/Deepfake_project/train_data_85'
    OUTPUT_ROOT = '/home/jplinux/Deepfake_project/data/processed'
    
    print(f"이미지 시퀀스 전처리 시작: {INPUT_ROOT} -> {OUTPUT_ROOT}")
    preprocess_images(INPUT_ROOT, OUTPUT_ROOT)
    print("전처리 완료.")
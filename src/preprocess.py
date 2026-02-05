import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# [파일 경로: /home/jplinux/Deepfake_project/src/preprocess.py]
import cv2
import numpy as np
import torch
from tqdm import tqdm

# [얼굴 감지기 설정]
# OpenCV의 기본 Haar Cascade 사용 (속도 최적화)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_video(video_path, output_path, seq_len=30, img_size=128):
    """
    비디오를 읽어 얼굴을 크롭하고, (C, T, H, W) 형태의 .npy 파일로 저장.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 전체 프레임 수 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 프레임 샘플링 간격 (전체에서 골고루 30개 뽑기)
    if total_frames > seq_len:
        indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
    else:
        indices = np.arange(total_frames) # 프레임 부족 시 있는 대로 다 씀

    current_frame = 0
    collected_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in indices:
            # 1. 그레이스케일 변환 (rPPG는 밝기 변화가 중요)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. 얼굴 감지
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # 가장 큰 얼굴 선택
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_img = gray[y:y+h, x:x+w]
                
                # 3. 리사이즈
                face_img = cv2.resize(face_img, (img_size, img_size))
                
                # 4. 정규화 (0~1)
                face_img = face_img / 255.0
                
                frames.append(face_img)
                collected_count += 1
                
                if collected_count >= seq_len:
                    break
        
        current_frame += 1
        
    cap.release()
    
    # 프레임이 부족할 경우 패딩 (0으로 채움)
    if len(frames) < seq_len:
        padding = [np.zeros((img_size, img_size)) for _ in range(seq_len - len(frames))]
        frames.extend(padding)
        
    # (T, H, W) -> (C, T, H, W) 변환 (C=1)
    # numpy stack: (30, 128, 128)
    video_tensor = np.stack(frames, axis=0)
    # expand dims: (1, 30, 128, 128)
    video_tensor = np.expand_dims(video_tensor, axis=0)
    
    # 저장
    np.save(output_path, video_tensor.astype(np.float32))

def process_dataset(input_root, output_root):
    """
    input_root의 real/fake 폴더를 읽어 output_root에 .npy로 저장
    """
    categories = ['real', 'fake']
    
    for category in categories:
        input_dir = os.path.join(input_root, category)
        output_dir = os.path.join(output_root, category)
        
        if not os.path.exists(input_dir):
            print(f"[스킵] 폴더 없음: {input_dir}")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        print(f"[{category.upper()}] 처리 시작: {len(files)}개 파일")
        
        for file_name in tqdm(files):
            video_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace('.mp4', '.npy'))
            
            if os.path.exists(output_path):
                continue # 이미 처리된 파일 스킵
                
            try:
                preprocess_video(video_path, output_path)
            except Exception as e:
                print(f"[오류] {file_name} 처리 실패: {e}")

if __name__ == "__main__":
    # [설정]
    # 사용자의 실제 데이터 경로
    INPUT_ROOT = '/home/jplinux/Deepfake_project/data/raw/train_data_85'
    # 전처리된 데이터가 저장될 경로
    OUTPUT_ROOT = os.path.join(project_root, 'data/processed')
    
    print(f"전처리 시작: {INPUT_ROOT} -> {OUTPUT_ROOT}")
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)
    print("전처리 완료.")
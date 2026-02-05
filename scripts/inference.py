import os
import sys
import cv2
import numpy as np
import torch
import random
from collections import defaultdict

# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core_model import RPPGDeepfakeModel

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

def preprocess_from_images(image_paths, seq_len=30, img_size=128):
    frames = []
    try:
        image_paths.sort(key=lambda x: int(x.split('_')[-1].replace('.jpg', '')))
    except:
        image_paths.sort()

    if len(image_paths) > seq_len:
        indices = np.linspace(0, len(image_paths) - 1, seq_len, dtype=int)
        selected_paths = [image_paths[i] for i in indices]
    else:
        selected_paths = image_paths

    for img_path in selected_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        frames.append(img)

    if not frames: return None

    if len(frames) < seq_len:
        padding = [np.zeros((img_size, img_size)) for _ in range(seq_len - len(frames))]
        frames.extend(padding)
        
    video_tensor = np.stack(frames, axis=0)
    video_tensor = np.expand_dims(video_tensor, axis=0)
    return torch.from_numpy(video_tensor.astype(np.float32)).unsqueeze(0)

def run_inference_random():
    # --- [설정] ---
    MODEL_PATH = os.path.join(project_root, 'models', 'rppg_best_real.pth') # [수정] 중앙 모델 저장소 경로
    DATA_ROOT = os.path.join(project_root, 'data/raw/train_data_85')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RPPGDeepfakeModel().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 학습된 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print("[성공] 모델 로드 완료.")
    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
        return

    print(f"[진행] '{DATA_ROOT}'에서 랜덤 비디오 샘플링 중...")
    category = random.choice(['real', 'fake'])
    data_dir = os.path.join(DATA_ROOT, category)
    
    video_groups = defaultdict(list)
    files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    for f in files:
        try:
            video_id = f.split('_')[1]
            video_groups[video_id].append(os.path.join(data_dir, f))
        except:
            continue
            
    if not video_groups:
        print(f"[오류] '{data_dir}'에서 이미지 파일을 찾을 수 없습니다.")
        return
        
    random_video_id = random.choice(list(video_groups.keys()))
    image_paths = video_groups[random_video_id]
    
    print(f"  >> 선택된 샘플: {category.upper()} / 비디오 ID: {random_video_id}")

    input_tensor = preprocess_from_images(image_paths)
    if input_tensor is None:
        print("[오류] 샘플 전처리 실패.")
        return

    input_tensor = input_tensor.to(device)

    print("[진행] 모델 추론 중...")
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    prediction = "FAKE" if probability > 0.5 else "REAL"
    
    print("\n" + "="*30)
    print("         >>> 최종 판별 결과 <<<")
    print("="*30)
    print(f"  - 실제 라벨: [ {category.upper()} ]")
    print(f"  - 모델 예측: [ {prediction} ]")
    print(f"  - 신뢰도 (Fake일 확률): {probability * 100:.2f}%")
    print("="*30)
    
    if category.upper() == prediction:
        print("  >> 결과: 정답입니다!")
    else:
        print("  >> 결과: 틀렸습니다.")

if __name__ == '__main__':
    run_inference_random()
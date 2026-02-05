import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report

# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core_model import RPPGDeepfakeModel
from src.dataset import DeepfakeDataset

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

def evaluate():
    # --- [설정] ---
    DATA_ROOT = os.path.join(project_root, 'data/processed')
    MODEL_PATH = os.path.join(project_root, 'models', 'rppg_best_real.pth') # [수정] 중앙 모델 저장소 경로
    BATCH_SIZE = 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] 평가 장치: {device}")

    model = RPPGDeepfakeModel().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print(f"[성공] 모델 로드 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
        return
    
    model.eval()

    print("[데이터] 평가용 데이터셋 로드 중...")
    full_dataset = DeepfakeDataset(data_root=DATA_ROOT)
    
    if len(full_dataset) == 0:
        print("[오류] 데이터셋이 비어있습니다.")
        return
        
    indices = list(range(len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_size = int(0.2 * len(full_dataset))
    val_indices = indices[:val_size]
    
    val_dataset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"[데이터] 평가 샘플 수: {len(val_dataset)}개 (Real/Fake 혼합)")

    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print("\n" + "="*50)
    print("               최종 모델 성능 평가 리포트")
    print("="*50)
    
    report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], digits=4)
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    print("="*50)
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    print(f"해석: 모델은 전체 {len(all_labels)}개의 비디오 중 {accuracy*100:.2f}%를 정확하게 분류했습니다.")

if __name__ == "__main__":
    evaluate()
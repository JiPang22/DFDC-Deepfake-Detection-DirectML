import os
import sys
import torch
import numpy as np
import matplotlib
try:
    matplotlib.use('Agg')
except:
    pass
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader, Subset

# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core_model import RPPGDeepfakeModel
from src.dataset import DeepfakeDataset

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

def visualize():
    # --- [설정] ---
    DATA_ROOT = os.path.join(project_root, 'data/processed')
    MODEL_PATH = os.path.join(project_root, 'models', 'rppg_best_real.pth') # [수정] 중앙 모델 저장소 경로
    RESULT_DIR = os.path.join(project_root, 'results')
    BATCH_SIZE = 32
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] 분석 장치: {device}")

    model = RPPGDeepfakeModel().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일 없음: {MODEL_PATH}")
        return
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
        return

    print("[데이터] 전체 데이터셋 로드 및 분석 중...")
    dataset = DeepfakeDataset(data_root=DATA_ROOT)
    
    if len(dataset) > 2000:
        indices = np.random.choice(len(dataset), 2000, replace=False)
        dataset = Subset(dataset, indices)
        print(f"[알림] 빠른 시각화를 위해 2000개 샘플만 무작위 추출했습니다.")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_true.extend(labels.numpy().flatten())
            y_scores.extend(probs.flatten())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int)

    print("[분석] 그래프 생성 중...")
    
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real', 'Fake'])
    plt.yticks(tick_marks, ['Real', 'Fake'])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', marker='.', linestyle='-', lw=1, markersize=4, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]
    plt.hist(real_scores, bins=30, alpha=0.6, color='blue', label='Real')
    plt.hist(fake_scores, bins=30, alpha=0.6, color='red', label='Fake')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Predicted Probability (Fake Score)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(RESULT_DIR, 'performance_report.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[완료] 결과 그래프가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    visualize()
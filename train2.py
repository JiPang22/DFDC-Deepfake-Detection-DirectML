import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit # 그룹 분할 도구

# 데이터 증강: 모델을 괴롭혀서 억지로 공부하게 만듦
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 로드 및 영상 ID 추출
df = pd.read_csv("dataset_labels.csv")
# 경로명에서 영상 이름을 분리 (예: 'video_001_frame_1.jpg' -> 'video_001')
df['video_id'] = df['path'].apply(lambda x: x.split('/')[-1].split('_')[0])

# 영상 단위로 8:2 분할 (같은 영상의 프레임은 같은 묶음으로 이동)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df['video_id']))

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import time


# 1. 데이터셋 클래스 정의
class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # CSV 파일 로드 시 경로가 틀어지지 않도록 확인
        self.data_info = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_info.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # 파일이 없을 경우 예외 처리
            return None, None

        label = 1 if self.data_info.iloc[idx, 2] == 'real' else 0

        if self.transform:
            image = self.transform(image)
        return image, label


# 2. 이미지 변환 설정 (RX 6600 최적화: 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 경로 설정 (사용자 경로에 맞게 절대경로 권장)
BASE_PATH = "/home/jplinux/Deepfake_project"
CSV_PATH = os.path.join(BASE_PATH, "train_list.csv")
IMG_PATH = os.path.join(BASE_PATH, "oldThings/train_data_85")

# 데이터셋 리스트 구성(라벨링)
dataset = DeepfakeDataset(csv_file=CSV_PATH, img_dir=IMG_PATH, transform=transform)
# 배치 사이즈 16: RX 6600에서 ViViT 모델 실행 시 가장 안정적인 크기
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

if __name__ == "__main__":
    print(f"총 정예 데이터 개수: {len(dataset)}개")
    start_time = time.time()
    images, labels = next(iter(train_loader))
    end_time = time.time()
    print(f"첫 배치 로딩 성공 (Image Shape: {images.shape})")
    print(f"로딩 소요 시간: {end_time - start_time:.2f}초 (ETA 완료)")
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "resnet18_deepfake.pth"

# 모델 구조 정의 및 가중치 로드
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 전처리 (학습 때와 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prob = torch.nn.functional.softmax(outputs, dim=1)

    label = "FAKE" if predicted.item() == 1 else "REAL"
    confidence = prob[0][predicted.item()].item() * 100
    print(f"📸 결과: {label} ({confidence:.2f}%) | 경로: {image_path}")


if __name__ == "__main__":
    # 테스트하고 싶은 이미지 경로를 넣으세요 (학습에 안 쓴 사진 추천)
    test_img = "경로를_입력하세요.jpg"
    if os.path.exists(test_img):
        predict(test_img)
    else:
        print("파일이 없습니다. 경로를 확인해주세요.")
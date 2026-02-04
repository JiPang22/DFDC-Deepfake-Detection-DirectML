import cv2
import os
import shutil
from tqdm import tqdm
import time

# 1. 경로 및 설정
SOURCE_DIR = "/home/jplinux/Deepfake_project/processed_faces"
DEST_DIR = "/home/jplinux/Deepfake_project/high_quality_samples"
SHARPNESS_THRESHOLD = 100  # 선명도 기준 (이보다 높아야 선명함)
SAMPLE_LIMIT = 500  # 검토용으로 뽑을 최대 이미지 수

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# 2. 파일 리스트 확보
image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_count = 0
start_time = time.time()

# 3. 필터링 및 복사 (ETA 포함)
print("데이터 품질 검수 및 샘플링 시작...")
for i, filename in enumerate(tqdm(image_files, desc="Processing")):
    img_path = os.path.join(SOURCE_DIR, filename)

    # 이미지 읽기 및 선명도 계산
    img = cv2.imread(img_path)
    if img is None: continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()  # 라플라시안 분산값 (선명도)

    # 기준 충족 시 복사
    if score > SHARPNESS_THRESHOLD:
        shutil.copy(img_path, os.path.join(DEST_DIR, f"{int(score)}_{filename}"))
        selected_count += 1

    if selected_count >= SAMPLE_LIMIT:
        break

# 4. 결과 출력
end_time = time.time()
print(f"완료! 총 {selected_count}개의 고품질 샘플이 '{DEST_DIR}'에 저장되었습니다.")
print(f"총 소요 시간: {end_time - start_time:.2f}초 (ETA 완료)")
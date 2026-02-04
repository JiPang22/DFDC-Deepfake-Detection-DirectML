import os
import pandas as pd
import random

# 1. 현재 파일 위치 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(base_dir, "oldThings/train_data_85")

print(f"탐색 시작: {data_root}")

# 2. real과 fake 폴더 내부 파일 찾기 (하위 폴더 포함)
def get_files(target_name):
    target_path = os.path.join(data_root, target_name)
    file_list = []
    if os.path.exists(target_path):
        for root, dirs, files in os.walk(target_path):
            for if in files:
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    # train_list.csv에는 'train_data_85/real/파일명' 형태의 상대경로로 저장
                    rel_path = os.path.relpath(os.path.join(root, f), data_root)
                    file_list.append(rel_path)
    return file_list

real_files = get_files("real")
fake_files = get_files("fake")

# 3. 데이터 확인 및 1:1 믹스
if not real_files or not fake_files:
    print(f"데이터 부족! - Real: {len(real_files)}개, Fake: {len(fake_files)}개")
    print("train_data_85/real 폴더와 fake 폴더 안에 이미지가 있는지 확인하세요.")
    exit()

min_count = min(len(real_files), len(fake_files))
real_sample = random.sample(real_files, min_count)
fake_sample = random.sample(fake_files, min_count)

# 4. 리스트 생성 및 저장 (데이터셋 리스트 구성(라벨링))
dataset = []
for f in real_sample: dataset.append({'path': f, 'label': 0})
for f in fake_sample: dataset.append({'path': f, 'label': 1})

random.shuffle(dataset)
df = pd.DataFrame(dataset)

# 프로젝트 루트와 데이터 폴더 양쪽에 저장해서 에러 방지
df.to_csv(os.path.join(base_dir, "train_list.csv"), index=False)
df.to_csv(os.path.join(data_root, "train_list.csv"), index=False)

print(f"✅ 생성 완료! 총 {len(df)}개 (Real: {min_count}, Fake: {min_count})")
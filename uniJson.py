import os
import json
import pandas as pd

# 1. DFDC 데이터 루트 경로
base_path = "/home/jplinux/Deepfake_project/dfdc_data"
combined_data = []

# 2. 모든 하위 폴더의 metadata.json 검색
for root, dirs, files in os.walk(base_path):
    if "metadata.json" in files:
        json_path = os.path.join(root, "metadata.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
            for video_file, info in data.items():
                # 영상 파일명, 라벨(REAL/FAKE), 그리고 해당 영상이 있는 폴더 경로 저장
                combined_data.append({
                    'video_path': os.path.join(root, video_file),
                    'label': info['label'], # 'REAL' or 'FAKE'
                    'split': root.split('/')[-1] # 어느 파트인지 구분
                })

# 3. 통합 CSV 저장 (데이터셋 리스트 구성(라벨링))
df = pd.DataFrame(combined_data)
df.to_csv("/home/jplinux/Deepfake_project/dataset_labels.csv", index=False)

print(f"통합 완료: 총 {len(df)}개의 영상 정보 수집됨.")
print(df['label'].value_counts()) # REAL/FAKE 비율 출력
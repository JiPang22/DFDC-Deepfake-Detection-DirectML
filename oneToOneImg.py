import pandas as pd
import os

# 1. 통합 라벨 로드
df = pd.read_csv("/home/jplinux/Deepfake_project/dataset_labels.csv")

# 2. 1:1 샘플링 (REAL 개수인 2,512개에 맞춤)
real_df = df[df['label'] == 'REAL']
fake_df = df[df['label'] == 'FAKE'].sample(n=len(real_df), random_state=42)

# 3. 최종 학습 대상 리스트 통합 (데이터셋 리스트 구성(라벨링))
balanced_df = pd.concat([real_df, fake_df]).sample(frac=1).reset_index(drop=True)

# 4. 저장
balanced_df.to_csv("/home/jplinux/Deepfake_project/balanced_video_list.csv", index=False)

print(f"균형 잡힌 데이터셋 구성 완료: 총 {len(balanced_df)}개 영상")
print(balanced_df['label'].value_counts())
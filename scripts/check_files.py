import os
import sys
# [경로 자동 설정] 프로젝트 루트를 참조하도록 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


def check_dir(path):
    print(f"--- Checking: {path} ---")
    if not os.path.exists(path):
        print("Path does not exist!")
        return

    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}[DIR] {os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        
        # 파일이 너무 많으면 앞 5개만 출력
        for i, f in enumerate(files):
            if i < 5:
                print(f"{subindent}{f}")
            elif i == 5:
                print(f"{subindent}... (Total {len(files)} files)")
                break

if __name__ == "__main__":
    check_dir('/home/jplinux/Deepfake_project/data/raw/train_data_85')

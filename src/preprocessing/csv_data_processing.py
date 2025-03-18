import re
import os
import csv

# 현재 스크립트(insert_data.py)의 절대 경로를 기준으로 CSV 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

processed_dir = os.path.join(BASE_DIR, "data/gsang_processed_data/Validation")

# 정규 표현식 패턴 정의
pattern = r'&[a-zA-Z0-9]+&|\(.*?\)|[)]|-[^-\n]*-|{.*?}'

# Training 폴더 안의 CSV 파일들 순회
for filename in os.listdir(processed_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(processed_dir, filename)

        # CSV 파일 열기
        with open(file_path, mode="r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            header = next(reader)  # 헤더 읽기

            # 결과를 담을 리스트
            result = []
            for row in reader:
                dialect = re.sub(pattern, "", row[0]).strip()
                standard = re.sub(pattern, "", row[1]).strip()
                result.append([dialect, standard])

        # 파일 이름을 바탕으로 새 CSV 파일 경로 설정
        csv_filename = filename  # 파일 이름 그대로 사용
        csv_file_path = os.path.join(processed_dir, csv_filename)

        # 처리된 데이터를 새 CSV로 저장
        with open(csv_file_path, mode="w", encoding="utf-8", newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # 헤더 작성
            writer.writerows(result)  # 데이터 작성

        print(f"파일 {csv_filename}이(가) 처리되어 저장되었습니다.")

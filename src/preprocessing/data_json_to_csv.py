import os
import json
import csv

# 현재 스크립트(insert_data.py)의 절대 경로를 기준으로 Training 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
original_dir = os.path.join(BASE_DIR, "data\gsang_original_data\Training")
processed_dir = os.path.join(BASE_DIR, "data\gsang_processed_data\Training")

# Training 폴더 안의 JSON 파일들 순회
for filename in os.listdir(original_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(original_dir, filename)

        # JSON 파일 열기
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 결과를 담을 리스트
        result = []
        for utter in data["utterance"]:
            dialect = utter["dialect_form"]
            standard = utter["standard_form"]
            result.append((dialect, standard))

        # 파일 이름을 바탕으로 CSV 파일 경로 설정
        csv_filename = filename.replace(".json", ".csv")
        csv_file_path = os.path.join(processed_dir, csv_filename)

        # CSV로 저장
        with open(csv_file_path, mode="w", encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["사투리", "표준어"])  # 헤더 작성
            for dialect, standard in result:
                writer.writerow([dialect, standard])  # 각 행 작성

        print(f"파일 {csv_filename}이(가) 생성되었습니다.")

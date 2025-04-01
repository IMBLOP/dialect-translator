import os
import csv

# BASE_DIR 및 처리된 CSV 파일들이 있는 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
processed_dir = os.path.join(BASE_DIR, "data/gsang_processed_data/Validation")

# 합칠 CSV 파일 경로 (예: processed_dir에 combined_filtered.csv라는 파일로 저장)
output_csv_path = os.path.join(processed_dir, "combined_filtered.csv")

# 결과를 저장할 리스트 (헤더 포함)
header = ["dialect", "standard"]
result_rows = []

# processed_dir 내의 모든 CSV 파일을 순회
for filename in os.listdir(processed_dir):
    # 이미 생성된 combined_filtered.csv는 건너뛰기
    if filename.endswith(".csv") and filename != "combined_filtered.csv":
        file_path = os.path.join(processed_dir, filename)
        with open(file_path, mode="r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            file_header = next(reader)  # 각 파일의 헤더 읽기
            for row in reader:
                if len(row) >= 2:
                    dialect = row[0].strip()
                    standard = row[1].strip()
                    # dialect와 standard가 다른 경우만 결과에 추가
                    if dialect != standard:
                        result_rows.append([dialect, standard])
        print(f"파일 '{filename}' 처리 완료.")

# 결과를 하나의 CSV 파일로 저장
with open(output_csv_path, mode="w", encoding="utf-8", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)
    writer.writerows(result_rows)

print(f"합쳐진 CSV 파일이 생성되었습니다: {output_csv_path}")

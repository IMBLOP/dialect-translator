import pandas as pd
import os
from src.database.db_connection import connect_db  # DB 연결 함수 임포트

# CSV 파일이 있는 base 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dirs = {
    "train": os.path.join(BASE_DIR, "data", "gsang_processed_data", "Training"),
    "validation": os.path.join(BASE_DIR, "data", "gsang_processed_data", "Validation")
}

# DB 연결
conn = connect_db()
if conn:
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO GsangDialectData (dialect, standard, data_type)
        VALUES (%s, %s, %s);
    """

    # 각 폴더(Training, Validation) 순회
    for data_type, folder_path in data_dirs.items():
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)

            # 첫 번째 줄은 header로 처리
            df = pd.read_csv(csv_path, header=0, names=["dialect", "standard"])

            df['data_type'] = data_type  # data_type 컬럼 추가
            df = df.where(pd.notna(df), None)  # NaN -> None

            for _, row in df.iterrows():
                cursor.execute(insert_query, (row["dialect"], row["standard"], row["data_type"]))

            print(f"✅ [{data_type}] {csv_file} 에서 {len(df)}개 데이터 삽입 완료")

    conn.commit()
    cursor.close()
    conn.close()
    print("🎉 Training & Validation 데이터 삽입 모두 완료!")

else:
    print("❌ DB 연결 실패")

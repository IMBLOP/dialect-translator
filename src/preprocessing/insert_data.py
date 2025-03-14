import pandas as pd
import psycopg2
import os
from src.database.db_connection import connect_db  # DB 연결 함수 임포트

# CSV 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_file = os.path.join(BASE_DIR, "data", "gsang_dialect.csv")

# CSV 데이터 불러오기
df = pd.read_csv(csv_file, header=None, names=["id", "source", "pos", "word", "definition", "sentence", "translation"])

# 첫 번째 id 컬럼 삭제 및 새로운 인덱스 부여
df.drop(columns=["id"], inplace=True)
df.insert(0, "region", "경상도")  # region 컬럼 추가

# 빈 값(NaN)을 None으로 변환 (DB 입력 시 Null 처리)
df = df.where(pd.notna(df), None)

# DB 연결
conn = connect_db()
if conn:
    cursor = conn.cursor()

    # 데이터 삽입 쿼리
    insert_query = """
        INSERT INTO DialectData (source, region, pos, word, definition, sentence, translation)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """

    # 데이터 INSERT 실행
    for _, row in df.iterrows():
        cursor.execute(insert_query, (row["source"], row["region"], row["pos"], row["word"], row["definition"], row["sentence"], row["translation"]))

    # 변경사항 저장
    conn.commit()
    print(f"✅ {len(df)}개의 데이터가 성공적으로 삽입되었습니다.")

    # 연결 종료
    cursor.close()
    conn.close()

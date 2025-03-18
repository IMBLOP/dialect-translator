from src.database.db_connection import connect_db  # DB 연결 함수 임포트

def init_db():
    conn = connect_db()
    if conn:
        cursor = conn.cursor()

        # 경상도 사투리 테이블 생성
        cursor.execute("""
            CREATE TABLE GsangDialectData (
                id SERIAL PRIMARY KEY,
                dialect TEXT NOT NULL,   -- 경상도 사투리 문장
                standard TEXT NOT NULL,  -- 표준어 번역
                data_type VARCHAR(50) NOT NULL CHECK (data_type IN ('train', 'validation'))
            );
        """)

        conn.commit()
        print("✅ DB 초기화 및 GsangDialectData 테이블 생성 완료!")
        cursor.close()
        conn.close()
    else:
        print("❌ DB 연결에 실패했습니다.")

if __name__ == "__main__":
    init_db()
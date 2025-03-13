from db_connection import connect_db

# 테이블 생성 함수
def create_tables():
    conn = connect_db()  # db_connection.py에서 연결을 시도

    if conn is None:  # 연결 실패 시 함수 종료
        return

    cursor = conn.cursor()

    # 스키마 생성 (필요한 경우)
    cursor.execute("CREATE SCHEMA IF NOT EXISTS public;")

    # DialectData 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS DialectData (
            id SERIAL PRIMARY KEY,
            source TEXT,
            region VARCHAR(50) NOT NULL,
            pos VARCHAR(50),
            word TEXT NOT NULL,
            definition TEXT NOT NULL,
            sentence TEXT,
            translation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # TrainingData 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TrainingData (
            id SERIAL PRIMARY KEY,
            dialect_id INTEGER REFERENCES DialectData(id) ON DELETE CASCADE,
            model_version VARCHAR(50) NOT NULL,
            accuracy FLOAT CHECK (accuracy BETWEEN 0 AND 1),
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # TranslationLog 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TranslationLog (
            id SERIAL PRIMARY KEY,
            dialect_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            translated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 커밋 후 연결 종료
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ 테이블이 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    create_tables()

import psycopg2

DB_CONFIG = {
    "dbname" : "dialect_db",
    "user" : "postgres",
    "password" : "rhkr1122!",
    "host" : "localhost",
    "port" : "5432"
}

def connect_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("DB 연결 성공")
        return conn
    except Exception as e:
        print(f"DB 연결 실패 : {e}")
        return None

# 연결 테스트
if __name__ == "__main__":
    conn = connect_db()
    if conn:
        conn.close()
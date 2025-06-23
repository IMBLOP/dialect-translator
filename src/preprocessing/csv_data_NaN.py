import os
import pandas as pd

# CSV들이 들어있는 폴더 경로
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
processed_dir = os.path.join(BASE_DIR, "Data/Training")

for filename in os.listdir(processed_dir):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(processed_dir, filename)
    # 1) 모든 칼럼을 문자열로 읽고, 2) NaN → 빈 문자열 로 통일
    df = pd.read_csv(file_path, dtype=str).fillna("")

    # NaN/빈 문자열이 하나라도 있는 행 마스크
    nan_mask = (
        df['dialect'].str.strip().eq('') |
        df['standard'].str.strip().eq('')
    )
    # dialect == standard 인 행 마스크
    equal_mask = df['dialect'].str.strip() == df['standard'].str.strip()

    # 둘 중 하나라도 True인 행(=삭제 대상) 제외
    mask_to_drop = nan_mask | equal_mask
    df_filtered = df[~mask_to_drop].reset_index(drop=True)

    # (원한다면) 삭제 전/후 개수 확인
    total_rows   = len(df)
    kept_rows    = len(df_filtered)
    dropped_rows = mask_to_drop.sum()
    print(f"{filename}: 전체 {total_rows}개 → 보관 {kept_rows}개, 삭제 {dropped_rows}개")

    # 필터링된 내용을 CSV로 덮어쓰기
    df_filtered.to_csv(file_path, index=False, encoding='utf-8')

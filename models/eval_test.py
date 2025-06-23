import os
import math
import torch
import pandas as pd
from transformers import BartForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

# 1) Validation CSV 경로
VAL_CSV = r"D:\PyCharm\dialect_translator\Data\Validation\jeolla_validation.csv"

# 2) CSV에서 그대로 읽어오기 (이미 전처리 완료된 상태)
df = pd.read_csv(VAL_CSV, dtype=str)

# 3) 앞 100개 샘플만 사용
max_samples = 100
df_subset = df.iloc[:max_samples].reset_index(drop=True)
dialects  = df_subset["dialect"].tolist()
standards = df_subset["standard"].tolist()

print(f"총 {len(dialects)}쌍의 데이터로 샘플링하여 출력합니다.\n")

# 4) 모델 로딩 (학습된 모델 경로)
MODEL_DIR = r"D:\PyCharm\dialect_translator\models\jeolla"
model     = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 5) 예측 결과 저장할 리스트
results = []

# 6) 배치 단위로 추론
batch_size  = 16
num_batches = math.ceil(len(dialects) / batch_size)

for i in tqdm(range(num_batches), desc="Translating"):
    batch_inputs = dialects[i*batch_size : (i+1)*batch_size]

    inputs = tokenizer(
        batch_inputs,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding="longest"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=64,
            early_stopping=True
        )

    batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # 결과 저장
    for src, pred, tgt in zip(batch_inputs, batch_preds, standards[i*batch_size : (i+1)*batch_size]):
        results.append({
            "dialect": src,
            "predict": pred,
            "standard": tgt
        })

# 7) 결과를 DataFrame으로 출력
result_df = pd.DataFrame(results, columns=["dialect", "predict", "standard"])

# 8) TSV 형식으로 출력 (Excel에 그대로 붙여넣기 OK)
print(result_df.to_csv(sep='\t', index=False))

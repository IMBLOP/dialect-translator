import os
import math
import torch
import pandas as pd
from transformers import BartForConditionalGeneration, AutoTokenizer
from bert_score import score
from tqdm import tqdm

# 1) Validation CSV 경로
VAL_CSV = r"D:\PyCharm\dialect_translator\Data\Validation\gangwon_vali.csv"

# 2) CSV에서 그대로 읽어오기 (이미 전처리 완료된 상태)
df = pd.read_csv(VAL_CSV, dtype=str)

# 3) 앞 30,000개 샘플만 사용
max_samples = 30000
dialects  = df["dialect"].tolist()[:max_samples]
standards = df["standard"].tolist()[:max_samples]

print(f"총 {len(dialects)}쌍의 데이터로 평가를 진행합니다.\n")

# 4) 모델 로딩 (학습된 모델 경로)
MODEL_DIR = r"D:\PyCharm\dialect_translator\models\gangwon"
model     = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 5) 모델 예측 결과 저장할 리스트
predictions = []

# 6) 배치 단위로 추론
batch_size  = 16
num_batches = math.ceil(len(dialects) / batch_size)

print("모델 추론(번역) 진행 중...\n")
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
    predictions.extend(batch_preds)

# 7) BERTScore 계산
print("\nBERTScore 계산 중...")
P, R, F1 = score(predictions, standards, lang="ko")

mean_p = P.mean().item()
mean_r = R.mean().item()
mean_f = F1.mean().item()

# 8) 결과 출력
print("=== BERTScore 결과 (한국어) ===")
print(f"Precision: {mean_p:.4f}")
print(f"Recall   : {mean_r:.4f}")
print(f"F1       : {mean_f:.4f}")
print("\n평가 완료!")

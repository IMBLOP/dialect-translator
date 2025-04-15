import math
import pandas as pd
import sacrebleu
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

# 1) 모델 & 토크나이저 로딩 (AutoTokenizer 사용)
MODEL_DIR = r"C:\INHATC\dialect_translator\models\model03"  # 학습 완료된 모델 폴더
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# GPU 사용 가능하면 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2) 검증 데이터 로딩
VAL_CSV = r"C:\INHATC\dialect_translator\data\gsang_processed_data\Validation\combined_filtered.csv"
df = pd.read_csv(VAL_CSV)
print("검증 데이터 개수:", len(df))

# 3) 배치 처리 설정
batch_size = 16
num_batches = math.ceil(len(df) / batch_size)

predictions = []
references = []

# 4) 배치 단위로 추론 (tqdm으로 진행 상황 표시)
for i in tqdm(range(num_batches), desc="Evaluating", ncols=100):
    batch_df = df.iloc[i * batch_size: (i + 1) * batch_size]
    batch_dialects = batch_df["dialect"].tolist()
    batch_standards = batch_df["standard"].tolist()

    # 배치 토크나이징 (padding 자동 처리)
    inputs = tokenizer(batch_dialects, return_tensors="pt", max_length=64, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 모델 추론 (generate) - 배치 처리
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=64,
            early_stopping=True
        )

    # 배치 결과 디코딩
    batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    predictions.extend(batch_preds)
    references.extend(batch_standards)

# 5) sacrebleu 형식에 맞춰 레퍼런스 변환 ([[정답1], [정답2], ...])
references_for_bleu = [[ref] for ref in references]

# 6) BLEU 점수 계산
bleu = sacrebleu.corpus_bleu(predictions, references_for_bleu)
print(f"\nBLEU score: {bleu.score:.2f}")

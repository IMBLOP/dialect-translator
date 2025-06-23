import os
import pandas as pd
from datasets import Dataset
from transformers import (
    BartForConditionalGeneration,
    AutoTokenizer,  # PreTrainedTokenizerFast 대신 AutoTokenizer 사용
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# 절대 경로로 저장 위치 지정
SAVE_DIR = r"D:\PyCharm\dialect_translator\models\gangwon"
CSV_PATH = r"D:\PyCharm\dialect_translator\Data\Training\gangwon_train.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "gogamza/kobart-base-v2"
MAX_LENGTH = 64

def load_dataset_from_csv():
    """CSV 파일에서 데이터 불러오기"""
    df = pd.read_csv(CSV_PATH)
    return df.to_dict(orient="records")

def preprocess_function(examples, tokenizer):
    """입력/출력 텍스트를 토크나이징 (Seq2SeqTrainer용)"""
    model_inputs = tokenizer(
        examples["dialect"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    # 타겟 시퀀스 토크나이징
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["standard"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("CSV에서 데이터 불러오는 중...")
    data = load_dataset_from_csv()
    if not data:
        print("데이터가 없습니다.")
        return

    # Dataset으로 변환
    dataset = Dataset.from_list(data)

    # 토크나이저, 모델 로드 (AutoTokenizer 사용)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    print("데이터 전처리 중...")
    tokenized = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    print("학습 시작...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(SAVE_DIR, "checkpoints"),
        eval_strategy="no",
        save_strategy="epoch",
        logging_dir=os.path.join(SAVE_DIR, "logs"),
        per_device_train_batch_size=16,
        num_train_epochs=3,
        learning_rate=3e-5,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
        logging_steps=100,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("모델 저장 중...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"모델 저장 완료: {SAVE_DIR}")

if __name__ == "__main__":
    main()

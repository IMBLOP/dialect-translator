import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from src.database.db_connection import connect_db

# 절대 경로로 저장 위치 지정
SAVE_DIR = r"C:\INHATC\dialect_translator\models"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "gogamza/kobart-base-v2"
MAX_LENGTH = 128


def load_dataset_from_db():
    """DB에서 GsangDialectData 테이블 데이터 불러오기"""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT dialect, standard, data_type FROM GsangDialectData")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [{"dialect": r[0], "standard": r[1], "data_type": r[2]} for r in rows]


def prepare_datasets(data):
    """train / validation DatasetDict로 분리"""
    df = pd.DataFrame(data)
    train_df = df[df["data_type"] == "train"]
    val_df = df[df["data_type"] == "validation"]
    return DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False)
    })


def preprocess_function(examples, tokenizer):
    """입력/출력 텍스트를 토크나이징"""
    return tokenizer(
        examples["dialect"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        text_target=examples["standard"]
    )


def main():
    print("데이터 불러오는 중...")
    data = load_dataset_from_db()
    if not data:
        print("데이터가 없습니다.")
        return

    datasets = prepare_datasets(data)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    print("데이터 전처리 중...")
    tokenized = datasets.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    print("학습 시작...")
    training_args = TrainingArguments(
        output_dir=os.path.join(SAVE_DIR, "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(SAVE_DIR, "logs"),
        per_device_train_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    print("모델 저장 중...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"모델 저장 완료: {SAVE_DIR}")


if __name__ == "__main__":
    main()

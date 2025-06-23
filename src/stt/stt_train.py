import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torch._dynamo
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# ── 최적화 설정 ─────────────────────────────────────────────────────────
# 1) TorchDynamo 컴파일 비활성화 (PyTorch 2.x)
torch._dynamo.disable()

# 2) cuDNN 벤치마크 모드 (GPU에서 가장 빠른 커널 선택)
torch.backends.cudnn.benchmark = True

# 🔧 환경 설정
os.environ["WANDB_DISABLED"] = "true"

# 📦 데이터셋 로딩
CACHE_TRAIN_PATH = "E:/hf_cache/processed/train_ds"
CACHE_EVAL_PATH  = "E:/hf_cache/processed/eval_ds"

print("📦 Loading datasets...")
train_ds = Dataset.load_from_disk(CACHE_TRAIN_PATH)
eval_ds  = Dataset.load_from_disk(CACHE_EVAL_PATH)
print(f"✅ Loaded: {len(train_ds)} train / {len(eval_ds)} eval samples")

# ✅ Whisper 모델과 Processor 로드 (Multilingual + 한국어 설정)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="Korean", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# ※ Gradient Checkpointing 제거: 속도 향상을 위해 주석 처리
# model.config.use_cache = False
# model.gradient_checkpointing_enable()

# 🧩 Collator 정의 (attention_mask 제거 포함)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[list, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1) feature_extractor로 pad → input_features + attention_mask 반환
        input_batch = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_batch, return_tensors="pt")

        # 2) 모델 forward에 불필요한 attention_mask 제거
        batch.pop("attention_mask", None)

        # 3) labels padding
        label_batch = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_batch, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 4) BOS 토큰만 있는 첫 열 제거
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 📊 CER metric 함수
cer_metric = evaluate.load("cer")
def compute_metrics(p):
    pred_ids  = p.predictions
    label_ids = p.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_strs  = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_strs, references=label_strs)
    return {"cer": cer * 100}

# 🏗️ 학습 설정 (Colab과 동일한 하이퍼파라미터)
training_args = Seq2SeqTrainingArguments(
    output_dir="D:/PyCharm/dialect_translator/src/stt/jeju_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_steps=4000,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    fp16=True,                   # GPU가 있을 때만 켜짐
    eval_accumulation_steps=4,
    load_best_model_at_end=False,
    report_to="none"
)

# 🧠 Trainer 설정
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,                # 전체 eval_ds 사용
    data_collator=collator,
    compute_metrics=None,                # CER 사용 시 compute_metrics=compute_metrics
    tokenizer=processor.feature_extractor
)

# 🚀 훈련 시작
trainer.train()

# 💾 모델 저장
trainer.save_model("D:/PyCharm/dialect_translator/src/stt/jeju_model")
processor.save_pretrained("D:/PyCharm/dialect_translator/src/stt/jeju_model")

print("🎉 Training complete. Model & processor saved!")

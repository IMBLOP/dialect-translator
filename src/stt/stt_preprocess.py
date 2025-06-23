# 📁 stt_step1_to_4_preprocess.py
import os

# 캐시 디렉토리 설정
os.environ["HF_HOME"]            = r"E:\hf_cache"
os.environ["HF_DATASETS_CACHE"]  = r"E:\hf_cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"E:\hf_cache\models"
os.environ["HF_METRICS_CACHE"]   = r"E:\hf_cache\metrics"

import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor

CACHE_TRAIN_PATH = "E:/hf_cache/processed/train_ds"
CACHE_EVAL_PATH  = "E:/hf_cache/processed/eval_ds"

print("[1/4] Loading Whisper processor (small)...")
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="Korean", task="transcribe"
)

print("[2/4] Loading manifest and splitting dataset...")
MANIFEST = "gangwon_manifest.jsonl"
ds = load_dataset("json", data_files={"train": MANIFEST}, split="train")
ds = ds.train_test_split(test_size=0.1, seed=42)

print("[3/4] Casting audio column to Audio type...")
ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16_000))

print("[4/4] Preprocessing and saving dataset...")
def preprocess(example):
    audio = example["audio_filepath"]["array"]
    sr = example["audio_filepath"]["sampling_rate"]
    segment = audio[int(example["start_time"] * sr):int(example["end_time"] * sr)]
    in_feat = processor.feature_extractor(segment, sampling_rate=sr).input_features[0]
    labels = processor.tokenizer(example["text"], add_special_tokens=True).input_ids
    return {"input_features": in_feat, "labels": labels}

print("    Preprocessing train set...")
train_ds = ds["train"].map(preprocess, remove_columns=ds["train"].column_names, desc="Preprocessing train")
print("    Preprocessing eval set...")
eval_ds  = ds["test"].map(preprocess, remove_columns=ds["test"].column_names, desc="Preprocessing eval")

print("    Saving to disk...")
train_ds.save_to_disk(CACHE_TRAIN_PATH)
eval_ds.save_to_disk(CACHE_EVAL_PATH)
print("✅ Preprocessing complete!")

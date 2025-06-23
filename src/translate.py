import os
import math
import torch
import torchaudio
import kss  # 문장 단위 분리
import pyttsx3
import numpy as np
from tqdm import tqdm
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    BartForConditionalGeneration, AutoTokenizer
)

# 1) 사용자에게 사투리 종류 선택 받기
print("사투리 번역기의 지역을 선택하세요:")
print("1. 경상도 사투리 번역")
print("2. 제주도 사투리 번역")
print("3. 전라도 사투리 번역")
print("4. 강원도 사투리 번역")
choice = input("번호 입력 (1 ~ 4): ").strip()
if choice == "1":
    stt_dir = r"D:\PyCharm\dialect_translator\src\stt\gsang_model"
    trans_dir = r"D:\PyCharm\dialect_translator\models\model03"
    wav_dir = r"D:\PyCharm\dialect_translator\Data\gsang_voice_data\wav1"
    print("[경상도 사투리 모델 및 파일 경로 설정]")
elif choice == "2":
    stt_dir = r"D:\PyCharm\dialect_translator\src\stt\jeju_model"
    trans_dir = r"D:\PyCharm\dialect_translator\models\jeju"
    wav_dir = r"D:\PyCharm\dialect_translator\Data\jeju_voice_data\wav1"
    print("[제주도 사투리 모델 및 파일 경로 설정]")
elif choice == "3":
    stt_dir = r"D:\PyCharm\dialect_translator\src\stt\jeolla_model"
    trans_dir = r"D:\PyCharm\dialect_translator\models\jeolla"
    wav_dir = r"D:\PyCharm\dialect_translator\Data\jeolla_voice_data\wav1"
    print("[전라도 사투리 모델 및 파일 경로 설정]")
elif choice == "4":
    stt_dir = r"D:\PyCharm\dialect_translator\src\stt\gangwon_model"
    trans_dir = r"D:\PyCharm\dialect_translator\models\gangwon"
    wav_dir = r"D:\PyCharm\dialect_translator\Data\gangwon_voice_data\wav1"
    print("[강원도 사투리 모델 및 파일 경로 설정]")
else:
    raise ValueError("잘못된 입력입니다. 1 ~ 4 중 입력해주세요.")

# 2) 장비 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 사용 장치: {device}")

# 3) STT 모델 & processor 로드
print("📥 STT 모델 로드 중...")
stt_model = WhisperForConditionalGeneration.from_pretrained(stt_dir).to(device)
stt_model.config.forced_decoder_ids = None
stt_model.config.suppress_tokens = []
stt_model.generation_config.forced_decoder_ids = None
stt_model.generation_config._from_model_config = True
stt_processor = WhisperProcessor.from_pretrained(stt_dir)
stt_model.eval()

# 4) 번역 모델 로드
print("📥 번역 모델 로드 중...")
trans_model = BartForConditionalGeneration.from_pretrained(trans_dir).to(device)
trans_tokenizer = AutoTokenizer.from_pretrained(trans_dir)
trans_model.eval()

# 5) pyttsx3 TTS 엔진 설정
print("🔊 TTS 엔진 설정 중...")
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)


# 6) 오디오 분할 함수 (수정됨)
def load_audio_chunks(path, chunk_sec=30):
    try:
        wav, sr = torchaudio.load(path)

        # 스테레오를 모노로 변환
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # 리샘플링
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        # (1, N) -> (N,) 형태로 변환
        wav = wav.squeeze()

        size = 16000 * chunk_sec
        chunks = []

        for i in range(0, len(wav), size):
            chunk = wav[i:i + size]
            # numpy array로 변환
            chunk_np = chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk
            chunks.append(chunk_np)

        return chunks

    except Exception as e:
        print(f"❌ 오디오 로드 실패: {e}")
        return []


# 7) STT + 문장 분리 (수정됨)
def transcribe_sentences(path):
    if not os.path.exists(path):
        print(f"❌ 파일을 찾을 수 없습니다: {path}")
        return []

    chunks = load_audio_chunks(path)
    if not chunks:
        return []

    print(f"🔍 총 {len(chunks)}개 chunk 처리")
    texts = []

    for i, chunk in enumerate(chunks, 1):
        try:
            # chunk가 이미 numpy array인지 확인
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.numpy()

            # processor에 단일 오디오로 전달 (리스트로 감싸지 않음)
            inp = stt_processor(chunk, sampling_rate=16000, return_tensors="pt")
            feats = inp.input_features.to(device)

            with torch.no_grad():
                preds = stt_model.generate(
                    feats,
                    max_new_tokens=440,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False,
                    task="transcribe",
                    language="ko"
                )

            txt = stt_processor.batch_decode(preds, skip_special_tokens=True)[0]
            texts.append(txt)
            print(f"  • chunk {i}/{len(chunks)} 완료: {txt[:30]}...")

        except Exception as e:
            print(f"❌ chunk {i} 처리 실패: {e}")
            continue

    if not texts:
        print("❌ 음성 인식 결과가 없습니다.")
        return []

    # 문장 분리
    combined_text = " ".join(texts)
    sentences = kss.split_sentences(combined_text)

    # 빈 문장 제거
    return [s.strip() for s in sentences if s.strip()]


# 8) 한 문장씩 번역
def translate_sentences(dialect_sents):
    if not dialect_sents:
        return []

    results = []
    for sent in tqdm(dialect_sents, desc="🔄 번역 중"):
        try:
            enc = trans_tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            ).to(device)

            with torch.no_grad():
                out = trans_model.generate(
                    enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=64,
                    eos_token_id=trans_tokenizer.eos_token_id,
                    pad_token_id=trans_tokenizer.pad_token_id,
                    num_beams=4,
                    early_stopping=True,
                )

            translated = trans_tokenizer.decode(out[0], skip_special_tokens=True)
            results.append(translated)

        except Exception as e:
            print(f"❌ 번역 실패: {sent[:20]}... - {e}")
            results.append(sent)  # 번역 실패시 원문 유지

    return results


# 9) 실행 및 결과 출력
if __name__ == "__main__":
    try:
        # wav 파일 선택: 디렉토리에서 원하는 파일명을 지정하세요
        filename = "1.wav"  # 예시 파일명
        wav_path = os.path.join(wav_dir, filename)

        print(f"🎵 처리할 파일: {wav_path}")

        # STT 처리
        print("\n🎙️ 음성 인식 시작...")
        dialect_sents = transcribe_sentences(wav_path)

        if not dialect_sents:
            print("❌ 음성 인식 결과가 없습니다. 프로그램을 종료합니다.")
            exit()

        print(f"✅ {len(dialect_sents)}개 문장 인식 완료")

        # 번역 처리
        print("\n🔄 번역 시작...")
        std_sents = translate_sentences(dialect_sents)

        # 사투리와 표준어가 다른 문장만 필터링
        filtered = [(d, s) for d, s in zip(dialect_sents, std_sents)
                    if d.strip() != s.strip() and len(d.strip()) > 0]

        if not filtered:
            print("❌ 번역할 사투리 문장이 없습니다.")
            exit()

        print(f"✅ {len(filtered)}개 문장 번역 완료")

        # 결과 출력
        print("\n📋 최종 결과 (번호 포함):\n")
        for idx, (d, s) in enumerate(filtered, 1):
            print(f"{idx}. [사투리] {d}")
            print(f"   [표준어] {s}")
            print("-" * 50)

        # 사용자 인터랙션
        print(f"\n🔢 읽을 문장 번호를 입력하세요 (1-{len(filtered)}, 종료하려면 엔터만 입력):")
        while True:
            choice = input("번호> ").strip()
            if not choice:
                print("프로그램을 종료합니다.")
                break

            if not choice.isdigit() or not (1 <= int(choice) <= len(filtered)):
                print(f"1부터 {len(filtered)} 사이의 숫자를 입력해주세요.")
                continue

            d, s = filtered[int(choice) - 1]
            print(f"\n[사투리] {d}")
            print(f"[표준어] {s}")
            print("-" * 50)

            try:
                print("🔊 음성 출력 중...")
                tts_engine.say(s)
                tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ TTS 실패: {e}")

    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류 발생: {e}")
        print(f"오류 타입: {type(e).__name__}")
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import kss  # 문장 분리 라이브러리
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 & processor
model = WhisperForConditionalGeneration.from_pretrained("D:\PyCharm\dialect_translator\src\stt\gangwon_model")
model.to(device)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None
model.generation_config._from_model_config = True
processor = WhisperProcessor.from_pretrained("D:\PyCharm\dialect_translator\src\stt\gangwon_model")


def load_audio_chunks(file_path, chunk_sec=30):
    waveform, sample_rate = torchaudio.load(file_path)

    # 스테레오를 모노로 변환 (채널이 여러 개인 경우)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 리샘플링
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # (1, N) -> (N,) 형태로 변환
    waveform = waveform.squeeze()

    chunk_size = 16000 * chunk_sec
    chunks = []

    for i in range(0, len(waveform), chunk_size):
        chunk = waveform[i:i + chunk_size]
        # numpy array로 변환
        chunk_np = chunk.numpy()
        chunks.append(chunk_np)

    return chunks


def transcribe_long_audio(file_path):
    audio_chunks = load_audio_chunks(file_path)
    full_transcription = []

    print(f"🔍 총 {len(audio_chunks)}개의 chunk로 나눔")

    for idx, chunk in enumerate(audio_chunks):
        # chunk가 numpy array인지 확인하고 변환
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.numpy()

        # processor에 단일 오디오로 전달 (리스트로 감싸지 않음)
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        predicted_ids = model.generate(
            input_features,
            max_new_tokens=440,
            num_beams=5,
            do_sample=False,
            early_stopping=True,
            task="transcribe",
            language="ko"
        )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcription.append(transcription)
        print(f"🧩 chunk {idx + 1}/{len(audio_chunks)} 완료: {transcription[:50]}...")

    combined_text = " ".join(full_transcription)
    return combined_text


# 실행
if __name__ == "__main__":
    wav_path = r"D:\PyCharm\dialect_translator\Data\gangwon_voice_data\wav1\1.wav"

    try:
        result = transcribe_long_audio(wav_path)

        print("\n🎙️ 전체 인식 결과 (문장 단위):")
        split_sentences = kss.split_sentences(result)

        for sentence in split_sentences:
            if sentence.strip():  # 빈 문장 제외
                print(sentence.strip())

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print(f"오류 타입: {type(e).__name__}")
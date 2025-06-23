import os, json
from glob import glob

# 이 스크립트는 JSON 메타데이터와 대응하는 WAV 파일들을 읽어
# "utterance" 단위로 분리하여 manifest JSONL을 생성합니다.

def make_manifest(json_dir: str, wav_dir: str, out_path: str):
    records = []
    for jp in glob(os.path.join(json_dir, "*.json")):
        meta = json.load(open(jp, encoding="utf-8"))
        session_id = meta.get("id")  # e.g. "DKSR20000890"
        wav_path   = os.path.join(wav_dir, f"{session_id}.wav")

        # 각 utterance 단위로 레코드를 추가
        for utt in meta.get("utterance", []):
            start = utt.get("start", 0.0)
            end   = utt.get("end", 0.0)
            # 너무 짧은 구간은 생략 (0.5초 미만)
            if end - start < 0.5:
                continue
            text = utt.get("dialect_form") or utt.get("standard_form") or ""
            records.append({
                "audio_filepath": wav_path,
                "start_time": start,
                "end_time":   end,
                "text":       text
            })

    # JSONL 형식으로 저장
    with open(out_path, "w", encoding="utf-8") as fw:
        for rec in records:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Manifest saved to {out_path}, total utterances: {len(records)}")


if __name__ == "__main__":
    # 경로를 실제 환경에 맞게 수정하세요
    JSON_DIR = r"D:\PyCharm\dialect_translator\Data\gangwon_voice_data\json1"
    WAV_DIR  = r"D:\PyCharm\dialect_translator\Data\gangwon_voice_data\wav1"
    OUT_MANIFEST = "gangwon_manifest.jsonl"

    make_manifest(JSON_DIR, WAV_DIR, OUT_MANIFEST)

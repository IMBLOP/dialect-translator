import torch
from transformers import BartForConditionalGeneration, AutoTokenizer

# 저장된 모델 경로
MODEL_DIR = r"D:\PyCharm\dialect_translator\models\model03"  # 저장된 모델 경로로 수정

# 모델 & 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)

# GPU 사용 가능하면 GPU로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 테스트 함수
def translate_dialect_to_standard(text: str) -> str:
    # padding="max_length"를 없애고 padding=True로 변경
    inputs = tokenizer([text], return_tensors="pt", max_length=64, truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # GPU로 보내기

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=64,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# CLI 입력
if __name__ == "__main__":
    print("💬 사투리 문장을 입력하면 표준어로 변환해드립니다!")
    while True:
        try:
            sentence = input("\n🗣 사투리 입력 (종료하려면 엔터): ").strip()
            if not sentence:
                print("종료합니다.")
                break
            result = translate_dialect_to_standard(sentence)
            print("➡️ 표준어 번역:", result)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break

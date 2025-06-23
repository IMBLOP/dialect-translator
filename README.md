# 사투리 번역기
Whisper 기반 음성 인식, KoBART 기반 기계 번역, pyttsx3 기반 음성 합성을 통합하여 한국어 방언 발화 음성 데이터를 표준어로 변환 후 재생하는 한국어 방언 번역기 프로젝트입니다.
  ### 학술대회 논문([Whisper·KoBART 기반 한국어 방언-표준어 번역 시스템 설계 및 구현]())
　

## 구현 계획
- 개발 환경 구축
- 데이터 수집 및 저장
  - AI HUB의 지역별 방언 발화 데이터 활용
- 데이터 가공
- AI 모델 선정
  - KoBART 모델 사용
- AI 모델 학습
  - 지역별 방언 데이터 학습
- 텍스트 번역기 기능 구현
- 음성 인식 후 텍스트로 변환하는 모듈 추가
  - Whisper 모델 학습 및 사용
- 번역한 표준어 문장을 TTS로 읽어주는 모듈 추가
  - pyttsx3 모듈 사용
- 음성인식-기계번역-음성합성 통합 파이프라인 구축
  - 지역별 통합 번역 기능 업데이트
    

## 개발 도구 및 기술 스택
### 1. 개발 언어
- #### Python 3.11
  
### 2. 핵심 라이브러리 및 프레임워크
- #### PyTorch 2.7.0 (+ torchaudio 2.7.0, torchvision 0.22.0)  
  - 딥러닝 학습·추론 핵심 프레임워크 및 오디오·비전 I/O 지원
- #### Accelerate 1.7.0  
  - bf16 혼합정밀, 그래디언트 누적, 멀티-디바이스 관리를 간소화
- #### Transformers 4.52.3  
  - Whisper·KoBART 등 Hugging Face 모델 로딩·추론 API 제공
- #### datasets 3.6.0  
  - 대용량 음성·텍스트를 Arrow 포맷으로 관리·캐싱
- #### SentencePiece 0.1.99 & tokenizers 0.21.1  
  - KoBART 서브워드 토크나이저 및 커스텀 BPE 처리
- #### KoNLPy 0.6.0 / python-mecab-ko 1.3.7  
  - 한국어 형태소 분석(품사 태깅, 불용어 제거 등)
- #### KSS 6.0.4  
  - 한국어 문장 단위 분리(句讀點 보정 포함)
- #### SpeechRecognition 3.14.1 + soundfile 0.13.1 + pydub 0.25.1  
  - WAV/마이크 입력 로딩 및 Whisper 전처리 지원
- #### pyttsx3 2.98  
  - 로컬 SAPI5 / espeak 기반 TTS 합성(프로토타입 출력)
- #### scikit-learn 1.4.1 · jiwer 3.1.0 · bert-score 0.3.13 · sacrebleu 2.5.1 · evaluate 0.4.3  
  - 모델 평가(WER·BERTScore·BLEU 등) 및 전처리·통계 툴킷
- #### tqdm 4.67.1 · numpy 1.26.4 · pandas 2.2.3  
  - 데이터 처리, 배열 연산, 진행률 출력 등 유틸리티 전반
    
### 3. AI 모델
- #### KoBART (gogamza/kobart-base-v2) [선정 과정](https://github.com/IMBLOP/dialect-translator/issues/2#issue-2995355902)
  - Text-to-Text Translation Model
  - 40GB 이상의 한국어 텍스트에 대해 학습한 encoder-decoder 언어 모델
- #### 모델 학습
  - [학습 코드](src/training/training.py)
  - [문제 해결](https://github.com/IMBLOP/dialect-translator/issues/1#issue-2995320637)
- #### 모델 평가(BERT_SCORE)
  - [평가 코드](src/evaluation/bert_score_eval.py)
  - [문제 해결](https://github.com/IMBLOP/dialect-translator/issues/3#issue-2995488377)

### 4. 개발 환경
- #### PyCharm
  - Python 개발을 위한 IDE
- #### Google Colab
  - TTS(Whisper) 모델 학습에 필요한 그래픽 메모리 최적화를 위해 사용
- #### GitHub
  - 버전 관리 및 프로젝트 진행사항 기록을 위한 플랫폼


## 사용 데이터
[AI HUB 한국어 방언 발화(경상도)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=119)
 
  [AI HUB 한국어 방언 발화(제주도)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=121)
  
  [AI HUB 한국어 방언 발화(전라도)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=120)
  
  [AI HUB 한국어 방언 발화(강원도)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=118)  


---


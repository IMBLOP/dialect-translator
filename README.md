# 사투리 번역기
Whisper 기반 음성 인식, KoBART 기반 기계 번역, pyttsx3 기반 음성 합성을 통합하여 한국어 방언 발화 음성 데이터를 표준어로 변환 후 재생하는 한국어 방언 번역기 프로젝트입니다.
  ### 학술대회 논문([Whisper·KoBART 기반 한국어 방언-표준어 번역 시스템 설계 및 구현]())
　

## 구현 계획
- 개발 환경 구축
- 데이터 수집 및 저장
  - AI HUB의 경상도 사투리 데이터 활용
- 데이터 가공
- AI 모델 선정
  - KoBART 모델 사용
- AI 모델 학습
  - 경상도 사투리 학습
- 텍스트 번역기 기능 구현
- 음성 인식 후 텍스트로 변환하는 모듈 추가
  - Whisper 모델 학습 및 사용
- 번역한 표준어 문장을 TTS로 읽어주는 모듈 추가
  - pyttsx3 모듈 사용
- 추후 다른 지역 데이터도 추가 및 학습
  - 지역별 번역 기능 업데이트
    

　


## 개발 도구 및 기술 스택
### 1. 개발 언어
- #### Python 3.11
### 2. 핵심 라이브러리 및 프레임워크
- #### PyTorch 2.6.0
  - 딥러닝 모델 학습 및 예측에 사용
- #### Transformers 4.49.0
  - 사투리 번역을 위한 Transformer 기반 모델 구현
- #### KoNLPy 0.6.0
  - 한국어 텍스트 형태소 분석 및 자연어 처리(NLP)에 사용
- #### SpeechRecognition 3.14.1
  - 음성 입력을 텍스트로 변환하는 기능 제공
- #### SentencePiece 0.1.99
  - KoBART tokenizer에 사용되는 subword segmentation 도구
- #### scikit-learn 1.4.1
  - 모델 성능 평가 및 전처리에 사용
- #### datasets 3.4.1
  - Hugging Face Dataset 포맷 기반 데이터 관리에 사용
- #### bert_score 0.3.13
  - Model Evaluation 단계의 의미적 유사도 측정에 사용
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


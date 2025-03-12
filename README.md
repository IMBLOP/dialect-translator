# 사투리 번역기
　대한민국의 다양한 지역에서 사용되고 있는 사투리를 표준말로 번역해주는 번역기를 제작하는 프로젝트입니다.

　

## 구현 계획
- 개발 환경 구축
- DB 설게
- AI 모델 선정
- 데이터 수집 및 저장
  - Kaggle의 경상, 제주 사투리 데이터 활용
- AI 모델 학습
  - 경상, 제주 사투리 학습
- 텍스트 번역기 기능 구현
- 음성 인식 후 텍스트로 변환하는 모듈 추가
- 추후 다른 지역 데이터도 추가 및 학습
  - 지역별 번역 기능 업데이트
- 특수한 언어 데이터 수집 및 학습
  - 신조어, 판교어 등 추가
    

　


## 개발 도구 및 기술 스택
### 1. 개발 언어
- #### Python 3.11
### 2. AI 모델 및 라이브러리
- #### PyTorch 2.6.0
  - 딥러닝 모델 학습 및 예측을 위한 프레임워크
- #### Transformers 4.49.0
  - 사투리 번역을 위한 Transformer 기반 모델(예: T5, BERT, GPT 등)
- #### KoNLPy 0.6.0
  - 한국어 텍스트 자연어 처리(NLP)를 위한 라이브러리
- #### SpeechRecognition 3.14.1
  - 음성 인식 및 텍스트 변환 기능을 제공하는 라이브러리
### 3. 데이터베이스
- #### PostgreSQL
  - 데이터 수집 및 저장을 위한 관계형 데이터베이스
  - 데이터 관리 및 쿼리 효율성을 위해 사용
### 4. 개발 환경
- #### PyCharm
  - Python 개발을 위한 IDE
- #### GitHub
  - 버전 관리 및 프로젝트 진행사항 기록 및 확인을 위한 플랫폼


## DATABASE ERD
![DialetTranslator (1)](https://github.com/user-attachments/assets/d6e60be5-0ff8-4e5d-8f49-c36022e29861)




## 사용 데이터 (제주, 경상북도)
　[Kaggle 한국어 지역 방언 분류 (제주, 경상도, 전라도)](https://www.kaggle.com/competitions/hai2023summer/overview)

　[Kaggle Korean Dialect Dictionary (제주, 경상북도)](https://www.kaggle.com/datasets/daraejang/korean-dialect-dictionary)

---


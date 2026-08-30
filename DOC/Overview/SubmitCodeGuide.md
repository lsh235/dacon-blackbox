# DACON 코드 제출 대회 가이드

## 1. 대회 방식 소개

### 1.1 왜 코드 제출 방식인가요?
* **공정한 모델 평가 보장**: 평가 데이터(X)를 참가자에게 공개하지 않고 서버 환경에서 직접 실행하므로 치팅이 불가능하며, 데이터에 특화된 편법이나 과적합 문제를 근본적으로 해결합니다.
* **실무 환경을 고려한 종합 평가**: 정확도뿐만 아니라 추론 속도, 메모리 효율성, 안정성 등 다면적 성능을 종합 평가하여 실제 배포 환경과 유사한 조건에서 성능을 검증합니다.
* **즉시 활용 가능한 실행 모델 확보**: 검증된 추론 파이프라인을 갖춘 완성된 모델을 확보하여 개발부터 배포까지의 시간과 비용을 절감합니다.

### 1.2 대회 진행 과정
1. **참가자 로컬 환경**: 데이터 학습 $\rightarrow$ 모델 가중치 생성(예: `model.pt`) $\rightarrow$ 추론 코드 작성 및 테스트
2. **평가 서버 환경**: 참가자 제출 코드 실행 $\rightarrow$ 실제 평가 데이터 예측 $\rightarrow$ 최종 점수 산출

### 1.3 평가 데이터 구성
* **샘플 평가 데이터 (참가자 제공)**: 실제 평가 데이터와 동일한 폴더 구조 및 형식을 가진 소량의 더미 데이터로, 로컬 테스트용입니다.
* **실제 평가 데이터 (서버 자동 적용)**: 실제 평가 시 사용되는 전체 데이터셋입니다.

---

## 2. 제출 파일 구성

제출은 ZIP 파일 형식(`your_submission.zip`)으로 이루어지며, 아래의 **필수 디렉터리 구조**를 엄격히 준수해야 합니다 (파일명 공백 및 한글 제외).

```text
your_submission.zip
├── model/                 # 학습된 모델 가중치 저장 디렉터리 (파일명 자유)
│   └── model.pt
├── script.py              # 추론 실행 코드 (반드시 이 파일명 사용)
└── requirements.txt       # 패키지(라이브러리) 의존성 명시
```

* **`model/`**: 로컬에서 훈련한 모델 가중치 파일(예: `model.pt`, `tokenizer.json` 등)을 자유롭게 저장합니다.
* **`script.py`**: 서버에서 자동 실행되는 추론 전용 코드입니다. **학습 과정을 포함해선 안 되며**, `데이터 로드 → 모델 로드 → 예측 → 결과 저장` 순서로 작성합니다.
* **`requirements.txt`**: 추가 패키지 설치용 (`pip install -r requirements.txt`). 서버에 이미 설치된 기본 패키지는 버전 충돌 방지를 위해 가급적 제외를 권장합니다.

---

## 3. 평가 서버 동작 과정

1. **환경 구성**: 서버에서 `data/`와 `output/` 폴더를 자동으로 추가하여 실행 환경을 세팅합니다.
2. **패키지 설치**: `requirements.txt` 설치 진행. (설치 실패 시 **설치 오류** 처리, 제출 횟수 미차감)
3. **추론 실행**: `python script.py` 실행. (실행 실패 또는 시간 초과 시 **제출 오류** 처리, 제출 횟수 차감)
4. **결과 확인**: `output/submission.csv` 생성 여부 및 내용 검증.

*(서버 하드웨어 사양 및 기본 설치 패키지는 각 대회 페이지 '평가' 탭에서 확인)*

---

## 4. 추론 코드 작성 가이드

### 권장 코드 구조 (`script.py`)
```python
import os
import pandas as pd

def load_model():
    # model/ 디렉터리에서 로컬 모델 가중치 로드
    model_path = os.path.join('model', 'your_model.pt')
    return model

def load_data():
    # data/ 디렉터리에서 평가 데이터 로드
    data_path = os.path.join('data', 'test.csv')
    return data

def predict(model, data):
    # 추론 수행
    predictions = model.predict(data)
    return predictions

def save_results(predictions):
    # output/submission.csv로 결과 저장 (필수)
    os.makedirs('output', exist_ok=True)
    submission = pd.DataFrame({'prediction': predictions})
    submission.to_csv('output/submission.csv', index=False)

if __name__ == "__main__":
    model = load_model()
    data = load_data()
    predictions = predict(model, data)
    save_results(predictions)
```

### ⚠️ 오프라인 환경 제약사항 (중요)
서버는 패키지 설치 이후 **완전한 오프라인 환경**으로 동작합니다.
* **불가능한 작업**: `model.from_pretrained()` 등을 통한 온라인 다운로드, 외부 API(OpenAI 등) 호출, 인터넷 연결.
* **올바른 접근**: 필요한 모든 파일(모델, 토크나이저 등)을 `model/`에 미리 담아 로컬 파일 경로로 접근해야 합니다.
  * ❌ 잘못된 예시: `AutoModel.from_pretrained("bert-base-uncased")`
  * ✅ 올바른 예시: `AutoModel.from_pretrained(os.path.join('model', 'bert-base-uncased'))`

---

## 5. 제출 오류 유형 및 해결방법

| 오류 유형 | 발생 원인 | 해결 방법 | 제출 횟수 |
| :--- | :--- | :--- | :--- |
| **설치 오류** | 파일 구조 불일치, 패키지 설치 실패 및 시간 초과 | 구조 및 버전 재확인, 불필요 패키지 제거 | 차감 안 됨 |
| **제출 오류** | `script.py` 실행 에러, `submission.csv` 미생성, 추론 시간 초과 | 로컬 철저 테스트, 예외 처리, 알고리즘 최적화 | **차감됨** |

---

## 6. 제출 전 최종 점검
* [ ] 최상위 경로에 `model/`, `script.py`, `requirements.txt`가 존재하는가?
* [ ] `script.py`가 로컬에서 정상 실행되며 `output/submission.csv`를 생성하는가?
* [ ] 모델 파라미터 로딩 등 코드 내 모든 경로가 **상대 경로**로 설정되었으며 **오프라인 동작**이 가능한가?
* [ ] `requirements.txt`에 꼭 필요한 패키지만 명시되었는가?

---

## 7. 성공적인 제출을 위한 팁
* **로컬 테스트 구축**: 샘플 데이터(더미 파일)를 활용하여 충분히 추론 코드를 테스트하세요.
* **오프라인 철저 대비**: 필요한 모델 파일을 빠짐없이 다운로드하여 ZIP에 포함시키세요.
* **예외 처리 & 최적화**: 다양한 오류에 대비한 예외 처리 코드를 넣고, 시간 제한에 맞춰 추론 속도를 최적화하세요.

---

## 8. 문의사항
* **이메일**: dacon@dacon.io
* **토크 탭**: 대회 페이지 내 토크(Talk) 탭

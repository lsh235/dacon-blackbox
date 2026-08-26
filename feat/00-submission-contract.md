# 제출 계약

## 필수 추론 함수

- `inference.py`는 `predict_stage1`, `predict_stage2`, `predict_stage3`를 모듈 최상위에 정의한다.
- 각 함수는 해당 Stage의 정확한 컬럼 순서를 가진 `pandas.DataFrame`을 반환한다.
- 제출 코드는 네트워크, 사용자 캐시, 절대 경로에 의존하지 않는다.

## Stage 1 출력

- `ID`는 비어 있지 않고 영상마다 유일하다.
- `answer`는 `ORIGINAL` 또는 `RERECORDED`다.

## Stage 2 출력

- `ID`는 비어 있지 않고 영상마다 유일하다.
- `collision_frame`과 `entry_frame`은 입력 이미지 파일명의 원본 프레임 번호다.
- `evasion_space`와 `entry_side`는 제출 계약에 허용된 값만 사용한다.

## Stage 3 출력

- `(ID, sample_index)` 조합은 유일하다.
- 각 영상의 `sample_index`는 0부터 시작해 누락 없이 증가한다.
- 가감속과 조향 레이블은 제출 계약에 허용된 값만 사용한다.

## 제출 ZIP

- ZIP은 필수 추론 파일, requirements와 Stage별 모델 파일을 포함한다.
- ZIP 경로에는 절대 경로와 상위 디렉터리 이동이 없다.
- ZIP 크기와 압축 해제 후 크기는 대회 제한을 넘지 않는다.

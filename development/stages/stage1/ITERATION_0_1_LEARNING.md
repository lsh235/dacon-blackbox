# Stage 1 Iteration 0~1 학습 노트

이 문서는 `ORIGINAL`과 `RERECORDED`를 구분하는 Stage 1을 개선하면서 적용한 가설, 구현 경계, 검증 방법을 정리한다. 목표는 **공개 예제 점수를 과도하게 맞추는 것**이 아니라, 실제 학습 데이터가 주어졌을 때 재현 가능하고 누수가 적은 실험을 실행하는 기반을 갖추는 것이다.

> 공개 예제는 `DOC/Data/detail/data/stage1`의 10개 영상(클래스별 5개)이며, 개발에서는 원본을 보존하기 위해 바이트 동일한 작업 복사본 `data/raw/public_example/stage1`을 사용한다. 이 데이터는 입출력·실행 검증용이므로, 여기서 얻은 점수를 일반화 성능으로 해석하지 않는다.

## 1. 시작점: 베이스라인의 한계

초기 베이스라인은 RGB 영상 클립을 MViTv2-S에 넣고 Cross Entropy로 학습했다. 빠르게 동작하는 기준점으로는 유용했지만, 다음 문제를 해결하지 못했다.

- 재녹화에서 두드러질 수 있는 모아레, 스크린 격자, 반복 노이즈 같은 **주파수 특성**을 입력에 명시하지 않았다.
- 영상의 모든 프레임을 먼저 메모리에 올린 뒤 일부를 고르는 방식은 긴 영상에서 메모리 사용량이 커질 수 있었다.
- 검증 분할과 Macro F1/혼동 행렬이 없어, 모델이 어느 클래스로 편향됐는지 또는 개선이 우연인지 확인할 수 없었다.

## 2. Iteration 0: RGB + FFT와 Focal Loss

### 2.1 가설: 모아레는 공간 도메인보다 주파수 도메인에서 더 분리될 수 있다

모니터를 다시 촬영하면 픽셀 격자와 카메라 센서의 샘플링 간섭으로 반복적인 고주파 패턴이 나타날 수 있다. 따라서 프레임의 RGB 정보만 쓰는 대신, RGB 각 채널의 2D 로그 진폭 스펙트럼을 추가 채널로 연결한다.

```text
raw RGB [0, 1]
  ├─ RGB 정규화 ──────────────────────────────────────┐
  └─ 채널별 spatial mean 제거 → FFT2 → fftshift       │
       → log(1 + |FFT|) → 채널/프레임별 표준화 ────────┤
                                                        ↓
                                      concat → 6-channel clip → MViTv2-S
```

FFT 특성은 `spatial_log_spectrum()`에서 계산한다. 평균 성분(DC)이 나머지 주파수 성분을 압도하지 않게 공간 평균을 먼저 제거하고, `log1p`로 큰 진폭을 압축한다. FFT 채널은 RGB 이미지 값이 아니므로 ImageNet RGB 평균·표준편차를 재사용하지 않고 별도로 안정화한다.

이 설계는 “모아레를 잡을 수 있다”는 검증 가능한 가설이지, FFT가 항상 성능을 올린다는 보장은 아니다. 공간 픽셀과 주파수 bin을 초반 합성곱에서 조기에 섞는 한계도 있으므로, 향후에는 성능 근거가 쌓였을 때 2-branch encoder를 비교 후보로 둔다.

### 2.2 입력/체크포인트 호환성

`rgb` 모드는 3채널, `rgb_fft` 모드는 6채널이다. 따라서 첫 3D projection convolution의 입력 채널 수도 같이 바뀌어야 한다. 학습 체크포인트에는 아래 메타데이터를 저장하고, 추론은 이를 읽어 같은 feature mode로 모델과 전처리를 재구성한다.

- `feature_mode`
- `input_channels`
- `frames`, `size`
- `loss` 설정
- sampling 설정

이 규칙 덕분에 새 6채널 체크포인트와 이전 3채널 체크포인트를 각각 알맞은 입력 경로로 불러올 수 있다. 학습과 추론의 crop, RGB 정규화, FFT 순서는 반드시 동일해야 한다.

### 2.3 Focal Loss: 어려운 오분류에 더 큰 가중치

Focal Loss는 정답 클래스의 예측 확률을 \(p_t\)라 할 때 다음과 같다.

\[
L = -(1 - p_t)^\gamma \log(p_t)
\]

`gamma=0`이면 정확히 일반 Cross Entropy와 같다. 기본 실험값 `gamma=2`는 이미 쉽게 맞히는 샘플의 손실을 줄이고 어려운 샘플에 상대적으로 집중한다. 단, 10개 공개 예제처럼 데이터가 작고 라벨 노이즈 가능성이 있으면 과도하게 불안정해질 수 있으므로, RGB+CE를 항상 대조군으로 둔다.

### 2.4 OOM 방지: 메모리 제한 균일 샘플링

`uniform_frame_indices()`는 전체 프레임 수에서 필요한 `N`개 인덱스만 균일하게 정한다. 디코더는 요청된 프레임만 순차적으로 읽고, 짧은 영상에서 필요한 중복 인덱스도 정확히 유지한다. 즉 긴 영상을 Python 리스트로 전부 쌓아 놓지 않는다.

이 변경은 Stage 1의 **프레임 디코딩 메모리**를 `O(전체 프레임 수)`가 아니라 `O(샘플 수 × frame size)`에 가깝게 제한한다. 모델 activation 메모리와 전체 제출의 시간 제한을 자동으로 해결한다는 뜻은 아니므로, 실제 제출 규모에서 별도 측정이 필요하다.

### 2.5 Iteration 0에서 확인한 것과 확인하지 못한 것

확인한 항목:

- FFT feature의 shape, 유한값, 결정성
- RGB/FFT 채널 수와 MViT 입력 채널의 일치
- Focal Loss의 `gamma=0 == Cross Entropy`, 극단 logits에서의 finite gradient
- 3채널 legacy 및 6채널 checkpoint loading
- 공개 영상 디코딩 결과와 순차 디코딩 기준의 일치

확인하지 못한 항목:

- FFT 또는 Focal Loss가 비공개 데이터 성능을 올리는지 여부
- 전체 영상 길이와 전체 Stage 1~3를 포함한 제출 시간/VRAM 상한

## 3. Iteration 1: 누수 방지 로컬 평가 파이프라인

### 3.1 왜 Group Fold인가

같은 블랙박스, 같은 장면, 같은 원본에서 파생된 영상이 train과 validation에 동시에 있으면 모델은 재녹화 특성이 아니라 배경·카메라 고유 특성을 외울 수 있다. 단순 랜덤 분할의 점수는 이 경우 실제보다 높게 보인다.

`make_stratified_group_folds()`는 다음 우선순위의 메타데이터로 그룹을 만든다.

1. 사용자가 지정한 `--group-column`
2. `source_content_id`, `scene_id`, `original_video_id`, `group_id`
3. 공개 예제에 한해서만 `path`의 파일 stem

실제 `train.csv`에는 원본/장면 식별자를 제공해야 한다. 없다면 안전한 그룹 분할을 추정할 수 없으므로 오류를 내는 것이 임의의 랜덤 분할보다 낫다. 공개 예제의 `original/000001.mp4`와 `rerecorded/000001.mp4`는 파일 stem이 같아 동일 그룹으로 묶는다.

### 3.2 평가지표

로컬 평가는 `ORIGINAL`, `RERECORDED` 각각의 F1을 평균한 Macro F1을 사용한다.

\[
\mathrm{MacroF1} = \frac{F1_{ORIGINAL} + F1_{RERECORDED}}{2}
\]

함께 Accuracy, 2×2 confusion matrix, 클래스별 precision/recall/F1/support를 저장한다. 클래스별 결과를 같이 봐야 한쪽 클래스로만 예측해도 Accuracy가 높아 보이는 문제를 피할 수 있다.

`DOC/Overview/Evaluation.yaml`은 현재 비어 있으므로, 이 Macro F1은 로컬 진단 및 사용자 요청을 위해 구현한 지표다. 공식 평가 규칙이 공개되면 해당 정의와 averaging 방식을 다시 대조해야 한다.

### 3.3 동일 fold에서 실행하는 A/B 실험

`scripts/evaluate/run_stage1_cv.py`는 fold를 한 번 만든 뒤 아래 두 실험에 똑같이 사용한다.

| 실험 | feature mode | loss |
| --- | --- | --- |
| A (대조군) | `rgb` | Cross Entropy (`focal_gamma=0`) |
| B (제안) | `rgb_fft` | Focal Loss (`focal_gamma=2`) |

실행 예시는 다음과 같다.

```bash
cd development
PYTHONPATH=src .venv/bin/python scripts/evaluate/run_stage1_cv.py \
  --data-dir data/raw/public_example/stage1 \
  --folds 2 --epochs 1 --seed 20260825 \
  --output-root artifacts/experiments/iteration1-public-cv
```

공개 예제에서 실제 실행한 결과는 아래와 같다.

| 실험 | OOF Macro F1 | Accuracy | OOF confusion matrix (true rows, pred columns: ORIGINAL/RERECORDED) | 경과 시간 |
| --- | ---: | ---: | --- | ---: |
| A: RGB + CE | 0.494949 | 0.500000 | `[[3, 2], [3, 2]]` | 5.949초 |
| B: RGB+FFT + Focal | 0.494949 | 0.500000 | `[[3, 2], [3, 2]]` | 5.676초 |

두 모드가 같다는 사실은 평가 모듈과 OOF 보고서가 정상 동작함을 보여 주지만, 제안 모드의 성능 향상을 입증하지는 않는다. 특히 공개 예제는 작고 실제 재녹화 데이터 분포를 대표하지 않는다고 안내되어 있으므로 모델 선택 근거로 사용하지 않는다. 상세 실행 기록은 [`../../reports/experiments/S1-iteration1-public-cv.md`](../../reports/experiments/S1-iteration1-public-cv.md)에 있다.

## 4. 실제 학습 데이터가 왔을 때의 실험 순서

1. CSV에서 원본 영상/장면 단위의 group column을 확인하고 `--group-column`으로 고정한다.
2. split seed, fold 수, epoch, 영상 sampling 수를 실험 메타데이터에 기록한다.
3. A(RGB+CE)를 먼저 실행해 OOF Macro F1과 클래스별 confusion matrix를 기준선으로 남긴다.
4. B에서는 FFT와 Focal Loss를 동시에 바꿨다는 교란 요인을 인지한다. 원인 분리가 필요하면 `RGB+Focal`, `RGB+FFT+CE`를 추가해 한 요인씩 비교한다.
5. fold별 편차와 오분류 영상을 점검한 뒤에만 다음 augmentation/architecture 변경을 결정한다.
6. 검증 OOF를 사용해 threshold를 조정했다면, 그 threshold와 선택 절차도 기록한다. 비공개 테스트 라벨이나 제출 결과로 반복 튜닝하지 않는다.

## 5. 코드 탐색 지도

- 데이터, sampling, RGB/FFT feature: [`../../src/blackbox/stages/stage1/dataset.py`](../../src/blackbox/stages/stage1/dataset.py)
- Focal Loss: [`../../src/blackbox/stages/stage1/losses.py`](../../src/blackbox/stages/stage1/losses.py)
- 모델, train/inference checkpoint 호환: [`../../src/blackbox/stages/stage1/baseline.py`](../../src/blackbox/stages/stage1/baseline.py)
- group fold: [`../../src/blackbox/stages/stage1/splits.py`](../../src/blackbox/stages/stage1/splits.py)
- Metrics/JSON/Markdown report: [`../../src/blackbox/evaluation/stage1.py`](../../src/blackbox/evaluation/stage1.py)
- A/B CV runner: [`../../scripts/evaluate/run_stage1_cv.py`](../../scripts/evaluate/run_stage1_cv.py)
- 회귀 테스트: [`../../tests/unit/test_stage1_iteration0.py`](../../tests/unit/test_stage1_iteration0.py), [`../../tests/unit/test_stage1_evaluation.py`](../../tests/unit/test_stage1_evaluation.py)

## 6. 핵심 원칙

- 공개 예제는 제출 형식과 파이프라인 검증에 사용하고, 일반화 성능 주장에는 사용하지 않는다.
- 그룹은 가능한 한 원본 콘텐츠/장면 기준으로 정하고, train과 validation 사이에서 절대 겹치지 않게 한다.
- 입력 feature를 바꾸면 모델의 입력 채널, checkpoint metadata, 추론 전처리를 함께 바꾼다.
- 실험 결과에는 점수만 적지 말고 seed, fold, 데이터 그룹 기준, confusion matrix, 실행 설정을 함께 남긴다.

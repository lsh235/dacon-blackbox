# Stage 3 Ego-motion — Iteration 8 Handover Plan

이 문서는 다음 개발 세션의 시작점이다. 다음 세션은 먼저 이 문서를 끝까지 읽고, Stage 3의 자차 거동 인식 성능 개선만 진행한다. Stage 1과 Stage 2는 현재 상태를 안정화 기준선으로 동결한다. 특히 Stage 2 소스, 입출력 계약, 체크포인트 형식과 추론 결과를 임의로 변경하지 않는다.

`DOC/`는 대회 원본 자료를 보관하는 읽기 전용 디렉터리다. 코드, 설정, 캐시, 체크포인트와 실험 결과는 계속 `DOC/` 밖에 둔다.

## 1. Current Status

기준 커밋은 `e77aa15` (`feat(inference): add AMP smoothing and fold ensembles`)이다. 이 커밋에서 `make -C development check`는 66개 단위 테스트와 제출 계약 스모크를 통과했다.

### Stage 1 — 안정화 및 동결

- RGB + 2D FFT 입력, Focal Loss, clip-consistent augmentation과 temporal TTA가 구현돼 있다.
- FFT는 `preprocess_data.py`에서 오프라인으로 생성하고 학습 로더는 전처리 파일만 읽는다.
- YAML의 `training.use_amp`로 AMP를 제어하고, 여러 체크포인트의 확률을 평균하는 순차 soft-voting 추론을 지원한다.
- 다음 세션의 Stage 3 작업을 위해 Stage 1 코드를 수정하지 않는다.

### Stage 2 — 안정화 및 코드 동결

- RGB + Farneback Optical Flow Two-Stream, BiLSTM, Gaussian Soft Target, overlap aggregation과 원본 프레임 역매핑이 구현돼 있다.
- RGB/Flow window는 `data/processed/`에 오프라인 전처리되며 학습 중 Farneback를 다시 계산하지 않는다.
- AMP와 체크포인트 목록 기반 순차 soft-voting을 지원한다.
- `predict_stage2(data_dir, model_dir)`의 공식 출력 계약은 유지돼 있다.
- **Iteration 8에서는 Stage 2 모델, 데이터셋, target mapping 및 추론 로직을 변경하지 않는다.** 공유 모듈 변경이 불가피하면 Stage 2 기본 파라미터가 기존 동작과 정확히 같고 기존 가중치를 strict load할 수 있음을 회귀 테스트로 증명해야 한다.

### Stage 3 — 현재 구현

- `dataset_stage3.py`는 영상별 FPS에서 `frames_per_sample = round(FPS / 10)`을 계산하고, 원본 RGB/Flow 프레임을 0.1초 구간별로 평균한다.
- 이 시간축 평균은 Flow의 H×W 공간 배열을 유지한다. 공간 정보가 데이터셋에서 즉시 스칼라로 축소되는 것은 아니다.
- `Stage3TwoStreamBiLSTM`은 Stage 2와 공유하는 `TwoStreamBiLSTMEncoder`를 상속한다. RGB와 Flow를 각각 ResNet18에 통과시키고 결합한 뒤 BiLSTM과 가감속/조향 head로 각 타임스텝의 logits를 출력한다.
- Stage 3 추론에는 10 Hz 투영 이후 확률 기반 moving-average smoothing과 fold soft-voting 구조가 준비돼 있다.
- 현재 통합 제출 경로의 안정 기준은 baseline 체크포인트다. Stage 3 Two-Stream은 아직 실데이터 성능과 제출 시간·VRAM을 검증하지 않은 연구 경로이므로 성능 모델로 간주하지 않는다.

## 2. Problem Definition

현재 공유 Flow encoder는 전체 화면의 dense flow를 ResNet18에 입력한다. ResNet convolution 구간에서는 2차원 feature map이 유지되지만, ResNet18의 마지막 기본 `AdaptiveAvgPool2d((1, 1))`와 flatten에서 프레임 전체가 하나의 벡터로 축약된다.

이 구조에는 다음 한계가 있다.

- 하늘의 구름, 나뭇가지, 전광판과 같은 상단 영역의 움직임이 도로 영역과 같은 비중으로 Flow feature에 들어간다.
- 맞은편 차량, 보행자와 주변 차량의 독립 운동이 자차의 병진·회전 운동과 섞인다.
- 좌·우 차선의 서로 다른 수평 Flow 패턴과 가까운 도로/먼 도로의 수직 Flow 차이가 1×1 global average pooling에서 소실된다.
- 결과적으로 모델이 자차의 가감속과 조향에 대응하는 ego-motion 대신 장면 구성이나 우연한 객체 움직임을 학습할 위험이 크다.

Iteration 8의 핵심 질문은 다음과 같다.

> 전체 화면의 움직임 중 도로 지면에서 발생하는 자차 유도 Flow를 분리하고, 그 공간적 방향성을 BiLSTM 입력까지 보존하면 0.1초 단위 가감속·조향 분류가 개선되는가?

## 3. Next Objectives — Iteration 8

### 3.1 ROI 기반 Flow 추출

초기 ROI는 화면 하단 1/2을 사용한다. 도로와 차선이 주로 존재하는 영역만 남겨 상단 환경 노이즈를 차단한다.

권장 구현 순서:

1. Stage 3 전용 설정에 `flow_roi_top_ratio: 0.5`와 `flow_roi_mode: mask`를 추가한다.
2. 첫 실험은 하단 ROI 밖 Flow를 0으로 만드는 masking을 사용한다. 입력 크기와 캐시 형식이 유지되므로 기존 오프라인 Flow를 재생성하지 않고 비교할 수 있다.
3. `crop_resize`는 별도 ablation으로 둔다. Flow를 resize하면 벡터 단위도 변하므로 `dx`에는 출력/입력 폭 비율, `dy`에는 출력/입력 높이 비율을 적용해야 한다.
4. ROI 설정은 YAML과 Stage 3 checkpoint metadata에 모두 저장한다.

ROI는 Stage 3 경로에서만 활성화한다. 공유 encoder의 기본값은 전체 화면이어야 하며 Stage 2는 기존 입력을 그대로 받아야 한다.

### 3.2 Spatial Grid Pooling

Flow branch의 1×1 GAP를 3×3 grid pooling으로 교체하는 실험을 우선한다. 4×4는 두 번째 후보로 둔다.

권장 구조:

```text
Full-frame cached Flow [B,T,2,H,W]
  -> bottom-half ROI mask/crop
  -> Flow ResNet convolution trunk [B*T,512,h,w]
  -> AdaptiveAvgPool2d((3,3))
  -> flatten by fixed grid order [B,T,512*9]
  -> projection + normalization [B,T,D_flow]
                                         ┐
RGB ResNet + legacy 1x1 GAP [B,T,D_rgb] --+-> concatenate -> BiLSTM
Physical Flow vector [B,T,D_physics] -----┘                  ├-> accel head
                                                             └-> steer head
```

`512 × 3 × 3`을 BiLSTM에 그대로 연결하면 파라미터와 VRAM이 크게 증가한다. grid flatten 뒤 `Linear` 또는 작은 MLP로 `D_flow` 차원에 투영하고 `LayerNorm`을 적용하는 구성을 우선한다. projection은 grid cell 순서가 고정된 flatten 벡터를 받으므로 좌·우·상·하 위치 정보를 학습할 수 있다.

공유 모듈에는 다음과 같이 기본값이 기존 동작을 보존하는 파라미터를 둔다.

```python
TwoStreamBiLSTMEncoder(
    flow_grid_size=1,       # Stage 2와 legacy checkpoint의 기본값
    flow_roi_top_ratio=0.0, # 전체 화면
    use_physics_vector=False,
)
```

Stage 3 Iteration 8 설정에서만 `flow_grid_size=3`, `flow_roi_top_ratio=0.5`를 명시한다. 단순한 전역 `use_grid_pooling=True` 플래그만 두기보다 grid 크기와 ROI를 checkpoint에 재현 가능하게 기록한다.

### 3.3 명시적 물리 벡터 결합 — Optional Ablation

하단 ROI의 정규화된 Flow에서 각 0.1초 타임스텝마다 다음 값을 계산한다.

- `mean_dx`, `mean_dy`
- `std_dx`, `std_dy`

초기 물리 벡터는 4차원으로 제한한다. 필요할 때만 평균 magnitude나 좌/우 절반의 `mean_dx` 차이를 별도 ablation으로 추가한다. 한 실험에서 여러 통계량을 동시에 늘리지 않는다.

물리 벡터는 작은 MLP와 normalization을 거친 뒤 RGB/Grid Flow feature와 함께 BiLSTM 입력에 concatenate한다. padding 타임스텝은 valid mask 밖에서 0으로 유지한다. 통계는 반드시 ROI 안의 valid pixel만 사용하고 Flow clip/normalization 규칙과 일치해야 한다.

## 4. Shared Module Safety and Stage 2 Freeze

`development/src/blackbox/stages/two_stream.py`는 Stage 2와 Stage 3가 공유한다. Grid Pooling이나 ROI를 무조건 활성화하면 Stage 2 feature 차원, LSTM 입력 크기, state-dict shape와 기존 체크포인트 로드가 깨진다.

필수 안전 규칙:

1. 모든 신규 옵션의 기본값은 현재 Stage 2와 동일한 `1×1 GAP + full-frame Flow + no physics vector`다.
2. Stage 3가 명시적으로 옵션을 전달할 때만 ROI/Grid/physics 분기가 활성화돼야 한다.
3. Stage 2의 생성자 기본 인자와 `predict_stage2(data_dir, model_dir)` 시그니처를 변경하지 않는다.
4. Stage 2 default model의 parameter shape와 state-dict key를 가능한 한 유지한다.
5. 신규 Stage 3 checkpoint에는 architecture version, ROI, grid size, projection dimension과 physics feature 목록을 저장한다.
6. legacy Stage 3/Stage 2 checkpoint에 신규 구조를 암묵적으로 load하지 않는다. 구조 metadata가 다르면 명시적 오류를 내거나 legacy default 경로로만 load한다.

Stage 2 동결 회귀 테스트는 최소한 다음을 포함한다.

- `Stage2TwoStreamBiLSTM()` 기본 생성자의 기존 입력/출력 shape 유지
- default model state-dict 저장 후 strict reload 성공
- Stage 2 Gaussian target, padding mask와 original-frame reverse mapping 기존 테스트 통과
- 전체 `make -C development check` 통과

## 5. Implementation Plan

### Step 1 — Stage 3 전용 ROI helper와 설정

- `dataset_stage3.py` 또는 Stage 3 전용 feature helper에 bottom ROI masking을 추가한다.
- cached full-frame Flow를 입력으로 받아 shape와 dtype을 바꾸지 않는 경로부터 구현한다.
- synthetic Flow로 ROI 밖 값이 최종 ROI tensor와 물리 통계에 영향을 주지 않는지 테스트한다.
- YAML의 Stage 3 section과 checkpoint metadata에 ROI 설정을 연결한다.

### Step 2 — 파라미터화된 Grid Flow encoder

- `two_stream.py`의 기본 1×1 경로를 그대로 유지하면서 선택적인 `flow_grid_size`를 추가한다.
- ResNet18 convolution trunk의 출력을 grid pool한 뒤 projection한다.
- 1×1 default의 shape와 strict state-dict load 회귀 테스트를 먼저 통과시킨다.
- 3×3 synthetic feature map에서 좌측/우측 활성 패턴이 서로 다른 embedding 입력을 만드는지 테스트한다.

### Step 3 — Stage 3 모델 연결

- `Stage3TwoStreamBiLSTM` 생성자에서 ROI/Grid 옵션을 명시적으로 전달한다.
- 가감속·조향 multi-head와 valid mask 계약을 유지한다.
- `use_physics_vector=False`를 기본으로 구현한 뒤 ROI+Grid 기준선이 안정화된 경우에만 물리 벡터를 추가한다.

### Step 4 — 학습·추론·설정 연결

- `train_stage3.py`와 YAML에 신규 옵션을 연결한다.
- AMP, early stopping, JSONL logging과 offline preprocessing 경로를 유지한다.
- Stage 3 Two-Stream checkpoint를 통합 제출에 연결하기 전 strict load, 10 Hz reverse mapping, smoothing, ensemble, VRAM과 실행 시간을 검증한다.

### Step 5 — Controlled Ablation

실제 정답 데이터가 제공되면 동일한 group split, seed, window와 학습 조건에서 한 번에 하나의 핵심 변경만 비교한다.

1. A0: 현재 full-frame + 1×1 GAP
2. A1: bottom-half ROI mask + 1×1 GAP
3. A2: bottom-half ROI mask + 3×3 Grid Pooling
4. A3: A2 + 4D physics vector
5. 선택: A2의 3×3과 4×4 비교

각 실험은 accel Macro F1, steer Macro F1, 두 metric의 평균, 클래스별 F1과 confusion matrix를 기록한다. 또한 예측 전환 횟수, 1-step jitter 비율과 smoothing 전후 결과를 별도로 남긴다. 공개 예제나 복제 체크포인트 스모크는 실행 검증일 뿐 일반화 성능 근거로 사용하지 않는다.

## 6. Tests and Definition of Done

Iteration 8 완료 조건:

- ROI mask가 상단 Flow 노이즈를 제거하고 H×W 계약을 유지한다.
- Grid Pooling이 3×3 또는 4×4 cell 순서를 보존하고 projection 차원이 고정된다.
- Optional physics vector가 synthetic constant/variable Flow에서 기대한 평균과 표준편차를 반환한다.
- Stage 3 forward가 `[B,T,4]` accel logits, `[B,T,3]` steer logits와 valid mask를 유지한다.
- Stage 2 default 경로와 legacy-compatible state-dict 회귀 테스트가 통과한다.
- `make -C development check`가 통과한다.
- 공개 영상 1개로 AMP inference/training smoke가 OOM 없이 완료된다.
- 실제 라벨이 있을 때만 controlled ablation metric을 보고한다.

## 7. Known Issues and Constraints

### FPS / 10 Hz mapping conflict

공개 Stage 3 MP4는 OpenCV에서 약 478–480 FPS로 읽히지만 라벨의 frame/time 관계는 약 20 FPS를 가리킨다. `verify_video_metadata.py`는 이 충돌을 탐지하고 20 FPS를 추천하지만, 공식 평가 규칙이 확정된 것은 아니다. 다음 세션은 신규 공식 자료를 먼저 확인하고 임의로 20 FPS를 영구 고정하지 않는다.

### Offline feature compatibility

현재 Stage 3 cache는 full-frame RGB와 Flow를 저장한다. 첫 ROI mask 실험은 캐시를 재사용할 수 있다. crop/resize 또는 전처리 단계에서 ROI를 영구 적용한다면 cache schema/version과 manifest에 ROI 및 resize-vector scaling 설정을 포함하고 기존 cache를 잘못 재사용하지 않도록 한다.

### Memory and checkpoint compatibility

- Grid Pooling은 Flow embedding 크기를 증가시키므로 projection dimension과 `frame_batch_size`를 함께 관리한다.
- AMP는 YAML로 계속 제어하고 NaN/Inf를 모니터링한다.
- Grid/physics 모델은 기존 Stage 3 checkpoint와 shape가 다르다. 구조 버전 검증 없이 `strict=False`로 조용히 일부 가중치만 load하지 않는다.
- 여러 fold는 한 모델씩 순차 로드하고 확률을 CPU에 누적한 뒤 VRAM을 해제하는 Iteration 7 정책을 유지한다.

## 8. First Commands for the Next Session

```bash
cd /home/sra235/2026/dacon/blackbox
git status --short --branch
sed -n '1,320p' STAGE3_EGO_MOTION_PLAN.md
sed -n '1,260p' development/src/blackbox/stages/two_stream.py
sed -n '1,260p' development/src/blackbox/stages/stage3/model_stage3.py
sed -n '220,360p' development/src/blackbox/stages/stage3/dataset_stage3.py
make -C development check
```

그다음 `DOC/Overview/`, `DOC/Data/`, `DOC/Code/`를 읽기 전용으로 확인해 Stage 3 라벨, 평가 방식, FPS/10 Hz 규칙에 새 공식 정보가 추가됐는지 대조한다. 충돌이 계속되면 현재 방어 로직을 유지하고, 확인되지 않은 해석을 공식 규칙으로 단정하지 않는다.

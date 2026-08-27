# Stage 2 Iteration 3 Handover Plan

다음 세션은 이 문서와 `development/stages/stage2/`의 Iteration 2 문서를 먼저 읽고 시작한다. `DOC/`는 대회 원본 자료이므로 계속 읽기 전용으로 유지하며, 코드·캐시·체크포인트·실험 결과는 모두 `development/` 밖으로 나가지 않게 한다.

## 1. Current Status — Iteration 2 완료 상태

### Stage 1: 안정화 단계

- RGB+2D FFT 6채널 입력과 Focal Loss를 구현했다. checkpoint에는 feature mode, input channel, loss, sampling metadata를 저장해 RGB 3채널 legacy checkpoint와의 호환 경로도 유지한다.
- 메모리 제한 uniform frame sampling을 사용한다. 전체 영상을 프레임 리스트로 적재하지 않는다.
- 원본 콘텐츠/장면 metadata 기반 Group K-Fold 및 OOF Macro F1, accuracy, confusion matrix, precision/recall/F1 보고서를 구현했다. 공개 예제에서의 점수는 구조 검증일 뿐 일반화 성능 근거가 아니다.
- train 전용으로 clip-consistent ColorJitter와 약한 RandomAffine을 적용한다. blur와 강한 random crop/resize는 모아레 단서 훼손을 피하기 위해 넣지 않았다.
- 추론은 초반·중반·후반 3개 uniform slot의 `P(RERECORDED)`를 평균하는 temporal TTA를 사용한다. TTA slot 수는 checkpoint metadata에 저장된다.
- 관련 문서: `development/stages/stage1/ITERATION_0_1_LEARNING.md`, `development/stages/stage1/ITERATION_2_ROBUSTNESS.md`

### Stage 2: 메모리 안전 스켈레톤 완료

- `development/src/blackbox/stages/stage2/dataset_stage2.py`
  - `Stage2SlidingWindowDataset`이 전체 영상을 로드하지 않고 fixed-size window만 지연 decode한다.
  - `frame_numbers`, `valid_length`를 반환해 local prediction을 원본 frame 번호로 복원할 수 있다.
  - 공개 label의 `-1`은 class 0으로 바꾸지 않고 `IGNORE_INDEX`로 유지한다.
- `model_stage2.py`
  - `Stage2CnnBiLSTM`이 RGB frame chunk를 작은 CNN batch로 encoder에 통과시킨 뒤 BiLSTM으로 time logits와 scene logits를 만든다.
- `train_stage2.py`
  - chunk 내부에 존재하는 target만 loss에 넣는 실험용 trainer다. 아직 제출 trainer나 `predict_stage2()`에 연결하지 않았다.
- 공식 video-level 출력은 `ID`, `collision_frame`, `entry_frame`, `evasion_space`, `entry_side`다. `collision_frame`/`entry_frame`은 **원본 frame 번호**여야 한다.
- Iteration 2 검증: `make -C development check` 통과(46 tests 및 submission contract smoke). 공개 Stage 2 영상에서 실제 16-frame chunk decode → CNN+BiLSTM forward → original frame mapping도 통과했다.

### 현재 한계

- 공개 Stage 2 예제는 `t_collision` 외 `t_entry`, `evasion_space`, `entry_side`가 `-1`이다. 세 과업의 label 정의와 충분한 실제 학습 데이터 없이 성능/제출 모델을 주장하면 안 된다.
- 스켈레톤은 window-local logits만 만든다. overlapping window의 score를 원본 frame axis에서 합치는 global aggregation은 아직 구현하지 않았다.
- Stage 2 baseline/스켈레톤은 제출 모델이 아니다. 전체 Stage 1~3 실 제출 시간·메모리와 package run은 최종 checkpoint가 준비된 뒤 별도 검증해야 한다.

## 2. Next Objective — Iteration 3

딥러닝 optical-flow model이나 외부 weight를 추가하지 않고, **OpenCV 내장 `cv2.calcOpticalFlowFarneback`**로 window 내 motion feature를 만든다. 이는 평가 환경의 인터넷 차단과 설치 시간 제약을 유지하면서 RGB-only 모델이 놓치기 쉬운 차선 진입·충돌 직전의 상대 운동 변화를 보조하는 목적이다.

Iteration 3은 RGB-only baseline과 optical-flow 추가 모델을 같은 group split, seed, window 설정에서 비교한다. 공개 예제만으로 승자를 고르지 않는다.

## 3. Architecture to Implement

### 3.1 Two-Stream input/output 계약

```text
Stage2SlidingWindowDataset
  ├─ RGB:  [T, 3, H, W]                 ─→ spatial CNN ─┐
  ├─ Flow: [T, 2, H, W] (t=0 is zero)   ─→ temporal CNN ─┼→ concatenate
  ├─ valid_length / valid_mask                                ↓
  └─ frame_numbers (original indices)                     BiLSTM
                                                           ├─ collision logits [T]
                                                           ├─ entry logits [T]
                                                           ├─ evasion logits [2]
                                                           └─ entry-side logits [2]
```

1. Dataset은 기존 window decode 후, 동일한 crop/resize 결과를 grayscale으로 바꿔 인접 frame `(t-1, t)`의 Farneback flow를 계산한다.
2. `flow[0]`은 zero vector로 둔다. 따라서 RGB와 Flow의 time length `T`가 일치한다.
3. flow tensor는 `(dx, dy)` 순서의 `[T, 2, H, W]` float32이며, valid mask 밖 padding frame의 flow/logit은 손실과 argmax에서 제외한다.
4. spatial CNN과 temporal CNN은 각각 frame batch chunking을 지원해야 한다. 두 feature를 같은 local time index에서 concatenate해 BiLSTM으로 보낸다.
5. scene head는 valid local hidden state만 pooling한다.

### 3.2 Farneback 고정 설정과 재현성

초기 구현은 parameters를 하나의 frozen config/dataclass로 만들고 checkpoint/experiment report에 저장한다. 권장 출발값은 `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, `flags=0`이다. 이는 출발점일 뿐 성능 주장값이 아니다.

- resize/crop, grayscale 변환, flow 계산 순서는 train/inference에서 정확히 같아야 한다.
- `dx`, `dy`의 clip/normalization 방식은 RGB normalization과 분리한다. 예: clip 후 고정 scale로 나누고, 수치 범위를 unit test한다.
- blackbox ego-motion도 flow에 섞인다. RGB-only 대조군과의 fold별 비교, 전역 motion 보정 가능성, 실제 실패 영상을 함께 기록한다.

### 3.3 Target Mapping — local distribution과 reverse mapping

새 helper는 원본 정답 frame 번호를 window-local target으로 바꾼다.

1. target이 `-1`이거나 `[window_start, window_start + valid_length)` 밖이면 `IGNORE_INDEX`다. 없는 target을 negative로 라벨링하지 않는다.
2. target이 window 안에 있으면 local index `j = target_frame - window_start`를 계산한다.
3. 두 가지 mode를 구현해 같은 split에서 비교한다.
   - `binary_mask`: `j`만 1인 one-hot/CE target. 필요하면 반경 `r`의 tolerance mask도 config로 명시한다.
   - `gaussian`: `y_i = exp(-(i-j)^2 / (2σ^2))`를 valid positions에서만 normalize한 soft distribution. masked cross entropy/KL로 학습한다.
4. 추론에서는 invalid/padded position을 `-inf`로 masking하고 local peak `argmax`를 찾는다. 반드시 `frame_numbers[local_peak]`를 사용해 원본 `collision_frame` 또는 `entry_frame`으로 역매핑한다.
5. 여러 overlapping windows의 같은 original frame score는 original frame axis에서 `mean` 또는 `max`로 aggregate한 뒤 global argmax를 한다. aggregation policy도 config/checkpoint에 저장한다.

## 4. Implementation Order

1. `dataset_stage2.py`에 `FarnebackConfig`, grayscale/flow helper, Flow 포함 sample contract를 추가한다. 같은 영상/같은 window에서 RGB와 Flow의 shape 및 frame map이 맞는 unit test를 먼저 쓴다.
2. Flow가 없는 기존 RGB sample contract도 option으로 유지해 ablation이 가능하게 한다.
3. `model_stage2.py`에 temporal flow encoder와 two-stream fusion model을 추가한다. RGB-only skeleton을 삭제하거나 제출 API를 바꾸지 않는다.
4. `train_stage2.py`에 binary-mask/gaussian target builder, masked soft target loss, overlap aggregation helper를 추가한다.
5. synthetic target mapping tests, short public-video decode smoke, `make -C development check`, CPU/GPU window-memory/runtime measurement을 실행한다.
6. 실제 label이 준비된 경우에만 같은 Group CV split에서 RGB-only vs RGB+Flow를 비교하고, frame error/recall/scene-task metrics와 failure cases를 report한다.

## 5. Known Issues and Operational Considerations

### Farneback data-loading bottleneck

Farneback는 CPU 연산이므로 DataLoader worker에서 매 epoch 같은 window flow를 재계산하면 GPU가 기다릴 수 있다. Iteration 3에서 다음 둘을 분리해 측정한다.

- **on-the-fly mode:** 처음에는 구현 단순성과 correctness를 위해 사용한다. `num_workers`, decode time, flow time, GPU idle time을 기록한다.
- **cache/pre-extraction mode:** 병목이 확인되면 `DOC/` 밖의 `development/artifacts/features/stage2/`에 `.npz` 또는 `.pt` cache를 만든다. cache key에는 video content hash, original frame range, size/crop policy, Farneback config, OpenCV version을 포함해 stale feature 재사용을 막는다.

cache는 학습/검증 분할 이전에 label을 사용하지 않는 video-level feature만 담고, checkpoint/제출 ZIP에는 무단으로 대용량 cache를 포함하지 않는다. 생성·삭제 대상은 항상 `DOC/` 밖의 명시된 artifact path로 제한한다.

### Submission and validation boundary

- `predict_stage2(data_dir, model_dir)`의 official DataFrame contract는 유지한다. Two-stream checkpoint를 제출 경로에 연결하기 전에는 strict checkpoint loading, original-frame mapping, output schema, offline install, 전체 time/VRAM을 검증한다.
- `DOC/Overview/Evaluation.yaml`의 공식 metric 정보가 비어 있으므로, metric/threshold/tolerance는 문서 없이 임의로 대회 규칙이라고 선언하지 않는다.
- YOLO/object tracking은 이번 iteration의 필수 의존성이 아니다. 실제 data에서 flow의 부족이 확인되고, weight·license·offline package 조건이 정리된 뒤 별도 ablation으로 다룬다.

## 6. First Commands for the Next Session

```bash
cd /home/sra235/2026/dacon/blackbox
git status --short --branch
make -C development check
sed -n '1,260p' STAGE2_ITERATION3_PLAN.md
sed -n '1,260p' development/stages/stage2/ITERATION_2_SKELETON_AND_ARCHITECTURE.md
```

이후 수정 전에는 `DOC/Overview/`, `DOC/Data/`, `DOC/Code/`에서 Stage 2의 새 공식 공지·라벨 정의가 추가됐는지 읽기 전용으로 재확인한다.

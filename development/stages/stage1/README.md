# Stage 1 — 재녹화 여부 판별

입력 영상을 `ORIGINAL` 또는 `RERECORDED`로 분류한다.

- 먼저 읽을 문서: [`STUDY_AND_IMPROVEMENT_GUIDE.md`](./STUDY_AND_IMPROVEMENT_GUIDE.md)
- 개선 과정 학습 노트: [`ITERATION_0_1_LEARNING.md`](./ITERATION_0_1_LEARNING.md)
- Iteration 2 강건성 노트: [`ITERATION_2_ROBUSTNESS.md`](./ITERATION_2_ROBUSTNESS.md)
- 현재 구현: [`../../src/blackbox/stages/stage1/baseline.py`](../../src/blackbox/stages/stage1/baseline.py)
- 공통 전처리: [`../../src/blackbox/common/runtime.py`](../../src/blackbox/common/runtime.py)
- 작업 상태: [`../../TASKS.md`](../../TASKS.md)의 M3
- 공개 예제 한계: [`../../reports/experiments/M2-public-example-inventory.md`](../../reports/experiments/M2-public-example-inventory.md)

현재 구현은 RGB MViTv2-S, frame별 FFT 2-D CNN, luminance/row-profile Dilated Conv1d와 correlation-ConvGRU motion branch를 feature fusion하는 멀티스트림 구조다. Motion branch는 stride 16의 14×14 피처에서 모든 픽셀 쌍의 4-D correlation을 계산하고, 마지막 두 좌표축을 1/2/4/8 크기로 평균 풀링한다. 로컬 correlation lookup과 tied ConvGRU를 세 번 반복해 변위를 갱신하며, 각 step은 `1/(iteration+1)`로 감쇠하고 `tanh`로 제한한다. soft explainability mask와 target reconstruction도 함께 출력한다.

학습 손실은 clip Focal loss에 frame-level Focal loss, `0.05` 가중치와 `tau=4`를 쓰는 adjacent-frame truncated MSE, `0.05 * (mask-weighted Charbonnier reconstruction + 0.2 * BCE-to-one)` explainability loss를 결합한다. 보조 손실별 값은 checkpoint history와 JSONL에 따로 기록된다. 학습은 16/24/32프레임 중 하나를 샘플마다 고르고, MViT 입력만 시간축을 16으로 보간한다. FFT·flicker·motion 분기는 원래 길이를 그대로 받는다. 추론은 16프레임 초반·중반·후반 3구간을 사용하며, 학습 clip은 전처리 context 안에서 시작점을 jitter한다. 공개 예제에는 실제 기기 재촬영 데이터가 없으므로, 현재 결과를 일반화 성능의 증거로 사용하지 않는다.

기존 6채널 `.npy` 캐시 및 v2 멀티스트림 checkpoint는 새 v3 correlation-GRU 구조와 호환되지 않는다. 16프레임 raw-view cache도 가변 길이 학습에 부족하므로 아래 명령으로 32프레임 cache를 다시 만들고 모델 checkpoint를 재학습해야 한다.

```bash
./preprocess_data.py \
  --data-root development/data/raw/public_example \
  --stages stage1 \
  --stage1-frames 32 --stage1-slots 3 \
  --stage1-jitter-frames 4 --stage1-forensic-size 320
```

## Local CV

로컬 비교는 `scripts/evaluate/run_stage1_cv.py`로 수행한다. `source_content_id`, `scene_id`, `original_video_id`, `group_id` 중 하나가 있다면 자동 사용하며, 실제 데이터에서는 원본 콘텐츠 또는 씬 열을 명시하는 것을 권장한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate/run_stage1_cv.py \
  --data-dir data/raw/train/stage1 \
  --labels-csv data/raw/train/stage1/train.csv \
  --group-column source_content_id \
  --folds 5 \
  --epochs 30 \
  --output-root artifacts/experiments/s1-cv
```

각 fold는 최대 30 epoch, 최소 10 epoch, patience 7을 사용한다. 최초 3 epoch에는 MViT `1e-6 -> 1e-5`, 나머지 분기 `1e-6 -> 1e-4` 선형 warm-up을 적용하고, 이후 두 parameter group 모두 `eta_min=1e-6`인 `CosineAnnealingLR`로 감쇠한다. MViT는 기본적으로 random initialization이며, 로컬 가중치를 사용할 때만 `--pretrained-backbone-checkpoint /path/to/state_dict.pt`를 명시한다.

같은 fold에서 A(RGB+Cross Entropy)와 B(RGB+FFT+Focal)를 실행하고, 각 실험의 OOF Macro F1, confusion matrix, precision/recall/F1을 저장한다. 그룹 열이 없을 때의 `path` 파일명 fallback은 공개 예제 쌍에만 검증됐으므로, 실제 데이터에서 원본 콘텐츠 식별자로 쓸 수 없으면 실행을 중단하고 메타데이터를 추가해야 한다.

## Model evaluation and diagnostics

학습이 끝난 체크포인트는 다음 명령으로 손실 수렴, 클래스별 지표, 영상 길이별 편차, 분기 기여도와 시간 구간별 확률 변동을 함께 평가한다. `--trace-windows`는 프레임별 출력이 아니라 시간순 clip window 진단 개수다. 분류 지표는 이 값과 무관하게 체크포인트의 공식 3구간 추론 정책으로 계산한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate/evaluate_stage1_model.py \
  --data-dir data/raw/train/stage1 \
  --labels-csv data/raw/train/stage1/train.csv \
  --checkpoint artifacts/models/stage1/best.pt \
  --fold-metrics-json artifacts/experiments/s1-cv/exp_b_rgb_fft_focal/fold_metrics.json \
  --trace-windows 9 \
  --output-dir artifacts/evaluation/stage1
```

주요 산출물은 다음과 같다.

- `loss_curve.svg`, `training_history.csv`: Train/Validation loss 및 fixed-point 진단 근거
- `evaluation.json`, `evaluation.md`: Macro F1, 클래스별 precision/recall/F1, 길이 그룹별 성능, fold F1 표준편차
- `confusion_matrix.csv`: 행이 정답, 열이 예측인 혼동 행렬
- `predictions.csv`: 영상별 지속 시간, 길이 그룹, 최종 확률과 시간 일관성 지표
- `temporal_branch_diagnostics.csv`: 시간순 window 확률, RGB/공간/시간 분기 활성화 RMS, 첫 fusion layer 가중 활성화, 분기 제거 시 확률 변화

길이 그룹은 검증 영상 지속 시간의 1/3·2/3 분위수를 경계로 short/medium/long을 나누며, 같은 길이의 영상은 서로 다른 그룹으로 억지 분리하지 않는다. GroupKFold는 `session_id`, `capture_session_id`, `device_id`, `capture_device_id`, `camera_id` 또는 source/scene 식별자를 그룹으로 사용할 수 있으며, `split_audit.json`과 `fold_generalization.json`에 그룹 분리 및 fold별 F1 분산을 기록한다. 같은 길이의 공개 예제만으로는 길이 강건성을 판단할 수 없으므로 실제 검증 데이터 결과를 사용해야 한다.

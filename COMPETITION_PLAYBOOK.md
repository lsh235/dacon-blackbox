# DACON 블랙박스 대회 운영 Playbook

이 문서는 실제 데이터가 공개된 뒤 환경 구성부터 최종 `submit.zip` 생성까지 프로젝트를 명령어 단위로 운영하기 위한 마스터 매뉴얼이다. 모든 명령은 프로젝트 루트에서 실행한다.

```bash
cd /home/sra235/2026/dacon/blackbox
```

`DOC/`는 대회 원본 참고 자료이므로 읽기 전용으로 유지한다. 데이터, 캐시, 체크포인트, 로그와 제출물은 모두 `development/` 아래의 Git 제외 경로에 저장한다.

## 1. Pipeline Overview

```text
공식 학습/평가 데이터
  -> 데이터 구조 확인 및 영상 메타데이터 진단
  -> 오프라인 전처리
       Stage 1: uniform RGB clip + 2D FFT
       Stage 2/3: RGB window + Farneback dense flow cache
  -> 학습 및 모델 선택
       stable baseline / Stage별 Two-Stream 실험 / Stage 3 Optuna HPO
  -> Stage 1 -> Stage 2 -> Stage 3 순차 추론
       fold soft-voting -> 10 Hz 투영 -> moving average -> Viterbi constraint
  -> Stage별 계약 CSV 생성
  -> submission.csv scaffold + inference.py + models + requirements.txt
  -> submit.zip 계약 검증 및 SHA-256 기록
```

### Stage별 구성

| Stage | 입력과 목표 | 주요 모델 및 처리 |
| --- | --- | --- |
| Stage 1 | 영상의 원본/재녹화 이진 분류 | 고정 개수 uniform clip, RGB+2D FFT 6채널, MViTv2-S, Focal Loss, clip-consistent augmentation, 3-slot temporal TTA, fold soft-voting |
| Stage 2 | 진입·충돌 프레임, 진입 방향, 회피 공간 | 제출 baseline은 ResNet18+BiGRU다. 실험 경로는 RGB+Farneback Two-Stream ResNet18, BiLSTM, Gaussian temporal target, overlapping-window 원본 프레임 aggregation을 사용한다. |
| Stage 3 | 10 Hz 가감속 4종, 조향 3종 시퀀스 | 제출 baseline은 MViTv2-S multi-head다. Ego-motion 실험 경로는 하단 ROI Flow masking, Spatial Grid Pooling, 물리 Flow 벡터, RGB/Flow Two-Stream, BiLSTM과 Focal Loss를 사용한다. 추론 후 moving average와 물리 전이 그래프 기반 Viterbi 제약을 적용한다. |

### 운영 모드의 구분

- `development/configs/baseline.yaml`: `run_all.sh`가 지원하는 제출 호환 안정 경로다. Stage별 `best.pt`를 만들고 세 Stage CSV를 순차 생성한다.
- `development/configs/experiment_two_stream.yaml`: Stage 2/3 Two-Stream 및 Stage 3 HPO를 위한 실험 경로다. Stage별 학습 CLI로 실행한다.
- `run_all.sh`는 K-Fold를 생성하는 스크립트가 아니다. 한 개의 baseline 체크포인트를 Stage별로 학습하며, `inference.checkpoints`에 기존 fold 체크포인트가 지정된 경우에만 추론 단계에서 soft-voting한다.
- 현재 `build_submission.sh`는 `model_root`의 Stage별 단일 `best.pt`를 패키징한다. 로컬 fold ensemble CSV와 code-submission ZIP의 단일 체크포인트 실행을 혼동하지 않는다.

## 2. 대회 제약과 실행 전 체크

저장된 공식 자료 기준 제출 환경은 다음과 같다.

- 전체 추론 60분 이내
- 패키지 설치 10분 이내
- ZIP 10GB 이하, 압축 해제 후 32GB 이하
- 평가 서버 인터넷 비활성화
- NVIDIA L40S 44.7GiB, CPU 7 vCPU, RAM 60GB, shared memory 30GB

대회 시작 후 최신 공지와 평가 탭을 다시 확인하고, 변경된 제한이 있으면 최신 규칙을 우선한다.

환경을 만들고 전체 회귀 검사를 먼저 실행한다.

```bash
make -C development setup
make -C development check
```

`make check`는 패키지 버전/CUDA 확인, Python 컴파일, 단위 테스트와 제출 계약 스모크를 순서대로 수행한다.

## 3. 데이터 세팅과 설정 파일 준비

### 3.1 학습 데이터 복사

공식 압축을 프로젝트 밖에서 해제한 뒤 격리된 개발 경로로 복사한다. 복사 도구는 대상 경로가 이미 있으면 덮어쓰지 않고 중단한다.

```bash
PYTHONPATH=development/src development/.venv/bin/python \
  development/scripts/data/prepare_public_example.py \
  --source /absolute/path/to/official_training_data \
  --destination development/data/raw/competition \
  --decode
```

학습 루트는 최소한 다음 Stage 구조를 가져야 한다. Stage 1/2 영상의 실제 상대 경로는 각 `labels.csv`의 `path` 열을 따른다.

```text
development/data/raw/competition/
├── stage1/
│   ├── labels.csv
│   └── ... referenced videos ...
├── stage2/
│   ├── labels.csv
│   └── ... referenced videos ...
└── stage3/
    ├── labels.csv
    └── videos/*.mp4
```

평가 또는 로컬 추론 입력은 predictor 계약에 맞춰 다음 구조를 사용한다.

```text
development/data/interim/competition_eval/
├── stage1/videos/*
├── stage2/images/<ID>/*.{jpg,jpeg,png}
└── stage3/videos/*.mp4
```

### 3.2 대회 전용 YAML 생성

추적 중인 기준 설정은 보존하고 대회용 복사본을 만든다.

```bash
cp development/configs/baseline.yaml development/configs/competition_baseline.yaml
cp development/configs/experiment_two_stream.yaml development/configs/competition_two_stream.yaml
```

YAML 경로는 설정 파일 위치를 기준으로 해석된다. 두 파일의 경로를 실제 데이터에 맞게 수정한다.

```yaml
data:
  root: ../data/raw/competition
  inference_root: ../data/interim/competition_eval
  processed_root: ../data/processed

run:
  model_root: ../artifacts/competition/models
  output_root: ../artifacts/competition/submissions
```

`run.epochs`, `training.validation_fraction`, Stage별 batch/window 설정도 실제 데이터 규모와 VRAM 측정값에 맞춰 확정한다.

## 4. Step-by-Step Execution Guide

### Step 1. 영상 메타데이터 진단

특히 Stage 3은 컨테이너 FPS, 실제 디코딩 프레임 수와 라벨 시간축을 학습 전에 비교한다.

```bash
mkdir -p development/artifacts/reports

PYTHONPATH=development/src development/.venv/bin/python \
  development/scripts/verify_video_metadata.py \
  --data-dir development/data/raw/competition/stage3 \
  --threshold 0.10 \
  --output development/artifacts/reports/stage3_video_metadata.json
```

판정 기준:

- `flagged_count == 0`: 메타데이터와 디코딩 결과에 설정 임계치 이상의 충돌이 없다.
- `cap_prop_fps_vs_label_source_fps`: OpenCV FPS와 라벨에서 계산한 source FPS가 충돌한다.
- `cap_prop_frame_count_vs_decoded_count`: 컨테이너 프레임 수와 실제 decode 수가 충돌한다.
- `label_sample_hz_not_10`: 라벨 시간축이 10 Hz와 맞지 않는다.

경고가 발생해도 FPS를 임의로 고정하지 않는다. 현재 Stage 3 평가 경로는 공식 10 Hz 영상의 디코딩 프레임 하나를 `sample_index` 하나로 취급하고, 공개 sparse 학습 라벨만 `frame_index / sample_index` 비율로 매핑한다.

Stage 1/2에도 `videos/` 디렉터리가 있으면 같은 명령의 `--data-dir`만 바꿔 decode/프레임 수 진단을 수행할 수 있다.

### Step 2. 오프라인 전처리

Stage 1과 Stage 2/3은 캐시 형식이 다르며, Stage 2와 Stage 3의 window/stride도 다르다. 한 번에 묶지 말고 아래처럼 Stage별로 실행한다.

Stage 1 RGB+FFT feature:

```bash
./preprocess_data.py \
  --data-root development/data/raw/competition \
  --processed-root development/data/processed \
  --stages stage1 \
  --size 224 \
  --stage1-frames 16 \
  --stage1-slots 1 \
  --stage1-feature-mode rgb_fft
```

Stage 2 RGB+Farneback window:

```bash
./preprocess_data.py \
  --data-root development/data/raw/competition \
  --processed-root development/data/processed \
  --stages stage2 \
  --size 224 \
  --window-frames 64 \
  --stride 32
```

Stage 3 RGB+Farneback window:

```bash
./preprocess_data.py \
  --data-root development/data/raw/competition \
  --processed-root development/data/processed \
  --stages stage3 \
  --size 224 \
  --window-frames 96 \
  --stride 48
```

정상 종료 JSON의 `created`, `reused`, `videos/windows` 수를 기록한다. 파라미터가 바뀌었거나 원본 데이터가 교체된 경우에만 `--overwrite` 사용을 검토한다. `--max-windows-per-video`는 스모크용 제한이므로 최종 학습에서는 지정하지 않는다.

### Step 3. Stage 3 Optuna HPO

먼저 1~2 Trial 합성 스모크로 GPU/모델 경로를 확인할 수 있다.

```bash
./tune_hyperparams.py \
  --smoke \
  --trials 2 \
  --epochs 1 \
  --output-dir development/artifacts/hpo-stage3-smoke
```

실데이터 HPO는 Stage 3 video ID를 기준으로 train/validation group holdout을 만들고, 가감속 Macro F1과 조향 Macro F1의 평균을 최대화한다. 최소 두 개 이상의 video group이 필요하다.

```bash
mkdir -p development/artifacts/hpo-stage3-competition

./tune_hyperparams.py \
  --config development/configs/competition_two_stream.yaml \
  --data-dir development/data/raw/competition/stage3 \
  --processed-root development/data/processed \
  --output-dir development/artifacts/hpo-stage3-competition \
  --study-name stage3-ego-motion-competition \
  --storage sqlite:///development/artifacts/hpo-stage3-competition/study.db \
  --trials 50 \
  --epochs 3 \
  --use-physics-vector \
  --use-amp
```

탐색 대상은 다음 네 항목이다.

- `flow_roi_top_ratio`: 0.35, 0.50, 0.65
- `flow_grid_size`: 2, 3, 4
- `learning_rate`: `1e-5`~`1e-3`, log scale
- `focal_gamma`: 0, 1, 2, 3

주요 산출물:

- `best_stage3_hparams.yaml`: 최적 조합이 반영된 재사용 설정
- `trials.csv`: Trial별 상태, 파라미터와 metric
- `study_summary.json`: 최고 Trial과 고정 모델 설정
- `study.db`: `--storage`를 사용한 경우 재시작 가능한 Optuna study

합성 스모크 F1을 실제 성능으로 사용하지 않는다. 실제 데이터의 group holdout 또는 대회 리더보드 결과만 모델 선택 근거로 기록한다.

최적 YAML로 Stage 3 Two-Stream 후보를 학습하려면 Stage별 CLI를 사용한다.

```bash
PYTHONPATH=development/src development/.venv/bin/python \
  -m blackbox.stages.stage3.train_stage3 \
  --config development/artifacts/hpo-stage3-competition/best_stage3_hparams.yaml \
  --architecture two-stream \
  --epochs 20
```

이 명령의 결과는 `stage3_two_stream_experimental.pt`다. 현재 `run_all.sh`와 `build_submission.sh`가 요구하는 baseline `best.pt`와 형식이 다르므로, 구조가 다른 체크포인트를 이름만 바꿔 패키징하지 않는다.

### Step 4. 최종 학습과 앙상블 추론

#### 4.1 제출 호환 baseline 전체 실행

`competition_baseline.yaml`의 epoch와 경로를 확정한 뒤 실행한다.

```bash
./run_all.sh --config development/configs/competition_baseline.yaml
```

실행 순서:

1. Stage 1 baseline 학습 및 `stage1/best.pt` 저장
2. Stage 1 객체와 GPU cache 해제
3. Stage 2 baseline 학습 및 `stage2/best.pt`, ResNet18 backbone 저장
4. Stage 2 객체와 GPU cache 해제
5. Stage 3 baseline 학습 및 `stage3/best.pt` 저장
6. Stage 1→2→3 순차 추론, 계약 검증과 CSV 저장

#### 4.2 Stage 1 Group K-Fold 생성

Stage 1은 source/scene group을 fold 사이에서 분리하는 실행 스크립트가 있다. 실제 데이터에 신뢰 가능한 group 열이 없으면 임의 분할하지 않고 중단한다.

```bash
PYTHONPATH=development/src development/.venv/bin/python \
  development/scripts/evaluate/run_stage1_cv.py \
  --data-dir development/data/raw/competition/stage1 \
  --group-column source_content_id \
  --folds 5 \
  --epochs 20 \
  --output-root development/artifacts/cv/stage1-competition
```

선택한 RGB+FFT+Focal 모델은 다음 위치에 저장된다.

```text
development/artifacts/cv/stage1-competition/
└── exp_b_rgb_fft_focal/
    ├── fold_0/model/best.pt
    ├── fold_1/model/best.pt
    └── ...
```

현재 Stage 2/3에는 전체 K-Fold를 생성하는 통합 스크립트가 없다. 별도 절차로 만든 fold 체크포인트가 있을 때만 다음 앙상블 추론에 전달한다.

#### 4.3 기존 fold 체크포인트 soft-voting

fold는 한 번에 하나씩 로드되고 확률만 CPU에 누적된다. Stage 3은 fold 평균 후 10 Hz 투영, moving average와 Viterbi 전이 제약을 적용한다.

```bash
PYTHONPATH=development/src development/.venv/bin/python \
  development/scripts/submission/generate_submission.py \
  --data-root development/data/interim/competition_eval \
  --model-root development/artifacts/competition/models \
  --output-root development/artifacts/competition/ensemble-submissions \
  --stage1-checkpoints \
    development/artifacts/cv/stage1-competition/exp_b_rgb_fft_focal/fold_0/model/best.pt \
    development/artifacts/cv/stage1-competition/exp_b_rgb_fft_focal/fold_1/model/best.pt \
    development/artifacts/cv/stage1-competition/exp_b_rgb_fft_focal/fold_2/model/best.pt \
    development/artifacts/cv/stage1-competition/exp_b_rgb_fft_focal/fold_3/model/best.pt \
    development/artifacts/cv/stage1-competition/exp_b_rgb_fft_focal/fold_4/model/best.pt \
  --smoothing-window 3 \
  --use-transition-constraints \
  --transition-penalty -1000000000
```

Stage 2/3 fold가 준비돼 있다면 각각 `--stage2-checkpoints`, `--stage3-checkpoints`에 추가한다. Stage 2의 각 `best.pt` 옆에는 해당 fold의 `resnet18-f37072fd.pth`가 있어야 한다.

같은 목록을 `competition_baseline.yaml`의 `inference.checkpoints.stage1~3`에 기록하면 `run_all.sh`의 마지막 추론에서도 soft-voting을 사용할 수 있다. 목록 경로는 YAML 기준 상대 경로다.

### Step 5. 제출물 생성

단일 제출 체크포인트와 Stage별 CSV를 기준으로 최종 빌더를 실행한다.

```bash
./build_submission.sh \
  --input-dir development/artifacts/competition/submissions \
  --model-root development/artifacts/competition/models \
  --output-dir development/artifacts/final-submission
```

빌더는 다음을 자동 수행한다.

1. `stage1_submission.csv`, `stage2_submission.csv`, `stage3_submission.csv` 계약 검증
2. 서로 다른 Stage schema를 `stage` 열이 있는 long-form `submission.csv` 뼈대로 병합
3. `inference.py`, `blackbox` 소스, `requirements.txt`와 모델 패키징
4. 필수 파일, top-level predictor 함수, ZIP/압축 해제 크기 검증
5. SHA-256과 파일 수를 `build_manifest.json`에 기록

결과:

```text
development/artifacts/final-submission/
├── submission.csv
├── submit.zip
└── build_manifest.json
```

`submission.csv`는 서로 다른 세 Stage 결과를 확인하기 위한 로컬 long-form scaffold다. 실제 대회 제출 계약의 기준은 predictor 함수와 모델을 포함한 `submit.zip`이다.

필요하면 검증기를 다시 실행한다.

```bash
PYTHONPATH=development/src development/.venv/bin/python \
  development/scripts/submission/validate_submission.py \
  development/artifacts/final-submission/submit.zip

sha256sum development/artifacts/final-submission/submit.zip
```

## 5. Directory Structure and Artifacts

| 경로 | 내용 | Git 추적 |
| --- | --- | --- |
| `DOC/` | 대회 원본 문서와 예제 | 추적, 읽기 전용 |
| `development/configs/` | baseline/Two-Stream YAML과 대회용 설정 복사본 | 기준 설정은 추적, 로컬 대회 설정은 커밋 전 검토 |
| `development/data/raw/` | 복사한 학습 데이터 | 제외 |
| `development/data/interim/` | 로컬/대회 추론 입력 구조 | 제외 |
| `development/data/processed/stage1/features/` | RGB 또는 RGB+FFT `.npy` | 제외 |
| `development/data/processed/stage2/windows/` | Stage 2 RGB/Flow window와 manifest | 제외 |
| `development/data/processed/stage3/windows/` | Stage 3 RGB/Flow window와 manifest | 제외 |
| `development/artifacts/competition/models/` | Stage별 제출 baseline 체크포인트 | 제외 |
| `development/artifacts/cv/` | fold 모델, split과 OOF metric | 제외 |
| `development/artifacts/hpo-stage3-competition/` | Optuna YAML, CSV, JSON, SQLite study | 제외 |
| `development/artifacts/competition/submissions/` | Stage 1~3 CSV와 추론 manifest | 제외 |
| `development/artifacts/final-submission/` | 최종 scaffold, ZIP, build manifest | 제외 |
| `development/logs/stage1.jsonl` | Stage 1 epoch/loss/validation 기록 | 제외 |
| `development/logs/stage2.jsonl` | Stage 2 epoch/loss/validation 기록 | 제외 |
| `development/logs/stage3.jsonl` | Stage 3 epoch/loss/validation 기록 | 제외 |

전처리 cache는 원본 절대 경로와 geometry/Farneback 설정을 manifest에 기록한다. 데이터를 이동하거나 window/size 설정을 바꾼 경우 기존 cache가 선택되지 않는지 확인한다.

## 6. 장애 대응

### `missing offline ...; run preprocess_data.py`

학습 YAML과 전처리 명령의 `size`, `frames`, `window_frames`, `stride`, feature mode가 다르다. Stage별 전처리 명령을 동일 파라미터로 다시 실행한다.

### 메타데이터 진단이 `flagged`를 반환함

JSON의 `reasons`, label-derived 값과 실제 decode 수를 보존한다. 컨테이너 FPS 하나만 보고 Stage 3 stride를 바꾸지 않는다.

### CUDA OOM

우선 `batch_size`, `frame_batch_size`를 줄인다. 그다음 window 길이 또는 fold 동시 실행 여부를 확인한다. 앙상블은 여러 모델을 동시에 GPU에 올리지 않는다.

### Stage 전체가 fallback CSV를 생성함

`submission_manifest.json`의 `fallback_stages`와 Stage별 `fallback` 오류를 확인한다. fallback CSV는 계약을 유지하기 위한 안전 출력이지 정상 추론 성공을 의미하지 않는다.

### ZIP 빌더가 모델 누락으로 중단됨

다음 네 파일이 비어 있지 않은지 확인한다.

```text
model_root/stage1/best.pt
model_root/stage2/best.pt
model_root/stage2/resnet18-f37072fd.pth
model_root/stage3/best.pt
```

### 평가 서버에서 패키지 설치 실패

평가 서버는 오프라인이다. `requirements.txt`의 고정 버전이 제공 wheel/환경과 호환되는지 제출 전에 공식 샘플 실행으로 검증한다. 런타임 다운로드나 외부 URL에 의존하지 않는다.

## 7. 최종 제출 체크리스트

- [ ] 최신 대회 공지, Stage 정의, 평가 환경과 시간/용량 제한을 다시 확인했다.
- [ ] `git status --short DOC/`가 비어 있고 `DOC/`를 변경하지 않았다.
- [ ] 공식 데이터의 ID/label schema와 모든 영상 decode를 확인했다.
- [ ] Stage 3 메타데이터 보고서를 보존했다.
- [ ] 최종 YAML과 동일한 설정으로 Stage별 cache를 생성했다.
- [ ] HPO 결과가 `synthetic_smoke`가 아닌 실데이터 study인지 확인했다.
- [ ] group split에 source/scene leakage가 없는지 확인했다.
- [ ] `make -C development check`가 통과했다.
- [ ] 최종 평가 형식의 입력으로 세 predictor를 실행했다.
- [ ] `submission_manifest.json`에 fallback Stage가 없다.
- [ ] Stage별 CSV row/key/label 계약을 확인했다.
- [ ] `build_submission.sh`가 성공했고 ZIP validator가 통과했다.
- [ ] ZIP과 압축 해제 크기가 최신 제한 이내다.
- [ ] `build_manifest.json`의 SHA-256과 실제 `sha256sum`이 같다.
- [ ] 인터넷 없이 새 환경에서 `inference.py` import와 짧은 추론을 확인했다.

## 8. Quick Run Sequence

경로와 epoch/Trial 수를 확정한 후의 최소 실행 순서는 다음과 같다.

```bash
make -C development check

PYTHONPATH=development/src development/.venv/bin/python \
  development/scripts/verify_video_metadata.py \
  --data-dir development/data/raw/competition/stage3 \
  --output development/artifacts/reports/stage3_video_metadata.json

./preprocess_data.py --data-root development/data/raw/competition \
  --processed-root development/data/processed --stages stage1 \
  --size 224 --stage1-frames 16 --stage1-slots 1 --stage1-feature-mode rgb_fft

./preprocess_data.py --data-root development/data/raw/competition \
  --processed-root development/data/processed --stages stage2 \
  --size 224 --window-frames 64 --stride 32

./preprocess_data.py --data-root development/data/raw/competition \
  --processed-root development/data/processed --stages stage3 \
  --size 224 --window-frames 96 --stride 48

./tune_hyperparams.py --config development/configs/competition_two_stream.yaml \
  --data-dir development/data/raw/competition/stage3 \
  --processed-root development/data/processed \
  --output-dir development/artifacts/hpo-stage3-competition \
  --trials 50 --epochs 3 --use-physics-vector --use-amp

./run_all.sh --config development/configs/competition_baseline.yaml

./build_submission.sh \
  --input-dir development/artifacts/competition/submissions \
  --model-root development/artifacts/competition/models \
  --output-dir development/artifacts/final-submission
```

성능 수치, 실행시간과 VRAM은 실제 대회 데이터와 최종 평가 형식에서 새로 측정해 run manifest에 남긴다. 구조 스모크나 공개 예제 결과를 실제 일반화 성능으로 보고하지 않는다.

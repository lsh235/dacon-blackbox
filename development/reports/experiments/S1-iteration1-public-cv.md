# S1 Iteration 1 — Group CV 및 Local Evaluation 파이프라인

- 실행일: 2026-08-27
- 목적: 원본 콘텐츠 그룹이 train/validation에 동시에 들어가지 않도록 분할하고, Stage 1의 Macro F1·confusion matrix·클래스별 precision/recall/F1 산출을 검증한다.
- 데이터: `data/raw/public_example/stage1`의 10개 영상
- 중요 한계: 공개 `RERECORDED` 5개는 실제 기기 재녹화가 아니라 특성을 모사한 예제다. 따라서 아래 수치는 평가 파이프라인의 실행 증거일 뿐 실제 일반화 성능 또는 모델 선택 근거가 아니다.

## 분할

공개 예제에는 `source_content_id`나 `scene_id`가 없으므로 `path`의 파일명 stem을 그룹 키로 사용했다.

- 그룹: `000001`~`000005` (5개)
- fold: 2
- 각 그룹의 `original/00000N.mp4`, `rerecorded/00000N.mp4`는 같은 fold에만 배정됨
- 새 실제 `train.csv`에서는 `--group-column source_content_id` 또는 `--group-column scene_id`를 명시해야 한다. 해당 메타데이터가 있으면 path-stem fallback보다 우선한다.

## 실행 명령

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate/run_stage1_cv.py \
  --data-dir data/raw/public_example/stage1 \
  --folds 2 \
  --epochs 1 \
  --seed 20260825 \
  --output-root artifacts/experiments/iteration1-public-cv
```

## Out-of-fold 결과

| 실험 | 입력 / 손실 | Macro F1 | Accuracy | 실행 시간 |
| --- | --- | ---: | ---: | ---: |
| A | RGB 3채널 + Cross Entropy (`gamma=0`) | 0.494949 | 0.500000 | 5.949초 |
| B | RGB+FFT 6채널 + Focal (`gamma=2`) | 0.494949 | 0.500000 | 5.676초 |

두 실험의 OOF confusion matrix는 동일했다. 행은 정답, 열은 예측이다.

| true \\ predicted | ORIGINAL | RERECORDED |
| --- | ---: | ---: |
| ORIGINAL | 3 | 2 |
| RERECORDED | 3 | 2 |

클래스별 지표는 ORIGINAL precision/recall/F1 = 0.500000/0.600000/0.545455, RERECORDED = 0.500000/0.400000/0.444444이다.

## 판정

- 그룹 누수 없는 2-fold split, OOF 예측, Macro F1, confusion matrix, precision/recall 리포트 생성은 정상 동작했다.
- 1 epoch·10개 모사 예제에서 A와 B는 동률이므로 B의 성능 우위를 주장하지 않는다.
- 다음 실제 데이터 실험은 원본 콘텐츠 또는 씬 메타데이터를 `--group-column`으로 고정하고, 같은 fold·seed·epoch에서 A/B를 다시 비교한다.

## 생성 산출물

- `artifacts/experiments/iteration1-public-cv/split_assignments.csv`
- `artifacts/experiments/iteration1-public-cv/comparison.md`
- 각 실험의 `oof_predictions.csv`, `metrics.json`, `metrics.md`, fold별 체크포인트와 검증 예측

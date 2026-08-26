# Stage 1 — 재녹화 여부 판별

입력 영상을 `ORIGINAL` 또는 `RERECORDED`로 분류한다.

- 먼저 읽을 문서: [`STUDY_AND_IMPROVEMENT_GUIDE.md`](./STUDY_AND_IMPROVEMENT_GUIDE.md)
- 개선 과정 학습 노트: [`ITERATION_0_1_LEARNING.md`](./ITERATION_0_1_LEARNING.md)
- 현재 구현: [`../../src/blackbox/stages/stage1/baseline.py`](../../src/blackbox/stages/stage1/baseline.py)
- 공통 전처리: [`../../src/blackbox/common/runtime.py`](../../src/blackbox/common/runtime.py)
- 작업 상태: [`../../TASKS.md`](../../TASKS.md)의 M3
- 공개 예제 한계: [`../../reports/experiments/M2-public-example-inventory.md`](../../reports/experiments/M2-public-example-inventory.md)

현재 구현은 입출력·GPU 실행을 검증한 구조 베이스라인이다. 공개 예제에는 실제 기기 재촬영 데이터가 없고 공식 평가 지표도 확인되지 않았으므로, 현재 결과를 성능 베이스라인이나 일반화 성능의 증거로 사용하지 않는다.

## Local CV

로컬 비교는 `scripts/evaluate/run_stage1_cv.py`로 수행한다. `source_content_id`, `scene_id`, `original_video_id`, `group_id` 중 하나가 있다면 자동 사용하며, 실제 데이터에서는 원본 콘텐츠 또는 씬 열을 명시하는 것을 권장한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate/run_stage1_cv.py \
  --data-dir data/raw/train/stage1 \
  --labels-csv data/raw/train/stage1/train.csv \
  --group-column source_content_id \
  --folds 5 \
  --epochs 2 \
  --output-root artifacts/experiments/s1-cv
```

같은 fold에서 A(RGB+Cross Entropy)와 B(RGB+FFT+Focal)를 실행하고, 각 실험의 OOF Macro F1, confusion matrix, precision/recall/F1을 저장한다. 그룹 열이 없을 때의 `path` 파일명 fallback은 공개 예제 쌍에만 검증됐으므로, 실제 데이터에서 원본 콘텐츠 식별자로 쓸 수 없으면 실행을 중단하고 메타데이터를 추가해야 한다.

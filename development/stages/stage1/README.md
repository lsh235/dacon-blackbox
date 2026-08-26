# Stage 1 — 재녹화 여부 판별

입력 영상을 `ORIGINAL` 또는 `RERECORDED`로 분류한다.

- 먼저 읽을 문서: [`STUDY_AND_IMPROVEMENT_GUIDE.md`](./STUDY_AND_IMPROVEMENT_GUIDE.md)
- 현재 구현: [`../../src/blackbox/stages/stage1/baseline.py`](../../src/blackbox/stages/stage1/baseline.py)
- 공통 전처리: [`../../src/blackbox/common/runtime.py`](../../src/blackbox/common/runtime.py)
- 작업 상태: [`../../TASKS.md`](../../TASKS.md)의 M3
- 공개 예제 한계: [`../../reports/experiments/M2-public-example-inventory.md`](../../reports/experiments/M2-public-example-inventory.md)

현재 구현은 입출력·GPU 실행을 검증한 구조 베이스라인이다. 공개 예제에는 실제 기기 재촬영 데이터가 없고 공식 평가 지표도 확인되지 않았으므로, 현재 결과를 성능 베이스라인이나 일반화 성능의 증거로 사용하지 않는다.

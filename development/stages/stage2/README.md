# Stage 2 — 사고 주요 시점·상황 분석

충돌 프레임, 진입 프레임, 회피 공간 여부와 진입 방향을 예측한다.

- 현재 구현: [`../../src/blackbox/stages/stage2/baseline.py`](../../src/blackbox/stages/stage2/baseline.py)
- Iteration 2 연구용 스켈레톤·아키텍처 제안: [`ITERATION_2_SKELETON_AND_ARCHITECTURE.md`](./ITERATION_2_SKELETON_AND_ARCHITECTURE.md)
- 작업 상태: [`../../TASKS.md`](../../TASKS.md)의 M4
- 출력 계약: [`../../configs/submission_contract.json`](../../configs/submission_contract.json)

현재 ResNet18 특징과 BiGRU 구현은 구조 스모크를 통과했다. 공개 예제는 충돌 시점 외 세 과업의 충분한 정답을 제공하지 않으므로, 실제 데이터와 라벨 정의를 확보한 뒤 성능 실험을 시작한다.

# Stage별 개발 작업 영역

세 Stage의 문제 이해, 실험 계획과 결과를 서로 섞지 않고 관리하기 위한 디렉터리다. 대회 원본 자료는 계속 `../../DOC/`에 읽기 전용으로 두며, 이 디렉터리에는 원본 자료를 복사하지 않는다.

| Stage | 작업 문서 | 실행 소스 | 주요 출력 |
| --- | --- | --- | --- |
| Stage 1 | [`stage1/`](./stage1/) | [`../src/blackbox/stages/stage1/`](../src/blackbox/stages/stage1/) | `ID`, `answer` |
| Stage 2 | [`stage2/`](./stage2/) | [`../src/blackbox/stages/stage2/`](../src/blackbox/stages/stage2/) | 주요 시점·상황 |
| Stage 3 | [`stage3/`](./stage3/) | [`../src/blackbox/stages/stage3/`](../src/blackbox/stages/stage3/) | 0.1초 단위 차량 거동 |

공통 영상 I/O와 장치 관리는 [`../src/blackbox/common/`](../src/blackbox/common/), 출력 계약은 [`../src/blackbox/contracts.py`](../src/blackbox/contracts.py), 세 Stage 통합 진입점은 [`../src/blackbox/inference.py`](../src/blackbox/inference.py)에 둔다. 데이터와 체크포인트는 기존처럼 `data/<stage>/`, `artifacts/checkpoints/<stage>/` 아래에서 분리한다.

진행 상태는 [`../TASKS.md`](../TASKS.md)에서 단일 관리한다. Stage별 문서에는 문제 정의와 실험 근거를 기록하고, 상태 변경은 `TASKS.md`의 명령·결과·산출물 근거와 함께 반영한다.

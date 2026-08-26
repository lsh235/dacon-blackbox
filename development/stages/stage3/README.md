# Stage 3 — 차량 거동 특성 분석

0.1초 단위로 가감속 범주와 조향 범주를 예측한다.

- 현재 구현: [`../../src/blackbox/stages/stage3/baseline.py`](../../src/blackbox/stages/stage3/baseline.py)
- 작업 상태: [`../../TASKS.md`](../../TASKS.md)의 M5
- 환경·시간축 쟁점: [`../../reports/requirements-audit.md`](../../reports/requirements-audit.md)

현재 MViTv2-S 다중 헤드 구현은 구조 스모크를 통과했다. 공개 영상의 메타데이터 FPS와 라벨 시간축, 문서의 10Hz 정의가 일치하지 않으므로 공식 매핑을 확인하기 전에는 시간축 후처리를 확정하지 않는다.

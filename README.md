# Dacon Blackbox

블랙박스 영상 기반 재녹화 여부 판별, 사고 주요상황 분석과 차량 거동 분석을 위한 3-Stage 개발 프로젝트다.

## 디렉터리

- [`DOC/`](./DOC/): 대회 원본 자료. 개발 중 읽기 전용 참고 자료로 사용한다.
- [`development/`](./development/): 학습·추론 소스, 설정, 테스트, 계획과 실험 보고서
- [`development/stages/`](./development/stages/): Stage별 문제 이해와 실험 문서
- [`feat/`](./feat/): 제출·데이터·환경 기능 명세
- [`test/`](./test/): 명세 기반 스모크 테스트와 결과 양식

가상환경, 복제 데이터, 체크포인트, 예측 결과와 제출 ZIP은 저장소에서 제외한다. 재현에 필요한 버전과 명령은 코드·설정·보고서로 관리한다.

## Ubuntu 24.04 개발 환경

```bash
cd development
make setup
make check
```

자세한 진행 상태는 [`development/TASKS.md`](./development/TASKS.md), 전체 계획은 [`development/PLAN.md`](./development/PLAN.md), Stage 1 조사 순서는 [`development/stages/stage1/STUDY_AND_IMPROVEMENT_GUIDE.md`](./development/stages/stage1/STUDY_AND_IMPROVEMENT_GUIDE.md)를 참고한다.

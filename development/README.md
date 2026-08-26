# Development

대회 원본 자료인 `../DOC/`를 변경하지 않고 학습·추론·검증을 수행하는 개발 영역이다. 전체 작업 상태는 [`TASKS.md`](./TASKS.md), 단계별 계획은 [`PLAN.md`](./PLAN.md)에서 관리한다.

## Stage별 작업 영역

- Stage별 문제 이해와 실험 문서: [`stages/`](./stages/)
- Stage 1 학습·성능 개선 조사 가이드: [`stages/stage1/STUDY_AND_IMPROVEMENT_GUIDE.md`](./stages/stage1/STUDY_AND_IMPROVEMENT_GUIDE.md)
- Stage별 실행 코드: [`src/blackbox/stages/`](./src/blackbox/stages/)

공통 코드와 통합 진입점은 `src/blackbox/`에 유지한다. Stage별 구현에서 `DOC/` 파일을 import하거나 런타임 경로로 참조하지 않는다.

## Ubuntu 24.04 환경 구성

```bash
cd development
make setup
```

`make setup`은 `.venv`를 만들고 `requirements.txt`의 고정 버전을 설치한다. Ubuntu에 `python3-venv` 또는 `ensurepip`이 없어도 시스템 pip의 `--python` 기능으로 가상환경 내부에만 pip를 설치한다.

가상환경을 직접 활성화할 필요는 없다. Makefile은 `.venv/bin/python`이 있으면 자동으로 사용한다.

## 전체 검증

```bash
make check
```

검증 순서:

1. Ubuntu, Python, 패키지 버전과 CUDA 확인
2. Python 소스 컴파일
3. 단위 테스트
4. 예측·제출 계약 런타임 스모크

개별 명령은 다음과 같다.

```bash
make env-check
make build
make test
make smoke
```

## 베이스라인 실행

```bash
PYTHONPATH=src .venv/bin/python scripts/train/train_baseline.py --help
PYTHONPATH=src .venv/bin/python scripts/evaluate/run_baseline_inference.py --help
```

데이터, 체크포인트, 예측, 로그와 제출 ZIP은 `.gitignore` 대상이며 모두 `development/` 내부에 저장한다.

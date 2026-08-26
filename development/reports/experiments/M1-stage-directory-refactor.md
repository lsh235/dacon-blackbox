# M1 Stage별 디렉터리 분리 검증

- 날짜: 2026-08-27
- 목적: Stage 1·2·3 구현을 독립 소스 경로로 이동하고 기존 통합 계약·체크포인트 호환성을 확인
- 원본 자료 변경: 없음

## 변경 구조

- 작업 문서: `stages/stage1/`, `stages/stage2/`, `stages/stage3/`
- 실행 소스: `src/blackbox/stages/stage1/`, `stage2/`, `stage3/`
- 공통 코드: `src/blackbox/common/`, `contracts.py`, `training.py`, `inference.py`

예전 `src/blackbox/stage1~3/`에 남은 생성 캐시만 휴지통으로 이동했다. 모델 체크포인트, 데이터와 기존 예측 결과는 변경하지 않았다.

## 환경 및 전체 검증

```bash
cd development
make check
```

결과:

- Ubuntu 24.04.4 LTS, Python 3.12.3
- torch 2.8.0, torchvision 0.23.0, CUDA 12.8
- NVIDIA GeForce RTX 4060 확인
- Python 컴파일 PASS
- 단위 테스트 26개 PASS
- 예측·제출 ZIP 계약 스모크 PASS

## 실제 GPU 추론

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate/run_baseline_inference.py \
  --data-root data/interim/smoke_eval \
  --model-root artifacts/checkpoints/ubuntu2404-epoch1 \
  --output-root artifacts/predictions/stage-layout-refactor-smoke \
  --stages 1 2 3
```

| Stage | 행 수 | 실행시간 | 결과 |
| --- | ---: | ---: | --- |
| Stage 1 | 1 | 1.346초 | PASS |
| Stage 2 | 1 | 0.457초 | PASS |
| Stage 3 | 8 | 0.739초 | PASS |

세 CSV는 `ubuntu2404-epoch1-integrated-smoke`의 대응 파일과 바이트 단위로 동일했다. 이 결과는 경로 이동과 체크포인트 호환성의 증거이며 모델 정확도나 대회 성능의 증거는 아니다.

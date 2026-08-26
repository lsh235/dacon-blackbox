# M1 베이스라인 구조 스모크

- 실행일: 2026-08-26
- 목적: 학습 성능 측정이 아니라 Stage별 모델 저장·로딩, GPU 추론 및 출력 계약 검증
- 장치: NVIDIA GeForce RTX 4060
- 환경 주의: 제공 requirements와 현재 torch, torchvision, tqdm 버전이 다름

## 입력

- `DOC/Data/detail/data`를 `development/data/raw/public_example`로 복사
- Stage 1: 영상 1개
- Stage 2: 영상 1개에서 원본 번호를 유지한 프레임 8개
- Stage 3: 영상 1개에서 구조 검사용 8프레임 영상을 10FPS로 재작성

`DOC/` 원본에서는 노트북이나 전처리를 실행하지 않았다.

## 모델

- Stage 1: MViTv2-S 2-class head
- Stage 2: ResNet18 특징 + 2-layer BiGRU
- Stage 3: MViTv2-S 공유 백본 + 가감속·조향 head
- epoch: 0
- Stage 2 pretrained backbone: 사용하지 않음

0 epoch 체크포인트는 학습 성능이 없는 임의 초기화 모델이다.

## 실행 명령

```bash
PYTHONPATH=src python3 scripts/train/train_baseline.py \
  --data-root data/raw/public_example \
  --model-root artifacts/checkpoints/baseline-structure \
  --stages 1 2 3 \
  --epochs 0

BLACKBOX_STAGE3_BATCH_SIZE=1 PYTHONPATH=src \
python3 scripts/evaluate/run_baseline_inference.py \
  --data-root data/interim/smoke_eval \
  --model-root artifacts/checkpoints/baseline-structure \
  --output-root artifacts/predictions/baseline-structure-smoke-rerun \
  --stages 1 2 3
```

## 결과

| Stage | 행 수 | 실행시간 | 계약 검사 |
| --- | ---: | ---: | --- |
| Stage 1 | 1 | 1.279초 | PASS |
| Stage 2 | 1 | 0.487초 | PASS |
| Stage 3 | 8 | 0.825초 | PASS |

모델 파일 크기:

- Stage 1 `best.pt`: 130.7MiB
- Stage 2 `best.pt`: 6.2MiB
- Stage 2 ResNet18: 44.7MiB
- Stage 3 `best.pt`: 130.7MiB

## 판정과 한계

- 세 필수 함수의 모델 로딩, GPU 실행, CSV 생성과 스키마 검사는 통과했다.
- 공개 예제 전체 추론시간, 학습 수렴, 정확도 및 공식 점수는 측정하지 않았다.
- 이 결과는 평가 서버 호환 또는 모델 성능 증거가 아니다.
- 다음 성능 실험 전에 공식 지표, 전체 데이터, Stage 3 시간축 정의와 요구 버전 환경이 필요하다.

## 1 epoch 역전파 추가 확인

구조 스모크 이후 세 Stage를 각각 1 epoch로 학습하고 새 체크포인트로 동일한 소규모 추론을 실행했다.

| Stage | 학습 결과 | 추론 행 수 | 추론시간 | 계약 검사 |
| --- | --- | ---: | ---: | --- |
| Stage 1 | PASS | 1 | 1.150초 | PASS |
| Stage 2 | PASS | 1 | 0.637초 | PASS |
| Stage 3 | PASS | 8 | 1.024초 | PASS |

Stage 2는 네트워크 다운로드를 피하기 위해 무작위 초기화 ResNet18을 사용했다. Stage 3 학습은 희소 라벨 50건을 처리하는 데 약 66초가 걸렸으며, 현재 기준선은 라벨 한 건마다 영상을 처음부터 다시 디코딩한다. 이 수치는 데이터 로딩 최적화 필요성을 보여주는 로컬 관찰일 뿐 평가 서버 시간 추정치는 아니다.

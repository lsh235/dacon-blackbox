# Ubuntu 24.04 환경 재검증

- 실행일: 2026-08-27
- 목적: OS 업그레이드 후 기존 개발 진행상황과 체크포인트 호환성 확인
- 결과: PASS

## 환경 변화

업그레이드 직후 시스템 Python은 3.12.3으로 변경됐고 기존 Python 3.10에 설치돼 있던 torch, torchvision, numpy, pandas, OpenCV, tqdm과 Pillow를 찾을 수 없었다. GPU 드라이버와 RTX 4060은 정상 인식됐다.

시스템 Python을 변경하지 않고 `development/.venv`를 구성했다. Ubuntu 최소 구성에 `ensurepip`이 없어 전역 pip의 `--python` 기능으로 가상환경 내부에 pip를 설치했다.

## 최종 환경

| 항목 | 값 |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.12.3 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| opencv-python | 4.10.0.84 |
| tqdm | 4.66.5 |
| CUDA / cuDNN | 12.8 / 9.10.2 |
| GPU | NVIDIA GeForce RTX 4060 |

`pip check`와 CUDA 텐서 연산을 통과했다.

## 회귀 검증

- Python 3.12 컴파일: PASS
- 단위 테스트: 26개 PASS
- 제출 계약 스모크: PASS
- 공개 예제 첫 프레임 디코딩: 20개 PASS
- 기존 PyTorch 2.4 체크포인트를 PyTorch 2.8에서 로드: PASS
- 업그레이드 전후 기존 체크포인트의 Stage별 CSV 비교: 동일

## 새 환경 학습과 통합 추론

공개 예제에서 Stage 1·2·3을 각각 1 epoch 학습했다. 전체 학습 명령은 103.33초에 완료됐으며 세 Stage 체크포인트가 생성됐다. Stage 2는 네트워크 다운로드를 사용하지 않는 무작위 초기화 ResNet18 구조 스모크다.

새 체크포인트를 한 프로세스에서 순차 실행한 결과:

| Stage | 출력 행 | 추론시간 | 계약 검사 |
| --- | ---: | ---: | --- |
| Stage 1 | 1 | 1.100초 | PASS |
| Stage 2 | 1 | 0.328초 | PASS |
| Stage 3 | 8 | 0.714초 | PASS |

## 수정 사항

- Makefile이 `.venv/bin/python`을 자동 선택하도록 변경
- Ubuntu 24.04용 `make setup` 부트스트랩 추가
- 고정 패키지 버전과 CUDA를 검사하는 `make env-check` 추가
- 환경 감사 단위 테스트와 기능 명세 추가
- 실행 환경, 체크포인트와 예측을 연결한 JSON 매니페스트 생성

## 증거 한계

- 공개 예제와 1 epoch 결과는 구조·실행 호환 증거이며 정확도 근거가 아니다.
- RTX 4060 결과는 평가 서버의 L40S 시간·메모리 충족 증거가 아니다.
- 공식 평가 지표, 전체 데이터와 Stage 3 시간축 정의에 대한 기존 차단 항목은 그대로다.

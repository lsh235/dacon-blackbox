# Ubuntu 24.04 개발 환경 호환성

## 프로젝트 전용 환경

- 시스템 Python에 패키지를 직접 설치하지 않는다.
- `development/.venv`의 Python과 pip를 모든 개발·검증 명령에서 사용한다.
- Ubuntu에 `ensurepip`이 없어도 전역 pip의 `--python` 기능으로 가상환경 pip를 복구할 수 있다.

## 패키지 계약

- `development/requirements.txt`의 모든 직접 의존성은 정확한 버전으로 설치한다.
- 설치 후 `pip check`와 자동 환경 감사를 모두 통과한다.
- torch 2.8.0과 torchvision 0.23.0 조합을 유지한다.

## GPU 실행

- PyTorch가 CUDA GPU와 CUDA 런타임을 인식해야 한다.
- 최소 CUDA 텐서 연산을 실행하여 드라이버와 런타임 연결을 확인한다.
- 기존 체크포인트를 새 환경에서 로드하고 세 Stage의 실제 추론을 다시 실행한다.

## 업그레이드 후 회귀 검사

- Python 소스를 새 인터프리터로 다시 컴파일한다.
- 전체 단위 테스트와 제출 계약 스모크를 다시 실행한다.
- OpenCV로 공개 예제 영상을 디코딩하고 기존 메타데이터 인벤토리를 재검증한다.
- 로컬 RTX 4060 결과는 평가 서버 L40S 성능 증거로 사용하지 않는다.

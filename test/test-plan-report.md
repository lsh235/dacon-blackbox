# Test Plan Report

- Generated at (UTC): `2026-08-26 15:28:25`
- Source mode: spec markdown -> normalized test cases
- Specs used:
  - `feat/00-submission-contract.md`
  - `feat/01-reference-data.md`
  - `feat/02-environment-compatibility.md`

## Traceability

| ID | Priority | Category | Type | Source | Scenario | Status |
|---|---|---|---|---|---|---|
| TC-001 | low | ui | functional | `feat/00-submission-contract.md` | 필수 추론 함수 | TODO |
| TC-002 | low | ui | functional | `feat/00-submission-contract.md` | `inference.py`는 `predict_stage1`, `predict_stage2`, `predict_stage3`를 모듈 최상위에 정의한다. | TODO |
| TC-003 | low | ui | functional | `feat/00-submission-contract.md` | 각 함수는 해당 Stage의 정확한 컬럼 순서를 가진 `pandas.DataFrame`을 반환한다. | TODO |
| TC-004 | low | ui | functional | `feat/00-submission-contract.md` | 제출 코드는 네트워크, 사용자 캐시, 절대 경로에 의존하지 않는다. | TODO |
| TC-005 | low | ui | functional | `feat/00-submission-contract.md` | Stage 1 출력 | TODO |
| TC-006 | low | ui | functional | `feat/00-submission-contract.md` | `ID`는 비어 있지 않고 영상마다 유일하다. | TODO |
| TC-007 | low | ui | functional | `feat/00-submission-contract.md` | `answer`는 `ORIGINAL` 또는 `RERECORDED`다. | TODO |
| TC-008 | low | ui | functional | `feat/00-submission-contract.md` | Stage 2 출력 | TODO |
| TC-009 | low | ui | functional | `feat/00-submission-contract.md` | `collision_frame`과 `entry_frame`은 입력 이미지 파일명의 원본 프레임 번호다. | TODO |
| TC-010 | low | ui | functional | `feat/00-submission-contract.md` | `evasion_space`와 `entry_side`는 제출 계약에 허용된 값만 사용한다. | TODO |
| TC-011 | low | ui | functional | `feat/00-submission-contract.md` | Stage 3 출력 | TODO |
| TC-012 | low | ui | functional | `feat/00-submission-contract.md` | `(ID, sample_index)` 조합은 유일하다. | TODO |
| TC-013 | low | ui | functional | `feat/00-submission-contract.md` | 각 영상의 `sample_index`는 0부터 시작해 누락 없이 증가한다. | TODO |
| TC-014 | low | ui | functional | `feat/00-submission-contract.md` | 가감속과 조향 레이블은 제출 계약에 허용된 값만 사용한다. | TODO |
| TC-015 | low | ui | functional | `feat/00-submission-contract.md` | 제출 ZIP | TODO |
| TC-016 | low | ui | functional | `feat/00-submission-contract.md` | ZIP은 필수 추론 파일, requirements와 Stage별 모델 파일을 포함한다. | TODO |
| TC-017 | low | ui | functional | `feat/00-submission-contract.md` | ZIP 경로에는 절대 경로와 상위 디렉터리 이동이 없다. | TODO |
| TC-018 | low | ui | functional | `feat/00-submission-contract.md` | ZIP 크기와 압축 해제 후 크기는 대회 제한을 넘지 않는다. | TODO |
| TC-019 | low | ui | functional | `feat/01-reference-data.md` | 공개 예제 데이터 준비 | TODO |
| TC-020 | low | ui | functional | `feat/01-reference-data.md` | `DOC/` 원본을 수정하거나 직접 실행 산출물을 기록하지 않는다. | TODO |
| TC-021 | low | ui | functional | `feat/01-reference-data.md` | 공개 예제는 명시적인 개발 경로로 복사한 뒤 사용한다. | TODO |
| TC-022 | low | ui | functional | `feat/01-reference-data.md` | 이미 존재하는 개발 데이터를 묵시적으로 덮어쓰지 않는다. | TODO |
| TC-023 | low | ui | functional | `feat/01-reference-data.md` | Stage 1 labels의 영상 경로와 분류 레이블을 검사한다. | TODO |
| TC-024 | low | ui | functional | `feat/01-reference-data.md` | Stage 2 labels의 영상 경로와 시점·장면 컬럼을 검사한다. | TODO |
| TC-025 | low | ui | functional | `feat/01-reference-data.md` | Stage 3 labels의 ID별 영상과 표본·프레임 컬럼을 검사한다. | TODO |
| TC-026 | low | ui | functional | `feat/01-reference-data.md` | 선택적으로 각 영상의 첫 프레임이 디코딩되는지 검사한다. | TODO |
| TC-027 | low | ui | functional | `feat/01-reference-data.md` | 공개 예제 통과는 실행·입출력 호환 증거로만 사용한다. | TODO |
| TC-028 | low | ui | functional | `feat/01-reference-data.md` | 공개 예제로 측정한 값은 일반화 성능이나 공식 대회 점수로 보고하지 않는다. | TODO |
| TC-029 | low | ui | functional | `feat/02-environment-compatibility.md` | Ubuntu 24.04 개발 환경 호환성 | TODO |
| TC-030 | low | ui | functional | `feat/02-environment-compatibility.md` | 프로젝트 전용 환경 | TODO |
| TC-031 | low | ui | functional | `feat/02-environment-compatibility.md` | 시스템 Python에 패키지를 직접 설치하지 않는다. | TODO |
| TC-032 | low | ui | functional | `feat/02-environment-compatibility.md` | `development/.venv`의 Python과 pip를 모든 개발·검증 명령에서 사용한다. | TODO |
| TC-033 | low | ui | functional | `feat/02-environment-compatibility.md` | Ubuntu에 `ensurepip`이 없어도 전역 pip의 `--python` 기능으로 가상환경 pip를 복구할 수 있다. | TODO |
| TC-034 | low | ui | functional | `feat/02-environment-compatibility.md` | 패키지 계약 | TODO |
| TC-035 | low | ui | functional | `feat/02-environment-compatibility.md` | `development/requirements.txt`의 모든 직접 의존성은 정확한 버전으로 설치한다. | TODO |
| TC-036 | low | ui | functional | `feat/02-environment-compatibility.md` | 설치 후 `pip check`와 자동 환경 감사를 모두 통과한다. | TODO |
| TC-037 | low | ui | functional | `feat/02-environment-compatibility.md` | torch 2.8.0과 torchvision 0.23.0 조합을 유지한다. | TODO |
| TC-038 | low | ui | functional | `feat/02-environment-compatibility.md` | GPU 실행 | TODO |
| TC-039 | low | ui | functional | `feat/02-environment-compatibility.md` | PyTorch가 CUDA GPU와 CUDA 런타임을 인식해야 한다. | TODO |
| TC-040 | low | ui | functional | `feat/02-environment-compatibility.md` | 최소 CUDA 텐서 연산을 실행하여 드라이버와 런타임 연결을 확인한다. | TODO |
| TC-041 | low | ui | functional | `feat/02-environment-compatibility.md` | 기존 체크포인트를 새 환경에서 로드하고 세 Stage의 실제 추론을 다시 실행한다. | TODO |
| TC-042 | low | ui | functional | `feat/02-environment-compatibility.md` | 업그레이드 후 회귀 검사 | TODO |
| TC-043 | low | ui | functional | `feat/02-environment-compatibility.md` | Python 소스를 새 인터프리터로 다시 컴파일한다. | TODO |
| TC-044 | low | ui | functional | `feat/02-environment-compatibility.md` | 전체 단위 테스트와 제출 계약 스모크를 다시 실행한다. | TODO |
| TC-045 | low | ui | functional | `feat/02-environment-compatibility.md` | OpenCV로 공개 예제 영상을 디코딩하고 기존 메타데이터 인벤토리를 재검증한다. | TODO |
| TC-046 | low | ui | functional | `feat/02-environment-compatibility.md` | 로컬 RTX 4060 결과는 평가 서버 L40S 성능 증거로 사용하지 않는다. | TODO |

## Execution Notes

- Fill `Status` with `PASS`, `FAIL`, `BLOCKED`, or `SKIPPED`.
- Add command output snippets and error logs below after running tests.

### 2026-08-27 Ubuntu 24.04 execution

- Specs: `feat/00-submission-contract.md`, `feat/01-reference-data.md`, `feat/02-environment-compatibility.md`.
- `node test/run-spec-smoke.mjs`: PASS, 46 cases loaded.
- `make -C development check`: PASS, environment audit + build + 26 unit tests + runtime smoke.
- Public example first-frame decode: PASS, 20 videos.
- Existing PyTorch 2.4 checkpoints loaded with PyTorch 2.8: PASS; output CSVs unchanged.
- New 1 epoch Stage 1/2/3 training and integrated GPU inference: PASS.
- Accuracy and official-metric cases remain blocked by missing official definitions and full training data.

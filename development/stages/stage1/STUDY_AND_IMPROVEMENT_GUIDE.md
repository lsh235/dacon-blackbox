# Stage 1 이해 및 성능 개선 조사 가이드

- 대상: 원본 블랙박스 영상과 다른 화면·기기를 통해 다시 촬영한 영상의 이진 분류
- 출력 계약: `pandas.DataFrame`의 `ID`, `answer`; `answer`는 `ORIGINAL` 또는 `RERECORDED`
- 기준 소스: [`../../src/blackbox/stages/stage1/baseline.py`](../../src/blackbox/stages/stage1/baseline.py)
- 진행 작업: [`../../TASKS.md`](../../TASKS.md)의 M3-001~M3-005
- 대회 원본: `../../../DOC/` — 읽기 전용 참고 자료이며 실행 경로로 사용하지 않는다.

이 문서는 “무엇을 공부할지”와 “어떤 순서로 성능을 개선할지”를 분리해 정리한다. 논문의 수치가 이 대회의 성능을 보장하지 않으므로, 모든 방법은 동일한 누수 방지 분할과 공식 지표에서 다시 검증한다.

## 1. 현재 구현을 먼저 정확히 이해하기

현재 코드는 제출 입출력과 GPU 실행을 검증하기 위한 구조 베이스라인이다.

| 항목 | 현재 구현 | 확인할 위치 |
| --- | --- | --- |
| 모델 | RGB MViTv2-S + FFT 2-D CNN + Flicker Dilated Conv1d + 4-D Correlation/Tied ConvGRU fusion | `Stage1MViT.__init__` |
| 보조 학습 | frame Focal + truncated smoothing + masked reconstruction + BCE-to-one mask regularization | `Stage1MultiTaskLoss` |
| 학습 입력 | 초반·중반·후반에서 연속 16프레임, 시작점 random jitter | `Stage1TrainingDataset` |
| 추론 입력 | 동일 3구간에서 중앙 연속 16프레임 | `Stage1InferenceDataset` |
| 전처리 | RGB 224 crop, 별도 320 forensic crop, 증강 후 FFT·flicker 계산 | `prepare_stage1_inputs` |
| 집계 | 각 구간의 `RERECORDED` softmax 확률 평균 | `score_stage1_checkpoint` |
| 결정 | 평균 확률이 0.5 이상이면 `RERECORDED` | `predict_stage1` |
| 오류 정책 | 모든 구간 디코딩 실패 시 `RERECORDED` | `_Stage1Clips`, `predict_stage1` |

주의할 점:

1. `weights=None`이므로 현재 모델은 사전학습 가중치를 사용하지 않는다.
2. 현재 1 epoch 실행 결과는 역전파·체크포인트·추론 계약을 검증한 스모크이지 성능 측정값이 아니다.
3. 공개 Stage 1 예제는 원본 5개와 특성을 모사한 파생 재녹화 5개뿐이며 실제 기기로 재촬영한 검증셋이 아니다.
4. 학습·검증·추론은 같은 3구간 정책을 사용하지만, clip 길이·jitter 폭은 고정 Group CV에서 별도로 비교해야 한다.
5. 사전학습 가중치와 외부·합성 데이터의 사용 허용 여부는 아직 공식 확인되지 않았다. 확인 전에는 최종 학습에 넣지 않는다.

## 2. 이해를 위해 공부할 주제

### 2.1 재녹화가 만들어지는 물리적 파이프라인

다음 흐름에서 어떤 흔적이 생기고, 어떤 후처리로 흔적이 약해지는지 설명할 수 있어야 한다.

```text
원본 영상 → 디스플레이 표시 → 화면 화소/주사율 → 카메라 광학계·센서
          → 노출/화이트밸런스/ISP → 재인코딩 → 재녹화 영상
```

찾아볼 개념과 검색어:

- `screen recapture detection`, `display-camera pipeline`, `recaptured video forensics`
- 디스플레이 화소 격자와 카메라 CFA 간 간섭, 모아레와 에일리어싱
- 디스플레이 주사율과 카메라 셔터/FPS 불일치, 플리커와 밴딩
- rolling shutter, 자동 노출, 자동 초점, 화이트밸런스 변화
- 감마·색역·chromaticity 변화, 화면 반사·경계·원근 왜곡
- 재인코딩, double compression, resize와 sharpening 흔적

목표는 특정 흔적 하나를 외우는 것이 아니라 “장치·각도·조명·코덱이 바뀌어도 남는 단서”와 “쉽게 사라지는 단서”를 구분하는 것이다.

### 2.2 영상 텐서와 시간 샘플링

확인할 내용:

- 모델 입력 `B × C × T × H × W`에서 각 축의 의미
- 연속 16프레임과 영상 전체 균등 16프레임이 포착하는 정보의 차이
- 실제 시간 간격을 결정하는 FPS, frame stride와 clip duration의 관계
- 짧은 영상의 프레임 반복, 긴 영상의 구간 누락, variable frame rate 처리
- clip-level 예측을 video-level 예측으로 바꾸는 mean probability, mean logit, max, voting의 차이

현재 `_clip_ids`는 메타데이터의 프레임 수를 사용한다. 잘못된 프레임 수·VFR·디코딩 중단이 샘플 위치에 미치는 영향을 별도 테스트해야 한다.

### 2.3 MViTv2 구조와 사전학습

다음 질문에 답할 수 있을 정도로 본다.

- 멀티스케일 토큰과 pooling attention이 일반 ViT와 어떻게 다른가?
- MViTv2의 residual pooling과 decomposed relative positional embedding은 무엇을 개선하는가?
- 16프레임·224 입력과 공식 가중치의 전처리 계약은 무엇인가?
- 무작위 초기화와 Kinetics 사전학습 미세조정의 데이터 요구량은 어떻게 다른가?
- 전체 미세조정, backbone 동결, 단계적 unfreezing의 비용과 위험은 무엇인가?

사전학습 가중치를 실험할 때는 허용 규칙을 먼저 확인하고, 평가 서버가 오프라인이므로 필요한 가중치를 제출물 안에서 상대 경로로 로드해야 한다.

### 2.4 전처리와 증강

리사이즈·크롭·압축은 재녹화 단서를 지울 수도 만들 수도 있다. 다음을 각각 검증한다.

- interpolation 방식이 모아레와 고주파 성분에 주는 영향
- 224 중앙 크롭이 화면 경계나 주변 반사를 제거하는 영향
- 색상 변환이 chromaticity 단서를 훼손하는 정도
- H.264 재인코딩, JPEG, blur, resize, noise 증강이 양쪽 클래스에 공정한지
- 시간 증강이 프레임 순서와 실제 주기성 단서를 보존하는지

증강을 적용했다는 사실보다 “라벨 의미를 보존하며 실제 평가 분포에 가까운가”가 중요하다. 재녹화 클래스에만 특정 코덱이나 테두리를 넣으면 모델이 합성 규칙을 외우는 누수가 된다.

### 2.5 분할, 누수와 도메인 일반화

가장 먼저 고정할 부분이다. 같은 원본 콘텐츠에서 만든 원본/파생/재녹화 영상이나 같은 촬영 세션의 클립이 train과 valid에 나뉘면 점수가 과대평가된다.

분할 그룹 후보:

- 원본 콘텐츠 또는 사고 영상 ID
- 재생 디스플레이 장치
- 재촬영 카메라 장치
- 촬영 세션, 장소, 각도와 조명 조건
- 코덱·해상도·편집 파이프라인

최소한 `source_content_id` 단위 그룹 분할을 하고, 메타데이터가 있다면 장치·세션을 포함한 교차 도메인 검증을 별도로 둔다. 랜덤 파일 분할의 결과만으로 모델을 선택하지 않는다.

### 2.6 지표, 임계값과 오류 분석

공식 지표가 공개되면 그 구현을 유일한 모델 선택 기준으로 둔다. 그 전에는 accuracy 하나로 결론 내리지 말고 다음은 진단용으로만 기록한다.

- confusion matrix, 클래스별 precision/recall/F1
- ROC-AUC 또는 PR-AUC
- 임계값별 false original과 false rerecorded 비용
- Brier score 또는 reliability diagram을 이용한 확률 보정 상태
- 장치·조명·각도·해상도·코덱별 성능과 실패 샘플

검증셋에서 임계값을 고른 뒤 같은 검증셋의 점수를 최종 성능처럼 보고하지 않는다. 임계값 선택용과 최종 확인용 데이터를 분리하거나 교차검증의 out-of-fold 예측을 사용한다.

## 3. 현재 코드를 읽는 순서

1. [`../../src/blackbox/contracts.py`](../../src/blackbox/contracts.py): Stage 1 출력 컬럼과 허용 값
2. [`../../src/blackbox/common/runtime.py`](../../src/blackbox/common/runtime.py): 프레임 디코딩, 크롭, 정규화 상수, 장치·시드
3. [`../../src/blackbox/stages/stage1/baseline.py`](../../src/blackbox/stages/stage1/baseline.py): 모델, 학습, 3구간 추론, 집계
4. [`../../src/blackbox/training.py`](../../src/blackbox/training.py): Stage별 학습 오케스트레이션
5. [`../../src/blackbox/inference.py`](../../src/blackbox/inference.py): 대회가 호출할 통합 함수 노출
6. [`../../scripts/train/train_baseline.py`](../../scripts/train/train_baseline.py): 학습 CLI
7. [`../../scripts/evaluate/run_baseline_inference.py`](../../scripts/evaluate/run_baseline_inference.py): 추론 CLI와 시간 기록
8. [`../../tests/unit/test_baseline_api.py`](../../tests/unit/test_baseline_api.py): 필수 함수 시그니처와 Stage API 검사

읽으면서 입력 shape, dtype, 값 범위, 프레임 인덱스, 체크포인트 키, 오류 시 반환 정책을 직접 표로 적는다. 함수 이름만 보고 동작을 추측하지 않는다.

## 4. 성능 개선을 위해 조사할 우선순위

### P0. 데이터와 평가 기준 고정

가장 높은 우선순위다.

- 공식 Stage 1 지표와 통합 점수 계산법 확인
- 사전학습·외부·합성 데이터 허용 및 보고 규칙 확인
- 실제 기기 재녹화 데이터 확보: 디스플레이, 카메라, 각도, 거리, 조명, 주사율, 해상도, 코덱을 다양화
- 원본 콘텐츠·장치·세션 기반 그룹 분할과 중복 해시 검사
- 공개 파생 예제와 실제 재촬영 검증셋을 분리해 결과 기록

이 단계가 없으면 이후 실험의 개선값을 신뢰할 수 없다.

### P1. 강한 재현 기준선

비교 후보:

1. 현재 `weights=None`
2. 규칙상 허용될 경우 `MViT_V2_S_Weights.KINETICS400_V1`
3. backbone 동결 후 head 학습 → 일부 블록 unfreeze → 전체 미세조정
4. class weight 또는 balanced sampler
5. learning rate, weight decay, warmup, scheduler, epoch와 early stopping

모든 실험은 같은 split, seed 목록, clip 수, 해상도와 지표를 사용한다. AMP는 학습과 추론에 적용되며, 속도·VRAM뿐 아니라 지표와 수치 안정성도 함께 기록한다.

### P2. 시간 샘플링과 video-level 집계

우선 작은 조합부터 비교한다.

- 구간 수: 1 / 3 / 5
- 한 구간 프레임 수: 8 / 16 / 32
- 연속 프레임 대비 FPS 정규화 stride
- 균등 샘플 대비 random temporal jitter
- probability mean 대비 logit mean, max, majority vote
- 검증 데이터로 선택한 threshold와 확률 calibration

구간 수와 해상도를 동시에 바꾸지 않는다. 정확도뿐 아니라 영상당 디코딩 시간, GPU 시간과 VRAM을 함께 기록한다.

### P3. 재녹화 단서 보존형 전처리와 증강

실험 후보:

- center crop 대비 multi-crop 또는 patch sampling
- 약한 resize/blur/noise/exposure/color jitter
- 양 클래스 모두에 적용하는 codec 재인코딩 강건성
- frame drop, FPS 변화, temporal jitter
- 화면 경계가 없는 patch와 경계가 포함된 patch의 앙상블

원본에만 없는 인공 패턴을 재녹화에 삽입하는 식의 label shortcut은 금지한다. 증강 강도별로 원본 영상과 실제 재녹화 영상의 단서가 얼마나 보존되는지 시각화한다.

### P4. 단서·모델 확장

아래는 P0~P3가 고정된 뒤 독립 가설로 검증한다.

- RGB video backbone 외에 FFT/power spectrum 또는 chromaticity 특징 보조 분기
- 전체 프레임 모델과 고주파 patch 모델의 결합
- 영상당 다양한 공간 patch를 평가한 뒤 robust voting
- 동일 예산에서 R3D, R(2+1)D, S3D, Video Swin 등과 비교
- 모델 앙상블은 단일 모델의 오류 상관과 60분 제한을 확인한 뒤 적용

문서 이미지나 일반 객체 재촬영 연구의 기법은 블랙박스 동영상에 그대로 성립한다고 가정하지 않는다. 이 대회에서는 “시도할 특징 가설”로만 가져오고 실제 장치 교차 검증으로 판단한다.

### P5. 추론 효율과 제출 안정성

- 한 번의 순차 디코딩으로 여러 clip을 만드는 방식과 현재 clip별 seek 비교
- DataLoader worker 0~4와 batch size별 CPU·GPU 사용률
- FP16/autocast 정확도와 처리량
- checkpoint 용량, 오프라인 로딩, 상대 경로 의존성
- 손상 영상·짧은 영상·VFR에서 fallback 정책
- Stage 1 최적화가 Stage 2·3 포함 전체 60분 예산에 미치는 영향

로컬 RTX 4060 결과는 상대 비교에 사용하고, 최종 시간 충족 근거는 평가 서버와 가까운 L40S 조건에서 다시 측정한다.

## 5. 추천 실험 순서

| 순서 | 작업 | 변경 변수 | 완료 기준 | 연결 작업 |
| --- | --- | --- | --- | --- |
| 1 | 공식 규칙·실데이터 확보 | 없음 | 지표·허용 데이터·그룹 키 확정 | M0-002, M0-006, M2 |
| 2 | 누수 방지 split 고정 | split만 | 콘텐츠·장치·세션 중복 0건 | M2-002 |
| 3 | 현재 기준선 측정 | 없음 | seed별 점수·시간·VRAM 기록 | M3-001 |
| 4 | 사전학습·미세조정 비교 | 초기 가중치·freeze | 기준선과 동일 조건 비교 | M3-001 |
| 5 | 시간 샘플링 비교 | clip/frames/stride 하나씩 | 성능-시간 Pareto 후보 선택 | M3-002 |
| 6 | 증강 비교 | 증강 하나씩 | 실제 재촬영 검증셋에서 개선 | M3-003 |
| 7 | 집계·임계값 비교 | aggregation/threshold | 고정 OOF 예측으로 선택 | M3-004 |
| 8 | 장치·조건별 오류 분석 | 없음 | 실패 유형과 다음 가설 연결 | M3-005 |

Stage 1의 현재 상태에서는 1~2가 차단되어 있다. 그동안 할 수 있는 작업은 코드 단위 테스트, 디코딩 프로파일, 실험 설정 템플릿과 데이터 메타데이터 스키마 준비까지다.

## 6. 실험 기록 템플릿

각 실험은 `../../reports/experiments/`에 다음 내용을 남긴다.

```markdown
# S1-XXX 실험명

- 가설:
- 기준 실행 ID:
- 변경 변수(하나):
- 고정 변수: 데이터 버전, split 해시, seed, 전처리, metric 버전
- 학습 설정:
- 추론 설정:
- 결과: 공식 지표, 보조 지표, seed 평균/편차
- 비용: 학습 시간, 영상당 추론 시간, 최대 VRAM, checkpoint 크기
- 조건별 오류: 장치, 조명, 각도, 해상도, 코덱
- 결론: 채택 / 보류 / 기각
- 다음 실험:
```

설정과 데이터·split·가중치·예측 파일 해시는 실행 매니페스트에 연결한다. 결과가 좋아도 분할이나 전처리가 달라졌다면 동일 실험으로 비교하지 않는다.

## 7. 우선 읽을 원문 자료

### 모델과 구현

- [Torchvision Video MViT 문서](https://docs.pytorch.org/vision/stable/models/video_mvit.html): 제공 모델과 입력 계열 확인
- [Torchvision `mvit_v2_s` API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.mvit_v2_s.html): 공식 가중치, 입력 크기와 transforms 확인
- [Multiscale Vision Transformers, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf): MViT의 멀티스케일 구조
- [MViTv2, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.html): MViTv2의 개선점
- [Torchvision transforms 문서](https://docs.pytorch.org/vision/main/transforms.html): resize·crop·dtype·보간 방식의 구현 기준
- [PyTorch AMP 문서](https://docs.pytorch.org/docs/stable/amp.html): 학습·추론 mixed precision 적용 방법

### 재녹화 단서와 일반화

- [CMA: Chromaticity Map Adapter, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_CMA_A_Chromaticity_Map_Adapter_for_Robust_Detection_of_Screen-Recapture_CVPR_2024_paper.html): chromaticity와 압축·저해상도 강건성 가설. 문서 이미지 연구이므로 블랙박스 영상에 별도 검증 필요
- [Domain-Generalized Object Anti-Spoofing, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Lee_Domain-Generalized_Object_Anti-Spoofing_Bridging_Gaps_and_Patch_Selection_for_Robust_WACV_2025_paper.html): power spectrum, patch selection과 도메인 일반화. 객체 이미지 과제에서 전이된 가설로만 사용
- [Raw Screen Image and Video Demoireing, 2023](https://arxiv.org/abs/2310.20332): 디스플레이 격자와 센서 샘플링이 만드는 모아레의 물리적 이해
- [DynaAugment, 2022](https://arxiv.org/abs/2206.15015): 비디오의 시간적으로 변화하는 증강 아이디어
- [Identification of Recaptured Photographs on LCD Screens, ICIP 2010](https://ieeexplore.ieee.org/document/5495419/): LCD 재촬영 탐지의 초기 영상 포렌식 관점
- [Single-View Recaptured Image Detection, ICIP 2010](https://ieeexplore.ieee.org/document/5583280/): 단일 이미지의 재촬영 물리 특징 관점

논문은 초록과 결론만 보지 말고 데이터 수집 장치, train/test 도메인 분리, 압축 조건, 해상도, 실패 사례를 함께 확인한다. 우리 데이터 조건과 다른 부분을 실험 보고서에 명시한다.

## 8. 바로 실행할 체크리스트

- [ ] 공식 Stage 1 평가 지표와 통합 점수 확인
- [ ] 사전학습·외부·합성 데이터 허용 범위 확인
- [ ] 실제 재녹화 데이터의 장치·세션·원본 콘텐츠 메타데이터 확보
- [ ] `source_content_id` 기반 train/valid/test 분할 및 중복 검사
- [ ] 현재 `weights=None`, 16프레임, 3구간 기준선 측정
- [ ] 학습과 추론의 시간 샘플 분포를 맞춘 대조 실험
- [ ] 허용 시 Kinetics 사전학습 기준선 측정
- [ ] sampler → augmentation → aggregation 순서로 한 변수씩 비교
- [ ] 장치·각도·조명·코덱별 오류 분석
- [ ] 최종 후보의 시간·VRAM·오프라인 제출 계약 검증

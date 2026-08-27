# Stage 1 Iteration 2 — 재녹화 도메인 강건성

Iteration 2의 목적은 공개 예제 점수를 높이는 것이 아니라, 재녹화 과정에서 일어날 수 있는 색·각도 변화와 영상 구간 편차에 덜 민감한 학습/추론 경로를 만드는 것이다.

## 학습 증강

`Stage1TrainAugmentation`은 crop 뒤, RGB 정규화와 FFT feature 계산 전에 적용된다. 따라서 RGB와 FFT가 같은 변형된 프레임에서 계산되며, train과 inference 전처리의 feature 정의는 변하지 않는다.

| 변형 | 기본값 | 재녹화 가설 | 보존한 제약 |
| --- | --- | --- | --- |
| ColorJitter | 확률 0.80, brightness/contrast ±0.15, saturation ±0.12 | 모니터 반사·노출·색 응답 차이 | blur, random resize/crop 미사용 |
| RandomAffine | 확률 0.35, 회전 ±2°, 이동 가로/세로 ±2% | 스마트폰/카메라의 미세한 촬영 각도 차이 | scale·강한 회전 미사용, bilinear 보간 |

한 클립의 모든 프레임에 **같은 파라미터**를 적용한다. 프레임마다 서로 다른 색/기하 변형을 만들면 실제 재촬영과 무관한 시간 깜빡임을 학습할 수 있기 때문이다. 모아레 단서를 보존하기 위해 강한 crop, resize jitter, blur는 넣지 않았다.

학습 명령에서 증강을 끄려면 `--stage1-no-augmentation`을 사용한다. 저장 checkpoint의 `augmentation` 항목에 실제 활성 여부와 파라미터가 기록된다.

## Temporal TTA

추론은 기본적으로 영상 전체를 초반·중반·후반의 3개 uniform slot으로 나누고, 각 slot에서 고정 길이 clip을 읽는다. 각 clip의 `RERECORDED` softmax 확률을 평균해 영상 하나의 확률로 만든다.

```text
video → early clip ─┐
      → middle clip ├→ model → P(RERECORDED) 평균 → final label
      → late clip ──┘
```

`inference_tta_slots`는 checkpoint `sampling` metadata에 저장된다. 과거 checkpoint에는 이 값이 없으므로 호환 경로에서 기본값 3을 사용한다. CLI에서는 `--stage1-tta-slots N`으로 실험할 수 있다. TTA 수를 늘리면 특정 구간 노이즈의 영향은 줄 수 있지만, 영상 디코딩·추론 시간은 거의 비례해 늘어난다.

## 확인 범위

- clip-consistent ColorJitter/RandomAffine의 shape, range, finite 값을 단위 테스트로 검증한다.
- 변형하지 않는 설정은 입력 clip을 정확히 유지한다.
- legacy/new checkpoint의 TTA slot 해석과 유효하지 않은 metadata 검증을 테스트한다.

이는 파이프라인의 동작 증거이며, 증강 또는 TTA가 비공개 성능을 올린다는 증거는 아니다. 실제 데이터가 주어지면 고정된 Group CV split에서 `augmentation off/on`, `TTA 1/3`을 각각 한 요인씩 비교한다.

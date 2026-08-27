# Stage 2 Iteration 2 — Sliding Window Skeleton과 아키텍처 제안

## 확인된 공식 입출력

Stage 2는 영상별로 아래 한 행을 반환한다.

| 컬럼 | 의미 |
| --- | --- |
| `ID` | 평가 영상 식별자 |
| `collision_frame` | 충돌 시점의 **원본 프레임 번호** |
| `entry_frame` | 피해차량의 최초 진입 원본 프레임 번호 |
| `evasion_space` | 충돌 당시 회피 공간 여부 |
| `entry_side` | 피해차량 진입 방향 (`LEFT`/`RIGHT`) |

공개 `labels.csv`에는 `t_collision`만 채워져 있고, `t_entry`, `evasion_space`, `entry_side`는 `-1`이다. 따라서 `-1`을 0이나 별도 클래스라고 추측하지 않는다. 스켈레톤의 loss는 이를 `IGNORE_INDEX`로 남겨 해당 항목을 학습에서 제외한다.

## 추가한 연구용 모듈

- [`dataset_stage2.py`](../../src/blackbox/stages/stage2/dataset_stage2.py): CSV metadata만 먼저 읽고, `window_frames` 크기의 프레임 청크만 지연 디코딩하는 `Stage2SlidingWindowDataset`
- [`model_stage2.py`](../../src/blackbox/stages/stage2/model_stage2.py): 작은 배치로 CNN 특징을 뽑은 뒤 BiLSTM으로 시퀀스를 처리하는 `Stage2CnnBiLSTM`
- [`train_stage2.py`](../../src/blackbox/stages/stage2/train_stage2.py): chunk-local loss와 실험용 checkpoint를 만드는 독립 trainer

```text
video metadata only
        ↓
window [start, end) ── lazy decode ── RGB frames [T, 3, H, W]
        ↓                                  ↓
original-frame map                    CNN (small frame batches)
        ↓                                  ↓
collision_frame / entry_frame ← local temporal logits ← BiLSTM
                                                  └→ evasion / entry-side logits
```

마지막 tail window는 마지막 프레임까지 포함하며, 짧은 video는 마지막 frame을 반복해 고정 길이 tensor로 맞춘다. 하지만 `valid_length`와 `frame_numbers`를 함께 돌려 주므로 padding 위치를 사건으로 선택하지 않으며, local argmax를 원본 프레임 번호로 다시 변환할 수 있다.

이 코드는 현재 제출 모델에 연결하지 않은 연구용 스켈레톤이다. 전체 video에 분산된 window score를 합쳐 최종 한 개의 사건 시점을 고르는 규칙, 실제 라벨 정의와 성능 평가는 실제 학습 데이터가 온 뒤에 확정해야 한다.

## Optical Flow vs Object Tracking 제안

첫 실전 후보는 **RGB CNN+BiLSTM에 optical-flow 기반 motion feature를 보조 입력으로 결합**하는 방식이다. YOLO 기반 object tracking을 필수 입력으로 두지는 않는다.

| 선택지 | 장점 | 위험 | Iteration 2 결정 |
| --- | --- | --- | --- |
| Optical Flow | 충돌 직전 상대 운동·차선 진입의 속도/방향 변화를 직접 표현, detector 라벨 불필요 | ego-motion과 조명 변화에도 반응 | 보조 motion stream으로 우선 검증 |
| Object Tracking | 차량별 궤적·차선 관계를 설명하기 좋음 | 가림/충돌/야간에서 track 단절, YOLO weights·라이선스·오프라인 패키징 검증 필요 | 라벨·규칙 확인 후 후보로 추가 |

구체적으로는 동일 sliding window에서 RGB 특징과 인접 frame flow 요약(크기, 방향 histogram 또는 작은 flow encoder)을 함께 만든다. 블랙박스 자체의 전진 운동도 flow에 강하게 섞이므로, 전역 camera motion을 보정하거나 RGB-only 대조군과 반드시 비교한다. 이후 실제 라벨이 충분할 때 detector/tracker의 차량별 상대 궤적을 세 번째 stream으로 추가해, flow가 놓치는 정지 차량·가림 상황의 이득을 측정한다.

비교 실험은 동일 Group split에서 다음 순서를 따른다.

1. RGB CNN+BiLSTM
2. RGB + motion feature
3. RGB + motion + optional tracking

각 단계에서 frame error, 사건 검출 recall, entry/evasion/side 별 지표, video별 실패 사례, 실행 시간과 GPU 메모리를 함께 기록한다. 현재 공개 예제만으로는 모델 우열이나 YOLO 사용의 효과를 판단하지 않는다.

# M2 공개 예제 인벤토리

- 실행일: 2026-08-26
- 대상: `development/data/raw/public_example`
- 목적: 전체 데이터가 제공될 때 사용할 해시·영상 메타데이터 인벤토리 도구 검증

## 실행 명령

```bash
PYTHONPATH=src python3 scripts/data/build_inventory.py \
  data/raw/public_example \
  --csv data/metadata/public_example_inventory.csv \
  --summary data/metadata/public_example_summary.json
```

## 결과

- 영상 수: 20
- 총 크기: 195,351,697바이트
- 첫 프레임 디코딩 성공: 20
- 디코딩 실패: 0
- 동일 콘텐츠 해시 그룹: 5

동일 콘텐츠 그룹은 각각 `stage1/original/00000N.mp4`와 `stage2/videos/00000N.mp4`의 쌍이다. Stage간 공개 예제가 동일 원본을 재사용한다는 뜻이다. 전체 데이터에서는 파일명 대신 콘텐츠 해시와 촬영 원본 식별자를 기준으로 분할 누수를 검사해야 한다.

Stage 3 파일은 컨테이너 FPS와 라벨 시간축이 일치하지 않으므로 인벤토리의 FPS를 실제 평가 샘플링 규칙으로 해석하지 않는다.

## 산출물

- `data/metadata/public_example_inventory.csv`
- `data/metadata/public_example_summary.json`

## 남은 작업

- 전체 학습데이터가 확보되면 같은 도구로 인벤토리를 다시 생성한다.
- 장치, 촬영 세션, 원본·파생 관계 메타데이터를 추가한다.
- 그룹 분할은 위 메타데이터와 콘텐츠 해시를 함께 사용한다.

# Evaluation dataset guide

이 폴더는 검색 정책을 같은 기준으로 비교하기 위한 사람 작성 평가셋이다. 현재
CSV/JSON은 의도적으로 비어 있다. 실제 이미지나 관련도 판단을 만들어 넣지 않는다.

## Query manifest

`query_manifest.csv`의 한 행이 한 query다. 초기 목표는 20~30개이며 카테고리와
난이도, 장면 유형을 가능한 한 고르게 구성한다. 잘 검색되는 상품 사진만 고르지 않고
착용 사진, 다중 의류, 복잡한 배경, 가림이 있는 사례도 포함한다.

- `image_path`: 이 폴더 기준 상대 경로. 권장 형식은 `queries/Q001.jpg`.
- `category`: 서비스가 사용하는 query 카테고리.
- `difficulty`: `easy`, `medium`, `hard`.
- `scene_type`: `clean_product`, `person_wearing`, `multi_item`,
  `complex_background`, `occluded`.
- `split`: 파라미터를 고르는 `dev` 또는 마지막 확인용 `holdout`.
- `important_features`: 검색 결과에서 보존되길 기대하는 특징. `|`로 구분한다.

정확히 30개나 20/10 split을 강제하지 않는다. 이는 데이터 수집이 진행된 뒤 설명
가능한 균형을 잡기 위한 목표이며, 작은 포트폴리오 프로젝트에서 숫자를 맞추려고
부적절한 query를 채우는 것보다 실제 사용 사례를 대표하는 것이 중요하다.

## Relevance labels

`labels.json`은 query별 후보 상품의 관련도를 저장한다.

- `0`: 관련 없음 — 카테고리나 핵심 형태가 다름.
- `1`: 관련 있음 — 카테고리와 실루엣이 유사하지만 핵심 디테일 일부가 다름.
- `2`: 매우 관련 있음 — 카테고리, 실루엣, 주요 디자인 특징이 강하게 일치함.

판단 순서는 `카테고리 → 실루엣 → 디자인 디테일 → 패턴 → 색상`을 출발점으로
사용한다. 색상만 같다는 이유로 높은 관련도를 주지 않는다. 애매한 사례는 메모로
숨기기보다 팀/작성자가 기준을 다시 설명할 수 있을 때까지 동일 규칙으로 판단한다.

Ground truth는 전체 카탈로그를 전수 라벨링한 것이 아니라 여러 검색 설정의 상위
결과를 합친 candidate pool을 라벨링한다. 따라서 **pooled, non-exhaustive ground
truth**이며 pool 밖의 관련 상품을 놓칠 수 있다. 이후 후보 수집기가 생기면 어떤
실험에서 후보가 들어왔는지 추적하는 필드는 TODO로 확장한다.

## Validation

실험 실행 시에는 기본값으로 이미지 존재까지 확인한다.

```python
from evaluation.dataset import load_dataset

dataset = load_dataset("evaluation/dataset")
```

이미지를 복사하기 전 CSV/JSON 형식만 확인할 때만 `require_images=False`를 사용한다.

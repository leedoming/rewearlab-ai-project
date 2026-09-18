# Milestone 3 — Dataset Support: Evidence Record

**Branch:** `feat/m3-dataset-support` (created from merged `main` at `5dc2c77`)
**PR:** https://github.com/leedoming/rewearlab-ai-project/pull/3

## 1. Objective

검색 정책을 동일한 query/ground truth로 비교할 수 있도록 평가 데이터의 형식과
로더를 만든다. 이번 milestone은 실제 평가 이미지를 수집하거나 검색 결과를
라벨링하지 않는다. 실제로 판단하지 않은 relevance나 실험 결과를 채우는 대신,
사람이 작성할 빈 템플릿과 오류를 조기에 잡는 검증 경계를 제공한다.

## 2. Design decisions

### CSV query manifest + JSON relevance labels

- Query 메타데이터는 행 단위로 보고 수정하기 쉬운 CSV를 사용한다.
- 상품 후보 수가 query마다 다르고 0/1/2 relevance가 중첩되므로 labels는 JSON으로
  분리한다.
- loader 결과는 immutable dataclass로 반환해 이후 runner가 원본 작성 데이터를
  실수로 변경하지 않게 한다.

SQLite나 데이터베이스를 사용하지 않은 이유는 초기 목표가 20~30 query인 작은
수동 평가셋이기 때문이다. 현 단계에서는 검토 가능한 텍스트 파일이 운영 복잡도보다
중요하다.

### 결과 신뢰성에 영향을 주는 검증만 강제

다음 오류는 실험 결과를 왜곡하거나 재현을 막으므로 즉시 실패시킨다.

- 중복 query/product ID
- 허용되지 않은 difficulty, scene type, dev/holdout split
- 0/1/2 밖의 relevance
- manifest에 없는 query의 label
- dataset 폴더 밖을 가리키는 image path
- 실제 실험 실행 시 존재하지 않는 query image

반면 정확히 30개, 카테고리별 고정 개수, 20/10 split은 강제하지 않는다. 이는 좋은
초기 구성 목표지만 숫자를 맞추기 위해 대표성이 낮은 query를 채우는 것이 더 나쁘다.
데이터를 채운 뒤 분포를 출력하는 summary 검사는 TODO로 남긴다.

### Authoring과 실행 검증 분리

`load_dataset(..., require_images=False)`는 이미지 복사 전 스키마 작성 단계에만 쓴다.
실험 runner는 기본값 `True`를 사용해 누락 이미지를 조용히 건너뛰지 못하게 한다.

## 3. Relevance and pooling policy

Relevance는 `0=not relevant`, `1=relevant`, `2=highly relevant`로 저장한다. 판단의
출발점은 `category → silhouette → design detail → pattern → color`다. 색상 하나만
같은 결과가 높은 점수를 받지 않도록 README에 명시했다.

Ground truth는 전체 catalog 전수 라벨이 아니라 여러 retrieval 설정의 top results를
합친 candidate pool 기반이다. 따라서 non-exhaustive하며 pool 밖의 relevant item을
놓칠 수 있다. 이 제한은 labels의 `pooled_ground_truth: true`와 README에 명시했다.

후보가 어느 experiment에서 들어왔는지 저장하는 provenance는 candidate pooling을
실제로 구현하는 milestone에서 추가할 TODO다. 아직 존재하지 않는 E0~E4 결과 구조를
이번 schema에 추측으로 고정하지 않았다.

## 4. Files

- `evaluation/dataset/query_manifest.csv`: 빈 query manifest 템플릿
- `evaluation/dataset/labels.json`: 빈 `eval-v1` pooled labels 템플릿
- `evaluation/dataset/labels.schema.json`: 편집기/사람 검토용 JSON Schema
- `evaluation/dataset/loader.py`: third-party dependency 없는 loader와 validation
- `evaluation/dataset/README.md`: 수집·라벨링 기준과 한계
- `tests/unit/test_dataset_loader.py`: 정상 로드와 핵심 invalid cases

## 5. Validation

Repository root:

```bash
python -m compileall -q main/evaluation main/tests
```

Result: exit 0, no output.

From `main/`:

```bash
python -m pytest tests/unit -q
```

Initial result: `100 passed in 0.60s`.
Result after review fixes: `105 passed in 0.76s`.

실제 query image/label은 아직 없으므로 dataset quality나 retrieval 성능 수치는 없다.

## 6. Known limitations / TODO

- 실제 20~30 query 수집과 dev/holdout 배정은 사람이 해야 한다.
- 후보 pooling과 source-experiment provenance 기록은 baseline/experiment runner가 생긴
  뒤 추가한다.
- category/difficulty/scene 분포 summary는 실제 데이터가 채워진 뒤 추가한다.
- 라벨러 간 일치도 측정은 단일 작성자의 포트폴리오 규모에서는 필수가 아니다. 두 명
  이상이 라벨링하게 되면 그때 TODO로 추가한다.
- Exact duplicate 방지는 이미지 해시를 계산하는 수집 도구가 생기기 전까지 수동
  확인 항목이다.

## 7. Review round 1

PR review에서 실제 평가값을 조용히 왜곡할 수 있는 네 가지 경로를 확인했다.

- manifest query에 labels entry가 없어도 빈 ground truth로 로드됨
- category 오타가 검증 없이 통과함
- product ID 앞뒤 공백 때문에 retrieval ID와 매칭이 실패할 수 있음
- frozen dataclass 내부의 labels dict는 여전히 변경 가능함

모두 확인된 문제로 판단해 수정했다. 모든 manifest query는 labels에 명시적으로
존재해야 하며, truly-empty ground truth도 `[]`로 의도를 표현한다. Category는 별도
목록을 중복 작성하지 않고 `retrieval.config.COLLECTION_NAMES`를 재사용한다. Product
ID는 trim 후 중복 검사/저장하고, labels는 `MappingProxyType`으로 노출해 실제로
불변성을 보장한다. 추가로 labels JSON 최상위가 object가 아닐 때 일관된
`DatasetValidationError`를 반환하도록 보강했다.

각 경로에 unit test를 추가했으며 전체 suite는 105개가 통과했다.

## 8. Commit SHAs

```text
99f07d2  feat: add evaluation dataset support
bd8ceb5  docs: add milestone 3 evidence record
efdbe72  fix: harden dataset validation
```

이 리뷰 기록 업데이트는 다음 문서 커밋에 포함된다.

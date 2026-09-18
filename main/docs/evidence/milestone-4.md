# Milestone 4 — E0 RAW Baseline Runner: Evidence Record

**Branch:** `feat/m4-raw-baseline` (created from merged `main` at `de1562e`)
**PR:** TODO — branch push 후 생성

## 1. Objective

동일 평가셋에서 이후 bbox/category 실험의 기준점이 될 E0 RAW baseline을 실행하고,
query별 결과와 metric, aggregate summary를 재현 가능한 파일로 저장한다. 이번 단계는
실행 구조를 구현하는 milestone이며 실제 query image와 live ChromaDB가 없으므로
성능 숫자를 생성하지 않는다.

## 2. Baseline definition

```text
RAW RGB query image
→ embedding model default transform
→ embedding
→ every configured collection Top-K
→ raw Chroma distance global ranking
→ global Top-K
```

Detection, bbox crop, category filter는 사용하지 않는다. `preprocessing.mode=raw`,
`top_k>=10`, `dedupe=true`를 runner가 검증해 E0라는 이름으로 다른 설정이 실행되는
것을 막는다. Top-K를 최소 10으로 강제하는 이유는 이 runner가 NDCG@10,
Precision@10, Recall@10을 계산하기 때문이다.

중복 제거를 켠 이유는 같은 product ID가 여러 collection에서 반복될 경우 동일한
관련 상품이 여러 번 gain을 얻어 ranking metric이 왜곡되는 것을 막기 위해서다.

## 3. Dev / holdout policy

기본 config는 `split=dev`다. 파라미터와 정책을 고르는 동안 holdout 결과를 반복해서
보는 leakage를 피하기 위해서다. 최종 확인이 필요할 때만 config를 명시적으로
`holdout`으로 바꾼다. 작은 프로젝트이므로 별도 권한 시스템까지 만들지 않고,
config와 output 기록으로 의도를 추적한다.

## 4. Metrics and aggregation

Query 결과의 product ID를 pooled labels의 0/1/2 relevance와 매칭해 다음을 계산한다.

- Precision@5 / Precision@10
- Recall@5 / Recall@10
- MRR / NDCG@10
- Incompatible Category Rate@10
- Relevant Exclusion Rate (E0에는 filter가 없으므로 relevant query에서 0)

Recall처럼 undefined인 `None`은 aggregate에서 제외하며, 각 metric에 실제 포함된
query `count`를 함께 기록한다. 전체 mean/median/population stddev와 category,
difficulty, scene_type slice를 저장한다.

## 5. Outputs

- `query_results.jsonl`: config, RAW preprocessing metadata, raw-distance results, metrics
- `metrics_by_query.csv`: 사람이 비교하기 쉬운 query별 metric table
- `summary.json`: dataset version, config, overall/sliced aggregate

실제 collection distance metric은 실행 시 collection metadata의 `hnsw:space`를 읽어
config에 기록한다. 명시되지 않은 collection은 추측으로 `l2`라 쓰지 않고
`not_explicitly_declared`로 기록한다.

## 6. Test strategy and validation

실제 모델/DB 대신 retrieval callable을 주입해 ranking→metric→serialization 전체를
검증했다. 테스트는 pooled label 매칭, graded NDCG, category incompatibility,
undefined metric 제외, slice aggregate, dev-only 실행, 세 산출물, E0 통제값 검증을
포함한다.

Repository root:

```bash
python -m compileall -q main/evaluation main/tests
```

Result: exit 0, no output.

From `main/`:

```bash
python -m pytest tests/unit -q
```

Result: `113 passed in 0.72s`.

## 7. Real execution command

데이터와 DB가 준비된 환경에서 `main/` 기준:

```bash
python -m evaluation.run_experiment \
  --dataset-dir evaluation/dataset \
  --output-dir evaluation/results/E0 \
  --config evaluation/configs/e0_raw.json
```

Azure ChromaDB는 `--host`와 필요 시 `--port`를 추가한다. 이 명령은 현재 환경에서
실행하지 않았다. Query manifest가 비어 있고 torch/chromadb/open_clip이 설치되지
않았기 때문이다.

## 8. Known limitations / TODO

- 실제 E0 수치는 query 수집·라벨링과 DB 연결 후 생성한다.
- DB snapshot/version과 git commit을 자동 manifest로 기록하는 기능은 실제 첫 run
  전에 추가할 TODO다. 현재는 config와 dataset version만 output에 포함된다.
- Collection metadata가 distance metric을 명시하지 않으면 실제 Chroma 설정을 운영
  환경에서 별도로 확인해야 한다.
- 실패 taxonomy와 poor-query extraction은 Milestone 8 범위다.
- Experiment 간 delta/paired comparison은 비교 대상 E1+가 생긴 뒤 추가한다.

## 9. Commit SHAs

```text
bac82ae  feat: add E0 raw baseline runner
```

이 Evidence 문서는 다음 문서 커밋에 추가된다. PR URL과 Evidence 커밋 SHA는 PR
생성 후 갱신한다.

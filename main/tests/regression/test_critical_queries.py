"""Critical query regression checks (IMPLEMENTATION_SPEC.md section 57).

Section 57's own example is a real production case: "Q017 first relevant
rank <= 5" -- an assertion pinned to a specific query the team already
knows matters. No such query exists yet in this project: the evaluation
dataset (`main/evaluation/dataset/labels.json`) is still an empty scaffold
(`{"queries": {}}`), so there is no real "Q017" to point at. Inventing one
would mean fabricating a critical-query result this project's own
discipline (see every milestone's evidence doc) explicitly forbids.

This file instead demonstrates `evaluation.regression`'s critical-query
helpers (`first_relevant_rank`/`passes_critical_query_rank`) against a
synthetic, clearly-fictional query ID, so the mechanism is tested and ready
to receive real critical queries the moment the dataset has some -- see
docs/evidence/milestone-10.md section 3.
"""

from evaluation.regression import first_relevant_rank, passes_critical_query_rank


def result(product_id):
    return {"product_id": product_id}


def test_synthetic_critical_query_within_rank_bound():
    # Stand-in for a real "CRITICAL-QUERY-EXAMPLE first relevant rank <= 5"
    # assertion -- not a real project finding, see module docstring.
    results = [result("X1"), result("X2"), result("P_TARGET")]
    labels = {"P_TARGET": 2}

    rank = first_relevant_rank(results, labels)

    assert rank == 3
    assert passes_critical_query_rank(rank, max_rank=5)


def test_synthetic_critical_query_beyond_rank_bound_fails():
    results = [result(f"X{i}") for i in range(6)] + [result("P_TARGET")]
    labels = {"P_TARGET": 2}

    rank = first_relevant_rank(results, labels)

    assert rank == 7
    assert not passes_critical_query_rank(rank, max_rank=5)

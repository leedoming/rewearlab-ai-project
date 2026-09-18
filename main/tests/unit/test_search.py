"""Tests for retrieval.search: collection failure isolation/logging,
per-collection diagnostics, and dedupe=True/False behavior.

Uses a fake ChromaDB client/collection (dependency-injected via the
`client` parameter) — never imports the real chromadb package.
"""

import logging

import pytest

from retrieval.search import search_collections


class FakeCollection:
    def __init__(self, query_result=None, count=0, raise_on_query=False):
        self._query_result = query_result or {"metadatas": [[]], "distances": [[]]}
        self._count = count
        self.raise_on_query = raise_on_query

    def count(self):
        return self._count

    def query(self, **kwargs):
        if self.raise_on_query:
            raise RuntimeError("simulated collection failure")
        return self._query_result


class FakeClient:
    def __init__(self, collections):
        self._collections = collections

    def get_collection(self, name, embedding_function=None):
        return self._collections[name]


def make_query_result(ids_and_distances):
    metadatas = [{"id": product_id, "product_id": product_id} for product_id, _ in ids_and_distances]
    distances = [distance for _, distance in ids_and_distances]
    return {"metadatas": [metadatas], "distances": [distances]}


def test_failed_collection_is_isolated_and_logged(caplog):
    good = FakeCollection(query_result=make_query_result([("P1", 0.1)]), count=5)
    bad = FakeCollection(raise_on_query=True, count=3)
    client = FakeClient({"good": good, "bad": bad})

    with caplog.at_level(logging.ERROR, logger="retrieval.search"):
        results = search_collections(client, ["good", "bad"], query_embedding=[0.0], top_k=10)

    # Good collection's results still come back...
    assert [r["product_id"] for r in results] == ["P1"]
    # ...and the failure was logged with the collection name, not swallowed silently.
    assert any("bad" in record.message for record in caplog.records)


def test_search_collection_logs_item_count_diagnostic(caplog):
    collection = FakeCollection(query_result=make_query_result([("P1", 0.1)]), count=42)
    client = FakeClient({"top": collection})

    with caplog.at_level(logging.INFO, logger="retrieval.search"):
        search_collections(client, ["top"], query_embedding=[0.0], top_k=10)

    assert any("top" in record.message and "42" in record.message for record in caplog.records)


def test_dedupe_true_collapses_duplicate_product_ids_across_collections():
    # P1 appears in both collections at different distances; P1's best (smallest)
    # distance should be the one that determines its rank once deduped.
    coll_a = FakeCollection(query_result=make_query_result([("P1", 0.5), ("P2", 0.9)]))
    coll_b = FakeCollection(query_result=make_query_result([("P1", 0.2)]))
    client = FakeClient({"a": coll_a, "b": coll_b})

    results = search_collections(client, ["a", "b"], query_embedding=[0.0], top_k=10, dedupe=True)

    product_ids = [r["product_id"] for r in results]
    assert product_ids.count("P1") == 1
    assert set(product_ids) == {"P1", "P2"}
    # Sorted by raw_distance ascending -> P1 (0.2, from collection b) beats P2 (0.9).
    assert product_ids[0] == "P1"
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))


def test_dedupe_false_keeps_duplicate_product_ids():
    coll_a = FakeCollection(query_result=make_query_result([("P1", 0.5)]))
    coll_b = FakeCollection(query_result=make_query_result([("P1", 0.2)]))
    client = FakeClient({"a": coll_a, "b": coll_b})

    results = search_collections(client, ["a", "b"], query_embedding=[0.0], top_k=10, dedupe=False)

    product_ids = [r["product_id"] for r in results]
    assert product_ids.count("P1") == 2
    # Still globally ranked by raw_distance ascending across both collections.
    assert results[0]["raw_distance"] == pytest.approx(0.2)
    assert results[1]["raw_distance"] == pytest.approx(0.5)
    assert [r["rank"] for r in results] == [1, 2]


def test_dedupe_true_and_false_both_respect_top_k_truncation():
    coll = FakeCollection(query_result=make_query_result([("P1", 0.1), ("P2", 0.2), ("P3", 0.3)]))
    client = FakeClient({"only": coll})

    deduped = search_collections(client, ["only"], query_embedding=[0.0], top_k=2, dedupe=True)
    not_deduped = search_collections(client, ["only"], query_embedding=[0.0], top_k=2, dedupe=False)

    assert len(deduped) == 2
    assert len(not_deduped) == 2
    assert [r["rank"] for r in deduped] == [1, 2]
    assert [r["rank"] for r in not_deduped] == [1, 2]

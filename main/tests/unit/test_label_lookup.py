from evaluation.label_lookup import count_relevant_excluded_by_filter, resolve_label_collections


class FakeCollection:
    def __init__(self, ids_present):
        self._ids_present = set(ids_present)

    def get(self, ids):
        found = [pid for pid in ids if pid in self._ids_present]
        return {"ids": found}


class FakeClient:
    def __init__(self, collections):
        self._collections = collections

    def get_collection(self, name):
        return self._collections[name]


def test_resolve_label_collections_finds_ids_across_collections():
    client = FakeClient(
        {
            "top": FakeCollection(["P1"]),
            "outer": FakeCollection(["P2"]),
            "pants": FakeCollection([]),
        }
    )

    result = resolve_label_collections(client, ["P1", "P2", "P3"], ["top", "outer", "pants"])

    assert result == {"P1": "top", "P2": "outer"}


def test_resolve_label_collections_stops_once_everything_is_found():
    calls = []

    class CountingCollection(FakeCollection):
        def get(self, ids):
            calls.append(list(ids))
            return super().get(ids)

    client = FakeClient(
        {
            "top": CountingCollection(["P1"]),
            "outer": CountingCollection(["P2"]),
            "pants": CountingCollection(["P3"]),
        }
    )

    resolve_label_collections(client, ["P1"], ["top", "outer", "pants"])

    # Only the first collection needed to be queried once P1 was found.
    assert calls == [["P1"]]


def test_resolve_label_collections_omits_ids_not_found_anywhere():
    client = FakeClient({"top": FakeCollection([])})
    result = resolve_label_collections(client, ["ghost"], ["top"])
    assert result == {}


def test_count_relevant_excluded_by_filter_only_counts_relevant_and_excluded():
    labels = {"P1": 2, "P2": 1, "P3": 0, "P4": 1}
    label_collections = {"P1": "top", "P2": "outer", "P3": "pants", "P4": "dress_skirts"}

    # Filter only searched "top": P2 (relevant, excluded) and P4 (relevant,
    # excluded) count; P1 (relevant, but searched) doesn't; P3 (not
    # relevant) never counts regardless of exclusion.
    excluded = count_relevant_excluded_by_filter(labels, label_collections, searched_collections=["top"])
    assert excluded == 2


def test_count_relevant_excluded_by_filter_ignores_unresolved_products():
    # "P1" is relevant but its true collection couldn't be resolved (e.g.
    # removed from the DB) -- that's a DB-coverage question, not a filter
    # exclusion, so it must not be counted either way.
    labels = {"P1": 2}
    excluded = count_relevant_excluded_by_filter(labels, label_collections={}, searched_collections=["top"])
    assert excluded == 0


def test_count_relevant_excluded_by_filter_zero_when_nothing_filtered_out():
    labels = {"P1": 2, "P2": 1}
    label_collections = {"P1": "top", "P2": "outer"}
    excluded = count_relevant_excluded_by_filter(
        labels, label_collections, searched_collections=["pants", "top", "outer", "dress_skirts"]
    )
    assert excluded == 0

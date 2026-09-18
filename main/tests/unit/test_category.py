import pytest

from retrieval.category import (
    get_allowed_labels,
    get_category_from_filename,
    get_collection_name,
    get_filtered_collections,
    is_label_allowed_for_category,
)


def test_get_collection_name_known_category():
    assert get_collection_name("바지") == "pants"
    assert get_collection_name("상의") == "top"
    assert get_collection_name("아우터") == "outer"
    assert get_collection_name("원피스_스커트") == "dress_skirts"


def test_get_collection_name_unknown_category():
    assert get_collection_name("모자") == "unknown"


def test_get_allowed_labels():
    assert get_allowed_labels("바지") == ["bottom"]
    assert get_allowed_labels("상의") == ["top", "outer"]
    assert get_allowed_labels("모자") == []


def test_get_allowed_labels_accepts_collection_names_too():
    # The evaluation dataset (main/evaluation/dataset) labels queries with a
    # ChromaDB collection name (e.g. "outer"), not the Musinsa category
    # string ("아우터") -- category-based bbox selection must resolve both,
    # or category_confidence/category_largest would silently see zero
    # compatible labels for every dataset query and always fall back.
    assert get_allowed_labels("pants") == ["bottom"]
    assert get_allowed_labels("top") == ["top", "outer"]
    assert get_allowed_labels("outer") == ["top", "outer"]
    assert get_allowed_labels("dress_skirts") == ["bottom", "dress"]
    assert get_allowed_labels("not_a_real_collection") == []


def test_is_label_allowed_for_category():
    assert is_label_allowed_for_category("outer", "상의") is True
    assert is_label_allowed_for_category("bottom", "상의") is False


def test_get_filtered_collections_none_policy_searches_everything():
    assert get_filtered_collections("top", "none") == ["pants", "top", "outer", "dress_skirts"]
    # "none" ignores category entirely, even an unknown one.
    assert get_filtered_collections("not_a_category", "none") == ["pants", "top", "outer", "dress_skirts"]


def test_get_filtered_collections_hard_policy_is_exact_match_only():
    assert get_filtered_collections("top", "hard") == ["top"]
    assert get_filtered_collections("pants", "hard") == ["pants"]


def test_get_filtered_collections_soft_policy_allows_adjacent_categories():
    assert get_filtered_collections("top", "soft") == ["top", "outer"]
    assert get_filtered_collections("outer", "soft") == ["outer", "top"]
    # pants/dress_skirts have no adjacency in the documented soft mapping.
    assert get_filtered_collections("pants", "soft") == ["pants"]


def test_get_filtered_collections_unknown_category_falls_back_to_no_filter():
    # An unrecognized category means "we don't know", not "search nothing".
    assert get_filtered_collections("not_a_category", "hard") == ["pants", "top", "outer", "dress_skirts"]
    assert get_filtered_collections("not_a_category", "soft") == ["pants", "top", "outer", "dress_skirts"]


def test_get_filtered_collections_rejects_unknown_policy():
    with pytest.raises(ValueError):
        get_filtered_collections("top", "extreme")


def test_get_category_from_filename():
    assert get_category_from_filename("musinsa_바지.json") == "바지"
    assert get_category_from_filename("/some/dir/musinsa_상의.json") == "상의"
    assert get_category_from_filename("musinsa_모자.json") is None

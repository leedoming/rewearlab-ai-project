from retrieval.category import (
    get_allowed_labels,
    get_category_from_filename,
    get_collection_name,
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


def test_is_label_allowed_for_category():
    assert is_label_allowed_for_category("outer", "상의") is True
    assert is_label_allowed_for_category("bottom", "상의") is False


def test_get_category_from_filename():
    assert get_category_from_filename("musinsa_바지.json") == "바지"
    assert get_category_from_filename("/some/dir/musinsa_상의.json") == "상의"
    assert get_category_from_filename("musinsa_모자.json") is None

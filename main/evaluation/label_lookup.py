"""Resolve which ChromaDB collection each labeled ground-truth product lives in.

Needed to honestly compute `relevant_exclusion_rate` for the category-filter
experiments (E5/E6): a labeled relevant item that never appears in a
filtered search's results might be genuinely dissimilar to the query, OR it
might live in a collection the filter excluded before search ever ran.
Distinguishing those two cases requires knowing each labeled product's true
collection -- which `main/evaluation/dataset/labels.json` does not record
(IMPLEMENTATION_SPEC.md sections 11-12 only specify product_id + relevance).
Rather than extend the Milestone 3 dataset schema for this, this module
resolves it directly from ChromaDB metadata (a cheap `.get(ids=...)` lookup,
not a similarity search).
"""


def resolve_label_collections(client, product_ids, collection_names):
    """Return {product_id: collection_name} for every id actually found.

    Product ids that aren't found in any collection are omitted (the
    ground truth may reference an item that's since been removed from the
    DB -- a database-coverage issue, not something this function should
    guess about).
    """
    found = {}
    remaining = list(dict.fromkeys(product_ids))  # de-duplicate, keep order
    for collection_name in collection_names:
        if not remaining:
            break
        collection = client.get_collection(name=collection_name)
        try:
            fetched = collection.get(ids=remaining)
        except Exception:
            continue
        for product_id in fetched.get("ids", []):
            found[product_id] = collection_name
        remaining = [pid for pid in remaining if pid not in found]
    return found


def count_relevant_excluded_by_filter(labels, label_collections, searched_collections):
    """How many relevant (grade >= 1) labeled items were excluded by a
    category filter, i.e. their known collection isn't in
    `searched_collections`.

    Items whose collection couldn't be resolved (missing from
    `label_collections`) are never counted here -- that's a DB-coverage
    question, not a filter-exclusion one; counting them would conflate the
    two failure modes IMPLEMENTATION_SPEC.md section 28 keeps distinct.
    """
    searched = set(searched_collections)
    return sum(
        1
        for product_id, grade in labels.items()
        if grade >= 1
        and product_id in label_collections
        and label_collections[product_id] not in searched
    )

"""Aggregate per-query metrics without hiding undefined values."""

from statistics import mean, median, pstdev


def summarize_values(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "stddev": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "stddev": pstdev(values),
    }


def aggregate_metrics(rows):
    """Return overall and category/difficulty/scene_type metric summaries."""
    metric_names = sorted({name for row in rows for name in row["metrics"]})

    def summarize(group):
        return {
            name: summarize_values([row["metrics"].get(name) for row in group])
            for name in metric_names
        }

    summary = {"query_count": len(rows), "overall": summarize(rows)}
    for field in ("category", "difficulty", "scene_type"):
        values = sorted({row[field] for row in rows})
        summary[f"by_{field}"] = {
            value: summarize([row for row in rows if row[field] == value])
            for value in values
        }
    return summary

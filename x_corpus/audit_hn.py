"""Paired robustness checks on existing AI stance ratings, with coding choices explicit."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean


def read(model):
    return {
        row["i"]: row
        for row in json.loads(Path(f"results/stance_ratings_{model}.json").read_text())
    }


def category(v):
    return "null" if v is None else "zero" if v == 0 else "direction"


a, b = read("gpt-5.4-mini"), read("gpt-4.1")
assert a.keys() == b.keys()
matrix = Counter()
conditions = {}
for idx, x in a.items():
    y = b[idx]
    assert x["text"] == y["text"] and x["bin"] == y["bin"] and x["ok"] and y["ok"]
    m, g = x["stance"], y["stance"]
    matrix[f"{category(m)} / {category(g)}"] += 1
    for rule, keep in [
        ("both_numeric", m is not None and g is not None),
        ("both_directional", bool(m) and bool(g)),
        ("treat_missing_as_zero", True),
    ]:
        if keep:
            conditions.setdefault(rule, {}).setdefault(x["label"], []).append(
                ((m or 0) + (g or 0)) / 2
            )
out = {
    "matrix_mini_then_4_1": matrix,
    "by_rule": {
        rule: {k: {"n": len(v), "mean": mean(v)} for k, v in bins.items()}
        for rule, bins in conditions.items()
    },
    "example_ids": [467, 1174],
}
Path("x_corpus/hn_audit.json").write_text(json.dumps(out, indent=2) + "\n")

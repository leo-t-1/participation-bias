"""Run from any directory: python3 path/to/breadth_experiment/run.py"""
from collections import Counter, defaultdict
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"


def read_days():
    days = []
    seen = set()
    for day in range(14):
        rows = []
        for item in json.loads((DATA / f"day{day:02d}.json").read_text()):
            cid = item["objectID"]
            if cid in seen:
                raise ValueError(f"duplicate comment {cid}")
            seen.add(cid)
            if item.get("story_id") and item.get("author"):
                rows.append((item["story_id"], item["author"], item.get("story_title")))
        days.append(rows)
    return days, len(seen)


def summarize(days):
    authors = defaultdict(Counter)
    titles = {}
    for day in days:
        for story, author, title in day:
            authors[story][author] += 1
            if title:
                titles[story] = title
    scores = {
        "comments": {s: sum(a.values()) for s, a in authors.items()},
        "people": {s: len(a) for s, a in authors.items()},
        "cap_two": {s: sum(min(n, 2) for n in a.values()) for s, a in authors.items()},
    }
    # Use the same ascending story ID to break ties under each counting rule.
    orders = {k: sorted(v, key=lambda s: (-v[s], s)) for k, v in scores.items()}
    ranks = {k: {s: i for i, s in enumerate(order, 1)} for k, order in orders.items()}
    def describe(s):
        return dict(story=s, title=titles.get(s),
                    **{k: scores[k][s] for k in scores},
                    **{k + "_rank": ranks[k][s] for k in ranks},
                    most_active_two=sum(n for _, n in authors[s].most_common(2)))
    overlap = {}
    for k in (10, 20, 50):
        original = set(orders["comments"][:k])
        overlap[k] = {m: len(original.intersection(orders[m][:k])) for m in ("cap_two", "people")}
    # Strongest strict inversion between a story in each of the two top-20 lists.
    # The product measures the size of the reversal on a ratio scale.
    pairs = [(a,b) for a in orders["comments"][:20] for b in orders["people"][:20]
             if scores["comments"][a] > scores["comments"][b] and scores["people"][a] < scores["people"][b]]
    pair = max(pairs, key=lambda p: (scores["comments"][p[0]] / scores["comments"][p[1]]
                                     * scores["people"][p[1]] / scores["people"][p[0]]), default=None)
    # Among high-volume stories within one comment of each other, choose the
    # pair with the largest proportional gap in distinct authors.
    top = orders["comments"][:20]
    near_pairs = [(a,b) for i,a in enumerate(top) for b in top[i+1:]
                  if abs(scores["comments"][a] - scores["comments"][b]) <= 1]
    near = max(near_pairs, key=lambda p: max(scores["people"][s] for s in p)
               / min(scores["people"][s] for s in p), default=None)
    return dict(matched_volume=[describe(s) for s in near] if near else [], stories=len(authors), comments=sum(scores["comments"].values()),
                overlap=overlap, example=[describe(s) for s in pair] if pair else [],
                cutoff_scores={key:[scores[key][s] for s in order[19:21]] for key,order in orders.items()},
                top20={key:[describe(s) for s in orders[key][:20]] for key in ("comments","people")})


if __name__ == "__main__":
    days, collected = read_days()
    out = {"input_rows": collected, "periods": {}}
    for name, sample in (("fortnight", days), ("week1", days[:7]), ("week2", days[7:])):
        out["periods"][name] = summarize(sample)
    (HERE / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for name, stats in out["periods"].items():
        print(name, stats["stories"], stats["comments"], stats["overlap"], stats["cutoff_scores"], stats["example"])

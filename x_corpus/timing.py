"""Conditional timing null, preserving each account's count and both period totals."""

import json
from pathlib import Path
import duckdb
import sys

con = duckdb.connect(config={"threads": 2, "memory_limit": "800MB"})
source = "hn" if "--hn" in sys.argv else "x"
if source == "hn":
    from collections import Counter
    from collections import defaultdict

    threads = defaultdict(set)
    periods = [Counter(), Counter()]
    for day in range(14):
        rows = json.loads(Path(f"data/day{day:02}.json").read_text())
        for x in rows:
            if x.get("author"):
                periods[day // 7][x["author"]] += 1
                threads[x["author"]].add(x.get("story_id"))
    con.execute("CREATE TABLE counts(early INTEGER,late INTEGER,n INTEGER)")
    a, b = periods
    con.executemany(
        "INSERT INTO counts VALUES (?,?,?)",
        [(a[u], b[u], a[u] + b[u]) for u in a.keys() | b.keys()],
    )
else:
    con.execute(
        """CREATE VIEW counts AS SELECT early,late,early+late n FROM 'xdata/account_counts.parquet'"""
    )
early, late = con.execute("SELECT sum(early),sum(late) FROM counts").fetchone()
total = early + late
out = {"early_posts": int(early), "late_posts": int(late), "by_total_posts": {}}
if source == "hn":
    out["two_commenters_by_story"] = {}
    for name, same in [("same_story", True), ("different_stories", False)]:
        users = [
            u
            for u in threads
            if a[u] + b[u] == 2
            and (len(threads[u]) == 1) == same
            and None not in threads[u]
        ]
        out["two_commenters_by_story"][name] = {
            "n": len(users),
            "both_weeks": sum(a[u] == 1 for u in users),
        }
# Exact sampling without replacement from the two period labels.
for n, num, split in con.execute(
    """SELECT n,count(*),countif(early>0 AND late>0)
                               FROM counts WHERE n IN (2,3,5,10,20,50) GROUP BY n ORDER BY n"""
).fetchall():

    def same_period(pop):
        p = 1.0
        for k in range(n):
            p *= (pop - k) / (total - k)
        return p

    out["by_total_posts"][n] = dict(
        accounts=num,
        both_periods=split,
        observed=split / num,
        shuffled=1 - same_period(early) - same_period(late),
    )
Path(f"x_corpus/timing_{source}.json").write_text(json.dumps(out, indent=2) + "\n")

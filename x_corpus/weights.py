"""Exact finite-corpus reweighting of late-period replies and concentration.
Read account counts built by count.py. Run from repository root.
"""

import json
from pathlib import Path
import duckdb

db = duckdb.connect(config={"threads": 2, "memory_limit": "1500MB"})
db.execute("SET temp_directory='xdata/count_cache/spill'")
db.execute(
    "CREATE VIEW users AS SELECT *, early+late AS total FROM 'xdata/account_counts.parquet'"
)


def scalar_row(sql):
    q = db.sql(sql)
    return dict(zip(q.columns, q.fetchall()[0]))


out = scalar_row(
    """SELECT count(*) AS accounts,sum(total)::BIGINT AS posts,
    avg(total) AS posts_per_account,sum(total::DOUBLE*total)/sum(total) AS posts_per_post_author,
    sum(late)::BIGINT AS late_posts,countif(late>0) AS late_accounts,
    sum(replies)::BIGINT AS late_replies,sum(replies)/sum(late) AS post_reply_share,
    avg(replies/nullif(late,0)) AS account_reply_share FROM users"""
)
top = scalar_row(
    """WITH ranked AS (SELECT total,row_number() OVER(ORDER BY total DESC, account) AS rn
    FROM users) SELECT sum(total) FILTER (WHERE rn<=?) /sum(total) AS top_decile_share FROM ranked""".replace(
        "?", str(out["accounts"] // 10)
    )
)
out.update(top)
out["bias_percentage_points"] = 100 * (
    out["post_reply_share"] - out["account_reply_share"]
)
# Exact decomposition: moving from author to post weights changes the mixture
# of early bins and changes the weighting of accounts inside each early bin.
bins = json.loads(Path("x_corpus/count_summary.json").read_text())
within = across = 0.0
for row in bins:
    p_post = row["late_posts"] / out["late_posts"]
    p_account = row["late_accounts"] / out["late_accounts"]
    within += p_post * (row["post_reply_share"] - row["author_reply_share"])
    across += (p_post - p_account) * row["author_reply_share"]
assert abs(100 * (within + across) - out["bias_percentage_points"]) < 1e-7
out.update(within_early_bins_pp=100 * within, change_in_bin_mix_pp=100 * across)
Path("x_corpus/weighting.json").write_text(json.dumps(out, indent=2) + "\n")

"""Sensitivity check: compare early-low and early-high accounts at the same captured terms."""

import json
from pathlib import Path
import duckdb

con = duckdb.connect(config={"threads": 2, "memory_limit": "1600MB"})
con.execute("SET temp_directory='xdata/term_spill'")
paths = [f"xdata/s{i:02}.parquet" for i in (0, 24, 48)]
con.execute(
    """CREATE TEMP TABLE records AS SELECT lower(username) account,search_term term,
    date>='2026-05-01' later,coalesce(len("replying-to")>0,false)::INT reply
    FROM read_parquet(?) WHERE "is-retweet"=false AND length(trim(username))>1 AND
    date>='2026-01-01' AND date<'2026-07-28'""",
    [paths],
)
con.execute(
    """CREATE TEMP TABLE rates AS SELECT account,countif(NOT later) early,countif(later) late
    FROM records GROUP BY account"""
)
# Each late-active author contributes total weight 1. A term's rate is the
# author's weighted post mass that is replies, divided by all such post mass.
con.execute(
    """CREATE TEMP TABLE cells AS SELECT term, early=1 AS low,
    sum(1.0/late) mass,sum(reply::DOUBLE/late) reply_mass,count(*) posts
    FROM records JOIN rates USING(account) WHERE later AND (early=1 OR early>=26)
    GROUP BY term,low"""
)
out = {}
for low, key in [(True, "one"), (False, "26+")]:
    mass, rate = con.execute(
        "SELECT sum(mass),sum(reply_mass)/sum(mass) FROM cells WHERE low=?", [low]
    ).fetchall()[0]
    out[key] = {"authors": round(mass), "reply_share": rate}
for threshold in [1, 5, 20]:
    # Weight each retained term by the smaller of its two relative masses.
    # This gives both groups an identical distribution over terms.
    q = con.sql(
        """WITH pairs AS (SELECT a.term,a.reply_mass/a.mass ra,b.reply_mass/b.mass rb,
        a.mass / ? AS pa,b.mass / ? AS pb,least(pa,pb) AS w FROM cells a JOIN cells b
        ON a.term=b.term WHERE a.low AND NOT b.low AND a.posts>=? AND b.posts>=?)
        SELECT count(*) terms,sum(pa) retained_low_mass,sum(pb) retained_high_mass,
        sum(w*ra)/sum(w) standardized_low,sum(w*rb)/sum(w) standardized_high
        FROM pairs""",
        params=[out["one"]["authors"], out["26+"]["authors"], threshold, threshold],
    )
    out[f"min_posts_per_group_{threshold}"] = dict(zip(q.columns, q.fetchall()[0]))
Path("x_corpus/term_sensitivity.json").write_text(json.dumps(out, indent=2) + "\n")

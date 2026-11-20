"""Build compact account counts from the pinned source, without storing full shards.
Run from the repository root. One sparse temporary file per concurrent worker.
"""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import struct
import time

import duckdb
import pyarrow.parquet as pq
import requests

REV = "254bbe4cea800035ec544cf047d8b3bebc24fe99"
BASE = f"https://huggingface.co/datasets/SinclairSchneider/tweets_sample_2026/resolve/{REV}/data"
CACHE = Path("xdata/count_cache")
COLUMNS = ("username", "date", "is-retweet", "replying-to")
QUERY = """
SELECT lower(username) AS account,
       count(*) FILTER (WHERE date < '2026-05-01')::INTEGER AS early,
       count(*) FILTER (WHERE date >= '2026-05-01')::INTEGER AS late,
       count(*) FILTER (WHERE date >= '2026-05-01' AND len("replying-to") > 0)::INTEGER AS replies
FROM read_parquet(?) WHERE "is-retweet" = false
AND length(trim(username)) > 1 AND date >= '2026-01-01' AND date < '2026-07-28'
GROUP BY 1
"""


def get_range(session, url, spec):
    for attempt in range(5):
        try:
            # Vary query with the byte range to isolate caches and signed redirects.
            r = session.get(
                url + "?part=" + spec,
                headers={"Range": "bytes=" + spec},
                timeout=(20, 100),
            )
            r.raise_for_status()
            if r.status_code != 206:
                raise ValueError("Expected partial response")
            bounds, size = r.headers["content-range"].split()[1].split("/")
            start, stop = map(int, bounds.split("-"))
            if spec.startswith("-"):
                expected = (int(size) - int(spec[1:]), int(size) - 1)
            else:
                expected = tuple(map(int, spec.split("-")))
            if (start, stop) != expected or len(r.content) != stop - start + 1:
                raise ValueError("Invalid range")
            return r.content, start, int(size)
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def process(i):
    dest = CACHE / f"counts{i:02}.parquet"
    if dest.exists():
        return
    url = f"{BASE}/train-{i:05}-of-00073.parquet"
    temp = CACHE / f"partial{i:02}.parquet"
    with requests.Session() as session:
        tail, start, size = get_range(session, url, "-65536")
        n = struct.unpack("<I", tail[-8:-4])[0]
        if n + 8 > len(tail):
            tail, start, size = get_range(session, url, f"-{n+8}")
        meta = pq.read_metadata(io.BytesIO(b"PAR1" + tail[-8 - n :]))
        with temp.open("wb") as f:
            f.write(b"PAR1")
            f.seek(start)
            f.write(tail)
            for group in range(meta.num_row_groups):
                rg = meta.row_group(group)
                for j in range(rg.num_columns):
                    col = rg.column(j)
                    if col.path_in_schema.split(".")[0] not in COLUMNS:
                        continue
                    pos = (
                        col.dictionary_page_offset
                        if col.has_dictionary_page
                        else col.data_page_offset
                    )
                    data, _, _ = get_range(
                        session, url, f"{pos}-{pos+col.total_compressed_size-1}"
                    )
                    f.seek(pos)
                    f.write(data)
    con = duckdb.connect(config={"threads": 1})
    stage = dest.with_suffix(".tmp")
    try:
        con.sql(QUERY, params=[str(temp)]).write_parquet(str(stage))
        stage.replace(dest)
    finally:
        con.close()
        temp.unlink(missing_ok=True)
    print(f"finished shard {i}", flush=True)


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(12) as pool:
        list(pool.map(process, range(73)))
    con = duckdb.connect(config={"threads": 3, "memory_limit": "1800MB"})
    con.execute("SET temp_directory='xdata/count_cache/spill'")
    con.sql(
        """SELECT account,sum(early)::INT early,sum(late)::INT late,sum(replies)::INT replies
           FROM read_parquet('xdata/count_cache/counts*.parquet') GROUP BY account"""
    ).write_parquet("xdata/account_counts.parquet")
    bins = "CASE WHEN early=0 THEN '0' WHEN early=1 THEN '1' WHEN early<=4 THEN '2-4' WHEN early<=10 THEN '5-10' WHEN early<=25 THEN '11-25' ELSE '26+' END"
    stats = con.sql(
        f"""SELECT {bins} AS bin,count(*) accounts,sum(early) early_posts,sum(late) late_posts,
               countif(late>0) late_accounts, CASE WHEN max(early)>0 THEN avg((late>0)::int) END reappearance,
               sum(replies)/sum(late) post_reply_share, avg(replies/nullif(late,0)) author_reply_share
               FROM 'xdata/account_counts.parquet' GROUP BY 1 ORDER BY min(early)"""
    )
    Path("x_corpus/count_summary.json").write_text(
        json.dumps(
            [dict(zip(stats.columns, row)) for row in stats.fetchall()],
            default=int,
            indent=2,
        )
        + "\n"
    )

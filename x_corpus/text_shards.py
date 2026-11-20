"""Fetch the three original shards and verify SHA-256 against the pinned manifest."""

import hashlib
from pathlib import Path
import sys
import time
import requests
from count import BASE, REV


def verify_or_fetch(i, meta, wait):
    path = Path(f"xdata/s{i:02}.parquet")
    if wait:
        deadline = time.monotonic() + 1500
        while (
            not path.exists() or path.stat().st_size < meta["size"]
        ) and time.monotonic() < deadline:
            time.sleep(4)

    def valid(p):
        if not p.exists() or p.stat().st_size != meta["size"]:
            return False
        with p.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest() == meta["lfs"]["oid"]

    if not valid(path):
        tmp = path.with_suffix(".download")
        with requests.get(
            f"{BASE}/train-{i:05}-of-00073.parquet", stream=True, timeout=(30, 150)
        ) as r, tmp.open("wb") as out:
            r.raise_for_status()
            for chunk in r.iter_content(2**20):
                out.write(chunk)
        if not valid(tmp):
            raise ValueError(f"Checksum mismatch for shard {i}")
        tmp.replace(path)
    print(f"Verified shard {i}", flush=True)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    Path("xdata").mkdir(exist_ok=True)
    r = requests.get(
        f"https://huggingface.co/api/datasets/SinclairSchneider/tweets_sample_2026/tree/{REV}/data",
        timeout=30,
    )
    r.raise_for_status()
    by_name = {x["path"].split("/")[-1]: x for x in r.json()}
    with ThreadPoolExecutor(3) as pool:
        list(
            pool.map(
                lambda i: verify_or_fetch(
                    i, by_name[f"train-{i:05}-of-00073.parquet"], "--wait" in sys.argv
                ),
                (0, 24, 48),
            )
        )

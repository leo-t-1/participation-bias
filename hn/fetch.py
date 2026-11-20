import json, sys, time, urllib.request, urllib.parse, datetime as dt, os, ssl
from concurrent.futures import ThreadPoolExecutor
OUT = sys.argv[1]
START = dt.datetime(2026,11,1,tzinfo=dt.timezone.utc)
DAYS = 14
ATTRS = "author,created_at_i,story_id,parent_id,comment_text,story_title"
def get(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            err = e
    raise err
def day(d):
    a = int((START+dt.timedelta(days=d)).timestamp()); b = a+86400
    rows = {}; cur = b; n=0
    while True:
        q = urllib.parse.urlencode({"tags":"comment","hitsPerPage":1000,
            "numericFilters":f"created_at_i>={a},created_at_i<{cur}",
            "attributesToRetrieve":ATTRS,"attributesToHighlight":""})
        j = get("https://hn.algolia.com/api/v1/search_by_date?"+q); n+=1
        hits = j["hits"]
        if not hits: break
        for h in hits: rows[h["objectID"]] = h
        m = min(h["created_at_i"] for h in hits)
        if len(hits) < 1000: break
        cur = m+1 if m+1 < cur else m   # include boundary second, dedupe by id
        if cur == m and all(h["created_at_i"]==m for h in hits): cur = m
    with open(os.path.join(OUT,f"day{d:02d}.json"),"w") as f: json.dump(list(rows.values()), f)
    return d, len(rows), n
with ThreadPoolExecutor(7) as ex:
    for r in ex.map(day, range(DAYS)): print(r, flush=True)

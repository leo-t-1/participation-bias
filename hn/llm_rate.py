import json, glob, re, html, sys, os, collections, urllib.request, random
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
DATA, OUT, ENV, MODEL = sys.argv[1:5]
KEY = [l.split("=",1)[1].strip() for l in open(ENV) if l.startswith("OPENAI_API_KEY=")][0]
random.seed(0)
rows = {}
for f in sorted(glob.glob(os.path.join(DATA,"day*.json"))):
    d = int(re.search(r"day(\d+)", f).group(1))
    for h in json.load(open(f)):
        if h.get("author") and h.get("comment_text"): h["day"]=d; rows[h["objectID"]]=h
rows = list(rows.values())
TAG = re.compile(r"<[^>]+>"); clean = lambda t: html.unescape(TAG.sub(" ", t)).strip()
c1 = collections.Counter(r["author"] for r in rows if r["day"]<7)
BINS = [(0,0,"0"),(1,1,"1"),(2,4,"2-4"),(5,9,"5-9"),(10,24,"10-24"),(25,10**9,"25+")]
by_user = collections.defaultdict(list)
for r in rows:
    if r["day"]>=7:
        t = clean(r["comment_text"]); w = len(t.split())
        if 20<=w<=150: by_user[r["author"]].append((t,w))
sample = []
for j,(lo,hi,l) in enumerate(BINS):
    us = [u for u in by_user if lo<=c1[u]<=hi]; random.shuffle(us)
    for u in us[:250]:
        t,w = random.choice(by_user[u]); sample.append(dict(bin=j, label=l, text=t, words=w))
random.shuffle(sample)
SYS = ("You rate the tone of single online forum comments. Return JSON with two integer fields, each 1-5. "
       "hostility: 1 = friendly or neutral, 3 = dismissive or irritated, 5 = openly hostile or contemptuous toward a person or group. "
       "assertiveness: 1 = tentative, heavily qualified, 3 = ordinary statements, 5 = categorical, presents opinion as certain fact. "
       "Judge tone toward people and certainty of claims, not whether the topic itself is negative or technical.")
def rate(s):
    body = json.dumps({"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":s["text"][:1500]}],"response_format":{"type":"json_object"}}).encode()
    for k in range(4):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                j = json.loads(json.load(r)["choices"][0]["message"]["content"])
                s["hostility"]=int(j["hostility"]); s["assertiveness"]=int(j["assertiveness"]); return s
        except Exception as e: s["err"]=str(e)[:200]
    return s
with ThreadPoolExecutor(16) as ex: sample = list(ex.map(rate, sample))
ok = [s for s in sample if "hostility" in s]
print("rated", len(ok), "of", len(sample), "| first error:", next((s["err"] for s in sample if "hostility" not in s), None))
va = SentimentIntensityAnalyzer()
for s in ok: s["vader"] = va.polarity_scores(s["text"].lower())["compound"]
b = np.array([s["bin"] for s in ok]); lw = np.log([s["words"] for s in ok]); out = {"model":MODEL,"n":len(ok)}
for k in ("hostility","assertiveness"):
    v = np.array([s[k] for s in ok], float)
    X = np.column_stack([np.ones(len(v)), b, lw]); beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    resid = v - X@beta; se = np.sqrt(np.diag(np.linalg.inv(X.T@X))*resid.var(ddof=3))
    out[k] = dict(by_bin=[dict(bin=BINS[j][2], n=int((b==j).sum()), mean=float(v[b==j].mean()), se=float(v[b==j].std(ddof=1)/np.sqrt((b==j).sum())), share_ge3=float((v[b==j]>=3).mean())) for j in range(6)],
        spearman_bin=float(stats.spearmanr(b,v)[0]), spearman_p=float(stats.spearmanr(b,v)[1]),
        slope_per_bin_controlling_length=float(beta[1]), slope_se=float(se[1]),
        spearman_with_vader=float(stats.spearmanr(v,[s["vader"] for s in ok])[0]))
json.dump(out, open(os.path.join(OUT,"llm_validation.json"),"w"), indent=1)
json.dump(ok, open(os.path.join(OUT,"llm_ratings.json"),"w"))
print(json.dumps(out, indent=1))

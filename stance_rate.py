"""Stance experiment. Hypothesis (fixed before running): on AI threads, heavier posters hold more extreme
stances on AI (further from neutral) than light posters. Posting rate = week-1 comments (all topics);
stance = LLM rating of one week-2 comment on an AI thread per user, blind to author.
python3 stance_rate.py data results .env <model>"""
import json, glob, re, html, sys, os, collections, urllib.request, random
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
DATA, OUT, ENV, MODEL = sys.argv[1:5]
KEY = [l.split("=",1)[1].strip() for l in open(ENV) if l.startswith("OPENAI_API_KEY=")][0]
random.seed(0)
AI = re.compile(r"\b(ai|a\.i\.|llm|llms|gpt|chatgpt|claude|openai|anthropic|gemini|copilot|agents?|agentic|machine learning|neural|deepmind|mistral|llama|transformer|diffusion|agi)\b", re.I)
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
    if r["day"]>=7 and AI.search(r.get("story_title") or ""):
        t = clean(r["comment_text"]); w = len(t.split())
        if 20<=w<=150: by_user[r["author"]].append((t,w,r.get("story_title")))
sample = []
for j,(lo,hi,l) in enumerate(BINS):
    us = [u for u in by_user if lo<=c1[u]<=hi]; random.shuffle(us)
    for u in us[:300]:
        t,w,st = random.choice(by_user[u]); sample.append(dict(bin=j, label=l, text=t, words=w, title=st))
random.shuffle(sample)
for i,x in enumerate(sample): x["i"]=i
SYS = ("You rate the stance toward AI expressed in a single online forum comment from a thread about AI. Return JSON with one field, "
       "stance: an integer from -3 to 3, or null. -3 = strongly negative or dismissive about AI (overhyped, useless, harmful, a bubble); "
       "-1 = mildly sceptical; 0 = expresses a view that is balanced or neutral; 1 = mildly positive; "
       "3 = strongly positive or enthusiastic (transformative, highly capable, clearly valuable). "
       "Use null if the comment expresses no view about AI at all (off-topic, purely factual or technical detail, a question). "
       "Rate the position taken, not how politely it is worded.")
def rate(s):
    body = json.dumps({"model":MODEL,"messages":[{"role":"system","content":SYS},{"role":"user","content":"Thread title: %s\n\nComment: %s" % (s["title"], s["text"][:1500])}],"response_format":{"type":"json_object"}}).encode()
    for k in range(4):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                j = json.loads(json.load(r)["choices"][0]["message"]["content"])
                s["stance"] = None if j["stance"] is None else int(j["stance"]); s["ok"]=True; return s
        except Exception as e: s["err"]=str(e)[:200]
    return s
with ThreadPoolExecutor(16) as ex: sample = list(ex.map(rate, sample))
ok = [s for s in sample if s.get("ok")]
print("rated", len(ok), "of", len(sample), "| first error:", next((s["err"] for s in sample if not s.get("ok")), None))
rng = np.random.default_rng(0); out = {"model":MODEL, "n_rated":len(ok), "by_bin":[]}
for j in range(6):
    a = [s for s in ok if s["bin"]==j]; st = np.array([s["stance"] for s in a if s["stance"] is not None], float)
    ex_ = np.abs(st)
    out["by_bin"].append(dict(bin=BINS[j][2], n=len(a), share_with_stance=float(len(st)/max(len(a),1)), n_stance=int(len(st)),
        mean_stance=float(st.mean()), mean_extremity=float(ex_.mean()), se_extremity=float(ex_.std(ddof=1)/np.sqrt(len(ex_))),
        share_strong=float((ex_>=2).mean()), share_negative=float((st<0).mean()), share_positive=float((st>0).mean())))
hs = [s for s in ok if s["stance"] is not None]
b = np.array([s["bin"] for s in hs]); e = np.abs([s["stance"] for s in hs]).astype(float); sv = np.array([s["stance"] for s in hs], float)
ball = np.array([s["bin"] for s in ok]); has = np.array([s["stance"] is not None for s in ok], float)
out["spearman_bin_vs_extremity"] = dict(rho=float(stats.spearmanr(b,e)[0]), p=float(stats.spearmanr(b,e)[1]))
out["spearman_bin_vs_stance"] = dict(rho=float(stats.spearmanr(b,sv)[0]), p=float(stats.spearmanr(b,sv)[1]))
out["spearman_bin_vs_has_stance"] = dict(rho=float(stats.spearmanr(ball,has)[0]), p=float(stats.spearmanr(ball,has)[1]))
json.dump(out, open(os.path.join(OUT,"stance_%s.json" % MODEL),"w"), indent=1)
json.dump(ok, open(os.path.join(OUT,"stance_ratings_%s.json" % MODEL),"w"))
for x in out["by_bin"]: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in x.items()})
print({k:v for k,v in out.items() if k.startswith("spearman")})

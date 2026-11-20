import json, glob, re, html, sys, os, collections
import numpy as np
from scipy import stats
DATA, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
rows = {}
for f in sorted(glob.glob(os.path.join(DATA,"day*.json"))):
    d = int(re.search(r"day(\d+)", f).group(1))
    for h in json.load(open(f)):
        if h.get("author") and h.get("comment_text"):
            h["day"] = d; rows[h["objectID"]] = h
rows = list(rows.values())
M = len(rows)
TAG = re.compile(r"<[^>]+>")
def clean(t): return html.unescape(TAG.sub(" ", t)).lower()
AI = re.compile(r"\b(ai|a\.i\.|llm|llms|gpt|chatgpt|claude|openai|anthropic|gemini|copilot|agents?|agentic|machine learning|neural|deepmind|mistral|llama|transformer|diffusion|agi)\b", re.I)
HEDGE = ["i think","i guess","i believe","maybe","perhaps","probably","possibly","might","seems","seem","not sure","i suspect","imo","imho","afaik","iirc","arguably","sort of","kind of","i wonder","could be","as far as i know","in my experience","i feel"]
ABS = ["absolutely","all","always","complete","completely","constant","constantly","definitely","entire","ever","every","everyone","everything","full","must","never","nothing","totally","whole","obviously","clearly","of course","literally","nobody","no one","undoubtedly","certainly"]
rx = lambda L: re.compile(r"\b(" + "|".join(re.escape(x) for x in sorted(L,key=len,reverse=True)) + r")\b")
HR, AR = rx(HEDGE), rx(ABS)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
va = SentimentIntensityAnalyzer()
res = {}
# ---------- H1 concentration (all 14 days)
cnt = collections.Counter(r["author"] for r in rows)
p = np.array(sorted(cnt.values()), float); N = len(p)
srt = np.sort(p)[::-1]; cum = np.cumsum(srt)/p.sum()
gini = 1 - 2*np.trapezoid(np.cumsum(np.sort(p))/p.sum(), dx=1/N)
res["H1"] = dict(comments=M, users=N, mean_p=p.mean(), median_p=float(np.median(p)),
    size_biased_mean=(p**2).mean()/p.mean(), ratio=(p**2).mean()/p.mean()**2,
    top1_share=float(cum[int(np.ceil(N*0.01))-1]), top10_share=float(cum[int(np.ceil(N*0.10))-1]),
    one_comment_user_share=float((p==1).mean()), one_comment_comment_share=float((p==1).sum()/p.sum()),
    users_for_half_of_comments=float((np.searchsorted(cum,0.5)+1)/N), gini=float(gini),
    top_authors=cnt.most_common(8))
# ---------- H2 stability
c1 = collections.Counter(r["author"] for r in rows if r["day"]<7)
c2 = collections.Counter(r["author"] for r in rows if r["day"]>=7)
users = sorted(set(c1)|set(c2))
a = np.array([c1[u] for u in users]); b = np.array([c2[u] for u in users])
BINS = [(0,0,"0"),(1,1,"1"),(2,4,"2-4"),(5,9,"5-9"),(10,24,"10-24"),(25,10**9,"25+")]
def binof(x):
    for i,(lo,hi,_) in enumerate(BINS):
        if lo<=x<=hi: return i
w1 = a>0
res["H2_time"] = dict(spearman_all=float(stats.spearmanr(a,b)[0]),
    spearman_log_active_both=float(stats.pearsonr(np.log(a[(a>0)&(b>0)]),np.log(b[(a>0)&(b>0)]))[0]),
    by_week1_bin=[dict(bin=l, n=int(((a>=lo)&(a<=hi)).sum()), p_return=float((b[(a>=lo)&(a<=hi)]>0).mean()), mean_w2=float(b[(a>=lo)&(a<=hi)].mean())) for lo,hi,l in BINS[1:]])
isai = lambda r: bool(AI.search(r.get("story_title") or ""))
# cross-topic, cross-time: non-AI rate in week 1 -> AI-thread participation in week 2
n1 = collections.Counter(r["author"] for r in rows if r["day"]<7 and not isai(r))
ai2 = collections.Counter(r["author"] for r in rows if r["day"]>=7 and isai(r))
x = np.array([n1[u] for u in users]); y = np.array([ai2[u] for u in users])
res["H2_topic"] = dict(ai_comment_share=float(np.mean([isai(r) for r in rows])), spearman=float(stats.spearmanr(x,y)[0]),
    by_bin=[dict(bin=l, n=int(((x>=lo)&(x<=hi)).sum()), p_any_ai_w2=float((y[(x>=lo)&(x<=hi)]>0).mean()), mean_ai_w2=float(y[(x>=lo)&(x<=hi)].mean())) for lo,hi,l in BINS])
# ---------- H3 traits measured in week 2, propensity measured in week 1
U = collections.defaultdict(lambda: dict(n=0,words=0,hedge=0,abs=0,reply=0,neg=0,comp=0.0))
for r in rows:
    if r["day"]<7: continue
    t = clean(r["comment_text"]); w = len(t.split())
    if w==0: continue
    u = U[r["author"]]; u["n"]+=1; u["words"]+=w
    u["hedge"]+=len(HR.findall(t)); u["abs"]+=len(AR.findall(t))
    u["reply"]+= int(r.get("parent_id")!=r.get("story_id"))
    c = va.polarity_scores(t[:3000])["compound"]; u["neg"]+= int(c<=-0.5); u["comp"]+=c
us = list(U); n = np.array([U[u]["n"] for u in us], float); prior = np.array([c1[u] for u in us])
T = {"hedges per 100 words": np.array([100*U[u]["hedge"]/U[u]["words"] for u in us]),
     "absolutist words per 100 words": np.array([100*U[u]["abs"]/U[u]["words"] for u in us]),
     "share of comments that are replies": np.array([U[u]["reply"]/U[u]["n"] for u in us]),
     "words per comment": np.array([U[u]["words"]/U[u]["n"] for u in us]),
     "share of strongly negative comments": np.array([U[u]["neg"]/U[u]["n"] for u in us])}
bidx = np.array([binof(v) for v in prior])
res["H3"] = {}
for k,t in T.items():
    person, post = t.mean(), (n*t).sum()/n.sum()
    ident = np.cov(n,t,bias=True)[0,1]/n.mean()
    bs = []
    for _ in range(1000):
        i = rng.integers(0,len(t),len(t)); bs.append((n[i]*t[i]).sum()/n[i].sum() - t[i].mean())
    lo,hi = np.percentile(bs,[2.5,97.5])
    bins = []
    for j,(_,_,l) in enumerate(BINS):
        m = bidx==j; v = t[m]
        se = v.std(ddof=1)/np.sqrt(len(v))
        bins.append(dict(bin=l, n=int(m.sum()), mean=float(v.mean()), se=float(se)))
    res["H3"][k] = dict(person_weighted=float(person), post_weighted=float(post), gap=float(post-person), gap_pct=float(100*(post-person)/person),
        cov_identity=float(ident), ci95=[float(lo),float(hi)], spearman_prior_vs_trait=float(stats.spearmanr(prior,t)[0]),
        spearman_p=float(stats.spearmanr(prior,t)[1]), by_week1_bin=bins)
res["H3_n_users_week2"] = len(us)
json.dump(res, open(os.path.join(OUT,"results.json"),"w"), indent=1)
np.savez(os.path.join(OUT,"arrays.npz"), p=p, a=a, b=b)
print(json.dumps(res, indent=1))

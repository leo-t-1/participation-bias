import json, glob, re, html, sys, os, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
DATA, OUT = sys.argv[1], sys.argv[2]
res = json.load(open(os.path.join(OUT,"results.json")))
rows = {}
for f in sorted(glob.glob(os.path.join(DATA,"day*.json"))):
    d = int(re.search(r"day(\d+)", f).group(1))
    for h in json.load(open(f)):
        if h.get("author") and h.get("comment_text"): h["day"]=d; rows[h["objectID"]]=h
rows = list(rows.values())
TAG = re.compile(r"<[^>]+>"); clean = lambda t: html.unescape(TAG.sub(" ", t)).lower()
va = SentimentIntensityAnalyzer()
c1 = collections.Counter(r["author"] for r in rows if r["day"]<7)
call = collections.Counter(r["author"] for r in rows)
top = {u for u,_ in call.most_common(max(1,len(call)//1000))}   # top 0.1% accounts
BINS = [(0,0,"0"),(1,1,"1"),(2,4,"2-4"),(5,9,"5-9"),(10,24,"10-24"),(25,10**9,"25+")]
def run(filt):
    U = collections.defaultdict(lambda: [0,0,0])  # n, neg, reply
    for r in rows:
        if r["day"]<7: continue
        t = clean(r["comment_text"]); w = len(t.split())
        if not filt(r,w): continue
        u = U[r["author"]]; u[0]+=1; u[1]+= int(va.polarity_scores(t)["compound"]<=-0.5); u[2]+= int(r.get("parent_id")!=r.get("story_id"))
    us=list(U); n=np.array([U[u][0] for u in us],float); pr=np.array([c1[u] for u in us])
    out={}
    for name,j in (("neg",1),("reply",2)):
        t=np.array([U[u][j]/U[u][0] for u in us])
        out[name]=dict(person=float(t.mean()), post=float((n*t).sum()/n.sum()),
            bins=[float(t[(pr>=lo)&(pr<=hi)].mean()) for lo,hi,_ in BINS])
    return out
rob = dict(length_matched_20_60_words=run(lambda r,w: 20<=w<=60),
           excluding_top_0p1pct_accounts=run(lambda r,w: w>0 and r["author"] not in top))
json.dump(rob, open(os.path.join(OUT,"robustness.json"),"w"), indent=1); print(json.dumps(rob,indent=1))
# ---------------- figure
arr = np.load(os.path.join(OUT,"arrays.npz")); p = np.sort(arr["p"])[::-1]
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig, ax = plt.subplots(1,3, figsize=(15,4.6))
x = np.arange(1,len(p)+1)/len(p); y = np.cumsum(p)/p.sum()
ax[0].plot(100*x,100*y,color="#1f4e79",lw=2); ax[0].plot([0,100],[0,100],color="#999",ls="--",lw=1)
for q in (1,10):
    yy = 100*y[int(np.ceil(len(p)*q/100))-1]; ax[0].scatter([q],[yy],color="#c0392b",zorder=3)
    ax[0].annotate(f"top {q}% of users\nwrite {yy:.0f}% of comments",(q,yy),xytext=(q+12,yy-(14 if q==10 else 16)),arrowprops=dict(arrowstyle="-",color="#555"))
H=res["H1"]
ax[0].text(47,24,f"random commenter: {H['mean_p']:.1f} comments / 2 wks\nauthor of a random comment: {H['size_biased_mean']:.1f}",fontsize=8.5,bbox=dict(fc="#f4f4f4",ec="none"))
ax[0].set(xlabel="% of commenters (most active first)",ylabel="% of all comments",title="H1  A few people write most of what you read")
labs=[b["bin"] for b in res["H2_time"]["by_week1_bin"]]
ax[1].plot(labs,[100*b["p_return"] for b in res["H2_time"]["by_week1_bin"]],"o-",color="#1f4e79",lw=2,label="comments again in week 2 (any topic)")
ax[1].plot(labs,[100*b["p_any_ai_w2"] for b in res["H2_topic"]["by_bin"][1:]],"s-",color="#c0392b",lw=2,label="comments on an AI thread in week 2\n(x-axis: week-1 comments on non-AI threads)")
ax[1].set(ylim=(0,100),xlabel="comments in week 1",ylabel="% of users",title="H2  Posting is a stable, topic-general disposition"); ax[1].legend(frameon=False,fontsize=8.5,loc="lower right")
names={"share of comments that are replies":"replies (vs. top-level)","share of strongly negative comments":"strongly negative tone","words per comment":"comment length","absolutist words per 100 words":"absolutist words","hedges per 100 words":"hedging words"}
ks=list(names); g=[res["H3"][k]["gap_pct"] for k in ks]
lo=[100*res["H3"][k]["ci95"][0]/res["H3"][k]["person_weighted"] for k in ks]; hi=[100*res["H3"][k]["ci95"][1]/res["H3"][k]["person_weighted"] for k in ks]
order=np.argsort(g); yy=np.arange(len(ks))
ax[2].barh(yy,[g[i] for i in order],xerr=[[g[i]-lo[i] for i in order],[hi[i]-g[i] for i in order]],color=["#c0392b" if abs(g[i])>1 and lo[i]*hi[i]>0 else "#bbb" for i in order],capsize=3)
ax[2].set_yticks(yy); ax[2].set_yticklabels([names[ks[i]] for i in order]); ax[2].axvline(0,color="#333",lw=1)
ax[2].set(xlabel="% difference: average comment vs. average commenter\n(= Cov(p, t) / E[p], 95% bootstrap CI)",title="H3  What you read vs. who is there")
fig.suptitle(f"Hacker News, 1-14 Nov 2026: {H['comments']:,} comments by {H['users']:,} users. Posting rate measured in week 1, traits in week 2.",fontsize=10,y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"participation_bias_hn.png"),dpi=170,bbox_inches="tight")

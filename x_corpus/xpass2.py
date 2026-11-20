import sys, glob, collections, random, re, json, os, pyarrow.parquet as pq, numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
D, OUT = sys.argv[1], sys.argv[2]; random.seed(0)
EN = set("the and is to of a in that it you for on with this are was be have not but they".split())
DE = set("der die das und nicht ist ich zu den mit von auf für ein eine sich auch es dem".split())
HEDGE = ["i think","i guess","i believe","maybe","perhaps","probably","possibly","might","seems","seem","not sure","i suspect","imo","imho","afaik","iirc","arguably","sort of","kind of","i wonder","could be","as far as i know","in my experience","i feel"]
ABS = ["absolutely","all","always","complete","completely","constant","constantly","definitely","entire","ever","every","everyone","everything","full","must","never","nothing","totally","whole","obviously","clearly","of course","literally","nobody","no one","undoubtedly","certainly"]
rx = lambda L: re.compile(r"\b(" + "|".join(re.escape(x) for x in sorted(L,key=len,reverse=True)) + r")\b")
HR, AR = rx(HEDGE), rx(ABS); URL = re.compile(r"https?://\S+|@\w+")
A = collections.Counter(); Bt = collections.defaultdict(list); Ball = collections.Counter()
for f in sorted(glob.glob(D+"/s*.parquet")):
    t = pq.read_table(f, columns=["username","date","is-retweet","text","replying-to"])
    for u,d,r,x,rep in zip(t["username"].to_pylist(), t["date"].to_pylist(), t["is-retweet"].to_pylist(), t["text"].to_pylist(), t["replying-to"].to_pylist()):
        if r or len(u)<2 or not ("2026-01-01" <= d[:10] <= "2026-07-27"): continue
        if d[:10] < "2026-05-01": A[u]+=1
        else:
            Ball[u]+=1
            if len(Bt[u])<20: Bt[u].append((x or "", bool(rep)))
BINS = [(0,0,"0"),(1,1,"1"),(2,4,"2-4"),(5,10,"5-10"),(11,25,"11-25"),(26,10**9,"26+")]
res = {"n_users":len(set(A)|set(Ball)), "A_tweets":sum(A.values()), "B_tweets":sum(Ball.values())}
res["H2"] = [dict(bin=l, n=int(sum(1 for u,c in A.items() if lo<=c<=hi)), p_seen_again=float(np.mean([Ball[u]>0 for u,c in A.items() if lo<=c<=hi])), mean_B=float(np.mean([Ball[u] for u,c in A.items() if lo<=c<=hi]))) for lo,hi,l in BINS[1:]]
va = SentimentIntensityAnalyzer(); H3 = []
for lo,hi,l in BINS:
    us = [u for u in Bt if lo<=A[u]<=hi]; random.shuffle(us); rowsU = []
    for u in us:
        n=w=h=a=neg=rp=0
        for x,rep in Bt[u]:
            tl = URL.sub(" ", x).lower(); tok = re.findall(r"[a-zäöüß']+", tl)
            if len(tok)<5 or sum(k in EN for k in tok)<2 or sum(k in DE for k in tok)>=2: continue
            n+=1; w+=len(tok); h+=len(HR.findall(tl)); a+=len(AR.findall(tl)); rp+=rep; neg+= va.polarity_scores(tl)["compound"]<=-0.5
        if n: rowsU.append((n,w,h,a,neg,rp))
        if len(rowsU)>=12000: break
    R = np.array(rowsU,float); m = lambda v: (float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v))))
    H3.append(dict(bin=l, users=len(R), neg=m(R[:,4]/R[:,0]), absolutist=m(100*R[:,3]/R[:,1]), hedge=m(100*R[:,2]/R[:,1]), reply=m(R[:,5]/R[:,0]), words=m(R[:,1]/R[:,0])))
    print(H3[-1], flush=True)
res["H3"] = H3
json.dump(res, open(os.path.join(OUT,"x_results.json"),"w"), indent=1); print(json.dumps(res["H2"],indent=1)); print({k:res[k] for k in ("n_users","A_tweets","B_tweets")})

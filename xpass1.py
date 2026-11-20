import sys, glob, collections, pyarrow.parquet as pq, numpy as np
D = sys.argv[1]
A = collections.Counter(); B = collections.Counter(); n=0; rt=0; terms=set(); dmin="9"; dmax="0"
for f in sorted(glob.glob(D+"/s*.parquet")):
    t = pq.read_table(f, columns=["username","date","is-retweet","search_term"])
    u = t["username"].to_pylist(); d = t["date"].to_pylist(); r = t["is-retweet"].to_pylist(); s = t["search_term"].to_pylist()
    print(f, len(u), "terms", s[0], "->", s[-1], "| dates", d[0][:10], d[len(d)//2][:10], d[-1][:10], flush=True)
    for uu,dd,rr in zip(u,d,r):
        n+=1
        if rr: rt+=1; continue
        if not ("2026-01-01" <= dd[:10] <= "2026-07-27"): continue
        (A if dd[:10] < "2026-05-01" else B)[uu]+=1
tot = A+B
c = np.array(list(tot.values()))
print("rows",n,"retweets",rt,"users",len(tot),"A tweets",sum(A.values()),"B tweets",sum(B.values()))
print("count dist (total):", {k:int((c==k).sum()) for k in (1,2,3,4,5)}, ">=6:",int((c>=6).sum()), ">=11:",int((c>=11).sum()), ">=26:", int((c>=26).sum()), "max", c.max())
a = np.array(list(A.values())); print("A count dist:", {k:int((a==k).sum()) for k in (1,2,3,4,5)}, "6-10:",int(((a>=6)&(a<=10)).sum()), "11+:",int((a>=11).sum()))
print("top:", tot.most_common(8))

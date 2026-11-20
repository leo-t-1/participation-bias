"""Collider (selection) simulation. Expertise and conviction are independent in the population.
A person posts if expertise + conviction + noise > threshold. Among posters they become negatively correlated.
python3 simulation/collider_sim.py results"""
import sys, json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; rng = np.random.default_rng(0); N = 200_000
e, c = rng.standard_normal(N), rng.standard_normal(N)
res = {"population_corr": float(np.corrcoef(e, c)[0, 1]), "by_share_posting": []}
for share in (0.5, 0.2, 0.1, 0.05, 0.01):
    rs = []
    for seed in range(20):
        r = np.random.default_rng(seed); e_, c_ = r.standard_normal(N), r.standard_normal(N)
        s = e_ + c_ + r.standard_normal(N); m = s > np.quantile(s, 1 - share)
        rs.append(np.corrcoef(e_[m], c_[m])[0, 1])
    res["by_share_posting"].append({"share": share, "corr_among_posters": float(np.mean(rs)), "sd_over_20_runs": float(np.std(rs, ddof=1))})
s = e + c + rng.standard_normal(N); m = s > np.quantile(s, 0.9)
res["top10_example"] = {"corr": float(np.corrcoef(e[m], c[m])[0, 1]), "mean_conviction_posters": float(c[m].mean()), "mean_expertise_posters": float(e[m].mean())}
json.dump(res, open(os.path.join(OUT, "collider.json"), "w"), indent=1); print(json.dumps(res, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
i = rng.choice(N, 6000, replace=False)
ax[0].scatter(e[i][~m[i]], c[i][~m[i]], s=4, c="#bbbbbb", label="silent (90%)")
ax[0].scatter(e[i][m[i]], c[i][m[i]], s=6, c="#c0392b", label="posters (10%)")
ax[0].set_xlabel("expertise", fontsize=13); ax[0].set_ylabel("conviction", fontsize=13); ax[0].legend(fontsize=11, frameon=False)
ax[0].set_title(f"Population r = {res['population_corr']:.2f}; among posters r = {res['top10_example']['corr']:.2f}", fontsize=13)
sh = [b["share"] * 100 for b in res["by_share_posting"]]; cr = [b["corr_among_posters"] for b in res["by_share_posting"]]
ax[1].plot(sh, cr, "o-", c="#1f4e79"); ax[1].axhline(0, c="gray", ls="--"); ax[1].set_xscale("log")
ax[1].set_xlabel("% of population that posts", fontsize=13); ax[1].set_ylabel("expertise-conviction correlation\namong posters", fontsize=13)
ax[1].set_title("Stricter selection, stronger spurious trade-off", fontsize=13)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Simulation: post if expertise + conviction + noise > threshold (traits independent in the population)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "collider_sim.png"), dpi=150)

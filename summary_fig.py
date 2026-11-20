"""One summary figure from the saved result files. python3 summary_fig.py results"""
import json, sys, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = sys.argv[1]; J = lambda *p: json.load(open(os.path.join(*p)))
HN, X, GREY, INK = "#1f4e79", "#c0392b", "#9aa0a6", "#222222"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
res, val, xl, col = J(R, "results.json"), J(R, "llm_validation.json"), J(R, "x_llm_gpt-4.1.json"), J(R, "collider.json")
cs, wt = J("x_corpus", "count_summary.json"), J("x_corpus", "weighting.json")
se = lambda p, n: np.sqrt(p * (1 - p) / n)
fig, ax = plt.subplots(2, 3, figsize=(17, 10)); ax = ax.ravel()
BL = ["0", "1", "2-4", "5-9", "10-24", "25+"]

# A: size bias
a = ax[0]; vals = [res["H1"]["mean_p"], res["H1"]["size_biased_mean"], wt["posts_per_account"], wt["posts_per_post_author"]]
ypos = [3.2, 2.4, 1.0, 0.2]; cols = [GREY, HN, GREY, X]
a.barh(ypos, vals, height=0.62, color=cols); a.set_xscale("log"); a.set_xlim(1, 2500)
for y, v, lab in zip(ypos, vals, ["a random commenter", "the author of a random comment", "a random account", "the author of a random post"]):
    a.text(v * 1.12, y, "%s: %.3g" % (lab, v), va="center", fontsize=11.5)
a.set_yticks([2.8, 0.6]); a.set_yticklabels(["Hacker News\n(2 weeks)", "X corpus\n(7 months)"]); a.set_xlabel("posts written (log scale)")
a.set_title("A  You read the prolific, not the typical")

# B: disposition
a = ax[1]; xs = range(5)
a.plot(xs, [100 * b["p_return"] for b in res["H2_time"]["by_week1_bin"]], "o-", c=HN, lw=2, ms=8, label="HN: comments again next week")
a.plot(xs, [100 * b["p_any_ai_w2"] for b in res["H2_topic"]["by_bin"][1:]], "s--", c=HN, lw=2, ms=7, label="HN: joins an AI thread next week\n(x = earlier non-AI comments)")
a.plot(xs, [100 * b["reappearance"] for b in cs[1:6]], "D-", c=X, lw=2, ms=7, label="X: captured again in May-July")
a.set_xticks(list(xs)); a.set_xticklabels(["1", "2-4", "5-9", "10-24", "25+"]); a.set_ylim(-32, 102); a.set_yticks([0, 20, 40, 60, 80, 100]); a.set_ylabel("% of users")
a.set_xlabel("posts in the earlier period"); a.legend(frameon=False, fontsize=10, loc="lower right", bbox_to_anchor=(1.02, 0.0))
a.set_title("B  Speaking up is a stable, general trait")

# C: hostility
a = ax[2]; xs = range(6)
h = [b["share_ge3"] for b in val["hostility"]["by_bin"]]; n = [b["n"] for b in val["hostility"]["by_bin"]]
xb = xl["X2_tone_persons"]["by_bin"]; hx = [b["share_hostile_ge3"] for b in xb]; nx = [b["hostility"]["n"] for b in xb]
a.errorbar(xs, 100 * np.array(h), yerr=100 * se(np.array(h), np.array(n)), fmt="o-", c=HN, lw=2, ms=8, capsize=3, label="Hacker News comments")
a.errorbar(xs, 100 * np.array(hx), yerr=100 * se(np.array(hx), np.array(nx)), fmt="D-", c=X, lw=2, ms=7, capsize=3, label="X posts by individuals")
a.set_xticks(list(xs)); a.set_xticklabels(BL); a.set_ylim(0, 60); a.set_ylabel("% rated dismissive or hostile (LLM, blind)")
a.set_xlabel("posts in the earlier period"); a.legend(frameon=False, fontsize=11, loc="center right")
a.set_title("C  Hostility: rises on HN, flat on X")

# D: AI stance on HN, two judges
a = ax[3]
for model, mk, ls in (("gpt-4.1", "o", "-"), ("gpt-5.4-mini", "s", "--")):
    rs = J(R, "stance_ratings_%s.json" % model); m, e = [], []
    for j in range(6):
        v = np.array([s["stance"] for s in rs if s["bin"] == j and s["stance"] is not None], float)
        m.append(v.mean()); e.append(v.std(ddof=1) / np.sqrt(len(v)))
    a.errorbar(range(6), m, yerr=e, fmt=mk + ls, c=HN, lw=2, ms=7, capsize=3, label="judge: " + model)
a.axhline(0, c="#888888", lw=1, ls=":"); a.set_xticks(range(6)); a.set_xticklabels(BL); a.set_ylim(-1.1, 0.5)
a.set_ylabel("mean stance on AI  (-3 sceptical ... +3 enthusiastic)"); a.set_xlabel("comments in week 1")
a.legend(frameon=False, fontsize=11, loc="lower left")
a.text(0.98, 0.95, "extremity |stance|: no robust trend\n(hypothesis failed)", transform=a.transAxes, ha="right", va="top", fontsize=11)
a.set_title("D  Stance on AI (HN): shifted, not more extreme")

# E: account type on X
a = ax[4]; tb = xl["X1_account_type"]["by_bin"]
org = np.array([b["organisation"] for b in tb]); bot = np.array([b["automated"] for b in tb]); nn = np.array([b["n"] for b in tb])
a.bar(range(6), 100 * org, width=0.62, color=X, label="organisations", edgecolor="white", linewidth=2)
a.bar(range(6), 100 * bot, width=0.62, bottom=100 * org, color="#e8a39a", label="automated accounts", edgecolor="white", linewidth=2)
a.errorbar(range(6), 100 * (org + bot), yerr=100 * se(org + bot, nn), fmt="none", ecolor=INK, capsize=3)
for i, v in enumerate(100 * (org + bot)): a.text(i, v + 3.2, "%.0f%%" % v, ha="center", fontsize=11)
a.set_xticks(range(6)); a.set_xticklabels(["0", "1", "2-4", "5-10", "11-25", "26+"]); a.set_ylim(0, 50); a.set_ylabel("% of accounts (LLM-classified)")
a.set_xlabel("posts captured in January-April (3 of 73 shards)"); a.legend(frameon=False, fontsize=11, loc="upper left")
a.set_title("E  On X the heavy tail is partly not people")

# F: collider simulation
a = ax[5]; rng = np.random.default_rng(0); N = 200_000
e_, c_ = rng.standard_normal(N), rng.standard_normal(N); s = e_ + c_ + rng.standard_normal(N); m = s > np.quantile(s, 0.9)
i = rng.choice(N, 5000, replace=False)
a.scatter(e_[i][~m[i]], c_[i][~m[i]], s=5, c="#c9cdd2", label="silent (90%), simulated", linewidths=0)
a.scatter(e_[i][m[i]], c_[i][m[i]], s=9, c=X, label="posters (10%)", linewidths=0)
a.set_xlabel("expertise"); a.set_ylabel("conviction"); a.legend(frameon=False, fontsize=11, loc="lower left", markerscale=2.5)
a.text(0.98, 0.97, "population r = %.2f\namong posters r = %.2f" % (abs(col["population_corr"]), col["top10_example"]["corr"]), transform=a.transAxes, ha="right", va="top", fontsize=11.5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
a.set_title("F  Selection fakes a trade-off")

fig.suptitle("What you read = what people think + Cov(posting rate, trait) / mean posting rate", fontsize=16, fontweight="bold", x=0.01, ha="left")
fig.text(0.01, 0.005, "Hacker News: all 144,483 comments, 1-14 Nov 2026. X: SinclairSchneider/tweets_sample_2026 (keyword-collected, not a random sample). Error bars: standard errors.", fontsize=10, color="#555555")
fig.tight_layout(rect=(0, 0.02, 1, 0.96), w_pad=2.5, h_pad=2.5); fig.savefig(os.path.join(R, "summary.png"), dpi=150)
print("saved")

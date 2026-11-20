"""Plot exact reply reweighting and the two independent draws of political posts."""

from pathlib import Path
import json
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

folder = Path(__file__).resolve().parent


def read(filename):
    return json.loads((folder / filename).read_text())


def wilson(count, n):
    p = count / n
    denom = 1 + 1.96**2 / n
    mid = (p + 1.96**2 / (2 * n)) / denom
    half = 1.96 * math.sqrt(p * (1 - p) / n + 1.96**2 / (4 * n * n)) / denom
    return (mid - half, mid + half)


weights = read("weighting.json")
text = read("text_summary.json")
fig, (left, right) = plt.subplots(
    1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.6]}
)
colors = ["#257369", "#944629"]
for i, (key, label) in enumerate(
    [("account_reply_share", "Account first"), ("post_reply_share", "Post first")]
):
    p = weights[key]
    left.bar(i, p, color=colors[i], width=0.53)
    left.text(i, p + 0.014, f"{p:.1%}", ha="center", fontsize=11)
left.set_xticks([0, 1], ["Account first", "Post first"])
left.set_title("Share of late posts that are replies", fontsize=12, loc="left")
left.set_ylim(0, 1)
labels = ["oppose", "support", "mixed", "none"]
for j, arm in enumerate(("politics_accounts", "politics_posts")):
    row = text[arm]["all"]["all"]
    vals = [row["stance"][k] / row["n"] for k in labels]
    intervals = [wilson(row["stance"][k], row["n"]) for k in labels]
    right.errorbar(
        [i + (-0.09 if j == 0 else 0.09) for i in range(4)],
        vals,
        yerr=[
            [p - lo for p, (lo, hi) in zip(vals, intervals)],
            [hi - p for p, (lo, hi) in zip(vals, intervals)],
        ],
        color=colors[j],
        linestyle="none",
        marker="o",
        capsize=3,
        label=["Account first", "Post first"][j],
    )
right.set_xticks(range(4), ["Oppose", "Support", "Mixed", "No position"])
right.set_title("Expressed stance in posts mentioning Trump", fontsize=12, loc="left")
right.set_ylim(0, 1)
right.legend(frameon=False, loc="upper right")
for ax in (left, right):
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#dddddd", linewidth=0.75)
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.tick_params(length=0, pad=6)
fig.suptitle(
    "What changes when every author counts equally?", fontsize=16, x=0.07, ha="left"
)
fig.text(
    0.07,
    0.045,
    "May 1–July 27, 2026. Replies use all 73 shards. Stance uses three shards and 600 draws per arm.\nIntervals show 95% sampling uncertainty within the text sample frame. Model and collection errors are additional.",
    fontsize=9,
    color="#454545",
)
fig.subplots_adjust(top=0.80, bottom=0.23, left=0.07, right=0.98, wspace=0.28)
fig.savefig(folder / "x_weighting.png", dpi=170)

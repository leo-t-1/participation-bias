"""Format aggregate contrasts. Sampling intervals are conditional on fixed model labels."""

import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
text = json.loads((root / "text_summary.json").read_text())
tab = [
    "| Sampling rule | n | Oppose | Support | Mixed | No position |",
    "|---|---:|---:|---:|---:|---:|",
]


def pct(num, den):
    return f"{num/den:.1%}" if den else "n/a"


pol = []
for arm, title in [
    ("politics_accounts", "One post per sampled account"),
    ("politics_posts", "Sample posts directly"),
]:
    g = text[arm]["all"]["all"]
    pol.append(g)
    tab.append(
        "| "
        + " | ".join(
            [title, str(g["n"])]
            + [
                pct(g["stance"][k], g["n"])
                for k in ("oppose", "support", "mixed", "none")
            ]
        )
        + " |"
    )
p1, p2 = [g["stance"]["oppose"] / g["n"] for g in pol]
se = math.sqrt(sum(p * (1 - p) / g["n"] for p, g in zip((p1, p2), pol)))
gap = p2 - p1
tab += [
    "",
    f"The post-minus-account difference for expressed opposition was {100*gap:+.1f} percentage points (approximate 95% interval {100*(gap-1.96*se):+.1f} to {100*(gap+1.96*se):+.1f}). The interval describes sampling error assuming independent draws and fixed labels. Collector bias and model classification error are additional sources of uncertainty.",
    "",
    "| Earlier posts in the three shards | n | Hostile | Replies | Person-like | Organization-like | Overt automation | Unclear | Replies in person-like group | Hostile in person-like group |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for label, r in text["general"].items():
    g = r["all"]
    n = g["n"]
    h = r["person"]
    values = [label, str(n), pct(g["hostile"], n), pct(g["reply"], n)]
    values += [
        pct(r[k]["n"], n) for k in ("person", "organization", "automated", "unclear")
    ]
    values += [pct(h["reply"], h["n"]), pct(h["hostile"], h["n"])]
    tab.append("| " + " | ".join(values) + " |")
(root / "text_results.md").write_text(
    "# Text sampling results\n\n" + "\n".join(tab) + "\n"
)

"""Check the stance rubric on invented controls with known intended meanings."""

import json
import os
from pathlib import Path
from rate import rate_batch, MODEL

CASES = [
    ("I support Trump's approach to this issue.", "support"),
    (
        "The presenter read a viewer message aloud: “I support Trump’s approach to this issue.”",
        "none",
    ),
    ("I oppose Trump's approach to this issue.", "oppose"),
    (
        "The presenter read a viewer message aloud: “I oppose Trump’s approach to this issue.”",
        "none",
    ),
    ("I think Trump made exactly the right call today.", "support"),
    (
        "According to its press release, the campaign says Trump made exactly the right call today.",
        "none",
    ),
    ("Trump's decision today was indefensible.", "oppose"),
    (
        "Tuesday’s headline quotes a senator calling Trump’s decision “indefensible.”",
        "none",
    ),
    ("I support Trump's first proposal and oppose his second.", "mixed"),
    (
        "The committee will review Trump's first and second proposals at two o'clock.",
        "none",
    ),
    ("Does anyone have a transcript of Trump's speech?", "none"),
    ("The transcript of Trump's speech came online an hour ago.", "none"),
    ("I wish Trump would reverse this decision. It will hurt people.", "oppose"),
    ("The amendment Trump signed yesterday takes effect next month.", "none"),
    ("Trump deserves credit for this decision.", "support"),
    (
        "A caller said Trump deserved credit for the decision. The broadcast then ended.",
        "none",
    ),
]
if __name__ == "__main__":
    key = os.getenv("OPENAI_API_KEY") or next(
        l.split("=", 1)[1].strip()
        for l in Path(".env").read_text().splitlines()
        if l.startswith("OPENAI_API_KEY=")
    )
    items = [{"id": i, "text": s} for i, (s, _) in enumerate(CASES)]
    labels = dict(rate_batch(("content", items, key)))
    report = [
        dict(text=s, intended=y, rated=labels[i]["stance"])
        for i, (s, y) in enumerate(CASES)
    ]
    Path("x_corpus/controls.json").write_text(
        json.dumps(dict(model=MODEL, cases=report), indent=2) + "\n"
    )
    print(sum(r["rated"] == r["intended"] for r in report), "of", len(report))

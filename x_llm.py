"""LLM-rated tests on the X sample (SinclairSchneider/tweets_sample_2026, three shards in xdata/s*.parquet).
Posting rate = an account's sampled posts in Jan-Apr 2026; everything else is measured on its May-Jul posts.

Hypotheses, written before running:
  X1  The share of accounts that are not individual people (organisations, automated accounts) rises with posting rate.
  X2  Among individual people, hostility rises with posting rate (as on Hacker News).
  X3  Among individual people posting about politics, stance extremity (|left-right stance|) rises with posting rate.

python3 x_llm.py xdata results .env <model> [accounts_per_bin]
"""
import json, re, sys, os, urllib.request, collections
import numpy as np, duckdb
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
D, OUT, ENV, MODEL = sys.argv[1:5]; NPER = int(sys.argv[5]) if len(sys.argv) > 5 else 400
KEY = [l.split("=", 1)[1].strip() for l in open(ENV) if l.startswith("OPENAI_API_KEY=")][0]
rng = np.random.default_rng(0)
BINS = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 10, "5-10"), (11, 25, "11-25"), (26, 10**9, "26+")]
binof = lambda a: next(j for j, (lo, hi, _) in enumerate(BINS) if lo <= a <= hi)
EN = set("the and is to of a in that it you for on with this are was be have not but they".split())
DE = set("der die das und nicht ist ich zu den mit von auf für ein eine sich auch es dem".split())
URL = re.compile(r"https?://\S+")
def english(x):
    tok = re.findall(r"[a-zäöüß']+", URL.sub(" ", x or "").lower())
    return len(tok) >= 8 and sum(k in EN for k in tok) >= 2 and sum(k in DE for k in tok) < 2
POL = r"\b(trump|biden|harris|democrats?|republicans?|maga|liberals?|conservatives?|immigration|immigrants?|congress|senate|elections?|leftists?|left-wing|right-wing|labour|tories|tory|starmer|farage|woke|gop|socialis[mt]|fascis[mt]s?|abortion|gun control|deportations?|taxes|government)\b"

con = duckdb.connect()
con.execute("""create table t as select username, name, text, id, date >= '2026-05-01' as late, len("replying-to") > 0 as rep
               from read_parquet('%s/s*.parquet') where "is-retweet" = false and length(username) > 1
               and date >= '2026-01-01' and date < '2026-07-28'""" % D)
users = con.execute("select username, count(*) filter (not late) a, count(*) filter (late) b from t group by 1 order by 1").fetchall()
A = {u: a for u, a, b in users}
print("accounts", len(users), "posts", con.execute("select count(*) from t").fetchone()[0], flush=True)

def draw(rows, per_user):
    """rows: (username, name, text, rep). Keep English posts, group by user."""
    g = collections.defaultdict(list)
    for u, n, x, rep in rows:
        if english(x) and len(g[u]) < per_user: g[u].append((n, x, bool(rep)))
    return g

# --- sample 1: tone. Random accounts with a May-Jul post, per posting-rate bin.
cand = []
for j, (lo, hi, _) in enumerate(BINS):
    us = [u for u, a, b in users if b > 0 and lo <= a <= hi]
    cand += list(rng.choice(us, min(len(us), NPER * 5), replace=False))
con.execute("create table cand(username varchar)"); con.executemany("insert into cand values (?)", [(u,) for u in cand])
g = draw(con.execute("""select username, name, text, rep from (select t.*, row_number() over (partition by t.username order by hash(id)) rn
                        from t join cand using (username) where late) where rn <= 8 order by username, rn""").fetchall(), 5)
tone = []
for j in range(6):
    us = sorted(u for u in g if binof(A[u]) == j); us = list(rng.permutation(us))[:NPER]
    tone += [dict(kind="tone", bin=j, user=u, name=g[u][0][0], posts=[x for _, x, _ in g[u]], text=g[u][0][1],
                  reply_share=float(np.mean([r for _, _, r in g[u]]))) for u in us]
# --- sample 2: politics. Accounts with a May-Jul post matching a political keyword, one such post each.
g2 = draw(con.execute("""select username, name, text, rep from (select t.*, row_number() over (partition by username order by hash(id)) rn
                         from t where late and regexp_matches(lower(text), ?)) where rn <= 3 order by username, rn""", [POL]).fetchall(), 1)
con.execute("create table cand2(username varchar)"); pol = []
for j in range(6):
    us = sorted(u for u in g2 if binof(A[u]) == j); us = list(rng.permutation(us))[:NPER]
    pol += [dict(kind="pol", bin=j, user=u, name=g2[u][0][0], text=g2[u][0][1]) for u in us]
con.executemany("insert into cand2 values (?)", [(s["user"],) for s in pol])
extra = draw(con.execute("""select username, name, text, rep from (select t.*, row_number() over (partition by t.username order by hash(id)) rn
                            from t join cand2 using (username) where late) where rn <= 8 order by username, rn""").fetchall(), 5)
for s in pol: s["posts"] = [x for _, x, _ in extra.get(s["user"], [])] or [s["text"]]
print("tone sample", collections.Counter(s["bin"] for s in tone), "politics sample", collections.Counter(s["bin"] for s in pol), flush=True)

SYS_TYPE = ("You classify an X (Twitter) account from its handle, display name and a few of its posts. Return JSON with one field, type, one of: "
            "\"person\" (an individual posting in their own voice, including pseudonymous accounts and public figures), "
            "\"organisation\" (company, brand, news outlet, customer service, institution, campaign), "
            "\"automated\" (bot, spam, feed, aggregator, repetitive promotion), \"unclear\".")
SYS_TONE = ("You rate the tone of a single social media post. Return JSON with two integer fields, each 1-5. "
            "hostility: 1 = friendly or neutral, 3 = dismissive or irritated, 5 = openly hostile or contemptuous toward a person or group. "
            "assertiveness: 1 = tentative, heavily qualified, 3 = ordinary statements, 5 = categorical, presents opinion as certain fact. "
            "Judge tone toward people and certainty of claims, not whether the topic itself is negative.")
SYS_POL = ("You rate the political position expressed in a single social media post. Return JSON with two fields. "
           "stance: an integer from -3 to 3, or null. -3 = strongly left-wing or progressive, -1 = leans left, 0 = political but balanced or centrist, "
           "1 = leans right, 3 = strongly right-wing or conservative. Use null if the post takes no political position or cannot be placed on a left-right axis. "
           "Rate the position, not how politely it is worded. hostility: integer 1-5, 1 = friendly or neutral, 3 = dismissive or irritated, "
           "5 = openly hostile or contemptuous toward a person or group.")
def call(system, user):
    body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       "response_format": {"type": "json_object"}}).encode()
    err = None
    for k in range(4):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r: return json.loads(json.load(r)["choices"][0]["message"]["content"])
        except Exception as e: err = str(e)[:200]
    return {"err": err}
def rate(s):
    t = call(SYS_TYPE, "Handle: %s\nDisplay name: %s\nPosts:\n%s" % (s["user"], s["name"], "\n".join("- " + p[:400] for p in s["posts"])))
    s["type"] = t.get("type"); r = call(SYS_TONE if s["kind"] == "tone" else SYS_POL, s["text"][:1500])
    try:
        s["hostility"] = int(r["hostility"])
        if s["kind"] == "tone": s["assertiveness"] = int(r["assertiveness"])
        else: s["stance"] = None if r["stance"] is None else int(r["stance"])
        s["ok"] = s["type"] is not None
    except Exception: s["err"] = str(r)[:200]
    return s
with ThreadPoolExecutor(32) as ex: allr = list(ex.map(rate, tone + pol))
ok = [s for s in allr if s.get("ok")]
print("rated", len(ok), "of", len(allr), "| first error:", next((s.get("err") for s in allr if not s.get("ok")), None), flush=True)

sp = lambda x, y: dict(rho=float(stats.spearmanr(x, y)[0]), p=float(stats.spearmanr(x, y)[1]), n=len(x))
ms = lambda v: dict(mean=float(np.mean(v)), se=float(np.std(v, ddof=1) / np.sqrt(len(v))), n=len(v)) if len(v) > 1 else dict(mean=None, se=None, n=len(v))
out = {"model": MODEL, "accounts_per_bin": NPER, "n_rated": len(ok), "bins": [b[2] for b in BINS]}
T = [s for s in ok if s["kind"] == "tone"]; P = [s for s in ok if s["kind"] == "pol"]
out["X1_account_type"] = dict(by_bin=[dict(bin=BINS[j][2], n=sum(s["bin"] == j for s in T),
        **{k: float(np.mean([s["type"] == k for s in T if s["bin"] == j])) for k in ("person", "organisation", "automated", "unclear")}) for j in range(6)],
    spearman_bin_vs_not_person=sp([s["bin"] for s in T], [s["type"] != "person" for s in T]))
for label, sub in (("all_accounts", T), ("persons", [s for s in T if s["type"] == "person"])):
    out["X2_tone_" + label] = dict(by_bin=[dict(bin=BINS[j][2], hostility=ms([s["hostility"] for s in sub if s["bin"] == j]),
            share_hostile_ge3=float(np.mean([s["hostility"] >= 3 for s in sub if s["bin"] == j])),
            assertiveness=ms([s["assertiveness"] for s in sub if s["bin"] == j]), reply_share=ms([s["reply_share"] for s in sub if s["bin"] == j])) for j in range(6)],
        spearman_bin_vs_hostility=sp([s["bin"] for s in sub], [s["hostility"] for s in sub]),
        spearman_bin_vs_assertiveness=sp([s["bin"] for s in sub], [s["assertiveness"] for s in sub]),
        spearman_bin_vs_reply_share=sp([s["bin"] for s in sub], [s["reply_share"] for s in sub]))
for label, sub in (("all_accounts", P), ("persons", [s for s in P if s["type"] == "person"])):
    st = [s for s in sub if s["stance"] is not None]
    out["X3_politics_" + label] = dict(n=len(sub), share_with_stance=len(st) / max(len(sub), 1),
        by_bin=[dict(bin=BINS[j][2], n=sum(s["bin"] == j for s in st), extremity=ms([abs(s["stance"]) for s in st if s["bin"] == j]),
            share_strong=float(np.mean([abs(s["stance"]) >= 2 for s in st if s["bin"] == j])), mean_stance=ms([s["stance"] for s in st if s["bin"] == j]),
            hostility=ms([s["hostility"] for s in st if s["bin"] == j])) for j in range(6)],
        spearman_bin_vs_extremity=sp([s["bin"] for s in st], [abs(s["stance"]) for s in st]),
        spearman_bin_vs_stance=sp([s["bin"] for s in st], [s["stance"] for s in st]),
        spearman_bin_vs_hostility=sp([s["bin"] for s in sub], [s["hostility"] for s in sub]))
json.dump(out, open(os.path.join(OUT, "x_llm_%s.json" % MODEL), "w"), indent=1)
json.dump(ok, open(os.path.join(D, "x_llm_ratings_%s.json" % MODEL), "w"))  # kept local: contains handles and post text
print(json.dumps(out, indent=1))

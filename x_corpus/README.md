# Who gets heard in a vocabulary search of X?

The [dataset card](https://huggingface.co/datasets/SinclairSchneider/tweets_sample_2026/blob/254bbe4cea800035ec544cf047d8b3bebc24fe99/README.md) documents a Nitter search over 65,689 terms from a newspaper vocabulary. The collector merged the results and removed duplicate post IDs. Search ranking and coverage varied by term and date. This permits descriptive comparisons inside the corpus. The probabilities that a given X user or post was captured are unknown. The publisher explicitly advises against using it to estimate platform prevalence or study particular accounts.

## Reproduce

From the repository root in a Python 3.13 environment with DuckDB, PyArrow and requests installed:

1. Run `python x_corpus/count.py` to retrieve the date, handle, retweet and reply fields across 73 shards and build local per-handle counts. Downloads are pinned to revision `254bbe4cea800035ec544cf047d8b3bebc24fe99`. Completed shards are cached. Run `python x_corpus/weights.py` for the exact post-versus-account contrast.
2. Run `python x_corpus/text_shards.py` to fetch and checksum shards 0, 24 and 48. To reproduce the earlier three-shard lexical analysis, install NumPy and vaderSentiment and run `python xpass2.py xdata results`.
3. Run `python x_corpus/rate.py` for the exploratory text experiments (paid OpenAI API, GPT-4.1). A key may be supplied in the environment or a local .env file. The raw samples, handles and cached model labels stay in the ignored xdata directory. The public output contains aggregate counts. Future model calls can differ.
4. Run `python x_corpus/audit_hn.py` to compare the two previously saved HN judges without making API calls.

## What was measured

For corpus counts the unit is a lowercased handle with at least one observed, dated, non-retweet post between January 1 and July 27, 2026, inclusive. Early counts cover January through April. Late counts cover May through July 27. A reply has a nonempty replying-to field. Periods differ in collection intensity. Reappearance measures persistence in retrieval and can reflect persistent language, topic, or account activity. The same person could use several handles. No impressions or ranked feed were observed.

The full-corpus reply comparison uses every eligible late post in every language. Under post weighting every post counts once. Under account weighting every observed late author contributes their own fraction of replies with equal weight. The early-bin 0 group contains only accounts retrieved later, so its return rate is undefined.

The text analysis estimates the political sampling contrast and the account-role breakdown. The full-corpus count measures reply reweighting and concentration.

The text sample uses its own early counts from just the three specified shards, plus a simple English-word screen in the late period. For tone, 160 distinct accounts were selected in each of four early-count bins, with one sampled late post per account. For political stance, we selected late posts containing the word Trump and used two draws: 600 sampled posts and 600 sampled accounts contributing one eligible post each. Hash-based ordering makes each draw reproducible. A uniform draw from posts gives frequent contributors more weight. The political arms have a shared sampling frame and may overlap. Posts about Trump without that literal token fall outside this comparison.

The content judge saw post text without activity labels or account metadata. Stance has categories support, oppose, mixed and no expressed position. A separate model call labeled apparent public account role from name, handle and one post. These are imperfect presentation cues; the automated category captures overt automation, and opaque cases can land in any class. Subgroup comparisons are descriptive. Conditional stance percentages among opinionated posts answer a different question from percentages calculated over all sampled posts.

For HN, 760/1,800 posts received a nonzero stance from both judges. Another 334 received a nonzero stance from just one. The 1,247 paired numeric answers include zeros. Comparing the 0 versus 25+ early-comment groups, average paired scores were -0.05 versus -0.62 on paired numeric responses, -0.10 versus -0.97 when both scores were directional, and -0.04 versus -0.43 when absent scores counted as zero. In example 467, criticism of Google's advertising business received a mini score of -3 and a null from GPT-4.1. Example 1174 criticizes a company's financials and received -2 and null. A zero or a missing score represents a judgment about expression and context, not independent evidence of a user's attitude. Full sensitivity results appear in hn_audit.json.

The additional search-term sensitivity analysis (`python x_corpus/terms.py`) uses late posts in any language from the three text shards. It compares authors with one early capture to those with 26 or more. Each author contributes total weight one across their late posts. After computing reply shares inside each recorded term and group, both groups are standardized to the same term distribution, proportional to the smaller of the two groups' relative mass for each term. The analysis repeats with minimum cell sizes of 1, 5 and 20 posts per group. Search terms are lossy: the corpus supplies only one matched term per post, even if several queries found it. Equalizing this field does not equalize collection probabilities.

For a rubric check, `python x_corpus/controls.py` submits 16 constructed examples with explicit intended labels. GPT-4.1 classified all 16 as intended, including attributed quotations without endorsement. They cover straightforward cases; accuracy on real X posts requires separate validation. `python x_corpus/plot.py` regenerates the figure using matplotlib.

A timing control (`python x_corpus/timing.py`) preserves every author's total number of captures and both period sizes while shuffling the early/late labels over posts. For each account total it computes the exact expected chance of appearing in both windows by sampling without replacement. A lower observed share is consistent with activity bursts, account entry or exit, or time-varying retrieval. The null assumes all captured posts are exchangeable in time; it is a benchmark for additional temporal clustering.

The [Pew linked survey](https://www.pewresearch.org/politics/2019/10/23/national-politics-on-twitter-small-share-of-u-s-adults-produce-majority-of-tweets/) supplies the population benchmark. Running the timing check with `--hn` uses the original fourteen Hacker News day files.

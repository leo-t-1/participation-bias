**Short answer.** By itself the bias explains one thing: what you read over-represents prolific posters. It explains anything further (hostile tone, skewed opinion) only where posting rate correlates with that trait, which has to be measured. In my data that held for some traits and failed for others. I model it as size-biased sampling of people by posting rate.

**Reading the sentence.** It links an act (posting now) to a disposition (being someone who posts). It is only "near" tautological because it makes an empirical claim: participation is driven mainly by who you are, not by the situation.

**Model.** Give each person a posting rate p and any trait t (opinion, tone, expertise). A reader samples posts, not people, so person i is weighted by p_i:

  mean of t over posts = E[p·t]/E[p] = E[t] + Cov(p, t)/E[p]

Setting t = p gives the author of a random post a rate of E[p] + Var(p)/E[p] (size-biased sampling). The bias explains something only if (1) p is unequal and stable, and (2) p correlates with t. If Cov(p, t) = 0, discourse is a fair sample however concentrated it is.

**Test.** I fixed hypotheses first, then took every Hacker News comment from 1-14 Nov 2026 (144,483 comments, 26,624 users; rate from week 1, traits from week 2).

- Concentration: supported. A random commenter wrote 5.4 comments; the author of a random comment wrote 35.9.
- Disposition: supported, but situation matters too. Commenting again in week 2 rises from 31% (one comment in week 1) to 98% (25+), and activity on non-AI threads predicts commenting on AI threads (13% to 88%). Yet users with exactly two comments spread them over both weeks only 31% of the time (50% by chance). So p is a stable baseline times topical bursts.
- Tone: partly. Replies are 77% of comments but 66% for the average commenter. LLM-rated hostility, blind to author, rises from 6% (silent in week 1) to 19% (25+).
- Opinion: two LLM judges rated stance on AI for 1,800 comments. Heavy posters are not more extreme (hypothesis failed) but, exploratory, more sceptical (mean -0.05 to -0.62 on a -3 to 3 scale).

**X.** A public keyword-collected corpus (72M posts, 8.9M accounts, 2026; not a random sample) shows the same disposition: reappearing in May-July rises from 43% (one captured post in January-April) to 97% (26+). But the heavy tail is partly not people: in LLM ratings of 2,400 accounts, organisations and bots rise from 8% to 38%. Among individuals, neither hostility nor political extremity rises with rate (both hypotheses failed). Pew, matching surveys to accounts: strong Trump disapprovers were 55% of US adult users but wrote 72% of tweets on national politics.

**What it explains.**

1. Misjudging the population when Cov(p, t) is not zero: Hacker News reads as more hostile and AI-sceptical than its users, political Twitter as more anti-Trump.
2. Spurious trade-offs (collider bias). Simulate independent expertise and conviction, and let the top 10% by expertise + conviction + noise post: among posters the two correlate at -0.38. Confident people look less informed with no real link.
3. Anything calibrated on posts inherits the p-weighting: sentiment "polls", LLM training data.

**What it does not explain.** Why p correlates with a trait. It also does not imply polarisation: my extremity hypothesis failed on both platforms. Algorithmic amplification is a separate reweighting on top. Limits: lurkers are invisible; LLM ratings are unvalidated.

Code and figures: https://github.com/leo-t-1/participation-bias

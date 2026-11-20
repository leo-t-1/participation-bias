**Short answer.** Read as a statement about acts, the sentence is empty: the people who post are the people who post. Read as a statement about dispositions, it is a testable claim: seeing someone speak on one occasion helps predict whether they speak on another, even on a different topic. If so, public discourse is not a sample of the public. It weights people by how readily they speak. By itself this explains only that ready speakers are over-represented. It explains more exactly when the disposition to speak correlates with something else.

**Model.** Give each person a posting rate p and any trait t (an opinion, tone, expertise). A reader samples posts, not people, so person i is weighted by p_i:

  mean of t over posts = E[t] + Cov(p, t)/E[p]

This covers two selections at once: who speaks at all (the silent have p = 0 and are never heard) and how often speakers speak (ten posts give ten chances to be read). Setting t = p gives the tautology itself: the author of a random post posts at E[p] + Var(p)/E[p]. For any other trait the gap is a covariance, and its sign is open. Concentration alone tells you nothing: if Cov(p, t) = 0, the post-weighted mean of t equals its population mean. I would model p as a stable personal baseline times interest in the topic and occasion.

**What it explains.**

1. The gap between discourse and public opinion. Whatever travels with the urge to speak looks more common than it is, so readers misjudge what others think.
2. Spurious patterns among speakers. If either expertise or conviction gets you to post, the two are negatively correlated among posters even when independent in the population (collider bias; r = -0.38 in my simulation). "The loudest know least" can be pure selection.
3. Why anything calibrated on posts (sentiment "polls", a politician's sense of the mood, LLM training data) can misread the public.

**What it does not explain.** Which traits correlate with p, or why. Polarisation does not follow. Algorithmic amplification is a second reweighting on top. And selection can feed back: if a harsh tone drives moderates out, p itself changes.

**Evidence** (hypotheses fixed beforehand; figures in the link). On Hacker News (144,483 comments, two weeks) the chance of commenting again next week rises from 31% to 98% with prior activity, and activity on non-AI threads predicts showing up on AI threads (13% to 88%). Occasion matters too: users with exactly two comments spread them over both weeks 31% of the time (50% by chance). A demonstrated consequence: a random comment is a reply 77% of the time; pick a commenter first and it is 66%. A limit: I predicted heavy posters would hold more extreme views on AI, and they did not (though they were more sceptical and more hostile). In a 72M-post X corpus the disposition replicates (43% to 97%), but individuals who post more were neither more hostile nor more politically extreme. All of this compares people who spoke at least once. Pew's survey-matched data reaches further, because it measures the attitude independently of posting: strong Trump disapprovers were 55% of US adult Twitter users and wrote 72% of tweets mentioning national politics. That shows whose voices were over-represented, not what the tweets said.

Code, data and figures: https://github.com/leo-t-1/participation-bias

**Short answer.** Read as a statement about acts, the sentence is empty: the people who post are the people who post. Read as a statement about dispositions, it is an empirical claim: speaking up is a stable trait of a person, not a response to the situation. If that is true, public discourse is not a sample of the public. It is a record of one type of person. By itself this explains only that prolific speakers are over-represented. It explains more exactly when the disposition to speak correlates with something else.

**Model.** Give each person a posting rate p and any trait t (an opinion, tone, expertise). A reader samples posts, not people, so person i is weighted by p_i:

  mean of t over posts = E[t] + Cov(p, t)/E[p]

Setting t = p gives the tautology itself: the author of a random post posts at E[p] + Var(p)/E[p], more than the average person whenever rates differ. For every other trait the gap is a covariance, and its sign is open. Concentration alone tells you nothing: if Cov(p, t) = 0, a discourse written by 1% of people is still a fair sample of t. I would model p as a stable personal baseline times topical interest, with the silent (p = 0) as the largest group.

**What it explains.**

1. The gap between discourse and public opinion. Whatever travels with the urge to speak looks more common than it is, so readers misjudge what others think.
2. Why debates on different topics feel alike: they are held by the same people.
3. Spurious patterns among speakers. If either expertise or conviction gets you to post, the two are negatively correlated among posters even when independent in the population (collider bias; r = -0.38 in my simulation). "The loudest know least" can be pure selection.
4. Anything calibrated on posts inherits the weighting: sentiment "polls", a politician's sense of the mood, LLM training data.

**What it does not explain.** Which traits correlate with p, or why. Polarisation does not follow. Algorithmic amplification is a second reweighting on top. And selection can feed back: if a harsh tone drives moderates out, p itself changes, which a static model misses.

**Evidence** (hypotheses fixed before looking; figures and code in the link). On Hacker News (144,483 comments, two weeks) the disposition is real and topic-general: the chance of commenting again next week rises from 31% to 98% with prior activity, and activity on non-AI threads predicts showing up on AI threads (13% to 88%). The author of a random comment wrote 35.9 comments, a random commenter 5.4. The covariance was trait-specific: heavy posters were more hostile (6% to 19% of comments, LLM-rated) and more AI-sceptical, but not more extreme, which I had predicted. In a 72M-post X corpus the disposition replicates (43% to 97%), but individuals who post more were neither more hostile nor more politically extreme, and the heavy tail is increasingly organisations and bots (8% to 38%). Pew's matched survey data shows a case where it bites: strong Trump disapprovers were 55% of US adult Twitter users and wrote 72% of political tweets.

Code, data and figures: https://github.com/leo-t-1/participation-bias

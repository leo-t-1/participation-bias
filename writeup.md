“The people who post are the people who post” is a tautology. The word “tend” gives it a testable implication. Hearing someone speak today should help me predict whether they’ll speak next week, even on a different topic. This can explain why counting posts can give a distorted impression of public opinion and why two traits can appear related among speakers even when they’re independent in the population.

Give each person a posting rate p and a trait t, such as support for a policy. Someone with p = 0 goes unheard. Someone who writes ten posts gets ten chances to be read. If I draw a post uniformly, the expected trait of its author is

E[p × t] / E[p] = E[t] + Cov(p, t) / E[p],

where expectations are over people. Concentration alone leaves the direction of the bias open. Even if a few people dominate the conversation, the mean shifts only when posting rate covaries with the trait. An AI trained on these posts can inherit that weighting. A ranked feed adds another weight through the posts it shows.

Selection can also create an apparent trade-off. In my simulation, expertise and conviction were independent, and people posted when expertise plus conviction plus noise crossed a threshold. Someone with little expertise generally needed stronger conviction to speak. Among speakers the correlation was −0.38.

I would model posting rate as a personal baseline multiplied by interest in the topic and occasion. In 144,483 Hacker News comments over two weeks, 31% of people who commented once in week one returned in week two, rising to 98% among those who commented at least 25 times. Activity outside AI threads also predicted later participation in AI threads. Yet among people who commented exactly twice, only 31% appeared in both weeks, against 50% if dates were shuffled.

What difference did selection make? A random Hacker News comment had a 77% chance of being a reply. Picking a commenter first, then one of their comments, gave 66%. I also predicted that frequent posters would express more extreme views about AI. Two model judges found no consistent increase in extremity. In a keyword-collected X corpus of 72 million posts, earlier appearances likewise predicted later ones, but a rated subsample showed no clear increase in political extremity among accounts classified as individuals.

These comparisons reach only people who spoke. Pew linked survey answers to Twitter records and found that strong Trump disapprovers made up 55% of U.S. adult Twitter users and wrote 72% of tweets mentioning national politics. The 72% tells us whose voices were heard. How many tweets actually expressed disapproval is a different empirical question.

Code, data and figures: https://github.com/leo-t-1/participation-bias
Pew study: https://www.pewresearch.org/data-labs/2019/10/23/national-politics-on-twitter-small-share-of-u-s-adults-produce-majority-of-tweets/

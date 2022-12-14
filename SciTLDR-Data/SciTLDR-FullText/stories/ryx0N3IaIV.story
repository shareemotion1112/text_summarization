Consistently checking the statistical significance of experimental results is the first mandatory step towards reproducible science.

This paper presents a hitchhiker's guide to rigorous comparisons of reinforcement learning algorithms.

After introducing the concepts of statistical testing, we review the relevant statistical tests and compare them empirically in terms of false positive rate and statistical power as a function of the sample size (number of seeds) and effect size.

We further investigate the robustness of these tests to violations of the most common hypotheses (normal distributions, same distributions, equal variances).

Beside simulations, we compare empirical distributions obtained by running Soft-Actor Critic and Twin-Delayed Deep Deterministic Policy Gradient on Half-Cheetah.

We conclude by providing guidelines and code to perform rigorous comparisons of RL algorithm performances.

Reproducibility in Machine Learning and Reinforcement Learning in particular (RL) has become a serious issue in the recent years.

As pointed out in Islam et al. BID0 and Henderson et al. BID1 , reproducing the results of an RL paper can turn out to be much more complicated than expected.

In a thorough investigation, Henderson et al. BID1 showed it can be caused by differences in codebases, hyperparameters (e.g. size of the network, activation functions) or the number of random seeds used by the original study.

Henderson et al. BID1 states the obvious: the claim that an algorithm performs better than another should be supported by evidence, which requires the use of statistical tests.

Building on these observations, this paper presents a hitchhiker's guide for statistical comparisons of RL algorithms.

The performances of RL algorithm have specific characteristics (they are independent of each other, they are not paired between algorithms etc.).

This paper reviews some statistical tests relevant in that context and compares them in terms of false positive rate and statistical power.

Beside simulations, it compares empirical distributions obtained by running Soft-Actor Critic (SAC) BID2 and Twin-Delayed DDPG (TD3) BID3 on Half-Cheetah BID4 .

We finally provide guidelines to perform robust difference testing in the context of RL.

A repository containing the raw results and the code to reproduce all experiments is available at https://github.com/ccolas/rl_stats.

In this paper, we consider the problem of conducting meaningful comparisons of Algorithm 1 and Algorithm 2.

Because the seed of the random generator is different for each run BID1 , two runs of a same algorithm yield different measures of performance.

An algorithm performance can therefore be modeled as a random variable X, characterized by a distribution.

Measuring the performance x at the end of a particular run is equivalent to measuring a realization of that random variable.

Repeating this N times, we obtain a sample x = (x 1 , ..., x N ) of size N .

To compare RL algorithms on the basis of their performances, we focus on the comparisons of the central tendencies (?? 1 , ?? 2 ): the means or the medians of the associated random variables X 1 , X 2 .

BID2 Unfortunately, we cannot know ?? 1 , ?? 2 exactly.

Given a sample x i of X i , we can estimate ?? i by the empirical mean: DISPLAYFORM0 (resp.

the empirical median).

However, comparing central performances does not simply boil down to the comparison of their estimates.

As an illustration, FIG0 shows two normal distributions describing the distributions of two algorithm performances X 1 and X 2 .

Two samples of sample size N = 3 are collected.

In this example, we have ?? 1 < ?? 2 but x 1 > x 2 .

The rest of this text uses central performance to refer to either the mean or the median of the performance distribution i.

It is noted ?? i while its empirical estimate is noted x i .

The distinction is made where necessary.

Statistical difference testing.

Statistical difference testing offers a principled way to compare the central performances of two algorithms.

It defines two hypothesis: 1) the null hypothesis H 0 : ????? = ?? 1 ????? 2 = 0 and 2) the alternative hypothesis H a : |?????| > 0.

When performing a test, one initially assumes the null hypothesis to be true.

After having observed (x 1 , x 2 ), statistical tests usually estimate the probability to observe two samples whose empirical central difference is at least as extreme as the observed one (|???x| = |x 1 ???x 2 |) under H 0 (e.g. given ????? = 0).

This probability is called the p-value.

If the p-value is very low, the test rejects H 0 and concludes that a true underlying difference (H a ) is likely.

When the p-value is high, the test does not have enough evidence to conclude.

This could be due to the lack of true difference, or to the lack of statistical power (too few measurements given how noisy they are).

The significance level ?? (usually ??? 0.05) draws the line between rejection and conservation of H 0 : if p-value < ??, H 0 is rejected.

Further experiments demonstrate it is not always the case, which is why we prefer to note the false positive rate ?? * .

False negatives occur when the statistical test fails to recognize a true difference in the central performances.

This depends on the size of the underlying difference: the larger the difference, the lower the risk of false negative.

The false negative rate is noted ?? * .Trade-off between false positive and statistical power.

Ideally, we would like to set ?? = 0 to ensure the lowest possible false positive rate ?? * .

However, decreasing the confidence level makes the statistical test more conservative.

The test requires even bigger empirical differences ???x to reject H 0 , which decreases the probability of true positive.

This probability of true positive 1????? * is called the statistical power of a test.

It is the probability to reject H 0 when H a holds.

It is directly impacted by the effect size: the larger the effect size, the easier it is to detect (larger statistical power).

It is also a direct function of the sample size: larger samples bring more evidence to support the rejection of H 0 .

Generally, the sample size is chosen so as to obtain a theoretical statistical power of 1????? * = 0.8.

Different tests have different statistical powers depending on the assumptions they make, whether they are met, how the p-value is derived etc.

Parametric vs. non-parametric.

Parametric tests usually compare the means of two distributions by making assumptions on the distributions of the two algorithms' performances.

Non-parametric tests on the other hand usually compare the medians and do not require assumptions on the type of distributions.

Non-parametric tests are often recommended when one wants to compare median rather than means, when the data is skewed or when the sample size is small.

Section 4.2 shows that these recommendations are not always justified.

Test statistic.

Statistical tests usually use a test statistic.

It is a numerical quantity computed from the samples that summarizes the data.

In the t-test for instance, the statistic t ?? is computed as t ?? = |???x|/?? pool , where ?? pool is the pooled standard deviation (?? pool = (?? 2 1 + ?? 2 2 )/2).

Under the t-test assumptions, this statistic follows the analytic Student's distribution with density function f S (t).

The probability to obtain a difference more important than the sample difference ???x (p-value) can be rewritten p-value = P (|t| > t ?? ) and can be computed as the area under f S (t) such that |t| > t ?? .Relative effect size.

The relative effect size is the absolute effect size |?????|, scaled by the pooled standard deviation ?? pool , such that = |?????|/?? pool .

The relative effect size is independent of the spread of the considered distributions.3 Statistical Tests for RL

Each test makes some assumptions (e.g. about the nature of the performance distributions, their variances, the sample sizes etc.).

In the context of RL, some assumptions are reasonable while others are not.

It is reasonable to assume that RL performances are measured at random and independently from one another.

The samples are not paired, and here we assume they have the same size.

BID3 Other common assumptions might be discussed:??? Normal distributions of performances: this might not be the case (skewed distributions, bimodal distributions, truncated distributions).??? Continuous performances: the support of the performance distribution might be bounded:e.g.

in the Fetch environments of Gym BID4 , the performance is a success rate in [0, 1].??? Known standard deviations: this is not the case in RL.??? Equal standard deviations: this is often not the case (see BID1 ).

This section briefly presents various statistical tests relevant to the comparison of RL performances.

It focuses on the underlying assumptions BID5 and provides the corresponding implementation from the Python Scipy library when available.

Further details can be found in any statistical textbook.

Contrary to Henderson et al. BID1 , we do not recommend using the Kolmogorov-Smirnov test as it tests for the equality of the two distributions and does not test for a difference in their central tendencies BID6 .T-test.

This parametric test compares the means of two distributions and assumes the two distributions have equal variances BID7 .

If this variance is known, a more powerful test is available: the Z-test for two population means.

The test is accurate when the two distributions are normal, it gives an approximate guide otherwise.

Implementation: scipy.stats.ttest_ind(x1, x2, equal_var=True).Welch's t-test.

It is a t-test where the assumption of equal variances is relaxed BID8 .

Implementation: scipy.stats.ttest_ind(x1, x2, equal_var=False).Wilcoxon Mann-Whitney rank sum test.

This non-parametric test compares the median of two distributions.

It does not make assumptions about the type of distributions but assumes they are continuous and have the same shape and spread BID9 .

Implementation: scipy.stats.mannwhitneyu(x1, x2, alternative='two-sided').Ranked t-test.

In this non-parametric test that compares the medians, all realizations are ranked together before being fed to a traditional t-test.

Conover and Iman BID10 shows that the computed statistic is a monotonically increasing function of the statistic computed by the Wilcoxon MannWhitney test, making them really close.

Implemented in our code.

Bootstrap confidence interval test.

In the bootstrap test, the sample is considered to be an approximation of the original distribution BID11 .

Given two observed samples (x 1 , x 2 ) of size N , we obtain two bootstrap samples (x 1 ,x 2 ) of size N by sampling with replacement in (x 1 , x 2 ) respectively and compute the difference in empirical means ???x.

This procedure is repeated a large number of times (e.g. 103 ).

The distance between percentiles ????100 2 and 100(1??? ?? 2 ) is considered to be the 100(1?????)% confidence interval around the true mean difference ?????. If it does not include 0, the test rejects the null hypothesis with confidence level ??.

This test does not require any assumptions on the performance distributions, but we will see it requires large sample sizes.

Implementation: https://github.com/facebookincubator/bootstrapped.

Permutation test.

Under the null hypothesis, the realizations of both samples would come from distributions with the same mean.

The empirical mean difference (???x) should not be affected by the relabelling of the different realization (in average).

The permutation test performs permutations of the realization labels and computes ???x =x 1 ???x 2 .

This procedure is repeated many times (e.g. 103 ).

H 0 is rejected if the proportion of |???x| that falls below the original difference |???x| is higher than 1?????.

Implemented in our code.

This section compares the above statistical tests in terms of their false positive rates and statistical powers.

A false positive rate estimates the probability to claim that two algorithms perform differently when H 0 holds.

It impacts directly the reproducibility of a piece of research and should be as low as possible.

Statistical power is the true positive rate and refers to the probability to find evidence for an existing effect.

The following study is an extension of the one performed in BID12 .

We conduct experiments using models of RL distributions (analytic distributions) and true empirical RL distributions collected by running 192 trials of both SAC BID2 and TD3 BID3 on Half-Cheetah-v2 BID4 for 2M timesteps.

Investigating the case of non-normal distributions.

Several candidate distributions are selected to model RL performance distributions ( FIG2 ): a standard normal distribution, a lognormal distribution and a bimodal distribution that is an even mixture of two normal distributions.

All these distributions are tuned so that ?? = 0, ?? = 1.

In addition we use two empirical distributions of size 192 collected from SAC and TD3.Investigating the case of unequal standard deviations.

To investigate the effect of unequal standard deviations, we tune the distribution parameters to double the standard deviation of Algorithm 2 as compared to Algorithm 1.

We also compare SAC and TD3 which have different standard deviations (?? T D3 = 1.15 ?? SAC ).Measuring false positive rates.

To test for false positive rates ?? * , we simply enforce H 0 by aligning the central performances of the two distributions: ?? 1 = ?? 2 = 0 (the median for the MannWhitney test and the ranked t-test, the mean for others).

Given one test, two distributions and a sample size, we sample x 1 and x 2 from distributions X 1 , X 2 and compare them using the test with ?? = 0.05.

We repeat this procedure N r = 10 3 times and estimate ?? * as the proportion of H 0 rejection.

BID4 Using the spinning up implementation of OpenAI: https://github.com/openai/spinningupThe standard error of this estimate is: se(?? * ) = (?? * (1????? * )/N r .

It is smaller than the widths of the lines on the reported figures.

This procedure is repeated for every test, every combination of distributions and for several sample sizes (see pseudo-code in the supplementary material).Measuring true positive rates (statistical power).

Here, we enforce the alternative hypothesis H a by sampling x 1 from a given distribution centered in 0 (mean or median depending on the test), and x 2 from a distribution whose mean (resp.

median) is shifted by an effect size ?????. Given one test, two distributions (the second being shifted) and the sample size, we repeat the procedure above and obtain an estimate of the true positive rate.

Tables reporting the statistical powers for various effect sizes, sample sizes, tests and assumptions are made available in the supplementary results.

Same distributions, equal standard deviations. , we can directly compare the mean estimates (the lines) to the significance level ?? = 0.05, the standard errors being smaller than the widths of these lines.

BID5 ?? * is very large when using bootstrap tests, unless large sample sizes are used (>40).

Using small sample sizes (<5), the permutation and the ranked t-test also show large ?? * .

Results using two log-normal distributions show similar behaviors and can be found in the supplementary results.

Same distributions, unequal standard deviations.

Here, we sample x 1 from a distribution, and x 2 from the same type of distribution with doubled standard deviation.

Comparing two normal distributions with different standard deviation does not differ much from the case with equal standard deviations.

Figure 4 (a) (bimodal distributions) shows that Mann-Whitney and ranked t-test (median tests) constantly overestimate ?? * , no matter the sample size (?? * > 0.1).

For log-normal distributions on the other hand ( Figure 4(b) ), the false positive rate using these tests respects the confidence level (?? * ??? ??) with sample sizes higher than N = 10.

However, other tests tend to show large ?? * , even for large sample sizes (?? * ??? 0.07 up to N > 50).Different distributions, equal standard deviations.

Now we compare samples coming from different distributions with equal standard deviations.

Comparing normal and bimodal distributions of equal standard deviation does not impact much the false positive rates curves (similar to FIG3 (a)).

However, FIG6 (a) and 5(b) show that when one of the two distributions is skewed (log-normal), the Mann-Whitney and the ranked t-test demonstrate very important false positive rate, a phenomenon that gets worse with larger sample sizes.

Section 4.5 discusses why it might be the case.

Different distributions, unequal standard deviations.

We now combine different distributions and different standard deviations.

As before, comparing a skewed distribution (log-normal) and a symmetric one leads to high false positive rates for the Mann-Whitney test and the ranked t-test BID5 We reproduced all the results twice, hardly seeing any difference in the figures.

All tests show similar estimations of statistical power.

More than 50 samples are needed to detect a relative effect size = 0.5 with 80% probability, close to 20 with = 1 and a bit more than 10 with = 2.

Tables reporting statistical power for Finally, we compare two empirical distributions obtained from running two RL algorithms (SAC, TD3) 192 times each, on Half-Cheetah.

We observe a small increase in false positive rates when using the ranked t-test ( Figure 7 ).

The relative effect size estimated from the empirical distributions is = 0.80 (median), or = 0.93 (mean), in favor of SAC.

For such relative effect sizes, the sample sizes required to achieve a statistical power of 0.8 are between 10 and 15 for tests comparing the mean and between 15 and 20 for tests comparing the median (see full table in supplementary results).

Using a sample size N = 5 with the Welch's t-test, the effect size would need to be 3 to 4 times larger to be detected with 0.8 probability.

No matter the distributions.

From the above results, it seems clear that the bootstrap test should never be used for sample sizes below N = 50 and the permutation test should never be used for sample sizes below N = 10.

The bootstrap test in particular, uses the sample as an estimate of the true performance distribution.

A small sample is a very noisy estimate, which leads to very high false positive rates.

The ranked t-test shows a false positive rate of 0 and a statistical power of 0 when N = 2 in all conditions.

As noted in BID12 , comparing two samples of size N = 2 can result in only four possible p-values (only 4 possible orders when ranked), none of which falls below ?? = 0.05.

Such quantization issues make this test unreliable for small sample sizes, see BID12 for further comments and references on this issue.

When distributions do not meet assumptions.

In addition to the behaviors reported above, Section 4.2 shows that non-parametric tests (Mann-Whitney and ranked t-test) can demonstrate very high false positive rates when comparing a symmetric distribution with a skewed one (log-normal).

This effect gets worse linearly with the sample size.

When the sample size increases, the number of samples drawn in the skewed tail of the log-normal increases.

All these realizations will be ranked above any realizations from the other distribution.

Therefore, the larger the sample size, the more realization are ranked first in favor of the log-normal, which leads to a bias in the statistical test.

This problem does not occur when two log-normal are compared to one another.

Comparing a skewed distribution to a symmetric one violates the Mann-Whitney assumptions stating that distributions must have the same shape and spread.

The false positive rates of Mann-Whitney and ranked t-test are also above the confidence level whenever a bimodal distribution is compared to another distribution.

The traditional recommendation to use non-parametric tests when the distributions are not normal seems to be failing when the two distributions are different.

Most robust tests.

The t-test and the Welch's t-test were found to be more robust than others to violations of their assumptions.

However, ?? * was found to be slightly above the required level (?? * > ??) when at least one of the two distributions is skewed (?? * ??? 0.1) no matter the sample size, and when one of the two distributions is bimodal, for small sample sizes N < 10.

Welch's ?? * is always a bit lower than the t-test's ?? * .Statistical power.

Except for the anomalies in small sample size mentioned above due to overconfident tests like the bootstrap or the permutation tests, statistical powers stay qualitatively stable no matter the distributions compared, or the test used: = 0.5: N ??? 100; = 1: N ??? 20 and = 2: N ??? 5, 10.

Measuring the performance of RL Algorithms.

Before using any statistical test, one must obtain measures of performance.

RL algorithms should ideally be evaluated offline.

The algorithm performance after t steps is measured as the average of the returns over E evaluation episodes conducted independently from training, usually using a deterministic version of the current policy (e.g. E = 20).

Evaluating agents by averaging performances over several training episodes results in a much less interpretable performance measure and should be stated clearly.

The collection of performance measures forms a learning curve.

Representing learning curves.

After obtaining a learning curve for each of the N runs, it can be rendered on a plot.

At each evaluation, one can represent either the empirical mean or median.

Whereas the empirical median directly represents the center of the collected sample, the empirical mean tries to model the sample as coming from a Gaussian distribution, and under this assumptions represents the maximum likelihood estimate of that center.

Error bars should also be added to this plot.

The standard deviation (SD) represents the variability of the performances, but is only representative when the values are approximately normally distributed.

When it is not normal, one should prefer to represent interpercentile ranges (e.g. 10% ??? 90%).

If the sample size is small (e.g. <10), the most informative solution is to represent all learning curves in addition to the mean or median.

When performances are normally distributed, the standard error of the mean (SE) or confidence intervals can be used to represent estimates of the uncertainty on the mean estimate.

Robust comparisons.

Which test, which sample sizes?

The results in Section 4.2 advocate for the use of the Welch's t-test, which shows lower false positive rate and similar statistical powers than other tests.

However, the false positive rate often remains superior to the confidence level ?? * > ?? when the distributions are not normal.

When in doubt, we recommend using lower confidence levels ?? < 0.05 (e.g. ?? = 0.01) to ensure that ?? * < 0.05.

The number of random seeds to be used to achieve a statistical power of 0.8 depends on the expected relative effect size: = 0.5: N ??? 100; = 1: N ??? 20 and = 2: N ??? 5,10.

The analysis of a real case comparing SAC and TD3 algorithms, required a sample size between N = 10 and N = 15 for a relatively strong effect = 0.93 when comparing the means, and about 5 more seeds when comparing the medians ( = 0.80).

Small sample sizes like N = 5 would require 3 to 4 times larger effects.

A word on multiple comparisons.

When performing multiple comparisons (e.g. between different pairs of algorithms evaluated in the same setting), the probability to have at least one false positive increases linearly with the number of comparisons n c .

This probability is called the Family-Wise Error Rate (FWER).

To correct for this effect, one must apply corrections.

The Bonferroni correction for instance adapts the confidence level ?? Bonf.

= ??/n c BID13 .

This ensures that the FWER stays below the initial ??.

Using this corrections makes each test more conservative and decreases its statistical power.

Comparing full learning curves.

Instead of only comparing the final performances of the two algorithms after T timesteps in the environment, we can compare performances along learning.

This consists in performing a statistical comparison for every evaluation step.

This might reveal differences in speed of convergence and can provide more robust comparisons.

Further discussions on how this relates to the problem of multiple comparison is given in the supplementary materials.

In conclusion, this paper advocates for the use of Welch's t-test with low confidence level (?? < 0.05) to ensure a false positive rate below ?? * < 0.05.

The sample size must be selected carefully depending on the expected relative effect size.

It also warns against the use of other unreliable tests, such as the bootstrap test (for N < 50), the Mann-Whitney and the ranked t-test (unless assumptions are carefully checked), or the permutation test (for N < 10).

Using the t-test or the Welch's t-test with small sample sizes (<5) usually leads to high false positive rate and would require very large relative effect sizes (over = 2) to show good statistical power.

Sample sizes above N = 20 generally meet the requirement of a 0.8 statistical power for a relative effect size = 1.

Algorithm 1 represents the pseudo-code of the experiment.

The whole code can be found at https: //github.com/ccolas/rl_stats.

distributions refers to a list of pairs of distributions.

When comparing tests for an equal distribution setting, the pairs represent twice the same type of distribution.

When comparing for an unequal variance setting, the standard deviation of the second distribution is doubled.

The number of repetitions is set to 10.000.

The rejection variable refers to the rejection of the null hypothesis.

The false positive error rates can be found in results_array [ for i_t, test in tests do 5: for i_e, effect_size in effect_sizes do 6: for i_ss, N in sample_sizes do for i_r = 1: nb_repets do 9: distrib BID0 .shift(effect) 10: sample1 = distrib [0] .sample(N) 11: sample2 = distrib BID0 .sample(N) 12: rejection_list.append(test.test(sample1, sample2, ??)) 13: results_array[i_d, i_t, i_e, i_ss] = mean(rejection_list)

The correction to apply when comparing two learning curves depends 1) on the number of comparisons, 2) on the criteria that is used to conclude whether an algorithm is better than the other.

The criteria used to draw a conclusion must be decided before running any test.

An example can be: if when comparing the last 100 performance measures of the two algorithms, more than 50 comparisons show a significant difference, then Algorithm 1 is better than Algorithm 2.

In that case, the number of comparisons performed is N c = 100, and the criterion is N rejection > N crit = 50.

We want to constrain the probability FWER that our criterion is met by pure chance to a confidence level ??=0.05.

This probability is: FWER = ????N c /N crit .

To make it satisfy FWER = 0.05, we need to correct ?? such as ?? corrected = ????N crit /N c (?? corrected = ??/2 in our case).

Table 6 : Statistical power when comparing samples from two bimodal distribution with different standard deviation.

The first is centered in 0 (?? 1 = 0, ?? 1 = 1 mean or median depending on the test), the other shifted by the relative effect size (?? 2 = ?? pool , ?? 2 = 2).

Both have same standard deviation ?? 1 = ?? 2 = 1.

Each result represents the percentage of true positive over 10.000 repetitions.

In bold are results satisfying a true positive rate above 0.8.

@highlight

This paper compares statistical tests for RL comparisons (false positive, statistical power), checks robustness to assumptions using simulated distributions and empirical distributions (SAC, TD3), provides guidelines for RL students and researchers.
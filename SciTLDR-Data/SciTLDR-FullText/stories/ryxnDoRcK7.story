We develop the Y-learner for estimating heterogeneous treatment effects in experimental and observational studies.

The Y-learner is designed to leverage the abilities of neural networks to optimize multiple objectives and continually update, which allows for better pooling of underlying feature information between treatment and control groups.

We evaluate the Y-learner on three test problems: (1) A set of six simulated data benchmarks from the literature.

(2) A real-world large-scale experiment on voter persuasion.

(3) A task from the literature that estimates artificially generated treatment effects on MNIST didgits.

The Y-learner achieves state of the art results on two of the three tasks.

On the MNIST task, it gets the second best results.

We consider the problem of estimating the Conditional Average Treatment Effect (CATE) in randomized experiments and observational studies.

The CATE is a desirable quantity to estimate, because it allows us to measure how well a given treatment works for an individual conditioned on their observed covariates.

Thus, the CATE allows us to better understand the underlying causal mechanisms at play and better personalize treatments at an individual level.

Because of its promise, CATE estimation appears across a wide range of disciplines including political science, medicine, economics, and digital experiments BID7 BID15 BID20 BID2 BID6 BID21 BID9 BID12 .

CATE estimation has been an especially active area of research in the past year.

In BID10 , the authors develop the X-learner and the U-learner.

Both of these methods are so-called "meta-learners," CATE estimation strategies that can be carried out with any sufficiently generic function approximator as a base learner.

The authors primarily consider Random Forests and Bayesian Additive Regression Trees (BART) as base learners.

In doing so, they are able to provide several convergence guarantees that relate the size of the treatment and control groups to the efficiency of the estimation strategy.

Meanwhile, in BID14 the R-learner is introduced.

The authors show that the R-learner delivers excellent performance on extant benchmarks, especially when it is parameterized by deep neural networks.

The paper also provides a "quasi-oracle" regret bound for non-parametric regression problems, which they apply to the R-learner.

Motivated by these recent advances, we seek to answer the question: is there a more efficient neural network architecture for CATE estimation?

Recent work has been constrained, both by its desire to incorporate formal guarantees and by its desire to work with any general function approximator.

While these are worthwhile goals, we are curious how much performance can be improved by designing a CATE estimation strategy that takes advantage of the unique properties of neural networks.

In particular, deep neural networks can be continually optimized.

This stands in contrast to other estimators like RF and BART, which can not be meaningfully updated once trained.

While this distinction may seem small, it crucially allows a single neural networks to be asynchronously optimized with respect to several distinct objectives.

It also allows multiple networks to "co-learn," continually training on small amounts of data and staying in step with one another.

Ultimately, we show how one can leverage these properties of neural networks to create a learner that achieves state of the art performance with only a fraction of the data on several CATE estimation tasks.

We call our new learner the Y-learner.

Code for our experiments will be released at publication and is available to reviewers upon request.

Consider a randomized experiment.

In this experiment, there is a population P. Each member of the population shares a common feature space X .

Denote by X i ??? R d the features for population member i.

We are interested in how each of these population members responds to some treatment.

Let W i ??? {0, 1} be 0 if X i is in the control group (does not receive treatment) and 1 if X i does receive treatment.

Further, let Y i (1) ??? R be the response of member i when receiving treatment and Y i (0) ??? R be the response of member i when not receiving treatment.

Within the causal inference literature, Y i (0) and Y i (1) are called potential outcomes and the above framework is called the potential outcomes framework BID18 .Let us consider a concrete example that is a favorite in introduction to economics courses.

We would like to measure the impact of going to college on future income.

It is our intuition that individuals who go to college should earn more.

To verify this, we can measure the average treatment effect DISPLAYFORM0 .

While the ATE is a useful diagnostic, it only tells us the impact of treatment over an entire population.

It can tell us that, on average, going to college will improve future earnings.

It can not recommend whether individual i should go to college based on his profile X i .

As an additional problem, the ATE is also susceptible to treatment and control group selection bias.

On average, people with greater academic skills tend to go to college.

But then who's to say whether their improved income is because of their college education or because they were simply more skilled to begin with?

To offer more personalized recommendations, we need to consider the Conditional Average Treatment Effect (CATE), defined by DISPLAYFORM1 (1) Unfortunately, Equation (1) is difficult to estimate.

For a given individual X i , it is not possible observe both the outcome under treatment Y (1) and the outcome under control Y (0).

You cannot clone a college-bound individual and force the clone to skip college instead just so you can measure both outcomes.

While the situation may seem grim, if we are willing to make two strong assumptions then we can make progress on estimating the CATE.

The first assumption, called Ignorability, addresses the selection bias issue we discussed above BID17 .

It prevents the existence of a random variable that influences the probability of treatment and the potential outcomes.

The second assumption, called Overlap, ensures that no part of X i lets you uniquely identify whether individual i will be assigned to treatment or control BID3 .

For example, it prevents a situation wherein every individual under the age of 18 is automatically in the control group.

These assumptions are strong, but nevertheless standard.

They are true by design in randomized experiments BID3 BID10 BID14 .

DISPLAYFORM2 Assumption 2 (Overlap) Then there exists constant 0 < e min , e max < 1 such that for all x ??? Support(X), 0 < e min < e(x) < e max < 1.

Where e(x), the propensity score of x is defined by e(x) := P(W = 1|X = x).These two assumptions, plus regularity conditions, allow one to identify the CATE.

We can estimate the CATE by proceeding as follows.

Define DISPLAYFORM3 , where ?? 1 is the treatment response function.

It denotes the outcomes of the units who received treatment.

?? 0 is defined analogously for control units.

To estimate ?? (x) (the CATE), we compute estimates?? 0 ,?? 1 for ?? 1 and ?? 0 and then subtract to get DISPLAYFORM4 Below, we will discuss four common strategies for estimating?? .

In the T-Learner, we estimate?? 0 and?? 1 directly with any arbitrary function approximator.

Let f i be such a function approximator.

Then?? DISPLAYFORM0 We then estimate the CATE by taking differences: DISPLAYFORM1 Under strong assumptions, it is possible to provide convergence rate guarantees for the T-learner BID10 BID14 .

In spite of its simplicity, the T-learner is almost always insufficient because it does not share information across the treatment and control outcome estimators BID1 .

The S-learner tries to be more efficient than the T-learner by sharing information across the treatment and control estimators.

It uses only a single function approximator, f .

In addition to the input features x i , this function approximator also receives the binary treatment indicator w. DISPLAYFORM0 The CATE is then estimated as before.

We first consider the U-Learner.

Let M is the main treatment effect function defined by DISPLAYFORM0 and e be the treatment propensity given by P (W = 1|X = x].

The U-learner weights the estimated treatment effect by an estimated treatment propensity to form the CATE estimator.

More concretely,?? DISPLAYFORM1 The R-Learner learner was proposed in BID14 .

It is an extension to the U-learner that provides some regularization and breaks estimation into a two step process.

The authors prove the R-learner has several nice convergence guarantees.

They also demonstrate that it achieves state of the art performance on several problems.

See BID14 for more details on the R-Learner.

This procedure makes use of imputed treatment effects to transfer information between treatment and control.

Define?? 0 and?? 1 as in the T-learner.

For each of these estimates, we can produce a corresponding imputed treatment effect DISPLAYFORM0 Note that this learner does in fact use the control estimator ?? 0 on the treatment data X 1 and similarly for ?? 1 and X 0 .

This is the correct way to impute the treatment effect estimate from?? 0 and?? 1 .

From here, we can get to the CATE by estimating the imputed treatment effects and then summing the estimates?? DISPLAYFORM1 For a theoretically grounded justification of this procedure, including convergence rate analysis, see BID10 .

In addition to the work discussed above BID14 BID10 BID1 , there exists a variety of interesting work.

We are particularly interested in work that develops better CATE estimation strategies, and in work about estimating causal effects with neural networks.

In BID16 , the author shows how to use autoencoders for generalized neighborhood matching, which one method of generating contractuals for estimating individual and average treatment effects.

BID13 is concerned with domain adaptation using causal inference to handle shifts caused by measurements taken in different contexts.

Meanwhile, in BID9 , representation learning via neural networks and domain adaptation are is used to answer problems from counterfactural inference.

BID12 considers the use of deep latent variable models to handle confounders.

In BID0 , parallels are drawn between causal inference and multi-task learning.

Finally, in ) the authors develop an Integral Probability Metric based algorithm for measuring the ITE.

We are eager to hear about more related work in this area, so please let us know if we have missed anything.

5/17/2018 y_learner Our development of the Y-learner started by examining a deficiency in the X-learner.

Recall that the X-learner is a two step procedure.

In the first stage, the outcome functions,?? 0 and?? 1 , are estimated and the individual treatment effects are imputed: DISPLAYFORM0 In the second stage, estimators for the CATE are derived by regressing the features X on the imputed treatment effects.

DISPLAYFORM1 In the X-learner paper, random forests were used to obtain the estimates?? 0 and?? 1 .

However, suppose we used neural networks instead.

In fact, suppose f ??0 and f ??1 estimate ?? 0 and ?? 1 .

Then we can write DISPLAYFORM2 .

Suppose we also want to use neural networks for the second stage.

Then we can writ?? DISPLAYFORM3 When written in this way, it is clear that we should at least try to jointly optimize f ?? and f ??i .

That is, when we are optimizing f ??1 , we should also backprop through the netwowk f ??0 and similarly for f ??0 and f ??1 .

If we were using random forests, capturing this dependence would not be possible since random forests are largely fixed once trained.

However, with neural networks this presents no problem.

Neural networks also allow us to do some additional housekeeping.

For instance, we only need to keep a single neural network to output the imputed treatment effects under this joint optimization strategy.

Further, a two-stage estimation procedure is no longer necessary.

We can simply train the imputation networks and the CATE estimation networks concurrently on the same data.

The algorithm is presented as Algorithm 1 and also as a diagram in FIG0 .

DISPLAYFORM4 Update the network f ??0 to predict Y obs i 3:Update the network f ??1 to predict Y DISPLAYFORM5 Update the network f ?? to predict f ??1 ( DISPLAYFORM6 Update the network f ??0 to predict DISPLAYFORM7 Update the network f ??1 to predict Y obs i

Update the network f ?? to predict DISPLAYFORM0

While testing the Y-learner, we made an curious discovery.

We noticed that it almost always obtained much better performance than the X-learner.

This was not surprising, because we figured the joint optimization strategy of backpropgating through f ??1 and f ??0 in lines 4 and 9 of Algorithm 1 would allow those estimators to more directly benefit the final CATE estimation network f ?? .

However, if we stopped gradients from going through f ??0 and f ??1 when backpropogating through f ?? , we saw there was no major loss in performance.

The Y-learner still outperformed the X-learner by a large amount.

This seemed strange to us, since the Y-learner is structurally quite similar to the X-learner.

One key difference between the two is that the Y-learner updates f ??0 , f ??1 , and f ?? continuously and in-step with one another, whereas in the X-learner the imputation networks D 0 and D 1 are fixed before training the CATE estimation networks f ??1 and f ??0 .

We hypothesized that perhaps this continual ' 'co-learning" process may help improve training.

In other problems, such as generative adversarial networks, it is well known that the learning rate for co-learning networks is important.

If one network learns too fast or too slow, it will make the other network unstable BID5 .

In certain imitation learning algorithms, there is a more direct analogy.

In these algorithms, one co-learns two networks: One critic network to tell the agent what to do and another action network to actually do it.

Suppose this algorithm is run to completion.

Subsequently we use the fully trained critic network to train a new action network from scratch.

This seems like it should work, but it will usually fail BID8 .To test the effect of co-learning on the Y-learner, we ran the following experiment.

On one of the simulated datasets from Section 4.1, we ran 4 learners.

First, the standard X-learner with neural networks.

Second, the Y-learner with full backpropogation through f ??0 , f ??1 when training f ?? .

This is labeled 'Y.' Third, the Y-learner with no backpropogation through f ??0 , f ??1 when training f ?? .

This is labeled 'Y no backprop.'

For the final learner, we train a Y-learner to completion.

We then hold the trained f ?? fixed and use the same dataset to train a new f ??1 and f ??2 from scratch.

Finally, we hold the f ??1 and f ??2 that we just trained fixed and use them to train a new f ?? from scratch.

The goal of the last learner is to test the importance of co-learning f ??1 , f ??2 , and f ?? for the Y-learner.

We label this experiment 'no co-learning.'

To our surprise, the no co-learning experiment performed much worse than the standard y-learner and the y-learner no backprop experiments.

This is evidence supporting our conjecture that co-learning is an important component of the Y-learner.

Further research in this area is likely needed to draw more definitive conclusions.

Under review as a conference paper at ICLR 2019 The first task we consider consider is a synthetic data benchmark used in BID10 .

This benchmark has six different data generating process.

Each synthetic dataset is designed to present some difficulty in estimating the CATE.

For example, the treatment propensity might be unbalanced, the relationship between the treatment effect and the outcome might be complex, or there might be confounding variables.

See BID10 for a full description of all of the data generating processes.

Figure 3: Performance of R, X, Y, S, and T learners on six simulated data benchmark tasks.

The data is synthetically generated to make estimating the CATE difficult.

We see that the Y-learner delivers the best performance on simulations 1, 2, and 4.

On simulations 3, 5, and 6 it delivers comparable final performance to all extant methods.

On most simulations, the Y-learner requires the least data to learn a good CATE estimate. .

Figure 4 : Total training time in seconds for the S, T, R, X, and Y-learners on simulated dataset 2.

We see that the X and Y learners are roughly twice as expensive as the simpler T learners.

The S-learner requires about half the compute of the T-learner, making it the cheapest option.

Due to the R-learner's two step estimation procedure, it takes an order of magnitude longer.

This experiment was designed to measure the impact of social pressure on voter turnout in US elections BID4 .

This first version of this task was developed in BID14 , though it was later removed from that paper for unknown reasons.

A newer version was proposed in BID11 .

In this task, MNIST digits are given a treatment effect.

The value of the treatment effect is a function of the number depicted in the image.

The task is interesting because the input data is an image.

Traditional CATE estimation strategies were not capable of learning treatment effects from raw image inputs.

However, when CATE estimators are parameterized by neural networks, image inputs present no special challenges.

In this paper, we proposed the Y-learner for CATE estimation.

The Y-learner was designed specifically with neural networks in mind.

It takes advantage of the ability of neural networks to continually optimize against multiple objectives.

We noted that the Y-learner was differentiated from the Xlearner by its co-learning strategy.

The Y-learner achieves excellent performance on three benchmark Figure 6 : Results on the MNIST task.

The X, R, and T learners have fairly flat learning curves and end with MSEs of 12.8, 8.6, and 14.1 respectively, so they are omitted here.

The S-learner does much better than the Y-learner until there are around 14,000 points in the training set.problems, including one simulated data benchmark, one real data benchmark, and one benchmark that estimated CATEs over images.

We are left with several open questions.

While we did not perform a theoretical analysis on the convergence rate of the Y-learner, it seems likely that the tools from BID14 BID10 would allow us to do so.

There exists a body of related work on imputing or otherwise handling missing counterfacturals using deep learning techniques.

The Y-learner too provides a technique for imputing the missing counterfacturals needed for CATE estimation.

It would be investigate the links between our scheme and the the recently proposed methods surveyed in this paper.

As always, the problem of dealing with confounding variables remains an interesting one.

It would be interesting to adapt the Y-learner so that it can tackle this problem more directly.

@highlight

We develop a CATE estimation strategy that takes advantage some of the intriguing properties of neural networks. 

@highlight

Shows improvements to X-learner by modeling the treatment response function, the control response function, and the mapping from imputed treatment effect to the conditional average treatment effect, as neural networks.

@highlight

The authors propose the Y-learner to estimate conditional average treatment effect(CATE), which simultaneously updates the parameters of the outcome functions and the CATE estimator.
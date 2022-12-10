We develop new algorithms for estimating heterogeneous treatment effects, combining recent developments in transfer learning for neural networks with insights from the causal inference literature.

By taking advantage of transfer learning, we are able to efficiently use different data sources that are related to the same underlying causal mechanisms.

We compare our algorithms with those in the extant literature using extensive simulation studies based on large-scale voter persuasion experiments and the MNIST database.

Our methods can perform an order of magnitude better than existing benchmarks while using a fraction of the data.

The rise of massive datasets that provide fine-grained information about human beings and their behavior provides unprecedented opportunities for evaluating the effectiveness of treatments.

Researchers want to exploit these large and heterogeneous datasets, and they often seek to estimate how well a given treatment works for individuals conditioning on their observed covariates.

This problem is important in medicine (where it is sometimes called personalized medicine) (Henderson et al., 2016; Powers et al., 2018) , digital experiments (Taddy et al., 2016) , economics (Athey and Imbens, 2016) , political science (Green and Kern, 2012) , statistics (Tian et al., 2014) , and many other fields.

Although a large number of articles are being written on this topic, many outstanding questions remain.

We present the first paper that applies transfer learning to this problem.

In the simplest case, treatment effects are estimated by splitting a training set into a treatment and a control group.

The treatment group receives the treatment, while the control group does not.

The outcomes in those groups are then used to construct an estimator for the Conditional Average Treatment Effect (CATE), which is defined as the expected outcome under treatment minus the expected outcome under control given a particular feature vector (Athey and Imbens, 2015) .

This is a challenging task because, for every unit, we either observe its outcome under treatment or control, but never both.

Assumptions, such as the random assignment of treatment and additional regularity conditions, are needed to make progress.

Even with these assumptions, the resulting estimates are often noisy and unstable because the CATE is a vector parameter.

Recent research has shown that it is important to use estimators which consider both treatment groups simultaneously (Künzel et al., 2017; Wager and Athey, 2017; Nie and Wager, 2017; Hill, 2011) .

Unfortunately, these recent advances are often still insufficient to train robust CATE estimators because of the large sample sizes required when the number of covariates is not small.

In this paper, we show how these difficulties in estimating the CATE can sometimes be overcome through the use of transfer learning.

In particular, we provide several strategies for utilizing ancillary datasets that are related to the causal mechanism under investigation.

Examples of such datasets include observations from: experiments in different locations on different populations, different treatment arms, different outcomes, and non-experimental observational studies.

We show that, by transferring information from these ancillary datasets, CATE estimators can converge to better solutions with fewer samples.

This is particularly important for CATE estimation, as the cost of collecting additional data is quite high and often requires real-world data collection.

Our contributions are as follows:1.

We introduce the new problem of transfer learning for estimating heterogeneous treatment effects.2.

MLRW Transfer for CATE Estimation adapts the idea of meta-learning regression weights (MLRW) to CATE estimation.

By using a learned initialization, regression problems can be optimized much more quickly than with random initializations.

Though a variety of MLRW algorithms exist, it is not immediately obvious how one should use these methods for CATE estimation.

The principal difficulty is that CATE estimation requires the simultaneous estimation of outcomes under both treatment and control, but we only observe one of the outcomes for any individual unit.

Most MLRW transfer methods optimize on a per-task basis to estimate a single quantity.

We show that one can overcome this problem with clever use of the Reptile algorithm (Nichol et al., 2018) .3.

We provide several additional methods for transfer learning for CATE estimation: warm start, frozen-features, multi-head, and joint training.4.

We apply our methods to difficult data problems and show that they perform better than existing benchmarks.

We reanalyze a set of large field experiments that evaluate the effect of a mailer on voter turnout in the 2014 U.S. midterm elections (Gerber et al., 2017) .

This includes 17 experiments with 1.96 million individuals in total.

We also simulate several randomized controlled trials using image data of handwritten digits found in the MNIST database (LeCun, 1998) .

We show that our methods, MLRW in particular, obtain better than state-of-the-art performance in estimating CATE, and that they require far fewer observations than extant methods.

We provide open source code for our algorithms.

We begin by formally introducing the CATE estimation problem.

Following the potential outcomes framework (Rubin, 1974) , assume there exists a single experiment wherein we observe N i.i.d.

distributed units from some super population, DISPLAYFORM0 0) 2 R denotes the potential outcome of unit i if it is in the control group, Y i (1) 2 R is the potential outcome of i if it is in the treatment group, X i 2 R d is a d-dimensional feature vector, and W i 2 {0, 1} is the treatment assignment.

For each unit in the treatment group (W i = 1), we only observe the outcome under treatment, Y i (1).

For each unit under control (W i = 0), we only observe the outcome under control.

Crucially, there cannot exist overlap between the set of units for which W i = 1 and the set for which W i = 0.

It is impossible to observe both potential outcomes for any unit.

This is commonly referred to as the fundamental problem of causal inference.

However, not all hope is lost.

We can still estimate the Conditional Average Treatment Effect (CATE) of the treatment.

Let x be an individual feature vector.

Then the CATE of x, denoted ⌧ (x), is defined by DISPLAYFORM1 Estimating ⌧ is impossible without making further assumptions on the distribution of DISPLAYFORM2 In particular, we need to place two assumptions on our data.

Assumption 1 (Strong Ignorability, Rosenbaum and Rubin (1983) ) DISPLAYFORM3 Define the propensity score of x as, e(x) := P(W = 1|X = x).

Then there exists constant 0 < e min , e max < 1 such that for all x 2 Support(X), 0 < e min < e(x) < e max < 1.In words, e(x) is bounded away from 0 and 1.1 The software will be released once anonymity is no longer needed.

We can also provide an anynomized copy to reviewers upon request.

Assumption 1 ensures that there is no unobserved confounder, a random variable which influences both the probability of treatment and the potential outcomes, which would make the CATE unidentifiable.

The assumption is particularly strong and difficult to check in applications.

Meanwhile, Assumption 2 rectifies the situation wherein a certain part of the population is always treated or always in the control group.

If, for example, all women were in the control group, one cannot identify the treatment effect for women.

Though both assumptions are strong, they are nevertheless satisfied by design in randomized controlled trials.

While the estimators we discuss would be sensible in observational studies when the assumptions are satisfied, we warn practitioners to be cautious in such studies, especially when the number of covariates is large (D'Amour et al., 2017) .Given these two assumptions, there exist many valid CATE estimators.

The crux of these methods is to estimate two quantities: the control response function, DISPLAYFORM4 , and the treatment response function, DISPLAYFORM5 If we denote our learned estimates asμ 0 (x) andμ 1 (x), then we can form the CATE estimate as the difference between the two⌧ (x) =μ 1 (x) μ 0 (x).

The astute reader may be wondering why we don't simply estimate µ 0 and µ 1 with our favorite function approximation algorithm at this point and then all go home.

After all, we have access to the ground truths µ 0 and µ 1 and the corresponding inputs x.

In fact, it is commonplace to do exactly that.

When people directly estimate µ 0 and µ 1 with their favorite model, we call the procedure a T-learner (Künzel et al., 2017) .

Common choices of models include linear models and random forests, though neural networks have recently been considered (Nie and Wager, 2017) .A practitioner of deep learning might find the T-learner quite trivial.

After all, it amounts to using neural networks to fit two quantities, µ 0 and µ 1 .

However, it is important to note that the T-learner is a baseline method.

We use it in this paper only to ease exposition, especially as it relates to transfer learning.

The T-learner has many drawbacks (Athey and Imbens, 2015) .

It is almost always an inefficient estimator.

For example, it will often perform poorly when one can borrow information across the treatment conditions.

For these reasons, more sophisticated learners such as the S, X, T, R, and Y learners are almost always used instead of the T-learner (Hill, 2011; Athey and Imbens, 2016; Nie and Wager, 2017; Künzel et al., 2017; Stadie et al., 2018) .

Although much of our exposition will focus on transfer learning in the context of the T-learner, in practice we extend the discussed methods to these other more advanced learners, as shown in the Evaluation section.

Descriptions of these more advanced estimators are given in the appendix.

In this section, we consider a scenario wherein one has access to many related causal inference experiments.

The goal is to use the results from some old experiments to obtain faster training with less data on other new experiments.

Since direct transfer between different populations is wrought with difficulty, we will instead achieve transfer by using previous experiments to help find an initialization for new experiments which leads to faster optimization.

2 We consider two kinds of algorithms.

First, there are transfer algorithms that sit on top of existing CATE estimators.

These transfer algorithms take a CATE estimation strategy, such as the S-learner, and provide a recipe for transforming it into a transfer learning CATE estimator.

The second class of algorithms does not sit on top of existing CATE estimation strategies.

Instead, they are built from the ground up to take advantage of transfer learning.

These algorithms are joint training and MLRW.Across all experiments, the input space X is the same.

Let i index an experiment.

Each experiment has its own distinct outcome when treatment is received, µ 1,i (x), and when no treatment is received, µ 0,i (x).

Together, these quantities define the CATE ⌧ ·,i (x) = µ 1,i (x) µ 0,i (x), In standard CATE estimation, we define a strategy that takes x as input and outputs predictionsμ 0,i (x) andμ 1,i (x).

In transfer learning, the hope is that we can transfer knowledge between experiments.

The model parameters that allowed us predict µ 0,i (x), µ 1,i (x) and ⌧ ·,i (x) from experiment i should help us predict µ 1,j (x), µ 0,j (x), and ⌧ ·,j (x) from experiment j.

Let ⇡ ✓ be a generic expression for a neural network parameterized by ✓.

Parameters will have two subscripts.

The index on the left indicates if their neural network predicts treatment or control (0 for control and 1 for treatment).

The index on the right is for the experiment.

For example, ✓ 0,2 parametrizes ⇡ ✓0,2 (x) to predict µ 0,2 (x), the outcome under control for Experiment 2.

All of the transfer algorithms described here are presented in full detail as pseudo-code in the appendix.

These algorithms extend existing CATE estimation techniques to the transfer learning setting.

The following exposition is largely motivated by transfer learning with the T-Learner as a base CATE estimator.

This is only for ease of exposition.

The discussed procedures can extend to other, more complicated, CATE estimators such as the R, X, Y, and S learners.

Warm start: Experiment 0 predicts ⇡ ✓0,0 (x) =μ 0,0 (x) and ⇡ ✓1,0 (x) =μ 1,0 (x) to form the CATE estimator⌧ ·,0 =μ 1,0 (x) μ 0,0 (x).

Suppose ✓ 0,0 , ✓ 1,0 are fully trained and produce a good CATE estimate.

For experiment 1, the input space X is identical to the input space for experiment 0, but the outcomes µ 0,1 (x) and µ 1,1 (x) are different.

However, we suspect the underlying data representations learned by ⇡ ✓0,0 and ⇡ ✓1,0 are still useful.

Hence, rather than randomly initializing ✓ 0,1 and ✓ 1,1 for experiment 1, we set ✓ 0,1 = ✓ 0,0 and ✓ 1,1 = ✓ 1,0 .

We then train ⇡ ✓0,1 (x) =μ 0,1 (x) and ⇡ ✓1,1 (x) =μ 1,1 (x).

See FIG0 and Algorithm 8 in the appendix.

Frozen-features: Begin by training ⇡ ✓0,0 and ⇡ ✓1,0 to produce good CATE estimates for experiment 0.

Assuming ✓ 0,0 and ✓ 1,0 have more than k layers, let 0 be the parameters corresponding to the first k layers of ✓ 0,0 .

Define 1 analogously.

Since we think the features encoded by ⇡ i (X) would make a more informative input than the raw features X, we want to use those features as a transformed input space for ⇡ ✓0,1 and ⇡ ✓1,1 .

To wit, set z 0 = ⇡ 0 (x) and z 1 = ⇡ 1 (x).

Then form the estimates ⇡ ✓0,1 (z 0 ) =μ 0,1 (x) and ⇡ ✓1,1 (z 1 ) =μ 1,1 (x).

During training of experiment 1, we only backpropagate through ✓ 0,1 and ✓ 1,1 and not through 0 and 1 .

See FIG0 and Algorithm 9 in the appendix.

Multi-head: In this setup, all experiments share base layers that are followed by experiment-specific layers.

The intuition is that the base layers should learn general features, and the experimentspecific layers should transform those features into estimates of µ j,i (x).

More concretely, let 0 and 1 be shared base layers for estimating µ 0,· (x) and µ 1,· (x) respectively.

Set z 0 = ⇡ 0 (x 0 ) and z 1 = ⇡ 1 (x 1 ).

The base layers are followed by experiment-specific layers 0,i and 1,i .

Let DISPLAYFORM0 Training alternates between experiments: each ✓ 0,i and ✓ 1,i is trained for some small number of iterations, and then the experiment and head being trained are switched.

Every head is usually trained several times.

See FIG0 Algorithm 10 in the appendix.

SF Reptile transfer for CATE estimators: Pick your favorite CATE estimator.

The goal is to learn an initialization for that CATE estimator's weights that leads to fast convergence on new experiments.

More concretely, starting from good initializers ✓ 0 and ✓ 1 , one can train neural networks ⇡ ✓0 and ⇡ ✓1 to estimate µ 0,i (x) and µ 1,i (x) much faster and with less data than starting from random initializations.

To learn these good initializations, we use a transfer learning technique called Reptile.

The idea is to perform experiment-specific inner updates U (✓) and then aggregate them into outer updates of the form ✓ new = ✏ · U (✓) + (1 ✏) · ✓.

In this paper, we consider a slight variation of Reptile.

In standard Reptile, ✏ is either a scalar or correlated to per-parameter weights furnished via SGD.

For our problem, we would like to encourage our network layers to learn at different rates.

The hope is that the lower layers can learn more general, slowly-changing features like in the frozen features method, and the higher layers can learn comparatively faster features that more quickly adapt to new tasks after ingesting the stable lower-level features.

To accomplish this, we take the path of least resistance and make ✏ a vector which assigns a different learning rate to each neural network layer.

Because our intuition involves slow and fast weights, we will refer to this modification in this paper as SF Reptile: Slow Fast Reptile.

Though this change is seemingly small, we found it boosted performance on our problems.

See Algorithm 11.

Joint training: All predictions share base layers ✓.

From these base layers, there are two heads per experiment i: one to predict µ 0,i (x) and one to predict µ 1,i (x).

Every head and the base features are trained simultaneously by optimizing with respect to the loss function DISPLAYFORM0 k and minimizing over all weights.

This will encourage the base layers to learn generally applicable features and the heads to learn features specific to predicting a single µ j,i (x).

See Algorithm 6.

MLRW transfer: In this method, there exists one single set of weights ✓.

There are no experimentspecific weights.

Furthermore, we do not use separate networks to estimate µ 0 and µ 1 .

Instead,

✓ is trained to estimate one µ i,j (x) at a time.

We train ✓ with SF Reptile so that in the future ⇡ ✓ requires minimal samples to fit µ i,j (x) from any experiment.

To actually form the CATE estimate, we use a small number of training samples to fit ⇡ ✓ to µ 0,i (x) and then a small number of training samples to fit ⇡ ✓ to µ 1,i (x).

We call ✓ meta-learned regression weights (MLRW) because they are meta-learned over many experiments to quickly regress onto any µ i,j (x).

The full MLRW algorithm is presented as Algorithm 5.

We evaluate our transfer learning estimators on both real and simulated data.

In our data example, we consider the important problem of voter encouragement.

Analyzing a large data set of 1.96 million potential voters, we show how transfer learning across elections and geographic regions can dramatically improve our CATE estimators.

To the best of our knowledge, this is the first successful demonstration of transfer learning for CATE estimation.

The simulated data has been intentionally chosen to be different in character from our real-world example.

In particular, the simulated input space is images and the estimated outcome variable is continuous.

To evaluate transfer learning for CATE estimation on real data, we reanalyze a set of large field experiments with more than 1.96 million potential voters (Gerber et al., 2017) .

The authors conducted 17 experiments to evaluate the effect of a mailer on voter turnout in the 2014 U.S. Midterm Elections.

The mailer informs the targeted individual whether or not they voted in the past four major elections (2006, 2008, 2010, and 2012) , and it compares their voting behavior with that of the people in the same state.

The mailer finishes with a reminder that their voting behavior will be monitored.

The idea is that social pressure-i.e., the social norm of voting-will encourage people to vote.

The likelihood of voting increases by about 2.2% (s.e.=0.001) when given the mailer.

Each of the experiments targets a different state.

This results in different populations, different ballots, and different electoral environments.

In addition to this, the treatment is slightly different in each experiment, as the median voting behavior in each state is different.

However, there are still many similarities across the experiments, so there should be gains from transferring information.

In this example, the input X is a voter's demographic data including age, past voting turnout in 2006, 2008, 2009, 2010, 2011, 2012, and 2013 , marital status, race, and gender.

The treatment response functionμ 1 (x) estimates the voting propensity for a potential voter who receives a mailer encouraging them to vote.

The control response functionμ 0 estimates the voting propensity if that voter did not receive a mailer.

The CATE ⌧ is thus the change in the probability of voting when a unit receives a mailer.

The complete dataset has this data over 17 different states.

Treating each state as a separate experiment, we can perform transfer learning across them.

Being able to estimate the treatment effect of sending a mailer is an important problem in elections.

DISPLAYFORM0 We may wish to only treat people whose likelihood of voting would significantly increase when receiving the mailer, to justify the cost for these mailers.

Furthermore, we wish to avoid sending mailers to voters who will respond negatively to them.

This negative response has been previously observed and is therefore feasible and a relevant problem-e.g., some recipients call their Secretary of State's office or local election registrar to complain (Mann, 2010; Michelson, 2016) .Evaluating a CATE estimator on real data is difficult.

The primary difficulty is that we do not get to observe the true CATE for any unit, due to the fundamental problem of causal inference.

By definition, only one of the two outcomes is observed for any unit.

One could use the original features and simulate the outcome features, but this would require us to create a response model.

Instead, we estimate the "truth" on the real data using linear models (version 1) or random forests (version 2).

We then construct the data based on these estimates.

For a detailed description, see Appendix A.2.

We then ask the question: How do the various methods perform when they have less data than the entire sample?

We evaluate all the algorithms discussed in section 3 on the GOTV dataset.

For the algorithms in section 3.0.1 that require a base CATE estimator, we use the Y learner because we found it delivered the best performance.

3 For baselines, we compare against the non-transfer Y-learner and the S learner with random forests.

4 In previous work, state of the art results on this problem have been achieved with both non-transfer tree-based estimators such as S-RF (Künzel et al., 2017; Green and Kern, 2012) and neural-network-based learners such as the R and Y-learners (Nie and Wager, 2017; Stadie et al., 2018) .

The best estimator is MLRW.

This algorithm consistently converges to a very good solution with very few observations.

Looking at Tables 1, 2 , and 3, we observe that MLRW is the best performing transfer learner for GOTV version 1 in 8 out of 17 trials.

In GOTV version 2, it is the best in 11 out of 17 trials.

In FIG1 , its average performance is dominant over all other algorithms.

We hypothesize that this method does best because it does not try to artificially bottleneck the flow of information between outcomes and experiments.

MLRW also seems more resilient to data-poisoning when it encounters outlier data, though we did not concretely test against this.

We also observe that multi-head, frozen-features, and SF all generally improve upon non-transfer baselines.

The faster learning rate of these algorithms indicates that positive transfer between experiments is occurring.

Warm start, however, does not work well and often even leads to worse results than the baseline estimators.

This is consistent with prior findings on warm start (Finn et al., 2017) .

In the previous experiment, we observed that the MLRW estimator performed most favorably, and transfer learning significantly improved upon the baseline.

To confirm that this conclusion is not specific to voter persuasion studies, we intentionally consider a very different type of data.

Recently, (Nie and Wager, 2017) introduced a simulation study wherein MNIST digits are rotated by some number of degrees ↵; with ↵ furnished via a single data generating process that depends on the value of the depicted digit.

They then attempt to do CATE estimation to measure the heterogeneous treatment effect of a digit's label.

Motivated by this example, we develop a data generating process using MNIST digits wherein transfer learning for CATE estimation is applicable.

In our example, the input X is an MNIST image.

We have k data-generating processes which return different outcomes for each input when given either treatment or control.

Thus, under some fixed data-generating process, µ 0 represents the outcome when the input image X is given the control, µ 1 represents the outcome when X is given the treatment, and ⌧ is the difference in outcomes given the placement of X in the treatment or control group.

Each data-generating process has different response functions (µ 0 and µ 1 ) and thus different CATEs (⌧ ), but each of these functions only depends on the label presented in the image X. We thus hope that transfer learning could expedite the process of learning features which are indicative of the label.

See Appendix A for full details of the data generation process.

In FIG11 , we confirm that a transfer learning strategy outperforms its non-transfer learning counterpart, even on image data.

We also see that MLRW performs well, though in this case multi-head is competitive.

We also see that several of the transfer methods are worse than non-transfer baselines.

FIG11 : MNIST task.

The baseline is the S-learner.

All transfer CATE estimators for this task are built on top the S-learner, rather than the Y-learner, because we found it delivered better performance for this problem.

In this paper, we proposed the problem of transfer learning for CATE estimation.

One immediate question the reader may be left with is why we chose the transfer learning techniques we did.

We only considered two common types of transfer: (1) Basic fine tuning and weights sharing techniques common in the computer vision literature (Welinder et al., 2010; Saenko and Darrell, 2010; Bourdev et al., 2011; Donahue et al., 2014; Koch, 2015) , (2) Techniques for learning an initialization that can be quickly optimized (Finn et al., 2017; Ravi and Larochelle, 2017; Nichol et al., 2018) .

However, many further techniques exist.

Yet, transfer learning is an extensively studied and perennial problem (Schmidhuber, 1992; Bengio et al., 1992; Thrun, 1996; Thrun and Pratt, 1998; Taylor and Stone, 2009; Silver et al., 2013) .

In (Vinyals et al., 2016) , the authors attempt to combine feature embeddings that can be utilized with non-parametric methods for transfer. (Snell et al., 2017) is an extension of this work that modifies the procedure for sampling examples from the support set during training.

BID1 and related techniques try to meta-learn an optimizer that can more quickly solve new tasks. (Rusu et al., 2016) attempts to overcome forgetting during transfer by systematically introducing new network layers with lateral connections to old frozen layers. (Munkhdalai and Yu, 2017) uses networks with memory to adapt to new tasks.

We invite the reader to review (Finn et al., 2017) for an excellent overview of the current transfer learning landscape.

Though the majority of the discussed techniques could be extended to CATE estimation, our implementations of (Rusu et al., 2016; BID1 proved difficult to tune and consequently learned very little.

Furthermore, we were not able to successfully adapt (Snell et al., 2017) to the problem of regression.

We decided to instead focus our attention on algorithms for obtaining good initializations, which were easy to adapt to our problem and quickly delivered good results without extensive tuning.

On the topic of using neural networks to improve causal inference algorithms, a flurry of relevant work exists (Ramachandra, 2018; Magliacane et al., 2017; Johansson et al., 2016; Louizos et al., 2017; BID0 Shalit et al., 2017; Nie and Wager, 2017) .

We found that these papers either did not allow us to better estimate the CATE, or else provided worse performance than the baseline methods we did consider in this paper.

Extending transfer to other causal inference algorithms is an ongoing and interesting area of research.

We are left with several open questions.

Can transfer learning still be applied to CATE estimation when the experiment input spaces differ?

How should one properly deal with missing and incomplete data?

Do there exist better methods for interpretability, highlighting which features are most important for transfer and why?

Can these techniques be extended to causal models outside of CATE estimation?

How can one properly encode causal relationships into a neural network?

Answering these questions would have a positive impact on fields such as causal inference, deep learning, and reinforcement learning.

Thrun (1996) .

Is learning the n-th thing any easier than learning the first?

NIPS.

and we define the response functions and the propensity score as DISPLAYFORM0 we fist sample a (X i , C i ) from the MNIST data set, and we then generate Y i (0), Y i (1), and W i in the following way: DISPLAYFORM1 , and Y i are made available to the convolutional neural network, which then predicts⌧ given a test image X i and a treatment W i .

⌧ is the difference in the outcome given the difference in treatment and control.

Having access to multiple DGPs can be interpreted as having access to prior experiments done on a similar population of images, allowing us to explore the effects of different transfer learning methods when predicting the effect of a treatment in a new image.

In this section, we describe how the simulations for the GOTV example in the main paper were done and we discuss the results of a much bigger simulation study with 51 experiments which is summarized in Tables 1, 2 , and 3.

For our data example, we took one of the experiments conducted by Gerber et al. (2017) .

The study took place in 2014 in Alaska and 252,576 potential voters were randomly assigned in a control and a treatment group.

Subjects in the treatment group were sent a mailer as described in the main text and their voting turnout was recorded.

To evaluate the performance of different CATE estimators we need to know the true CATEs, which are unknown due to the fundamental problem of causal inference.

To still be able to evaluate CATE estimators researchers usually estimate the potential outcomes using some machine learning method and then generate the data from this estimate.

This is to some extend also a simulation, but unlike classical simulation studies it is not up to the researcher to determine the data generating distribution.

The only choice of the researcher lies in the type of estimator she uses to estimate the response functions.

To avoid being mislead by artifacts created by a particular method, we used a linear model in Real World Data Set 1 and random forests estimator in Real World Data Set 2.Specifically, we generate for each experiment a true CATE and we simulate new observed outcomes based on the real data in four steps.1.

We first use the estimator of choice (e.g., a random forests estimator) and train it on the treated units and on the control units separately to get estimates for the response functions, µ 0 and µ 1 .

2.

Next, we sample N units from the underlying experiment to get the features and the treatment assignment of our samples DISPLAYFORM0 We then generate the true underlying CATE for each unit using DISPLAYFORM1

around mean µ i .

DISPLAYFORM0 After this procedure, we have 17 data sets corresponding to the 17 experiments for which we know the true CATE function, which we can now use to evaluate CATE estimators and CATE transfer learners.

Simulations motivated by real-world experiments are important to assess whether our methods work well for voter persuasion data sets, but it is important to also consider other settings to evaluate the generalizability of our conclusions.

We then use each of the 17 experiments to generate a simulated experiment in the following way:1.

We sample N units from the underlying experiment to get the features and the treatment assignment of our samples DISPLAYFORM0 We then generate the true underlying CATE for each unit using DISPLAYFORM1 Finally we generate the observed outcome by sampling a Bernoulli distributed variable around mean µ i .

DISPLAYFORM2 The experiments range in size from 5,000 units to 400,000 units per experiment and the covariate vector is 11 dimensional and the same as in the main part of the paper.

We will present here three different setup.

Simulation LM (Table 1) Simulation RF (Table 2) : We choose here N to be all units in the corresponding experiment.1.

Train a random forests estimator on the real data set and define µ 0 to be the resulting estimator, 2.

Sample a covariate f (e.g., age), 3.

ample a random value in the support of f (e.g., 38), 4.

Sample a shift s ⇠ N (0, 4).Now define the potential outcomes as follows: DISPLAYFORM3 Simulation RFt TAB12

In this section, we will present pseudo code for the CATE estimators in this paper.

We present code for the meta learning algorithms in Section C. We denote by Y 0 and Y 1 the observed outcomes for the control and the treated group.

For example, Y 1 i is the observed outcome of the ith unit in the treated group.

X 0 and X 1 are the features of the control and treated units, and hence, X 1 i corresponds to the feature vector of the ith unit in the treated group.

M k (Y ⇠ X) is the notation for a regression estimator, which estimates x 7 !

E[Y |X = x].

It can be any regression/machine learning estimator, but in this paper we only choose it to be a neural network or random forest.

These algorithms first appeared in (Künzel et al., 2017; Stadie et al., 2018) .

We reproduce them here for completeness.

DISPLAYFORM0 end procedure M0 and M1 are here some, possibly different machine learning/regression algorithms.

Algorithm 2 S-learner Algorithm 3 X-learner 1: procedure X-LEARNER(X, Y obs , W, g) DISPLAYFORM1 DISPLAYFORM2 .

Estimate CATE for treated and control DISPLAYFORM3 .

Average the estimates 9: end procedure g(x) 2 [0, 1] is a weighing function which is chosen to minimize the variance of⌧ (x).

It is sometimes possible to estimate Cov(⌧0(x), ⌧1(x)), and compute the best g based on this estimate.

However, we have made good experiences by choosing g to be an estimate of the propensity score, but also choosing it to be constant and equal to the ratio of treated units usually leads to a good estimator of the CATE.

Sample X 0 and X 1 : control and treatment units from experiment i 10: DISPLAYFORM4 .

j iterating over treatment and control 11: DISPLAYFORM5 for k < inneriters do 13: DISPLAYFORM6 Compute r ✓ L.

Use ADAM with r ✓ L to obtain U k+1 (✓).16: DISPLAYFORM0 end for

for p < N do 19: Sample X 0 and X 1 : control and treatment units from experiment i DISPLAYFORM0

Sample X: test units from experiment i.

for j = [0, 1] do .

j iterating over treatment and control 30: DISPLAYFORM0 Compute r ✓ L.

Use ADAM with r ✓ L to obtain U k+1 (✓).

end for be the full prediction network for µ 0 in experiment i. DISPLAYFORM0 DISPLAYFORM1 1 i be the full prediction network for µ 1 in experiment i. DISPLAYFORM2 j be all trainable parameters.

10: Let numiters be the total number of training iterations 11: for iter < numiters do Sample X 0 and X 1 : control and treatment units from experiment i 15: DISPLAYFORM3 .

j iterating over treatment and control 16: DISPLAYFORM4 DISPLAYFORM5 Apply ADAM with gradients given by r ⌦ L.

for i < numexps do

Here, we present full pseudo code for the algorithms from Section 3 using the T-learner as a base learner.

All of these algorithms can be extended to other learners including S, R, X, and Y .

See the released code for implementations.

Algorithm 7 Vanilla T-learner (also referred to as Baseline T-learner)1: Let µ 0 and µ 1 be the outcome under treatment and control.

2: Let X be the experimental data.

Let X t be the test data.

3: Let ⇡ ✓0 and ⇡ ✓1 be a neural networks parameterized by ✓ 0 and ✓ 1 .

4: Let ✓ = ✓ 0 [ ✓ 1 .

5: Let numiters be the total number of training iterations.

6: Let batchsize be the number of units sampled.

We use 64.

7: for i < numiters do 8:Sample X 0 and X 1 : control and treatment units.

Sample batchsize units.9: DISPLAYFORM0 Compute r ✓ L = @L @✓ .

Apply ADAM with gradients given by r ✓ L. 14: end for DISPLAYFORM0 Algorithm 8 Warm Start T-learner 1: Let µ i 0 and µ i 1 be the outcome under treatment and control for experiment i. 2: Let X i be the data for experiment i. Let X i t be the test data for experiment i. DISPLAYFORM1 Compute r ✓ L = @L @✓ .

Apply ADAM with gradients given by r ✓ L. 14: end for 15: for i < numiters do DISPLAYFORM0 Compute r ✓ L = @L @✓ .

Apply ADAM with gradients given by r ✓ L. 22: end for DISPLAYFORM0 Algorithm 9 Frozen Features T-learner 1: Let µ i 0 and µ i 1 be the outcome under treatment and control for experiment i. 2:

Let X i be the data for experiment i. Let X i t be the test data for experiment i. DISPLAYFORM1 Compute r ✓ L = @L @✓ .

Apply ADAM with gradients given by r ✓ 0 L. 16: end for 17: for i < numiters do DISPLAYFORM0 DISPLAYFORM1 be all trainable parameters used to predict µ i 0 .

DISPLAYFORM2 DISPLAYFORM3 18: DISPLAYFORM4 19: DISPLAYFORM5 Compute r ⌦ i L = @L @⌦ i .

Apply ADAM with gradients given by r ✓ L.

end for 25: end for 26: Let C = [] 27: for j < numexps do 28: DISPLAYFORM0 29: for i < numexps do 12: DISPLAYFORM1 DISPLAYFORM2 14:for k< numinneriters do 15:Sample X i 0 and X i 1 : control and treatment units.

Sample batchsize units.16: DISPLAYFORM3 Compute r ✓ L = @L @✓ .

Use ADAM with gradients given by r ✓ L to obtain U k+1 (✓ 0 ) and U k+1 (✓ 1 ).

Set DISPLAYFORM0 end for

for p < N do 24: DISPLAYFORM0 DISPLAYFORM1 for k< numinneriters do 34:Sample X i 0 and X i 1 : control and treatment units.

Sample batchsize units.35: DISPLAYFORM2 Compute r ✓ L = @L @✓ .

Use ADAM with gradients given by r ✓ L to obtain U k+1 (✓ 0 ) and U k+1 (✓ 1 ).40: DISPLAYFORM0 C.append(⌧ i ).

46: end for 47: return C.

Below, we include the full results for the GOTV and MNIST experiments.

In particular, we use show results for transfer CATE learners with S, T, X, and Y base learners.

We also provide full tables of more comprehensive results for all methods and all train-test splits.

Number of units in the training set in 1000 units MSE for the CATE

<|TLDR|>

@highlight

Transfer learning for estimating causal effects using neural networks.

@highlight

Develops algorithms to estimate conditional average treatment effect by auxiliary dataset in different environments, both with and without base learner.

@highlight

The authors propose methods to address a novel task of transfer learning for estimating the CATE function, and evaluate them using a synthetic setting and a real-world experimental dataset.

@highlight

Using neural network regression and comparing transfer learning frameworks to estimate a conditional average treatment effect under string ignorability assumptions
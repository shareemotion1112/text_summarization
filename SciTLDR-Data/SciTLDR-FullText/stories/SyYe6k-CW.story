Recent advances in deep reinforcement learning have made significant strides in performance on applications such as Go and Atari games.

However, developing practical methods to balance exploration and exploitation in complex domains remains largely unsolved.

Thompson Sampling and its extension to reinforcement learning provide an elegant approach to exploration that only requires access to posterior samples of the model.

At the same time, advances in approximate Bayesian methods have made posterior approximation for flexible neural network models practical.

Thus, it is attractive to consider approximate Bayesian neural networks in a Thompson Sampling framework.

To understand the impact of using an approximate posterior on Thompson Sampling, we benchmark well-established and recently developed methods for approximate posterior sampling combined with Thompson Sampling over a series of contextual bandit problems.

We found that many approaches that have been successful in the supervised learning setting underperformed in the sequential decision-making scenario.

In particular, we highlight the challenge of adapting slowly converging uncertainty estimates to the online setting.

Recent advances in reinforcement learning have sparked renewed interest in sequential decision making with deep neural networks.

Neural networks have proven to be powerful and flexible function approximators, allowing one to learn mappings directly from complex states (e.g., pixels) to estimates of expected return.

While such models can be accurate on data they have been trained on, quantifying model uncertainty on new data remains challenging.

However, having an understanding of what is not yet known or well understood is critical to some central tasks of machine intelligence, such as effective exploration for decision making.

A fundamental aspect of sequential decision making is the exploration-exploitation dilemma: in order to maximize cumulative reward, agents need to trade-off what is expected to be best at the moment, (i.e., exploitation), with potentially sub-optimal exploratory actions.

Solving this trade-off in an efficient manner to maximize cumulative reward is a significant challenge as it requires uncertainty estimates.

Furthermore, exploratory actions should be coordinated throughout the entire decision making process, known as deep exploration, rather than performed independently at each state.

Thompson Sampling (Thompson, 1933) and its extension to reinforcement learning, known as Posterior Sampling, provide an elegant approach that tackles the exploration-exploitation dilemma by maintaining a posterior over models and choosing actions in proportion to the probability that they are optimal.

Unfortunately, maintaining such a posterior is intractable for all but the simplest models.

As such, significant effort has been dedicated to approximate Bayesian methods for deep neural networks.

These range from variational methods BID17 BID6 BID23 to stochastic minibatch Markov Chain Monte Carlo (Neal, 1994; Welling & Teh, 2011; BID25 BID1 BID26 , among others.

Because the exact posterior is intractable, evaluating these approaches is hard.

Furthermore, these methods are rarely compared on benchmarks that measure the quality of their estimates of uncertainty for downstream tasks.

To address this challenge, we develop a benchmark for exploration methods using deep neural networks.

We compare a variety of well-established and recent Bayesian approximations under the lens of Thompson Sampling for contextual bandits, a classical task in sequential decision making.

All code and implementations to reproduce the experiments will be available open-source, to provide a reproducible benchmark for future development.

1 Exploration in the context of reinforcement learning is a highly active area of research.

Simple strategies such as epsilon-greedy remain extremely competitive (Mnih et al., 2015; Schaul et al., 2016) .

However, a number of promising techniques have recently emerged that encourage exploration though carefully adding random noise to the parameters (Plappert et al., 2017; BID12 BID13 or bootstrap sampling (Osband et al., 2016) before making decisions.

These methods rely explicitly or implicitly on posterior sampling for exploration.

In this paper, we investigate how different posterior approximations affect the performance of Thompson Sampling from an empirical standpoint.

For simplicity, we restrict ourselves to one of the most basic sequential decision making scenarios: that of contextual bandits.

No single algorithm bested the others in every bandit problem, however, we observed some general trends.

We found that dropout, injecting random noise, and bootstrapping did provide a strong boost in performance on some tasks, but was not able to solve challenging synthetic exploration tasks.

Other algorithms, like Variational Inference, Black Box ??-divergence, and minibatch Markov Chain Monte Carlo approaches, strongly couple their complex representation and uncertainty estimates.

This proves problematic when decisions are made based on partial optimization of both, as online scenarios usually require.

On the other hand, making decisions according to a Bayesian linear regression on the representation provided by the last layer of a deep network offers a robust and easy-to-tune approach.

It would be interesting to try this approach on more complex reinforcement learning domains.

In Section 2 we discuss Thompson Sampling, and present the contextual bandit problem.

The different algorithmic approaches that approximate the posterior distribution fed to Thompson Sampling are introduced in Section 3, while the linear case is described in Section 4.

The main experimental results are presented in Section 5, and discussed in Section 6.

Finally, Section 7 concludes.

The contextual bandit problem works as follows.

At time t = 1, . . .

, n a new context X t ??? R d arrives and is presented to algorithm A. The algorithm -based on its internal model and X t -selects one of the k available actions, a t .

Some reward r t = r t (X t , a t ) is then generated and returned to the algorithm, that may update its internal model with the new data.

At the end of the process, the reward for the algorithm is given by r = n t=1 r t , and cumulative regret is defined as R A = E[r * ??? r], where r * is the cumulative reward of the optimal policy (i.e., the policy that always selects the action with highest expected reward given the context).

The goal is to minimize R A .The main research question we address in this paper is how approximated model posteriors affect the performance of decision making via Thompson Sampling (Algorithm 1) in contextual bandits.

We study a variety of algorithmic approaches to approximate a posterior distribution, together with different empirical and synthetic data problems that highlight several aspects of decision making.

We consider distributions ?? over the space of parameters that completely define a problem instance ?? ??? ??. For example, ?? could encode the reward distributions of a set of arms in the multi-armed bandit scenario, or -more generally-all the parameters of an MDP in reinforcement learning.

Thompson Sampling is a classic algorithm (Thompson, 1933) which requires only that one can sample from the posterior distribution over plausible problem instances (for example, values or rewards).

At each round, it draws a sample and takes a greedy action under the optimal policy for the sample.

The posterior distribution is then updated after the result of the action is observed.

Thompson Sampling has been shown to be extremely effective for bandit problems both in practice BID9 BID16 and theory BID0 .

It is especially appealing for deep neural networks as one rarely has access to the full posterior but can often approximately sample from it.

In this section, we describe the different algorithmic design principles that we considered in our simulations of Section 5.

These algorithms include linear methods, Neural Linear and Neural Greedy, variational inference, expectation-propagation, dropout, Monte Carlo methods, bootstrapping, direct noise injection, and Gaussian Processes.

In FIG10 in the appendix, we visualize the posteriors of the nonlinear algorithms on a synthetic one dimensional problem.

Linear Methods We apply well-known closed-form updates for Bayesian linear regression for exact posterior inference in linear models BID5 .

We provide the specific formulas below, and note that they admit a computationally-efficient online version.

We consider exact linear posteriors as a baseline; i.e., these formulas compute the posterior when the data was generated according to Y = X T ?? + where ??? N (0, ?? 2 ), and Y represents the reward.

Importantly, we model the joint distribution of ?? and ?? 2 for each action.

Sequentially estimating the noise level ?? 2 for each action allows the algorithm to adaptively improve its understanding of the volume of the hyperellipsoid of plausible ??'s; in general, this leads to a more aggressive initial exploration phase (in both ?? and ?? 2 ).The posterior at time t for action i, after observing X, Y , is DISPLAYFORM0 , where we assume ?? 2 ??? IG(a t , b t ), and ?? | ?? 2 ??? N (?? t , ?? 2 ?? t ), an Inverse Gamma and Gaussian distribution, respectively.

Their parameters are given by DISPLAYFORM1 DISPLAYFORM2 We set the prior hyperparameters to ?? 0 = 0, and ?? 0 = ?? Id, while a 0 = b 0 = ?? > 1.

It follows that initially, for ?? We consider two approximations to (1) motivated by function approximators where d is large.

While posterior distributions or confidence ellipsoids should capture dependencies across parameters as shown above (say, a dense ?? t ), in practice, computing the correlations across all pairs of parameters is too expensive, and diagonal covariance approximations are common.

For linear models it may still be feasible to exactly compute (1), whereas in the case of Bayesian neural networks, unfortunately, this may no longer be possible.

Accordingly, we study two linear approximations where ?? t is diagonal.

Our goal is to understand the impact of such approximations in the simplest case, to properly set our expectations for the loss in performance of equivalent approximations in more complex approaches, like mean-field variational inference or Stochastic Gradient Langevin Dynamics.

Assume for simplicity the noise standard deviation is known.

In FIG1 , for d = 2, we see the posterior distribution ?? t ??? N (?? t , ?? t ) of a linear model based on (1), in green, together with two diagonal approximations.

Each approximation tries to minimize a different objective.

In blue, the PrecisionDiag posterior approximation finds the diagonal?? ??? R d??d minimizing KL(N (?? t ,??) || N (?? t , ?? t )), like in mean-field variational inference.

In particular,?? = Diag(?? ???1 t ) ???1 .

On the other hand, in orange, the Diag posterior approximation finds the diagonal matrix?? minimizing KL(N (?? t , ?? t ) || N (?? t ,??)) instead.

In this case, the solution is simply?? = Diag(?? t ).We add linear baselines that do not model the uncertainty in the action noise ?? 2 .

In addition, we also consider simple greedy and epsilon greedy linear baselines (i.e., not based on Thompson Sampling).

The main problem linear algorithms face is their lack of representational power, which they complement with accurate uncertainty estimates.

A natural attempt at getting the best of both worlds consists in performing a Bayesian linear regression on top of the representation of the last layer of a neural network, similarly to Snoek et al. (2015) .

The predicted value v i for each action a i is given by v i = ?? T i z x , where z x is the output of the last hidden layer of the network for context x. While linear methods directly try to regress values v on x, we can independently train a deep net to learn a representation z, and then use a Bayesian linear regression to regress v on z, obtain uncertainty estimates on the ??'s, and make decisions accordingly via Thompson Sampling.

Note that we do not explicitly consider the weights of the linear output layer of the network to make decisions; further, the network is only used to find good representations z. In addition, we can update the network and the linear regression at different time-scales.

It makes sense to keep an exact linear regression (as in FORMULA1 and FORMULA2 ) at all times, adding each new data point as soon as it arrives.

However, we only update the network after a number of points have been collected.

In our experiments, after updating the network, we perform a forward pass on all the training data to obtain z x , which is then fed to the Bayesian regression.

In practice this may be too expensive, and z could be updated periodically with online updates on the regression.

We call this algorithm Neural Linear.

Neural Greedy We refer to the algorithm that simply trains a neural network and acts greedily (i.e., takes the action whose predicted score for the current context is highest) as RMS, as we train it using the RMSProp optimizer.

This is our non-linear baseline, and we tested several versions of it (based on whether the training step was decayed, reset to its initial value for each re-training or not, and how long the network was trained for).

We also tried the -greedy version of the algorithm, where a random action was selected with probability for some decaying schedule of .Variational Inference Variational approaches approximate the posterior by finding a distribution within a tractable family that minimizes the KL divergence to the posterior BID20 .

These approaches formulate and solve an optimization problem, as opposed, for example, to sampling methods like MCMC BID22 Wainwright et al., 2008) .

Typically (and in our experiments), the posterior is approximated by a mean-field or factorized distribution where strong independence assumptions are made.

For instance, each neural network weight can be modeled via a -conditionally independent-Gaussian distribution whose mean and variance are estimated from data.

Recent advances have scaled these approaches to estimate the posterior of neural networks with millions of parameters BID6 .

A common criticism of variational inference is that it underestimates uncertainty (e.g., BID5 ), which could lead to under-exploration.

The family of expectation-propagation algorithms (Opper & Winther, 2000; BID30 a) is based on the message passing framework (Pearl, 1986) .

They iteratively approximate the posterior by updating a single approximation factor (or site) at a time, which usually corresponds to the likelihood of one data point.

The algorithm sequentially minimizes a set of local KL divergences, one for each site.

Most often, and for computational reasons, likelihoods are chosen to lie in the exponential family.

In this case, the minimization corresponds to moment matching.

See BID14 for further details.

We focus on methods that directly optimize the global EP objective via stochastic gradient descent, as, for instance, Power EP BID28 .

In particular, in this work, we implement the black-box ??-divergence minimization algorithm BID19 , where local parameter sharing is applied to the Power EP energy function.

Note that different values of ?? ??? R\{0} correspond to common algorithms: ?? = 1 to EP, and ?? ??? 0 to Variational Bayes.

The optimal ?? value is problem-dependent BID19 .Dropout Dropout is a training technique where the output of each neuron is independently zeroed out with probability p at each forward pass (Srivastava et al., 2014) .

Once the network has been trained, dropout can still be used to obtain a distribution of predictions for a specific input.

Following the best action with respect to the random dropout prediction can be interpreted as an implicit form of Thompson sampling.

Dropout can be seen as optimizing a variational objective BID23 BID13 BID21 .Monte Carlo Monte Carlo sampling remains one of the simplest and reliable tools in the Bayesian toolbox.

Rather than parameterizing the full posterior, Monte Carlo methods estimate the posterior through drawing samples.

This is naturally appealing for highly parameterized deep neural networks for which the posterior is intractable in general and even simple approximations such as multivariate Gaussian are too expensive (i.e. require computing and inverting a covariance matrix over all parameters).

Among Monte Carlo methods, Hamiltonian Monte Carlo (Neal, 1994) (HMC) is often regarded as a gold standard algorithm for neural networks as it takes advantage of gradient information and momentum to more effectively draw samples.

However, it remains unfeasible for larger datasets as it involves a Metropolis accept-reject step that requires computing the log likelihood over the whole data set.

A variety of methods have been developed to approximate HMC using mini-batch stochastic gradients.

These Stochastic Gradient Langevin Dynamics (SGLD) methods (Neal, 1994; Welling & Teh, 2011) add Gaussian noise to the model gradients during stochastic gradient updates in such a manner that each update results in an approximate sample from the posterior.

Different strategies have been developed for augmenting the gradients and noise according to a preconditioning matrix.

BID25 show that a preconditioner based on the RMSprop algorithm performs well on deep neural networks.

Patterson & Teh (2013) suggested using the Fisher information matrix as a preconditioner in SGLD.

Unfortunately the approximations of SGLD hold only if the learning rate is asymptotically annealed to zero.

BID1 introduced Stochastic Gradient Fisher Scoring to elegantly remove this requirement by preconditioning according to the Fisher information (or a diagonal approximation thereof).

BID26 develop methods for approximately sampling from the posterior using a constant learning rate in stochastic gradient descent and develop a prescription for a stable version of SGFS.

We evaluate the diagonal-SGFS and constant-SGD algorithms from BID26 in this work.

Specifically for constant-SGD we use a constant learning rate for stochastic gradient descent, where the learning rate is given by = 2 S N BB T where S is the batch size, N the number of data points and BB T is an online average of the diagonal empirical Fisher information matrix.

For Stochastic Gradient Fisher Scoring we use the following stochastic gradient update for the model parameters ?? at step t: DISPLAYFORM0 where we take the noise covariance EE T to also be BB DISPLAYFORM1 Bootstrap A simple empirical approach to approximate the sampling distribution of any estimator is the Bootstrap BID10 .

The main idea is to simultaneously train q models, where each model i is based on a different dataset D i .

When all the data D is available in advance, D i is typically created by sampling |D| elements from D at random with replacement.

In our case, however, the data grows one example at a time.

Accordingly, we set a parameter p ??? (0, 1], and append the new datapoint to each D i independently at random with probability p.

In order to emulate Thompson Sampling, we sample a model uniformly at random (i.e., with probability p i = 1/q.) and take the action predicted to be best by the sampled model.

We mainly tested cases q = 5, 10 and p = 0.8, 1.0, with neural network models.

Note that even when p = 1 and the datasets are identical, the random initialization of each network, together with the randomness from SGD, lead to different predictions.

Direct Noise Injection Parameter-Noise (Plappert et al., 2017 ) is a recently proposed approach for exploration in deep RL that has shown promising results.

The training updates for the network are unchanged, but when selecting actions, the network weights ?? are perturbed with isotropic Gaussian noise.

Crucially, the network uses layer normalization BID3 , which ensures that all weights are on the same scale.

The magnitude of the Gaussian noise is adjusted so that the overall effect of the perturbations is similar in scale to -greedy with a linearly decaying schedule (see (Plappert et al., 2017) for details).

Because the perturbations are done on the model parameters, we might hope that the actions produced by the perturbations are more sensible than -greedy.

Bayesian Non-parametric Gaussian processes (Rasmussen & Williams, 2005) are a gold-standard method for modeling distributions over non-linear continuous functions.

It can be shown that, in the limit of infinite hidden units and under a Gaussian prior, a Bayesian neural network converges to a Gaussian process (Neal, 1994) .

As such, GPs would appear to be a natural baseline.

Unfortunately, standard GPs computationally scale cubically in the number of observations, limiting their applicability to relatively small datasets.

There are a wide variety of methods to approximate Gaussian processes using, for example, pseudo-observations (Snelson & Ghahramani, 2006) or variational inference (Titsias, 2009).

We implemented both standard and sparse GPs but only report the former due to similar performance.

For the standard GP, due to the scaling issue, we stop adding inputs to the GP after 1000 observations.

This performed significantly better than randomly sampling inputs.

Our implementation is a multi-task Gaussian process BID7 with a linear and Matern 3/2 product kernel over the inputs and an exponentiated quadratic kernel over latent vectors for the different tasks.

The hyperparameters of this model and the latent task vectors are optimized over the GP marginal likelihood.

This allows the model to learn correlations between the outputs of the model.

Specifically, the covariance function K(??) of the GP is given by: DISPLAYFORM2 and the task kernel between tasks t and l are DISPLAYFORM3 2 ) where v l indexes the latent vector for task l and r ?? (x,x) = |(x ??) ??? (x ??)|.

The length-scales, ?? m and ?? l , and amplitude parameters ??, ?? are optimized via the log marginal likelihood.

For the sparse version we used a Sparse Variational GP BID18 with the same kernel and with 300 inducing points, trained via minibatch stochastic gradient descent .

In this section, we illustrate some of the subtleties that arise when uncertainty estimates drive sequential decision-making using simple linear examples.

There is a fundamental difference between static and dynamic scenarios.

In a static scenario, e.g. supervised learning, we are given a model family ?? (like the set of linear models, trees, or neural networks with specific dimensions), a prior distribution ?? 0 over ??, and some observed data D that -importantly-is assumed i.i.d.

Our goal is to return an approximate posterior distribution: DISPLAYFORM0 We define the quality of our approximation by means of some distance d(??, ??).On the other hand, in dynamic settings, our estimate at time t, say?? t , will be used via some mechanism M, in this case Thompson sampling, to collect the next data-point, which is then appended to D t .

In this case, the data-points in D t are no longer independent.

D t will now determine two distributions: the posterior given the data that was actually observed, ?? t+1 = P(?? | D t ), and our new estimate?? t+1 .

When the goal is to make good sequential decisions in terms of cumulative regret, the distance d(?? t , ?? t ) is in general no longer a definitive proxy for performance.

For instance, a poorly-approximated decision boundary could lead an algorithm, based on??, to get stuck repeatedly selecting a single sub-optimal action a. After collecting lots of data for that action,?? t and ?? t could start to agree (to their capacity) on the models that explain what was observed for a, while both would stick to something close to the prior regarding the other actions.

At that point, d(?? t , ?? t ) may show relatively little disagreement, but the regret would already be terrible.

Let ?? * t be the posterior distribution P(?? | D t ) under Thompson Sampling's assumption, that is, data was always collected according to ?? * j for j < t. We follow the idea that?? t being close to ?? * t for all t leads to strong performance.

However, this concept is difficult to formalize: once different decisions are made, data for different actions is collected and it is hard to compare posterior distributions.

We illustrate the previous points with a simple example, see FIG2 .

Data is generated according to a bandit with k = 6 arms.

For a given context X ??? N (??, ??), the reward obtained by pulling arm i follows a linear model r i,X = X T ?? i + with ??? N (0, ?? DISPLAYFORM1 can be exactly computed using the standard Bayesian linear regression formulas presented in Section 3.

We set the contextual dimension d = 20, and the prior to be ?? ??? N (0, ?? I d ), for ?? > 0.In FIG2 , we show the posterior distribution for two dimensions of ?? i for each arm i after n = 500 pulls.

In particular, in FIG2 , two independent runs of Thompson Sampling with their posterior distribution are displayed in red and green.

While strongly aligned, the estimates for some arms disagree (especially for arms that are best only for a small fraction of the contexts, like Arm 2 and 3, where fewer data-points are available).

In FIG2 , we also consider Thompson Sampling with an approximate posterior with diagonal covariance matrix, Diag in red, as defined in Section 3.

Each algorithm collects its own data based on its current posterior (or approximation).

In this case, the posterior disagreement after n = 500 decisions is certainly stronger.

However, as shown in FIG2 , if we computed the approximate posterior with a diagonal covariance matrix based on the data collected by the actual posterior, the disagreement would be reduced as much as possible within the approximation capacity (i.e., it still cannot capture correlations in this case).

FIG2 shows then the effect of the feedback loop.

We look next at the impact that this mismatch has on regret.

We illustrate with a similar example how inaccurate posteriors sometimes lead to quite different behaviors in terms of regret.

In FIG1 , we see the posterior distribution ?? ??? N (??, ??) of a linear model in green, together with the two diagonal linear approximations introduced in Section 3: the Diag (in orange) and the PrecisionDiag (in blue) approximations, respectively.

We now assume there are k linear arms, ?? i ??? R d for i = 1, . . .

, k, and decisions are made according to the posteriors in FIG1 .

In FIG1 we plot the regret of Thompson Sampling when there are k = 20 arms, for both d = 15 and d = 30.

We see that, while the PrecisionDiag approximation does even outperform the actual posterior, the diagonal covariance approximation truly suffers poor regret when we increase the dimension d, as it is heavily penalized by simultaneously over-exploring in a large number of dimensions and repeateadly acting according to implausible models.

In this section, we present the simulations and outcomes of several synthetic and real-world data bandit problems with each of the algorithms introduced in Section 3.

In particular, we first explain how the simulations were set up and run, and the metrics we report.

We then split the experiments according to how data was generated, and the underlying models fit by the algorithms from Section 3.

We run the contextual bandit experiments as described at the beginning of Section 2, and discuss below some implementation details of both experiments and algorithms.

A detailed summary of the key parameters used for each algorithm can be found in Table 2 in the appendix.

Neural Network Architectures All algorithms based on neural networks as function approximators share the same architecture.

In particular, we fit a simple fully-connected feedforward network with two hidden layers with 100 units each and ReLu activations.

The input of the network has dimension d (same as the contexts), and there are k outputs, one per action.

Note that for each training point (X t , a t , r t ) only one action was observed (and algorithms usually only take into account the loss corresponding to the prediction for the observed action).Updating Models A key question is how often and for how long models are updated.

Ideally, we would like to train after each new observation and for as long as possible.

However, this may limit the applicability of our algorithms in online scenarios where decisions must be made immediately.

We update linear algorithms after each time-step by means of (1) and (2).

For neural networks, the default behavior was to train for t s = 20 or 100 mini-batches every t f = 20 timesteps.

2 The size of each mini-batch was 512.

We experimented with increasing values of t s , and it proved essential for some algorithms like variational inference approaches.

See the details in Table 2 .Metrics We report two metrics: cumulative regret and simple regret.

We approximate the latter as the mean cumulative regret in the last 500 time-steps, a proxy for the quality of the final policy (see further discussion on pure exploration settings, BID8 ).

Cumulative regret is computed based on the best expected reward, as is standard.

For most real datasets (Statlog, Covertype, Jester, Adult, Census, and Song), the rewards were deterministic, in which case, the definition of regret also corresponds to the highest realized reward (i.e., possibly leading to a hard task, which helps to understand why in some cases all regrets look linear).

We reshuffle the order of the contexts, and rerun the experiment 50 times to obtain the cumulative regret distribution and report its statistics.

Hyper-Parameter Tuning Deep learning methods are known to be very sensitive to the selection of a wide variety of hyperparameters, and many of the algorithms presented are no exception.

Moreover, that choice is known to be highly dataset dependent.

Unfortunately, in the bandits scenario, we commonly do not have access to each problem a-priori to perform tuning.

For the vast majority of algorithms, we report the outcome for three versions of the algorithm defined as follows.

First, we use one version where hyper-parameters take values we guessed to be reasonable a-priori.

Then, we add two additional instances whose hyper-parameters were optimized on two different datasets via Bayesian Optimization.

For example, in the case of Dropout, the former version is named Dropout, while the optimized versions are named Dropout-MR (using the Mushroom dataset) and Dropout-SL (using the Statlog dataset) respectively.

Some algorithms truly benefit from hyper-parameter DISPLAYFORM0 Figure 3: Wheel bandits for increasing values of ?? ??? (0, 1).

Optimal action for blue, red, green, black, and yellow regions, are actions 1, 2, 3, 4, and 5, respectively.optimization, while others do not show remarkable differences in performance; the latter are more appropriate in settings where access to the real environment for tuning is not possible in advance.

Buffer After some experimentation, we decided not to use a data buffer as evidence of catastrophic forgetting was observed, and datasets are relatively small.

Accordingly, all observations are sampled with equal probability to be part of a mini-batch.

In addition, as is standard in bandit algorithms, each action was initially selected s = 3 times using round-robin independently of the context.

We evaluated the algorithms on a range of bandit problems created from real-world data.

In particular, we test on the Mushroom, Statlog, Covertype, Financial, Jester, Adult, Census, and Song datasets (see Appendix Section A for details on each dataset and bandit problem).

They exhibit a broad range of properties: small and large sizes, one dominating action versus more homogeneous optimality, learnable or little signal, stochastic or deterministic rewards, etc.

For space reasons, the outcome of some simulations are presented in the Appendix.

The Statlog, Covertype, Adult, and Census datasets were originally tested in BID11 .

We summarize the final cumulative regret for Mushroom, Statlog, Covertype, Financial, and Jester datasets in TAB0 .

In Figure 5 at the appendix, we show a box plot of the ranks achieved by each algorithm across the suite of bandit problems (see Appendix Table 6 and 7 for the full results).

As most of the algorithms from Section 3 can be implemented for any model architecture, in this subsection we use linear models as a baseline comparison across algorithms (i.e., neural networks that contain a single linear layer).

This allows us to directly compare the approximate methods against methods that can compute the exact posterior.

The specific hyper-parameter configurations used in the experiments are described in TAB2 in the appendix.

Datasets are the same as in the previous subsection.

The cumulative and simple regret results are provided in appendix Tables 4 and 5.

Some of the real-data problems presented above do not require significant exploration.

We design an artificial problem where the need for exploration is smoothly parameterized.

The wheel bandit is defined as follows (see Figure 3 ).

Set d = 2, and ?? ??? (0, 1), the exploration parameter.

Contexts are sampled uniformly at random in the unit circle in R 2 , X ??? U (D).

There are k = 5 possible actions.

The first action a 1 always offers reward r ??? N (?? 1 , ?? 2 ), independently of the context.

On the other hand, for contexts such that X ??? ??, i.e. inside the blue circle in Figure 3 , the other four actions are equally distributed and sub-optimal, with r ??? N (?? 2 , ?? 2 ) for ?? 2 < ?? 1 .

When X > ??, we are outside the blue circle, and only one of the actions a 2 , . . .

, a 5 is optimal depending on the sign of context components X = (X 1 , X 2 ).

If X 1 , X 2 > 0, action 2 is optimal.

If X 1 > 0, X 2 < 0, action 3 is optimal, and so on.

Non-optimal actions still deliver r ??? N (?? 2 , ?? 2 ) in this region, except a 1 whose mean reward is always ?? 1 , while the optimal action provides r ??? N (?? 3 , ?? 2 ), with ?? 3 ?? 1 .

We set ?? 1 = 1.2, ?? 2 = 1.0, ?? 3 = 50.0, and ?? = 0.01.

Note that the probability of a context randomly falling in the high-reward region is 1 ??? ?? 2 (not blue).

The difficulty of the problem increases with ??, and we expect algorithms to get stuck repeatedly selecting action a 1 for large ??.

The problem can be easily generalized for d > 2.

Results are shown in Table 9 .

100.00 ?? 0.15 100.00 ?? 0.03 100.00 ?? 0.01 100.00 ?? 1.48 100.00 ?? 1.01

Overall, we found that there is significant room for improvement in uncertainty estimation for neural networks in sequential decision-making problems.

First, unlike in supervised learning, sequential decision-making requires the model to be frequently updated as data is accumulated.

As a result, methods that converge slowly are at a disadvantage because we must truncate optimization to make the method practical for the online setting.

In these cases, we found that partially optimized uncertainty estimates can lead to catastrophic decisions and poor performance.

Second, and while it deserves further investigation, it seems that decoupling representation learning and uncertainty estimation improves performance.

The NeuralLinear algorithm is an example of this decoupling.

With such a model, the uncertainty estimates can be solved for in closed form (but may be erroneous due to the simplistic model), so there is no issue with partial optimization.

We suspect that this may be the reason for the improved performance.

In addition, we observed that many algorithms are sensitive to their hyperparameters, so that best configurations are problem-dependent.

Finally, we found that in many cases, the inherit randomness in Stochastic Gradient Descent provided sufficient exploration.

Accordingly, in some scenarios it may be hard to justify the use of complicated (and less transparent) variations of simple methods.

However, Stochastic Gradient Descent is by no The suffix of the BBB legend label indicates the number of training epochs in each training step.

We emphasize that in this evaluation, all algorithms use the same family of models (i.e., linear).

While PrecisionDiag exactly solves the mean field problem, BBB relies on partial optimization via SGD.

As the number of training epochs increases, BBB improves performance, but is always outperformed by PrecisionDiag.means always enough: in our synthetic exploration-oriented problem (the Wheel bandit) additional exploration was necessary.

Next, we discuss our main findings for each class of algorithms.

Linear Methods.

Linear methods offer a reasonable baseline, surprisingly strong in many cases.

While their representation power is certainly a limiting factor, their ability to compute informative uncertainty measures seems to payoff and balance their initial disadvantage.

They do well in several datasets, and are able to react fast to unexpected or extreme rewards (maybe as single points can have a heavy impact in fitted models, and their updates are immediate, deterministic, and exact).

Some datasets clearly need more complex non-linear representations, and linear methods are unable to efficiently solve those.

In addition, linear methods obviously offer computational advantages, and it would be interesting to investigate how their performance degrades when a finite data buffer feeds the estimates as various real-world online applications may require (instead of all collected data).In terms of the diagonal linear approximations described in Section 3, we found that diagonalizing the precision matrix (as in mean-field Variational Inference) performs dramatically better than diagonalizing the covariance matrix.

NeuralLinear.

The NeuralLinear algorithm sits near a sweet spot that is worth further studying.

In general it seems to improve the RMS neural network it is based on, suggesting its exploration mechanisms add concrete value.

We believe its main strength is that it is able to simultaneously learn a data representation that greatly simplifies the task at hand, and to accurately quantify the uncertainty over linear models that explain the observed rewards in terms of the proposed representation.

While the former process may be noisier and heavily dependent on the amount of training steps that were taken and available data, the latter always offers the exact solution to its approximate parent problem.

This, together with the partial success of linear methods with poor representations, may explain its promising results.

In some sense, it knows what it knows.

In the Wheel problem, which requires increasingly good exploration mechanisms, NeuralLinear is probably the best algorithm.

Its performance is almost an order of magnitude better than any RMS algorithm (and its spinoffs, like Bootstrapped NN, Dropout, or Parameter Noise), and all greedy linear approaches.

On the other hand, it is able to successfully solve problems that require non-linear representations (as Statlog or Covertype) where linear approaches fail.

In addition, the algorithm is remarkably easy to tune, and robust in terms of hyper-parameter configurations.

While conceptually simple, its deployment to large scale systems may involve some technical difficulties; mainly, to update the Bayesian estimates when the network is re-trained.

We believe, however, standard solutions to similar problems (like running averages) could greatly mitigate these issues.

In our experiments and compared to other algorithms, as shown in Table 8 , NeuralLinear is fast from a computational standpoint.

Variational Inference.

Overall, Bayes By Backprop performed poorly, ranking in the bottom half of algorithms across datasets TAB0 .

To investigate if this was due to underestimating uncertainty (as variational methods are known to BID5 ), to the mean field approximation, or to stochastic optimization, we applied BBB to a linear model, where the mean field optimization problem can be solved in closed form FIG5 .

We found that the performance of BBB slowly improved as the number of training epochs increased, but underperformed compared to the exact mean field solution.

Moreover, the difference in performance due to the number of training steps dwarfed the difference between the mean field solution and the exact posterior.

This suggests that it is not sufficient to partially optimize the variational parameters when the uncertainty estimates directly affect the data being collected.

In supervised learning, optimizing to convergence is acceptable, however in the online setting, optimizing to convergence at every step incurs unreasonable computational cost.

Expectation-Propagation.

The performance of Black Box ??-divergence algorithms was poor.

Because this class of algorithms is similar to BBB (in fact, as ?? ??? 0, it converges to the BBB objective), we suspect that partial convergence was also the cause of their poor performance.

We found these algorithms to be sensitive to the number of training steps between actions, requiring a large number to achieve marginal performance.

Their terrible performance in the Mushroom bandit is remarkable, while in the other datasets they perform slightly worse than their variational inference counterpart.

Given the successes of Black Box ??-divergence in other domains BID19 , investigating approaches to sidestep the slow convergence of the uncertainty estimates is a promising direction for future work.

Monte Carlo.

Constant-SGD comes out as the winner on Covertype, which requires non-linearity and exploration as evidenced by performance of the linear baseline approaches TAB0 ).

The method is especially appealing as it does not require tuning learning rates or exploration parameters.

SGFS, however, performs better on average.

The additional injected noise in SGFS may cause the model to explore more and thus perform better, as shown in the Wheel Bandit problem where SGFS strongly outperforms Constant-SGD.Bootstrap.

The bootstrap offers significant gains with respect to its parent algorithm (RMS) in several datasets.

Note that in Statlog one of the actions is optimal around 80% of the time, and the bootstrapped predictions may help to avoid getting stuck, something from which RMS methods may suffer.

In other scenarios, the randomness from SGD may be enough for exploration, and the bootstrap may not offer important benefits.

In those cases, it might not justify the heavy computational overhead of the method.

We found it surprising that the optimized versions of BootstrappedNN decided to use only q = 2 and q = 3 networks respectively (while we set its value to q = 10 in the manually tuned version, and the extra networks did not improve performance significantly).

Unfortunately, Bootstrapped NNs were not able to solve the Wheel problem, and its performance was fairly similar to that of RMS.

One possible explanation is that -given the sparsity of the rewardall the bootstrapped networks agreed for the most part, and the algorithm simply got stuck selecting action a 1 .

As opposed to linear models, reacting to unusual rewards could take Bootstrapped NNs some time as good predictions could be randomly overlooked (and useful data discarded if p 1).Direct Noise Injection.

When properly tuned, Parameter-Noise provided an important boost in performance across datasets over the learner that it was based on (RMS), average rank of ParamNoise-SL is 20.9 compared to RMS at 28.7 TAB0 .

However, we found the algorithm hard to tune and sensitive to the heuristic controlling the injected noise-level.

On the synthetic Wheel problem -where exploration is necessary-both parameter-noise and RMS suffer from underexploration and perform similarly, except ParamNoise-MR which does a good job.

In addition, developing an intuition for the heuristic is not straightforward as it lacks transparency and a principled grounding, and thus may require repeated access to the decision-making process for tuning.

Dropout.

We initially experimented with two dropout versions: fixed p = 0.5, and p = 0.8.

The latter consistently delivered better results, and it is the one we manually picked.

The optimized versions of the algorithm provided decent improvements over its base RMS (specially Dropout-MR).In the Wheel problem, dropout performance is somewhat poor: Dropout is outperformed by RMS, while Dropout-MR offers gains with respect to all versions of RMS but it is not competitive with the best algorithms.

Overall, the algorithm seems to heavily depend on its hyper-parameters (see cum-regret performance of the raw Dropout, for example).

Dropout was used both for training and for decision-making; unfortunately, we did not add a baseline where dropout only applies during training.

Consequently, it is not obvious how to disentangle the contribution of better training from that of better exploration.

This remains as future work.

Bayesian Non-parametrics.

Perhaps unsurprisingly, Gaussian processes perform reasonably well on problems with little data but struggle on larger problems.

While this motivated the use of sparse GP, the latter was not able to perform similarly to stronger (and definitively simpler) methods.

In this work, we empirically studied the impact on performance of approximate model posteriors for decision making via Thompson Sampling in contextual bandits.

We found that the most robust methods exactly measured uncertainty (possibly under the wrong model assumptions) on top of complex representations learned in parallel.

More complicated approaches that learn the representation and its uncertainty together seemed to require heavier training, an important drawback in online scenarios, and exhibited stronger hyper-parameter dependence.

Further exploring and developing the promising approaches is an exciting avenue for future work.

Greedy NN approach, fixed learning rate (?? = 0.01).

Learning rate decays, and it is reset every training period.

Learning rate decays, and it is not reset at all.

It starts at ?? = 1.

RMS Based on RMS3 net.

Learning decay rate is 0.55, initial learning rate is 1.0.

Trained for ts = 100, t f = 20.

Based on RMS3 net.

Learning decay rate is 2.5, initial learning rate is 1.0.

Trained for ts = 50, t f = 20.

Based on RMS3 net.

Learning decay rate is 0.4, initial learning rate is 1.1.

Trained for ts = 100, t f = 20.

SGFS Burning = 500, learning rate ?? = 0.014, EMA decay = 0.9, noise ?? = 0.75.

Takes each action at random with equal probability.

BayesByBackprop with noise ?? = 0.5. (ts = 100, first 100 times linear decay from ts = 10000).

BayesByBackprop with noise ?? = 0.75. (ts = 100, first 100 times linear decay from ts = 10000).

BayesByBackprop with noise ?? = 1.0. (ts = 100, first 100 times linear decay from ts = 10000).

Bootstrapped NN Bootstrapped with q = 5 models, and p = 0.85.

Based on RMS3 net.

Bootstrapped NN2Bootstrapped with q = 5 models, and p = 1.0.

Based on RMS3 net.

Bootstrapped NN3Bootstrapped with q = 10 models, and p = 1.0.

Based on RMS3 net.

Dropout (RMS3) Dropout with probability p = 0.8.

Based on RMS3 net.

Dropout (RMS2) Dropout with probability p = 0.8.

Based on RMS2 net.

Greedy NN approach, fixed learning rate (?? = 0.01).

Learning rate decays, and it is reset every training period.

RMS2bSimilar to RMS2, but training for longer (ts = 800).

Learning rate decays, and it is not reset at all.

Starts at ?? = 1.

SGFS Burning = 500, learning rate ?? = 0.014, EMA decay = 0.9, noise ?? = 0.75.

ConstSGD Burning = 500, EMA decay = 0.9, noise ?? = 0.5.

Initial noise ?? = 0.01, and level = 0.01.

Based on RMS3 net.

Trained for longer: ts = 800.

Takes each action at random with equal probability.

Published as a conference paper at ICLR 2018 Table 4 : Cumulative regret incurred by linear models using algorithms in Section 3 on the bandits described in Section A. Values reported are the mean over 50 independent trials with standard error of the mean.

Published as a conference paper at ICLR 2018 Table 5 : Simple regret incurred by linear models using algorithms in Section 3 on the bandits described in Section A. Simple regret was approximated by averaging the regret over the final 500 steps.

Values reported are the mean over 50 independent trials with standard error of the mean.

Statlog Covertype Financial Jester Adult

Alpha Divergences 0.68 ?? 0.04 0.07 ?? 0.00 0.31 ?? 0.00 0.00 ?? 0.00 2.91 ?? 0.04 0.75 ?? 0.00 Alpha Divergences FORMULA1 1.50 ?? 0.05 0.08 ?? 0.00 0.31 ?? 0.00 0.00 ?? 0.00 2.98 ?? 0.03 0.75 ?? 0.00 Alpha Divergences FORMULA2 1.51 ?? 0.05 0.13 ?? 0.00 0.32 ?? 0.00 0.00 ?? 0.00 3.42 ?? 0.05 0.77 ?? 0.00 Alpha Divergences (3)1.50 ?? 0.05 0.08 ?? 0.00 0.31 ?? 0.00 0.00 ?? 0.00 2.

0.07 ?? 0.01 0.07 ?? 0.00 0.29 ?? 0.00 0.01 ?? 0.00 2.80 ?? 0.03 0.67 ?? 0.00 LinGreedy (eps=0.05) 0.24 ?? 0.02 0.10 ?? 0.00 0.31 ?? 0.00 0.06 ?? 0.00 2.86 ?? 0.03 0.68 ?? 0.00 LinPost 0.29 ?? 0.03 0.06 ?? 0.00 0.28 ?? 0.00 0.01 ?? 0.00 2.74 ?? 0.04 0.69 ?? 0.00 LinfullDiagPost 4.10 ?? 0.07 0.18 ?? 0.00 0.63 ?? 0.00 0.00 ?? 0.00 2.86 ?? 0.03 0.89 ?? 0.00 LinfullDiagPrecPost 0.19 ?? 0.02 0.05 ?? 0.00 0.28 ?? 0.00 0.00 ?? 0.00 2.82 ?? 0.03 0.67 ?? 0.00 LinfullPost 0.08 ?? 0.01 0.05 ?? 0.00 0.28 ?? 0.00 0.00 ?? 0.00 2.86 ?? 0.03 0.67 ?? 0.00 Param-Noise 0.49 ?? 0.07 0.05 ?? 0.00 0.32 ?? 0.00 0.01 ?? 0.00 2.87 ?? 0.04 0.69 ?? 0.00 Param-Noise2 0.36 ?? 0.05 0.05 ?? 0.00 0.33 ?? 0.00 0.01 ?? 0.00 2.83 ?? 0.04 0.69 ?? 0.00 Uniform 4.88 ?? 0.07 0.86 ?? 0.00 0.86 ?? 0.00 1.25 ?? 0.02 5.03 ?? 0.07 0.93 ?? 0.00Published as a conference paper at ICLR 2018 Table 6 : Cumulative regret incurred by models using algorithms in Section 3 on the bandits described in Section A. Values reported are the mean over 50 independent trials with standard error of the mean.

Normalized with respect to the performance of Uniform.

Published as a conference paper at ICLR 2018 Table 7 : Simple regret incurred by models using algorithms in Section 3 on the bandits described in Section A. Simple regret was approximated by averaging the regret over the final 500 steps.

Values reported are the mean over 50 independent trials with standard error of the mean.

Normalized with respect to the performance of Uniform.

Published as a conference paper at ICLR 2018 Table 8 : Elapsed time for algorithms in Section 3 on the bandits described in Section A. Values reported are the mean over 50 independent trials with standard error of the mean.

Normalized with respect to the elapsed time required by RMS (which uses t s = 100 and t f = 20).Published as a conference paper at ICLR 2018 Table 9 : Cumulative regret incurred on the Wheel Bandit problem with increasing values of ??.

Values reported are the mean over 50 independent trials with standard error of the mean.

Normalized with respect to the performance of Uniform.

We qualitatively compare plots of the sample distribution from various methods, similarly to BID19 .

We plot the mean and standard deviation of 100 samples drawn from each method conditioned on a small set of observations with three outputs (two are from the same underlying function and thus strongly correlated while the third (bottom) is independent).

The true underlying functions are plotted in red.

Mushroom.

The Mushroom Dataset (Schlimmer, 1981) contains 22 attributes per mushroom, and two classes: poisonous and safe.

As in BID6 , we create a bandit problem where the agent must decide whether to eat or not a given mushroom.

Eating a safe mushroom provides reward +5.

Eating a poisonous mushroom delivers reward +5 with probability 1/2 and reward -35 otherwise.

If the agent does not eat a mushroom, then the reward is 0.

We set n = 50000.Statlog.

The Shuttle Statlog Dataset BID2 provides the value of d = 9 indicators during a space shuttle flight, and the goal is to predict the state of the radiator subsystem of the shuttle.

There are k = 7 possible states, and if the agent selects the right state, then reward 1 is generated.

Otherwise, the agent obtains no reward (r = 0).

The most interesting aspect of the dataset is that one action is the optimal one in 80% of the cases, and some algorithms may commit to this action instead of further exploring.

In this case, n = 43500.Covertype.

The Covertype Dataset BID2 classifies the cover type of northern Colorado forest areas in k = 7 classes, based on d = 54 features, including elevation, slope, aspect, and soil type.

Again, the agent obtains reward 1 if the correct class is selected, and 0 otherwise.

We run the bandit for n = 150000.Financial.

We created the Financial Dataset by pulling the stock prices of d = 21 publicly traded companies in NYSE and Nasdaq, for the last 14 years (n = 3713).

For each day, the context was the price difference between the beginning and end of the session for each stock.

We synthetically created the arms, to be a linear combination of the contexts, representing k = 8 different potential portfolios.

By far, this was the smallest dataset, and many algorithms over-explored at the beginning with no time to amortize their investment (Thompson Sampling does not account for the horizon).Jester.

We create a recommendation system bandit problem as follows.

The Jester Dataset BID15 ) provides continuous ratings in [???10, 10] for 100 jokes from 73421 users.

We find a complete subset of n = 19181 users rating all 40 jokes.

Following Riquelme et al. FORMULA1 , we take d = 32 of the ratings as the context of the user, and k = 8 as the arms.

The agent recommends one joke, and obtains the reward corresponding to the rating of the user for the selected joke.

Adult.

The Adult Dataset BID24 BID2 comprises personal information from the US Census Bureau database, and the standard prediction task is to determine if a person makes over $50K a year or not.

However, we consider the k = 14 different occupations as feasible actions, based on d = 94 covariates (many of them binarized).

As in previous datasets, the agent obtains reward 1 for making the right prediction, and 0 otherwise.

We set n = 45222.Census.

The US Census (1990) Dataset BID2 contains a number of personal features (age, native language, education...) which we summarize in d = 389 covariates, including binary dummy variables for categorical features.

Our goal again is to predict the occupation of the individual among k = 9 classes.

The agent obtains reward 1 for making the right prediction, and 0 otherwise, for each of the n = 250000 randomly selected data points.

Song.

The YearPredictionMSD Dataset is a subset of the Million Song Dataset BID4 .

The goal is to predict the year a given song was released (1922-2011) based on d = 90 technical audio features.

We divided the years in k = 10 contiguous year buckets containing the same number of songs, and provided decreasing Gaussian rewards as a function of the distance between the interval chosen by the agent and the one containing the year the song was actually released.

We initially selected n = 250000 songs at random from the training set.

The Statlog, Covertype, Adult, and Census datasets were tested in BID11 .

@highlight

An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling
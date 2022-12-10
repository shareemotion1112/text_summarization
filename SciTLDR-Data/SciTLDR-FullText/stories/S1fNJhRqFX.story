We establish the relation between Distributional RL and the Upper Confidence Bound (UCB) approach to exploration.

In this paper we show that the density of the Q function estimated by Distributional RL can be successfully used for the estimation of UCB.

This approach does not require counting and, therefore, generalizes well to the Deep RL.

We also point to the asymmetry of the empirical densities estimated by the Distributional RL algorithms like QR-DQN.

This observation leads to the reexamination of the variance's performance in the UCB type approach to exploration.

We introduce truncated variance as an alternative estimator of the UCB and a novel algorithm based on it.

We empirically show that newly introduced algorithm achieves better performance in multi-armed bandits setting.

Finally, we extend this approach to high-dimensional setting and test it on the Atari 2600 games.

New approach achieves better performance compared to QR-DQN in 26 of games, 13 ties out of 49 games.

Exploration is a long standing problem in Reinforcement Learning (RL).

It's been the main focus of the multi-armed bandits literature.

Here the algorithms are easier to design and analyze.

However, these solutions are quite unfeasible for high dimensional Deep RL setting, where the complication comes from the presence of the function approximator.

The multi-armed bandit can be represented by a slot machine with several arms.

Each arms' expected reward is unknown to the gambler.

Her/his goal is to maximize cumulative reward by pulling bandit's arms.

If the true expected rewards are known, then the best strategy is to pull the arm with the highest value.

However, gambler only observes stochastic reward after the arm is pulled.

One possible solution described by BID18 is to initialize values of arms' estimated means optimistically and then improve the estimates by pulling the same arm again.

Arm with a lower true mean will get its estimate decreased over time.

Eventually, the best arm will be discovered.

The drawback is that the set of the arms has to be enumerated and every arm has to be pulled infinitely many times.

In the RL setting an arm corresponds to a state-action pair, which implies that both assumptions are too strong for the Deep RL.Another line of reasoning is Upper Confidence Bound (UCB) type algorithms, e.g. UCB-1, introduce by BID10 .

The essence of the approach is nicely summarized by BID0 : 'optimism in the face of uncertainty principle'.

The idea is statistically intuitive: pull the arm which has the highest upper confidence bound, hoping for a better mean.

Estimation of the arm's UCB is performed via Hoeffdings Inequality 1 which is entirely based on counting the number of times the arm was pulled.

UCB extends to the tree search case in the form of UCT developed by BID10 .

Although this idea was successfully applied to the problem when perfect model is accessible, i.e. AlphaGo by BID17 , it does not generalize in a straightforward fashion to the general Deep RL setting without perfect model.

The main obstacle is the requirement of counting of the state-action pairs.

Another popular variation is UCB-V introduced by BID0 .

It estimates UCB via the empirical variance, which again involves counting.

Therefore, the requirement of counting prevents UCB ideas from successful generalization to the high dimensional setting of Deep RL.The generalization of exploration ideas from multi-armed bandits to Deep RL is challenging.

Therefore, one the most popular exploration approaches in Deep RL is the annealed epsilon greedy approach popularized by BID13 .

However, epsilon greedy approach is not very efficient, especially in Deep RL.

It does not take into account the underlying structure of the environment.

Therefore, researchers have been looking for other more efficient ways of exploration in Deep RL setting.

For example the idea of parametric noise was explored by BID4 .

Posterior sampling for reinforcement learning BID15 ) in Deep RL setting was developed by BID16 .

Uncertainty Bellman Equation proposed by BID14 , generalizes Bellman equation to the uncertainty measure.

The closest UCB type approach was developed by BID2 .

In order to avoid counting authors estimate UCB based on the empirical distribution of the Q function produced by Bootstrapped DQN BID16 ).

The approach reduces to estimating an ensemble of randomly initialized Q functions.

According to the averaged human normalized learning curve the performance improvement was insignificant.

Currently, there is a much better approach to estimating empirical distributions of Q function, i.e. distributional RL , ).

The results in the distributional RL are both theoretically sound and achieve state of the art performance in Deep RL environments, like Atari 2600.

However, we should note that Distributional RL does not use the whole distribution, but only the mean.

Another important characteristic of Distributional RL is that both C51 ) and Quantile Regression DQN (QR-DQN) ) are non parametric in the sense that the estimated distribution is not assumed to belong to any specific parametric family.

Hence, it is not assumed to be symmetric or even unimodal 2 .

We argue that in the case of asymmetric distributions, variance might become less sensitive in estimating UCB.

This problem seems to be overseen by the existing literature.

However, this issue might become more important in a more general setting, when symmetric assumption is not simply relaxed but is a very rare case.

We empirically show in the Section 4 that symmetry is in fact rare in Distributional RL.In this paper we build upon generic UCB idea.

We generalize it to the asymmetric distributions and high-dimensional setting.

In order to extend UCB approach to asymmetric distributions, we introduce truncated variability measure and show empirically that it achieves higher performance than variance in bandits setting.

Extension of this measure to rich visual environments provided by Atari 2600 platform is based on recent advances in Distributional RL.

As it was mentioned in the introduction UCB is based on the statistical relation between the number of observations of a random variable and the tightness of the mean estimates based on these observations.

More formally the connection is provided the inequality proved by BID7 :Theorem 1 (Hoeffdings Inequality) Let X 1 , . . .

X t be independent random variables bounded by DISPLAYFORM0 Therefore, Theorem 1 quantifies the relation between upper confidence bound forX and the number of realizations of X. On the other hand if the estimate of the probability density function (PDF) is available, then UCB can be estimated directly: DISPLAYFORM1 whereσ 2 is the variance computer fromP (X) and c is a constant reflecting the confidence level.

Now the question is how to estimate the empirical PDF.

Bayes-UCB introduced by BID9 uses restricted family of distributions which allows for the closed form Bayesian update.

On the other hand it is possible to model P [X] in a more expressive way using Neural Networks as it is done in the Distributional RL.

We lose closed form solutions, but gain more realistic estimates of the underlying distribution.

In this paper we explore Quantile Regression approach towards Distributional RL, which we introduce next.

The core idea behind QR-DQN is the Quantile Regression introduced by BID11 .

Let us first describe QR in the supervised machine learning setting.

Given data {(y i , x i )} i , τ -th linear quantile regression loss is defined as: DISPLAYFORM0 where DISPLAYFORM1 is the weighted sum of residuals.

Weights are proportional to the counts of the residual signs and order of the estimated quantile τ .

For higher quantiles positive residuals get higher weight and vice versa.

If τ = 0.5, then the estimate of the median for DISPLAYFORM2 Dabney et al. FORMULA0 introduced QR in a more general setting of RL with function approximator.

The design of the neural network is the same as in the original DQN introduced by BID13 except for the last linear layer, which outputs N quantiles {θ i } i instead of the a single estimate of Q. For a given transition (x, a, r, x ) and a discount factor γ the Bellman update is: DISPLAYFORM3 Note the similarity between loss in 3 and Algorithm 1.

The difference is in the type of the function approximator: linear in the former and neural network in the later.

This work is closely related to that of Bellemare et al. FORMULA0 with a major contribution in the way the empirical distribution is estimated.

use Boltzmann distribution in conjunction with a projection step.

QR-DQN seems to be a more elegant solution, since it does not involve the projection step and and there is no need for explicitly bounding the support.

It is worth emphasizing that function approximator in this case is crucial for generalization of UCB approach to DeepRL, since it eliminates the need for state-action pair counting.

Algorithm 1 Quantile Regression Q-learning DISPLAYFORM4 QR approach allows for a very elegant estimation of distributions in the RL setting.

We apply this idea to the multi-armed bandits.

This environment is more tractable and easier to explore.

Since the distributions are not assumed to have any regularities, like symmetry, we conjecture that variance might not the best UCB measure.

Therefore, we explore truncated variability measure.

We, then, generalize this idea to the Deep RL setting.

We propose to estimate empirical distribution of returns for each arm by the means of the QR.

The basic idea is to estimate mean and variance of the return for each arm based on the empirical distribution provided by QR and pick the arm according to the highest mean plus standard deviation.

We call this algorithm QUCB.In the setting of multi-armed bandits denote quantiles by {θ i } i and observed reward for by R. Then for a single arm the set of estimates of quantiles is the solution to: DISPLAYFORM0 As opposed to the supervised example there are no features x i .

Algorithm 2 outlines the QUCB.

Note the definitions ofθ t,k and V ar(θ t,k ): DISPLAYFORM1 Note the presence of the multiplier c t in the Algorithm 2.

To ensure the optimality in the limit t → ∞, {c t } t have to be chosen so that lim t→∞ c t = 0.

In case the number of quantiles is big compared to the sample size, it might help to warm-up the initial estimates by performing a few steps of pure exploration ( = 1).Hence, the algorithm estimates empirical quantiles.

QR approach allows to estimate any quantile.

In fact it allows to estimate multiple quantiles in one update step producing empirical density.

More importantly, having empirical distribution at hand opens up the way for new possible approaches to computing upper confidence bound, exploration bonuses or some other means of ordering empirical distributions when choosing action/arm.

One such approach is developed in the next section.

Input: if t ≤ number of burn-in steps then 3: DISPLAYFORM0 DISPLAYFORM1 Pick an arm randomly 4:else 5: DISPLAYFORM2 end if DISPLAYFORM3 draw reward R t 8: DISPLAYFORM4 Update quantiles of the corresponding arm 9: end for

In the case of non parametric approaches or parametric approaches that are not exclusively restricted to symmetric distributions the symmetry is not guaranteed.

In fact, it might be a very rare case as it can be seen from FIG1 .

Note that the game of Pong from Atari 2600 is the simplest one.

In the end of training presented in FIG1 , agent achieves almost the perfect score.

Hence the distributions in the end of training correspond to the near optimal policy and these distributions are not symmetric.

That is, the asymmetry of empirical distributions is the regular case, but not an exception.

Hence, the question: is the variance a 'good' measure of the upper confidence bound for asymmetric distributions in the setting of multi-armed bandits and RL?For the sake of the argument consider a simple decomposition of the variance into the two truncated variability measures: lower and upper variance 3 : DISPLAYFORM0 It is clear that in the case of a symmetric probability density function (PDF) the lower and upper variances are equivalent.

However, in the case of asymmetric distribution the equality σ In case of the UCB type approach to the exploration problem upper tail variability seem to be more relevant than lower tail one, especially if the estimated PDF is asymmetric.

Intuitively speaking, σ 2 u is an optimistic measure of variability.

σ 2 u is biased towards rare 'positive' rewards.

However, the availability of empirical PDF makes it possible to truncate the variance in many different ways.

σ 2 u might be a good candidate, although one potential draw back is the robustness of the estimator of the mean.

In order to mitigate that, we propose the following truncated measure of the variability based on the median rather than the mean 5 : DISPLAYFORM1 where θ i 's are i N -th quantiles.

As opposed to σ 2 u , σ 2 + captures some 'negative' variability but still being optimistic.

We propose QUCB+, the algorithm based on σ 2 + which is a slight modification of QUCB.

Instead of the V ar(θ t,k ) we propose to use σ 2 + .

We hypothesize that σ 2 + might be a more robust upper tail variability measure.

We support our hypothesis by empirical results in multi-armed bandits setting and Atari 2600, presented in Section 4.

The ideas presented in the previous section generalize in a straightforward fashion to the tabular RL and most importantly to the Deep RL setting.

As it was mentioned QR-DQN's output is the quantile distribution with equispaced quantiles: DISPLAYFORM0 .

The update step does not change.

Action selection step incorporates bias in the form of σ 2 + from Equation 10.

Algorithm 3 outlines DQN-QUCB+.

In the presence of the function approximator the variance encoded in the θ i is largely effected by the variation in the parameters.

Therefore, the variance produced by QR-DQN has at least two sources: intrinsic variation coming from the underlying MDP and parametric variation.

The important question is the dynamics of the parametric uncertainty during training.

As it can be seen from FIG1 the variance drops significantly meaning that the parametric uncertainty goes down as the network approaches optimal solution.

Hence, if the model tends to learn then the parametric component in the variance decreases.

4 Consider discrete empirical distribution with support {−1, 0, 2} and probability atoms { }.

5 For details see BID8 BID5 Algorithm 3 DQN-QUCB+ Input: w, w − , (x, a, r, x ), γ ∈ [0, 1) network weights, sampled transition, discount factor 1: Q(x , a ) = j q j θ j (x , a ; w − ) DISPLAYFORM1

Following BID18 we applied QUCB and QUCB+ to the multi-armed bandits test bed.

In order to study the effect of asymmetric distributions we set up two configurations of the test bed: with normally and asymmetrically distributed rewards.

Both configurations consist of 10 arms.

In both configurations true means of arms {µ i } K k=1 are drawn from the normal distribution with µ = 1, σ = 1.In the first configuration.

During each step the the reward for the k-th arm is drawn from N (µ k , 1).

As it can be seen from 2 there is no statistically significant difference between QUCB and QUCB+.

It is expected, since the the rewards are normally distributed, hence, symmetric around the mean.

In addition median and mean of the normal distribution coincide.

Therefore, estimate of σ 2 is close to that of σ 2 + .

In the second configuration the reward for the best arm is drawn from the lognormal distribution centered at µ k and variance 1.

And the rewards for other arms are drawn from the reflected about µ k lognormal distribution with variance one.

Hence true variances of all arms are the same, however in the presence of slight asymmetry QUCB+ performs better, see FIG3 .

As it was claimed earlier in the paper the QUCB generalizes to the Deep RL setting in the straightforward fashion.

The architecture of the network is that of QR-DQN.

experimented with two losses: the original QR loss and Huber loss with κ = 1.

Both architectures proved to be stable.

For our experiments we chose only one loss: the Huber loss with κ = 1 6 , due to high computational costs of experiments.

Another reason for picking the Huber loss is its smoothness compared to L1 loss of QR.

Smoothness is better suited for gradient descent methods.

Overall, we followed closely in setting the hyper parameters, except for the learning rate of the Adam optimizer which we set to α = 0.0001.The most significant distinction is the way the exploration is performed in DQN-QUCB.

As opposed to QR-DQN there is no epsilon greedy exploration schedule in DQN-QUCB.

The exploration is performed via the σ 2 + term only.

An important hyper parameter which is introduced by DQN-QUCB is the schedule, i.e. the sequence of multipliers for σ 2 + , {c t } t .

The choice depends on the specific problem.

In case of the stationary environment the UCB term involving σ 2 + should eventually vanish, i.e. c t → ∞. In the non stationary environment the agent might always need to explore, so that c t is eventually a constant 7 .In our experiments we used the following schedule: DISPLAYFORM0 The motivation behind the schedule is to gradually vanish the exploration term as it is done in QR-DQN.

This makes performance comparison more adequate.

During experiments we observed that DQN-QUCB is sensitive to the schedule to some extent, see for example FIG4 .

We conjecture that tuning the schedule might yield better performance across games.

We evaluated DQN-QUCB on the set of 49 Atari games initially proposed by BID13 .

Algorithms were evaluated on 40 million frames 8 3 runs per game.

presented in FIG5 .

DQN-QUCB achieves better performance (gain of over 3%) with respect to cumulative reward measure in 26 games.

We argue that cumulative reward is a suitable performance measure for our experiments, since none of the learning curves exhibit plummeting behaviour 9 .

A more detailed discussion of this point is presented in BID12 .

Recent advancements in RL, namely Distributional RL, not only established new theoretically sound principles but also achieved state-of-the-art performance in challenging high dimensional environments like Atari 2600.

The by-product of the Distributional RL is the empirical PDF for the Q function which is not directly used except for the mean computation.

UCB on the other hand is a very attractive exploration algorithm in the multi-armed bandits setting, which does not generalize in a straightforward fashion to Deep RL.In this paper we established the connection between the UCB idea and Distributional RL.

We also pointed to the asymmetry of the PDFs estimated by Distributional RL, which is not a rare exception but rather the only case.

We introduced truncated variability measure as an alternative to the variance and empirically showed that it can be successfully applied to multi-armed bandits and rich visual environments like Atari 2600.

It is highly likely that DQN-QUCB+ might be improved through schedule tuning.

DQN-QUCB+ might be combined with other advancements in Deep RL, e.g. Rainbow by BID6 , to yield better results.

@highlight

Exploration using Distributional RL and truncagted variance.

@highlight

Presents an RL method to manage exploration-explotation trade-offs via UCB techniques.

@highlight

A method to use the distribution learned by Quantile Regression DQN for exploration, in place of the usual epsilon-greedy strategy.

@highlight

Proposes new algorithsms (QUCB and QUCB+) to handle the exploration tradeoff in Multi-Armed Bendits and more generally in Reinforcement Learning
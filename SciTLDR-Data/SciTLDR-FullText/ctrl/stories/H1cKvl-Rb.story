We show how an ensemble of $Q^*$-functions can be leveraged for more effective exploration in deep reinforcement learning.

We build on well established algorithms from the bandit setting, and adapt them to the $Q$-learning setting.

We propose an exploration strategy based on upper-confidence bounds (UCB).

Our experiments show significant gains on the Atari benchmark.

Deep reinforcement learning seeks to learn mappings from high-dimensional observations to actions.

Deep Q-learning BID15 ) is a leading technique that has been used successfully, especially for video game benchmarks.

However, fundamental challenges remain, for example, improving sample efficiency and ensuring convergence to high quality solutions.

Provably optimal solutions exist in the bandit setting and for small MDPs, and at the core of these solutions are exploration schemes.

However these provably optimal exploration techniques do not extend to deep RL in a straightforward way.

Bootstrapped DQN BID16 ) is a previous attempt at adapting a theoretically verified approach to deep RL.

In particular, it draws inspiration from posterior sampling for reinforcement learning (PSRL, BID17 ; BID16 ), which has near-optimal regret bounds.

PSRL samples an MDP from its posterior each episode and exactly solves Q * , its optimal Q-function.

However, in high-dimensional settings, both approximating the posterior over MDPs and solving the sampled MDP are intractable.

Bootstrapped DQN avoids having to establish and sample from the posterior over MDPs by instead approximating the posterior over Q * .

In addition, bootstrapped DQN uses a multi-headed neural network to represent the Q-ensemble.

While the authors proposed bootstrapping to estimate the posterior distribution, their empirical findings show best performance is attained by simply relying on different initializations for the different heads, not requiring the sampling-with-replacement process that is prescribed by bootstrapping.

In this paper, we design new algorithms that build on the Q-ensemble approach from BID16 .

However, instead of using posterior sampling for exploration, we construct uncertainty estimates from the Q-ensemble.

Specifically, we first propose the Ensemble Voting algorithm where the agent takes action by a majority vote from the Q-ensemble.

Next, we propose the UCB exploration strategy.

This strategy is inspired by established UCB algorithms in the bandit setting and constructs uncertainty estimates of the Q-values.

In this strategy, agents are optimistic and take actions with the highest UCB.

We demonstrate that our algorithms significantly improve performance on the Atari benchmark.

We model reinforcement learning as a Markov decision process (MDP).

We define an MDP as (S, A, T, R, p 0 , γ), in which both the state space S and action space A are discrete, T : S × A × S → R + is the transition distribution, R : S × A → R is the reward function, assumed deterministic given the state and action, and γ ∈ (0, 1] is a discount factor, and p 0 is the initial state distribution.

We denote a transition experience as τ = (s, a, r, s ) where s ∼ T (s |s, a) and r = R(s, a).

A policy π : S → A specifies the action taken after observing a state.

We denote the Q-function for policy π as Q π (s, a) := E π ∞ t=0 γ t r t |s 0 = s, a 0 = a where r t = R(s t , a t ).

The optimal Q * -function corresponds to taking the optimal policy Q * (s, a) := sup π Q π (s, a)and satisfies the Bellman equation Q * (s, a) = E s ∼T (·|s,a) r + γ · max a Q * (s , a ) .

A notable early optimality result in reinforcement learning was the proof by Watkins and Dayan Watkins (1989); Watkins and Dayan (1992) that an online Q-learning algorithm is guaranteed to converge to the optimal policy, provided that every state is visited an infinite number of times.

However, the convergence of Watkins' Q-learning can be prohibitively slow in MDPs wheregreedy action selection explores state space randomly.

Later work developed reinforcement learning algorithms with provably fast (polynomial-time) convergence BID10 BID4 Strehl et al. (2006) ).

At the core of these provably-optimal learning methods is some exploration strategy, which actively encourages the agent to visit novel state-action pairs.

For example, R-MAX optimistically assumes that infrequently-visited states provide maximal reward, and delayed Q-learning initializes the Q-function with high values to ensure that each state-action is chosen enough times to drive the value down.

Since the theoretically sound RL algorithms are not computationally practical in the deep RL setting, deep RL implementations often use simple exploration methods such as -greedy and Boltzmann exploration, which are often sample-inefficient and fail to find good policies.

One common approach of exploration in deep RL is to construct an exploration bonus, which adds a reward for visiting state-action pairs that are deemed to be novel or informative.

In particular, several prior methods define an exploration bonus based on a density model or dynamics model.

Examples include VIME by BID9 , which uses variational inference on the forward-dynamics model, and Tang et al. FORMULA8 , BID2 , Ostrovski et al. (2017) , BID8 .

While these methods yield successful exploration in some problems, a major drawback is that this exploration bonus does not depend on the rewards, so the exploration may focus on irrelevant aspects of the environment, which are unrelated to reward.

Earlier works on Bayesian reinforcement learning include BID6 BID7 .

BID6 studied Bayesian Q-learning in the model-free setting and learned the distribution of Q * -values through Bayesian updates.

The prior and posterior specification relied on several simplifying assumptions, some of which are not compatible with the MDP setting.

BID7 took a model-based approach that updates the posterior distribution of the MDP.

The algorithm samples from the MDP posterior multiple times and solving the Q * values at every step.

This approach is only feasible for RL problems with very small state space and action space.

Strens (2000) proposed posterior sampling for reinforcement learning (PSRL).

PSRL instead takes a single sample of the MDP from the posterior in each episode and solves the Q * values.

Recent works including BID17 and BID16 established near-optimal Bayesian regret bounds for episodic RL.

Sorg et al. (2012) models the environment and constructs exploration bonus from variance of model parameters.

These methods are experimented on low dimensional problems only, because the computational cost of these methods is intractable for high dimensional RL.

Inspired by PSRL, but wanting to reduce computational cost, prior work developed approximate methods.

Osband et al. (2014) proposed randomized least-square value iteration for linearly-parameterized value functions.

Bootstrapped DQN Osband et al. (2016) applies to Q-functions parameterized by deep neural networks.

Bootstrapped DQN BID16 ) maintains a Q-ensemble, represented by a multi-head neural net structure to parameterize K ∈ N + Q-functions.

This multi-head structure shares the convolution layers but includes multiple "heads", each of which defines a Q-function Q k .Bootstrapped DQN diversifies the Q-ensemble through two mechanisms.

The first mechanism is independent initialization.

The second mechanism applies different samples to train each Q-function.

These Q-functions can be trained simultaneously by combining their loss functions with the help of a random mask DISPLAYFORM0 where y Q k τ is the target of the kth Q-function.

Thus, the transition τ updates Q k only if m k τ is nonzero.

To avoid the overestimation issue in DQN, bootstrapped DQN calculates the target value y Q k τ using the approach of Double DQN (Van Hasselt et al. (2016) ), such that the current Q k (·; θ t ) network determines the optimal action and the target network Q k (·; θ − ) estimates the value DISPLAYFORM1 In their experiments on Atari games, BID16 set the mask m τ = (1, . . . , 1) such that all {Q k } are trained with the same samples and their only difference is initialization.

Bootstrapped DQN picks one Q k uniformly at random at the start of an episode and follows the greedy action a t = argmax a Q k (s t , a) for the whole episode.

Ignoring computational costs, the ideal Bayesian approach to reinforcement learning is to maintain a posterior over the MDP.

However, with limited computation and model capacity, it is more tractable to maintain a posterior of the Q * -function.

This motivates using a Q-ensemble as a particle filter-based approach to approximate the posterior over Q * -function and we display our first proposed method, Ensemble Voting, in Algorithm 1.

DISPLAYFORM0 is parametrized with a deep neural network whose parameters are initialized independently at the start of training.

Each Q k proposes an action that maximizes the Q-value according to Q k at every time step and the agent chooses the action by a majority vote DISPLAYFORM1 At each learning interval, a minibatch of transitions is sampled from the replay buffer and each Q k takes a Bellman update based on this minibatch.

For stability, Algorithm 1 also uses a target network for each Q k as in Double DQN in the batched update.

We point out that the difference among the parameters of the Q-ensemble {Q k } comes only from the independent random initialization.

The deep neural network parametrization of the Q-ensemble introduces nonconvexity into the objective function of Bellman update, so the Q-ensemble {Q k } do not converge to the same Q-function during training even though they are trained with the same minibatches at every update.

We also experimented with bagging by updating each Q k using an independently drawn minibatch.

However, bagging led to inferior learning performance.

This phenomenon that that bagging deteriorates the performance of deep ensembles is also observed in supervised learning settings.

BID13 observed that supervised learning trained with deep ensembles with random initializations perform better than bagging for deep ensembles.

BID12 used deep ensembles for uncertainty estimates and also observed that bagging deteriorated performance in their experiments.

Lu and Van Roy (2017) develop ensemble sampling for bandit problems with deep neural network parametrized policies and the theoretical justification.

We derive a posterior update rule for the Q * function and approximations to the posterior update using ensembles in Appendix C. We note that in bootstrapped DQN, ensemble voting is applied for evaluation while Algorithm 1 uses ensemble voting during learning.

In the experiments (Sec. 5), we demonstrate that Algorithm 1 is superior to bootstrapped DQN.

The action choice of Algorithm 1 is exploitation only.

In the next section, we propose our UCB exploration strategy.

Pick an action according to DISPLAYFORM2 Execute a t .

Receive state s t+1 and reward r t from the environment 8:Add (s t , a t , r t , s t+1 ) to replay buffer B

At learning interval, sample random minibatch and update {Q k } 10:end for 11: end for

In this section, we propose optimism-based exploration by adapting the UCB algorithms BID1 ; BID0 ) from the bandit setting.

The UCB algorithms maintain an upper-confidence bound for each arm, such that the expected reward from pulling each arm is smaller than this bound with high probability.

At every time step, the agent optimistically chooses the arm with the highest UCB.

BID1 constructed the UCB based on empirical reward and the number of times each arm is chosen.

BID0 incorporated the empirical variance of each arm's reward into the UCB, such that at time step t, an arm A t is pulled according to DISPLAYFORM0 wherer i,t andV i,t are the empirical reward and variance of arm i at time t, n i,t is the number of times arm i has been pulled up to time t, and c 1 , c 2 are positive constants.

We extend the intuition of UCB algorithms to the RL setting.

Using the outputs of the {Q k } functions, we construct a UCB by adding the empirical standard deviationσ( DISPLAYFORM1 .

The agent chooses the action that maximizes this UCB a t ∈ argmax a μ(s t , a) + λ ·σ(s t , a) , where λ ∈ R + is a hyperparameter.

We present Algorithm 2, which incorporates the UCB exploration.

The hyperparemeter λ controls the degrees of exploration.

In Section 5, we compare the performance of our algorithms on Atari games using a consistent set of parameters.

Pick an action according to a t ∈ argmax a μ(s t , a) + λ ·σ(s t , a)

Receive state s t+1 and reward r t from environment, having taken action a t 8:Add (s t , a t , r t , s t+1 ) to replay buffer B 9:At learning interval, sample random minibatch and update {Q k } We evaluate the algorithms on each Atari game of the Arcade Learning Environment BID3 ).

We use the multi-head neural net architecture of BID16 .

We fix the common hyperparameters of all algorithms based on a well-tuned double DQN implementation, which uses the Adam optimizer BID11 ), different learning rate and exploration schedules compared to BID15 .

Appendix A tabulates the hyperparameters.

The number of {Q k } functions is K = 10.

Experiments are conducted on the OpenAI Gym platform BID5 ) and trained with 40 million frames and 2 trials on each game.

We take the following directions to evaluate the performance of our algorithms:1.

we compare Algorithm 1 against Double DQN and bootstrapped DQN, 2.

we isolate the impact of UCB exploration by comparing Algorithm 2 with λ = 0.1, denoted as ucb exploration, against Algorithm 1, Double DQN, and bootstrapped DQN.3.

we compare Algorithm 1 and Algorithm 2 with the count-based exploration method of BID2 .4.

we aggregate the comparison according to different categories of games, to understand when our methods are suprior.

In Appendix B, we tabulate detailed results that compare our algorithms, Ensemble Voting and ucb exploration, against prior methods.

In TAB5 , we tabulate the maximal mean reward in 100 consecutive episodes for Ensemble Voting, ucb exploration, bootstrapped DQN and Double DQN.

Without exploration, Ensemble Voting already achieves higher maximal mean reward than both Double DQN and bootstrapped DQN in a majority of Atari games.

Ensemble Voting performs better than Double DQN in 37 games out of the total 49 games evaluated, better than bootstrapped DQN in 41 games.

ucb exploration achieves the highest maximal mean reward among these four algorithms in 30 games out of the total 49 games evaluated.

Specifically, ucb exploration performs better than Double DQN in 38 out of 49 games evaluated, better than bootstrapped DQN in 45 games, and better than Ensemble Voting in 35 games.

Figure 2 displays the learning curves of these five algorithms on a set of six Atari games.

Ensemble Voting outperforms Double DQN and bootstrapped DQN.

ucb exploration outperforms Ensemble Voting.

In TAB6 , we compare our proposed methods with the count-based exploration method A3C+ of BID2 based on their published results of A3C+ trained with 200 million frames.

We point out that even though our methods were trained with only 40 million frames, much less than A3C+'s 200 million frames, UCB exploration achieves the highest average reward in 28 games, Ensemble Voting in 10 games, and A3C+ in 10 games.

Our approach outperforms A3C+.Finally to understand why and when the proposed methods are superior, we aggregate the comparison results according to four categories: Human Optimal, Score Explicit, Dense Reward, and Sparse Reward.

These categories follow the taxonomy in Probability of random action ingreedy exploration, as a function of the iteration t .replay start size 50000 Number of uniform random actions taken before learning starts.

Table 4 : Comparison of each method across different game categories.

The Atari games are separated into four categories: human optimal, score explicit, dense reward, and sparse reward.

In each row, we present the number of games in this category, the total number of games where each algorithm achieves the optimal performance according to TAB5 .

The game categories follow the taxonomy in TAB0 of Ostrovski et al. (2017) C APPROXIMATING BAYESIAN Q-LEARNING WITH Q-ENSEMBLESIn this section, we first derive a posterior update formula for the Q * -function under full exploration assumption and this formula turns out to depend on the transition Markov chain.

Next, we approximate the posterior update with Q-ensembles {Q k } and demonstrate that the Bellman equation emerges as the approximate update rule for each Q k .

*

-FUNCTION An MDP is specified by the transition probability T and the reward function R. Unlike prior works outlined in Section 2.3 which learned the posterior of the MDP, we will consider the joint distribution over (Q * , T ).

Note that R can be recovered from Q * given T .

So (Q * , T ) determines a unique MDP.

In this section, we assume that the agent samples (s, a) according to a fixed distribution.

The corresponding reward r and next state s given by the MDP append to (s, a) to form a transition τ = (s, a, r, s ), for updating the posterior of (Q * , T ).

Recall that the Q * -function satisfies the Bellman equation DISPLAYFORM0 Denote the joint prior distribution as p(Q * , T ) and the posterior asp.

We apply Bayes' formula to expand the posterior: DISPLAYFORM1 where Z(τ ) is a normalizing constant and the second equality is because s and a are sampled randomly from S and A. Next, we calculate the two conditional probabilities in (1).

First, DISPLAYFORM2 where the first equality is because given T , Q * does not influence the transition.

Second, DISPLAYFORM3 where 1 {·} is the indicator function and in the last equation we abbreviate it as 1(Q * , T ).

Substituting FORMULA9 and FORMULA10 into FORMULA8 , we obtain the joint posterior of Q * and T after observing an additional randomly sampled transition τ DISPLAYFORM4

The exact Q * -posterior update (4) is intractable in high-dimensional RL due to the large space of (Q * , T ).

Thus, we make several approximations to the Q * -posterior update.

First, we approximate the prior of Q * by sampling K ∈ N + independently initialized Q * -functions {Q k } K k=1 .

Next, we update them as more transitions are sampled.

The resulting {Q k } approximate samples drawn from the posterior.

The agent chooses the action by taking a majority vote from the actions determined by each Q k .We derive the update rule for {Q k } after observing a new transition τ = (s, a, r, s ).

At iteration i, given Q * = Q k,i (·; θ k ) parametrized by θ k the joint probability of (Q * , T ) factors into DISPLAYFORM0 Substitute FORMULA12 into FORMULA11 and we obtain the corresponding posterior for each Q k,i+1 at iteration i + 1 as DISPLAYFORM1 We update Q k,i to Q k,i+1 according to DISPLAYFORM2 We first derive a lower bound of the the posteriorp(Q k,i+1 |τ ): DISPLAYFORM3 where we apply a limit representation of the indicator function in the third equation.

The fourth equation is due to the bounded convergence theorem.

The inequality is Jensen's inequality.

The last equation (9) replaces the limit with an indicator function.

A sufficient condition for FORMULA14 is to maximize the lower-bound of the posterior distribution in (9) by ensuring the indicator function in (9) to hold.

We can replace (8) with the following update DISPLAYFORM4 (10) However, FORMULA8 is not tractable because the expectation in (10) is taken with respect to the posterior p(T |Q k,i , τ ) of the transition T .

To overcome this challenge, we approximate the posterior update by reusing the one-sample next state s from τ .

Solving the exact minimal for each Q k,i+1 is impractical, thus we take a gradient step on Q k,i+1 according to the following gradient DISPLAYFORM5 where η is the step size.

Instead of updating Q k after each transition, we use an experience replay buffer B to store observed transitions and sample a minibatch B mini of transitions (s, a, r, s ) for each update.

In this case, the batched update of each Q k,i to Q k,i+1 becomes a standard Bellman update DISPLAYFORM6

In this section, we also studied an "InfoGain" exploration bonus, which encourages agents to gain information about the Q * -function and examine its effectiveness.

We found it had some benefits on top of Ensemble Voting, but no uniform additional benefits once already using Q-ensembles on top of Double DQN.

We describe the approach and our experimental findings.

Similar to Sun et al. (2011) , we define the information gain from observing an additional transition τ n as DISPLAYFORM0 wherep(Q * |τ 1 , . . .

, τ n ) is the posterior distribution of Q * after observing a sequence of transitions (τ 1 , . . . , τ n ).

The total information gain is DISPLAYFORM1 Our Ensemble Voting, Algorithm 1, does not maintain the posteriorp, thus we cannot calculate (11) explicitly.

Instead, inspired by BID12 , we define an InfoGain exploration bonus that measures the disagreement among {Q k }.

Note that DISPLAYFORM2 where H(·) is the entropy.

If H τ1,...,τ N is small, then the posterior distribution has high entropy and high residual information.

Since {Q k } are approximate samples from the posterior, high entropy of the posterior leads to large discrepancy among {Q k }.

Thus, the exploration bonus is monotonous with respect to the residual information in the posterior H(p(Q * |τ 1 , . . .

, τ N )).

We first compute the Boltzmann distribution for each Q k DISPLAYFORM3 where T > 0 is a temperature parameter.

Next, calculate the average Boltzmann distribution DISPLAYFORM4 The InfoGain exploration bonus is the average KL-divergence from DISPLAYFORM5 The modified reward isr (s, a, s ) = r(s, a) DISPLAYFORM6 where ρ ∈ R + is a hyperparameter that controls the degree of exploration.

The exploration bonus b T (s t ) encourages the agent to explore where {Q k } disagree.

The temperature parameter T controls the sensitivity to discrepancies among {Q k }.

When T → +∞, {P T,k } converge to the uniform distribution on the action space and b T (s) → 0.

When T is small, the differences among {Q k } are magnified and b T (s) is large.

We display Algorithrim 3, which incorporates our InfoGain exploration bonus into Algorithm 2.

The hyperparameters λ, T and ρ vary for each game.

We demonstrate the performance of the combined UCB+InfoGain exploration in FIG4 and FIG4 .

We augment the previous figures in Section 5 with the performance of ucb+infogain exploration, where we set λ = 0.1, ρ = 1, and T = 1 in Algorithm 3.

FIG4 shows that combining UCB and InfoGain exploration does not lead to uniform improvement in the normalized learning curve.

At the individual game level, FIG4 shows that the impact of InfoGain exploration varies.

UCB exploration achieves sufficient exploration in games including Demon Attack and Kangaroo and Riverraid, while InfoGain exploration further improves learning on Enduro, Seaquest, and Up N Down.

The effect of InfoGain exploration depends on the choice of the temperature T. The optimal temperature parameter varies across games.

In FIG5 , we display the behavior of ucb+infogain exploration with different temperature values.

Thus, we see the InfoGain exploration bonus, tuned with the appropriate temperature parameter, can lead to improved learning for games that require extra exploration, such as ChopperCommand, KungFuMaster, Seaquest, UpNDown.

<|TLDR|>

@highlight

Adapting UCB exploration to ensemble Q-learning improves over prior methods such as Double DQN, A3C+ on Atari benchmark
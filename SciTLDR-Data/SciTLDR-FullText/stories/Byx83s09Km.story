Efficient exploration remains a major challenge for reinforcement learning.

One reason is that the variability of the returns often depends on the current state and action, and is therefore heteroscedastic.

Classical exploration strategies such as upper confidence bound algorithms and Thompson sampling fail to appropriately account for heteroscedasticity, even in the bandit setting.

Motivated by recent findings that address this issue in bandits, we propose to use Information-Directed Sampling (IDS) for exploration in reinforcement learning.

As our main contribution, we build on recent advances in distributional reinforcement learning and propose a novel, tractable approximation of IDS for deep Q-learning.

The resulting exploration strategy explicitly accounts for both parametric uncertainty and heteroscedastic observation noise.

We evaluate our method on Atari games and demonstrate a significant improvement over alternative approaches.

In Reinforcement Learning (RL), an agent seeks to maximize the cumulative rewards obtained from interactions with an unknown environment.

Given only knowledge based on previously observed trajectories, the agent faces the exploration-exploitation dilemma: Should the agent take actions that maximize rewards based on its current knowledge or instead investigate poorly understood states and actions to potentially improve future performance.

Thus, in order to find the optimal policy the agent needs to use an appropriate exploration strategy.

Popular exploration strategies, such as -greedy BID37 , rely on random perturbations of the agent's policy, which leads to undirected exploration.

The theoretical RL literature offers a variety of statistically-efficient methods that are based on a measure of uncertainty in the agent's model.

Examples include upper confidence bound (UCB) BID0 and Thompson sampling (TS) BID40 .

In recent years, these have been extended to practical exploration algorithms for large state-spaces and shown to improve performance BID27 O'Donoghue et al., 2018; BID15 .

However, these methods assume that the observation noise distribution is independent of the evaluation point, while in practice heteroscedastic observation noise is omnipresent in RL.

This means that the noise depends on the evaluation point, rather than being identically distributed (homoscedastic).

For instance, the return distribution typically depends on a sequence of interactions and, potentially, on hidden states or inherently heteroscedastic reward observations.

BID20 recently demonstrated that, even in the simpler bandit setting, classical approaches such as UCB and TS fail to efficiently account for heteroscedastic noise.

In this work, we propose to use Information-Directed Sampling (IDS) BID31 BID20 for efficient exploration in RL.

The IDS framework can be used to design exploration-exploitation strategies that balance the estimated instantaneous regret and the expected information gain.

Importantly, through the choice of an appropriate information-gain function, IDS is able to account for parametric uncertainty and heteroscedastic observation noise during exploration.

As our main contribution, we propose a novel, tractable RL algorithm based on the IDS principle.

We combine recent advances in distributional RL BID4 BID12 and approximate parameter uncertainty methods in order to develop both homoscedastic and heteroscedastic variants of an agent that is similar to DQN BID25 , but uses informationdirected exploration.

Our evaluation on Atari 2600 games shows the importance of accounting for heteroscedastic noise and indicates that at our approach can substantially outperform alternative state-of-the-art algorithms that focus on modeling either only epistemic or only aleatoric uncertainty.

To the best of our knowledge, we are the first to develop a tractable IDS algorithm for RL in large state spaces.

Exploration algorithms are well understood in bandits and have inspired successful extensions to RL BID8 BID22 .

Many strategies rely on the "optimism in the face of uncertainty" BID21 principle.

These algorithms act greedily w.r.t.

an augmented reward function that incorporates an exploration bonus.

One prominent example is the upper confidence bound (UCB) algorithm BID0 , which uses a bonus based on confidence intervals.

A related strategy is Thompson sampling (TS) BID40 , which samples actions according to their posterior probability of being optimal in a Bayesian model.

This approach often provides better empirical results than optimistic strategies BID9 .In order to extend TS to RL, one needs to maintain a distribution over Markov Decision Processes (MDPs), which is difficult in general.

Similar to TS, BID28 propose randomized linear value functions to maintain a Bayesian posterior distribution over value functions.

Bootstrapped DQN BID27 extends this idea to deep neural networks by using an ensemble of Qfunctions.

To explore, Bootstrapped DQN randomly samples a Q-function from the ensemble and acts greedily w.r.t.

the sample.

BID15 and BID29 investigate a similar idea and propose to adaptively perturb the parameter-space, which can also be thought of as tracking an approximate parameter posterior.

O'Donoghue et al. (2018) propose TS in combination with an uncertainty Bellman equation, which propagates agent's uncertainty in the Q-values over multiple time steps.

Additionally, propose to use the Q-ensemble of Bootstrapped DQN to obtain approximate confidence intervals for a UCB policy.

There are also multiple other ways to approximate parametric posterior in neural networks, including Neural Bayesian Linear Regression BID34 BID1 , Variational Inference BID6 , Monte Carlo methods (Neal, 1995; BID24 BID44 , and Bayesian Dropout BID16 .

For an empirical comparison of these, we refer the reader to BID30 .A shortcoming of all approaches mentioned above is that, while they consider parametric uncertainty, they do not account for heteroscedastic noise during exploration.

In contrast, distributional RL algorithms, such as Categorical DQN (C51) BID4 and Quantile Regression DQN (QR-DQN) BID12 , approximate the distribution over the Q-values directly.

However, both methods do not take advantage of the return distribution for exploration and use -greedy exploration.

Implicit Quantile Networks (IQN) BID11 instead use a risksensitive policy based on a return distribution learned via quantile regression and outperform both C51 and QR-DQN on Atari-57.

Similarly, Moerland et al. (2018) and BID13 act optimistically w.r.t.

the return distribution in deterministic MDPs.

However, these approaches to not consider parametric uncertainty.

Return and parametric uncertainty have previously been combined for exploration by BID39 and BID26 .

Both methods account for parametric uncertainty by sampling parameters that define a distribution over Q-values.

The former then act greedily with respect to the expectation of this distribution, while the latter additionally samples a return for each action and then acts greedily with respect to it.

However, like Thompson sampling, these approaches do not appropriately exploit the heteroscedastic nature of the return.

In particular, noisier actions are more likely to be chosen, which can slow down learning.

Our method is based on Information-Directed Sampling (IDS), which can explicitly account for parametric uncertainty and heteroscedasticity in the return distribution.

IDS has been primarily studied in the bandit setting BID31 BID20 .

BID45 extend it to finite MDPs, but their approach remains impractical for large state spaces, since it requires to find the optimal policies for a set of MDPs at the beginning of each episode.

We model the agent-environment interaction with a MDP (S, A, R, P, ??), where S and A are the state and action spaces, R(s, a) is the stochastic reward function, P (s |s, a) is the probability of transitioning from state s to state s after taking action a, and ?? ??? [0, 1) is the discount factor.

A policy ??(??|s) ??? P(A) maps a state s ??? S to a distribution over actions.

For a fixed policy ??, the discounted return of action a in state s is a random variable Z ?? (s, a) = ??? t=0 ?? t R(s t , a t ), with initial state s = s 0 and action a = a 0 and transition probabilities s t ??? P (??|s t???1 , a t???1 ), a t ??? ??(??|s t ).

The return distribution Z statisfied the Bellman equation, DISPLAYFORM0 where D = denotes distributional equality.

If we take the expectation of (1), the usual Bellman equation BID5 for the Q-function, DISPLAYFORM1 The objective is to find an optimal policy ?? * that maximizes the expected total discounted return DISPLAYFORM2

To find such an optimal policy, the majority of RL algorithms use a point estimate of the Q-function, Q(s, a).

However, such methods can be inefficient, because they can be overconfident about the performance of suboptimal actions if the optimal ones have not been evaluated before.

A natural solution for more efficient exploration is to use uncertainty information.

In this context, there are two source of uncertainty.

Parametric (epistemic) uncertainty is a result of ambiguity over the class of models that explain that data seen so far, while intrinsic (aleatoric) uncertainty is caused by stochasticity in the environment or policy, and is captured by the distribution over returns BID26 .

BID27 estimate parametric uncertainty with a Bootstrapped DQN.

They maintain an ensemble of K Q-functions, DISPLAYFORM0 , which is represented by a multi-headed deep neural network.

To train the network, the standard bootstrap method BID14 BID17 ) constructs K different datasets by sampling with replacement from the global data pool.

Instead, BID27 trains all network heads on the exact same data and diversifies the Q-ensemble via two other mechanisms.

First, each head Q k (s, a; ??) is trained on its own independent target head Q k (s, a; ?? ??? ), which is periodically updated BID25 .

Further, each head is randomly initialized, which, combined with the nonlinear parameterization and the independently targets, provides sufficient diversification.

Intrinsic uncertainty is captured by the return distribution Z ?? .

While Q-learning (Watkins, 1989) aims to estimate the expected discounted return DISPLAYFORM1 , distributional RL approximates the random return Z ?? (s, a) directly.

As in standard Q-learning BID43 , one can define a distributional Bellman optimality operator based on (1), DISPLAYFORM2 To estimate the distribution of Z, we use the approach of C51 BID4 in the following.

It parameterizes the return as a categorical distribution over a set of equidistant atoms in a fixed interval [V min , V max ].

The atom probabilities are parameterized by a softmax distribution over the outputs of a parametric model.

Since the parameterization Z ?? and the Bellman update T Z ?? have disjoint supports, the algorithm requires an additional step ?? that projects the shifted support DISPLAYFORM3

In RL, heteroscedasticity means that the variance of the return distribution Z depends on the state and action.

This can occur in a number of ways.

The variance Var(R|s, a) of the reward function itself may depend on s or a. Even with deterministic or homoscedastic rewards, in stochastic environments the variance of the observed return is a function of the stochasticity in the transitions over a sequence of steps.

Furthermore, Partially Observable MDPs (Monahan, 1982) are also heteroscedastic due to the possibility of different states aliasing to the same observation.

Interestingly, heteroscedasticity also occurs in value-based RL regardless of the environment.

This is due to Bellman targets being generated based on an evolving policy ??.

To demonstrate this, consider a standard observation model used in supervised learning y t = f (x t ) + t (x t ), with true function f and Gaussian noise t (x t ).

In Temporal Difference (TD) algorithms BID37 , given a sample transition (s t , a t , r t , s t+1 ), the learning target is generated as y t = r t + ??Q ?? (s t+1 , a ), for some action a .

Similarly to the observation model above, we can describe TD-targets for learning Q * being generated as DISPLAYFORM0 The last term clearly shows the dependence of the noise function ?? t (s, a) on the policy ??, used to generate the Bellman target.

Note additionally that heteroscedastic targets are not limited to TDlearning methods, but also occur in TD(??) and Monte-Carlo learning BID37 , no matter if the environment is stochastic or not.

Information-Directed Sampling (IDS) is a bandit algorithm, which was first introduced in the Bayesian setting by BID31 , and later adapted to the frequentist framework by BID20 .

Here, we concentrate on the latter formulation in order to avoid keeping track of a posterior distribution over the environment, which itself is a difficult problem in RL.

The bandit problem is equivalent to a single state MDP with stochastic reward function R(a, s) = R(a) and optimal action a * = arg max a???A E[R(a)].

We define the (expected) regret DISPLAYFORM0 , which is the loss in reward for choosing an suboptimal action a. Note, however, that we cannot directly compute ???(a), since it depends on R and the unknown optimal action a * .

Instead, IDS uses a conservative regret estimate??? t (a) = max a ???A u t (a ) ??? l t (a), where [l t (a), u t (a)] is a confidence interval which contains the true expected reward E[R(a)] with high probability.

In addition, assume for now that we are given an information gain function I t (a).

Then, at any time step t, the IDS policy is defined by DISPLAYFORM1 Technically, this is known as deterministic IDS which, for simplicity, we refer to as IDS throughout this work.

Intuitively, IDS chooses actions with small regret-information ratio?? t (a) :=??? DISPLAYFORM2 It(a) to balance between incurring regret and acquiring new information at each step.

BID20 introduce several information-gain functions and derive a high-probability bound on the cumulative regret, DISPLAYFORM3 Here, ?? T is an upper bound on the total information gain T t=1 I t (a t ), which has a sublinear dependence in T for different function classes and the specific information-gain function we use in the following BID35 .

The overall regret bound for IDS matches the best bound known for the widely used UCB policy for linear and kernelized reward functions.

One particular choice of the information gain function, that works well empirically and we focus on in the following, is I t (a) = log 1 + ?? t (a) 2 /??(a) 2 BID20 .

Here ?? t (a) 2 is the variance in the parametric estimate of E[R(a)] and ??(a) 2 = Var[R(a)] is the variance of the observed reward.

In particular, the information gain I t (a) is small for actions with little uncertainty in the true expected reward or with reward that is subject to high observation noise.

Importantly, note that ??(a) 2 may explicitly depend on the selected action a, which allows the policy to account for heteroscedastic noise.

We demonstrate the advantage of such a strategy in the Gaussian Process setting (Murphy, 2012).

In particular, for an arbitrary set of actions a 1 , . . .

, a N , we model the distribution of R(a 1 ), . . .

, R(a N ) by a multivariate Gaussian, with covariance Cov[R(a i ), R(a j )] = ??(x i , x j ), where ?? is a positive definite kernel.

In our toy example, the goal is to maximize R(x) under heteroscedastic observation noise with variance ??(x) 2 ( FIG0 ).

As UCB and TS do not consider observation noise in the acquisition function, they may sample at points where ??(x) 2 is large.

Instead, by exploiting kernel correlation, IDS is able to shrink the uncertainty in the high-noise region with fewer samples, by selecting a nearby point with potentially higher regret but small noise.

In this section, we use the IDS strategy from the previous section in the context of deep RL.

In order to do so, we have to define a tractable notion of regret ??? t and information gain I t .

In the context of RL, it is natural to extend the definition of instantaneous regret of action a in state s using the Q-function DISPLAYFORM0 where F t = {s 1 , a 1 , r 1 , . . .

s t , a t , r t } is the history of observations at time t. The regret definition in eq. (6) captures the loss in return when selecting action a in state s rather than the optimal action.

This is similar to the notion of the advantage function.

Since ??? ?? t (s, a) depends on the true Qfunction Q ?? , which is not available in practice and can only be estimated based on finite data, the IDS framework instead uses a conservative estimate.

To do so, we must characterize the parametric uncertainty in the Q-function.

Since we use neural networks as function approximators, we can obtain approximate confidence bounds using a Bootstrapped DQN BID27 .

In particular, given an ensemble of K action-value functions, we compute the empirical mean and variance of the estimated Q-values, DISPLAYFORM1 Based on the mean and variance estimate in the Q-values, we can define a surrogate for the regret using confidence intervals, DISPLAYFORM2 where ?? t is a scaling hyperparameter.

The first term corresponds to the maximum plausible value that the Q-function could take at a given state, while the right term lower-bounds the Q-value given the chosen action.

As a result, eq. (8) provides a conservative estimate of the regret in eq. (6).

DISPLAYFORM3 Execute action a t = arg min a???A?? (s t , a), observe r t and state s t+1 end for end for Given the regret surrogate, the only missing component to use the IDS strategy in eq. FORMULA9 is to compute the information gain function I t .

In particular, we use I t (a) = log 1 + ?? t (a) 2 /??(a) 2 based on the discussion in BID20 .

In addition to the previously defined predictive parameteric variance estimates for the regret, it depends on the variance of the noise distribution, ??.

While in the bandit setting we track one-step rewards, in RL we focus on learning from returns from complete trajectories.

Therefore, instantaneous reward observation noise variance ??(a) 2 in the bandit setting transfers to the variance of the return distribution Var (Z(s, a)) in RL.

We point out that the scale of Var (Z(s, a)) can substantially vary depending on the stochasticity of the policy and the environment, as well as the reward scaling.

This directly affects the scale of the information gain and the degree to which the agent chooses to explore.

Since the weighting between regret and information gain in the IDS ratio is implicit, for stable performance across a range of environments, we propose computing the information gain I(s, a) = log 1 + ??(s,a) 2 ??(s,a) 2 + 2 using the normalized variance DISPLAYFORM4 where 1 , 2 are small constants that prevent division by 0.

This normalization step brings the mean of all variances to 1, while keeping their values positive.

Importantly, it preserves the signal needed for noise-sensitive exploration and allows the agent to account for numerical differences across environments and favor the same amount of risk.

We also experimentally found this version to give better results compared to the unnormalized variance ??(s, a) 2 = Var (Z(s, a)).

Using the estimates for regret and information gain, we provide the complete control algorithm in Algorithm 1.

At each step, we compute the parametric uncertainty over Q(s, a) as well as the distribution over returns Z(s, a).

We then follow the steps from Section 4.1 to compute the regret and the information gain of each action, and select the one that minimizes the regret-information ratio??(s, a).To estimate parametric uncertainty, we use the exact same training procedure and architecture as Bootstrapped DQN BID27 : we split the DQN architecture BID25 into K bootstrap heads after the convolutional layers.

Each head Q k (s, a; ??) is trained against its own target head Q k (s, a; ?? ??? ) and all heads are trained on the exact same data.

We use Double DQN targets BID41 and normalize gradients propagated by each head by 1/K.To estimate Z(s, a), it makes sense to share some of the weights ?? from the Bootstrapped DQN.

We propose to use the output of the last convolutional layer ??(s) as input to a separate head that estimates Z(s, a).

The output of this head is the only one used for computing ??(s, a) 2 and is also not included in the bootstrap estimate.

For instance, this head can be trained using C51 or QR- DISPLAYFORM0 2 , where z i denotes the atoms of the distribution support, p i , their corresponding probabilities, and E[Z(s, a)] = i p i z i .

To isolate the effect of noise-sensitive exploration from the advantages of distributional training, we do not propagate distributional loss gradients in the convolutional layers and use the representation ??(s) learned only from the bootstrap branch.

This is not a limitation of our approach and both (or either) bootstrap and distributional gradients can be propagated through the convolutional layers.

Importantly, our method can account for deep exploration, since both the parametric uncertainty ??(s, a) 2 and the intrinsic uncertainty ??(s, a) 2 estimates in the information gain are extended beyond a single time step and propagate information over sequences of states.

We note the difference with intrinsic motivation methods, which augment the reward function by adding an exploration bonus to the step reward BID18 BID36 BID33 BID3 BID38 .

While the bonus is sometimes based on an information-gain measure, the estimated optimal policy is often affected by the augmentation of the rewards.

We now provide experimental results on 55 of the Atari 2600 games from the Arcade Learning Environment (ALE) BID2 , simulated via the OpenAI gym interface BID7 .

We exclude Defender and Surround from the standard Atari-57 selection, since they are not available in OpenAI gym.

Our method builds on the standard DQN architecture and we expect it to benefit from recent improvements such as Dueling DQN BID42 and prioritized replay .

However, in order to separately study the effect of changing the exploration strategy, we compare our method without these additions.

Our code can be found at https:// github.com/nikonikolov/rltf/tree/ids-drl.We evaluate two versions of our method: a homoscedastic one, called DQN-IDS, for which we do not estimate Z(s, a) and set ??(s, a)2 to a constant, and a heteroscedastic one, C51-IDS, for which we estimate Z(s, a) using C51 as previously described.

DQN-IDS uses the exact same network architecture as Bootstrapped DQN.

For C51-IDS, we add the fully-connected part of the C51 network BID4 on top of the last convolutional layer of the DQN-IDS architecture, but we do not propagate distributional loss gradients into the convolutional layers.

We use a target network to compute Bellman updates, with double DQN targets only for the bootstrap heads, but not for the distributional update.

Weights are updated using the Adam optimizer BID19 .

We evaluate the performance of our method using a mean greedy policy that is computed on the bootstrap heads arg max DISPLAYFORM0 Due to computational limitations, we did not perform an extensive hyperparameter search.

Our final algorithm uses ?? = 0.1, ??(s, a) 2 = 1.0 (for DQN-IDS) and target update frequency of 40000 agent steps, based on a parameter search over ?? ??? {0.1, 1.0}, ?? 2 ??? {0.5, 1.0}, and target update in {10000, 40000}. For C51-IDS, we put a heuristically chosen lower bound of 0.25 on ??(s, a) 2 to prevent the agent from fixating on "noiseless" actions.

This bound is introduced primarily for numerical reasons, since, even in the bandit setting, the strategy may degenerate as the noise variance of a single action goes to zero.

We also ran separate experiments without this lower bound and while the per-game scores slightly differ, the overall change in mean human-normalized score was only 23%.

We also use the suggested hyperparameters from C51 and Bootstrapped DQN, and set learning rate ?? = 0.00005, ADAM = 0.01/32, number of heads K = 10, number of atoms N = 51.

The rest of our training procedure is identical to that of BID25 , with the difference that we do not use -greedy exploration.

All episodes begin with up to 30 random no-ops BID25 and the horizon is capped at 108K frames BID41 .

Complete details are provided in Appendix A.To provide comparable results with existing work we report evaluation results under the best agent protocol.

Every 1M training frames, learning is frozen, the agent is evaluated for 500K frames and performance is computed as the average episode return from this latest evaluation run.

TAB0 shows the mean and median human-normalized scores BID41 of the best agent performance after 200M training frames.

Additionally, we illustrate the distributions learned by C51 and C51-IDS in FIG2 .We first point out the results of DQN-IDS and Bootstrapped DQN.

While both methods use the same architecture and similar optimization procedures, DQN-IDS outperforms Bootstrapped DQN by around 200%.

This suggests that simply changing the exploration strategy from TS to IDS (along with the type of optimizer), even without accounting for heteroscedastic noise, can substantially improve performance.

Furthermore, DQN-IDS slightly outperforms C51, even though C51 has the benefits of distributional learning.

We also see that C51-IDS outperforms C51 and QR-DQN and achieves slightly better results than IQN.

Importantly, the fact that C51-IDS substantially outperforms DQN-IDS, highlights the significance of accounting for heteroscedastic noise.

We also experimented with a QRDQN-IDS version, which uses QR-DQN instead of C51 to estimate Z(s, a) and noticed that our method can benefit from better approximation of the return distribution.

While we expect the performance over IQN to be higher, we do not include QRDQN-IDS scores since we were unable to reproduce the reported QR-DQN results on some games.

We also note that, unlike C51-IDS, IQN is specifically tuned for risk sensitivity.

One way to get a risk-sensitive IDS policy is by tuning for ?? in the additive IDS formulation??(s, a) =???(s, a) 2 ??? ??I(s, a), proposed by BID31 .

We verified on several games that C51-IDS scores can be improved by using this additive formulation and we believe such gains can be extended to the rest of the games.

We extended the idea of frequentist Information-Directed Sampling to a practical RL exploration algorithm that can account for heteroscedastic noise.

To the best of our knowledge, we are the first to propose a tractable IDS algorithm for RL in large state spaces.

Our method suggests a new way to use the return distribution in combination with parametric uncertainty for efficient deep exploration and demonstrates substantial gains on Atari games.

We also identified several sources of heteroscedasticity in RL and demonstrated the importance of accounting for heteroscedastic noise for efficient exploration.

Additionally, our evaluation results demonstrated that similarly to the bandit setting, IDS has the potential to outperform alternative strategies such as TS in RL.There remain promising directions for future work.

Our preliminary results show that similar improvements can be observed when IDS is combined with continuous control RL methods such as the Deep Deterministic Policy Gradient (DDPG) BID23 .

Developing a computationally efficient approximation of the randomized IDS version, which minimizes the regret-information ratio over the set of stochastic policies, is another idea to investigate.

Additionally, as indicated by BID31 , IDS should be seen as a design principle rather than a specific algorithm, and thus alternative information gain functions are an important direction for future research.

where agent, human and random represent the per-game raw scores.

The return distributions learned by C51-IDS and C51.

Plots obtained by sampling a random batch of 32 states from the replay buffer every 50000 steps and computing the estimates for ?? 2 (s, a) based on eq. (9).

A histogram over the resulting values is then computed and displayed as a distribution (by interpolation).

From top to bottom, the lines on each plot correspond to standard deviation boundaries of a normal distribution [max, ?? + 1.5??, ?? + ??, ?? + 0.5??, ??, ?? ??? 0.5??, ?? ??? ??, ?? ??? 1.5??, min].

The x-axis indicates number of training frames.

BID12 and BID11 .

C51-IDS averaged over 3 seeds.

@highlight

We develop a practical extension of Information-Directed Sampling for Reinforcement Learning, which accounts for parametric uncertainty and heteroscedasticity in the return distribution for exploration.

@highlight

The authors propose a way of extending Information-Directed Sampling to reinforcement learning by combining two types of uncertainty to obtain a simple exploration strategy based on IDS. 

@highlight

This paper investigates sophistical exploration approaches for reinforcement learning built on Information Direct Sampling and on Distributional Reinforcement Learning
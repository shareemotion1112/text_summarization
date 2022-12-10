Sample inefficiency is a long-lasting problem in reinforcement learning (RL).

The state-of-the-art uses action value function to derive policy while it usually involves an extensive search over the state-action space and unstable optimization.

Towards the sample-efficient RL, we propose ranking policy gradient (RPG), a policy gradient method that learns the optimal rank of a set of discrete actions.

To accelerate the learning of policy gradient methods, we establish the equivalence between maximizing the lower bound of return and imitating a near-optimal policy without accessing any oracles.

These results lead to a general off-policy learning framework, which preserves the optimality, reduces variance, and improves the sample-efficiency.

We conduct extensive experiments showing that when consolidating with the off-policy learning framework, RPG substantially reduces the sample complexity, comparing to the state-of-the-art.

One of the major challenges in reinforcement learning (RL) is the high sample complexity (Kakade et al., 2003) , which is the number of samples must be collected to conduct successful learning.

There are different reasons leading to poor sample efficiency of RL (Yu, 2018) .

Because policy gradient algorithms directly optimizing return estimated from rollouts (e.g., REINFORCE (Williams, 1992) ) could suffer from high variance (Sutton & Barto, 2018) , value function baselines were introduced by actor-critic methods to reduce the variance and improve the sample-efficiency.

However, since a value function is associated with a certain policy, the samples collected by former policies cannot be readily used without complicated manipulations (Degris et al., 2012) and extensive parameter tuning (Nachum et al., 2017) .

Such an on-policy requirement increases the difficulty of sampleefficient learning.

On the other hand, off-policy methods, such as one-step Q-learning (Watkins & Dayan, 1992) and variants of deep Q networks (DQN) (Mnih et al., 2015; Hessel et al., 2017; Dabney et al., 2018; Van Hasselt et al., 2016; Schaul et al., 2015) , enjoys the advantage of learning from any trajectory sampled from the same environment (i.e., off-policy learning), are currently among the most sampleefficient algorithms.

These algorithms, however, often require extensive searching (Bertsekas & Tsitsiklis, 1996, Chap.

5) over the large state-action space to estimate the optimal action value function.

Another deficiency is that, the combination of off-policy learning, bootstrapping, and function approximation, making up what Sutton & Barto (2018) called the "deadly triad", can easily lead to unstable or even divergent learning (Sutton & Barto, 2018, Chap.

11) .

These inherent issues limit their sample-efficiency.

Towards addressing the aforementioned challenge, we approach the sample-efficient reinforcement learning from a ranking perspective.

Instead of estimating optimal action value function, we concentrate on learning optimal rank of actions.

The rank of actions depends on the relative action values.

As long as the relative action values preserve the same rank of actions as the optimal action values (Q-values), we choose the same optimal action.

To learn optimal relative action values, we propose the ranking policy gradient (RPG) that optimizes the actions' rank with respect to the long-term reward by learning the pairwise relationship among actions.

Ranking Policy Gradient (RPG) that directly optimizes relative action values to maximize the return is a policy gradient method.

The track of off-policy actor-critic methods (Degris et al., 2012; Gu et al., 2016; Wang et al., 2016) have made substantial progress on improving the sample-efficiency of policy gradient.

However, the fundamental difficulty of learning stability associated with the bias-variance trade-off remains (Nachum et al., 2017) .

In this work, we first exploit the equivalence between RL optimizing the lower bound of return and supervised learning that imitates a specific optimal policy.

Build upon this theoretical foundation, we propose a general off-policy learning framework that equips the generalized policy iteration (Sutton & Barto, 2018, Chap.

4) with an external step of supervised learning.

The proposed off-policy learning not only enjoys the property of optimality preserving (unbiasedness), but also largely reduces the variance of policy gradient because of its independence of the horizon and reward scale.

Besides, we empirically show that there is a trade-off between optimality and sample-efficiency.

Last but not least, we demonstrate that the proposed approach, consolidating the RPG with off-policy learning, significantly outperforms the state-of-the-art (Hessel et al., 2017; Bellemare et al., 2017; Dabney et al., 2018; Mnih et al., 2015) .

Sample Efficiency.

The sample efficient reinforcement learning can be roughly divided into two categories.

The first category includes variants of Q-learning (Mnih et al., 2015; Schaul et al., 2015; Van Hasselt et al., 2016; Hessel et al., 2017) .

The main advantage of Q-learning methods is the use of off-policy learning, which is essential towards sample efficiency.

The representative DQN (Mnih et al., 2015) introduced deep neural network in Q-learning, which further inspried a track of successful DQN variants such as Double DQN (Van Hasselt et al., 2016) , Dueling networks (Wang et al., 2015) , prioritized experience replay (Schaul et al., 2015) , and RAINBOW (Hessel et al., 2017) .

The second category is the actor-critic approaches.

Most of recent works (Degris et al., 2012; Wang et al., 2016; Gruslys et al., 2018) in this category leverage importance sampling by re-weighting the samples to correct the estimation bias and reduce variance.

Its main advantage is in the wall-clock times due to the distributed framework, firstly presented in (Mnih et al., 2016) , instead of the sampleefficiency.

As of the time of writing, the variants of DQN (Hessel et al., 2017; Dabney et al., 2018; Bellemare et al., 2017; Schaul et al., 2015; Van Hasselt et al., 2016) are among the algorithms of most sample efficiency, which are adopted as our baselines for comparison.

and supervised learning such as Expectation-Maximization algorithms (Dayan & Hinton, 1997; Peters & Schaal, 2007; Kober & Peters, 2009; Abdolmaleki et al., 2018) , Entropy-Regularized RL (Oh et al., 2018; Haarnoja et al., 2018) , and Interactive Imitation Learning (IIL) (Daumé et al., 2009; Syed & Schapire, 2010; Ross & Bagnell, 2010; Ross et al., 2011; Sun et al., 2017; Hester et al., 2018; Osa et al., 2018) .

EM-based approaches utilize the probabilistic framework to transfer RL maximizing lower bound of return as a re-weighted regression problem while it requires on-policy estimation on the expectation step.

Entropy-Regularized RL optimizing entropy augmented objectives can lead to off-policy learning without the usage of importance sampling while it converges to soft optimality (Haarnoja et al., 2018) .

Of the three tracks in prior works, the IIL is most closely related to our work.

The IIL works firstly pointed out the connection between imitation learning and reinforcement learning (Ross & Bagnell, 2010; Syed & Schapire, 2010; Ross et al., 2011) and explore the idea of facilitating reinforcement learning by imitating experts.

However, most of imitation learning algorithms assume the access to the expert policy or demonstrations.

Our off-policy learning framework can be interpreted as an online imitation learning approach that constructs expert demonstrations during the exploration without soliciting experts, and conducts supervised learning to maximize return at the same time.

In conclusion, our approach is different from the prior work in terms of at least one of the following aspects: objectives, oracle assumptions, the optimality of learned policy, and on-policy requirement.

More concretely, the proposed method is able to learn both deterministic and stochastic optimal policy in terms of long-term reward, without access to the oracle (such as expert policy or expert demonstration) and it can be trained both empirically and theoretically in an off-policy fashion.

Due to the space limits, we defer the detailed discussion of the related work in the Appendix Section 9.1.

In this paper, we consider a finite horizon T , discrete time Markov Decision Process (MDP) with a finite discrete state space S and for each state s ∈ S, the action space A s is finite.

The environment dynamics is denoted as P = {p(s ′ |s, a), ∀s, s ′ ∈ S, a ∈ A s }.

We note that the dimension of action space can vary given different states.

We use m = max s ∥A s ∥ to denote the maximal action dimension among all possible states.

Our goal is to maximize the expected sum of positive rewards, or return

Value function estimation is widely used in advanced RL algorithms (Mnih et al., 2015; 2016; Schulman et al., 2017; Gruslys et al., 2018; Hessel et al., 2017; Dabney et al., 2018) to facilitate the learning process.

In practice, the on-policy requirement of value function estimations in actor-critic methods has largely increased the difficulty of sample-efficient learning (Degris et al., 2012; Gruslys et al., 2018) .

With the advantage of off-policy learning, the DQN (Mnih et al., 2015) variants are currently among the most sample-efficient algorithms (Hessel et al., 2017; Dabney et al., 2018; Bellemare et al., 2017) .

For complicated tasks, the value function can align with the relative relationship of action's return, but the absolute values are hardly accurate (Mnih et al., 2015; Ilyas et al., 2018) .

The above observations motivate us to look at the decision phase of RL from a different prospect: Given a state, the decision making is to perform a relative comparison over available actions and then choose the best action, which can lead to relatively higher return than others.

Therefore, an alternative solution is to learn the optimal rank of the actions.

In this section, we show how to optimize the rank of actions to maximize the return, and thus avoid the necessity of accurate estimation for optimal action value function.

The optimal relative action values should preserve the same optimal action as the optimal action values:

where Q π * (s, a i ) and λ(s, a i ) represent the optimal action value and the relative action value of action a i , respectively.

We omit the model parameter θ in λ θ (s, a i ) for concise presentation.

A π (s, a) = Q π (s, a) − V π (s).

To learn the λ-values, we can construct a probabilistic model of λ-values such that the best action has the highest probability to be selected than others.

Inspired by learning to rank (Burges et al., 2005) , we consider the pairwise relationship among all actions, by modeling the probability (denoted as p ij ) of an action a i to be ranked higher than any action a j as follows:

where p ij = 0.5 means the relative action value of a i is same as that of the action a j , p ij > 0.5 indicates that the action a i is ranked higher than a j .

Given the independent Assumption 1, we can represent the probability of selecting one action as the multiplication of a set of pairwise probabilities in Eq (1).

Formally, we define the pairwise ranking policy in Eq (2).

Please refer to Section 9.10 in the Appendix for the discussions on feasibility of Assumption 1.

Definition 2.

The pairwise ranking policy is defined as:

where the p ij is defined in Eq (1 Our ultimate goal is to maximize the long-term reward through optimizing the pairwise ranking policy or equivalently optimizing pairwise relationship among the action pairs.

Ideally, we would like the pairwise ranking policy selects the best action with the highest probability and the highest λ-value.

To achieve this goal, we resort to the policy gradient method.

Formally, we propose the ranking policy gradient method (RPG), as shown in Theorem 1.

and the deterministic pairwise ranking policy π θ is:

The proof of Theorem 1 is available in Appendix Section 9.2.

Theorem 1 states that optimizing the discrepancy between the relative action values of the best action and all other actions, is optimizing the pairwise relationships that maximize the return.

One limitation of RPG is that it is not convenient for the tasks where only optimal stochastic policies exist since the pairwise ranking policy takes extra efforts to construct a probability distribution [see Section 9.3 in Appendix].

In order to learn the stochastic policy, we introduce Listwise Policy Gradient (LPG) that optimizes the probability of ranking a specific action on the top of a set of actions, with respect to the return.

In the context of RL, this top one probability is the probability of action a i to be chosen, which is equal to the sum of probability all possible permutations that map action a i in the top.

Inspired by listwise learning to rank approach (Cao et al., 2007) , the top one probability can be modeled by the softmax function.

Therefore, LPG is equivalent to the REINFORCE (Williams, 1992) algorithm with a softmax layer.

LPG provides another interpretation of REINFORCE algorithm from the perspective of learning the optimal ranking and enables the learning of both deterministic policy and stochastic policy.

Due to the space limit, we defer the detailed description of LPG in Appendix Section 9.4.

To this end, seeking sample-efficiency motivates us to learn the relative relationship (RPG (Theorem 1) and LPG (Theorem 4)) of actions, instead of seeking accurate estimation of optimal action values and then choosing action greedily.

However, both of the RPG and LPG belong to policy gradient methods, which suffers from large variance and the on-policy learning requirement (Sutton & Barto, 2018) .

Therefore, the direct implementation of RPG or LPG is still far from sample-efficient.

In the next section, we will describe a general off-policy learning framework empowered by supervised learning, which provides an alternative way to accelerate learning, preserve optimality, and reduce variance.

In this section, we discuss the connections and discrepancies between RL and supervised learning, and our results lead to a sample-efficient off-policy learning paradigm for RL.

The main result in this section is Theorem 2, which casts the problem of maximizing the lower bound of return into a supervised learning problem, given one relatively mild Assumption 2 and practical Assumptions 1,3.

As we show by Lemma 4 in the Appendix that assumptions are valid in a range of RL tasks.

The central idea is to collect only the near-optimal trajectories when the learning agent interacts with the environment, and imitate the near-optimal policy by maximizing the log likelihood of the stateaction pairs from near-optimal trajectories.

With the road map in mind, we then begin to introduce our approach as follows.

In a discrete action MDP with finite states and horizon, given the near-optimal policy π * , the stationary state distribution is given by:

, where p(s|τ ) is the probability of a certain state given a specific trajectory τ and is not associated with any policies, and only p π * (τ ) is related to the policy parameters.

The stationary distribution of state-action pairs is thus: p π * (s, a) = p π * (s)π * (a|s).

In this section, we consider the MDP that each initial state will lead to at least one (near)-optimal trajectory.

For a more general case, please refer to the discussion in Appendix 9.5.

In order to connect supervised learning (i.e., imitating a near-optimal policy) with RL and enable sample-efficient off-policy learning, we first introduce the trajectory reward shaping (TRS), defined as follows:

Definition 3 (Trajectory Reward Shaping, TRS).

Given a fixed trajectory τ , its trajectory reward is shaped as follows:

where c = R max −ϵ is a problem-dependent near-optimal trajectory reward threshold that indicates the least reward of near-optimal trajectory, ϵ ≥ 0 and ϵ ≪ R max .

We denote the set of all possible near-optimal trajectories as T = {τ |w(τ ) = 1}, i.e., w(τ ) = 1, ∀τ ∈ T .

Remark 2.

The threshold c indicates a trade-off between the sample-efficiency and the optimality.

The higher the threshold, the less frequently it will hit the near-optimal trajectories during exploration, which means it has higher sample complexity, while the final performance is better (see Figure 3) .

Eq (38) in Appendix, Section 9.6).

For the sake of simplicity, we set w(τ ) = 1.

Different from the reward shaping works (Ng et al., 1999) , we directly shape the trajectory reward, which will enable the smooth transform from RL to SL.

After shaping the trajectory reward, we can transfer the goal of RL from maximizing the return to maximize the long-term performance (Def 4).

The long-term performance is the expected shaped trajectory reward, as shown in Eq (4).

By Def 3, the expectation over all trajectories is the equal to that over the near-optimal trajectories in T , i.e.,

The optimality is preserved after trajectory reward shaping (ϵ = 0, c = R max ) since the optimal policy π * maximizing long-term performance is also an optimal policy for original MDP, i.e.,

∈ T (see Lemma 2 in Appendix 9.6).

Similarly, when ϵ > 0, the optimal policy after trajectory reward shaping is a near-optimal policy for original MDP.

Note that most policy gradient methods use softmax function, in which we have ∃τ / ∈ T , p π θ (τ ) > 0 (see Lemma 3 in Appendix 9.6).

Therefore when softmax is used to model a policy, it will not converge to an exact optimal policy.

On the other hand, ideally, the discrepancy of the performance between them can be arbitrarily small based on the universal approximation (Hornik et al., 1989) with general conditions on the activation function and Theorem 1.

in (Syed & Schapire, 2010) .

Essentially, we use TRS to filter out near-optimal trajectories and then we maximize the probabilities of near-optimal trajectories to maximize the long-term performance.

This procedure can be approximated by maximizing the log-likelihood of near-optimal state-action pairs, which is a supervised learning problem.

Before we state our main results, we first introduce the definition of uniformly near-optimal policy (Def 5) and a prerequisite (Asm.

2) specifying the applicability of the results.

Definition 5 (Uniformly Near-Optimal Policy, UNOP).

The Uniformly Near-Optimal Policy π * is the policy whose probability distribution over near-optimal trajectories (T ) is a uniform distribution.

i.e. p π * (τ ) =

|T | , ∀τ ∈ T , where |T | is the number of near-optimal trajectories.

When we set c = R max , it is an optimal policy in terms of both maximizing return and long-term performance.

In the case of c = R max , the corresponding uniform policy is an optimal policy, we denote this type of optimal policy as uniformly optimal policy (UOP).

Based on Lemma 4 in Appendix Section 9.9, Assumption 2 is satisfied for certain MDPs that have deterministic dynamics.

Other than Assumption 2, all other assumptions in this work (Assumptions 1,3) can almost always be satisfied in practice, based on empirical observation.

With these relatively mild assumptions, we present the following long-term performance theorem, which shows the close connection between supervised learning and RL.

Theorem 2 (Long-term Performance Theorem).

Maximizing the lower bound of expected longterm performance (Eq (4) ) is maximizing the log-likelihood of state-action pairs sampled from an uniformly (near)-optimal policy π * , which is a supervised learning problem:

The optimal policy of maximizing the lower bound is also the optimal policy of maximizing the long-term performance and the return.

Remark 4.

It is worth noting that Theorem 2 does not require a uniformly near-optimal policy π * to be deterministic.

The only requirement is the existence of a uniformly near-optimal policy.

Remark 5.

Maximizing the lower bound of long-term performance is to maximize the lower bound of long-term reward since we can set w(τ ) = r(τ ) and

.

An optimal policy of maximizing this lower bound is also an optimal policy of maximizing the longterm performance when c = R max , thus maximizing the return.

The proof of Theorem 2 can be found in Appendix, Section 9.6.

Theorem 2 indicates that we break the dependency between current policy π θ and the environment dynamics, which means off-policy learning is able to be conducted by the above supervised learning approach.

Furthermore, we point out that there is a potential discrepancy between imitating UNOP by maximizing log likelihood (even when the optimal policy's samples are given) and the reinforcement learning since we are maximizing a lower bound of expected long-term performance (or equivalently the return over the near-optimal trajectories only) instead of return over all trajectories.

In practice, the state-action pairs from an optimal policy is hard to construct while the uniform characteristic of UNOP can alleviate this issue (see Sec 6).

Towards sample-efficient RL, we apply Theorem 2 to RPG, which reduces the ranking policy gradient to a classification problem by Corollary 1.

Corollary 1 (Ranking performance policy gradient).

Optimizing the lower bound of expected longterm performance (defined in Eq (4)) using pairwise ranking policy (Eq (2)) can be approximately optimized by the following loss:

where margin is a small positive value.

We set margin equal to one in our experiments.

The proof of Corollary 1 can be found in Appendix, section 9.7.

Similarly, we can reduce LPG to a classification problem (see Appendix 9.7.1).

One advantage of casting RL to SL is variance reduction.

With the proposed off-policy supervised learning, we can reduce the upper bound of the policy gradient variance, as shown in the Corollary 2.

Before introducing the variance reduction results, we first make the following standard assumption similar to (Degris et al., 2012, A1) .

Furthermore, the assumption is guaranteed for bounded continuously differentiable policy such as softmax function.

Assumption 3.

We assume the maximum norm of policy gradient is finite, i.e.

2 C 2 R 2 max ).

The

2 R 2 max ), given R max ≥ 1, T ≥ 1, which

The proof of Corollary 2 can be found in Appendix 9.8.

This corollary shows that the variance of regular policy gradient is upper-bounded by the square of time horizon and the maximum trajectory reward.

It is aligned with our intuition and empirical observation: the longer the horizon the harder the learning.

Also, the common reward shaping tricks such as truncating the reward to [− learning, we concentrate the difficulty of long-time horizon into the exploration phase, which is an inevitable issue for all RL algorithms, and we drop the dependence on T and R max for policy variance.

Thus, it is more stable and efficient to train the policy using supervised learning.

One potential limitation of this method is that the trajectory reward threshold c is task-specific, which is crucial to the final performance and sample-efficiency.

In many applications such as Dialogue system (Li et al., 2017) , recommender system (Melville & Sindhwani, 2011), etc., we design the reward function to guide the learning process, in which c is naturally known.

For the cases that we have no prior knowledge on the reward function of MDP, we treat c as a tuning parameter to balance the optimality and efficiency, as we empirically verified in Figure 3 .

The major theoretical uncertainty on general tasks is the existence of a uniformly near-optimal policy, which is negligible to the empirical performance.

The rigorous theoretical analysis of this problem is beyond the scope of this work.

Based on the discussions in Section 5, we exploit the advantage of reducing RL into supervised learning via a proposed two-stages off-policy learning framework.

As we illustrated in Figure 1 , the proposed framework contains the following two stages:

Generalized Policy Iteration for Exploration.

The goal of the exploration stage is to collect different near-optimal trajectories as frequently as possible.

Under the off-policy framework, the exploration agent and the learning agent can be separated.

Therefore, any existing RL algorithm can be used during the exploration.

The principle of this framework is using the most advanced RL agents as an exploration strategy in order to collect more near-optimal trajectories and leave the policy learning to the supervision stage.

Supervision.

In this stage, we imitate the uniformly near-optimal policy, UNOP (Def 5).

Although we have no access to the UNOP, we can approximate the state-action distribution from UNOP by collecting the near-optimal trajectories only.

The near-optimal samples are constructed online and we are not given any expert demonstration or expert policy beforehand.

This step provides a sampleefficient approach to conduct exploitation, which enjoys the superiority of stability (Figure 2 ), variance reduction (Corollary 2), and optimality preserving (Theorem 2).

The two-stage algorithmic framework can be directly incorporated in RPG and LPG to improve sample efficiency.

The implementation of RPG is given in Algorithm 1, and LPG follows the same procedure except for the difference in the loss function.

The main requirement of Alg.

1 is on the exploration efficiency and the MDP structure.

During the exploration stage, a sufficient amount of the different near-optimal trajectories need to be collected for constructing a representative supervised learning training dataset.

Theoretically, this requirement always holds [see Appendix Section 9.9, Lemma 5], while the number of episodes explored could be prohibitively large, which makes this algorithm sample-inefficient.

This could be a practical concern of the proposed algorithm.

However, according to our extensive empirical observations, we notice that long before the value function based state-of-the-art converges to near-optimal performance, enough amount of near-optimal trajectories are already explored.

Therefore, we point out that instead of estimating optimal action value functions and then choosing action greedily, using value function to facilitate the exploration and imitating UNOP is a more sample-efficient approach.

As illustrated in Figure 1 , value based methods with off-policy learning, bootstrapping, and function approximation could lead to a divergent optimization (Sutton & Barto, 2018, Chap.

11) .

In contrast to resolving the instability, we circumvent this issue via constructing a stationary target using the samples from (near)-optimal trajectories, and perform imitation learning.

This two-stage approach can avoid the extensive exploration of the suboptimal state-action space and reduce the substantial number of samples needed for estimating optimal action values.

In the MDP where we have a high probability of hitting the near-optimal trajectories (such as PONG), the supervision stage can further facilitate the exploration.

It should be emphasized that our work focuses on improving the sample-efficiency through more effective exploitation, rather than developing novel exploration method.

Please refer to the Appendix Section 9.11 for more discussion on exploration efficiency.

Figure 2: The training curves of the proposed RPG and state-of-the-art.

All results are averaged over random seeds from 1 to 5.

The x-axis represents the number of steps interacting with the environment (we update the model every four steps) and the y-axis represents the averaged training episodic return.

The error bars are plotted with a confidence interval of 95%.

To evaluate the sample-efficiency of Ranking Policy Gradient (RPG), we focus on Atari 2600 games in OpenAI gym Bellemare et al. (2013); Brockman et al. (2016) , without randomly repeating the previous action.

We compare our method with the state-of-the-art baselines including DQN Mnih et al. 2017), we report the training performance of all baselines as the increase of interactions with the environment, or proportionally the number of training iterations.

We run the algorithms with five random seeds and report the average rewards with 95% confidence intervals.

The implementation details of the proposed RPG and its variants are given as follows:

EPG: EPG is the stochastic listwise policy gradient (see Appendix Eq (18)) incorporated with the proposed off-policy learning.

More concretely, we apply trajectory reward shaping (TRS, Def 3) to all trajectories encountered during exploration and train vanilla policy gradient using the off-policy samples.

This is equivalent to minimizing the cross-entropy loss (see Appendix Eq (69)) over the near-optimal trajectories.

LPG: LPG is the deterministic listwise policy gradient with the proposed off-policy learning.

The only difference between EPG and LPG is that LPG chooses action deterministically (see Appendix Eq (17)) during evaluation.

RPG: RPG explores the environment using a separate EPG agent in PONG and IQN in other games.

Then RPG conducts supervised learning by minimizing the hinge loss Eq (6).

It is worth noting that the exploration agent (EPG or IQN) can be replaced by any existing exploration method.

In our RPG implementation, we collect all trajectories with the trajectory reward no less than the threshold c without eliminating the duplicated trajectories and we empirically found it is a reasonable simplification.

More details of hyperparameters are provided in the Appendix Section 9.12.

As the results shown in Figure 2 , our approach, RPG, significantly outperform the state-of-the-art baselines in terms of sample-efficiency at all tasks.

Furthermore, RPG not only achieved the most sample-efficient results, but also reached the highest final performance at ROB-OTANK, DOUBLEDUNK, PITFALL, and PONG, comparing to any model-free state-of-the-art.

In reinforcement learning, the stability of algorithm should be emphasized as an important issue.

As we can see from the results, the performance of baselines varies from task to task.

There is no single baseline consistently outperforms others.

In contrast, due to the reduction from RL to supervised learning, RPG is consistently stable and effective across different environments.

In addition to the stability and efficiency, RPG enjoys simplicity at the same time.

In the environment PONG, it is surprising that RPG without any complicated exploration method largely surpassed the sophisticated value-function based approaches.

The effectiveness of pairwise ranking policy and off-policy learning as supervised learning.

To get a better understanding of the underlying reasons that RPG is more sample-efficient than DQN variants, we performed ablation studies in the PONG environment by varying the combination of policy functions with the proposed off-policy learning.

The results of EPG, LPG, and RPG are shown in the bottom right, Figure 2 .

Recall that EPG and LPG use listwise policy gradient (vanilla policy gradient using softmax as policy function) to conduct exploration, the off-policy learning minimizes the cross-entropy loss Eq (69).

In contrast, RPG shares the same exploration method as EPG and LPG while uses pairwise ranking policy Eq (2) in off-policy learning that minimizes hinge loss Eq (6).

We can see that RPG is more sample-efficient than EPG/LPG.

We also compared the most advanced on-policy method Proximal Policy Optimization (PPO) Schulman et al. (2017) with EPG, LPG, and RPG.

The proposed off-policy learning largely surpassed the best on-policy method.

Therefore, we conclude that off-policy as supervised learning contributes to the sample-efficiency substantially, while pairwise ranking policy can further accelerate the learning.

In addition, we compare RPG to off-policy policy gradient approaches: ACER Wang et al. (2016) and self-imitation learning Oh et al. (2018) .

As the results shown, the proposed off-policy learning framework is more sample-efficient than the state-of-the-art off-policy policy gradient approaches.

The optimality-efficiency trade-off.

As reported in Figure 3 , we empirically demonstrated the trade-off between the sample-efficiency and optimality, which is controlled by the trajectory reward threshold (as defined in Def 3).

The higher value of trajectory reward threshold suggests we have higher requirement on defining near-optimal trajectory.

This will increase the difficulty of collecting near-optimal samples during exploration, while it ensures a better final performance.

These experimental results also justified that RPG is also effective in the absence of prior knowledge on trajectory reward threshold, with a mild cost on introducing an additional tuning parameter.

The trade-off between sample efficiency and optimality on DOUBLEDUNK,BREAKOUT, BANKHEIST.

As the trajectory reward threshold (c) increase, more samples are needed for the learning to converge, while it leads to better final performance.

We denote the value of c by the numbers at the end of legends.

In this work, we introduced ranking policy gradient (RPG) methods that, for the first time, resolve RL problem from a ranking perspective.

Furthermore, towards the sample-efficient RL, we propose an off-policy learning framework that allows RL agents to be trained in a supervised learning paradigm.

The off-policy learning framework uses generalized policy iteration for exploration and exploit the stableness of supervised learning for policy learning, which accomplishes the unbiasedness, variance reduction, off-policy learning, and sample efficiency at the same time.

Last but not least, empirical results show that RPG achieves superior performance as compared to the state-of-the-art.

In Table 1 we provide a brief summary of important notations used in the paper:

Notations Definition

The discrepancy of the relative action value of action i and action j. λ ij = λ i − λ j , where λ i = λ(s, a i ).

Notice that the value here is not the estimation of return, it represents which action will have relatively higher return if followed.

The action value function or equivalently the estimation of return taking action a at state s, following policy π.

p ij p ij = P (λ i > λ j ) denotes the probability that i-th action is to be ranked higher than j-th action.

Notice that p ij is controlled by θ through

collected from the environment.

It is worth noting that this trajectory is not associated with any policy.

It only represents a series of state-action pairs.

We also use the abbreviation s t = s(τ, t), a t = a(τ, t) .

The trajectory reward r(τ ) = ∑ T t=1 r(s t , a t ) is the sum of reward along one trajectory.

R max R max is the maximal possible trajectory reward, i.e., R max = max τ r(τ ).

Since we focus on MDPs with finite horizon and immediate reward, therefore the trajectory reward is bounded.

∑ τ The summation over all possible trajectories τ .

The probability of a specific trajectory is collected from the environment given policy

The set of all possible near-optimal trajectories.

|T | denotes the number of near-optimal trajectories in T .

n The number of training samples or equivalently state action pairs sampled from uniformly optimal policy.

m

The number of discrete actions.

There are two main distinctions between supervised learning and reinforcement learning.

In supervised learning, the data distribution D is static and training samples are assumed to be sampled i.i.d.

from D. On the contrary, the data distribution is dynamic in RL.

It is determined by both environment dynamics and the learning policy.

The policy keeps evolving during the learning process, which results in the dynamic data distribution in RL.

Secondly, the training samples we collected are not independently distributed due to the change of learning policy.

These intrinsic difficulties of RL make the learning algorithm unstable and sample-inefficient.

However, if we review the state-of-the-art in RL community, every algorithm eventually acquires the policy, either explicitly or implicitly, which is a mapping from the state to an action or a probability distribution over the action space.

Ultimately, there exists a supervised learning equivalent to the RL problem, if the optimal policies exist.

The paradox is that it is almost impossible to construct this supervised learning equivalent on the fly, without knowing any optimal policy.

Although what is the proper supervision still lingered in the RL community, pioneers have developed a set of insightful approaches to reduce RL into its SL counterpart over the past several decades.

Roughly, we can classify the prior work into the following categories: The early work in the EM track transfers objective by Jensen's inequality and the maximizing the lower bound of the original objective, which resembles Expectation-Maximization procedure and provides policy improvement

Require: The near-optimal trajectory reward threshold c, the number of maximal training episodes Nmax.

Maximum number of time steps in each episode T , and batch size b. 1: while episode < Nmax do 2: repeat 3:

Retrieve state st and sample action at by the specified exploration agent (can be random, ϵ-greedy, or any RL algorithms).

Collect the experience et = (st, at, rt, st+1) and store to the replay buffer.

5:

if t % update step == 0 then 7:

Sample a batch of experience {ej} b j=1 from the near-optimal replay buffer.

8:

Update π θ based on the hinge loss Eq (6) for RPG.

9:

Update exploration agent using samples from regular replay buffer (In simple MDPs such as PONG where we access to near-optimal trajectory frequently, we can use near-optimal replay buffer to update exploration agent).

10:

end if 11:

until terminal st or t − tstart >= T 12:

if return ∑ T t=1 rt ≥ c then 13:

Take the near-optimal trajectory et, t = 1, ..., T in the latest episode from the regular replay buffer into near-optimal replay buffer.

14:

end if 15:

if t % evaluation step == 0 then 16:

Evaluate the RPG agent by greedily choosing the action.

If the best performance is reached, then stop training.

end if 18: end while guarantee.

While pioneering at the time, these works typically focus on the simplified RL setting, such as in Dayan & Hinton (1997) the reward function is not associated with the state or in Peters & Schaal (2008) the goal is to maximize the expected immediate reward and the state distribution is assumed to be fixed.

Later on in Kober & Peters (2009) , the authors extended the EM framework from immediate reward into episodic return.

Recent advance Abdolmaleki et al. (2018) utilizes the EM-framework on a relative entropy objective, which adds a parameter prior as regularization.

As mentioned in the paper, the evaluation step using Retrace Munos et al. (2016) can be unstable even with linear function approximation Touati et al. (2017) .

In general, the estimation step in EM-based algorithms involves on-policy evaluation, which is one difficulty shared for any policy gradient methods.

One of the main motivation that we want to transfer the RL into a supervised learning task is the off-policy learning enable sample efficiency.

To achieve off-policy learning, PGQ O' Donoghue et al. (2016) connected the entropy-regularized policy gradient with Q-learning under the constraint of small regularization.

In the similar framework, Soft Actor-Critic Haarnoja et al. (2018) was proposed to enable sample-efficient and faster convergence under the framework of entropy-regularized RL.

It is able to converge to the optimal policy that optimizes the long-term reward along with policy entropy.

It is an efficient way to model the suboptimal behavior and empirically it is able to learn a reasonable policy.

Although recently the discrepancy between the entropy-regularized objective and original long-term reward has been discussed in O'Donoghue (2018); Eysenbach & Levine (2019), they focus on learning stochastic policy while the proposed framework is feasible for both learning deterministic optimal policy (Corollary 1) and stochastic optimal policy (Corollary 6).

In Oh et al. (2018) , this work shares similarity to our work in terms of the method we collecting the samples.

They collect good samples based on the past experience and then conduct the imitation learning w.r.t those good samples.

However, we differentiate at how do we look at the problem theoretically.

This self-imitation learning procedure was eventually connected to lower-bound-soft-Q-learning, which belongs to entropy-regularized reinforcement learning.

We comment that there is a trade-off between sample-efficiency and modeling suboptimal behaviors.

The more strict requirement we have on the samples collected we have less chance to hit the samples while we are more close to imitating the optimal behavior. (2010) firstly analyzed if the learned policy fails to imitate the expert with a certain probability, what is the performance degradation comparing to the expert.

While the theorem seems to resemble the long-term performance theorem 2, it considers the learning policy is trained through the state distribution induced by the expert, instead of state-action distribution as we did in Theorem 2.

Their theorem thus may be more applicable to the situation where an interactive procedure is needed, such as querying the expert during the training process.

On the contrary, we Table 2 : A comparison with prior work on reducing RL to SL.

The objective column denotes whether the goal is to maximize long-term reward.

The Cont.

Action column denotes whether the method is applicable for both continuous action space and discrete action space.

The Optimality denotes whether the algorithms can model the optimal policy.

The ✓ † denotes the optimality achieved by ERL is w.r.t.

the entropy regularize objective instead of return.

The Off-Policy column denotes if the algorithm enable off-policy learning.

The No Oracle column denotes if the algorithms need to access to certain type of oracle (expert policy or expert demonstrations).

focus on directly applying supervised learning approach without having access to the expert to label the data.

The optimal state-action pairs are collected during exploration and conducting supervised learning on the replay buffer will provide a performance guarantee in terms of long-term expected reward.

Concurrently, a resemble of theorem 2.1 in Ross & Bagnell (2010) is Theorem 1 in Syed & Schapire (2010), the authors reduce the apprenticeship learning to classification, under the assumption that the apprentice policy is deterministic and the misclassification rate at all time steps is bounded, which we do not make.

Within the IIL track, later on the AGGREVATE Ross & Bagnell (2014) was proposed to incorporate the information of action costs to facilitate imitation learning, and a differentiable version called AGGREVATED Sun et al. (2017) was recently developed and achieved impressive empirical results.

Recently, hinge loss was combined with regular Q-learning loss as a pre-training step for learning from demonstration Hester et al. (2018) or as a surrogate loss for imitating optimal trajectories Osa et al. (2018) .

In this work, we show that hinge loss constructs a new type of policy gradient method and can learn optimal policy directly.

In conclusion, our method approaches the problem of reducing RL to SL from a different perspective than all prior works mentioned.

With our construction from RL to SL, the samples collected in replay buffer can satisfy the i.i.d.

assumption exactly since the state action pairs are sampled from the data distribution of UNOP.

A summary of similarity and discrepancy between our method the current best method in each category is shown in the Table 2.

The proof of Theorem 1 can be found as follows:

Proof.

The following proof is based on direct policy differentiation Peters & Schaal (2008); Williams (1992) .

For concise presentation, the subscript t for action value λi, λj, and pij is omitted.

where the trajectory is a series of state-action pairs from t = 1, ..., T , i.e.τ = s1, a1, s2, a2, ..., sT .

From Eq (7) to Eq (8), we use the first-order Taylor expansion of log(1 + e x )|x=0 = log 2 + 1 2

x + O(x 2 ) to further simplify the ranking policy gradient.

Corollary 3.

The pairwise ranking policy as shown in Eq (2) constructs a probability distribution over the set of actions when the action space m is equal to 2, given any relative action values λi, i = 1, 2.

For the cases with m > 2, this conclusion does not hold in general.

It is easy to verify that π(ai|s) > 0, ∑ 2 i=1 π(ai|s) = 1 holds and the same conclusion cannot be applied to m > 2 by constructing counterexamples.

However, we can introduce a dummy action a ′ to form a probability distribution for RPG.

During policy learning, the algorithm will increase the probability of best actions and the probability of dummy action will decrease.

Ideally, if RPG converges to an optimal deterministic policy, the probability of taking best action is equal to one and π(a ′ |s) = 0.

Similarly, we can introduce a dummy trajectory τ ′ with trajectory reward r(τ

The trajectory probability forms a probability distribution since

The proof of a valid trajectory probability is similar to the following proof on π(a|s) is a valid probability distribution with a dummy action.

The practical influence of this is negligible since our goal is to increase the probability of (near)-optimal trajectories.

To present in a clear way, we avoid mentioning dummy trajectory τ ′ in Proof 9.2 while it can be seamlessly included.

This condition can be easily satisfied since in RPG we only focus on the relative relationship of λ-values and we can constrain its range so that λm satisfies the condition 1.

Furthermore, since we can see that m 1 m−1 > 1 is decreasing w.r.t to action dimension m. The larger the action dimension, the less constraint we have on the λ-values.

′ and set π(a = a ′ |s) = 1 − ∑ i π(a = ai|s), which will construct a valid probability distribution (π(a|s)) over the action space A ∪ a ′ .

Proof.

Since we have π(a = ai|s) > 0 ∀i = 1, ..., m and ∑ i π(a = ai|s) + π(a = a ′ |s) = 1.

To prove this is a valid probability distribution, we only need to show that π(a = a ′ |s) ≥ 0, ∀m ≥ 2, i.e.

9.4 LISTWISE POLICY GRADIENT

In order to learn the stochastic policy that optimizes the ranking of actions with respect to the return, we now introduce the Listwise Policy Gradient (LPG) method.

In RL, we want to optimize the probability of each action (ai) to be ranked higher among all actions, which is the sum of the probabilities of all permutations such that the action ai in the top position of the list.

This probability is computationally prohibitive since we need to consider the probability of m! permutations.

Luckily, based on Cao et al. (2007) [Theorem 6], we can model the such probability of action ai to be ranked highest given a set of relative action values by a simple softmax formulation, as described in Theorem 3.

Theorem 3 (Theorem 6 Cao et al. (2007)

where ϕ( * ) is any increasing, strictly positive function.

A common choice of ϕ is the exponential function.

Closely built upon the foundations from learning to rank Cao et al. (2007)

where the listwise ranking policy π θ parameterized by θ is given by Eq (17) for tasks with deterministic optimal policies: a = arg max

or Eq (18)

is the probability that action i being ranked highest, given the current state and all the relative action values λ1 . . .

λm.

The proof of Theorem 4 exactly follows the direct policy differentiation Peters & Schaal (2008); Williams (1992) by replacing the policy to the form of the softmax function.

The action probability π(ai|s), ∀i = 1, ..., m forms a probability distribution over the set of discrete actions [Cao et al. (2007) Lemma 7] .

Theorem 4 states that the vanilla policy gradient Williams (1992) parameterized by a softmax layer is optimizing the probability of each action to be ranked highest, with respect to the long-term reward.

Condition 2 If we want to preserve the optimality by TRS, the optimal trajectories of MDP needs to cover all initial states or equivalently, all initial states will lead to at least one optimal trajectory.

Similarly, the near-optimality is preserved for all MDPs that its near-optimal trajectories cover all initial states.

Theoretically, it is possible to transfer more general MDPs to satisfy Condition 2 and preserve the optimality with potential-based reward shaping Ng et al. (1999) .

More concretely, consider the deterministic binary tree MDP (M1) with the set of initial states S1 = {s1, s .

This reward shaping requires more prior knowledge, which may not be feasible in practice.

A more realistic method is to design a dynamic trajectory reward shaping approach.

In the beginning, we set c(s) = mins∈S 1 r(τ |s(τ, 1) = s), ∀s ∈ S1.

Take M1 as an example, c(s) = 3, ∀s ∈ S1.

During the exploration stage, we track the current best trajectory of each initial state and update c(s) with its trajectory reward.

Nevertheless, if the Condition 2 is not satisfied, we need more sophisticated prior knowledge other than a predefined trajectory reward threshold c to construct the replay buffer (training dataset of UNOP).

The practical implementation of trajectory reward shaping and rigorously theoretical study for general MDPs are beyond the scope of this work.

Under review as a conference paper at ICLR 2020

In this subsection, we reduce maximizing RL objective into a supervised learning problem with Theorem 2.

Before that, we first prove Lemma 1 to link the log probability of a trajectory τ to its state action distribution.

Then using this lemma, we can connect the trajectory probability of UNOP with its state-action distribution, from which we prove the Theorem 2.

Lemma 1.

Given a specific trajectory τ , the averaged state-action pair log-likelihood over horizon T is equal to the weighted sum over the entire state-action space, i.e.: , a1) , ..., (sT , aT )}, denote the unique state action pairs in this trajectory as

, where n is the number of unique stateaction pairs in τ and n ≤ T .

The number of occurrences of a state-action pair (si, ai) in the trajectory τ is denoted as |(si, ai)|.

where from Eq (24) to Eq (25) we use

This thus completes the proof.

Now we are ready to prove the Theorem 2:

Proof.

The following proof holds for arbitrary subset of trajectories T which is determined by the threshold c in Def 5.

The π * is associated with c and this subset of trajectories.

use Lemma 3 ∵ p θ (τ ) > 0 and

The lower bound holds when

, ∀τ ∈ T .

To this end, we maximize the lower bound of the expected long-term performance.

This is the reason that w(τ ) can be set as arbitrary positive constant (38)

log π θ (at|st) Use Assumption 2 the existence of UNOP.

(41)

where π * is a UNOP (Def 5).

Eq (44) can be established based on

The 2nd sum is over all possible state-action pairs. (s, a) represents a specific state-action pair.

In this proof we use st = s (τ, t) and at = a(τ, t) as abbreviations, which denote the t-th state and action in the trajectory τ , respectively.

|T | denotes the number of trajectories in T .

We also use the definition of w(τ ) to only focus on near-optimal trajectories.

We set w(τ ) = 1 for simplicity but it will not affect the conclusion if set to other constants.

Optimality: Furthermore, the optimal solution for the objective function Eq (49) is a uniformly (near)-optimal policy π * .

Therefore, the optimal solution of Eq (49) is also the (near)-optimal solution for the original RL problem since Proof.

We prove this by contradiction.

We assume π is an optimal policy.

If ∃τ

We can find a better policy π ′ by satisfying the following three conditions:

Since p π ′ (τ ) ≥ 0, ∀τ and ∑ τ p π ′ (τ ) = 1, therefore p π ′ constructs a valid probability distribution.

Then the expected long-term performance of π ′ is greater than that of π:

∵ τ ′ / ∈ T , ∴ w(τ ′ ) = 0 and τ1 ∈ T , ∴ w(τ ) = 1

Essentially, we can find a policy π ′ that has higher probability on the optimal trajectory τ1 and zero probability on τ ′ .

This indicates that it is a better policy than π.

Therefore, π is not an optimal policy and it contradicts our assumption, which proves that such τ ′ does not exist.

Therefore, ∀τ / ∈ T , we have pπ(τ ) = 0. (56) if p(st+1|st, at) = 0 or p(s1) = 0, then the probability of sampling τ from any policy is zero.

This trajectory does not exist.

This thus completes the proof.

where margin is a small positive value.

We set margin equal to one in our experiments.

Proof.

In RPG, the policy π θ (a|s) is defined as in Eq (2).

We then replace the action probability distribution in Eq (5) with the RPG policy.

∵ π(a = ai|s) = Π m j=1,j̸ =i pij (59) Because RPG is fitting a deterministic optimal policy, we denote the optimal action given sate s as ai, then we have

where the margin in Eq (66) is a small positive constant.

From Eq (65) to Eq (66), we consider learning a deterministic optimal policy ai = π * (s), where we use index i to denote the optimal action at each state.

The optimal λ-values minimizing Eq (65) (denoted by λ 1 ) need to satisfy λ

The proposed off-policy learning framework indicates the sample complexity is related to exploration efficiency and supervised learning efficiency.

Given a specific MDP, the exploration efficiency of an exploration strategy can be quantified by how frequently we can encounter different (near)-optimal trajectories in the first k episodes.

The supervised learning efficiency under the probably approximately correct framework Valiant (1984) is how many samples we need to collect so that we can achieve good generalization performance with high probability.

Jointly consider the efficiency in two stages, we can theoretically analyze the sample complexity of the proposed off-policy learning framework, which will be provided in the long version of this work.

Improving exploration efficiency is not the focus of this work.

In general, exploration efficiency is highly related to the properties of MDP, such as transition probabilities, horizon, action dimension, etc.

The exploration strategy should be designed according to certain domain knowledge of the MDP to improve the efficiency.

Therefore, we did not specify our exploration strategy but adopt the state-of-the-art to conduct exploration.

Based on the above discussion, we can see that how frequently we can encounter different (near)-optimal trajectories is a bottleneck of sample efficiency for RPG.

In the MDPs with small the transition probabilities of reaching the near-optimal trajectories, we rarely collect any near-optimal trajectories during the early stage of exploration.

The benefit of applying the proposed off-policy framework would be limited.

We present the training details of ranking policy gradient in Table 3 .

The network architecture is the same as the convolution neural network used in DQN Mnih et al. (2015) .

We update the RPG network every four timesteps with a minibatch of size 32.

The replay ratio is equal to eight for all baselines and RPG (except for ACER we use the default setting in openai baselines Dhariwal et al. (2017) for better performance).

@highlight

We propose ranking policy gradient that learns the optimal rank of actions to maximize return. We propose a general off-policy learning framework with the properties of optimality preserving, variance reduction, and sample-efficiency.

@highlight

This paper proposes to reparameterize the policy using a form of ranking to convert the RL problem into a supervised learning problem.

@highlight

This paper presents a new view on policy gradient methods from the perspective of ranking. 
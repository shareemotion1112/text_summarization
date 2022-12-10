Reinforcement learning in environments with large state-action spaces is challenging, as exploration can be highly inefficient.

Even if the dynamics are simple, the optimal policy can be combinatorially hard to discover.

In this work, we propose a hierarchical approach to structured exploration to improve the sample efficiency of on-policy exploration in large state-action spaces.

The key idea is to model a stochastic policy as a hierarchical latent variable model, which can learn low-dimensional structure in the state-action space, and to define exploration by sampling from the low-dimensional latent space.

This approach enables lower sample complexity, while preserving policy expressivity.

In order to make learning tractable, we derive a joint learning and exploration strategy by combining hierarchical variational inference with actor-critic learning.

The benefits of our learning approach are that 1) it is principled, 2) simple to implement, 3) easily scalable to settings with many actions and 4) easily composable with existing deep learning approaches.

We demonstrate the effectiveness of our approach on learning a deep centralized multi-agent policy, as multi-agent environments naturally have an exponentially large state-action space.

In this setting, the latent hierarchy implements a form of multi-agent coordination during exploration and execution (MACE).

We demonstrate empirically that MACE can more efficiently learn optimal policies in challenging multi-agent games with a large number (~20) of agents, compared to conventional baselines.

Moreover, we show that our hierarchical structure leads to meaningful agent coordination.

Reinforcement learning in environments with large state-action spaces is challenging, as exploration can be highly inefficient in high-dimensional spaces.

Hence, even if the environment dynamics are simple, the optimal policy can be combinatorially hard to discover.

However, for many large-scale environments, the high-dimensional state-action space has (often hidden or implicit) low-dimensional structure which can be exploited.

Many natural examples are in collaborative multi-agent problems, whose state-action space is exponentially large in the number of agents, but have a low-dimensional coordination structure.

Consider a simple variant of the Hare-Hunters problem (see FIG0 .

In this game, N = 2 identical hunters need to capture M = 2 identical static prey within T time-steps, and exactly H = 1 hunter is needed to capture each prey.

T is set such that no hunter can capture both preys.

There are two equivalent solutions: hunter 1 captures prey 1 and hunter 2 captures prey 2, or vice versa.

There are also two suboptimal choices: both hunters choose the same prey.

Hence, the hunters must coordinate over a (large) number of time-steps to maximize their reward.

This implies the solution space has low-dimensional structure that can be used to accelerate training.

In this work, we propose a principled approach to structured exploration to improve sample complexity in large state-action spaces, by learning deep hierarchical policies with a latent structure.

As a highlevel intuition, consider a tabular multi-agent policy, which maps discrete (joint) states to action probabilities.

For N agents with S states and A actions each, this policy has O((S · A) N ) weights.

However, the low-dimensional coordination structure can be captured by a factorized, low-rank matrix, where the factorization can be learned and, for instance, only has O(N K(S + A)) weights.

Similarly, our approach both 1) learns a low-dimensional factorization of the policy distribution and 2) defines exploration by also sampling from the low-dimensional latent space.

For instance, in the multi-agent setting, we can learn a centralized multi-agent policy with a latent structure that encodes coordination between agents and biases exploration towards policies that encode "good" coordination.

The key ideas of our approach are: 1) to utilize a shared stochastic latent variable model that defines the structured exploration policy, and 2) to employ a principled variational method to learn the posterior distribution over the latents jointly with the optimal policy.

Our approach has several desirable properties.

First we do not incorporate any form of prior domain knowledge, but rather discover the coordination structure purely from empirical experience during learning.

Second, our variational learning method enables fully differentiable end-to-end training of the entire policy class.

Finally, by utilizing a hierarchical policy class, our approach can easily scale to large action spaces (e.g. a large number of coordinating agents).

Our approach can also be seen as a deep hierarchical generalization of Thompson sampling, which is a historically popular way to capture correlations between actions (e.g. in the bandit setting BID2 ).To summarize, our contributions in this work are as follows:• We introduce a structured probabilistic policy class that uses a hierarchy of stochastic latent variables.• We propose an efficient and principled algorithm using variational methods to train the policy end-to-end.• To validate our learning framework, we introduce several synthetic multi-agent environments that explicitly require team coordination, and feature competitive pressures that are characteristic of many coordinated decision problems.• We empirically verify that our approach improves sample complexity on coordination games with a large number (N ∼ 20) of agents.• We show that learned latent structures correlate with meaningful coordination patterns.

We use multi-agent environments to show the efficacy of our approach to structured exploration, as they naturally exhibit exponentially large state-action spaces.

In this work we focus on efficiently learning a centralized policy: a joint policy model for all agents, in the full-information setting.

More generally, multi-agent problems can be generalized along many dimensions, e.g. one can learn decentralized policies in partial-information settings.

For an overview, see BID4 .In multi-agent RL, agents sequentially interact within an environment defined by the tuple: E ≡ (S, A, r, f P ).

Each agent i starts in an initial state s Figure 2 : Structured latent variable model of the multi-agent policy (actor, left) and instance of the multi-agent actor-critic interacting in the environment E (right).

The joint policy contains two stacked layers of stochastic latent variables (red), and deterministically receives states and computes actions (green).

Global variables λ are shared across agents.

On the right, a neural network instance of the actor-critic uses the reparametrization trick and receives the environment state, samples actions from the policy for all agents and computes value functions V• .where we sample M rollouts τ k by sampling actions from the policy that is being learned.

A central issue in reinforcement learning is the exploration-exploitation trade-off: how can agents sample rollouts and learn efficiently?

In particular, when the state-action space is exponentially large, discovering good (coordinated) policies when each agent samples independently becomes combinatorially intractable as N grows.

Hence, exploration in large state-action spaces poses a significant challenge.

We now formulate our multi-agent objective (1) using a hierarchical policy class that enables structured exploration.

Our approach, MACE ("Multi-Agent Coordinated Exploration"), builds upon two complementary approaches:• Encode structured exploration by sampling actions that are correlated between agents.

The correlation between actions encodes coordination.• Use a variational approach to derive and optimize a lower bound on the objective (1).Hierarchical Latent Model.

To encode coordination between agents, we assume the individual policies have shared structure, encoded by a latent variable λ t ∈ R n for all t, where n is the dimension of the latent space.

This leads to a hierarchical policy model P (a t , λ t |s t ), as shown in Figure 2 .

We first write the joint policy for a single time-step as: DISPLAYFORM0 where we introduced the conditional priors P (λ t |s t ).

The latent variables λ t introduce dependencies among the a t , hence this policy is more flexible compared to standard fully factorized policies BID23 ).

Note that this approach supports centralized learning and decentralized execution, by sharing a random seed amongst agents to sample λ t and actions a t during execution.

Computing the integral in the optimal policy (3) is hard, because the unknown distribution P (a i t |λ t , s t ) can be highly complex.

Hence, to make learning (3) tractable, we will use a variational approach.

Hierarchical Variational Lower Bound.

We next derive a tractable learning algorithm using variational methods.

Instead of directly optimizing (1), we cast it as a probabilistic inference problem, as in BID17 ; Vlassis et al. (2009) , and instead optimize a lower bound.

To do so, we assume that the total reward R i for each i to be non-negative and bounded.

Hence, we can view the total reward R(τ ) as a random variable, whose unnormalized distribution is defined as P (R|τ ) = R. We can then rewrite (1) as a maximum likelihood problem: DISPLAYFORM1 Hence, the RL objective is equivalent to a maximal likelihood problem: DISPLAYFORM2 In MACE we introduce a latent variable λ t in the probability of a rollout τ , using (3): DISPLAYFORM3 where computing the policy distribution P (a t , λ t |s t ; θ) is intractable, which makes the maximization in Equation FORMULA2 hard.

Hence, we derive a lower bound on the log-likelihood log P (R|τ )P (τ ; θ) in Equation FORMULA2 , using a variational approach.

Specifically, we use an approximate factorized variational distribution Q R that is weighted by the total reward R: DISPLAYFORM4 where φ are the parameters for the variational distribution Q R .

Using Jensen's inequality BID12 and (3) to factorize P (a t , λ t |s t ; θ), we can derive: DISPLAYFORM5 where the right-hand side is called the evidence lower bound (ELBO), which we can maximize as a proxy for (4).

For more details on the derivation, see the Appendix.

The standard choice for the prior P (λ t |s t ) is to use maximum-entropy standard-normal priors: P (λ t |s t ) = N (0, 1).

We can then optimize (8) using e.g. stochastic gradient ascent.

Formally, the MACE policy gradient is: DISPLAYFORM6 During a rollout τ k , we sample λ ∼ Q, observe rewards R ∼ P (R|τ ) and transitions s t+1 ∼ P (s t+1 |.), and use these to compute (10).

We can similarly compute g φ,Q = ∇ φ ELBO(Q R , θ, φ), the gradient for the variational posterior Q R .Actor-Critic and Bias-Variance.

Estimating policy gradients g θ using empirical rewards can suffer from high variance and instabilities.

It is thus useful to consider more general objectives F i : DISPLAYFORM7 such that the variance inĝ is reduced.

1 In practice, we find that using (10) with more general F , such as generalized advantages BID24 ), performs quite well.

To validate our approach, we created two grid-world games, depicted in Figure 2 , inspired by the classic Predator-Prey and Stag-Hunt games BID25 ).

In both games, the world is periodic and the initial positions of the hunters and prey are randomized.

Also, we consider two instances for both games: either the prey are moving or fixed.

Hare-Hunters.

Predator-Prey is a classic test environment for multi-agent learning, where 4 predators try to capture a prey by boxing it in.

We consider a variation defined by the settings (N, M, H, T ): N hunters and M prey.

Each prey can be captured by exactly H hunters: to capture the prey, a hunter gets next to it, after which the hunter is frozen.

Once a prey has had H hunters next to it, it is frozen and cannot be captured by another hunter.

The terminal rewards used are: DISPLAYFORM0 , if all prey are captured H times before the time limit T 0, otherwiseThe challenge of the game is for the agents to inactivate all prey within a finite time T .

Due to the time limit, the optimal strategy is for the agents to distribute targets efficiently, which can be challenging due to the combinatorially large number of possible hunter-to-prey assignments.

Stag-Hunters.

The Stag-Hunt is another classic multi-agent game designed to study coordination.

In this game, hunters have a choice: either they capture a hare for low reward, or, together with another hunter, capture a stag for a high reward.

We extend this to the multi-agent (N, M, H, T )-setting: N hunters hunt M prey (M/2 stags and M/2 hares).

Each stag has H hit-points, while hares and hunters have 1 hit-point.

Capturing is as in Hare-Hunters.

The spatial domain is similar to the Hare-Hunters game and we also use a time limit T .

The terminal reward is now defined as: DISPLAYFORM1 , if i captured a live stag that became inactive before the time limit T 0.1, if i captured a live hare before the time limit T 0, otherwiseThe challenge for the agents here is to discover that choosing to capture the same prey can yield substantially higher reward, but this requires coordinating with another hunter.

For experiments, we instantiated our multi-agent policy class (as in Figure 2 ) with deep neural networks.

For simplicity, we only used reactive policies without memory, although it is straightforward to apply MACE to policies with memory (e.g. LSTMs).

The model takes a joint state s t as input and computes features φ(s) using a 2-layer convolutional neural network.

To compute the latent variable λ ∈ R d , we use the reparametrization trick BID16 to learn the variational distribution (e.g. Q(λ|s)), sampling λ via ∼ N (0, 1) and distribution parameters µ, σ (omitting t): DISPLAYFORM0 Given λ, the model then computes the policies P (a i |λ, s) and value functions V i (s) as (omitting t): DISPLAYFORM1 where softmax(x) = exp x/ j exp x j .

In this way, the model can be trained end-to-end.

Training.

We used A3C BID20 ) with KL-controlled policy gradients (10), generalized advantage as F BID24 ).

and policy-entropy regularization.

For all experiments, we performed a hyper-parameter search and report the best 5 runs seen (see the Appendix for details).Baselines.

We compared MACE against two natural baselines:• Shared (shared actor-critic): agents share a deterministic hidden layer, but maintain individual weights θ i for their (stochastic) policy P (a|λ, s; θ i ) and value function V i (s; θ).

The key difference is that this model does not sample from the shared hidden layer.• Cloned (actor-critic): a model where each agent uses an identical policy and value function with shared weights.

There is shared information between the agents, and actions are sampled according to the agents' own policies.

Above, we defined a variational approach to train hierarchical multi-agent policies using structured exploration.

We now validate the efficacy of our approach by showing our method scales to environments with a large number of agents.

We ran experiments for both Hare-Hunters and Stag-Hunters for N = M = 10, 20 in a spatial domain of 50 × 50 grid cells.

Sample complexity.

In Table 1 we show the achieved rewards after a fixed number of training samples, and Figure 3 showcases the corresponding learning curves.

We see that MACE achieves up to 10× reward compared to the baselines.

Figure 4 shows the corresponding distribution of training episode lengths.

We see that MACE solves game instances more than 20% faster than baselines in 50% (10%) of Hare-Hunters (Stag-Hunters) episodes.

In particular, MACE learns to coordinate for higher reward more often: it achieves the highest average reward per-episode (e.g. for 10-10 Stag-Hunters with frozen prey, average rewards are 4.64 (Cloned), 6.22 (Shared), 6.61 (MACE)).

Hence, MACE coordinates successfully more often to capture the stags.

Together, these results show MACE enables more efficient learning.

Using the ELBO.

A salient difference between (10) and (2) is the KL-regularization, which stems from the derivation of the ELBO.

Since we use a more general objective F , c.f. (11), we also investigated the impact of using the KL-regularized policy gradient (10) versus the standard (2).

To Arrows show where agents move to in the next frame.

Top: at the start, predators explore via λ, but do not succeed before the time limit T (red dot).

Bottom: after convergence agents succeed consistently (green dot) before the time limit (purple dot) and λ encodes the two strategies from FIG0 .

Highlighted λ-components correlate with rollout under a 2-sided t-test at α = 0.1 significance.this end, we ran several instances of the above experiments both with and without KL-regularization.

We found that without KL-regularization, training is unstable and prone to mode collapse: the variance σ of the variational distribution can go to 0.

This reflects in essentially 0 achieved reward: the model does not solve the game for any reasonable hyperparameter settings.

Impact of dynamics and T .

Inspecting training performance, we see the relative difficulty of capturing moving or randomly moving prey.

Capturing moving prey is easier to learn than capturing fixed preys, as comparing rewards in Table 1 shows.

This shows a feature of the game dynamics: the expected distance between a hunter and an uncaptured prey are lower when the preys are randomly moving, resulting in an easier game.

Comparing Hare-Hunters and Stag-Hunters, we also see the impact of the time limit T .

Since we use terminal rewards only, as T gets larger, the reward becomes very sparse and models need more samples to discover good policies.

Beyond training benefits, we now demonstrate empirical evidence that suggest efficacy and meaningfulness of our approach to structured exploration.

We start by inspecting the behavior of the latent variable λ for a simple N = M = 2 Hare-Hunters game, which enables semantic inspection of the learned policies, as in FIG2 .

We make a number of observations.

First, λ is relevant: many components are statistically significantly correlated with the agents' actions.

This suggests the model does indeed use the latent λ: it (partly) controls the coordination between agents.

2 Second, the latent λ shows strong correlation during all phases of training.

This suggests that the model indeed is performing a form of structured exploration.

Third, the components of λ are correlated with semantic meaningful behavior.

We show a salient example in the bottom 2 rows in FIG2 : the correlated components of λ are disjoint and each component correlates with both agents.

The executed policies are exactly the two equivalent ways to assign 2 hunters to 2 preys, as illustrated in FIG0 .Coordination with a large N .

In the large N = M = 10 case, the dynamics of the agent collective are a generalization of the N = M = 2 case.

There are now redundancies in multi-agent hunter-prey assignments that are analogous to the N = M = 2 case that are prohibitively complex to analyze due to combinatorial complexity.

However our experiments strongly suggest (see e.g. FIG3 ) the latent code is again correlated with the agents' behavior during all phases of training, showing that λ induces meaningful multi-agent coordination.

Deep Structured Inference.

Recent works have focused on learning structured representations using expressive distributions, which enable more powerful probabilistic inference.

For instance, BID14 has proposed combining neural networks with graphical models, while BID23 learn hierarchical latent distributions.

Our work builds upon these approaches to learn structured policies in the reinforcement learning setting.

In the multi-agent setting, the RL problem has also been considered as an inference problem in e.g. BID18 Wu et al., 2013; BID19 .Variational methods in RL.

Neumann (2011); Furmston & Barber (2010) discuss variational approaches for RL problems, but did not consider end-to-end trainable models.

BID17 used variational methods for guided policy search.

BID13 learned exploration policies via information gain using variational methods.

However, these only consider 1 agent.

Multi-agent coordination has been studied in the RL community (e.g. BID11 ; BID15 ; BID5 ), for instance, as a method to reduce the instability of multiple agents learning simultaneously using RL.

The benefit of coordination was already demonstrated in simple multi-agent settings in e.g. BID27 .

The shared latent variable λ of our structured policy can also be interpreted as a learned correlation device (see BID3 for an example in the decentralized setting), which can be used to e.g. break ties between alternatives or induce coordination between agents.

More generally, they can be used to achieve correlated equilibria BID10 , a more general solution concept than Nash equilibria.

However, previous methods learned hand-crafted models and do not scale well to complex state spaces and many agents.

In contrast, our method learns coordination end-to-end via on-policy methods, learns the multi-agent exploration policy and scales well to many agents via its simple hierarchical structure.

In a sense, we studied the simplest setting that can benefit from structured exploration, in order to isolate the contribution of our work.

Our hierarchical model and variational approach are a simple way to implement multi-agent coordination, and easily combine with existing actor-critic methods.

Moving forward, there are many ways to expand on our work.

Firstly, for complex (partial-information) environments, instead of using reactive policies with simple priors P ∼ N (0, 1), memoryfull policies with flexible priors ) may be needed.

Secondly, our approach is complementary to richer forms of communication between agents.

Our hierarchical structure can be interpreted as a broadcast channel, where agents are passive receivers of the message λ.

Richer communication protocols could be encoded by policies with more complex inter-agent structure.

It would be interesting to investigate how to learn these richer structures.

We show details on how to derive a tractable learning method to the multi-agent reinforcement learning problem with a centralized controller: DISPLAYFORM0 Instead of directly optimizing (16), we cast it as a probabilistic inference problem, as in BID17 Vlassis et al. (2009) , and optimize a lower bound.

To do so, we assume that the total reward R i for each i to be non-negative and bounded.

Hence, we can view the total reward R(τ ) as a random variable, whose unnormalized distribution is defined as DISPLAYFORM1 We can then rewrite (16) as a maximum likelihood problem: DISPLAYFORM2 Hence, the RL objective is equivalent to a maximal likelihood problem: DISPLAYFORM3 where the probability of a rollout τ features a marginalization over the latent variables λ t : DISPLAYFORM4 DISPLAYFORM5 Here, we used the hierarchical decomposition for the policy: DISPLAYFORM6 DISPLAYFORM7 This policy distribution is intractable to learn exactly, as it involves margalization over λ t and an unknown flexible distribution P (a i t |λ t , s t ).

Hence the maximization in Equation FORMULA17 is hard.

Hence, we follow the variational approach and get a lower bound on the log-likelihood log P (R, τ ; θ) in Equation (19) .

For this, we use an approximate variational distribution Q R (λ 0:T |τ ; φ) and Jensen's inequality BID12 : DISPLAYFORM8 DISPLAYFORM9 where in the last line we used (20) .

By inspecting the quotient in (26), we see that the optimal Q R is a factorized distribution weighted by the total reward R: DISPLAYFORM10 P (s t+1 |s t , a t )Q(λ t |s t ; φ).We see that (26) simplifies to:dλ 0:T Q R (λ 0:T |τ ; φ) log P (R|τ )P (s 0 ) T t=0 P (s t+1 |s t , a t )P (a t , λ t |s t ; θ) P (R|τ )P (s 0 ) T t=0 P (s t+1 |s t , a t )Q(λ t |s t ; φ) (28) = dλ 0:T Q R (λ 0:T |τ ; φ) log T t=0 P (a t , λ t |s t ; θ) Q(λ t |s t , φ)= dλ 0:T Q R (λ 0:T |τ ; φ) T t=0 log P (a t , λ t |s t ; θ) Q(λ t |s t , φ)= dλ 0:T Q R (λ 0:T |τ ; φ) T t=0 log P (a t |λ t , s t ; θ)P (λ t |s t ) Q(λ t |s t , φ)= dλ 0:T Q R (λ 0:T |τ ; φ) DISPLAYFORM11 log P (a t |λ t , s t ; θ) + log P (λ t |s t ) Q(λ t |s t , φ)ELBO(Q R ,θ,φ).The right-hand side in Equation FORMULA30 is called the evidence lower bound (ELBO), which we can maximize as a proxy for (16).

The standard choice is to use maximum-entropy standard-normal priors: P (λ t |s t ) = N (0, 1).

We can then optimize (32) using e.g. stochastic gradient ascent.

Training method.

We used the A3C method with 5-20 threads for all our experiments.

Each thread performed SGD with (??).

The loss for the value function at each state s t is the standard L 2 -loss between the observed total rewards for each agent i and its value estimate: DISPLAYFORM0 In addition, in line with other work using actor-critic methods, we found that adding a small entropy regularization on the policy can sometimes positively influence performance, but this does not seem to be always required for our testbeds.

The entropy regularization is:H(P ) = −β t at P (a t |s t ; θ) log P (a t |s t ; θ).A3C additionally defines training minibatches in terms of a fixed number of environment steps L: a smaller L gives faster training with higher variance and a higher L vice versa.

<|TLDR|>

@highlight

Make deep reinforcement learning in large state-action spaces more efficient using structured exploration with deep hierarchical policies.

@highlight

A method to coordinate agent behaviour by using policies that have shared latent structure, a variational policy optimization method to optimize the coordinated policies, and a derivation of the authors' variational, hierarchical update.

@highlight

This paper suggests an algorithmic innovation consisting of hierarchical latent variables for coordinated exploration in multi-agent settings
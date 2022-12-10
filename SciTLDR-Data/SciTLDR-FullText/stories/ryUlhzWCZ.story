In this paper, we propose to combine imitation and reinforcement learning via the idea of reward shaping using an oracle.

We study the effectiveness of the near- optimal cost-to-go oracle on the planning horizon and demonstrate that the cost- to-go oracle shortens the learner’s planning horizon as function of its accuracy: a globally optimal oracle can shorten the planning horizon to one, leading to a one- step greedy Markov Decision Process which is much easier to optimize, while an oracle that is far away from the optimality requires planning over a longer horizon to achieve near-optimal performance.

Hence our new insight bridges the gap and interpolates between imitation learning and reinforcement learning.

Motivated by the above mentioned insights, we propose Truncated HORizon Policy Search (THOR), a method that focuses on searching for policies that maximize the total reshaped reward over a finite planning horizon when the oracle is sub-optimal.

We experimentally demonstrate that a gradient-based implementation of THOR can achieve superior performance compared to RL baselines and IL baselines even when the oracle is sub-optimal.

Reinforcement Learning (RL), equipped with modern deep learning techniques, has dramatically advanced the state-of-the-art in challenging sequential decision problems including high-dimensional robotics control tasks as well as video and board games BID13 BID23 .

However, these approaches typically require a large amount of training data and computational resources to succeed.

In response to these challenges, researchers have explored strategies for making RL more efficient by leveraging additional information to guide the learning process.

Imitation learning (IL) is one such approach.

In IL, the learner can reference expert demonstrations BID0 , or can access a cost-to-go oracle BID19 , providing additional information about the long-term effects of learner decisions.

Through these strategies, imitation learning lowers sample complexity by reducing random global exploration.

For example, BID25 shows that, with access to an optimal expert, imitation learning can exponentially lower sample complexity compared to pure RL approaches.

Experimentally, researchers also have demonstrated sample efficiency by leveraging expert demonstrations by adding demonstrations into a replay buffer BID28 BID14 , or mixing the policy gradient with a behavioral cloning-related gradient BID18 .Although imitating experts can speed up the learning process in RL tasks, the performance of the learned policies are generally limited to the performance of the expert, which is often sub-optimal in practice.

Previous imitation learning approaches with strong theoretical guarantees such as Data Aggregation (DAgger) BID20 and Aggregation with Values (AGGREVATE) BID19 can only guarantee a policy which performs as well as the expert policy or a one-step deviation improvement over the expert policy.1 Unfortunately, this implies that imitation learning with a sub-optimal expert will often return a sub-optimal policy.

Ideally, we want the best of both IL and RL: we want to use the expert to quickly learn a reasonable policy by imitation, while also exploring how to improve upon the expert with RL.

This would allow the learner to overcome the sample inefficiencies inherent in a pure RL strategy while also allowing the learner to eventually surpass a potentially sub-optimal expert.

Combining RL and IL is, in fact, not new.

BID5 attempted to combine IL and RL by stochastically interleaving incremental RL and IL updates.

By doing so, the learned policy will either perform as well as the expert policy-the property of IL BID19 , or eventually reach a local optimal policy-the property of policy iteration-based RL approaches.

Although, when the expert policy is sub-optimal, the learned locally optimal policy could potentially perform better than the expert policy, it is still difficult to precisely quantify how much the learner can improve over the expert.

In this work, we propose a novel way of combining IL and RL through the idea of Reward Shaping BID16 .

Throughout our paper we use cost instead of reward, and we refer to the concept of reward shaping with costs as cost shaping.

We assume access to a cost-to-go oracle that provides an estimate of expert cost-to-go during training.

The key idea is that the cost-to-go oracle can serve as a potential function for cost shaping.

For example, consider a task modeled by a Markov Decision Process (MDP).

Cost shaping with the cost-to-go oracle produces a new MDP with an optimal policy that is equivalent to the optimal policy of the original MDP BID16 .

The idea of cost shaping naturally suggests a strategy for IL: pick a favourite RL algorithm and run it on the new MDP reshaped using expert's cost-to-go oracle.

In fact, BID16 demonstrated that running SARSA BID26 on an MDP reshaped with a potential function that approximates the optimal policy's value-to-go, is an effective strategy.

We take this idea one step further and study the effectiveness of the cost shaping with the expert's cost-to-go oracle, with a focus on the setting where we only have an imperfect estimatorV e of the cost-to-go of some expert policy π e , i.e.,V e = V * , where V * is the optimal policy's cost-to-go in the original MDP.

We show that cost shaping with the cost-to-go oracle shortens the learner's planning horizon as a function of the accuracy of the oracleV e compared to V * .

Consider two extremes.

On one hand, when we reshape the cost of the original MDP with V * (i.e.,V e = V * ), the reshaped MDP has an effective planning horizon of one: a policy that minimizes the one-step cost of the reshaped MDP is in fact the optimal policy (hence the optimal policy of the original MDP).

On the other hand, when the cost-to-go oracle provides no information regarding V * , we have no choice but simply optimize the reshaped MDP (or just the original MDP) using RL over the entire planning horizon.

With the above insight, we propose the high-level strategy for combining IL and RL, which we name Truncated HORizon Policy Search with cost-to-go oracle (THOR).

The idea is to first shape the cost using the expert's cost-to-go oracleV e , and then truncate the planning horizon of the new MDP and search for a policy that optimizes over the truncated planning horizon.

For discrete MDPs, we mathematically formulate this strategy and guarantee that we will find a policy that performs better than the expert with a gap that can be exactly quantified (which is missing in the previous work of BID5 ).

In practice, we propose a gradient-based algorithm that is motivated from this insight.

The practical algorithm allows us to leverage complex function approximators to represent policies and can be applied to continuous state and action spaces.

We verify our approach on several MDPs with continuous state and action spaces and show that THOR can be much more sample efficient than strong RL baselines (we compared to Trust Region Policy Optimization with Generalized Advantage Estimation (TRPO-GAE) ), and can learn a significantly better policy than AGGREVATE (we compared to the policy gradient version of AGGREVATE from BID25 ) with access only to an imperfect cost-to-go oracle.

Previous work has shown that truncating the planning horizon can result in a tradeoff between accuracy and computational complexity.

BID8 proposed a model-based RL approach that focuses on a search for policies that maximize a sum of k-step rewards with a termination value that approximates the optimal value-to-go.

Their algorithm focuses on the model-based setting and the discrete state and action setting, as the algorithm needs to perform k-step value iteration to compute the policy.

Another use of the truncated planning horizon is to trade off bias and variance.

When the oracle is an approximation of the value function of the agent's current policy, by using k-step rollouts bottomed up by the oracle's return, truncating the planning horizon trades off bias and variance of the estimated reward-to-go.

The bias-variance tradeoff has been extensively studied in Temporal Difference Learning literature BID27 and policy iteration literature as well BID9 BID15 is perhaps the closest to our work.

In Theorem 5 in the Appendix of Ng's dissertation, Ng considers the setting where the potential function for reward shaping is close to the optimal value function and suggests that if one performs reward shaping with the potential function, then one can decrease the discount factor of the original MDP without losing the optimality that much.

Although in this work we consider truncating the planning steps directly, Theorem 5 in Ng's dissertation and our work both essentially considers trading off between the hardness of the reshaped MDP (the shorter the planning horizon, the easier the MDP to optimize) and optimality of the learned policy.

In addition to this tradeoff, our work suggests a path toward understanding previous imitation learning approaches through reward shaping, and tries to unify IL and RL by varying the planning horizon from 1 to infinity, based on how close the expert oracle is to the optimal value function.

Another contribution of our work is a lower bound analysis that shows that performance limitation of AGGREVATE with an imperfect oracle, which is missing in previous work BID19 .

The last contribution of our work is a model-free, actor-critic style algorithm that can be used for continuous state and action spaces.

We consider the problem of optimizing Markov Decision Process defined as M 0 = (S, A, P, C, γ).

Here, S is a set of S states and A is a set of A actions; P is the transition dynamics at such that for any s ∈ S, s ∈ S, a ∈ A, P (s |s, a) is the probability of transitioning to state s from state s by taking action a. For notation simplicity, in the rest of the paper, we will use short notation P sa to represent the distribution P (·|s, a).

The cost for a given pair of s and a is c(s, a), which is sampled from the cost distribution C(s, a) with mean valuec(s, a).

A stationary stochastic policy π(a|s) computes the probability of generating action a given state s.

The value function V π M0 and the state action cost-to-go Q π M0,h (s, a) of π on M 0 are defined as: DISPLAYFORM0 where the expectation is taken with respect to the randomness of M 0 and the stochastic policy π.

DISPLAYFORM1 The objective is to search for the optimal policy π * such that π * = arg min π V π (s), ∀s ∈ S.Throughout this work, we assume access to an cost-to-go oracleV e (s) : S → R. Note that we do not requireV e (s) to be equal to V * M0 .

For example,V e (s) could be obtained by learning from trajectories demonstrated by the expert π e (e.g., Temporal Difference Learning (Sutton & Barto, 1998)), orV e could be computed by near-optimal search algorithms via access to ground truth information BID7 BID5 BID24 or via access to a simulator using Dynamic Programming (DP) techniques BID6 BID17 .

In our experiment, we focus on the setting where we learn aV e (s) using TD methods from a set of expert demonstrations.

Given the original MDP M 0 and any potential functions Φ : S → R, we can reshape the cost c(s, a) sampled from C(s, a) to be: DISPLAYFORM0 Denote the new MDP M as the MDP obtained by replacing c by c in M 0 : M = (S, A, P, c , γ).

BID16 showed that the optimal policy π * M on M and the optimal policy π * M0 on the original MDP are the same: π * M (s) = π * M0 (s), ∀s.

In other words, if we can successfully find π * M on M, then we also find π * M0 , the optimal policy on the original MDP M 0 that we ultimately want to optimize.

In IL, when given a cost-to-go oracle V e , we can use it as a potential function for cost shaping.

.

As cost shaping does not change the optimal policy, we can rephrase the original policy search problem using the shaped cost: DISPLAYFORM0 for all s ∈ S. Though Eq. 2 provides an alternative objective for policy search, it could be as hard as the original problem as DISPLAYFORM1 , which can be easily verified using the definition of cost shaping and a telescoping sum trick.

As directly optimizing Eq 2 is as difficult as policy search in the original MDP, previous IL algorithms such as AGGREVATE essentially ignore temporal correlations between states and actions along the planning horizon and directly perform a policy iteration over the expert policy at every state, i.e., they are greedy with respect to A e asπ(s) = arg min a A e (s, a), ∀s ∈ S. The policy iteration theorem guarantees that such a greedy policyπ performs at least as well as the expert.

Hence, when the expert is optimal, the greedy policyπ is guaranteed to be optimal.

However when V e is not the optimal value function, the greedy policyπ over A e is a one-step deviation improvement over the expert but is not guaranteed to be close to the optimal π * .

We analyze in detail how poor the policy resulting from such a greedy policy improvement method could be when V e is far away from the optimal value function in Sec. 3.

In this section we study the dependency of effective planning horizon on the cost-to-go oracle.

We focus on the setting where we have access to an oracleV e (s) which approximates the cost-to-go of some expert policy π e (e.g., V e could be designed by domain knowledge BID16 or learned from a set of expert demonstrations).

We assume the oracle is close to V * M0 , but imperfect: |V e − V * M0 | = for some ∈ R + .

We first show that with such an imperfect oracle, previous IL algorithms AGGREVATE and AGGREVATE D BID19 BID25 are only guaranteed to learn a policy that is γ /(1−γ) away from the optimal.

Let us define the expected total cost for any policy π as J(π) = E s0∼v V π M0 (s 0 ) , measured under some initial state distribution v and the original MDP M 0 .Theorem 3.1.

There exists an MDP and an imperfect oracleV e (s) with |V e (s) − V * M0,h (s)| = , such that the performance of the induced policy from the cost-to-go oracleπ * = arg min a c(s, a) + γE s ∼Psa [V e (s )] is at least Ω(γ /(1 − γ)) away from the optimal policy π * : DISPLAYFORM0 The proof with the constructed example can be found in Appendix A. DenoteQ DISPLAYFORM1 , in high level, we construct an example whereQ e is close to Q * in terms of Q e − Q * ∞ , but the order of the actions induced byQ e is different from the order of the actions from Q * , hence forcing the induced policyπ * to make mistakes.

As AGGREVATE at best computes a policy that is one-step improvement over the oracle, i.e.,π * = arg min a c(s, a) + γE s ∼Psa [V e (s )] , it eventually has to suffer from the above lower bound.

This gap in fact is not surprising as AGGREVATE is a one-step greedy algorithm in a sense that it is only optimizing the one-step cost function c from the reshaped MDP M. To see this, note that the cost of the reshaped DISPLAYFORM2 , and we havê π * (s) = arg min a E[c (s, a)].

Hence AGGREVATE can be regarded as a special algorithm that aims to optimizing the one-step cost of MDP M that is reshaped from the original MDP M 0 using the cost-to-go oracle.

Though when the cost-to-go oracle is imperfect, AGGREVATE will suffer from the above lower bound due to being greedy, when the cost-to-go oracle is perfect, i.e.,V e = V * , being greedy on one-step cost makes perfect sense.

To see this, use the property of the cost shaping BID16 , we can verify that whenV e = V * : DISPLAYFORM3 Namely the optimal policy on the reshaped MDP M only optimizes the one-step cost, which indicates that the optimal cost-to-go oracle shortens the planning horizon to one: finding the optimal policy on M 0 becomes equivalent to optimizing the immediate cost function on M at every state s.

When the cost-to-go oracle is away from the optimality, we lose the one-step greedy property shown in Eq. 4.

In the next section, we show that how we can break the lower bound Ω( /(1 − γ)) only with access to an imperfect cost-to-go oracleV e , by being less greedy and looking head for more than one-step.

Given the reshaped MDP M withV e as the potential function, as we mentioned in Sec. 2.2, directly optimizing Eq. 2 is as difficult as the original policy search problem, we instead propose to minimize the total cost of a policy π over a finite k ≥ 1 steps at any state s ∈ S: DISPLAYFORM0 Using the definition of cost shaping and telescoping sum trick,we can re-write Eq. 5 in the following format, which we define as k-step disadvantage with respect to the cost-to-go oracle: DISPLAYFORM1 We assume that our policy class Π is rich enough that there always exists a policyπ * ∈ Π that can simultaneously minimizes the k−step disadvantage at every state (e.g., policies in tabular representation in discrete MDPs).

Note that when k = 1, minimizing Eq. 6 becomes the problem of finding a policy that minimizes the disadvantage A e M0 (s, a) with respect to the expert and reveals AGGREVATE.The following theorem shows that to outperform expert, we can optimize Eq. 6 with k > 1.

Let us denote the policy that minimizes Eq. 6 in every state asπ * , and the value function ofπ * as Vπ * .

Theorem 3.2.

Assumeπ * minimizes Eq. 6 for every state s ∈ S with k > 1 and |V e (s) − V * (s)| = Θ( ), ∀s.

We have : DISPLAYFORM2 Compare the above theorem to the lower bound shown in Theorem 3.1, we can see that when k > 1, we are able to learn a policy that performs better than the policy induced by the oracle (i.e.,π * (s) = arg min aQ e (s, a)) by at least ( DISPLAYFORM3 The proof can be found in Appendix B. Theorem 3.2 and Theorem 3.1 together summarize that when the expert is imperfect, simply computing a policy that minimizes the one-step disadvantage (i.e., (k = 1)) is not sufficient to guarantee near-optimal performance; however, optimizing a k-step disadvantage with k > 1 leads to a policy that guarantees to outperform the policy induced by the oracle (i.e., the best possible policy that can be learnt using AGGREVATE and AGGREVATED).

Also our theorem provides a concrete performance gap between the policy that optimizes Eq. 6 for k > 1 and the policy that induced by the oracle, which is missing in previous work (e.g., BID5 ).As we already showed, if we set k = 1, then optimizing Eq. 6 becomes optimizing the disadvantage over the expert A e M0 , which is exactly what AGGREVATE aims for.

When we set k = ∞, optimizing Eq. 6 or Eq. 5 just becomes optimizing the total cost of the original MDP.

Optimizing over a shorter Reset system.

Execute π θn to generate a set of trajectories {τ i } N i=1 .

Reshape cost c (s t , a t ) = c(s t , a t ) + V e t+1 (s t+1 ) − V e t (s t ), for every t ∈ [1, |τ i |] in every trajectory τ i , i ∈ [N ].

Compute gradient: DISPLAYFORM0 8:Update disadvantage estimator toÂ πn,k M using {τ i } i with reshaped cost c .

Update policy parameter to θ n+1 .

10: end for finite horizon is easier than optimizing over the entire infinite long horizon due to advantages such as smaller variance of the empirical estimation of the objective function, less temporal correlations between states and costs along a shorter trajectory.

Hence our main theorem essentially provides a tradeoff between the optimality of the solutionπ * and the difficulty of the underlying optimization problem.

Given the original MDP M 0 and the cost-to-go oracleV e , the reshaped MDP's cost function c is obtained from Eq. 1 using the cost-to-go oracle as a potential function.

Instead of directly applying RL algorithms on M 0 , we use the fact that the cost-to-go oracle shortens the effective planning horizon of M, and propose THOR: Truncated HORizon Policy Search summarized in Alg.

1.

The general idea of THOR is that instead of searching for policies that optimize the total cost over the entire infinitely long horizon, we focus on searching for polices that minimizes the total cost over a truncated horizon, i.e., a k−step time window.

Below we first show how we derive THOR from the insight we obtained in Sec. 3.Let us define a k-step truncated value function V π,k M and similar state action value function Q π,k M on the reshaped MDP M as: DISPLAYFORM0 At any time state s, V π,k M only considers (reshaped) cost signals c from a k-step time window.

We are interested in searching for a policy that can optimizes the total cost over a finite k-step horizon as shown in Eq. 5.

For MDPs with large or continuous state spaces, we cannot afford to enumerate all states s ∈ S to find a policy that minimizes the k−step disadvantage function as in Eq. 5.

Instead one can leverage the approximate policy iteration idea and minimize the weighted cost over state space using a state distribution ν BID11 BID2 : DISPLAYFORM1 For parameterized policy π (e.g., neural network policies), we can implement the minimization in Eq. 10 using gradient-based update procedures (e.g., Stochastic Gradient Descent, Natural Gradient BID10 BID1 ) in the policy's parameter space.

In the setting where the system cannot be reset to any state, a typical choice of exploration policy is the currently learned policy (possibly mixed with a random process BID12 to futher encourage exploration).

Denote π n as the currently learned policy after iteration n and P r πn (·) as the average state distribution induced by executing π n (parameterized by θ n ) on the MDP.

Replacing the exploration distribution by P r πn (·) in Eq. 10, and taking the derivative with respect to the policy parameter θ, the policy gradient is: DISPLAYFORM2 where τ k ∼ π n denotes a partial k−step trajectory τ k = {s 1 , a 1 , ..., s k , a k |s 1 = s} sampled from executing π n on the MDP from state s. Replacing the expectation by empirical samples from π n , replacing Q π,k M by a critic approximated by Generalized disadvantage Estimator (GAE)Â π,k M , we get back to the gradient used in Alg.

1: DISPLAYFORM3 where |τ | denotes the length of the trajectory τ .

If using the classic policy gradient formulation on the reshaped MDP M we should have the following expression, which is just a re-formulation of the classic policy gradient BID29 : DISPLAYFORM0 which is true since the cost c i (we denote c i (s, a) as c i for notation simplicity) at time step i is correlated with the actions at time step t = i all the way back to the beginning t = 1.

In other words, in the policy gradient format, the effectiveness of the cost c t is back-propagated through time all the way back the first step.

Our proposed gradient formulation in Alg.

1 shares a similar spirit of Truncated Back-Propagation Through Time BID30 , and can be regarded as a truncated version of the classic policy gradient formulation: at any time step t, the cost c is back-propagated through time at most k-steps: DISPLAYFORM1 In Eq. 13, for any time step t, we ignore the correlation between c t and the actions that are executed k-step before t, hence elimiates long temporal correlations between costs and old actions.

In fact, AGGREVATE D BID25 , a policy gradient version of AGGREVATE, sets k = 1 and can be regarded as No Back-Propagation Through Time.

The above gradient formulation provides a natural half-way point between IL and RL.

When k = 1 andV e = V * M0 (the optimal value function in the original MDP M 0 ): DISPLAYFORM0 where, for notation simplicity, we here use E τ to represent the expectation over trajectories sampled from executing policy π θ , and A π * M0 is the advantage function on the original MDP M 0 .

The fourth expression in the above equation is exactly the gradient proposed by AGGREVATED BID25 .

AGGREVATED performs gradient descent with gradient in the format of the fourth expression in Eq. 14 to discourage the log-likelihood of an action a t that has low advantage over π * at a given state s t .On the other hand, when we set k = ∞, i.e., no truncation on horizon, then we return back to the classic policy gradient on the MDP M obtained from cost shaping withV e .

As optimizing M is the same as optimizing the original MDP M 0 (Ng et al., 1999), our formulation is equivalent to a pure RL approach on M 0 .

In the extreme case when the oracleV e has nothing to do with the true optimal oracle V * , as there is no useful information we can distill from the oracle and RL becomes the only approach to solve M 0 .

We evaluated THOR on robotics simulators from OpenAI Gym BID4 .

Throughout this section, we report reward instead of cost, since OpenAI Gym by default uses reward.

The baseline we compare against is TRPO-GAE and AGGREVATED BID25 .To simulate oracles, we first train TRPO-GAE until convergence to obtain a policy as an expert π e .

We then collected a batch of trajectories by executing π e .

Finally, we use TD learning Sutton (1988) to train a value functionV e that approximates V e .

In all our experiments, we ignored π e and only used the pre-trainedV e for reward shaping.

Hence our experimental setting simulates the situation where we only have a batch of expert demonstrations available, and not the experts themselves.

This is a much harder setting than the interactive setting considered in previous work BID20 BID25 BID5 .

Note that π e is not guaranteed to be an optimal policy, and V e is only trained on the demonstrations from π e , therefore the oracleV e is just a coarse estimator of V * M0 .

Our goal is to show that, compared to AGGREVATED, THOR with k > 1 results in significantly better performance; compared to TRPO-GAE, THOR with some k << H converges faster and is more sample efficient.

For fair comparison to RL approaches, we do not pre-train policy or criticÂ using demonstration data, though initialization using demonstration data is suggested in theory and has been used in practice to boost the performance BID20 BID3 .For all methods we report statistics (mean and standard deviation) from 25 seeds that are i.i.d generated.

For trust region optimization on the actor π θ and GAE on the critic, we simply use the recommended parameters in the code-base from TRPO-GAE .

We did not tune any parameters except the truncation length k.

We consider two discrete action control tasks with sparse rewards: Mountain-Car, Acrobot and a modified sparse reward version of CartPole.

All simulations have sparse reward in the sense that no reward signal is given until the policy succeeds (e.g., Acrobot swings up).

In these settings, pure RL approaches that rely on random exploration strategies, suffer from the reward sparsity.

On the other Note that in our setting whereV e is imperfect, THOR with k > 1 works much better than AG-GREVATED (THOR with k = 1) in Acrobot.

In Mountain Car, we observe that AGGREVATED achieves good performance in terms of the mean, but THOR with k > 1 (especially k = 10) results in much higher mean+std, which means that once THOR receives the reward signal, it can leverage this signal to perform better than the oracles.

We also show that THOR with k > 1 (but much smaller than H) can perform better than TRPO-GAE.

In general, as k increases, we get better performance.

We make the acrobot setting even harder by setting H = 200 to even reduce the chance of a random policy to receive reward signals.

FIG1 to FIG1 , we can see that THOR with different settings of k always learns faster than TRPO-GAE, and THOR with k = 50 and k = 100 significantly outperform TRPO-GAE in both mean and mean+std.

This indicates that THOR can leverage both reward signals (to perform better than AGGREVATED) and the oracles (to learn faster or even outperform TRPO).

We tested our approach on simulators with continuous state and actions from MuJoCo simulators: a modified sparse reward Inverted Pendulum, a modifed sparse reward Inverted Double Pendulum, Hopper and Swimmer.

Note that, compared to the sparse reward setting, Hopper and Swimmer do not have reward sparsity and policy gradient methods have shown great results BID21 BID23 .

Also, due to the much larger and more complex state space and control space compared to the simulations we consider in the previous section, the value function estimatorV e is much less accurate in terms of estimating V * M0 since the trajectories demonstrated from experts may only cover a very small part of the state and control space.

FIG2 shows the results of our approach.

For all simulations, we require k to be around 20% ∼ 30% of the original planning horizon H to achieve good performance.

AGGREVATED (k = 1) learned very little due to the imperfect value function estimatorV e .

We also tested k = H, where we observe that reward shaping withV e gives better performance than TRPO-GAE.

This empirical observation is consistent with the observation from BID16 BID16 ) used SARSA (Sutton, 1988 , not policy gradient based methods).

This indicates that even whenV e is not close to V * , policy gradient methods can still employ the oracleV e just via reward shaping.

Finally, we also observed that our approach significantly reduces the variance of the performance of the learned polices (e.g., Swimmer in FIG2 ) in all experiments, including the sparse reward setting.

This is because truncation can significantly reduce the variance from the policy gradient estimation when k is small compared to H.

We propose a novel way of combining IL and RL through the idea of cost shaping with an expert oracle.

Our theory indicates that cost shaping with the oracle shortens the learner's planning horizon as a function of the accuracy of the oracle compared to the optimal policy's value function.

Specifically, when the oracle is the optimal value function, we show that by setting k = 1 reveals previous imitation learning algorithm AGGREVATED.

On the other hand, we show that when the oracle is imperfect, using planning horizon k > 1 can learn a policy that outperforms a policy that would been learned by AGGREVATE and AGGREVATED (i.e., k = 1).

With this insight, we propose THOR (Truncated HORizon policy search), a gradient based policy search algorithm that explicitly focusing on minimizing the total cost over a finite planning horizon.

Our formulation provides a natural half-way point between IL and RL, and experimentally we demonstrate that with a reasonably accurate oracle, our approach can outperform RL and IL baselines.

We believe our high-level idea of shaping the cost with the oracle and then focusing on optimizing a shorter planning horizon is not limited to the practical algorithm we proposed in this work.

In fact our idea can be combined with other RL techniques such as Deep Deterministic Policy Gradient (DDPG) BID12 , which has an extra potential advantage of storing extra information from the expert such as the offline demonstrations in its replay buffer BID28 ).

Though in our experiments, we simply used some expert's demonstrations to pre-trainV e using TD learning, there are other possible ways to learn a more accurateV e .

For instance, if an expert is available during training BID20 , one can online updateV e by querying expert's feedback.

A PROOF OF THEOREM 3.1 Figure 3 : The special MDP we constructed for theorem 3.1Proof.

We prove the theorem by constructing a special MDP shown in Fig 3, where H = ∞. The MDP has deterministic transition, 2H + 2 states, and each state has two actions a 1 and a 2 as shown in Fig. 3 .

Every episode starts at state s 0 .

For state s i (states on the top line), we have c(s i ) = 0 and for state s i (states at the bottom line) we have c(s i ) = 1.It is clear that for any state s i , we have Q * (s i , a 1 ) = 0, Q * (s i , a 2 ) = γ, Q * (s i , a 1 ) = 1 and Q * (s i , a 2 ) = 1 + γ, for i ≥ 1.

Let us assume that we have an oracleV e such thatV e (s i ) = 0.5 + δ and V e (s i ) = 0.5 − δ, for some positive real number δ.

Hence we can see that |V e (s) − V * (s)| = 0.5 + δ, for all s. DenoteQ e (s, a) = c(s, a) + γE s ∼Psa [V e (s )], we know thatQ e (s i , a 1 ) = γ(0.5 + δ),Q e (s i , a 2 ) = γ(0.5 − δ),Q e (s i , a 1 ) = 1 + γ(0.5 + δ) andQ e (s i , a 2 ) = 1 + γ(0.5 − δ).It is clear that the optimal policy π * has cost J(π * ) = 0.

Now let us compute the cost of the induced policy from oracleQ e :π(s) = arg min aQ e (s, a).

As we can seeπ makes a mistake at every state as arg min aQ e (s, a) = arg min a Q * (s, a).

Hence we have J(π) = γ 1−γ .

Recall that in our constructed example, we have = 0.5 + δ.

Now let δ → 0 + (by δ → 0 + we mean δ approaches to zero from the right side), we have → 0.5, hence J(π) = Proof of Theorem 3.2.

In this proof, for notation simplicity, we denote V π M0 as V π for any π.

Using the definition of value function V π , for any state s 1 ∈ S we have: DISPLAYFORM0

@highlight

Combining Imitation Learning and Reinforcement Learning to learn to outperform the expert
Multi-step greedy policies have been extensively used in model-based Reinforcement Learning (RL) and in the case when a model of the environment is available (e.g., in the game of Go).

In this work, we explore the benefits of multi-step greedy policies in model-free RL when employed in the framework of multi-step Dynamic Programming (DP): multi-step Policy and Value Iteration.

These algorithms iteratively solve short-horizon decision problems and converge to the optimal solution of the original one.

By using model-free algorithms as solvers of the short-horizon problems we derive fully model-free algorithms which are instances of the multi-step DP framework.

As model-free algorithms are prone to instabilities w.r.t.

the decision problem horizon, this simple approach can help in mitigating these instabilities and results in an improved model-free algorithms.

We test this approach and show results on both discrete and continuous control problems.

The field of Reinforcement learning (RL) span a wide variety of algorithms for solving decisionmaking problems through repeated interaction with the environment.

By incorporating deep neural networks into RL algorithms, the field of RL has recently witnessed remarkable empirical success (e.g., Mnih et al. 2015; Lillicrap et al. 2015; Silver et al. 2017 ).

Much of this success had been achieved by model-free RL algorithms, such as Q-learning and policy gradient.

These algorithms are known to suffer from high variance in their estimations (Greensmith et al., 2004) and to have difficulties handling function approximation (e.g., Thrun & Schwartz 1993; Baird 1995; Van Hasselt et al. 2016; Lu et al. 2018 ).

These problems are intensified in decision problems with long horizon, i.e., when the discount factor, γ, is large.

Although using smaller values of γ addresses the γ-dependent issues and leads to more stable algorithms (Petrik & Scherrer, 2009; Jiang et al., 2015) , it comes with a cost, as the algorithm may return a biased solution, i.e., it may not converge to an optimal solution of the original decision problem (the one with large value of γ).

Efroni et al. (2018a) recently proposed another approach to mitigate the γ-dependant instabilities in RL in which they study a multi-step greedy versions of the well-known dynamic programming (DP) algorithms policy iteration (PI) and value iteration (VI) (Bertsekas & Tsitsiklis, 1996) .

Efroni et al. (2018a) also proposed an alternative formulation of the multi-step greedy policy, called κ-greedy policy, and studied the convergence of the resulted PI and VI algorithms: κ-PI and κ-VI.

These two algorithms iteratively solve γκ-discounted decision problems, whose reward has been shaped by the solution of the decision problem at the previous iteration.

Unlike the biased solution obtained by solving the decision problem with a smaller value of γ, by iteratively solving decision problems with a smaller γκ horizon, the κ-PI and κ-VI algorithms could converge to an optimal policy of the original decision problem.

In this work, we derive and empirically validate model-free deep RL (DRL) implementations of κ-PI and κ-VI.

In these implementations, we use DQN (Mnih et al., 2015) and TRPO (Schulman et al., 2015) for (approximately) solving γκ-discounted decision problems (with shaped reward), which is the main component of the κ-PI and κ-VI algorithms.

The experiments illustrate the performance of model-free algorithms can be improved by using them as solvers of multi-step greedy PI and VI schemes, as well as emphasize important implementation details while doing so.

In this paper, we assume that the agent's interaction with the environment is modeled as a discrete time γ-discounted Markov Decision Process (MDP), defined by M γ = (S, A, P, R, γ, µ), where S and A are the state and action spaces; P ≡ P (s |s, a) is the transition kernel; R ≡ r(s, a)

is the reward function with the maximum value of R max ; γ ∈ (0, 1) is the discount factor; and µ is the initial state distribution.

Let π : S → P(A) be a stationary Markovian policy, where P(A) is a probability distribution on the set A. The value of π in any state s ∈ S is defined as V π (s) ≡ E[ t≥0 γ t r(s t , π(s t ))|s 0 = s, π], where the expectation is over all the randomness in policy, dynamics, and rewards.

Similarly, the action-value function of π is defined as Q π (s, a) = E[ t≥0 γ t r(s t , π(s t ))|s 0 = s, a 0 = a, π].

Since the rewards have the maximum value of R max , both V and Q functions have the maximum value of V max = R max /(1 − γ).

An optimal policy π * is the policy with maximum value at every state.

We call the value of π * the optimal value, and define it as V * (s) = max π E[ t≥0 γ t r(s t , π(s t ))|s 0 = s, π], ∀s ∈ S. Furthermore, we denote the stateaction value of π * as Q * (s, a) and remind the following relation holds V * (s) = max a Q * (s, a) for all s.

The algorithms by which an is be solved (obtain an optimal policy) are mainly based on two popular DP algorithms: Policy Iteration (PI) and Value Iteration (VI).

While VI relies on iteratively computing the optimal Bellman operator T applied to the current value function V (Eq. 1), PI relies on (iteratively) calculating a 1-step greedy policy π 1-step w.r.t.

to the value function of the current policy V (Eq. 2):

It is known that T is a γ-contraction w.r.t.

the max norm and its unique fixed point is V * , and the 1-step greedy policy w.r.t.

V * is an optimal policy π * .

In practice, the state space is often large, and thus, we can only approximately compute Eqs. 1 and 2, which results in approximate PI (API) and VI (AVI) algorithms.

These approximation errors then propagate through the iterations of the API and AVI algorithms.

However, it has been shown that this (propagated) error can be controlled (Munos, 2003; 2005; Farahmand et al., 2010) and after N steps, the algorithms approximately converge to a solution π N whose difference with the optimal value is bounded (see e.g., Scherrer 2014 for API):

is the expected value function at the initial state, 1 δ represents the per-iteration error, and C upper-bounds the mismatch between the sampling distribution and the distribution according to which the final value function is evaluated (µ in Eq. 3), and depends heavily on the dynamics.

Finally, the second term on the RHS of Eq. 3 is the error due to initial values of policy/value, and decays with the number of iterations N .

The optimal Bellman operator T (Eq. 1) and 1-step greedy policy π 1-step (Eq. 2) can be generalized to multi-step.

The most straightforward form of this generalization is by replacing T and π 1-step with h-optimal Bellman operator and h-step greedy policy (i.e., a lookahead of horizon h) that are defined by substituting the 1-step return in Eqs. 1 and 2, r(s 0 , a) + γV (s 1 ), with h-step return, h−1 t=0 r(s t , a t ) + γ h V (s h ), and computing the maximum over actions a 0 , . . .

, a h−1 , instead of just a 0 (Bertsekas & Tsitsiklis, 1996) .

Efroni et al. (2018a) proposed an alternative form of multi-step optimal Bellman operator and multi-step greedy policy, called κ-optimal Bellman operator, T κ , and κ-greedy policy, π κ , for κ ∈ [0, 1], i.e.,

1 Note that the LHS of Eq. 3 is the 1-norm of (V

where the shaped reward r t (κ, V ) w.r.t.

the value function V is defined as

It can be shown that the κ-greedy policy w.r.t.

the value function V is the optimal policy w.r.t.

a κ-weighted geometric average of all future h-step returns (from h = 0 to ∞).

This can be interpreted as TD(λ) (Sutton & Barto, 2018) for policy improvement (see Efroni et al., 2018a, Sec. 6) .

The important difference is that TD(λ) is used for policy evaluation and not for policy improvement.

From Eqs. 4 and 5, it is easy to see that solving these equations is equivalent to solving a surrogate γκ-discounted MDP with the shaped reward r t (κ, V ), which we denote by M γκ (V ) throughout the paper.

The optimal value of M γκ (V ) (the surrogate MDP) is T κ V and its optimal policy is the κ-greedy policy, π κ .

Using the notions of κ-optimal Bellman operator, T κ , and κ-greedy policy, π κ , Efroni et al. (2018a) derived κ-PI and κ-VI algorithms, whose pseudocode is shown in Algorithms 1 and 2.

κ-PI iteratively (i) evaluates the value of the current policy π i , and (ii) set the new policy, π i+1 , to the κ-greedy policy w.r.t.

the value of the current policy V πi , by solving Eq. 5.

On the other hand, κ-VI repeatedly applies the T κ operator to the current value function V i (solves Eq. 4) to obtain the next value function, V i+1 , and returns the κ-greedy policy w.r.t.

the final value V N (κ) .

Note that for κ = 0, the κ-greedy policy and κ-optimal Bellman operator are equivalent to their 1-step counterparts, defined by Eqs. 1 and 2, which indicates that κ-PI and κ-VI are generalizations of the seminal PI and VI algorithms.

It has been shown that both PI and VI converge to the optimal value with an exponential rate that depends on the discount factor γ, i.e.,

g., Bertsekas & Tsitsiklis, 1996; Scherrer, 2013) .

Analogously, Efroni et al. (2018a) showed that κ-PI and κ-VI converge with faster exponential rate of ξ(κ) =

, with the cost that each iteration of these algorithms is computationally more expensive than that of PI and VI.

Finally, we state the following two properties of κ-PI and κ-greedy policies that we use in our RL implementations of κ-PI and κ-VI algorithms in Sections 4 and 5: 1) Asymptotic performance depends on κ.

The following bound that is similar to the one reported in Eq. 3 was proved by Efroni et al. (2018b, Thm. 5) for the performance of κ-PI:

where δ(κ) and C(κ) are quantities similar to δ and C in Eq. 3.

Note that the first term on the RHS of Eq. 7 is independent of N (κ), while the second one decays with N (κ).

2) Soft updates w.r.t.

a κ-greedy policy does not necessarily improve the performance.

Let π κ be the κ-greedy policy w.r.t.

V π .

Then, unlike for 1-step greedy policies, the performance of (1−α)π+απ κ (soft update) is not necessarily better than that of π (Efroni et al., 2018b, Thm. 1) .

This hints that it would be advantages to use κ-greedy policies with 'hard' updates (using π κ as the new policy).

4 RL IMPLEMENTATIONS OF κ-PI AND κ-VI As described in Sec. 3, implementing κ-PI and κ-VI requires iteratively solving a γκ-discounted surrogate MDP with a shaped reward.

If a model of the environment is given, the surrogate MDP can be solved using a DP algorithm (see Efroni et al., 2018a, Sec. 7) .

When the model is not available, it can be approximately solved by any model-free RL algorithm.

In this paper, we focus on the case that the model is not available and propose RL implementations of κ-PI and κ-VI.

The main question we investigate in this work is how model-free RL algorithms should be implemented to efficiently solve the surrogate MDP in κ-PI and κ-VI.

In this paper, we use DQN (Mnih et al., 2015) and TRPO (Schulman et al., 2015) as subroutines for estimating a κ-greedy policy (Line 4 in κ-PI, Alg.

1 and Line 5 in κ-VI, Alg.

2) or for estimating an optimal value of the surrogate MDP (Line 3 in κ-VI, Alg.

2).

For estimating the value of the current policy (Line 3, in κ-PI, Alg.

1), we use standard policy evaluation deep RL (DRL) algorithms.

To implement κ-PI and κ-VI, we shall set the value of N (κ) ∈ N, i.e., the total number of iterations of these algorithms, and determine the number of samples for each iteration.

Since N (κ) only appears in the second term of Eq. 7, an appropriate choice of

Note that setting N (κ) to a higher value would not dramatically improve the

1: Initialize replay buffer D, Q-networks Q θ , Q φ , and target networks Q θ , Q φ ; 2: for i = 0, . . .

, N (κ) − 1 do 3: # Policy Improvement 4:

Act by an -greedy policy w.r.t.

Q θ (st, a), observe rt, st+1, and store (st, at, rt, st+1) in D; 6:

Sample a batch {(sj, aj, rj, sj+1)} N j=1 from D; 7:

Update θ by DQN rule with {(sj, aj, rj(κ, V φ ), sj+1)} N j=1 , where 8:

Copy θ to θ occasionally (θ ← θ); 10:

end for 11:

# Policy Evaluation of πi(s) ∈ arg maxa Q θ (s, a) 12:

Update φ by TD(0) off-policy rule with {(sj, aj, rj, sj+1)} N j=1 , and πi(s) ∈ arg maxa Q θ (s, a); 15:

Copy φ to φ occasionally (φ ← φ); 16: end for 17: end for performance, because the asymptotic term in Eq. 7 is independent of N (κ).

In practice, since δ(κ) and C(κ) are unknown, we set N (κ) to satisfy the following equality:

where C F A is a hyper-parameter that depends on the final-accuracy we are aiming for.

For example, if we expect the final accuracy being 90%, we would set C F A = 0.1.

Our results suggest that this approach leads to a reasonable choice for N (κ), e.g., N (κ = 0.99) 4 and N (κ = 0.5) 115, for C F A = 0.1 and γ = 0.99.

As we increase κ, we expect less iterations are needed for κ-PI and κ-VI to converge to a good policy.

Another important observation is that since the discount factor of the surrogate MDP that κ-PI and κ-VI solve at each iteration is γκ, the effective horizon (the effective horizon of a γκ-discounted MDP is 1/(1 − γκ)) of the surrogate MDP increases with κ.

Lastly, we need to determine the number of samples for each iteration of κ-PI and κ-VI.

We allocate equal number of samples per iteration, denoted by T (κ).

Since the total number of samples, T , is known beforehand, we set the number of samples per iteration to

5 DQN AND TRPO IMPLEMENTATIONS OF κ-PI AND κ-VI

In this section, we study the use of DQN (Mnih et al., 2015) and TRPO (Schulman et al., 2015) in κ-PI and κ-VI algorithms.

We first derive our DQN and TRPO implementations of κ-PI and κ-VI in Sections 5.1 and 5.2.

We refer to the resulting algorithms as κ-PI-DQN, κ-VI-DQN, κ-PI-TRPO, and κ-VI-TRPO.

It is important to note that for κ = 1, κ-PI-DQN and κ-VI-DQN are reduced to DQN, and κ-PI-TRPO and κ-VI-TRPO are reduced to TRPO.

We then conduct a set of experiments with these algorithms, in Sections 5.1.1 and 5.2.1, in which we carefully study the effect of κ and N (κ) (or equivalently the hyper-parameter C F A , defined by Eq. 8) on their performance.

In these experiments, we specifically focus on answering the following questions:

1.

Is the performance of DQN and TRPO improve when using them as κ-greedy solvers in κ-PI and κ-VI?

Is there a performance tradeoff w.r.t.

to κ?

2.

Following κ-PI and κ-VI, our DQN and TRPO implementations of these algorithms devote a significant number of sample T (κ) to each iteration.

Is this needed or a 'naive' choice of T (κ) = 1, or equivalently N (κ) = T , works just well, for all values of κ?

Algorithm 3 contains the pseudo-code of κ-PI-DQN.

Due to space constraints, we report its detailed pseudo-code in Appendix A.1 (Alg.

5).

In the policy improvement stage of κ-PI-DQN, we use DQN to solve the γκ-discounted surrogate MDP with the shaped reward r t (κ, V φ V πi−1 ), i.e., at the end of this stage M γκ (V φ ).

The output of the DQN is approximately the optimal Qfunction of M γκ (V φ ), and thus, the κ-greedy policy w.r.t.

V φ is equal to arg max a Q θ (·, a).

At the policy evaluation stage, we use off-policy TD(0) to evaluate the Q-function of the current policy

Although what is needed on Line 8 is an estimate of the value function of the current policy, V φ V πi−1 , we chose to evaluate the Q-function of π i : the data in our disposal (the transitions stored in the replay buffer) is an off-policy data and the Q-function of a fixed policy can be easily evaluated with this type of a data using off-policy TD(0), unlike the value function.

Remark 1 In order for V φ to be an accurate estimate of the value function of π i−1 on Line 8, we should use an additional target network, Q θ , that remains unchanged during the policy improvement stage.

This network should be used in π i−1 (·) = arg max a Q θ (·, a) on Line 8, and be only updated right after the improvement stage on Line 11.

However, to reduce the space complexity of the algorithm, we do not use this additional target network and compute π i−1 on Line 8 as arg max Q θ , despite the fact that Q θ changes during the improvement stage.

We report the pseudo-code of κ-VI-DQN in Appendix A.1 (Alg.

6).

Note that κ-VI simply repeats V ← T κ V and computes T κ V , which is the optimal value of the surrogate MDP M γκ (V ).

In κ-VI-DQN, we repeatedly solve M γκ (V ) by DQN, and use its optimal Q-function to shape the reward of the next iteration.

Let Q * γκ,V and V * γκ,V be the optimal Q and V functions of M γκ (V ).

, where the first equality is by definition (Sec. 2) and the second one holds since T κ V is the optimal value of M γκ (V ) (Sec. 3).

Therefore, in κ-VI-DQN, we shape the reward of each iteration by max a Q φ (s, a), where Q φ is the output of the DQN from the previous iteration, i.e., max a Q φ (s, a) T κ V i−1 .

In this section, we empirically analyze the performance of the κ-PI-DQN and κ-VI-DQN algorithms on the Atari domains: Breakout, Seaquest, SpaceInvaders, and Enduro (Bellemare et al., 2013) .

We start by performing an ablation test on three values of parameter C F A = {0.001, 0.05, 0.2} on the Breakout domain.

The value of C F A sets the number of samples per iteration T (κ) (Eq. 8) and the total number of iterations N (κ) (Eq. 9).

Aside from C F A , we set the total number of samples to T 10 6 .

This value represents the number of samples after which our DQN-based algorithms approximately converge.

For each value of C F A , we test κ-PI-DQN and κ-VI-DQN for several κ values.

In both algorithms, the best performance was obtained with C F A = 0.05, thus, we set C F A = 0.05 in our experiments with other Atari domains.

Alg.

Table 1 shows the final training performance of κ-PI-DQN and κ-VI-DQN on the Atari domains with C F A = 0.05.

Note that the scores reported in Table 1 are the actual returns of the Atari domains, while the vertical axis in the plots of Figure 1 corresponds to a scaled return.

We plot the scaled return, since this way it would be easier to reproduce our results using the OpenAI Baselines codebase (Hill et al., 2018) .

The results of Fig. 1 and Table 1 , as well as those in Appendix A.2, exhibit that both κ-PI-DQN and κ-VI-DQN improve the performance of DQN (κ = 1).

Moreover, they show that setting N (κ) = T leads to a clear degradation of the final training performance on all of the domains expect Enduro, which attains better performance for N (κ) = T .

Although the performance degrades, the results for N (κ) = T are still better than for DQN.

Algorithm 4 contains the pseudo-code of κ-PI-TRPO (detailed pseudo-code in Appendix A.1).

TRPO iteratively updates the current policy using its return and an estimate of its value function.

In our κ-PI-TRPO, at each iteration i: 1) we use the estimate of the current policy V φ V πi−1 (computed in the previous iteration) to calculate the return R(κ, V φ ) and an estimate of the value function V θ of the surrogate MDP M γκ (V φ ), 2) we use the return R(κ, V φ ) and V θ to compute the new policy π i , and 3) we estimate the value of the new policy V φ V πi on the original, γ discounted, MDP.

In Appendix B.1 we provide the pseudocode of κ-VI-TRPO derived by the κ-VI meta algorithm.

As previously noted, κ-VI iteratively solves the γκ discounted surrogate MDP and uses its optimal value T κ V i−1 to shape the reward of the surrogated MDP in the i'th iteration.

With that in mind, consider κ-PI-TRPO.

Notice that as π θ converges to the optimal policy of the surrogate γκ discounted MDP, Vθ converges to the optimal value of the surrogate MDP, i.e., it converges to

Thus, κ-PI-TRPO can be turn to κ-VI-TRPO by eliminating the policy evaluation stage, and simply copy φ ←θ, meaning, V φ ← Vθ = T κ V φ .

In this section, we empirically analyze the performance of the κ-PI-TRPO and κ-VI-TRPO algorithms on the MuJoCo domains: Walker2d-v2, Ant-v2, HalfCheetah-v2, HumanoidStandup-v2, and Swimmer-v2, (Todorov et al., 2012) .

As in Section 5.1.1, we start by performing an ablation test on the parameter C F A = {0.001, 0.05, 0.2} on the Walker domain.

We set the total number of iterations to 2000, with each iteration consisting 1000 samples.

Thus, the total number of samples is T 2 × 10 6 .

This is the number of samples after which our TRPO-based algorithms approximately converge.

For each value of C F A , we test κ-PI-TRPO and κ-VI-TRPO for several κ values.

In both algorithms, the best performance was obtained with C F A = 0.2, thus, we set C F A = 0.2 in our experiments with other MuJoCo domains.

1: Initialize V -networks V θ and V φ , policy network π ψ , and target network V φ ; 2: for i = 0, . . .

, N (κ) − 1 do 3:

for t = 1, . . .

, T (κ) do 4:

Simulate the current policy π ψ for M steps and calculate the following two returns for all steps j: 5:

Rj(κ, V φ ) = M t=j (γκ) t−j rt(κ, V φ ) and ρj = M t=j γ t−j rt;

Update θ by minimizing the batch loss function:

# Policy Improvement 8:

Update ψ using TRPO by the batch {(Rj(κ, V φ ), V θ (sj))} N j=1 ; 9:

# Policy Evaluation 10:

Update φ by minimizing the batch loss function:

end for 12:

Copy φ to φ (φ ← φ); 13: end for Table 2 shows the final training performance of κ-PI-TRPO and κ-VI-TRPO on the MuJoCo domains with C F A = 0.2.

The results of Figure 2 and Table 2 , as well as those in Appendix B.3, exhibit that both κ-PI-TRPO and κ-VI-TRPO yield better performance than TRPO (κ = 1).

Furthermore, they show that the algorithms with C F A = 0.2 perform better than with N (κ) = T .

However, the improvement is less significant relative to the DQN-based results in Section 5.1.1.

There is an intimate relation between κ-PI and the GAE algorithm Schulman et al. (2016) which we elaborate on in this section.

In GAE the policy is updated by the gradient:

which can be interpreted as a gradient step in a γλ discounted MDP with rewards δ(V ), which we refer here as M δ(V )

γλ .

As noted in Efroni et al. (2018a) , Section 6, the optimal policy of the MDP M δ(V ) γλ is the optimal policy of M γκ (V ) with κ = λ, i.e., the κ-greedy policy w.r.t.

V : thus, the Domain Alg. is the κ-greedy policy w.r.t.

V .

GAE, instead of solving the κ-greedy policy while keeping V fixed, changes the policy and updates V by the return concurrently.

Thus, this approach is conceptually similar to κ-PI-TRPO with N (κ) = T .

There, the value and policy are concurrently updated as well, without clear separation between the update of the policy and the value.

In Figure 2 and Table 2 the performance of GAE is compared to the one of κ-PI-TRPO and κ-VI-TRPO.

The performance of the latter two is slightly better than the one of GAE.

Remark 2 (Implementation of GAE) We used the OpenAI baseline implementation of GAE with a small modification.

In the baseline code, the value network is updated w.r.t.

to the target t (γλ) t r t , whereas in Schulman et al. (2016) the authors used the target t γ t r t (see Schulman et al. (2016) , Eq.28).

We chose the latter form in our implementation to be in accord with Schulman et al. (2016) .

To supply with a more complete view on our experiments, we tested the performance of the "vanilla" DQN and TRPO when trained with different γ values than the previously used one (γ = 0.99).

As evident in Figure 3 , only for the Ant domain this approach resulted in improved performance when for TRPO trained with γ = 0.68.

It is interesting to observe that for the Ant domain the performance of κ-PI-TRPO and especially of κ-VI-TRPO (Table 2 ) significantly surpassed the one of TRPO trained with γ = 0.68.

The performance of DQN and TRPO on the Breakout, SpaceInvaders and Walker domains decreased or remained unchanged in the tested γ values.

Thus, on these domains, changing the discount factor does not improve the DQN and TRPO algorithms, as using κ-PI or κ-VI with smaller κ value do.

It is interesting to observe that the performance on the Mujoco domains for small γ, e.g., γ = 0.68, achieved good performance, whereas for the Atari domains the performance degraded with lowering γ.

This fits the nature of these domains: in the Mujoco domains the decision problem inherently has much shorter horizon than in the Atari domains.

Furthermore, it is important to stress that γ and κ are two different parameters an algorithm designer may use.

For example, one can perform a scan of γ value, fix γ to the one with optimal performance, and then test the performance of different κ values.

In this work we formulated and empirically tested simple generalizations of DQN and TRPO derived by the theory of multi-step DP and, specifically, of κ-PI and κ-VI algorithms.

The empirical investigation reveals several points worth emphasizing.

1.

κ-PI is better than κ-VI for the Atari domains..

In most of the experiments on the Atari domains κ-PI-DQN has better performance than κ-VI-DQN.

This might be expected as the former uses extra information not used by the latter: κ-PI estimates the value of current policy whereas κ-VI ignores this information.

2.

For the Gym domains κ-VI performs slightly better than κ-PI.

For the Gym domains κ-VI-TRPO performs slightly better than κ-PI-TRPO.

We conjecture that the reason for the discrepancy relatively to the Atari domains lies in the inherent structure of the tasks of the Gym domains: they are inherently short horizon decision problems.

For this reason, the problems can be solved with smaller discount factor (as empirically demonstrated in Section 5.3) and information on the policy's value is not needed.

3.

Non trivial κ value improves the performance.

In the vast majority of our experiments both κ-PI and κ-VI improves over the performance of their vanilla counterparts (i.e., κ = 1), except for the Swimmer and BeamRider domains from Mujoco and Atari suites.

Importantly, the performance of the algorithms was shown to be 'smooth' in the parameter κ.

This suggests careful hyperparameter tuning of κ is not of great necessity.

4. Using the 'naive' choice of N (κ) = T deteriorates the performance.

Choosing the number of iteration by Eq. 8 improves the performance on the tested domains.

An interesting future work would be to test model-free algorithms which use other variants of greedy policies (Bertsekas & Tsitsiklis, 1996; Bertsekas, 2018; Efroni et al., 2018a; Sun et al., 2018; Shani et al., 2019) .

Furthermore, and although in this work we focused on model-free DRL, it is arguably more natural to use multi-step DP in model-based DRL (e.g., Kumar et al., 2016; Talvitie, 2017; Luo et al., 2018; Janner et al., 2019) .

Taking this approach, the multi-step greedy policy would be solved with an approximate model.

We conjecture that in this case one may set κ -or more generally, the planning horizon -as a function of the approximate model's 'quality': as the approximate model gets closer to the real model larger κ can be used.

We leave investigating such relation in theory and practice to future work.

Lastly, an important next step in continuation to our work is to study algorithms with an adaptive κ parameter.

This, we believe, would greatly improve the resulting methods, and possibly be done by studying the relation between the different approximation errors (i.e., errors in gradient and value estimation, Ilyas et al., 2018) , the performance and the κ value that should be used by the algorithm.

A DQN IMPLEMENTATION OF κ-PI AND κ-VI

In this section, we report the detailed pseudo-codes of the κ-PI-DQN and κ-VI-DQN algorithms, described in Section 5.1, side-by-side.

Algorithm 5 κ-PI-DQN 1: Initialize replay buffer D, and Q-networks Q θ and Q φ with random weights θ and φ; 2: Initialize target networks Q θ and Q φ with weights θ ← θ and φ ← φ;

# Policy Improvement 5:

Select a t as an -greedy action w.r.t.

Q θ (s t , a);

Execute a t , observe r t and s t+1 , and store the tuple (s t , a t , r t , s t+1 ) in D;

8:

Sample a random mini-batch {(s j , a j , r j , s j+1 )} N j=1 from D;

Update θ by minimizing the following loss function:

10:

11:

Copy θ to θ occasionally (θ ← θ); Set π i (s) ∈ arg max a Q θ (s, a);

16:

Update φ by minimizing the following loss function:

19:

Copy φ to φ occasionally (φ ← φ);

end for 22: end for Algorithm 6 κ-VI-DQN 1: Initialize replay buffer D, and Q-networks Q θ and Q φ with random weights θ and φ; 2: Initialize target network Q θ with weights θ ← θ;

# Evaluate T κ V φ and the κ-greedy policy w.r.t.

V φ 5:

Select a t as an -greedy action w.r.t.

Q θ (s t , a);

Execute a t , observe r t and s t+1 , and store the tuple (s t , a t , r t , s t+1 ) in D;

8:

Update θ by minimizing the following loss function:

10:

Copy θ to θ occasionally (θ ← θ); In this section, we report additional results of the application of κ-PI-DQN and κ-VI-DQN on the Atari domains.

A summary of these results has been reported in Table 1 in the main paper.

B TRPO IMPLEMENTATION OF κ-PI AND κ-VI

In this section, we report the detailed pseudo-codes of the κ-PI-TRPO and κ-VI-TRPO algorithms, described in Section 5.2, side-by-side.

Algorithm 7 κ-PI-TRPO 1: Initialize V -networks V θ and V φ , and policy network π ψ with random weights θ, φ, and ψ 2: Initialize target network V φ with weights φ ← φ 3: for i = 0, . . .

, N (κ) − 1 do 4:

Simulate the current policy π ψ for M time-steps;

6:

end for

Sample a random mini-batch {(s j , a j , r j , s j+1 )} N j=1 from the simulated M time-steps; 10:

Update θ by minimizing the loss function:

11:

# Policy Improvement 12:

Sample a random mini-batch {(s j , a j , r j , s j+1 )} N j=1 from the simulated M time-steps; 13:

Update ψ using TRPO with advantage function computed by Update φ by minimizing the loss function:

end for # Evaluate T κ V φ and the κ-greedy policy w.r.t.

V φ 5:

Simulate the current policy π ψ for M time-steps; 7:

end for 10:

Sample a random mini-batch {(s j , a j , r j , s j+1 )} N j=1 from the simulated M time-steps 11:

Update θ by minimizing the loss function:

Sample a random mini-batch {(s j , a j , r j , s j+1 )} N j=1 from the simulated M time-steps 13:

Update ψ using TRPO with advantage function computed by In this section, we report additional results of the application of κ-PI-TRPO and κ-VI-TRPO on the MuJoCo domains.

A summary of these results has been reported in Table 2 in the main paper.

C REBUTTAL RESULTS

In this section, we analyze the role κ plays in the proposed methods by reporting results on the simple CartPole environment for κ-PI TRPO.

For all experiments, we use a single layered value function network and a linear policy network.

Each hyperparameter configuration is run for 10 different random seeds and plots are shown for a 50% confidence interval.

Note that since the CartPole is extremely simple, we do not see a clear difference between the κ values that are closer to 1.0 (see Figure 16 ).

Below, we observe the performance when the discount factor γ is lowered (see Figure 17) .

Since, there is a ceiling of R = 200 on the maximum achievable return, it makes intuitive sense that observing the κ effect for a lower gamma value such as γ = 0.36 will allow us to see a clearer trade-off between κ values.

To this end, we also plot the results for when the discount factor is set to 0.36.

The intuitive idea behind κ-PI, and κ-VI similarly, is that at every time step, we wish to solve a simpler sub-problem, i.e. the γκ discounted MDP.

Although, we are solving an easier/shorter horizon problem, in doing so, the bias induced is taken care of by the modified reward in this new MDP.

Therefore, it becomes interesting to look at how κ affects its two contributions, one being the discounting, the other being the weighting of the shaped reward (see eq. 11).

Below we look at what happens when each of these terms are made κ independent, one at a time, while varying κ for the other term.

To make this clear, we introduce different notations for both such κ instances, one being κ d (responsible for discounting) and the other being κ s (responsible for shaping).

We see something interesting here.

For the CartPole domain, the shaping term does not seem to have any effect on the performance (Figure 18(b) ), while the discounting term does.

This implies that the problem does not suffer from any bias issues.

Thus, the correction provided by the shaped term is not needed.

However, this is not true for other more complex problems.

This is also why we see a similar result when lowering γ in this case, but not for more complex problems.

In this section, we report results for the Mountain Car environment.

Contrary to the CartPole results, where lowering the κ values degraded the performance, we observe that performance deteriorates when κ is increased.

We also plot a bar graph, with the cumulative score on the y axis and different κ values on the x axis.

We use the continuous Mountain Car domain here, which has been shown to create exploration issues.

Therefore, without receiving any positive reward, using a κ value of 0 in the case of discounting (solving the 1 step problem has the least negative reward) and of 1 in the case of shaping results in the best performance.

In this section, we move to the Pendulum environment, a domain where we see a non-trivial best κ value.

This is due to there not being a ceiling on the maximum possible return, which is the case in CartPole.

Under review as a conference paper at ICLR 2020

Choosing the best γ value and running κ-PI on it results in an improved performance for all κ values (see Figure 23 ).

To summarize, we believe that in inherently short horizon domains (dense, per time step reward), such as the Mujoco continuous control tasks, the discounting produced by κ-PI and VI is shown to cause major improvement in performance over the TRPO baselines.

This is reinforced by the results of lowering the discount factor experiments.

On the other hand, in inherently long horizon domains (sparse, end of trajectory reward), such as in Atari, the shaping produced by κ-PI and VI is supposed to cause the major improvement over the DQN baselines.

Again, this is supported by the fact that lowering the discount factor experiments actually result in deterioration in performance.

Figure 25: Cumulative training performance of κ-PI-TRPO on HalfCheetah (Left, corresponds to Figure 12 ) and Ant (Right, corresponds to Figure 11 ) environments.

<|TLDR|>

@highlight

Use model free algorithms like DQN/TRPO to solve short horizon problems (model free) iteratively in a Policy/Value Iteration fashion.
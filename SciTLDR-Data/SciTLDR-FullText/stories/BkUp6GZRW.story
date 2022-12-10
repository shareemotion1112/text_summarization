This paper proposes a new actor-critic-style algorithm called Dual Actor-Critic or Dual-AC.

It is derived in a principled way from the Lagrangian dual form of the Bellman optimality equation, which can be viewed as a two-player game between the actor and a critic-like function, which is named as dual critic.

Compared to its actor-critic relatives, Dual-AC has the desired property that the actor and dual critic are updated cooperatively to optimize the same objective function, providing a more transparent way for learning the critic that is directly related to the objective function of the actor.

We then provide a concrete algorithm that can effectively solve the minimax optimization problem, using techniques of multi-step bootstrapping, path regularization, and stochastic dual ascent algorithm.

We demonstrate that the proposed algorithm achieves the state-of-the-art performances across several benchmarks.

Reinforcement learning (RL) algorithms aim to learn a policy that maximizes the long-term return by sequentially interacting with an unknown environment.

Value-function-based algorithms first approximate the optimal value function, which can then be used to derive a good policy.

These methods BID23 BID28 often take advantage of the Bellman equation and use bootstrapping to make learning more sample efficient than Monte Carlo estimation BID25 .

However, the relation between the quality of the learned value function and the quality of the derived policy is fairly weak BID6 .

Policy-search-based algorithms such as REINFORCE BID29 and others (Kakade, 2002; BID18 , on the other hand, assume a fixed space of parameterized policies and search for the optimal policy parameter based on unbiased Monte Carlo estimates.

The parameters are often updated incrementally along stochastic directions that on average are guaranteed to increase the policy quality.

Unfortunately, they often have a greater variance that results in a higher sample complexity.

Actor-critic methods combine the benefits of these two classes, and have proved successful in a number of challenging problems such as robotics (Deisenroth et al., 2013) , meta-learning BID3 , and games (Mnih et al., 2016 ).

An actor-critic algorithm has two components: the actor (policy) and the critic (value function).

As in policy-search methods, actor is updated towards the direction of policy improvement.

However, the update directions are computed with the help of the critic, which can be more efficiently learned as in value-function-based methods BID24 Konda & Tsitsiklis, 2003; BID13 BID7 BID19 .

Although the use of a critic may introduce bias in learning the actor, its reduces variance and thus the sample complexity as well, compared to pure policy-search algorithms.

While the use of a critic is important for the efficiency of actor-critic algorithms, it is not entirely clear how the critic should be optimized to facilitate improvement of the actor.

For some parametric family of policies, it is known that a certain compatibility condition ensures the actor parameter update is an unbiased estimate of the true policy gradient BID24 .

In practice, temporaldifference methods are perhaps the most popular choice to learn the critic, especially when nonlinear function approximation is used (e.g., BID19 ).In this paper, we propose a new actor-critic-style algorithm where the actor and the critic-like function, which we named as dual critic, are trained cooperatively to optimize the same objective function.

The algorithm, called Dual Actor-Critic , is derived in a principled way by solving a dual form of the Bellman equation BID6 .

The algorithm can be viewed as a two-player game between the actor and the dual critic, and in principle can be solved by standard optimization algorithms like stochastic gradient descent (Section 2).

We emphasize the dual critic is not fitting the value function for current policy, but that of the optimal policy.

We then show that, when function approximation is used, direct application of standard optimization techniques can result in instability in training, because of the lack of convex-concavity in the objective function (Section 3).

Inspired by the augmented Lagrangian method (Luenberger & Ye, 2015; Boyd et al., 2010) , we propose path regularization for enhanced numerical stability.

We also generalize the two-player game formulation to the multi-step case to yield a better bias/variance tradeoff.

The full algorithm is derived and described in Section 4, and is compared to existing algorithms in Section 5.

Finally, our algorithm is evaluated on several locomotion tasks in the MuJoCo benchmark BID27 , and compares favorably to state-of-the-art algorithms across the board.

Notation.

We denote a discounted MDP by M = (S, A, P, R, γ), where S is the state space, A the action space, P (·|s, a) the transition probability kernel defining the distribution over next-state upon taking action a in state x, R(s, a) the corresponding immediate rewards, and γ ∈ (0, 1) the discount factor.

If there is no ambiguity, we will use a f (a) and f (a)da interchangeably.

In this section, we first describe the linear programming formula of the Bellman optimality equation BID5 BID14 , paving the path for a duality view of reinforcement learning via Lagrangian duality.

In the main text, we focus on MDPs with finite state and action spaces for simplicity of exposition.

We extend the duality view to continuous state and action spaces in Appendix A.2.Given an initial state distribution µ(s), the reinforcement learning problem aims to find a policy π(·|s) : S → P(A) that maximizes the total expected discounted reward with P(A) denoting all the probability measures over A, i.e., DISPLAYFORM0 where DISPLAYFORM1 which can be formulated as a linear program BID14 BID5 : DISPLAYFORM2 DISPLAYFORM3 For completeness, we provide the derivation of the above equivalence in Appendix A. Without loss of generality, we assume there exists an optimal policy for the given MDP, namely, the linear programming is solvable.

The optimal policy can be obtained from the solution to the linear program (3) via π DISPLAYFORM4 The dual form of the LP below is often easier to solve and yield more direct relations to the optimal policy.

DISPLAYFORM5 s.t.

a∈A ρ(s , a) = (1 − γ)µ(s ) + γ s,a∈S×A ρ(s, a)P (s |s, a)ds, ∀s ∈ S. Since the primal LP is solvable, the dual LP is also solvable, and P * − D * = 0.

The optimal dual variables ρ * (s, a) and optimal policy π * (a|s) are closely related in the following manner:Theorem 1 (Policy from dual variables) s,a∈S×A ρ * (s, a) = 1, and π DISPLAYFORM6 Since the goal of reinforcement learning is to learn an optimal policy, it is appealing to deal with the Lagrangian dual which optimizes the policy directly, or its equivalent saddle point problem that jointly learns the optimal policy and value function.

Theorem 2 (Competition in one-step setting) The optimal policy π * , actor, and its corresponding value function V * , dual critic, is the solution to the following saddle-point problem DISPLAYFORM7 where DISPLAYFORM8 The saddle point optimization (6) provides a game perspective in understanding the reinforcement learning problem (Goodfellow et al., 2014) .

The learning procedure can be thought as a game between the dual critic, i.e., value function for optimal policy, and the weighted actor, i.e., α(s)π(a|s): the dual critic V seeks the value function to satisfy the Bellman equation, while the actor π tries to generate state-action pairs that break the satisfaction.

Such a competition introduces new roles for the actor and the dual critic, and more importantly, bypasses the unnecessary separation of policy evaluation and policy improvement procedures needed in a traditional actor-critic framework.

To solve the dual problem in (6), a straightforward idea is to apply stochastic mirror prox BID8 or stochastic primal-dual algorithm (Chen et al., 2014) to address the saddle point problem in (6).

Unfortunately, such algorithms have limited use beyond special cases.

For example, for an MDP with finite state and action spaces, the one-step saddle-point problem (6) with tabular parametrization is convex-concave, and finite-sample convergence rates can be established; see e.g., Chen & Wang (2016) and Wang (2017) .

However, when the state/action spaces are large or continuous so that function approximation must be used, such convergence guarantees no longer hold due to lack of convex-concavity.

Consequently, directly solving (6) can suffer from severe bias and numerical issues, resulting in poor performance in practice (see, e.g., FIG1 ):1.

Large bias in one-step Bellman operator: It is well-known that one-step bootstrapping in temporal difference algorithms has lower variance than Monte Carlo methods and often require much fewer samples to learn.

But it produces biased estimates, especially when function approximation is used.

Such a bias is especially troublesome in our case as it introduces substantial noise in the gradients to update the policy parameters.2.

Absence of local convexity and duality: Using nonlinear parametrization will easily break the local convexity and duality between the original LP and the saddle point problem, which are known as the necessary conditions for the success of applying primal-dual algorithm to constrained problems (Luenberger & Ye, 2015) .

Thus none of the existing primal-dual type algorithms will remain stable and convergent when directly optimizing the saddle point problem without local convexity.3.

Biased stochastic gradient estimator with under-fitted value function: In the absence of local convexity, the stochastic gradient w.r.t.

the policy π constructed from under-fitted value function will presumably be biased and futile to provide any meaningful improvement of the policy.

Hence, naively extending the stochastic primal-dual algorithms in Chen & Wang (2016); Wang (2017) for the parametrized Lagrangian dual, will also lead to biased estimators and sample inefficiency.

In this section, we will introduce several techniques to bypass the three instability issues in the previous section: (1) generalization of the minimax game to the multi-step case to achieve a better bias-variance tradeoff; (2) use of path regularization in the objective function to promote local convexity and duality; and (3) use of stochastic dual ascent to ensure unbiased gradient estimates.

In this subsection, we will extend the minimax game between the actor and critic to the multi-step setting, which has been widely utilized in temporal-difference algorithms for better bias/variance tradeoffs BID25 Kearns & Singh, 2000) .

By the definition of the optimal value function, it is easy to derive the k-step Bellman optimality equation as DISPLAYFORM0 Similar to the one-step case, we can reformulate the multi-step Bellman optimality equation into a form similar to the LP formulation, and then we establish the duality, which leads to the following mimimax problem:Theorem 3 (Competition in multi-step setting) The optimal policy π * and its corresponding value function V * is the solution to the following saddle point problem DISPLAYFORM1 DISPLAYFORM2 The saddle-point problem FORMULA10 is similar to the one-step Lagrangian (6): the dual critic, V , and weighted k-step actor, α(s 0 ) k i=0 π(a i |s i ), are competing for an equilibrium, in which critic and actor become the optimal value function and optimal policy.

However, it should be emphasized that due to the existence of max-operator over the space of distributions P(A), rather than A, in the multi-step Bellman optimality equation FORMULA9 , the establishment of the competition in multi-step setting in Theorem 3 is not straightforward: i), its corresponding optimization is no longer a linear programming; ii), the strong duality in FORMULA10 is not obvious because of the lack of the convex-concave structure.

We first generalize the duality to multi-step setting.

Due to space limit, detailed analyses for generalizing the competition to multi-step setting are provided in Appendix B.

When function approximation is used, the one-step or multi-step saddle-point problems (8) will no longer be convex in the primal parameters.

This could lead to instability and even divergence when solved by brute-force stochastic primal-dual algorithms.

One then desires to partially convexify the objectives without affecting the optimal solutions.

The augmented Lagrangian method (Boyd et al., 2010; Luenberger & Ye, 2015) , also known as the method of multipliers, is designed and widely used for such purposes.

However, directly applying this method would require introducing penalty functions of the multi-step Bellman operator, which renders extra complexity and challenges in optimization.

Interested readers are referred to Appendix B.2 for further details.

Instead, we propose to use path regularization, as a stepping stone for promoting local convexity and computation efficiency.

The regularization term is motivated by the fact that the optimal value function satisfies the constraint DISPLAYFORM0 In the same spirit as augmented Lagrangian, we will introduce to the objective the simple penalty function DISPLAYFORM1 2 , leading to the following: DISPLAYFORM2 where η V 0 is a hyper-parameter controlling the strength of the regularization.

Note that in the penalty function above we use a behavior policy π b instead of an optimal policy, since the latter is unknown.

Adding such a regularization enables local duality in the primal parameters.

Indeed, this can be easily verified by showing the positive definiteness of the Hessian at a local solution.

We call this approach path regularization, since it exploits the rewards in the sample path to regularize the solution path of value function V in the optimization procedure.

As a by-product, the regularization also provides a mechanism to utilize off-policy samples from behavior policy π b .One can also see that the regularization indeed provides guidance and preference to search for the solution path.

Specifically, in each update of V the learning procedure, it tries to move towards the optimal value function while staying close to the value function of the behavior policy π b .

Intuitively, such regularization restricts the feasible domain of candidates V to be a ball centered at V π b .

Besides enhancing local convexity, such a penalty also avoids unboundedness of V in the learning procedure, and thus more numerical robust.

As long as the optimal value function is indeed in such a region, the introduced side-effect can be controlled.

Formally, we can show that with appropriate η V , the optimal solution (V * , α * , π * ) is not affected.

The main results of this subsection are summarized by the following theorem.

Theorem 4 (Property of path regularization) The local duality holds for L r (V, α, π).

Denote (V * , α * , π * ) as the solution to Bellman optimality equation, with some appropriate DISPLAYFORM3 The proof of the theorem is given in Appendix B.3.

We emphasize that the theorem holds when V is given enough capacity, i.e., in the nonparametric limit.

With parametrization introduced, definitely approximation error will be introduced, and the valid range of η V , which keeps optimal solution unchanged, will be affected.

However, the function approximation error is still an open problem for general class of parametrization, we omit such discussion here which is out of the range of this paper.

Rather than the primal form, i.e., min DISPLAYFORM0 The major reason is due to the sample efficiency consideration.

In the primal form, to apply the stochastic gradient descent algorithm at V t , one needs to solve max α∈P(S),π∈P(A) L r (V t , α, π) which involves sampling from each π and α during the solution path for the subproblem.

We define the regularized dual function r (α, π) := min V L r (V, α, π).

We first show the unbiased gradient estimator of r w.r.t.

θ ρ = (θ α , θ π ), which are parameters associated with α and π.

Then, we incorporate the stochastic update rule to the dual ascent algorithm (Boyd et al., 2010) , resulting in the dual actor-critic (Dual-AC) algorithm.

The gradient estimators of the dual functions can be derived using chain rule and are provided below.

Theorem 5 The regularized dual function r (α, π) has gradients estimators DISPLAYFORM1 DISPLAYFORM2 Therefore, we can apply stochastic mirror descent algorithm with the gradient estimator given in Theorem 5 to the regularized dual function r (α, π).

Since the dual variables are probabilistic distributions, it is natural to use KL-divergence as the prox-mapping to characterize the geometry in the family of parameters BID1 BID8 .

Specifically, in the t-th iteration, θ DISPLAYFORM3 DISPLAYFORM4 denotes the stochastic gradients estimated through (10) and (11) via given samples and KL(q(s, a)||p(s, a)) = q(s, a) log q(s,a) p(s,a) dsda.

Intuitively, such update rule emphasizes a trade-off between the current policy and possible improvements based on samples.

The update of π shares some similarity to the TRPO, which is derived to ensure monotonic improvement of the new policy BID18 .

We discuss the details in Section 4.4.Rather than just update V once via the stochastic gradient of ∇ V L r (V, α, π) in each iteration for solving saddle-point problem BID8 , which is only valid in convexconcave setting, Dual-AC exploits the stochastic dual ascent algorithm which requires V t = Published as a conference paper at ICLR 2018 Decay the stepsize: DISPLAYFORM5 Compute the stochastic gradients for θ π following (11).8:Update θ t π according to the exact prox-mapping (16) or the approximate closed-form (17).

9: end for argmin V L r (V, α t−1 , π t−1 ) in t-th iteration for estimating ∇ θρ r (θ α , θ π ).

As we discussed, such operation will keep the gradient estimator of dual variables unbiased, which provides better direction for convergence.

In Algorithm 1, we update V t by solving optimization min V L r (V, α t−1 , π t−1 ).

In fact, the V function in the path-regularized Lagrangian L r (V, α, π) plays two roles: i), inherited from the original Lagrangian, the first two terms in regularized Lagrangian (9) push the V towards the value function of the optimal policy with on-policy samples; ii), on the other hand, the path regularization enforces V to be close to the value function of behavior policy π b with off-policy samples.

Therefore, the V function in the Dual-AC algorithm can be understood as an interpolation between these two value functions learned from both on and off policy samples.

In above, we have introduced path regularization for recovering local duality property of the parametrized multi-step Lagrangian dual form and tailored stochastic mirror descent algorithm for optimizing the regularized dual function.

Here, we present several strategies for practical computation considerations.

Update rule of V t .

In each iteration, we need to solve V t = argmin θ V L r (V, α t−1 , π t−1 ), which depends on π b and η V , for estimating the gradient for dual variables.

In fact, the closer π b to π * is, DISPLAYFORM0 2 will be.

Therefore, we can set η V to be large for better local convexity and faster convergence.

Intuitively, the π t−1 is approaching to π * as the algorithm iterates.

Therefore, we can exploit the policy obtained in previous iteration, i.e., π t−1 , as the behavior policy.

The experience replay can also be used.

Furthermore, notice the L(V, α t−1 , π t−1 ) is a expectation of functions of V , we will use stochastic gradient descent algorithm for the subproblem.

Other efficient optimization algorithms can be used too.

Specifically, the unbiased gradient estimator for DISPLAYFORM1 We can use k-step Monte Carlo approximation for E DISPLAYFORM2 in the gradient estimator.

As k is large enough, the truncate error is negligible BID25 ).

We will iterate via θ DISPLAYFORM3 It should be emphasized that in our algorithm, V t is not trying to approximate the value or advantage function of π t , in contrast to most actor-critic algorithms.

Although V t eventually becomes an approximation of the optimal value function once the solution reaches the global optimum, in each update V t is merely a function that helps the current policy to be improved.

From this perspective, the Dual-AC bypasses the policy evaluation step.

Update rule of α t .

In practice, we may face with the situation that the initial sampling distribution is fixed, e.g., in MuJoCo tasks.

Therefore, we cannot obtain samples from α t (s) at each iteration.

We assume that ∃η µ ∈ (0, 1], such that α(s) = (1 − η µ )β(s) + η µ µ(s) with β(s) ∈ P(S).

Hence, we have s) .

Note that such an assumption is much weaker comparing with the requirement for popular policy gradient algorithms (e.g., BID24 ; BID22 ) that assumes µ(s) to be a stationary distribution.

In fact, we can obtain a closed-form update forα if a square-norm regularization term is introduced into the dual function.

Specifically, Theorem 6 In t-th iteration, given V t and π t−1 , DISPLAYFORM4 DISPLAYFORM5 Then, we can updateα t through FORMULA0 with Monte Carlo approximation of DISPLAYFORM6 , s k+1 , avoiding the parametrization ofα.

As we can see, theα t (s)reweights the samples based on the temporal differences and this offers a principled justification for the heuristic prioritized reweighting trick used in BID17 .Update rule of θ t π .

The parameters for dual function, θ ρ , are updated by the prox-mapping operator (12) following the stochastic mirror descent algorithm for the regularized dual function.

Specifically, in t-th iteration, given V t and α t , for θ π , the prox-mapping (12) reduces to DISPLAYFORM7 DISPLAYFORM8 .

Then, the update rule will become exactly the natural policy gradient (Kakade, 2002) with a principled way to compute the "policy gradient"ĝ t π .

This can be understood as the penalty version of the trust region policy optimization BID18 , in which the policy parameters conservative update in terms of KL-divergence is achieved by adding explicit constraints.

Exactly solving the prox-mapping for θ π requires another optimization, which may be expensive.

To further accelerate the prox-mapping, we approximate the KL-divergence with the second-order Taylor expansion, and obtain an approximate closed-form update given by DISPLAYFORM9 where DISPLAYFORM10 denotes the Fisher information matrix.

Empirically, we may normalize the gradient by its norm g t π F −1 t g t π BID15 for better performances.

Combining these practical tricks to the stochastic mirror descent update eventually gives rise to the dual actor-critic algorithm outlined in Algorithm 1.

The dual actor-critic algorithm includes both the learning of optimal value function and optimal policy in a unified framework based on the duality of the linear programming ( FORMULA0 , but they either do not focus on concrete algorithms for solving the optimization problem, or require certain knowledge of the transition probability function that may be hard to obtain in practice.

The duality view has also been exploited in BID10 .

Their algorithm is based on the duality of entropy-regularized Bellman equation BID26 BID16 Fox et al., 2016; Haarnoja et al., 2017; BID2 Nachum et al., 2017) , rather than the exact Bellman optimality equation we try to solve in this work.

Our dual actor-critic algorithm can be understood as a nontrivial extension of the (approximate) dual gradient method (Bertsekas, 1999, Chapter 6.

3) using stochastic gradient and Bregman divergence, which essentially parallels the view of (approximate) stochastic mirror descent algorithm (Nemirovski et al., 2009) in the primal space.

As a result, the algorithm converges with diminishing stepsizes and decaying errors from solving subproblems.

Particularly, the update rules of α and π in the dual actor-critic are related to several existing algorithms.

As we see in the update of α, the algorithm reweighs the samples which are not fitted well.

This is related to the heuristic prioritized experience replay BID17 .

For the update in π, the proposed algorithm bears some similarities with trust region poicy gradient (TRPO) BID18 and natural policy gradient (Kakade, 2002; BID15 .

Indeed, TRPO and NPR solve the same prox-mapping but are derived from different perspectives.

We emphasize that although the updating rules share some resemblance to several reinforcement learning algorithms in the literature, they are purely originated from a stochastic dual ascent algorithm for solving the two-play game derived from Bellman optimality equation.

We evaluated the dual actor-critic (Dual-AC) algorithm on several continuous control environments from the OpenAI Gym (Brockman et al., 2016) with MuJoCo physics simulator BID27 .

We compared Dual-AC with several representative actor-critic algorithms, including trust region policy optimization (TRPO) BID18 and proximal policy optimization (PPO) BID20 1 .

We ran the algorithms with 5 random seeds and reported the average rewards with 50% confidence interval.

Details of the tasks and setups of these experiments including the policy/value function architectures and the hyperparameters values, are provided in Appendix C.

To justify our analysis in identifying the sources of instability in directly optimizing the parametrized one-step Lagrangian duality and the effect of the corresponding components in the dual actor-critic algorithm, we perform comprehensive Ablation study in InvertedDoublePendulum-v1, Swimmerv1, and Hopper-v1 environments.

We also considered the effect of k = {10, 50} besides the one-step result in the study to demonstrate the benefits of multi-step.

We conducted comparison between the Dual-AC and its variants, including Dual-AC w/o multi-step, Dual-AC w/o path-regularization, Dual-AC w/o unbiased V , and the naive Dual-AC, for demonstrating the three instability sources in Section 3, respectively, as well as varying the k = {10, 50} in Dual-AC.

Specifically, Dual-AC w/o path-regularization removes the path-regularization components; Dual-AC w/o multi-step removes the multi-step extension and the path-regularization; Dual-AC w/o unbiased V calculates the stochastic gradient without achieving the convergence of inner optimization on V ; and the naive Dual-AC is the one without all components.

Moreover, Dual-AC with k = 10 and Dual-AC with k = 50 denote the length of steps set to be 10 and 50, respectively.

The empirical performances on InvertedDoublePendulum-v1, Swimmer-v1, and Hopper-v1 tasks are shown in FIG1 .

The results are consistent across the tasks with the analysis.

The naive Dual-AC performs the worst.

The performances of the Dual-AC found the optimal policy which solves the problem much faster than the alternative variants.

The Dual-AC w/o unbiased V converges slower, showing its sample inefficiency caused by the bias in gradient calculation.

The Dual-AC w/o multistep and Dual-AC w/o path-regularization cannot converge to the optimal policy, indicating the importance of the path-regularization in recovering the local duality.

Meanwhile, the performance of Dual-AC w/o multi-step is worse than Dual-AC w/o path-regularization, showing the bias in one- step can be alleviated via multi-step trajectories.

The performances of Dual-AC become better with the length of step k increasing on these three tasks.

We conjecture that the main reason may be that in these three MuJoCo environments, the bias dominates the variance.

Therefore, with the k increasing, the proposed Dual-AC obtains more accumulate rewards.

In this section, we evaluated the Dual-AC against TRPO and PPO across multiple tasks, including the InvertedDoublePendulum-v1, Hopper-v1, HalfCheetah-v1, Swimmer-v1 and Walker-v1.

These tasks have different dynamic properties, ranging from unstable to stable, Therefore, they provide sufficient benchmarks for testing the algorithms.

In Figure 2 , we reported the average rewards across 5 runs of each algorithm with 50% confidence interval during the training stage.

We also reported the average final rewards in TAB1 .

The proposed Dual-AC achieves the best performance in almost all environments, including Pendulum, InvertedDoublePendulum, Hopper, HalfCheetah and Walker.

These results demonstrate that Dual-AC is a viable and competitive RL algorithm for a wide spectrum of RL tasks with different dynamic properties.

A notable case is the InvertedDoublePendulum, where Dual-AC substantially outperforms TRPO and PPO in terms of the learning speed and sample efficiency, implying that Dual-AC is preferable to unstable dynamics.

We conjecture this advantage might come from the different meaning of V in our algorithm.

For unstable system, the failure will happen frequently, resulting the collected data are far away from the optimal trajectories.

Therefore, the policy improvement through the value function corresponding to current policy is slower, while our algorithm learns the optimal value function and enhances the sample efficiency.

In this paper, we revisited the linear program formulation of the Bellman optimality equation, whose Lagrangian dual form yields a game-theoretic view for the roles of the actor and the dual critic.

Although such a framework for actor and dual critic allows them to be optimized for the same objective function, parametering the actor and dual critic unfortunately induces instablity in optimization.

We analyze the sources of instability, which is corroborated by numerical experiments.

We then propose Dual Actor-Critic , which exploits stochastic dual ascent algorithm for the path regularized, DISPLAYFORM0 Figure 2: The results of Dual-AC against TRPO and PPO baselines.

Each plot shows average reward during training across 5 random seeded runs, with 50% confidence interval.

The x-axis is the number of training iterations.

The Dual-AC achieves comparable performances comparing with TRPO and PPO in some tasks, but outperforms on more challenging tasks.multi-step bootstrapping two-player game, to bypass these issues.

Proof We rewrite the linear programming 3 as DISPLAYFORM1 Recall the T is monotonic, i.e., if DISPLAYFORM2 Theorem 1 (Optimal policy from occupancy) s,a∈S×A ρ * (s, a) = 1, and π DISPLAYFORM3 a∈A ρ * (s,a) .

Proof For the optimal occupancy measure, it must satisfy DISPLAYFORM4 where P denotes the transition distribution and I denotes a |S| × |SA| matrix where I ij = 1 if and only if j ∈ [(i − 1) |A| + 1, . . .

, i |A|].

Multiply both sides with 1, due to µ and P are probabilities, we have 1, ρ * = 1.Without loss of generality, we assume there is only one best action in each state.

Therefore, by the KKT complementary conditions of (3), i.e., ρ(s, a) R(s, a) + γE s |s,a [V (s )] − V (s) = 0, which implies ρ * (s, a) = 0 if and only if a = a * , therefore, the π * by normalization.

Theorem 2 The optimal policy π * and its corresponding value function V * is the solution to the following saddle problem DISPLAYFORM5 Proof Due to the strong duality of the optimization (3), we have DISPLAYFORM6 Then, plugging the property of the optimum in Theorem 1, we achieve the final optimization (6).

In this section, we extend the linear programming and its duality to continuous state and action MDP.

In general, the only weak duality holds for infinite constraints, i.e., P * D * .

With a mild assumption, we will recover the strong duality for continuous state and action MDP, and most of the conclusions in discrete state and action MDP still holds.

Specifically, without loss of generality, we consider the solvable MDP, i.e., the optimal policy, π DISPLAYFORM0 where the first inequality comes from 2 f (x), g( DISPLAYFORM1 The constraints in the primal form of linear programming can be written as FORMULA1 , we have DISPLAYFORM2 DISPLAYFORM3 The solution (V * , * ) also satisfies the KKT conditions, DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 where denotes the conjugate operation.

By the KKT condition, we have DISPLAYFORM7 The strongly duality also holds, i.e., DISPLAYFORM8 s.t.( DISPLAYFORM9 Proof We compute the duality gap DISPLAYFORM10 which shows the strongly duality holds.

Once we establish the k-step Bellman optimality equation FORMULA9 , it is easy to derive the λ-Bellman optimality equation, i.e., DISPLAYFORM0 Proof Denote the optimal policy as π * (a|s), we have DISPLAYFORM1 holds for arbitrary ∀k ∈ N. Then, we conduct k ∼ Geo(λ) and take expectation over the countable infinite many equation, resulting DISPLAYFORM2 Next, we investigate the equivalent optimization form of the k-step and λ-Bellman optimality equation, which requires the following monotonic property of T k and T λ .Lemma 7 Both T k and T λ are monotonic.

Proof Assume U and V are the value functions corresponding to π 1 and π 2 , and U V , i.e., U (s) V (s), ∀s ∈ S, apply the operator T k on U and V , we have DISPLAYFORM3 , ∀π ∈ P, which leads to the first conclusion, DISPLAYFORM4 With the monotonicity of T k and T λ , we can rewrite the V * as the solution to an optimization,

The optimal value function V * is the solution to the optimization DISPLAYFORM0 where µ(s) is an arbitrary distribution over S. DISPLAYFORM1 where the last equality comes from the Banach fixed point theorem BID14 .

Similarly, we can also show that ∀V , V T ∞ λ V = V * .

By combining these two inequalities, we achieve the optimization.

We rewrite the optimization as min DISPLAYFORM2 (s, a) ∈ S × A, We emphasize that this optimization is no longer linear programming since the existence of maxoperator over distribution space in the constraints.

However, Theorem 1 still holds for the dual variables in (32).Proof Denote the optimal policy asπ * DISPLAYFORM3 the KKT condition of the optimization (29) can be written as DISPLAYFORM4 , we simplify the condition, i.e., DISPLAYFORM5 Due to the P π * V k (s |s, a) is a conditional probability for ∀V , with similar argument in Theorem 1, we have s,a ρ * (s, a) = 1.By the KKT complementary condition, the primal and dual solutions, i.e., V * and ρ * , satisfy DISPLAYFORM6 Recall V * denotes the value function of the optimal policy, then, based on the definition,π * V * = π * which denotes the optimal policy.

Then, the condition (30) implies ρ(s, a) = 0 if and only if a = a * , therefore, we can decompose ρ * (s, a) = α * (s)π * (a|s).The corresponding Lagrangian of optimization FORMULA1 is DISPLAYFORM7 where DISPLAYFORM8 We further simplify the optimization.

Since the dual variables are positive, we have After clarifying these properties of the optimization corresponding to the multi-step Bellman optimality equation, we are ready to prove the Theorem 3.Theorem 3 The optimal policy π * and its corresponding value function V * is the solution to the following saddle point problem max Proof By Theorem 1 in multi-step setting, we can decompose ρ(s, a) = α(s)π(a|s) without any loss.

Plugging such decomposition into the Lagrangian 32 and realizing the equivalence among the optimal policies, we arrive the optimization as min V max α∈P(S),π∈P(A) L k (V, α, π).

Then, because of the strong duality as we proved in Lemma 9, we can switch min and max operators in optimization 8 without any loss.

DISPLAYFORM9

The strong duality holds in optimization (8).Proof Specifically, for every α ∈ P(S), π ∈ P(A), DISPLAYFORM0 On the other hand, since L k (V, α * , π * ) is convex w.r.t.

V , we have V * ∈ argmin V L k (V, α * , π * ), by checking the first-order optimality.

Therefore, we have max α∈P(S),π∈P(A) (α, π) = max DISPLAYFORM1 Combine these two conditions, we achieve the strong duality even without convex-concave property (1 − γ k+1 )E s∼µ(s) [V * (s)] max α∈P(S),π∈P(A) (α, π) (1 − γ k+1 )E s∼µ(s) [V * (s)] .

We consider the one-step Lagrangian duality first.

Following the vanilla augmented Lagrangian method, one can achieve the dual function as The computation of P c is in general intractable due to the composition of max and the condition expectation in ∆[V ](s, a), which makes the optimization for augmented Lagrangian method difficult.

For the multi-step Lagrangian duality, the objective will become even more difficult due to constraints are on distribution family P(S) and P(A), rather than S × A.

The local duality holds for L r (V, α, π).

Denote (V * , α * , π * ) as the solution to Bellman optimality equation, with some appropriate η V , (V * , α * , π * ) = argmax α∈P(S),π∈P(A) argmin V L r (V, α, π).Proof The local duality can be verified by checking the Hessian of L r (θ V * ).

We apply the local duality theorem (Luenberger & Ye, 2015) [Chapter 14].

Suppose (Ṽ * ,α * ,π * ) is a local solution to min V max α∈P(S),π∈P(A) L r (V, α, π), then, max α∈P(S),π∈P(A) min V L r (V, α, π) has a local solutionṼ * with correspondingα * ,π * .Next, we show that with some appropriate η V , the path regularization does not change the optimum.

Let U π (s) = E

@highlight

We propose Dual Actor-Critic algorithm, which is derived in a principled way from the Lagrangian dual form of the Bellman optimality equation. The algorithm achieves the state-of-the-art performances across several benchmarks.
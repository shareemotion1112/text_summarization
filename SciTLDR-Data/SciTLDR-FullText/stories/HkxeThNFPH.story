We study continuous action reinforcement learning problems in which it is crucial that the agent interacts with the environment only through safe policies, i.e.,~policies that keep the agent in desirable situations, both during training and at convergence.

We formulate these problems as {\em constrained} Markov decision processes (CMDPs) and present safe policy optimization algorithms that are based on a Lyapunov approach to solve them.

Our algorithms can use any standard policy gradient (PG) method, such as deep deterministic policy gradient (DDPG) or proximal policy optimization (PPO), to train a neural network policy, while guaranteeing near-constraint satisfaction for every policy update by projecting either the policy parameter or the selected action onto the set of feasible solutions induced by the state-dependent linearized Lyapunov constraints.

Compared to the existing constrained PG algorithms, ours are more data efficient as they are able to utilize both on-policy and off-policy data.

Moreover, our action-projection algorithm often leads to less conservative policy updates and allows for natural integration into an end-to-end PG training pipeline.

We evaluate our algorithms and compare them with the state-of-the-art baselines on several simulated (MuJoCo) tasks, as well as a real-world robot obstacle-avoidance problem, demonstrating their effectiveness in terms of balancing performance and constraint satisfaction.

The field of reinforcement learning (RL) has witnessed tremendous success in many high-dimensional control problems, including video games (Mnih et al., 2015) , board games (Silver et al., 2016) , robot locomotion (Lillicrap et al., 2016) , manipulation (Levine et al., 2016; Kalashnikov et al., 2018) , navigation (Faust et al., 2018) , and obstacle avoidance (Chiang et al., 2019) .

In RL, the ultimate goal is to optimize the expected sum of rewards/costs, and the agent is free to explore any behavior as long as it leads to performance improvement.

Although this freedom might be acceptable in many problems, including those involving simulators, and could expedite learning a good policy, it might be harmful in many other problems and could cause damage to the agent (robot) or to the environment (objects or people nearby).

In such domains, it is absolutely crucial that while the agent optimizes long-term performance, it only executes safe policies both during training and at convergence.

A natural way to incorporate safety is via constraints.

A standard model for RL with constraints is constrained Markov decision process (CMDP) (Altman, 1999) , where in addition to its standard objective, the agent must satisfy constraints on expectations of auxiliary costs.

Although optimal policies for finite CMDPs with known models can be obtained by linear programming (Altman, 1999) , there are not many results for solving CMDPs when the model is unknown or the state and/or action spaces are large or infinite.

A common approach to solve CMDPs is to use the Lagrangian method (Altman, 1998; Geibel & Wysotzki, 2005) , which augments the original objective function with a penalty on constraint violation and computes the saddle-point of the constrained policy optimization via primal-dual methods (Chow et al., 2017) .

Although safety is ensured when the policy converges asymptotically, a major drawback of this approach is that it makes no guarantee with regards to the safety of the policies generated during training.

A few algorithms have been recently proposed to solve CMDPs at scale while remaining safe during training.

One such algorithm is constrained policy optimization (CPO) (Achiam et al., 2017) .

CPO extends the trust-region policy optimization (TRPO) algorithm (Schulman et al., 2015a) to handle the constraints in a principled way and has shown promising empirical results in terms scalability, performance, and constraint satisfaction, both during training and at convergence.

Another class of these algorithms is by Chow et al. (Chow et al., 2018) .

These algorithms use the notion of Lyapunov functions that have a long history in control theory to analyze the stability of dynamical systems (Khalil, 1996) .

Lyapunov functions have been used in RL to guarantee closed-loop stability (Perkins & Barto, 2002; Faust et al., 2014) .

They also have been used to guarantee that a model-based RL agent can be brought back to a "region of attraction" during exploration (Berkenkamp et al., 2017) .

Chow et al. (Chow et al., 2018) use the theoretical properties of the Lyapunov functions and propose safe approximate policy and value iteration algorithms.

They prove theories for their algorithms when the CMDP is finite with known dynamics, and empirically evaluate them in more general settings.

However, their algorithms are value-function-based, and thus are restricted to discrete-action domains.

In this paper, we build on the problem formulation and theoretical findings of the Lyapunov-based approach to solve CMDPs, and extend it to tackle continuous action problems that play an important role in control theory and robotics.

We propose Lyapunov-based safe RL algorithms that can handle problems with large or infinite action spaces, and return safe policies both during training and at convergence.

To do so, there are two major difficulties that need to be addressed: 1) the policy update becomes an optimization problem over the large or continuous action space (similar to standard MDPs with large actions), and 2) the policy update is a constrained optimization problem in which the (Lyapunov) constraints involve integration over the action space, and thus, it is often impossible to have them in closed-form.

Since the number of Lyapunov constraints is equal to the number of states, the situation is even more challenging when the problem has a large state space.

To address the first difficulty, we switch from value-function-based to policy gradient (PG) algorithms.

To address the second difficulty, we propose two approaches to solve our constrained policy optimization problem (a problem with infinite constraints, each involving an integral over the continuous action space) that can work with any standard on-policy (e.g., proximal policy optimization (PPO) (Schulman et al., 2017) ) and off-policy (e.g., deep deterministic policy gradient (DDPG) (Lillicrap et al., 2016) ) PG algorithm.

Our first approach, which we call policy parameter projection or θ-projection, is a constrained optimization method that combines PG with a projection of the policy parameters onto the set of feasible solutions induced by the Lyapunov constraints.

Our second approach, which we call action projection or a-projection, uses the concept of a safety layer introduced by (Dalal et al., 2018) to handle simple single-step constraints, extends this concept to general trajectorybased constraints, solves the constrained policy optimization problem in closed-form using Lyapunov functions, and integrates this closed-form into the policy network via safety-layer augmentation.

Since both approaches guarantee safety at every policy update, they manage to maintain safety throughout training (ignoring errors resulting from function approximation), ensuring that all intermediate policies are safe to be deployed.

To prevent constraint violations due to function approximation errors, similar to CPO, we offer a safeguard policy update rule that decreases constraint cost and ensures near-constraint satisfaction.

Our proposed algorithms have two main advantages over CPO.

First, since CPO is closely connected to TRPO, it can only be trivially combined with PG algorithms that are regularized with relative entropy, such as PPO.

This restricts CPO to on-policy PG algorithms.

On the contrary, our algorithms can work with any on-policy (e.g., PPO) and off-policy (e.g., DDPG) PG algorithm.

Having an off-policy implementation is beneficial, since off-policy algorithms are potentially more data-efficient, as they can use the data from the replay buffer.

Second, while CPO is not a back-propagatable algorithm, due to the backtracking line-search procedure and the conjugate gradient iterations for computing natural gradient in TRPO, our algorithms can be trained end-to-end, which is crucial for scalable and efficient implementation (Hafner et al., 2017) .

In fact, we show in Section 3.1 that CPO (minus the line search) can be viewed as a special case of the on-policy version (PPO version) of our θ-projection algorithm, corresponding to a specific approximation of the constraints.

We evaluate our algorithms and compare them with CPO and the Lagrangian method on several continuous control (MuJoCo) tasks and a real-world robot navigation problem, in which the robot must satisfy certain constraints, while minimizing its expected cumulative cost.

Results show that our algorithms outperform the baselines in terms of balancing the performance and constraint satisfaction (during training), and generalize better to new and more complex environments.

We consider the RL problem in which the agent's interaction with the environment is modeled as a Markov decision process (MDP).

A MDP is a tuple (X , A, γ, c, P, x0), where X and A are the state and action spaces; γ ∈ [0, 1) is a discounting factor; c(x, a) ∈ [0, Cmax] is the immediate cost function; P (·|x, a) is the transition probability distribution; and x0 ∈ X is the initial state.

Although we consider deterministic initial state and cost function, our results can be easily generalized to random initial states and costs.

We model the RL problems in which there are constraints on the cumulative cost using CMDPs.

The CMDP model extends MDP by introducing additional costs and the associated constraints, and is defined by (X , A, γ, c, P, x0, d, d0) , where the first six components are the same as in the unconstrained MDP; d(x) ∈ [0, Dmax] is the (state-dependent) immediate constraint cost; and d0 ∈ R ≥0 is an upper-bound on the expected cumulative constraint cost.

To formalize the optimization problem associated with CMDPs, let ∆ be the set of Markovian stationary policies, i.e., ∆ = {π : X × A → [0, 1], a π(a|x) = 1}. At each state x ∈ X , we define the generic Bellman operator w.r.t.

a policy π ∈ ∆ and a cost function h as

.

Given a policy π ∈ ∆, we define the expected cumulative cost and the safety constraint function (expected cumulative constraint cost) as Cπ(x0)

], respectively.

The safety constraint is then defined as Dπ(x0) ≤ d0.

The goal in CMDPs is to solve the constrained optimization problem

It has been shown that if the feasibility set is non-empty, then there exists an optimal policy in the class of stationary Markovian policies ∆ (Altman, 1999, Theorem 3.1).

2.1 POLICY GRADIENT ALGORITHMS Policy gradient (PG) algorithms optimize a policy by computing a sample estimate of the gradient of the expected cumulative cost induced by the policy, and then updating the policy parameter in the gradient direction.

In general, stochastic policies that give a probability distribution over actions are parameterized by a κ-dimensional vector θ, so the space of policies can be written as {π θ , θ ∈ Θ ⊂ R κ }.

Since in this setting a policy π is uniquely defined by its parameter θ, policy-dependent functions can be written as a function of θ or π interchangeably.

DDPG (Lillicrap et al., 2016) and PPO (Schulman et al., 2017) are two PG algorithms that have recently gained popularity in solving continuous control problems.

DDPG is an off-policy Q-learning style algorithm that jointly trains a deterministic policy π θ (x) and a Q-value approximator Q(x, a; φ).

The Q-value approximator is trained to fit the true Q-value function and the deterministic policy is trained to optimize Q(x, π θ (x); φ) via chain-rule.

The PPO algorithm we use in this paper is a penalty form of TRPO (Schulman et al., 2015a) with an adaptive rule to tune the DKL penalty weight β k .

PPO trains a policy π θ (x) by optimizing a loss function that consists of the standard policy gradient objective and a penalty on the KL-divergence between the current θ and previous θ policies, i.e., DKL(θ, θ )

Lagrangian method is a straightforward way to address the constraint Dπ θ (x0) ≤ d0 in CMDPs.

Lagrangian method adds the constraint costs d(x) to the task costs c(x, a) and transform the constrained optimization problem to a penalty form, i.e., min θ∈Θ max λ≥0 E[

The method then jointly optimizes θ and λ to find a saddle-point of the penalized objective.

The optimization of θ may be performed by any PG algorithm on the augmented cost c(x, a) + λd(x), while λ is optimized by stochastic gradient descent.

As described in Sec. 1, although the Lagrangian approach is easy to implement (see Appendix A for the details), in practice, it often violates the constraints during training.

While at each step during training, the objective encourages finding a safe solution, the current value of λ may lead to an unsafe policy.

This is why the Lagrangian method may not be suitable for solving problems in which safety is crucial during training.

Since in this paper, we extend the Lyapunov-based approach to CMDPs of (Chow et al., 2018) to PG algorithms, we end this section by introducing some terms and notations from (Chow et al., 2018) that are important in developing our safe PG algorithms.

We refer readers to Appendix B for details.

We define a set of Lyapunov functions w.r.t.

initial state x0 ∈ X and constraint threshold d0 as

, where πB is a feasible policy of (1), i.e., Dπ B (x0) ≤ d0.

We refer to the constraints in this feasibility set as the Lyapunov constraints.

For an arbitrary Lyapunov function L ∈ Lπ B (x0, d0), we denote by

, ∀x ∈ X , the set of L-induced Markov stationary policies.

The contraction property of T π,d , together with L(x0) ≤ d0, imply that any L-induced policy in FL is a feasible policy of (1).

However, FL(x) does not always contain an optimal solution of (1), and thus, it is necessary to design a Lyapunov function that provides this guarantee.

In other words, the main goal of the Lyapunov approach is to construct a Lyapunov function L ∈ Lπ B (x0, d0), such that FL contains an optimal policy π Chow et al. (2018) show in their Theorem 1 that without loss of optimality, the Lyapunov function that satisfies the above criterion can be expressed as Lπ B , (x) := E ∞ t=0 γ t d(xt) + (xt) | πB, x , in which (x) ≥ 0 is a specific immediate auxiliary constraint cost that keeps track of the maximum constraint budget available for policy improvement (from πB to π * ).

They propose ways to construct such , as well as an auxiliary constraint cost surrogate , which is a tight upper-bound on and can be computed more efficiently.

They use this construction to propose the safe (approximate) policy and value iteration algorithms, whose objective is to solve the following LP (Chow et al., 2018, Eq. 6 ) during policy improvement:

where x ) are the value and state-action value functions (w.r.t.

the cost function c), and

is the Lyapunov function.

In any iterative policy optimization method, such as those studied in this paper, the feasible policy πB at each iteration can be set to the policy computed at the previous iteration (which is feasible).

In LP (2), there are as many constraints as the number of states and each constraint involves an integral over the entire action space.

When the state space is large, even if the integral in the constraint has a closed-form (e.g., for finite actions), solving (2) becomes numerically intractable.

Chow et al. (Chow et al., 2018) assumed that the number of actions is finite and focused on value-function-based RL algorithms, and addressed the large state issue by policy distillation.

Since in this paper, we are interested in problems with large action spaces, solving (2) will be even more challenging.

To address this issue, in the next section, we first switch from value-function-based algorithms to PG algorithms, then propose an optimization problem with Lyapunov constraints, analogous to (2), that is suitable for PG, and finally present two methods to solve our proposed optimization problem efficiently.

We now present our approach to solve CMDPs in a way that guarantees safety both at convergence and during training.

Similar to (Chow et al., 2018) , our Lyapunov-based safe PG algorithms solve a constrained optimization problem analogous to (2).

In particular, our algorithms consist of two components, a baseline PG algorithm, such as DDPG or PPO, and an effective method to solve the general Lyapunov-based policy optimization problem, the analogous to (2), i.e,

In the next two sections, we present two approaches to solve (3) efficiently.

We call these approaches 1) θ-projection, a constrained optimization method that combines PG with projecting the policy parameter θ onto the set of feasible solutions induced by the Lyapunov constraints, and 2) a-projection, in which we embed the Lyapunov constraints into the policy network via a safety layer.

3.1 THE θ-PROJECTION APPROACH The θ-projection approach is based on the minorization-maximization technique in conservative PG (Kakade & Langford, 2002) and Taylor series expansion, and can be applied to both onpolicy and off-policy algorithms.

Following Theorem 4.1 in (Kakade & Langford, 2002) , we first have the following bound for the cumulative cost:

, where µ θ B ,x 0 is the γ-visiting distribution of π θ B starting at the initial state x0, and β is the weight for the entropy-based regularization.

1 Using this result, we denote by

the surrogate cumulative cost.

It has been shown in Eq. 10 of (Schulman et al., 2015a) that replacing the objective function Cπ θ (x0) with its surrogate C π θ (x0; π θ B ) in solving (3) will still lead to policy improvement.

In order to effectively compute the improved policy parameter θ+, one further approximates the function C π θ (x0; π θ B ) with its Taylor series expansion around θB. In particular, the term

is approximated up to its first order, and the term DKL(θ, θB) is approximated up to its second order.

These altogether allow us to replace the objective function in (3)

Similarly, regarding the constraints in (3), we can use the Taylor series expansion (around θB) to approximate the LHS of the Lyapunov constraints as

Using the above approximations, at each iteration, our safe PG algorithm updates the policy by solving the following constrained optimization problem with semi-infinite dimensional Lyapunov constraints:

the above max-operator is non-differentiable, this may still lead to numerical instability in gradient descent algorithms.

Similar to the surrogate constraint in TRPO (to transform the max D KL constraint to an average DKL constraint, see Eq. 12 in (Schulman et al., 2015a) ), a more numerically stable way is to approximate the Lyapunov constraint using the average constraint surrogate

where M is the number of on-policy sample trajectories of π θ B .

In order to effectively compute the gradient of the Lyapunov value function, consider the special case when the auxiliary constraint surrogate is chosen as = (1 − γ)(d0 − Dπ θ B (x0)) (see Appendix B for justification).

Using the fact that is θ-independent, the gradient term in (5) can be written as

are the constraint value functions, respectively.

Since the integral is equal to E a∼π θ [Q W θ B (x i , a)], the average constraint surrogate (5) can be approximated (approximation is because of the choice of ) by the inequality Dπ θ B (x0) +

, which is equivalent to the constraint used in CPO (see Section 6.1 in (Achiam et al., 2017) ).

This shows that CPO (minus the line search) belongs to the class of our Lyapunov-based PG algorithms with θ-projection.

We refer to the DDPG and PPO versions of our θ-projection safe PG algorithms as SDDPG and SPPO.

Derivation details and the pseudo-code (Algorithm 4) of these algorithms are given in Appendix C.

The main characteristic of the Lyapunov approach is to break down a trajectory-based constraint into a sequence of single-step state dependent constraints.

However, when the state space is infinite, the feasibility set is characterized by infinite dimensional constraints, and thus, it is counter-intuitive to directly enforce these Lyapunov constraints (as opposed to the original trajectory-based constraint) into the policy update optimization.

To address this, we leverage the idea of a safety layer from (Dalal et al., 2018) , that was applied to simple single-step constraints, and propose a novel approach to embed the set of Lyapunov constraints into the policy network.

This way, we reformulate the CMDP problem (1) as an unconstrained optimization problem and optimize its policy parameter θ (of the augmented network) using any standard unconstrained PG algorithm.

At every given state, the unconstrained action is first computed and then passed through the safety layer, where a feasible action mapping is constructed by projecting unconstrained actions onto the feasibility set w.r.t.

Lyapunov constraints.

This constraint projection approach can guarantee safety during training.

We now describe how the action mapping (to the set of Lyapunov constraints) works 2 .

Recall from the policy improvement problem in (3) that the Lyapunov constraint is imposed at every state x ∈ X .

Given a baseline feasible policy πB = π θ B , for any arbitrary policy parameter θ ∈ Θ, we denote by Ξ(πB, θ) = {θ ∈ Θ : QL π B (x, π θ (x)) − QL π B (x, πB(x)) ≤ (x), ∀x ∈ X }, the projection of θ onto the feasibility set induced by the Lyapunov constraints.

One way to construct a feasible policy π Ξ(π B ,θ) from a parameter θ is to solve the following 2 -projection problem:

We refer to this operation as the Lyapunov safety layer.

Intuitively, this projection perturbs the unconstrained action as little as possible in the Euclidean norm in order to satisfy the Lyapunov constraints.

Since this projection guarantees safety, if we have access to a closed form of the projection, we may insert it into the policy parameterization and simply solve an unconstrained policy optimization problem, i.e., θ+ ∈ arg min θ∈Θ Cπ Ξ(π B ,θ) (x0), using any standard PG algorithm.

To simplify the projection (6), we can approximate the LHS of the Lyapunov constraint with its first-order Taylor series (w.r.t.

action a = πB(x)).

Thus, at any given state x ∈ X , the safety layer solves the following projection problem:

where η(x) ∈ [0, 1) is the mixing parameter that controls the trade-off between projecting on unconstrained policy (for return maximization) and on baseline policy (for safety), and

is the action-gradient of the state-action Lyapunov function.

Similar to the analysis of Section 3.1, if the auxiliary cost is state-independent, one can readily find gL π B (x) by computing the gradient of the constraint action-value function ∇aQW θ B (x, a) | a=π B (x) .

Note that the objective function in (7) is positive-definite and quadratic, and the constraint approximation is linear.

Therefore, the solution of this (convex) projection problem can be effectively computed by an in-graph QP-solver, such as OPT-Net (Amos & Kolter, 2017) .

Combined with the above projection procedure, this further implies that the CMDP problem can be effectively solved using an end-to-end PG training pipeline (such as DDPG or PPO).

When the CMDP has a single constraint (and thus a single Lyapunov constraint), the policy π Ξ(π B ,θ) (x) has the following analytical solution.

Proposition 1.

At any given state x ∈ X , the solution to the optimization problem (7) has the form

The closed-form solution is essentially a linear projection of the unconstrained action π θ (x) onto the Lyapunov-safe hyper-plane with slope gL π B (x) and intercept (x) = (1 − γ)(d0 − Dπ B (x0)).

It is possible to extend this closed-form solution to handle multiple constraints, if there is at most one constraint active at a time (see Proposition 1 in (Dalal et al., 2018) ).We refer to the DDPG and PPO versions of our a-projection safe Lyapunov-based PG algorithms as SDDPG a-projection and SPPO a-projection.

Derivation and pseudo-code (Algorithm 5) of these algorithms are in Appendix C.

We empirically evaluate 3 our Lyapunov-based safe PG algorithms to assess their: (i) performance in terms of cost and safety during training, and (ii) robustness w.r.t.

constraint violation.

We use three simulated robot locomotion continuous control tasks in the MuJoCo simulator (Todorov et al., 2012) .

The notion of safety in these tasks is motivated by physical constraints: (i) HalfCheetah-Safe: this is a modification of the MuJoCo HalfCheetah problem in which we impose constraints on the speed of Cheetah in order to force it to run smoothly.

The video shows that the policy learned by our algorithm results in slower but much smoother movement of Cheetah compared to the policies learned by PPO and Lagrangian 4 ; (ii) Point-Circle: the agent is rewarded for running in a wide circle, but is constrained to stay within a safe region defined by |x| ≤ x lim ; (iii) Point-Gather & Ant-Gather: the agent is rewarded for collecting target objects in a terrain map, while being constrained to avoid bombs.

The last two tasks were first introduced in (Achiam et al., 2017) by adding constraints to the original MuJoCo tasks: Point and Ant.

Details of these tasks are given in Appendix D.

We compare our algorithms with two state-of-the-art unconstrained algorithms, DDPG and PPO, and two constrained methods, Lagrangian with optimized Lagrange multiplier (Appendix A) and on-policy CPO.

We use the CPO algorithm that is based on PPO (unlike the original CPO that is based on TRPO) and coincides with our SPPO algorithm derived in Section 4.1.

SPPO preserves the essence of CPO by adding the first-order constraint and relative entropy regularization to the policy optimization problem.

The main difference between CPO and SPPO is that the latter does not perform backtracking line-search in learning rate.

We compare with SPPO instead of CPO to 1) avoid the additional computational complexity of line-search in TRPO, while maintaining the performance of PG using PPO, 2) have a back-propagatable version of CPO, and 3) have a fair comparison with other back-propagatable safe PG algorithms, such as our DDPG and a-projection based algorithms.

Figures 1a, 1b , 2a, 2b, 8a, 8b, 9a, 9b show that our Lyapunov-based PG algorithms are stable in learning and all converge to feasible policies with reasonable performance.

Figures 1c, 1d , 2c, 2d, 8c, 8d, 9c, 9b show the algorithms in terms of constraint violation during training.

These figures indicate that our algorithms quickly stabilize the constraint cost below the threshold, while the unconstrained DDPG and PPO violate the constraints, and Lagrangian tends to jiggle around the threshold.

Moreover, it is worth-noting that the Lagrangian method can be sensitive to the initialization of the Lagrange multiplier λ 0 .

If λ 0 is too large, it would make policy updates overly conservative, and if it is too small, then we will have more constraint violation.

Without further knowledge about the environment, we treat λ 0 as a hyper-parameter and optimize it via grid-search.

See Appendix D for more details and for the experimental results of Ant-Gather and Point-Circle.

a-projection vs. θ-projection: The figures indicate that in many cases DDPG and PPO with aprojection converge faster and have lower constraint violation than their θ-projection counterparts (i.e., SDDPG and SPPO).

This corroborates with the hypothesis that a-projection is less conservative during policy updates than θ-projection (which is what CPO is based on) and generates smoother gradient updates during end-to-end training.

In most experiments (HalfCheetah, PointGather, and AntGather) the DDPG algorithms tend to have faster learning than their PPO counterparts, while the PPO algorithms perform better in terms of constraint satisfaction.

The faster learning behavior is due to the improved dataefficiency when using off-policy samples in PG, however, the covariate-shift 5 in off-policy data makes tight constraint control more challenging.

We now evaluate safe policy optimization algorithms on a real robot task -a map-less navigation task (Chiang et al., 2019) -where a noisy differential drive robot with limited sensors (Fig. 3a) is required to navigate to a goal outside of its field of view in unseen environments while avoiding collision.

The main goal is to learn a policy that drives the robot to goal as efficiently as possible, while limiting the impact energy of collisions, since the collision can damage the robot and environment.

Here the CMDP is non-discounting and has a fixed horizon.

The agent's observations consist of the relative goal position, agent's velocity, and Lidar measurements (Fig. 3a) .

The actions are the linear and angular velocity at the robot's center of the mass.

6 The transition probability captures the noisy robot's dynamics, whose exact formulation is unknown to the robot.

The robot must navigate to arbitrary goal positions collision-free in a previously unseen environment, and without access to the indoor map and any work-space topology.

We reward the agent for reaching the goal, which translates to an immediate cost that measures the relative distance to the goal.

To measure the total impact energy of obstacle collisions, we impose an immediate constraint cost to account for the speed during collision, with a constraint threshold d 0 that characterizes the agent's maximum tolerable collision impact energy to any object.

Different from the standard approach, where a constraint on collision speed is explicitly imposed to the learning problem at each time step, we emphasize that a CMDP constraint is required here because it allows the robot to lightly brush off the obstacle (such as walls) but prevent it from ramming into any objects.

Other use cases of CMDP constraints in robot navigation include collision avoidance (Pfeiffer et al., 2018) or limiting total battery usage of the task.

Experimental Results: We evaluate the learning algorithms on success rate and constraint control averaged over 100 episodes with random initialization.

The task is successful if the robot reaches the goal before the constraint threshold (total energy of collision) is exhausted.

While all methods converge to policies with reasonable performance, Figure 4a and 4b show that the Lyapunov-based PG algorithms have higher success rates, due to their robust abilities of controlling the total constraint, as well minimizing the distance to goal.

Although the unconstrained method often yields a lower distance to goal, it violates the constraint more frequently leading to a lower success rate.

Lagrangian approach is less robust to initialization of parameters, and therefore it generally has lower success rate and higher variability than the Lyapunov-based methods.

Unfortunately due to function approximation error and stochasticity of the problem, all the algorithms converged pre-maturely with constraints above the threshold, possibly due to the overly conservative constraint threshold (d 0 = 100).

Inspection of trajectories shows that the Lagrangian method tends to zigzag and has more collisions, while the SDDPG chooses a safer path to reach the goal (Figures 5a and 5b ).

Next, we evaluate how well the methods generalize to (i) longer trajectories, and (ii) new environments.

The tasks are trained in a 22 by 18 meters environment ( Fig. 7) with goals placed within 5 to 10 meters from the robot initial state.

In a much larger evaluation environment (60 by 47 meters) with goals placed up to 15 meters away from the goal, the success rate of all methods degrades as the goals are further away (Fig. 6a) .

The safety methods (a-projection -SL-DDPG, and θ-projection -SG-DDPG) outperform unconstrained and Lagrangian (DDPG and LA-DDPG), while retaining the lower constraints even when the task becomes more difficult (Fig. 6b) .

Finally, we deployed the SL-DDPG policy onto the real Fetch robot (Wise et al., 2016) in an everyday office environment.

7 Fetch robot weights 150 kilograms, and reaches maximum speed of 7 km/h making the collision force a safety paramount.

Figure 5c shows the top down view of the robot log.

Robot travelled, through narrow corridors and around people walking through the office, for a total of 500 meters to complete five repetitions of 12 tasks, each averaging about 10 meters to the goal.

The robot robustly avoids both static and dynamic (humans) obstacles coming into its path.

We observed additional "wobbling" effects, that was not present in simulation.

This is likely due to the wheel slippage at the floor that the policy was not trained for.

In several occasions when the robot could not find a clear path, the policy instructed the robot to stay put instead of narrowly passing by the obstacle.

This is precisely the safety behavior we want to achieve with the Lyapunov-based algorithms.

We used the notion of Lyapunov functions and developed a class of safe RL algorithms for continuous action problems.

Each algorithm in this class is a combination of one of our two proposed projections:

θ-projection and a-projection, with any on-policy (e.g., PPO) or off-policy (e.g., DDPG) PG algorithm.

We evaluated our algorithms on four high-dimensional simulated robot locomotion MuJoCo tasks and compared them with several baselines.

To demonstrate the effectiveness of our algorithms in solving real-world problems, we also applied them to an indoor robot navigation problem, to ensure that the robot's path is optimal and collision-free.

Our results indicate that our algorithms 1) achieve safe learning, 2) have better data-efficiency, 3) can be more naturally integrated within the standard end-to-end differentiable PG training pipeline, and 4) are scalable to tackle real-world problems.

Our work is a step forward in deploying RL to real-world problems in which safety guarantees are of paramount importance.

Future work includes 1) extending a-projection to stochastic policies and 2) extensions of the Lyapunov approach to model-based RL and use it for safe exploration.

We first state a number of mild technical and notational assumptions that we make throughout this section.

Assumption 1 (Differentiability).

For any state-action pair (x, a), π θ (a|x) is continuously differentiable in θ and ∇ θ π θ (a|x) is a Lipschitz function in θ for every x ∈ X and a ∈ A. Assumption 2 (Strict Feasibility).

There exists a transient policy π θ (·|x) such that D π θ (x 0 ) < d 0 in the constrained problem.

Assumption 3 (Step Sizes).

The step size schedules {α 3,k }, {α 2,k }, and {α 1,k } satisfy

Assumption 1 imposes smoothness on the optimal policy.

Assumption 2 guarantees the existence of a local saddle point in the Lagrangian analysis.

Assumption 3 refers to step sizes corresponding to policy updates and indicates that the update corresponding to {α 3,k } is on the fastest time-scale, the updates corresponding to {α 2,k } is on the intermediate time-scale, and the update corresponding to {α 1,k } is on the slowest time-scale.

As this assumption refers to user-defined parameters, they can always be chosen to be satisfied.

To solve the CMDP, we employ the Lagrangian relaxation procedure (Bertsekas, 1999) to convert it to the following unconstrained problem:

where λ is the Lagrange multiplier.

Notice that L(θ, λ) is a linear function in λ.

Then, there exists a local saddle point (θ * , λ * ) for the minimax optimization problem max λ≥0 min θ L(θ, λ), such that for some r > 0, ∀θ ∈ R κ ∩ B θ * (r), and ∀λ ∈ [0, λ max ], we have

where B θ * (r) is a hyper-dimensional ball centered at θ * with radius r > 0.

In the following, we present a policy gradient (PG) algorithm and an actor-critic (AC) algorithm.

While the PG algorithm updates its parameters after observing several trajectories, the AC algorithm is incremental and updates its parameters at each time-step.

We now present a policy gradient algorithm to solve the optimization problem (11).

The idea of the algorithm is to descend in θ and ascend in λ using the gradients of L(θ, λ) w.r.t.

θ and λ, i.e.,

The unit of observation in this algorithm is a system trajectory generated by following the current policy π θ k .

At each iteration, the algorithm generates N trajectories by following the current policy π θ k , uses them to estimate the gradients in (13), and then uses these estimates to update the parameters θ, λ.

Let ξ = {x 0 , a 0 , c 0 , x 1 , a 1 , c 1 , . . .

, x T −1 , a T −1 , c T −1 , x T } be a trajectory generated by following the policy θ, where x T = x Tar is the target state of the system and T is the (random) stopping time.

The cost, constraint cost, and probability of ξ are defined as C(ξ) =

respectively.

Based on the definition of P θ (ξ), one obtains ∇ θ log P θ (ξ) = T −1 k=0 ∇ θ log π θ (a k |x k ).

Algorithm 1 contains the pseudo-code of our proposed PG algorithm.

What appears inside the parentheses on the right-hand-side of the update equations are the estimates of the gradients of L(θ, λ) w.r.t.

θ, λ (estimates of the expressions in (13)).

Gradient estimates of the Lagrangian function are given by

Input: parameterized policy π(·|·; θ) Initialization: policy parameter θ = θ 0 , and the Lagrangian parameter λ = λ 0 for i = 0, 1, 2, . . .

do for j = 1, 2, . . .

do Generate N trajectories {ξ j,i } N j=1 by starting at x 0 and following the policy θ i . end for

end for where the likelihood gradient is

2 , which ensures the convergence of the algorithm.

Recall from Assumption 3 that the step-size schedules satisfy the standard conditions for stochastic approximation algorithms, and ensure that the policy parameter θ update is on the fast time-scale {α 2,i }, and the Lagrange multiplier λ update is on the slow time-scale {α 1,i }.

This results in a two time-scale stochastic approximation algorithm that has been shown to converge to a (local) saddle point of the objective function L(θ, λ).

This convergence proof makes use of standard results in stochastic approximation theory, because in the limit when the step-size is sufficiently small, analyzing the convergence of PG is equivalent to analyzing the stability of an ordinary differential equation (ODE) w.r.t.

its equilibrium point.

In PG, the unit of observation is a system trajectory.

This may result in high variance for the gradient estimates, especially when the length of the trajectories is long.

To address this issue, we propose two actor-critic algorithms that use value function approximation in the gradient estimates and update the parameters incrementally (after each state-action transition).

We present two actor-critic algorithms for optimizing (11).

These algorithms are still based on the above gradient estimates.

Algorithm 2 contains the pseudo-code of these algorithms.

The projection operator Γ Λ is necessary to ensure the convergence of the algorithms.

Recall from Assumption 3 that the step-size schedules satisfy the standard conditions for stochastic approximation algorithms, and ensure that the critic update is on the fastest time-scale α 3,k , the policy update α 2,k is on the intermediate timescale, and finally the Lagrange multiplier update is on the slowest time-scale α 1,k .

This results in three time-scale stochastic approximation algorithms.

Using the PG theorem from (Sutton et al., 2000) , one can show that

where µ θ is the discounted visiting distribution and Q θ is the action-value function of policy θ.

We can show that

, where

is the temporal-difference (TD) error, andV θ is an estimator of the value function V θ .

Traditionally, for convergence guarantees in actor-critic algorithms, the critic uses linear approximation for the value function

, where the feature vector ψ(·) belongs to a low-dimensional space R κ2 .

The linear approximationV θ,v belongs to a low-dimensional subspace Input: Parameterized policy π(·|·; θ) and value function feature vector φ(·) Initialization: policy parameters θ = θ0; Lagrangian parameter λ = λ0; value function weight v = v0

// NAC Algorithm: Critic Update:

, where Ψ is a short-hand notation for the set of features, i.e., Ψ(x) = ψ (x).

Recently with the advances in deep neural networks, it has become increasingly popular to model the critic with a deep neural network, based on the objective function of minimizing the MSE of Bellman residual w.r.t.

V θ or Q θ (Mnih et al., 2013) .

In this section, we revisit the Lyapunov approach to solving CMDPs that was proposed by (Chow et al., 2018) and report the mathematical results that are important in developing our safe policy optimization algorithms.

To start, without loss of generality, we assume that we have access to a baseline feasible policy of (1), π B ; i.e., π B satisfies D π B (x 0 ) ≤ d 0 .

We define a set of Lyapunov functions w.r.t.

initial state x 0 ∈ X and constraint threshold d 0 as

and call the constraints in this feasibility set the Lyapunov constraints.

For any arbitrary Lyapunov function L ∈ L π B (x 0 , d 0 ), we denote by

the set of L-induced Markov stationary policies.

Since T π,d is a contraction mapping (Bertsekas, 2005) , any L-induced policy π has the property

, ∀x ∈ X .

Together with the property that L(x 0 ) ≤ d 0 , they imply that any L-induced policy is a feasible policy of (1).

However, in general, the set F L (x) does not necessarily contain an optimal policy of (1), and thus, it is necessary to design a Lyapunov function (w.r.t.

a baseline policy π B ) that provides this guarantee.

In other words, the main goal is to construct a Lyapunov function

(21) Chow et al. (Chow et al., 2018) show in their Theorem 1 that 1) without loss of optimality, the Lyapunov function can be expressed as

where (x) ≥ 0 is some auxiliary constraint cost uniformly upper-bounded by

and 2) if the baseline policy π B satisfies the condition

where D = max x∈X max π D π (x) is the maximum constraint cost, then the Lyapunov function candidate L * also satisfies the properties of (21), and thus, its induced feasible policy set F L * contains an optimal policy.

Furthermore, suppose that the distance between the baseline and optimal policies can be estimated efficiently.

Using the set of L * -induced feasible policies and noting that the safe Bellman operator

, ∀x ∈ X , has a unique fixed point V * , such that V * (x 0 ) is a solution of (1) and an optimal policy can be constructed via greedification, i.e., π

.

This shows that under the above assumption, (1) can be solved using standard dynamic programming (DP) algorithms.

While this result connects CMDP with Bellman's principle of optimality, verifying whether π B satisfies this assumption is challenging when a good estimate of D T V (π * ||π B ) is not available.

To address this issue, Chow et al. (Chow et al., 2018) propose to approximate * with an auxiliary constraint cost , which is the largest auxiliary cost satisfying the Lyapunov condition

, ∀x ∈ X , and the safety condition L (x 0 ) ≤ d 0 .

The intuition here is that the larger , the larger the set of policies F L .

Thus, by choosing the largest such auxiliary cost, we hope to have a better chance of including the optimal policy π * in the set of feasible policies.

Specifically, is computed by solving the following linear program (LP):

where 1(x 0 ) represents a one-hot vector in which the non-zero element is located at x = x 0 .

When π B is a feasible policy, this problem has a non-empty solution.

Furthermore, according to the derivations in (Chow et al., 2018) , the maximizer of (22) has the following form:

Input: Initial feasible policy π0; for k = 0, 1, 2, . . .

do

Step 0: With π b = π k , evaluate the Lyapunov function L k , where k is a solution of (22) Step 1: Evaluate the cost value function Vπ k (x) = Cπ k (x); Then update the policy by solving the following problem:

They also show that by further restricting (x) to be a constant function, the maximizer is given by

Using the construction of the Lyapunov function L , (Chow et al., 2018) propose the safe policy iteration (SPI) algorithm (see Algorithm 3) in which the Lyapunov function is updated via bootstrapping, i.e., at each iteration L is recomputed using (22) w.r.t.

the current baseline policy.

At each iteration k, this algorithm has the following properties: 1) Consistent Feasibility, i.e., if the current policy π k is feasible, then π k+1 is also feasible; 2) Monotonic Policy Improvement, i.e., C π k+1 (x) ≤ C π k (x) for any x ∈ X ; and 3) Asymptotic Convergence.

Despite all these nice properties, SPI is still a value-function-based algorithm, and thus, it is not straightforward to use it in continuous action problems.

The main reason is that the greedification step becomes an optimization problem over the continuous set of actions that is not necessarily easy to solve.

In Section 3, we show how we use SPI and its nice properties to develop safe policy optimization algorithms that can handle continuous action problems.

Our algorithms can be thought as combinations of DDPG or PPO (or any other on-policy or off-policy policy optimization algorithm) with a SPI-inspired critic that evaluates the policy and computes its corresponding Lyapunov function.

The computed Lyapunov function is then used to guarantee safe policy update, i.e., the new policy is selected from a restricted set of safe policies defined by the Lyapunov function of the current policy.

In this section, we first provide the details of the derivation of the θ-projection and a-projection procedures described in Section 3, and then provide the pseudo-codes of our safe PG algorithms.

To derive our θ-projection algorithms, we first consider the original Lyapunov constraint in (3) that is given by

where the baseline policy is parameterized as π B = π θ B .

Using the first-order Taylor series expansion w.r.t.

θ = θ B , at any arbitrary x ∈ X , the term E a∼π θ Q L θ B (x, a) = a∈A π θ (a|x) Q Lπ B (x, a) da on left-hand-side of the above inequality can be written as

which implies that

Note that the objective function of the constrained minimization problem in (4) contains a regularization term:

that controls the distance θ − θ B to be small.

For most practical purposes, here one can assume the higher-order term O( θ − θ B 2 ) to be much smaller than the first-order term

Therefore, one can approximate the original Lyapunov constraint in (3) with the following constraint:

Furthermore, following the same line of arguments used in TRPO (to transform the max D KL constraint to an average D KL constraint, see Eq. 12 in (Schulman et al., 2015a) ), a more numerically stable way is to approximate the Lyapunov constraint using the average constraint surrogate, i.e.,

Now consider the special case when auxiliary constraint surrogate is chosen as a constant, i.e.,

.

The justification of such choice comes from analyzing the solution of optimization problem (22).

Then, one can write the Lyapunov action-value function Q L θ B (x, a) as

Since the second term is independent of θ, for any state x ∈ X , the gradient term

where

are the constraint value function and constraint state-action value function, respectively.

The second equality is based on the standard log-likelihood gradient property in PG algorithms (Sutton et al., 2000) .

Collectively, one can then re-write the Lyapunov average constraint surrogate as

where is the auxiliary constraint cost defined specifically by the Lyapunov-based approach, to guarantee constraint satisfaction.

By expanding the auxiliary constraint cost on the right-hand-side, the above constraint is equivalent to the constraint used in CPO, i.e.,

For any arbitrary state x ∈ X , consider the following constraint in the safety-layer projection problem given in (6):

Using first-order Taylor series expansion of the Lyapunov state-action value function Q Lπ B (x, a) w.r.t.

action a = π B (x), the Lyapunov value function Q Lπ B (x, a) can be re-written as

Note that the objective function of the action-projection problem in (7) contains a regularization term

2 that controls the distance a − π B (x) to be small.

For most practical purposes, here one can assume the higher-order term O( a − π B (x)

2 ) to be much smaller than the first-order term (a − π B (x)) g Lπ B (x).

Therefore, one can approximate the original action-based Lyapunov constraint in (6) with the constraint a − π B (x) g Lπ B (x) ≤ (x) that is the constraint in (7).

Similar to the analysis of the θ-projection approach, if the auxiliary cost is state-independent, the action-gradient term g Lπ B (x) is equal to the gradient of the constraint action-value function

, where Q W θ B is the state-action constraint value function w.r.t.

the baseline policy.

The rest of the proof follows the results from Proposition 1 in (Dalal et al., 2018) .

This completes the derivations of the a-projection approach.

Algorithms 4 and 5 contain the pseudo-code of our safe Lyapunov-based policy gradient (PG) algorithms with θ-projection and a-projection, respectively.

Due to function approximation errors, even with the Lyapunov constraints, in practice a safe PG algorithm may take a bad step and produce an infeasible policy update and cannot automatically recover from such a bad step.

To address this issue, similar to (Achiam et al., 2017) , we propose the following safeguard policy update rule to decrease the constraint cost:

where α sg,k is the learning rate for the safeguard update.

If α sg,k >> α k (learning rate of PG), then with the safeguard update, θ will quickly recover from the bad step, however, it might be overly conservative.

This approach is principled because as soon as π θ k is unsafe/infeasible w.r.t.

the CMDP constraints, the algorithm uses a limiting search direction.

One can directly extend this safeguard update to the multiple-constraint scenario by doing gradient descent over the constraint that has the worst violation.

Another remedy to reduce the chance of constraint violation is to do constraint tightening on the constraint cost threshold.

Specifically, instead of d 0 , one may pose the constraint based on d 0 · (1 − δ), where δ ∈ (0, 1) is the factor of safety for providing additional buffer to constraint violation.

Additional techniques in cost-shaping have been proposed in (Achiam et al., 2017) to smooth out the sparse constraint costs.

While these techniques can further ensure safety, construction of the cost-shaping term requires knowledge of the environment, which makes the safe PG algorithms more complicated.

Input: Initial feasible policy π0; for k = 0, 1, 2, . . .

do

Step 0:

of T steps by starting at x0 and following the policy θ k

Step 1: Using the trajectories {ξ j,k } N j=1 , estimate the critic Q θ (x, a) and the constraint critic Q D,θ (x, a); • For DDPG, these functions are trained by minimizing the MSE of Bellman residual, and one can also use off-policy samples from replay buffer (Schaul et al., 2015) ; • For PPO these functions can be estimated by the generalized advantage function technique from Schulman et al. (2015b) Step 2: Based on the closed form solution of a QP problem with an LP constraint in Section 10.2 of Achiam et al. (2017) , calculate λ * k with the following formula:

β k is the adaptive penalty weight of the DKL(π||π θ k ) regularizer, and

Step 3: Update the policy parameter by following the objective gradient;

• For DDPG

• For PPO,

Step 4: At any given state x ∈ X , compute the feasible action probability a * (x) via action projection in the safety layer, that takes inputs ∇aQL(x, a) = ∇aQ D,θ k (x, a) and (x) = (1 − γ)(d0 − Q D,θ k (x0, π k (x0))), for any a ∈ A. end for Return Final policy π θ k * ,

Our experiments are performed on safety-augmented versions of standard MuJoCo domains (Todorov et al., 2012) .

HalfCheetah-Safe.

The agent is a the standard HalfCheetah (a 2-legged simulated robot rewarded for running at high speed) augmented with safety constraints.

We choose the safety constraints to be defined on the speed limit.

We constrain the speed to be less than 1, i.e., constraint cost is thus 1[|v| > 1].

Episodes are of length 200.

The constraint threshold is 50.

Point Circle.

This environment is taken from (Achiam et al., 2017) .

The agent is a point mass (controlled via a pivot).

The agent is initialized at (0, 0) and rewarded for moving counter-clockwise along a circle of radius 15 according to the reward

, for position x, y and velocity dx, dy.

The safety constraint is defined as the agent staying in a position satisfying |x| ≤ 2.5.

The constraint cost is thus 1[|x| > 2.5].

Episodes are of length 65.

The constraint threshold is 7.

Input: Initial feasible policy π0; for k = 0, 1, 2, . . .

do

Step 0:

of T steps by starting at x0 and following the policy θ k

Step 1: Using the trajectories {ξ j,k } N j=1 , estimate the critic Q θ (x, a) and the constraint critic Q D,θ (x, a); • For DDPG, these functions are trained by minimizing the MSE of Bellman residual, and one can also use off-policy samples from replay buffer (Schaul et al., 2015) ; • For PPO these functions can be estimated by the generalized advantage function technique from Schulman et al. (2015b) Step 2: Update the policy parameter by following the objective gradient;

• For DDPG

• For PPO,

where β k is the adaptive penalty weight of the DKL(π||π θ k ) regularizer, and

Step 3: At any given state x ∈ X , compute the feasible action probability a * (x) via action projection in the safety layer, that takes inputs ∇aQL(x, a) = ∇aQ D,θ k (x, a) and

Point Gather.

This environment is taken from (Achiam et al., 2017) .

The agent is a point mass (controlled via a pivot) and the environment includes randomly positioned apples (2 apples) and bombs (8 bombs).

The agent given a reward of 10 for each apple collected and a penalty of −10 for each bomb.

The safety constraint is defined as the number of bombs collected during the episode.

Episodes are of length 15.

The constraint threshold is 4 for DDPG and 2 for PPO.

Ant Gather.

This environment is the same as Point Circle, only with an Ant agent (quadrapedal simulated robot).

Each episode is initialized with 8 apples and 8 bombs.

The agent receives a reward of 10 for each apple collected, a penalty of −20 for each bomb collected, and a penalty of −20 if the episode terminates prematurely (because the Ant falls).

Episodes are of length at most 500.

The constraint threshold is 10 and 5 for DDPG and PPO, respectively.

In these experiments, there are three different agents: (1) a point-mass (X ⊆ R 9 , A ⊆ R 2 ); an ant quadruped robot (X ⊆ R 32 , A ⊆ R 8 ); and (3) a half-cheetah (X ⊆ R 18 , A ⊆ R 6 ).

For all experiments, we use two neural networks with two hidden layers of size (100, 50) and ReLU activation to model the mean and log-variance of the Gaussian actor policy, and two neural networks with two hidden layers of size (200, 50) and tanh activation to model the critic and constraint critic.

To build a low variance sample gradient estimate, we use GAE-λ (Schulman et al., 2015b) On top of GAE-λ, in all experiments and for each algorithm (SDDPG, SPPO, SDDPG a-projection, SPPO a-projection, CPO, Lagrangian, and the unconstrained PG counterparts), we systematically explored different parameter settings by doing grid-search over the following factors: (i) learning rates in the actor-critic algorithm, (ii) batch size, (iii) regularization parameters of the policy relative entropy term, (iv) with-or-without natural policy gradient updates, (v) with-or-without the emergency safeguard PG updates (see Appendix C.4 for more details).

Although each algorithm might have a different parameter setting that leads to the optimal performance in training, the results reported here are the best ones for each algorithm, chosen by the same criteria (which is based on the value of return plus certain degree of constraint satisfaction).

To account for the variability during training, in each learning curve, a 1-SD confidence interval is also computed over 10 separate random runs (under the same parameter setting).

In all numerical experiments and for each algorithm (SPPO θ-projection, SDDPG θ-projection, SPPO a-projection, SDDPG a-projection, CPO, Lagrangian, and the unconstrained PG counterparts), we systematically explored various hyper-parameter settings by doing grid-search over the following factors: (i) learning rates in the actor-critic algorithm, (ii) batch size, (iii) regularization parameters of the policy relative entropy term, (iv) with-or-without natural policy gradient updates, (v) with-orwithout the emergency safeguard PG updates (see Appendix C.4 for more details).

Although each algorithm might have a different parameter setting that leads to the optimal training performance, the results reported in the paper are the best ones for each algorithm, chosen by the same criteria (which is based on value of return + certain degree of constraint satisfaction).

In our experiments, we compare the two classes of safe RL algorithms, one derived from θ-projection (constrained policy optimization) and one from the a-projection (safety layer), with the unconstrained and Lagrangian baselines in four problems: PointGather, AntGather, PointCircle, and HalfCheetahSafe.

We perform these experiments with both off-policy (DDPG) and on-policy (PPO) versions of the algorithms.

In PointCircle DDPG, although the Lagrangian algorithm significantly outperforms the safe RL algorithms in terms of return, it violates the constraint more often.

The only experiment in which Lagrangian performs similarly to the safe algorithms in terms of both return and constraint violation is PointCircle PPO.

In all other experiments that are performed in the HalfCheetahSafe, PointGather and AntGather domains, either (i) the policy learned by Lagrangian has a significantly lower performance than that learned by one of the safe algorithms (see HalfCheetahSafe DDPG, PointGather DDPG, AntGather DDPG), or (ii) the Lagrangian method violates the constraint during training, while the safe algorithms do not (see HalfCheetahSafe PPO, PointGather PPO, AntGather PPO).

This clearly illustrates the effectiveness of our Lyapunov-based safe RL algorithms, when compared to Lagrangian method.

Mapless navigation task is a continuous control task with a goal of navigating a robot to any arbitrary goal position collision-free and without memory of the workspace topology.

The goal is usually within 5 − 10 meters from the robot agent, but it is not visible to the agent before the task starts, due to both limited sensor range and the presence of obstacles that block a clear line of sight.

The agent's observations, x = (g,ġ, l) ∈ R 68 , consists of the relative goal position, the relative goal velocity, and the Lidar measurements.

Relative goal position, g, is the relative polar coordinates between the goal position and the current robot pose, andġ is the time derivative of g, which indicates the speed of the robot navigating to the goal.

This information is available from the robot's localization sensors.

Vector l is the noisy Lidar input (Fig. 3a) , which measures the nearest obstacle in a direction within a 220

• field of view split in 64 bins, up to 5 meters in depth.

The action is given by a ∈ R 2 , which is linear and angular velocity vector at the robot's center of the mass.

The transition probability P : X × A → X captures the noisy differential drive robot dynamics.

Without knowing the full nonlinear system dynamics, we here assume knowledge of a simplified blackbox kinematics simulator operating at 5Hz in which Gaussian noise, N (0, 0.1), is added to both the observations and actions in order to model the noise in sensing, dynamics, and action actuations in real-world.

The objective of the P2P task is to navigate the robot to reach within 30 centimeters from any real-time goal.

While the dynamics of this system is simpler than that of HalfCheetah.

But unlike the MuJoCo tasks where the underlying dynamics are deterministic, in this robot experiment the sensor, localization, and dynamics noise paired with partial world observations and unexpected obstacles make this safe RL much more challenging.

More descriptions about the indoor robot navigation problem and its implementation details can be found in Section 3 and 4 of (Chiang et al., 2019) .

Fetch robot weights 150 kilograms, and reaches maximum speed of 7 km/h making the collision force a safety paramount.

Here the CMDP is non-discounting and has a finite-horizon of T = 100.

We reward the agent for reaching the goal, which translates to an immediate cost of c(x, a) = g 2 , which measures the relative distance to goal.

To measure the impact energy of obstacle collisions, we impose an immediate constraint cost of d(x, a) = ġ · 1{ l ≤ r impact }/T , where r impact is the impact radius w.r.t.

the Lidar depth signal, to account for the speed during collision, with a constraint threshold d 0 that characterizes the agent's maximum tolerable collision impact energy to any objects.

(Here the total impact energy is proportional to the robot's speed during any collisions.)

Under this CMDP framework (Fig. 3b) , the main goal is to train a policy π * that drives the robot along the shortest path to the goal and to limit the average impact energy of obstacle collisions.

Furthermore, due to limited data any intermediate point-to-point policy is deployed on the robot to collect more samples for further training, therefore guaranteeing safety during training is critical in this application.

@highlight

A general framework for incorporating long-term safety constraints in policy-based reinforcement learning
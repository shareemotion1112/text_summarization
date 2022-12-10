Solving tasks in Reinforcement Learning is no easy feat.

As the goal of the agent is to maximize the accumulated reward, it often learns to exploit loopholes and misspecifications in the reward signal resulting in unwanted behavior.

While constraints may solve this issue, there is no closed form solution for general constraints.

In this work we present a novel multi-timescale approach for constrained policy optimization, called `Reward Constrained Policy Optimization' (RCPO), which uses an alternative penalty signal to guide the policy towards a constraint satisfying one.

We prove the convergence of our approach and provide empirical evidence of its ability to train constraint satisfying policies.

Applying Reinforcement Learning (RL) is generally a hard problem.

At each state, the agent performs an action which produces a reward.

The goal is to maximize the accumulated reward, hence the reward signal implicitly defines the behavior of the agent.

While in computer games (e.g. BID2 ) there exists a pre-defined reward signal, it is not such in many real applications.

An example is the Mujoco domain BID33 , in which the goal is to learn to control robotic agents in tasks such as: standing up, walking, navigation and more.

Considering the Humanoid domain, the agent is a 3 dimensional humanoid and the task is to walk forward as far as possible (without falling down) within a fixed amount of time.

Naturally, a reward is provided based on the forward velocity in order to encourage a larger distance; however, additional reward signals are provided in order to guide the agent, for instance a bonus for staying alive, a penalty for energy usage and a penalty based on the force of impact between the feet and the floor (which should encourage less erratic behavior).

Each signal is multiplied by it's own coefficient, which controls the emphasis placed on it.

This approach is a multi-objective problem BID20 ; in which for each set of penalty coefficients, there exists a different, optimal solution, also known as Pareto optimality BID34 .

In practice, the exact coefficient is selected through a time consuming and a computationally intensive process of hyper-parameter tuning.

As our experiments show, the coefficient is not shared across domains, a coefficient which leads to a satisfying behavior on one domain may lead to catastrophic failure on the other (issues also seen in BID17 and BID19 ).

Constraints are a natural and consistent approach, an approach which ensures a satisfying behavior without the need for manually selecting the penalty coefficients.

In constrained optimization, the task is to maximize a target function f (x) while satisfying an inequality constraint g(x) ≤ α.

While constraints are a promising solution to ensuring a satisfying behavior, existing methods are limited in the type of constraints they are able to handle and the algorithms that they may support -they require a parametrization of the policy (policy gradient methods) and propagation of the constraint violation signal over the entire trajectory (e.g. BID26 ).

This poses an issue, as Q-learning algorithms such as DQN BID21 do not learn a parametrization of the policy, and common Actor-Critic methods (e.g. BID27 BID22 BID0 Reward shaping BID29 3 BID29 ) build the reward-to-go based on an N-step sample and a bootstrap update from the critic.

In this paper, we propose the 'Reward Constrained Policy Optimization' (RCPO) algorithm.

RCPO incorporates the constraint as a penalty signal into the reward function.

This penalty signal guides the policy towards a constraint satisfying solution.

We prove that RCPO converges almost surely, under mild assumptions, to a constraint satisfying solution (Theorem 2).

In addition; we show, empirically on a toy domain and six robotics domains, that RCPO results in a constraint satisfying solution while demonstrating faster convergence and improved stability (compared to the standard constraint optimization methods).Related work: Constrained Markov Decision Processes BID1 are an active field of research.

CMDP applications cover a vast number of topics, such as: electric grids BID14 , networking BID11 , robotics BID8 BID10 BID0 BID9 and finance BID15 BID32 .The main approaches to solving such problems are (i) Lagrange multipliers BID5 BID4 , (ii) Trust Region BID0 , (iii) integrating prior knowledge BID9 and (iv) manual selection of the penalty coefficient BID31 BID18 BID25 .

The novelty of our work lies in the ability to tackle (1) general constraints (both discounted sum and mean value constraints), not only constraints which satisfy the recursive Bellman equation (i.e, discounted sum constraints) as in previous work.

The algorithm is (2) reward agnostic.

That is, invariant to scaling of the underlying reward signal, and (3) does not require the use of prior knowledge.

A comparison with the different approaches is provided in TAB0 .

A Markov Decision Processes M is defined by the tuple (S, A, R, P, µ, γ) (Sutton and Barto, 1998).

Where S is the set of states, A the available actions, R : S × A × S → R is the reward function, P : S × A × S → [0, 1] is the transition matrix, where P (s |s, a) is the probability of transitioning from state s to s assuming action a was taken, µ : S → [0, 1] is the initial state distribution and γ ∈ [0, 1) is the discount factor for future rewards.

A policy π : S → ∆ A is a probability distribution over actions and π(a|s) denotes the probability of taking action a at state s.

For each state s, the value of following policy π is denoted by: DISPLAYFORM0 1 A mean valued constraint takes the form of E[ DISPLAYFORM1 DISPLAYFORM2 The goal is then to maximize the expectation of the reward-to-go, given the initial state distribution µ: DISPLAYFORM3

A Constrained Markov Decision Process (CMDP) extends the MDP framework by introducing a penalty c(s, a), a constraint C(s t ) = F (c(s t , a t ), ..., c(s N , a N )) and a threshold α ∈ [0, 1].

A constraint may be a discounted sum (similar to the reward-to-go), the average sum and more (see BID1 for additional examples).

Throughout the paper we will refer to the collection of these constraints as general constraints.

We denote the expectation over the constraint by: DISPLAYFORM0 The problem thus becomes: DISPLAYFORM1

In this work we consider parametrized policies, such as neural networks.

The parameters of the policy are denoted by θ and a parametrized policy as π θ .

We make the following assumptions in order to ensure convergence to a constraint satisfying policy: DISPLAYFORM0 Assumption 2 is the minimal requirement in order to ensure convergence, given a general constraint, of a gradient algorithm to a feasible solution.

Stricter assumptions, such as convexity, may ensure convergence to the optimal solution; however, in practice constraints are non-convex and such assumptions do not hold.

Constrained MDP's are often solved using the Lagrange relaxation technique BID3 .

In Lagrange relaxation, the CMDP is converted into an equivalent unconstrained problem.

In addition to the objective, a penalty term is added for infeasibility, thus making infeasible solutions sub-optimal.

Given a CMDP (3), the unconstrained problem is DISPLAYFORM0 where L is the Lagrangian and λ ≥ 0 is the Lagrange multiplier (a penalty coefficient).

Notice, as λ increases, the solution to (4) converges to that of (3).

This suggests a twotimescale approach: on the faster timescale, θ is found by solving (4), while on the slower timescale, λ is increased until the constraint is satisfied.

The goal is to find a saddle point (θ * (λ * ), λ * ) of (4), which is a feasible solution.

Definition 1.

A feasible solution of the CMDP is a solution which satisfies J π C ≤ α.

We assume there isn't access to the MDP itself, but rather samples are obtained via simulation.

The simulation based algorithm for the constrained optimization problem (3) is: DISPLAYFORM1 DISPLAYFORM2 where Γ θ is a projection operator, which keeps the iterate θ k stable by projecting onto a compact and convex set.

Γ λ projects λ into the range [0, λ max 4 ].

∇ θ L and ∇ λ L are derived from (4), where the formulation for ∇ θ L is derivied using the log-likelihood trick BID35 : DISPLAYFORM3 DISPLAYFORM4 η 1 (k), η 2 (k) are step-sizes which ensure that the policy update is performed on a faster timescale than that of the penalty coefficient λ.

DISPLAYFORM5 Theorem 1.

Under Assumption 3, as well as the standard stability assumption for the iterates and bounded noise BID6 , the iterates (θ n , λ n ) converge to a fixed point (a local minima) almost surely.

Lemma 1.

Under assumptions 1 and 2, the fixed point of Theorem 1 is a feasible solution.

The proof to Theorem 1 is provided in Appendix C and to Lemma 1 in Appendix D.

Recently there has been a rise in the use of Actor-Critic based approaches, for example: A3C BID22 , TRPO BID27 and PPO BID29 .

The actor learns a policy π, whereas the critic learns the value (using temporal-difference learning -the recursive Bellman equation).

While the original use of the critic was for variance reduction, it also enables training using a finite number of samples (as opposed to Monte-Carlo sampling).Our goal is to tackle general constraints (Section 2.2), as such, they are not ensured to satisfy the recursive property required to train a critic.

We overcome this issue by training the actor (and critic) using an alternative, guiding, penalty -the discounted penalty.

The appropriate assumptions under which the process converges to a feasible solution are provided in Theorem 2.

It is important to note that; in order to ensure constraint satisfaction, λ is still optimized using Monte-Carlo sampling on the original constraint (8).Definition 2.

The value of the discounted (guiding) penalty is defined as: DISPLAYFORM0 Definition 3.

The penalized reward functions are defined as: DISPLAYFORM1 DISPLAYFORM2 As opposed to (4), for a fixed π and λ, the penalized value (11) can be estimated using TD-learning critic.

We denote a three-timescale (Constrained Actor Critic) process, in which the actor and critic are updated following (11) and λ is updated following (5), as the 'Reward Constrained Policy Optimization' (RCPO) algorithm.

Algorithm 1 illustrates such a procedure and a full RCPO Advantage-Actor-Critic algorithm is provided in Appendix A.Algorithm 1 Template for an RCPO implementation DISPLAYFORM3 Initialize actor parameters θ = θ 0 , critic parameters v = v 0 , Lagrange multipliers and DISPLAYFORM4 Initialize state s 0 ∼ µ 5: DISPLAYFORM5 Sample action a t ∼ π, observe next state s t+1 , reward r t and penalties c t 7: DISPLAYFORM6 Equation FORMULA3 8:Critic update: DISPLAYFORM7 9:Actor update: DISPLAYFORM8 10:Lagrange multiplier update: Cγ as Θ γ .

Assuming that Θ γ ⊆ Θ then the 'Reward Constrained Policy Optimization' (RCPO) algorithm converges almost surely to a fixed point (θ DISPLAYFORM9 DISPLAYFORM10 The proof to Theorem 2 is provided in Appendix E.The assumption in Theorem 2 demands a specific correlation between the guiding penalty signal C γ and the constraint C. Consider a robot with an average torque constraint.

A policy which uses 0 torque at each time-step is a feasible solution and in turn is a local minimum of both J C and J Cγ .

If such a policy is reachable from any θ (via gradient descent), this is enough in order to provide a theoretical guarantee such that J Cγ may be used as a guiding signal in order to converge to a fixed-point, which is a feasible solution.

We test the RCPO algorithm in various domains: a grid-world, and 6 tasks in the Mujoco simulator BID33 .

The grid-world serves as an experiment to show the benefits of RCPO over the standard Primal-Dual approach (solving (4) using Monte-Carlo simulations), whereas in the Mujoco domains we compare RCPO to reward shaping, a simpler (yet common) approach, and show the benefits of an adaptive approach to defining the cost value.

While we consider mean value constraints (robotics experiments) and probabilistic constraints (i.e., Mars rover), discounted sum constraints can be immediately incorporated into our setup.

We compare our approach with relevant baselines that can support these constraints.

Discounted sum approaches such as BID0 and per-state constraints such as BID9 are unsuitable for comparison given the considered constraints.

See TAB0 for more details.

For clarity, we provide exact details in Appendix B (architecture and simulation specifics).

The rover (red square) starts at the top left, a safe region of the grid, and is required to travel to the goal (orange square) which is located in the top right corner.

The transition function is stochastic, the rover will move in the selected direction with probability 1 − δ and randomly otherwise.

On each step, the agent receives a small negative reward r step and upon reaching the goal state a reward r goal .

Crashing into a rock (yellow) causes the episode to terminate and provides a negative reward −λ.

The domain is inspired by the Mars Rover domain presented in BID8 .

It is important to note that the domain is built such that a shorter path induces higher risk (more rocks along the path).

Given a minimal failure threshold (α ∈ (0, 1)), the task is to find λ, such that when solving for parameters δ, r step , r goal and λ, the policy will induce a path with P π θ µ (failure) ≤ α; e.g., find the shortest path while ensuring that the probability of failure is less or equal to α.

As this domain is characterized by a discrete action space, we solve it using the A2C algorithm (a synchronous version of A3C BID22 ).

We compare RCPO, using the discounted penalty C γ , with direct optimization of the Lagrange dual form (4).

FIG1 illustrates the domain and the policies the agent has learned based on different safety requirements.

Learning curves are provided in FIG0 .

The experiments show that, for both scenarios α = 0.01 and α = 0.5, RCPO is characterized by faster convergence (improved sample efficiency) and lower variance (a stabler learning regime).

Todorov et al. FORMULA3 ; BID7 and OpenAI (2017) provide interfaces for training agents in complex control problems.

These tasks attempt to imitate scenarios encountered by robots in real life, tasks such as teaching a humanoid robot to stand up, walk, and more.

The robot is composed of n joints; the state S ∈ R n×5 is composed of the coordinates (x, y, z) and angular velocity (ω θ , ω φ ) of each joint.

At each step the agent selects the amount of torque to apply to each joint.

We chose to use PPO BID29 in order to cope with the continuous action space.

In the following experiments; the aim is to prolong the motor life of the various robots, while still enabling the robot to perform the task at hand.

To do so, the robot motors need to be constrained from using high torque values.

This is accomplished by defining the constraint C as the average torque the agent has applied to each motor, and the per-state penalty c(s, a) becomes the amount of torque the agent decided to apply at each time step.

We compare RCPO to the reward shaping approach, in which the different values of λ are selected apriori and remain constant.

Learning curves are provided in FIG3 and the final values in TAB2 .

It is important to note that by preventing the agent from using high torque levels (limit the space of admissible policies), the agent may only be able to achieve a sub-optimal policy.

RCPO aims to find the best performing policy given the constraints; that is, the policy that achieves maximal value while at the same time satisfying the constraints.

Our experiments show that:1.

In all domains, RCPO finds a feasible (or near feasible) solution, and, besides the Walker2d-v2 domain, exhibits superior performance when compared to the relevant reward shaping variants (constant λ values resulting in constraint satisfaction).

Results are considered valid only if they are at or below the threshold.

RCPO is our approach, whereas each λ value is a PPO simulation with a fixed penalty coefficient.

Y axis is the average reward and the X axis represents the number of samples (steps).2.

Selecting a constant coefficient λ such that the policy satisfies the constraint is not a trivial task, resulting in different results across domains BID0 .

When performing reward shaping (selecting a fixed λ value), the experiments show that in domains where the agent attains a high value, the penalty coefficient is required to be larger in order for the solution to satisfy the constraints.

However, in domains where the agent attains a relatively low value, the same penalty coefficients can lead to drastically different behavior -often with severely sub-optimal solutions (e.g. Ant-v2 compared to Swimmer-v2).Additionally, in RL, the value (J π R ) increases as training progresses, this suggests that a non-adaptive approach is prone to converge to sub-optimal solutions; when the penalty is large, it is plausible that at the beginning of training the agent will only focus on constraint satisfaction and ignore the underlying reward signal, quickly converging to a local minima.

We introduced a novel constrained actor-critic approach, named 'Reward Constrained Policy Optimization' (RCPO).

RCPO uses a multi-timescale approach; on the fast timescale an alternative, discounted, objective is estimated using a TD-critic; on the intermediate timescale the policy is learned using policy gradient methods; and on the slow timescale the penalty coefficient λ is learned by ascending on the original constraint.

We validate our approach using simulations on both grid-world and robotics domains and show that RCPO converges in a stable and sample efficient manner to a constraint satisfying policy.

An exciting extension of this work is the combination of RCPO with CPO BID0 .

As they consider the discounted penalty, our guiding signal, it might be possible to combine both approaches.

Such an approach will be able to solve complex constraints while enjoying feasibility guarantees during training.

The original Advantage Actor Critic algorithm is in gray, whereas our additions are highlighted in black.1: Input: penalty function C(·), threshold α and learning rates η1, η2, η3 2: Initialize actor π(·|·; θp) and critic V (·; θv) with random weights 3: Initialize λ = 0, t = 0, s0 ∼ µ Restart 4: for T = 1, 2, ..., Tmax do 5:Reset gradients dθv ← 0, dθp ← 0 and ∀i :

dλi ← 0 6: tstart = t 7:while st not terminal and t − tstart < tmax do 8:Perform at according to policy π(at|st; θp) 9:Receive rt, st+1 and penalty scoreĈt 10:t ← t + 1 11: DISPLAYFORM0 for τ = t − 1, t − 2, ..., tstart do 13: DISPLAYFORM1 Update θv, θp and λ 21:Set λ = max(λ, 0) Ensure weights are non-negative (Equation 4)

The MDP was defined as follows: DISPLAYFORM0 In order to avoid the issue of exploration in this domain, we employ a linearly decaying random restart BID12 .

µ, the initial state distribution, follows the following rule: DISPLAYFORM1 where S denotes all the non-terminal states in the state space and s * is the state at the top left corner (red in FIG1 ).

Initially the agent starts at a random state, effectively improving the exploration and reducing convergence time.

As training progresses, with increasing probability, the agent starts at the top left corner, the state which we test against.

The A2C architecture is the standard non-recurrent architecture, where the actor and critic share the internal representation and only hold a separate final projection layer.

The input is fully-observable, being the whole grid.

The network is as follows: Published as a conference paper at ICLR 2019 between the layers we apply a ReLU non-linearity.

DISPLAYFORM2 As performance is noisy on such risk-sensitive environments, we evaluated the agent every 5120 episodes for a length of 1024 episodes.

To reduce the initial convergence time, we start λ at 0.6 and use a learning rate lr λ = 0.000025.

For these experiments we used a PyTorch BID24 implementation of PPO BID13 .

Notice that as in each domain the state represents the location and velocity of each joint, the number of inputs differs between domains.

The network is as follows: where DiagGaussian is a multivariate Gaussian distribution layer which learns a mean (as a function of the previous layers output) and std, per each motor, from which the torque is sampled.

Between each layer, a Tanh non-linearity is applied.

DISPLAYFORM0 We report the online performance of the agent and run each test for a total of 1M samples.

In these domains we start λ at 0 and use a learning rate lr λ = 5e − 7 which decays at a rate of κ = (1 − 1e − 9) in order to avoid oscillations.

The simulations were run using Generalized Advantage Estimation BID28 with coefficient τ = 0.95 and discount factor γ = 0.99.

We provide a brief proof for clarity.

We refer the reader to Chapter 6 of BID6 for a full proof of convergence for two-timescale stochastic approximation processes.

Initially, we assume nothing regarding the structure of the constraint as such λ max is given some finite value.

The special case in which Assumption 2 holds is handled in Lemma 1.The proof of convergence to a local saddle point of the Lagrangian (4) contains the following main steps:1.

Convergence of θ-recursion: We utilize the fact that owing to projection, the θ parameter is stable.

We show that the θ-recursion tracks an ODE in the asymptotic limit, for any given value of λ on the slowest timescale.2.

Convergence of λ-recursion: This step is similar to earlier analysis for constrained MDPs.

In particular, we show that λ-recursion in (4) converges and the overall convergence of (θ k , λ k ) is to a local saddle point (θ DISPLAYFORM0 Step 1: Due to the timescale separation, we can assume that the value of λ (updated on the slower timescale) is constant.

As such it is clear that the following ODE governs the evolution of θ:θ DISPLAYFORM1 where Γ θ is a projection operator which ensures that the evolution of the ODE stays within the compact and convex set Θ : DISPLAYFORM2 As λ is considered constant, the process over θ is: DISPLAYFORM3 Thus (6) can be seen as a discretization of the ODE (12).

Finally, using the standard stochastic approximation arguments from BID6 concludes step 1.Step 2: We start by showing that the λ-recursion converges and then show that the whole process converges to a local saddle point of L(λ, θ).The process governing the evolution of λ: DISPLAYFORM4 is the limiting point of the θ-recursion corresponding to λ k , can be seen as the following ODE:λ DISPLAYFORM5 (13) As shown in BID6 chapter 6, (λ n , θ n ) converges to the internally chain transitive invariant sets of the ODE (13), DISPLAYFORM6 Finally, as seen in Theorem 2 of Chapter 2 of BID6 , θ n → θ * a.s.

then λ n → λ(θ * ) a.s.

which completes the proof.

The proof is obtained by a simple extension to that of Theorem 1.

Assumption 2 states that any local minima π θ of 2 satisfies the constraints, e.g. J π θ C ≤ α; additionally, BID16 show that first order methods such as gradient descent, converge almost surely to a local minima (avoiding saddle points and local maxima).

Hence for λ max = ∞ (unbounded Lagrange multiplier), the process converges to a fixed point (θ * (λ * ), λ * ) which is a feasible solution.

As opposed to Theorem 1, in this case we are considering a three-timescale stochastic approximation scheme (the previous Theorem considered two-timescales).

The proof is similar in essence to that of BID26 .

The full process is described as follows: DISPLAYFORM0 s∼µ log π(s, a; θ)V (λ, s t ; v k ) ] DISPLAYFORM1 2 /∂v kStep 1: The value v k runs on the fastest timescale, hence it observes θ and λ as static.

As the TD operator is a contraction we conclude that v k → v(λ, θ).Step 2: For the policy recursion θ k , due to the timescale differences, we can assume that the critic v has converged and that λ is static.

Thus as seen in the proof of Theorem 1, θ k converges to the fixed point θ(λ, v).Step 3: As shown previously (and in BID26 ), (λ n , θ n , v n ) → (λ(θ * ), θ * , v(θ * )) a.s.

Denoting by Θ = {θ : J π θ C ≤ α} the set of feasible solutions and the set of local-minimas of J π θ Cγ as Θ γ .

We recall the assumption stated in Theorem 2: Assumption 4.

Θ γ ⊆ Θ.Given that the assumption above holds, we may conclude that for λ max → ∞, the set of stationary points of the process are limited to a sub-set of feasible solutions of (4).

As such the process converges a.s.

to a feasible solution.1.

Assumption 2 does not hold: As gradient descent algorithms descend until reaching a (local) stationary point.

In such a scenario, the algorithm is only ensured to converge to some stationary solution, yet said solution is not necessarily a feasible one.

As such we can only treat the constraint as a regularizing term for the policy in which λ max defines the maximal regularization allowed.2.

Assumption 4 does not hold: In this case, it is not safe to assume that the gradient of (2) may be used as a guide for solving (3).

A Monte-Carlo approach may be used (as seen in Section 5.1) to approximate the gradients, however this does not enjoy the benefits of reduced variance and smaller samples (due to the lack of a critic).

@highlight

For complex constraints in which it is not easy to estimate the gradient, we use the discounted penalty as a guiding signal. We prove that under certain assumptions it converges to a feasible solution.
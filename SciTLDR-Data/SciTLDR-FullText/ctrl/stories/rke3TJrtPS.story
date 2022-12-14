In this paper, we consider the problem of learning control policies that optimize areward function while satisfying constraints due to considerations of safety, fairness, or other costs.

We propose a new algorithm - Projection Based ConstrainedPolicy Optimization (PCPO), an iterative method for optimizing policies in a two-step process - the first step performs an unconstrained update while the secondstep reconciles the constraint violation by projection the policy back onto the constraint set.

We theoretically analyze PCPO and provide a lower bound on rewardimprovement, as well as an upper bound on constraint violation for each policy update.

We further characterize the convergence of PCPO with projection basedon two different metrics - L2 norm and Kullback-Leibler divergence.

Our empirical results over several control tasks demonstrate that our algorithm achievessuperior performance, averaging more than 3.5 times less constraint violation andaround 15% higher reward compared to state-of-the-art methods.

Recent advances in deep reinforcement learning (deep RL) have demonstrated excellent performance on several domains ranging from games like Go (Silver et al., 2017) and StarCraft (AlphaStar, 2019) to tasks like robotic control (Levine et al., 2016) .

In these settings, agents are allowed to explore the entire state space and experiment with all possible actions during training.

However, in many real-world applications such as self-driving cars and unmanned aerial vehicles, considerations of safety, fairness and other costs prevent the agent from having complete freedom to explore the environment.

For instance, an autonomous car, while optimizing for its driving policies, must not take any actions that could cause harm to pedestrians or property (including itself).

In effect, the agent is constrained to take actions that do not violate a specified set of constraints on state-action pairs.

In this work, we address the problem of learning control policies that optimize a reward function while satisfying predefined constraints.

The problem of policy learning with constraints is challenging since directly optimizing for the reward, like in Q-Learning (Mnih et al., 2013) or policy gradient (Sutton et al., 2000) approaches, would violate the constraints at some point.

One approach to incorporate constraints into the learning process is by formulating a constrained optimization problem (Achiam et al., 2017) .

This work performs policy updates using a conditional gradient descent with line search to ensure constraint satisfaction.

However, their base optimization problem becomes infeasible when the current policy violates the constraints.

Another approach (Tessler et al., 2018) adds weighted constraints to make the optimization easier, but requires extensive hyperparameter tuning of the weights.

To address the above issues, we propose projection based constrained policy optimization (PCPO) -an iterative algorithm that performs policy updates in two stages.

In the first stage, we maximize reward using a trust region optimization method (e.g., TRPO (Schulman et al., 2015a) ) without any constraints -this might result in a new intermediate policy that does not satisfy the provided constraints.

In the second state, we reconcile the constraint violation (if any) by projecting the policy back onto the constraint set, i.e., choosing the policy in the constraint set that is closest to the intermediate policy chosen.

This allows us to perform efficient updates while not violating the constraints, without requiring line search (Achiam et al., 2017) or constraint approximations (Tessler et al., 2018) .

Further, due to the projection step, PCPO offers efficient recovery from infeasible (i.e., constraint-violating) starting states, which existing methods cannot handle well.

We analyze PCPO theoretically and derive performance bounds for our algorithm.

Specifically, based on information geometry and policy optimization theory, we construct (1) a lower bound on reward improvement, and (2) an upper bound on constraint violations for each policy update.

We find that with a relatively small step size for each policy update, the worst-case constraint violation and reward degradation are tolerable.

We further analyze two distance measures for the projection step onto the constraint set.

We find that the convergence of PCPO is affected by the singular value of the Fisher information matrix used during training, providing a prescription for choosing the type of projection depending on the problem.

Empirically, we compare PCPO with state-of-the-art algorithms on four different control tasks, including two Mujoco environments with safety constraints introduced by Achiam et al. (2017) and two traffic management tasks with fairness constraints introduced by Vinitsky et al. (2018) .

In all cases, our algorithm achieves comparable or superior performance to prior approaches, averaging more reward with less cumulative constraint violations.

For instance, across these environments, PCPO performs 3.5 times less constraint violations and around 15% more reward.

This demonstrates the ability of PCPO robustly learn constraint-satisfying policies, and represents a step towards reliable deployment of RL in the real world.

We frame our policy learning as a constrained Markov Decision Process (CMDP) (Altman, 1999) , where policies will direct the agent to obtain the reward while avoiding the cost.

We define CMDP as the tuple < S, A, T, R, C >, where S is the set of states, A is the set of actions that the agent can take, T : S ?? A ?? S ??? [0, 1] is the transition probability of the CMDP, R : S ?? A ??? R is the reward function, and C : S ?? A ??? R is the cost function.

Given the agent's current state s, the policy ??(a|s) : S ??? A selects an action a for the agent to take.

Based on s and a, the agent transits to the next state (denoted by s ) according to the state transition model T (s |s, a), and receives a reward and pays a cost, denoted by R(s, a) and C(s, a), respectively.

We aim to learn a policy ?? that maximizes a cumulative discounted reward, denoted by

?? t R(s t , a t ) , while satisfying constraints, i.e., making a cumulative discounted cost constraint below a desired threshold h, denoted by

where ?? is the discount factor, ?? is the trajectory (?? = (s 0 , a 0 , s 1 , ?? ?? ?? )), and ?? ??? ?? is shorthand for showing that the distribution over the trajectory depends on ?? : s 0 ??? ??, a t ??? ??(a t |s t ), s t+1 ??? T (s t+1 |s t , a t ), where ?? is the initial state distribution.

Kakade & Langford (2002) derived an identity to express the performance of one policy ?? in terms of the advantage function over ?? :

where d ?? is the discounted future state distribution, denoted by d ?? (s) .

Here Q ?? R (s, a) is the discounted cumulative reward obtained by the policy ?? given the initial state s and action a, and V ?? R (s) is the discounted cumulative reward obtained by the policy ?? given the initial state s.

Lastly, we also have A

Learning constraint-satisfying policies is challenging because the policy optimization landscape is no longer smooth.

Further, in many cases, the constraints often conflict with the best direction of policy updates to maximize reward.

Therefore, we require an algorithm that can make progress in terms of policy improvement without being shackled by the constraints and potentially getting stuck in local minima.

A further challenge is that if we do end up with an infeasible (i.e., constraintviolating) policy, we need some efficient means of recovering back to a constraint-satisfying policy.

Figure 1: The update procedures for PCPO.

In the first step, PCPO follows the reward improvement direction in the trust region.

In the second step, PCPO projects the policy onto the constraint set.

To this end, we develop PCPO -a trust region method that performs policy updates corresponding to reward improvement, followed by projections onto the constraint set.

Formally, PCPO, inspired by projected gradient descent, is composed of two steps for each policy update -a reward improvement step and a projection step (See Fig. 1 for illustrating the procedure of PCPO).

Step.

First, we optimize a reward function by maximizing the reward advantage function A ?? R (s, a) subject to a Kullback-Leibler (KL) divergence constraint that constraints the intermediate policy ??

This update rule with the trust region, denoted by {?? : (Schulman et al., 2015a) .

It effectively constraints the policy changes and guarantees reward improvement.

Step.

Second, we project the intermediate policy ?? k+ 1 2 onto the constraint set by minimizing distance measure D subject to the constraint set:

The projection step says that the constraint-satisfying policy ?? k+1 is within the neighbourhood of ?? k+ 1 2 .

We consider two distance measures -L 2 norm and KL divergence.

If the neighbourhood is defined in the parameter space, a natural way is to use L 2 norm projection.

However, for the projection that defines in the parameter space, it is difficult to make connection to the policy defined in the probability distribution space, and hence hard to provide guarantees.

Fortunately, using KL divergence projection in the probability distribution space enables us to provide provable guarantees for PCPO with KL divergence projection.

To give performance guarantees for PCPO with KL divergence projection, we analyze worst-case performance degradation for each policy update when the current policy ?? k satisfies the constraint.

The following theorem provides (1) a lower bound on reward improvement, and (2) an upper bound on constraint violation for each policy update.

Theorem 3.1 (Worst-case Bound on Updating Constraint-satisfying Policies).

Define

, and

.

If the current policy ?? k satis-ment, and upper bound on constraint violation for each policy update are

(1 ??? ??) 2 , where ?? is the step size in the reward improvement step.

Proof.

See Appendix A.

Theorem 3.1 states that if ?? is small, the worst-case performance degradation is tolerable.

Due to approximation errors or random initialization of policies, PCPO may produce a constraintviolating policy.

To give performance guarantees for PCPO with KL divergence projection, we analyze worst-case performance degradation for each policy update when the current policy ?? k violates the constraint.

The following theorem provides (1) a lower bound on reward improvement, and (2) an upper bound on constraint violation for each policy update.

Theorem 3.2 (Worst-case Bound on Updating Constraint-violating Policies).

Define

, and

, where a is the gradient of the cost advantage function, and H is the Hessian of the KL divergence constraint.

If the current policy ?? k violates the constraint, and KL divergence projection is used, then the lower bound on reward improvement, and the upper bound on constraint violation for each policy update are

where ?? is the step size in the reward improvement step.

Proof.

See Appendix B.

Theorem 3.2 states that when the policy has more constraint violation (b + increases), its worst-case performance degradation increases.

Note that Theorem 3.2 boils down to Theorem 3.1 if the current policy ?? k satisfies the constraint (b + = 0).

The proofs of Theorem 3.1 and Theorem 3.2 follow the fact that the projection of the policy is non-expansive, i.e., the distance between the projected policies is smaller than that of the unprojected policies, so we are able to measure it and bound the KL divergence between the current policy and the new policy.

For a large neural network policy with many parameters, it is impractical to directly solve for the PCPO update due to the computational cost.

However, with a small step size ??, we can approximate the reward function and constraints with a first order expansion, and approximate the KL divergence constraint in reward improvement step, and the KL divergence measure in projection step with a second order expansion.

We now make several definitions:

] is the gradient of the cost advantage function, H is the Hessian of the KL divergence constraint (H is also called the Fisher information matrix, which guarantees positive semi-definite), b .

= J C (?? k ) ??? h, and ?? is the parameter of the policy.

Step.

First, we linearize the objective function at ?? k subject to second order approximation of KL divergence constraint in order to obtain the following updates:

Algorithm 1 Projection Based Constrained Policy Optimization (PCPO)

and store trajectories in D Compute g, a, H, and b using D Obtain ?? k+1 using update in Eq. (4) Empty D

Step.

Second, if the projection is defined in the parameter space, we can directly use L 2 norm projection.

On the other hand, if the projection is defined in the probability space, we can use KL divergence, which can be approximated through the second order expansion.

Again, we linearize the cost constraint at ?? k .

Finally, we have the following update for the projection step:

where L = I for L 2 norm projection, and L = H for KL divergence projection.

One may argue that using linear approximation to the constraint set is not enough to ensure constraint satisfaction since the real constraint set is non-convex and non-smooth in general.

However, if the step size ?? is small, then the linearization of the constraint set is accurate enough to locally approximate it.

We solve these two problems using convex programming (See Appendix C for the derivation).

For each policy update, we have

We assume that H is invertible to get a unique solution.

However, PCPO requires to invert H, which is impractical for huge neural network policies.

Hence we use the conjugate gradient method (Schulman et al., 2015a) .

Algorithm 1 shows the pseudocode.

(See Appendix F for more discussion of the tradeoff between the approximation error and computational efficiency of the conjugate gradient method.)

Analysis of PCPO Update Rule.

The update rule in Eq. (4) shows that the difference between PCPO with KL divergence and L 2 norm projections is the cost update direction, leading to reward improvement difference.

The policy iterate of L 2 norm projection has more reward fluctuation than KL divergence projection since L 2 norm projection does not use the Fisher information matrix to scale the cost update direction.

However, when the Fisher information matrix of KL divergence projection is ill-conditioned or not well-estimated, the reward and cost updates may be unstable because of pathological curvature.

In addition, these two projections converge to different stationary points with different converge rates related to the Fisher information matrix shown in Theorem 4.1.

To make our analysis valid, we consider the following assumptions are satisfied.

Assume that we minimize the negative reward objective function f : R n ??? R (We follow the convention of the literature that authors typically minimize the objective function) with L-smooth and twice continuously differentiable over the closed and convex constraint set C, and the Fisher information matrix H automatically guarantees positive semi-definite.

Theorem 4.1.

Define ?? as the coefficient for the reward updates in Eq. (4), i.e., ?? .

=

, where ?? is the step size for the reward improvement step, g is the gradient of f, H is the Fisher information matrix, ?? max (A) is the largest singular value of matrix A, and a is the gradient of cost constraint function.

Then PCPO with KL divergence projection converges to stationary points with g ??? ???a, and the objective value changes by

, and PCPO with L 2 norm projection converges to stationary points with H ???1 g ??? ???a, and if ?? max (H) ??? 1, then the objective value changes by

Proof.

See Appendix D.

Theorem 4.1 shows that the improvement of the objective value is affected by the singular value of the Fisher information matrix.

Specifically, the objective of KL divergence projection decreases when

And the objective of L 2 norm projection decreases when ?? < 2 L , implying that condition number of H is upper bounded:

L 2 ?? .

Observing the Fisher information matrix allows us to adaptively choose the type of projection to fit the landscape of the objective function.

Theorem 4.1 also states that a stationary point of PCPO with KL divergence projection has the property that the gradient of the objective is belong to the negative normal cone of the constraint set, i.e., the gradient of the cost constraint function.

On the other hand, a stationary point of PCPO with L2 norm projection has the property that the product of the inverse of the Fisher information matrix and the gradient of the objective is belong to the negative normal cone of the constraint set.

In Appendix G, we further use an example to compare the optimization trajectories and stationary points of KL divergence and L 2 norm projections.

Policy Learning with Constraints.

Learning constraint-satisfying policies has been explored in the context of safe RL (Garcia & Fernandez, 2015) .

The agent learns policies either by (1) exploration of the environment (Achiam et al., 2017; Tessler et al., 2018; Chow et al., 2017) or (2) through expert demonstrations (Ross et al., 2011; Rajeswaran et al., 2017; Gao et al., 2018) .

However, using expert demonstrations require humans to label the constraint-satisfying behavior for every possible situation.

The scalability of these rule-based approaches is an issue since many real autonomous systems such as self-driving cars and industrial robots are inherently complex.

To overcome this issue, our algorithm uses the first approach in which the agent learn by trial and error.

To prevent the agent from having constraint-violating behavior during exploring the environment, PCPO uses projection onto the constraint set to ensure constraint satisfaction throughout learning.

Using a projection onto a constraint set is an approach that has been explored for general constrained optimization in other contexts.

For example, Akrour et al. (2019) projected the policy from a parameter space onto the constraint that constrains the updated policy to stay in the neighbourhood of the previous policy.

In contrast to their work, we examine constraints that are defined in terms of states and actions.

Similarly, Chow et al. (2019) proposed ??-projection.

This projected the policy parameters ?? onto the constraint set.

However, they did not provide provable guarantees for their algorithm.

Moreover, they modelled the problem using a constrained optimization problem with the weighted constraint for step size added to the reward function.

Since the weight must be tuned, this incurs the cost of hyperparameter tuning.

In contrast to their work, PCPO eliminates the cost of the hyperparameter tuning, and provides provable guarantees on learning constraint-satisfying policies.

Comparison to CPO (Achiam et al., 2017) .

Perhaps the closest work to ours is the approach of Achiam et al. (2017) , who proposed the constrained policy optimization (CPO) algorithm to solve the following:

PCPO is different from CPO since PCPO first optimizes a reward and uses projection to satisfy the constraint, while CPO simultaneously considers the trust region and the constraint, and uses the line search to select a step size.

The update rule of CPO becomes infeasible when the current policy violates the constraint (b > 0).

CPO recovers by replacing Problem (5) with an update to purely decrease the constraint value:

This update rule may lead to a slow progress in learning constraint-satisfying policies.

In contrast, PCPO ensures a feasible solution, allowing the agent to improve the reward while ensuring constraint satisfaction simultaneously.

Figure 2: The gather, circle, grid and bottleneck tasks.

(a) Gather task: the agent is rewarded for gathering green apples but is constrained to collect a limited number of red fruit (Achiam et al., 2017) .

(b) Circle task: the agent is rewarded for moving in a specified wide circle, but is constrained to stay within a safe region smaller than the radius of the circle (Achiam et al., 2017) .

(c) Grid task: the agent controls the traffic lights in a grid road network and is rewarded for high throughput but constrained to let lights stay red for at most 7 consecutive seconds (Vinitsky et al., 2018) .

(d) Bottleneck task: the agent controls a set of autonomous vehicles (shown in red) in a traffic merge situation and is rewarded for achieving high throughput but constrained to ensure that human-driven vehicles (shown in white) have low speed for no more than 10 seconds (Vinitsky et al., 2018) .

Tasks.

We compare our method with existing approaches on four control tasks in total: two tasks with safety constraints ((a) and (b) in Fig. 2 ), and two tasks with fairness constraints ((c) and (d) in Fig. 2 ).

These tasks are briefly described in the caption of Fig. 2 .

The first two tasks -Gather and Circle -are Mujoco environments with state space constraints introduced by Achiam et al. (2017) .

The other two tasks -Grid and Bottleneck -are traffic management problems where the agent controls either a traffic light or a fleet of autonomous vehicles.

This is especially challenging since the dimensions of state and action spaces are larger, and the dynamics of the environment are inherently complex.

Baselines.

We compare our algorithm with four baselines outlined below.

(1) Constrained Policy Optimization (CPO) (Achiam et al., 2017) .

(2) Primal-dual Optimization (PDO) (Chow et al., 2017) .

In PDO, the weight (dual variables) is learned based on the current constraint satisfaction.

A PDO policy update solves:

where ?? k is updated using

Here ?? is a fixed learning rate.

(3) Fixed-point Policy Optimization (FPO).

A variant of PDO that solves Eq. (6) using a constant ??.

(4) Trust Region Policy Optimization (TRPO) (Schulman et al., 2015a) .

The TRPO policy update is an unconstrained one:

Note that TRPO ignores any constraints -we include it to serve as an upper bound baseline on the performance.

Since our main focus is to compare our algorithm with the state-of-the-art algorithm, CPO, PDO and FPO are not shown in the ant circle, ant gather, grid and bottleneck tasks for clarity.

Experimental Details.

For the gather and circle tasks we test two distinct agents: a point-mass (S ??? R 9 , A ??? R 2 ), and an ant robot (S ??? R 32 , A ??? R 8 ).

The agent in the grid task is S ??? R 156 , A ??? R 4 , and the agent in bottleneck task is S ??? R 141 , A ??? R 20 .

For the simulations in the gather and circle tasks, we use a neural network with two hidden layers of size (64, 32) to represent Gaussian policies.

For the simulations in the grid and bottleneck tasks, we use a neural network with two hidden layers of size (16, 16) and (50,25) to represent Gaussian policies, respectively.

In the experiments, since the step size is small, we reuse the Fisher of reward improvement step in the KL projection step to reduce the computational cost.

The step size ?? is set to 10 ???4 for all tasks and all tested algorithms.

For each task, we conduct 5 runs to get the mean and standard deviation for both the reward and the constraint value over the policy updates.

The experiments are implemented in rllab (Duan et al., 2016) , a tool for developing and evaluating RL algorithms.

Overall Performance.

The learning curves of the discounted reward and the undiscounted constraint value (the total number of constraint violation) over policy updates are shown for all tested algorithms and tasks in Fig. 3 .

The dashed line in the constraint figure is the cost constraint threshold h. The curves for baseline oracle, TRPO, indicate the reward and constraint value when the constraint is ignored.

Overall, we find that PCPO is able to improve the reward while having the fastest constraint satisfaction in all tasks.

In particular, PCPO is only algorithm that learns constraintsatisfying policies across all the tasks.

Moreover we observe that (1) CPO has more constraint violation than PCPO, (2) PDO is too conservative in optimizing the reward, and (3) FPO requires a significant effort to select a good vlaue of ??.

We also observe that in Grid and Bottleneck task, there is slightly more constraint violation than the easier task such as point circle and point gather.

This is due to complexity of the policy behavior and non-convexity of the constraint set.

However, even with linear approximation of the constraint set, PCPO still outperforms CPO with 85.15% and 5.42 times less constraint violation in Grid and Bottleneck task, respectively.

These observations suggest that projection step in PCPO drives the agent to learn the constraintsatisfying policy within few policy update, giving PCPO a great advantage for the real world applications.

To show that PCPO achieves the same reward with less constraint violation, we examine the reward versus the cumulative constraint value for the tested algorithms in point circle and point gather task shown in Fig. 4 .

We observe that PCPO outperforms CPO significantly with 66 times and 15 times less constraint violation under the same reward improvement in point circle and point gather tasks, respectively.

This observation suggests that PCPO enables the agent to cautiously explore the environment under the constraints.

Comparison of PCPO with KL Divergence vs. L 2 Norm Projections.

We observe that PCPO with L 2 norm projection is more constraint-satisfying than PCPO with KL divergence projection.

In addition, PCPO with L 2 norm projection tends to have reward fluctuation (point circle, ant circle, and ant gather tasks), while with KL divergence projection tends to have more stable reward improvement (all the tasks).

The above observations confirm our discussion in Section 4 that since the update direction of constraint does not scale by the Fisher information matrix, the update direction of the constraint is deviated from the update direction of the reward, which reduces the reward improvement.

However, when the Fisher information matrix is ill-conditioned or not well-estimated especially in the high dimensional policy space, the bad constraint update direction may hinder the constraint-satisfaction (ant circle, ant gather, grid and bottleneck tasks).

In addition, since the stationary points of KL divergence and L 2 norm projections are different, they converge to the policies with different reward (observe that PCPO with L 2 norm projection has higher reward than the one with KL divergence projection around 2250 iterations in ant circle task, and has less reward in point gather task).

Discussion of PDO and FPO.

For the PDO baseline, we see that its constraint values fluctuate especially in the point circle task.

This phenomena suggests that PDO is not able to adjust the weight ?? k quickly enough to meet the constraint threshold, which hinders the efficiency of learning constraint-satisfying policies.

If learning rate ?? is too big, the agent will be too conservative in improving the reward.

For the FPO, we also see that it learns near constraint-satisfying policies with slightly larger reward improvement compared to PDO.

However, in practice FPO requires a lot of engineering effort to select a good value of ??.

Since PCPO requites no hyperparameter tuning, it has the advantage of robustly learning constraint-satisfying policies over PDO and FPO.

We address the problem of finding constraint-satisfying policies.

Our algorithm -projection-based constrained policy optimization (PCPO) -optimizes for a reward function while using policy projections to ensure constraint satisfaction.

Our algorithm achieves comparable or superior performance to state-of-the-art approaches in terms of reward improvement and constraint satisfaction in all cases.

We further analyze the convergence of PCPO, and find that certain tasks may prefer either KL divergence projection or L 2 norm projection.

Future work will consider the following: (1) examining the Fisher information to iteratively prescribe the choice of projection for policy update, and hence robustly learn constraint-satisfying policies with more reward improvement, and (2) using expert demonstration or other domain knowledge to reduce the sample complexity.

Riad Akrour, Joni Pajarinen, Gerhard Neumann, and Jan Peters.

Projections for approximate policy iteration algorithms.

To prove the policy performance bound when the current policy is feasible, we prove KL divergence between ?? k and ?? k+1 for KL divergence projection.

We then prove our main theorem for worst-case performance degradation.

Lemma A.1.

If the current policy ?? k satisfies the constraint, the constraint set is closed and convex, the KL divergence constraint for the first step is E s???d ?? k D KL (??

, where ?? is the step size in the reward improvement step, and KL divergence projection is used, then we have

Proof.

By the Bregman divergence projection inequality, ?? k being in the constraint set, and ?? k+1 being the projection of the ?? k+ 1 2 onto the constraint set, we have

The derivation uses the fact that KL divergence is always greater than zero.

We know that KL divergence is asymptotically symmetric when updating the policy within a local neighbourhood.

Thus, we have

Now we use Lemma A.1 to prove our main theorem.

.

If the current policy ?? k satisfies the constraint, and KL divergence projection is used, then the lower bound on reward improvement, and the upper bound on constraint violation for each policy update are

(1 ??? ??) 2 , where ?? is the step size in the reward improvement step.

Proof.

By the theorem in Achiam et al. (2017) and Lemma A.1, we have the following reward degradation bound for each policy update:

Again, we have the following constraint violation bound for each policy update:

and

Combining Eq. (7) and Eq. (8), we have

To prove the policy performance bound when the current policy is infeasible (i.e., constraintviolating), we prove KL divergence between ?? k and ?? k+1 for KL divergence projection.

We then prove our main theorem for worst-case performance degradation.

Lemma B.1.

If the current policy ?? k violates the constraint, the constraint set is closed and convex, the KL divergence constraint for the first step is

, where ?? is the step size in the reward improvement step, and KL divergence projection is used, then we have

where

, a is the gradient of the cost advantage function, H is the Hessian of the KL divergence constraint, and

Proof.

We define the sublevel set of cost constraint function for the current infeasible policy ?? k :

This implies that the current policy ?? k lies in L By Three-point Lemma, for these three polices ?? k , ?? k+1 , and ?? k+1 l , with ??(x) .

5 shows these three polices), we have

The inequality

, and Lemma A.1.

If the constraint violation of the current policy ?? k is small, i.e., b + is small,

can be approximated by second order expansion.

By the update rule in Eq. (4), we have is the projection of ?? k+ 1 2 onto the sublevel set of the constraint set.

We want to find the KL divergence between ?? k and ?? k+1 .

where

And since ?? is small, we have ?????(?? k ) ??? ?????(?? k+1 l ) ??? 0 given s. Thus, the third term in Eq. (9) can be eliminated.

Combining Eq. (9) and Eq. (10), we have

Now we use Lemma B.1 to prove our main theorem.

, where a is the gradient of the cost advantage function, and H is the Hessian of the KL divergence constraint.

If the current policy ?? k violates the constraint, and the KL divergence projection is used, then the lower bound on the reward improvement, and the upper bound on the constraint violation for each policy update are

where ?? is the step size in the reward improvement step.

Proof.

Following the same proof in Theorem A.2, we complete the proof.

Note that the bounds we obtain for the infeasibe case; to the best of our knowledge, are new results.

Theorem C.1.

Consider the PCPO problem.

In the first step, we optimize the reward:

and in the second step, we project the policy onto the constraint set:

where g, a, ?? ??? R n , b, ?? ??? R, ?? > 0, and H, L ??? R n??n , L = H if using KL divergence projection, and L = I if using L 2 norm projection.

When there is at least one strictly feasible point, the optimal solution satisfies

assuming that H is invertible to get a unique solution.

Proof.

For the first problem, since H is the Fisher Information matrix, which automatically guarantees it is positive semi-definite.

Hence it is a convex program with quadratic inequality constraints.

Hence if the primal problem has a feasible point, then Slaters condition is satisfied and strong duality holds.

Let ?? * and ?? * denote the solutions to the primal and dual problems, respectively.

In addition, the primal objective function is continuously differentiable.

Hence the Karush-Kuhn-Tucker (KKT) conditions are necessary and sufficient for the optimality of ?? * and ?? * .

We now form the Lagrangian:

And we have the following KKT conditions:

By Eq. (11), we have ??

And by plugging Eq. (11) into Eq. (12), we have

.

Hence we have our optimal solution:

which also satisfies Eq. (13), Eq. (14), and Eq. (15).

Following the same reasoning, we now form the Lagrangian of the second problem:

And we have the following KKT conditions:

By Eq. (17), we have ??

And by plugging Eq. (17) into Eq. (18) and Eq. (20),

).

Hence we have our optimal solution:

which also satisfies Eq. (19) and Eq. (21).

Hence by Eq. (16) and Eq. (22), we have

To make our analysis valid, we consider the following assumptions are satisfied.

Assume that we minimize the negative reward objective function f : R n ??? R (We follow the convention of the literature that authors typically minimize the objective function) with L-smooth and twice continuously differentiable over the closed and convex constraint set C, and the Fisher information matrix H automatically guarantees positive semi-definite.

We include the following lemma to characterize the projection, and for the proof of Theorem D.2 (See Fig. 6 for semantic illustration) .

, and L = H if using KL divergence projection, and L = I if using L 2 norm projection.

for a given ?? ??? C, ?? ??? C be such that ?? = ?? * , and ?? ??? (0, 1).

Then we have

Since the right hand side of Eq. (23) can be made arbitrarily small for a given ??, and hence we have:

We show that ?? * must be the optimal solution.

Let ?? ??? C and ?? = ?? * .

Then we have

* is the optimal solution to the optimization problem, and ?? * = Proj

Based on Lemma D.1, we have the following theorem.

Theorem D.2.

Define ?? as the coefficient for the reward updates in Eq. (4), i.e., ?? .

= 2?? g T H ???1 g , where ?? is the step size for the reward improvement step, g is the gradient of f, H is the Fisher information matrix, ?? max (A) is the largest singular value of matrix A, and a is the gradient of cost constraint function.

Then PCPO with KL divergence projection converges to stationary points with g ??? ???a, and the objective value changes by

and PCPO with L 2 norm projection converges to stationary points with H ???1 g ??? ???a, and if ?? max (H) ??? 1, then the objective value changes by

Proof.

The proof of the theorem is based on working in a Hilbert space and the non-expansive property of the projection.

We first prove stationary points for PCPO with KL divergence and L 2 norm projections, and then prove the change of the objective value.

When in stationary points ?? * , we have

For KL divergence projection (L = H), Eq. (26) boils down to g ??? ???a, and for L 2 norm projection (L = I), Eq. (26) is equivalent to H ???1 g ??? ???a.

Now we prove the second part of the theorem.

Based on Lemma D.1, for KL divergence projection, we have

By Eq. (27), and L-smooth continuous function f, we have

For L 2 norm projection, we have

By Eq. (28), L-smooth continuous function f, and if ?? max (H) ??? 1, we have

To see why we need the assumption of ?? max (H) ??? 1, we define H = U ??U T as the singular value decomposition of H with u i being the column vector of U .

Then we have

If we want to have

then every singular value ?? i (H) of H needs to be smaller than 1, and hence ?? max (H) ??? 1, which justifies the assumption we use to prove the bound.

To make the objective value for PCPO with KL divergence projection improves, the right hand side of Eq. (24)

By the definition of the condition number and Eq. (29), we have

which justifies what we discuss.

For detailed explanation of the task in Achiam et al. (2017) , please refer to the appendix of Achiam et al. (2017) .

For detailed explanation of the task in Vinitsky et al. (2018) , please refer to Vinitsky et al. (2018) .

We use neural networks that take the input of state, and output the mean and variance to be the Gaussian policy in all experiments.

For the simulations in the gather and circle tasks, we use a neural network with two hidden layers of size (64, 32).

For the simulations in the grid and bottleneck tasks, we use a neural network with two hidden layers of size (16, 16) and (50, 25), respectively.

We use tanh as the activation function of the neural network.

We use GAE-?? approach (Schulman et al., 2015b) Note that we do not use a learned model to predict the probability of entering an undesirable state within a fixed time horizon as CPO did for cost shaping.

To examine the performance of the algorithms with different metrics, we provide the learning curves of the cumulative constraint value over policy update, and the reward versus the cumulative constraint value for the tested algorithms and task pairs in Section 6 shown in Fig. 7 .

The second metric enables us to compare the reward difference under the same number of cumulative constraint violation.

Overall, we find that, (a) CPO has more cumulative constraint violation than PCPO.

(b) PCPO with L 2 norm projection has less cumulative constraint violation than KL divergence projection except for the point circle and point gather tasks.

This observation suggests that the Fisher information matrix is not well-estimated in the high dimensional policy space, leading to have more constraint violation.

(c) PCPO has more reward improvement compared to CPO under the same number of cumulative constraint violation in point circle, point gather, ant circle, ant gather, and bottleneck task.

Due to approximation errors, CPO performs line search to check whether the updated policy satisfies the trust region and cost constraints.

To understand the necessity of line search in CPO, we conducted the experiment with and without line search shown in Fig. 8 .

The step size ?? is set to 0.01.

We find that CPO without line search tends to (1) have large reward variance especially in the point circle task, and (2) learn constraint-satisfying policies slightly faster.

These observations suggest that line search is more conservative in optimizing the policies since it usually take smaller steps.

However, we conjecture that if using smaller ??, the effect of line search is not significant.

To understand the stability of PCPO and CPO when deployed in more constraint-critical tasks, we increase the difficulty of the task by setting the constraint threshold to zero and reduce the safe area.

The learning curve of discounted reward and constraint value over policy updates are shown in Fig.  9 .

We observe that even with more difficult constraint, PCPO still has more reward improvement and constraint satisfaction than CPO, whereas CPO needs more feasible recovery steps to satisfy the constraint.

In addition, we observe that PCPO with L 2 norm projection has high constraint variance in point circle task, suggesting that the reward update direction is not well aligned with the cost update direction.

We also observe that PCPO with L 2 norm projection converges to a bad local optimum in terms of reward in point gather task, suggesting that in order to satisfy the constraint, the cost update direction destroys the reward update direction.

To learn policies under constraints, PCPO and CPO require to have a good estimation of the constraint set.

However, PCPO may project the policy onto the space that violates the constraint due to the assumption of approximating the constraint set by linear half space constraint.

To understand whether the estimation accuracy of the constraint set affects the performance, we conducted the experiments with batch sample size reducing to 1% of the previous experiments (only 500 samples for each policy update) shown in Fig. 10 .

We find that smaller training samples affects the performance of the algorithm, creating more reward and cost fluctuation.

However, we observe that even with smaller training samples, PCPO still has more reward improvement and constraint satisfaction than CPO.

In the Grid task, we observe that PCPO with KL divergence projection does worse in reward than TRPO, which is expected since TRPO ignores constraints.

However, TRPO actually outperforms PCPO with KL divergence projection in terms of constraint, which is unexpected since by trying to consider the constraint, PCPO with KL divergence projection has made constraint satisfaction worse.

The reason for this observation is that the Fisher information matrix is ill-conditioned, i.e., the condition number ?? max (H)/?? min (H) (?? max is the largest eigenvalue of the matrix) of the Fisher information matrix is large, causing conjugate gradient method that computes constraint update direction H ???1 a with small number of iteration output the inaccurate approximation.

Hence the inaccurate approximation of H ???1 a cause PCPO with KL divergence projection have more constraint violation than TRPO.

To solve this issue, one can have more epochs of conjugate gradient method.

This is because that the convergence of conjugate gradient method is controlled by the condition number (Shewchuk, 1994) ; the larger the condition number is, the more epochs the algorithm needs to get accurate approximation.

In our experiments, we set the number of iteration of conjugate gradient method to be 10 to tradeoff between the computational efficiency and the accuracy across all tested algorithms and task pairs.

To verify our observation, we compare the condition number of the Fisher information matrix, and the approximation error of the constraint update direction over training epochs with different number of iteration of the conjugate gradient method shown in Fig. 11 .

We observe that the Fisher information matrix is ill-conditioned, and the one with larger number of iteration has less error and more constraint satisfaction.

This observation confirms our discussion.

Figure 11: (1) The values of the reward and the constraint, (2) the condition number of the Fisher information matrix, and (3) the approximation error of the constraint update direction over training epochs with the conjugate gradient (CG) method's iteration of 10 and 20, respectively.

The one with larger number of iteration has more constraint satisfaction since it has more accurate approximation. (Best Fig. 12 for illustration.

To compare both stationary points, we consider the following example shown in Fig. 13 .

We maximize a non-convex function f (x) = x T diag(y)x subject to the constraint x T 1 ??? ???1, where y = [5, ???1] T , and 1 is an all-one vector.

An optimal solution to this constrained optimization problem is infinity.

Fig. 13(a) shows the update direction that combines the objective and the cost constraint update directions for both projections.

It shows that PCPO with KL divergence projection has stationary points with g ??? ???a in the boundary of the constraint set (observe that the update direction is zero for PCPO with KL divergence projection at T converges to a local optimum, whereas L 2 norm projection converges to infinity.

However, the above example does not necessary means that PCPO with L 2 norm projection always find a better optimum.

For example, if the gradient direction of the objective is zero in the constraint set or in the boundary, then both projections may converge to the same stationary point.

T (below).

The red star is the initial point, the red arrows are the optimization paths, and the region that is below to the black line is the constraint set.

We see that both projections converge to different solutions.

<|TLDR|>

@highlight

We propose a new algorithm that learns constraint-satisfying policies, and provide theoretical analysis and empirical demonstration in the context of reinforcement learning with constraints.

@highlight

This paper introduces a constrained policy optimization algorithm using a two-step optimization process, where policies that do not satisfy the constraint can be projected back into the constraint set.
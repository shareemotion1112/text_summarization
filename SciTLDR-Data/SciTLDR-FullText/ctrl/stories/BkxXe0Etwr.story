Reinforcement learning (RL) with value-based methods (e.g., Q-learning) has shown success in a variety of domains such as games and recommender systems (RSs).

When the action space is finite, these algorithms implicitly finds a policy by learning the optimal value function, which are often very efficient.

However, one major challenge of extending Q-learning to tackle continuous-action RL problems is that obtaining optimal Bellman backup requires solving a continuous action-maximization (max-Q) problem.

While it is common to restrict the parameterization of the Q-function to be concave in actions to simplify the max-Q problem, such a restriction might lead to performance degradation.

Alternatively, when the Q-function is parameterized with a generic feed-forward neural network (NN), the max-Q problem can be NP-hard.

In this work, we propose the CAQL method which minimizes the Bellman residual using Q-learning with one of several plug-and-play action optimizers.

In particular, leveraging the strides of optimization theories in deep NN, we show that max-Q problem can be solved optimally with mixed-integer programming (MIP)---when the Q-function has sufficient representation power, this MIP-based optimization induces better policies and is more robust than counterparts, e.g., CEM or GA, that approximate the max-Q solution.

To speed up training of CAQL, we develop three techniques, namely (i) dynamic tolerance, (ii) dual filtering, and (iii) clustering.

To speed up inference of CAQL, we introduce the action function that concurrently learns the optimal policy.

To demonstrate the efficiency of CAQL we compare it with state-of-the-art RL algorithms on benchmark continuous control problems that have different degrees of action constraints and show that CAQL significantly outperforms policy-based methods in heavily constrained environments.

Reinforcement learning (RL) has shown success in a variety of domains such as games (Mnih et al., 2013) and recommender systems (RSs) (Gauci et al., 2018) .

When the action space is finite, valuebased algorithms such as Q-learning (Watkins & Dayan, 1992) , which implicitly finds a policy by learning the optimal value function, are often very efficient because action optimization can be done by exhaustive enumeration.

By contrast, in problems with a continuous action spaces (e.g., robotics (Peters & Schaal, 2006) ), policy-based algorithms, such as policy gradient (PG) (Sutton et al., 2000; Silver et al., 2014) or cross-entropy policy search (CEPS) (Mannor et al., 2003; Kalashnikov et al., 2018) , which directly learn a return-maximizing policy, have proven more practical.

Recently, methods such as ensemble critic (Fujimoto et al., 2018) and entropy regularization (Haarnoja et al., 2018) have been developed to improve the performance of policy-based RL algorithms.

Policy-based approaches require a reasonable choice of policy parameterization.

In some continuous control problems, Gaussian distributions over actions conditioned on some state representation is used.

However, in applications such as RSs, where actions often take the form of high-dimensional item-feature vectors, policies cannot typically be modeled by common action distributions.

Furthermore, the admissible action set in RL is constrained in practice, for example, when actions must lie within a specific range for safety (Chow et al., 2018) .

In RSs, the admissible actions are often random functions of the state (Boutilier et al., 2018) .

In such cases, it is non-trivial to define policy parameterizations that handle such factors.

On the other hand, value-based algorithms are wellsuited to these settings, providing potential advantage over policy methods.

Moreover, at least with linear function approximation (Melo & Ribeiro, 2007) , under reasonable assumptions, Q-learning converges to optimality, while such optimality guarantees for non-convex policy-based methods are generally limited (Fazel et al., 2018) .

Empirical results also suggest that value-based methods are more data-efficient and less sensitive to hyper-parameters (Quillen et al., 2018) .

Of course, with large action spaces, exhaustive action enumeration in value-based algorithms can be expensive--one solution is to represent actions with continuous features (Dulac-Arnold et al., 2015) .

The main challenge in applying value-based algorithms to continuous-action domains is selecting optimal actions (both at training and inference time).

Previous work in this direction falls into three broad categories.

The first solves the inner maximization of the (optimal) Bellman residual loss using global nonlinear optimizers, such as the cross-entropy method (CEM) for QT-Opt (Kalashnikov et al., 2018) , gradient ascent (GA) for actor-expert (Lim et al., 2018) , and action discretization (Uther & Veloso, 1998; Smart & Kaelbling, 2000; Lazaric et al., 2008) .

However, these approaches do not guarantee optimality.

The second approach restricts the Q-function parameterization so that the optimization problem is tractable.

For instance, wire-fitting (Gaskett et al., 1999; III & Klopf, 1993) approximates Q-values piecewise-linearly over a discrete set of points, chosen to ensure the maximum action is one of the extreme points.

The normalized advantage function (NAF) (Gu et al., 2016) constructs the state-action advantage function to be quadratic, hence analytically solvable.

Parameterizing the Q-function with an input-convex neural network (Amos et al., 2017) ensures it is concave.

These restricted functional forms, however, may degrade performance if the domain does not conform to the imposed structure.

The third category replaces optimal Q-values with a "soft" counterpart (Haarnoja et al., 2018) : an entropy regularizer ensures that both the optimal Q-function and policy have closed-form solutions.

However, the sub-optimality gap of this soft policy scales with the interval and dimensionality of the action space (Neu et al., 2017) .

Motivated by the shortcomings of prior approaches, we propose Continuous Action Q-learning (CAQL), a Q-learning framework for continuous actions in which the Q-function is modeled by a generic feed-forward neural network.

1 Our contribution is three-fold.

First, we develop the CAQL framework, which minimizes the Bellman residual in Q-learning using one of several "plug-andplay" action optimizers.

We show that "max-Q" optimization, when the Q-function is approximated by a deep ReLU network, can be formulated as a mixed-integer program (MIP) that solves max-Q optimally.

When the Q-function has sufficient representation power, MIP-based optimization induces better policies and is more robust than methods (e.g., CEM, GA) that approximate the max-Q solution.

Second, to improve CAQL's practicality for larger-scale applications, we develop three speed-up techniques for computing max-Q values: (i) dynamic tolerance; (ii) dual filtering; and (iii) clustering.

Third, we compare CAQL with several state-of-the-art RL algorithms on several benchmark problems with varying degrees of action constraints.

Value-based CAQL is generally competitive, and outperforms policy-based methods in heavily constrained environments, sometimes significantly.

We also study the effects of our speed-ups through ablation analysis.

We consider an infinite-horizon, discounted Markov decision process (Puterman, 2014) with states X, (continuous) action space A, reward function R, transition kernel P , initial state distribution β and discount factor γ ∈ [0, 1), all having the usual meaning.

A (stationary, Markovian) policy π specifies a distribution π(·|x) over actions to be taken at state x. Let ∆ be the set of such policies.

The expected cumulative return of π ∈ ∆ is J(π) := E[ ∞ t=0 γ t r t | P, R, x 0 ∼ β, π].

An optimal policy π * satisfies π * ∈ arg max π∈∆ J(π).

The Bellman operator F [Q](x, a) = R(x, a) + γ x ∈X P (x |x, a) max a ∈A Q(x , a ) over state-action value function Q has unique fixed point Q * (x, a) (Puterman, 2014) , which is the optimal Q-function Q * (x, a) = E [ ∞ t=0 γ t R(x t , a t ) | x 0 = x, a 0 = a, π * ].

An optimal (deterministic) policy π * can be extracted from Q * : π * (a|x) = 1{a = a * (x)}, where a * (x) ∈ arg max a Q * (x, a).

For large or continuous state/action spaces, the optimal Q-function can be approximated, e.g., using a deep neural network (DNN) as in DQN (Mnih et al., 2013) .

In DQN, the value function Q θ is updated using the value label r + γ max a Q θ target (x , a ), where Q θ target is a target Q-function.

Instead of training these weights jointly, θ target is updated in a separate iterative fashion using the previous θ for a fixed number of training steps, or by averaging θ target ← τ θ + (1 − τ )θ target for some small momentum weight τ ∈ [0, 1] (Mnih et al., 2016) .

DQN is off-policy-the target is valid no matter how the experience was generated (as long as it is sufficiently exploratory).

Typically, the loss is minimized over mini-batches B of past data (x, a, r, x ) sampled from a large experience replay buffer R (Lin & Mitchell, 1992) .

One common loss function for training Q θ * is mean squared Bellman error:

2 .

Under this loss, RL can be viewed as 2 -regression of Q θ (·, ·) w.r.t.

target labels r + γ max a Q θ target (x , a ).

We augment DQN, using double Q-learning for more stable training (Hasselt et al., 2016) , whose loss is:

A hinge loss can also be used in Q-learning, and has connections to the linear programming (LP) formulation of the MDP (Puterman (2014) ).

The optimal Q-network weights can be specified as:

To stabilize training, we replace the Q-network of the inner maximization with the target Q-network and the optimal Q-value with the double-Q label, giving (see Appendix A for details):

In this work, we assume the Q-function approximation Q θ to be a feed-forward network.

Specifically, let Q θ be a K-layer feed-forward NN with state-action input (x, a) (where a lies in a ddimensional real vector space) and hidden layers arranged according to the equations:

where (W j , b j ) are the multiplicative and bias weights, c is the output weight of the Q-network,

are the weights of the Q-network,ẑ j denotes pre-activation values at layer j, and h(·) is the (component-wise) activation function.

For simplicity, in the following analysis, we restrict our attention to the case when the activation functions are ReLU's.

We also assume that the action space A is a d-dimensional ∞ -ball B ∞ (a, ∆) with some radius ∆ > 0 and center a. Therefore, at any arbitrary state x ∈ X the max-Q problem can be re-written as q *

While the above formulation is intuitive, the nonlinear equality constraints in the neural network formulation (3) makes this problem non-convex and NP-hard (Katz et al., 2017) .

Policy-based methods (Silver et al., 2014; Fujimoto et al., 2018; Haarnoja et al., 2018) have been widely-used to handle continuous actions in RL.

However, they suffer from several well-known difficulties, e.g., (i) modeling high-dimensional action distributions, (ii) handling action constraints, and (iii) data-inefficiency.

Motivated by earlier work on value-based RL methods, such as QTOpt (Kalashnikov et al., 2018) and actor-expert (Lim et al., 2018) , we propose Continuous Action Q-learning (CAQL), a general framework for continuous-action value-based RL, in which the Qfunction is parameterized by a NN (Eq. 3).

One novelty of CAQL is the formulation of the "max-Q" problem, i.e., the inner maximization in (1) and (2), as a mixed-integer programming (MIP).

The benefit of the MIP formulation is that it guarantees that we find the optimal action (and its true bootstrapped Q-value) when computing target labels (and at inference time).

We show empirically that this can induce better performance, especially when the Q-network has sufficient representation power.

Moreover, since MIP can readily model linear and combinatorial constraints, it offers considerable flexibility when incorporating complex action constraints in RL.

That said, finding the optimal Q-label (e.g., with MIP) is computationally intensive.

To alleviate this, we develop several approximation methods to systematically reduce the computational demands of the inner maximization.

In Sec. 3.2, we introduce the action function to approximate the arg max-policy at inference time, and in Sec. 4 we propose three techniques, dynamic tolerance, dual filtering, and clustering, to speed up max-Q computation during training.

In this section, we illustrate how the max-Q problem, with the Q-function represented by a ReLU network, can be formulated as a MIP, which can be solved using off-the-shelf optimization packages (e.g., SCIP (Gleixner et al., 2018) , CPLEX (CPLEX, 2019) , Gurobi (Gurobi, 2019) ).

In addition, we detail how approximate optimizers, specifically, gradient ascent (GA) and the cross-entropy method (CEM), can trade optimality for speed in max-Q computation within CAQL.

A trained feed-forward ReLU network can be modeled as a MIP by formulating the nonlinear activation function at each neuron with binary constraints.

Specifically, for a ReLU with pre-activation function of form z = max{0, w x + b}, where

and , u ∈ R d are the weights, bias and lower-upper bounds respectively, consider the following set with a binary variable ζ indicating whether the ReLU is active or not:

.

In this formulation, both M + = max x∈[ ,u] w x + b and M − = min x∈[ ,u] w x + b can be computed in linear time in d. We assume M + > 0 and M − < 0, otherwise the function can be replaced by z = 0 or z = w x + b. These constraints ensure that z is the output of the ReLU: If ζ = 0, then they are reduced to z = 0 ≥ w x+b, and if ζ = 1, then they become z = w x+b ≥ 0.

This can be extended to the ReLU network in (3) by chaining copies of intermediate ReLU formulations.

More precisely, if the ReLU Q-network has m j neurons in layer j ∈ {2, . . .

, K}, for any given state x ∈ X, the max-Q problem can be reformulated as the following MIP:

, . . . , K}, i ∈ {1, . . . , m j }, where 1 = a − ∆, u 1 = a + ∆ are the (action) input-bound vectors.

Since the output layer of the ReLU NN is linear, the MIP objective is linear as well.

Here, W j,i ∈ R mj and b j,i ∈ R are the weights and bias of neuron i in layer j. Furthermore, j , u j are interval bounds for the outputs of the neurons in layer j for j ≥ 2, and computing them can be done via interval arithmetic or other propagation methods (Weng et al., 2018) from the initial action space bounds (see Appendix C for details).

As detailed by Anderson et al. (2019) , this can be further tightened with additional constraints, and its implementation can be found in the tf.opt package described therein.

As long as these bounds are redundant, having these additional box constraints will not affect optimality.

We emphasize that the MIP returns provably global optima, unlike GA and CEM.

Even when interrupted with stopping conditions such as a time limit, MIP often produces high-quality solutions in practice.

In theory, this MIP formulation can be solved in time exponential on the number of ReLUs and polynomial on the input size (e.g., by naively solving an LP for each binary variable assignment).

In practice however, a modern MIP solver combines many different techniques to significantly speed up this process, such as branch-and-bound, cutting planes, preprocessing techniques, and primal heuristics (Linderoth & Savelsbergh, 1999) .

Versions of this MIP model have been used in neural network verification (Cheng et al., 2017; Lomuscio & Maganti, 2017; Bunel et al., 2018; Dutta et al., 2018; Fischetti & Jo, 2018; Anderson et al., 2019; Tjeng et al., 2019) and analysis (Serra et al., 2018; Kumar et al., 2019) , but its application to RL is novel.

While Say et al. (2017) also proposed a MIP formulation to solve the planning problem with non-linear state transition dynamics model learned with a NN, it is different than ours, which solves the max-Q problem.

Gradient Ascent GA (Nocedal & Wright, 2006 ) is a simple first-order optimization method for finding the (local) optimum of a differentiable objective function, such as a neural network Qfunction.

At any state x ∈ X, given a "seed" action a 0 , the optimal action arg max a Q θ (x, a) is computed iteratively by a t+1 ← a t + η∇ a Q θ (x, a), where η > 0 is a step size (either a tunable parameter or computed using back-tracking line search (Nocedal & Yuan, 1998) ).

This process repeats until convergence, |Q θ (x, a t+1 ) − Q θ (x, a t )| < , or a maximum iteration count is reached.

Cross-Entropy Method CEM (Rubinstein, 1999 ) is a derivative-free optimization algorithm.

At any given state x ∈ X, it samples a batch of N actions {a i } N i=1 from A using a fixed distribution (e.g., a Gaussian) and ranks the corresponding Q-values

.

Using the top K < N actions, it then updates the sampling distribution, e.g., using the sample mean and covariance to update the Gaussian.

This is repeated until convergence or a maximum iteration count is reached.

In traditional Q-learning, the policy π * is "implemented" by acting greedily w.r.t.

the learned Qfunction: π * (x) = arg max a Q θ (x, a).

3 However, computing the optimal action can be expensive in the continuous case, which may be especially problematic at inference time (e.g., when computational power is limited in, say embedded systems, or real-time response is critical).

To mitigate the problem, we can use an action function π w : X → A-effectively a trainable actor network-to approximate the greedy-action mapping π * .

We train π w using training data B = {(

, where q * i is the max-Q label at state x i .

Action function learning is then simply a supervised regression problem:

2 .

This is similar to the notion of "distilling" an optimal policy from max-Q labels, as in actor-expert (Lim et al., 2018) .

Unlike actorexpert-a separate stochastic policy network is jointly learned with the Q-function to maximize the likelihood with the underlying optimal policy-our method learns a state-action mapping to approximate arg max a Q θ (x, a)-this does not require distribution matching and is generally more stable.

The use of action function in CAQL is simply optional to accelerate data collection and inference.

In this section, we propose three methods to speed up the computationally-expensive max-Q solution during training: (i) dynamic tolerance, (ii) dual filtering, and (iii) clustering.

Dynamic Tolerance Tolerance plays a critical role in the stopping condition of nonlinear optimizers.

Intuitively, in the early phase of CAQL, when the Q-function estimate has high Bellman error, it may be wasteful to compute a highly accurate max-Q label when a crude estimate can already guide the gradient of CAQL to minimize the Bellman residual.

We can speed up the max-Q solver by dynamically adjusting its tolerance τ > 0 based on (a) the TD-error, which measures the estimation error of the optimal Q-function, and (b) the training step t > 0, which ensures the bias of the gradient (induced by the sub-optimality of max-Q solver) vanishes asymptotically so that CAQL converges to a stationary point.

While relating tolerance with the Bellman residual is intuitive, it is impossible to calculate that without knowing the max-Q label.

To resolve this circular dependency, notice that the action function π w approximates the optimal policy, i.e., π w (·|x) ≈ arg max a Q θ (x, ·).

We therefore replace the optimal policy with the action function in Bellman residual and propose the dynamic tolerance:

, where k 1 > 0 and k 2 ∈ [0, 1) are tunable parameters.

Under standard assumptions, CAQL with dynamic tolerance {τ t } converges a.s.

to a stationary point (Thm. 1, (Carden, 2014) ).

The main motivation of dual filtering is to reduce the number of max-Q problems at each CAQL training step.

For illustration, consider the formulation of hinge Q-learning in (2).

Denote by q * x ,θ target the max-Q label w.r.t.

the target Q-network and next state x .

The structure of the hinge penalty means the TD-error corresponding to sample (x, a, x , r) is inactive whenever q * x ,θ target ≤ (Q θ (x, a) − r)/γ-this data can be discarded.

In dual filtering, we efficiently estimate an upper bound on q * x ,θ target using some convex relaxation to determine which data can be discarded before max-Q optimization.

Specifically, recall that the main source of non-convexity in (3) comes from the equality constraint of the ReLU activation function at each NN layer.

Similar to MIP formulation, assume we have component-wise bounds

We use this approximation to define the relaxed NN equations, which replace the nonlinear equality constraints in (3) with the convex set H(l, u).

We denote the optimal Q-value w.r.t.

the relaxed NN asq * x , which is by definition an upper bound on q * x i

.

Hence, the condition:q * x ,θ target ≤ (Q θ (x, a) − r)/γ is a conservative certificate for checking whether the data (x, a, x , r) is inactive.

For further speed up, we estimateq * x with its dual upper bound (see Appendix C for derivations)q

, where ν is defined by the following recursion "dual" network:

) ·

1{s ∈ I j }, and replace the above certificate with an even more conservative one:

Although dual filtering is derived for hinge Q-learning, it also applies to the 2 -loss counterpart by replacing the optimal value q One can utilize the inactive samples in the π w -learning problem by replacing the max-Q label q * x ,θ with its dual approximation q x ,θ .

Since q x ,θ ≥ q * x ,θ , this replacement will not affect optimality.

Clustering To reduce the number of max-Q solves further still, we apply online state aggregation (Meyerson, 2001) , which picks a number of centroids from the batch of next states B as the centers of p-metric balls with radius b > 0, such that the union of these balls form a minimum covering of B .

Specifically, at training step t ∈ {0, 1, . . .}, denote by C t (b) ⊆ B the set of next-state centroids.

For each next state c ∈ C t (b), we compute the max-Q value q * c ,θ target = max a Q θ target (c , a ), where a * c is the corresponding optimal action.

For all remaining next states x ∈ B \ C t (b), we approximate their max-Q values via first-order Taylor series expansionq x ,θ target := q centroid to x , i.e., c ∈ arg min c ∈Ct(b) x −c p .

By the envelope theorem for arbitrary choice sets (Milgrom & Segal, 2002)

In this approach the cluster radius r > 0 controls the number of max-Q computations, which trades complexity for accuracy in Bellman residual estimation.

This parameter can either be a tuned or adjusted dynamically (similar to dynamic tolerance), e.g., r t = k 3 · k t 4 with hyperparameters k 3 > 0 and k 4 ∈ [0, 1).

Analogously, with this exponentially-decaying cluster radius schedule we can argue that the bias of CAQL gradient (induced by max-Q estimation error due to clustering) vanishes asymptotically, and the corresponding Q-function converges to a stationary point.

To combine clustering with dual filtering, we define B df as the batch of next states that are inconclusive after dual filtering, i.e., B df = {x ∈ B : q x ,θ target > (Q θ (x, a) − r)/γ}. Then instead of applying clustering to B we apply this method onto the refined batch B df .

To illustrate the effectiveness of CAQL, we (i) compare several CAQL variants with several state-ofthe-art RL methods on multiple domains, and (ii) assess the trade-off between max-Q computation speed and policy quality via ablation analysis.

Comparison with Baseline RL Algorithms We compare CAQL with three baseline methods, DDPG (Silver et al., 2014) and TD3 (Fujimoto et al., 2018 )-two popular policy-based deep RL algorithms-and NAF (Gu et al., 2016) , a value-based method using an action-quadratic Q-function.

We train CAQL using three different max-Q optimizers, MIP, GA, and CEM.

Note that CAQL-CEM counterpart is similar to QT-Opt (Kalashnikov et al., 2018) and CAQL-GA reflects some aspects actor-expert (Lim et al., 2018) .

These CAQL variants allow assessment of the degree to which policy quality is impacted by Q-learning with optimal Bellman residual (using MIP) rather than an approximation (using GA or CEM), at the cost of steeper computation.

To match the implementations of the baselines, we use 2 loss when training CAQL.

Further ablation analysis on CAQL with 2 loss vs. hinge loss is provided in Appendix E. We evaluate CAQL on one classical control benchmark (Pendulum) and five MuJoCo benchmarks (Hopper, Walker2D, HalfCheetah, Ant, Humanoid).

Different than most previous work, we evaluate the RL algorithms on domains not just with default action ranges, but also using smaller, constrained action ranges (see Table 6 in Appendix D for action ranges used in our experiments).

4 The motivation for this is two-fold: (i) To simulate real-world problems (Dulac-Arnold et al., 2019) , where the restricted ranges represent the safe/constrained action sets; (ii) To validate the hypothesis that action-distribution learning in policy-based methods cannot easily handle such constraints, while CAQL does so, illustrating its flexibility.

We reduce episode limits from 1000 to 200 steps and use small networks to accommodate the MIP.

Both changes lead to lower returns than that reported in state-of-the-art RL benchmarks (Duan et al., 2016) .

Details on network architectures and hyperparameters are described in Appendix D.

Policy performance is evaluated every 1000 training iterations, using a policy with no exploration.

Each measurement is an average return over 10 episodes, each generated using a separate random seed.

To smooth learning curves, data points are averaged over a sliding window of size 3.

Similar to the setting of Lim et al. (2018) , CAQL measurements are based on trajectories that are generated by the learned action function instead of the optimal action w.r.t.

the Q-function.

Table 1 shows the average return of CAQL and the baselines under the best hyperparameter configurations.

CAQL significantly outperforms NAF on most benchmarks, as well as DDPG and TD3 on 10 of 14 benchmarks.

Of all the CAQL policies, those trained using MIP are among the best performers in all the benchmarks except Ant [-0.25, 0 .25] and Humanoid [-0.25, 0.25] .

This verifies our conjecture about CAQL: Q-learning with optimal Bellman residual (using MIP) performs better than using approximation (using GA, CEM) when the Q-function has sufficient representation power (which is more likely in low-dimensional tasks).

Moreover, CAQL-MIP policies have slightly lower variance than those trained with GA and CEM on most benchmarks.

Table 2 shows summary statistics of the returns of CAQL and the baselines on all 320 configurations (32 hyperparameter combinations × 10 random seeds) and illustrates the sensitivity to hyperparameters of each method.

CAQL is least sensitive in 13 of 14 tasks, and policies trained using MIP optimization, specifically, are best in 8 of 14 tasks.

This corroborates the hypothesis that value- Table 2 : The mean ± standard deviation of (95-percentile) final returns over all 320 configurations (32 hyper parameter combinations×10 random seeds).

The full training curves are given in Figure 4 in Appendix E. CAQL-MIP policies are least sensitive to hyper parameters on 8/14 benchmarks.

based methods are generally more robust to hyperparameters than their policy-based counterparts.

Table 9 in Appendix E.1 compares the speed (in terms of average elapsed time) of various max-Q solvers (MIP, GA, and CEM), with MIP clearly the most computationally intensive.

We note that CAQL-MIP suffers from performance degradation in several high-dimensional environments with large action ranges (e.g., Ant [-0.25, 0 .25] and Humanoid [-0.25, 0.25] ).

In these experiments, its performance is even worse than that of CAQL-GA or CAQL-CEM.

We speculate that this is due to the fact that the small ReLU NN (32 × 16) doesn't have enough representation power to accurately model the Q-functions in more complex tasks, and therefore optimizing for the true max-Q value using an inaccurate function approximation impedes learning.

Ablation Analysis We now study the effects of using dynamic tolerance, dual filtering, and clustering on CAQL via two ablation analyses.

For simplicity, we experiment on standard benchmarks (with full action ranges), and primarily test CAQL-GA using an 2 loss.

Default values on tolerance and maximum iteration are 1e-6 and 200, respectively.

Table 3 shows how reducing the number of max-Q problems using dual filtering and clustering affects performance of CAQL.

Dual filtering (DF) manages to reduce the number of max-Q problems (from 3.2% to 26.5% across different benchmarks), while maintaining similar performance with the unfiltered CAQL-GA.

On top of dual filtering we apply clustering (C) to the set of inconclusive next states B df , in which the degree of approximation is controlled by the cluster radius.

With a small cluster radius (e.g., b = 0.1), clustering further reduces max-Q solves without significantly impacting training performance (and in some cases it actually improves performance), though further increasing the radius would significant degrade performance.

To illustrate the full trade-off of max-Q reduction versus policy quality, we also include the Dual method, which eliminates all max-Q computation with the dual approximation.

Table 4 shows how dynamic tolerance influences the quality of CAQL policies.

Compared with the standard algorithm, with a large tolerance (τ = 100) GA achieves a notable speed up (with only 1 step per max-Q optimization) in training but incurs a loss in performance.

GA with dynamic tolerance atttains the best of both worlds-it significantly Table 3 : Ablation analysis on CAQL-GA with dual filtering and clustering, where both the mean ± standard deviation of (95-percentile) final returns and the average %-max-Q-reduction (in parenthesis) are based on the best configuration.

See Figure 5 in Appendix E for training curves.

Table 4 : Ablation analysis on CAQL-GA with dynamic tolerance, where both the mean ± standard deviation of (95-percentile) final returns and the average number of GA iterations (in parenthesis) are based on the best configuration.

See Figure 7 in Appendix E for training curves.

NOTE: In ( * ) the performance significantly drops after hitting the peak, and learning curve does not converge.

reduces inner-maximization steps (from 29.5% to 77.3% across different problems and initial τ settings), while achieving good performance.

Additionally, Table 5 shows the results of CAQL-MIP with dynamic tolerance (i.e., optimality gap).

This method significantly reduces both median and variance of the MIP elapsed time, while having better performance.

Dynamic tolerance eliminates the high latency in MIP observed in the early phase of training (see Figure 1 and 2).

Table 5 : Ablation analysis on CAQL-MIP with dynamic tolerance, where both the mean ± standard deviation of (95-percentile) final returns and the (median, standard deviation) of the elapsed time κ (in msec) are based on the best configuration.

See Figure 11 in Appendix E for training curves.

We proposed Continuous Action Q-learning (CAQL), a general framework for handling continuous actions in value-based RL, in which the Q-function is parameterized by a neural network.

While generic nonlinear optimizers can be naturally integrated with CAQL, we illustrated how the inner maximization of Q-learning can be formulated as mixed-integer programming when the Qfunction is parameterized with a ReLU network.

CAQL (with action function learning) is a general Q-learning framework that includes many existing value-based methods such as QT-Opt and actorexpert.

Using several benchmarks with varying degrees of action constraint, we showed that the policy learned by CAQL-MIP generally outperforms those learned by CAQL-GA and CAQL-CEM; and CAQL is competitive with several state-of-the-art policy-based RL algorithms, and often outperforms them (and is more robust) in heavily-constrained environments.

Future work includes: extending CAQL to the full batch learning setting, in which the optimal Q-function is trained using only offline data; speeding up the MIP computation of the max-Q problem to make CAQL more scalable; and applying CAQL to real-world RL problems.

Consider an MDP with states X, actions A, transition probability function P , discount factor γ ∈ [0, 1), reward function R, and initial state distribution β.

We want to find an optimal Q-function by solving the following optimization problem:

The formulation is based on the LP formulation of MDP (see Puterman (2014) for more details).

Here the distribution p(x, a) is given by the data-generating distribution of the replay buffer B.

(We assume that the replay buffer is large enough such that it consists of experience from almost all state-action pairs.)

It is well-known that one can transform the above constrained optimization problem into an unconstrained one by applying a penalty-based approach (to the constraints).

For simplicity, here we stick with a single constant penalty parameter λ ≥ 0 (instead of going for a state-action Lagrange multiplier and maximizing that), and a hinge penalty function (·) + .

With a given penalty hyper-parameter λ ≥ 0 (that can be separately optimized), we propose finding the optimal Q-function by solving the following optimization problem:

Furthermore, recall that in many off-policy and offline RL algorithms (such as DQN), samples in form of

are independently drawn from the replay buffer, and instead of the optimizing the original objective function, one goes for its unbiased sample average approximation (SAA).

However, viewing from the objective function of problem (6), finding an unbiased SAA for this problem might be challenging, due to the non-linearity of hinge penalty function (·) + .

Therefore, alternatively we turn to study the following unconstrained optimization problem:

Using the Jensen's inequality for convex functions, one can see that the objective function in (7) is an upper-bound of that in (6).

Equality of the Jensen's inequality will hold in the case when transition function is deterministic.

(This is similar to the argument of PCL algorithm.) Using Jensen's inequality one justifies that optimization problem (7) is indeed an eligible upper-bound optimization to problem (6).

Recall that p(x, a) is the data-generation distribution of the replay buffer B. The unbiased SAA of problem (7) is therefore given by

where

are the N samples drawn independently from the replay buffer.

In the following, we will find the optimal Q function by solving this SAA problem.

In general when the state and action spaces are large/uncountable, instead of solving the Q-function exactly (as in the tabular case), we turn to approximate the Q-function with its parametrized form Q θ , and optimize the set of real weights θ (instead of Q) in problem (8).

Sample an initial state x 0 from the initial distribution 8:

Select action a = clip(π w (x ) + N (0, σ), l, u)

10:

Execute action a and observe reward r and new state x +1

Store transition (x , a , r , x +1 ) in Replay Buffer R

for s ← 1, . . .

, S do CAQL Training; S = 20 by default 13:

Sample a random minibatch B of |B| transitions {(

Initialize the refined batches B df ← B and B c ← B;

For each (x i , a i , r i , x i ) ∈ B df ∩ B c , compute optimal action a i using OPT(DTol):

and the corresponding TD targets:

For each (x i , a i , r i , x i ) ∈ B \ (B c ∩ B df ), compute the approximate TD target:

Update the Q-function parameters:

Update the action function parameters:

Update the target Q-function parameters:

Decay the Gaussian noise:

Recall that the Q-function NN has a nonlinear activation function, which can be viewed as a nonlinear equality constraint, according to the formulation in (3).

To tackle this constraint, Wong & Kolter (2017) proposed a convex relaxation of the ReLU non-linearity.

Specifically, first, they assume that for given x ∈ X and a ∈ B ∞ (a) such that z 1 = (x , a ), there exists a collection of component-wise bounds (l j , u j ), j = 2, . . .

, K − 1 such that l j ≤ẑ j ≤ u j .

As long as the bounds are redundant, adding these constraints into primal problem q * x does not affect the optimal value.

Second, the ReLU non-linear equality constraint is relaxed using a convex outer-approximation.

In particular, for a scalar input a within the real interval [l, u] , the exact ReLU non-linearity acting on a is captured by the set

Its convex outer-approximation is given by:

(9) Analogously to (3), define the relaxed NN equations as:

where the third equation above is understood to be component-wise across layer j for each j ∈ {2, . . .

, K − 1}, i.e.,

where n j is the dimension of hidden layer j. Using the relaxed NN equations, we now propose the following relaxed (convex) verification problem:

where δ Λ (·) is the indicator function for set Λ (i.e., δ Λ (x) = 0 if x ∈ Λ and ∞ otherwise).

Note that the indicator for the vector-ReLU in cost function above is understood to be component-wise, i.e.,

The optimal value of the relaxed problem, i.e.,q *

x is an upper bound on the optimal value for original problem q * x i

.

Thus, the certification one can obtain is the following: ifq * x ≤ (Q θ (x, a) − r)/γ, then the sample (x, a, x ) is discarded for inner maximization.

However, ifq * x > (Q θ (x, a) − r)/γ, the sample (x, a, x ) may or may not have any contribution to the TD-error in the hinge loss function.

To further speed up the computation of the verification problem, by looking into the dual variables of problem (11), in the next section we propose a numerically efficient technique to estimate a suboptimal, upper-bound estimate toq * x , namely q x .

Therefore, one verification criterion on whether a sample drawn from replay buffer should be discarded for inner-maximization is check whether the following inequality holds:

C.1 SUB-OPTIMAL SOLUTION TO THE RELAXED PROBLEM

In this section, we detail the sub-optimal lower bound solution to the relaxed problem in (11) as proposed in Wong & Kolter (2017) .

Let ν j , j = 2, . . .

, K denote the dual variables for the linear equality constraints in problem (11).

The Lagrangian for the relaxed problem in (11) is given by:

Defineν j :

= W j ν j+1 for j = 1, . . .

, K − 1, and defineν

Recall that for a real vector space X ⊆ R n , let X * denote the dual space of X with the standard pairing ·, · : X × X * → R. For a real-valued function f : X → R ∪ {∞, −∞}, let f * : X * → R ∪ {∞, −∞} be its convex conjugate, defined as: f * (y) = − inf x∈X (f (x) − y, x ) = sup x∈X ( y, x − f (x)), ∀y ∈ X * .

Therefore, the conjugate for the vector-ReLU indicator above takes the following component-wise structure:

Now, the convex conjugate of the set indicator function is given by the set support function.

Thus,

where · q is the l p -dual norm defined by the identity 1/p + 1/q = 1.

To compute the convex conjugate for the ReLU relaxation, we analyze the scalar definition as provided in (9).

Specifically, we characterize δ H (l,u) (p, q) defined by the scalar bounds (l, u), for the dual vector (p, q) ∈ R 2 .

There exist 3 possible cases:

Case I: l < u ≤ 0:

Case II: 0 ≤ l < u:

Case III: l < 0 < u: For this case, the sup will occur either on the line −ux + (u − l)y = −ul or at the origin.

Thus,

Applying these in context of equation (14), we calculate the Lagrange multipliers by considering the following cases.

Case I: l j (s) < u j (s) ≤ 0: In this case, since z j (s) = 0 regardless of the value ofẑ j (s), one can simply remove these variables from problems (10) and (11) by eliminating the j th row of W j−1 and b j−1 and the j th column of W i .

Equivalently, from (15), one can remove their contribution in (13) by setting ν j (s) = 0.

Case II: 0 ≤ l j (s) < u j (s): In this case, the ReLU non-linearity for (ẑ j (s), z j (s)) in problems (10) and (11) may be replaced with the convex linear equality constraint z j (s) =ẑ j (s) with associated dual variable µ. Within the Lagrangian, this would result in a modification of the term Minimizing this over (ẑ j (s), z j (s) ), a non-trivial lower bound (i.e., 0) is obtained only if ν j (s) = ν j (s) = µ. Equivalently, from (16), we set ν j (s) =ν j (s).

Case III: For the non-trivial third case, where l j (s) < 0 < u j (s), notice that due toν, the dual function g(ν) is not decoupled across the layers.

In order to get a sub-optimal, but analytical solution to the dual optimization, we will optimize each term within the first sum in (13) independently.

To do this, notice that the quantity in sub-case I in (17) is strictly greater than the other two sub-cases.

Thus, the best bound is obtained by using the third sub-case, which corresponds to setting:

.

Combining all the previous analysis, we now calculate the dual of the solution to problem (11).

Let I

Using the above case studies, a sub-optimal (upper-bound) dual solution to the primal solutionJ x in problem (11) is given by

where ν is defined by the following recursion, termed the "dual" network:

and D j is a diagonal matrix with

C.2 COMPUTING PRE-ACTIVATION BOUNDS For k ∈ {3, . . .

, K − 1}, define the k−partial NN as the set of equations:

Finding the lower bound l k forẑ k involves solving the following problem:

where e s is a one-hot vector with the non-zero element in the s-th entry, for s ∈ {1, . . .

, n k }.

Similarly, we obtain u k by maximizing the objective above.

Assuming we are given bounds {l j , u j } k−1 j=2 , we can employ the same convex relaxation technique and approximate dual solution as for the verification problem (since we are simply optimizing a linear function of the output of the first k layers of the NN).

Doing this recursively allows us to compute the bounds {l j , u j } for j = 3, . . .

, K − 1.

The recursion is given in Algorithm 1 in Wong & Kolter (2017) and is based on the matrix form of the recursion in (19), i.e., with c replaced with I and −I, so that the quantity in (18) We use a two hidden layer neural network with ReLU activation (32 units in the first layer and 16 units in the second layer) for both the Q-function and the action function.

The input layer for the Q-function is a concatenated vector of state representation and action variables.

The Q-function has a single output unit (without ReLU).

The input layer for the action function is only the state representation.

The output layer for the action function has d units (without ReLU), where d is the action dimension of a benchmark environment.

We use SCIP 6.0.0 (Gleixner et al., 2018) for the MIP solver.

A time limit of 60 seconds and a optimality gap limit of 10 −4 are used for all experiments.

For GA and CEM, a maximum iterations of 20 and a convergence threshold of 10

are used for all experiments.

E ADDITIONAL EXPERIMENTAL RESULTS E.1 OPTIMIZER SCALABILITY Table 9 shows the average elapsed time of various optimizers computing max-Q in the experiment setup described in Appendix D. MIP is more robust to action dimensions than GA and CEM.

MIP latency depends on the state of neural network weights.

It takes longer time with highly dense NN weights, but on the other hand, it can be substantially quicker with sparse NN weights.

Figure 1 shows the average elapsed time of MIP over training steps for various benchmarks.

We have observed that MIP is very slow in the beginning of the training phase but it quickly becomes faster.

This trend is observed for most benchmarks except Humanoid.

We speculate that the NN weights for the Q-function are dense in the beginning of the training phase, but it is gradually structurized (e.g, sparser weights) so that it becomes an easier problem for MIP.

Table 9 : The (median, standard deviation) for the average elapsed time κ (in msec) of various solvers computing max-Q problem.

(n) Humanoid [-0.25,0.25] Figure 4: The mean return over all 320 configurations (32 hyper parameter combinations × 10 random seeds).

Shaded area is ± standard deviation.

Data points are average over a sliding window of size 3.

The length of an episode is limited to 200 steps.

Table 11 : Ablation analysis on CAQL-GA with dynamic tolerance, where both the mean ± standard deviation of (95-percentile) final returns and the average number of GA iterations (in parenthesis) are over all 320 configurations.

See Figure 8 in Appendix E for training curves.

Table 13 : The mean ± standard deviation of (95-percentile) final returns over all 320 configurations (32 hyper parameter combinations × 10 random seeds).

The full training curves are given in Figure 10 in Appendix E.

<|TLDR|>

@highlight

A general framework of value-based reinforcement learning for continuous control
Trust region methods, such as TRPO, are often used to stabilize policy optimization algorithms in reinforcement learning (RL).

While current trust region strategies are effective for continuous control, they typically require a large amount of on-policy interaction with the environment.

To address this problem, we propose an off-policy trust region method, Trust-PCL, which exploits an observation that the optimal policy and state values of a maximum reward objective with a relative-entropy regularizer satisfy a set of multi-step pathwise consistencies along any path.

The introduction of relative entropy regularization allows Trust-PCL to maintain optimization stability while exploiting off-policy data to improve sample efficiency.

When evaluated on a number of continuous control tasks, Trust-PCL significantly improves the solution quality and sample efficiency of TRPO.

The goal of model-free reinforcement learning (RL) is to optimize an agent's behavior policy through trial and error interaction with a black box environment.

Value-based RL algorithms such as Q-learning BID36 and policy-based algorithms such as actor-critic BID15 have achieved well-known successes in environments with enumerable action spaces and predictable but possibly complex dynamics, e.g., as in Atari games BID19 BID34 .

However, when applied to environments with more sophisticated action spaces and dynamics (e.g., continuous control and robotics), success has been far more limited.

In an attempt to improve the applicability of Q-learning to continuous control, BID32 and BID16 developed an off-policy algorithm DDPG, leading to promising results on continuous control environments.

That said, current off-policy methods including DDPG often improve data efficiency at the cost of optimization stability.

The behaviour of DDPG is known to be highly dependent on hyperparameter selection and initialization BID18 ; even when using optimal hyperparameters, individual training runs can display highly varying outcomes.

On the other hand, in an attempt to improve the stability and convergence speed of policy-based RL methods, BID13 developed a natural policy gradient algorithm based on Amari (1998), which subsequently led to the development of trust region policy optimization (TRPO) BID28 .

TRPO has shown strong empirical performance on difficult continuous control tasks often outperforming value-based methods like DDPG.

However, a major drawback is that such methods are not able to exploit off-policy data and thus require a large amount of on-policy interaction with the environment, making them impractical for solving challenging real-world problems.

Efforts at combining the stability of trust region policy-based methods with the sample efficiency of value-based methods have focused on using off-policy data to better train a value estimate, which can be used as a control variate for variance reduction BID8 b) .In this paper, we investigate an alternative approach to improving the sample efficiency of trust region policy-based RL methods.

We exploit the key fact that, under entropy regularization, the optimal policy and value function satisfy a set of pathwise consistency properties along any sampled path BID21 , which allows both on and off-policy data to be incorporated in an actor-critic algorithm, PCL.

The original PCL algorithm optimized an entropy regularized maximum reward objective and was evaluated on relatively simple tasks.

Here we extend the ideas of PCL to achieve strong results on standard, challenging continuous control benchmarks.

The main observation is that by alternatively augmenting the maximum reward objective with a relative entropy regularizer, the optimal policy and values still satisfy a certain set of pathwise consistencies along any sampled trajectory.

The resulting objective is equivalent to maximizing expected reward subject to a penalty-based constraint on divergence from a reference (i.e., previous) policy.

We exploit this observation to propose a new off-policy trust region algorithm, Trust-PCL, that is able to exploit off-policy data to train policy and value estimates.

Moreover, we present a simple method for determining the coefficient on the relative entropy regularizer to remain agnostic to reward scale, hence ameliorating the task of hyperparameter tuning.

We find that the incorporation of a relative entropy regularizer is crucial for good and stable performance.

We evaluate Trust-PCL against TRPO, and observe that Trust-PCL is able to solve difficult continuous control tasks, while improving the performance of TRPO both in terms of the final reward achieved as well as sample-efficiency.

Trust Region Methods.

Gradient descent is the predominant optimization method for neural networks.

A gradient descent step is equivalent to solving a trust region constrained optimization, DISPLAYFORM0 which yields the locally optimal update dθ = −η∇ (θ) such that η = √ / ∇ (θ) ; hence by considering a Euclidean ball, gradient descent assumes the parameters lie in a Euclidean space.

However, in machine learning, particularly in the context of multi-layer neural network training, Euclidean geometry is not necessarily the best way to characterize proximity in parameter space.

It is often more effective to define an appropriate Riemannian metric that respects the loss surface (Amari, 2012), which allows much steeper descent directions to be identified within a local neighborhood (e.g., Amari (1998); BID17 ).

Whenever the loss is defined in terms of a Bregman divergence between an (unknown) optimal parameter θ * and model parameter θ, i.e., (θ) ≡ D F (θ * , θ), it is natural to use the same divergence to form the trust region: DISPLAYFORM1 The natural gradient (Amari, 1998) is a generalization of gradient descent where the Fisher information matrix F (θ) is used to define the local geometry of the parameter space around θ.

If a parameter update is constrained by dθ DISPLAYFORM2 is obtained.

This geometry is especially effective for optimizing the log-likelihood of a conditional probabilistic model, where the objective is in fact the KL divergence D KL (θ * , θ).

The local optimization is, DISPLAYFORM3 Thus, natural gradient approximates the trust region by DISPLAYFORM4 , which is accurate up to a second order Taylor approximation.

Previous work BID13 BID4 BID24 BID28 has applied natural gradient to policy optimization, locally improving expected reward subject to variants of dθ T F (θ)dθ ≤ .

Recently, TRPO BID28 has achieved state-of-the-art results in continuous control by adding several approximations to the natural gradient to make nonlinear policy optimization feasible.

Another approach to trust region optimization is given by proximal gradient methods BID23 .

The class of proximal gradient methods most similar to our work are those that replace the hard constraint in (2) with a penalty added to the objective.

These techniques have recently become popular in RL BID35 BID11 BID31 , although in terms of final reward performance on continuous control benchmarks, TRPO is still considered to be the state-of-the-art.

BID22 make the observation that entropy regularized expected reward may be expressed as a reversed KL divergence D KL (θ, θ * ), which suggests that an alternative to the constraint in (3) should be used when such regularization is present: DISPLAYFORM5 Unfortunately, this update requires computing the Fisher matrix at the endpoint of the update.

The use of F (θ) in previous work can be considered to be an approximation when entropy regularization is present, but it is not ideal, particularly if dθ is large.

In this paper, by contrast, we demonstrate that the optimal dθ under the reverse KL constraint D KL (θ + dθ, θ) ≤ can indeed be characterized.

Defining the constraint in this way appears to be more natural and effective than that of TRPO.Softmax Consistency.

To comply with the information geometry over policy parameters, previous work has used the relative entropy (i.e., KL divergence) to regularize policy optimization; resulting in a softmax relationship between the optimal policy and state values BID25 BID3 BID2 BID7 BID26 under single-step rollouts.

Our work is unique in that we leverage consistencies over multi-step rollouts.

The existence of multi-step softmax consistencies has been noted by prior work-first by BID21 in the presence of entropy regularization.

The existence of the same consistencies with relative entropy has been noted by BID30 .

Our work presents multi-step consistency relations for a hybrid relative entropy plus entropy regularized expected reward objective, interpreting relative entropy regularization as a trust region constraint.

This work is also distinct from prior work in that the coefficient of relative entropy can be automatically determined, which we have found to be especially crucial in cases where the reward distribution changes dramatically during training.

Most previous work on softmax consistency (e.g., BID7 ; Azar et al. FORMULA0 ; BID21 ) have only been evaluated on relatively simple tasks, including grid-world and discrete algorithmic environments.

BID26 conducted evaluations on simple variants of the CartPole and Pendulum continuous control tasks.

More recently, BID10 showed that soft Qlearning (a single-step special case of PCL) can succeed on more challenging environments, such as a variant of the Swimmer task we consider below.

By contrast, this paper presents a successful application of the softmax consistency concept to difficult and standard continuous-control benchmarks, resulting in performance that is competitive with and in some cases beats the state-of-the-art.

We model an agent's behavior by a policy distribution π(a | s) over a set of actions (possibly discrete or continuous).

At iteration t, the agent encounters a state s t and performs an action a t sampled from π(a | s t ).

The environment then returns a scalar reward r t ∼ r(s t , a t ) and transitions to the next state s t+1 ∼ ρ(s t , a t ).

When formulating expectations over actions, rewards, and state transitions we will often omit the sampling distributions, π, r, and ρ, respectively.

Maximizing Expected Reward.

The standard objective in RL is to maximize expected future discounted reward.

We formulate this objective on a per-state basis recursively as DISPLAYFORM0 The overall, state-agnostic objective is the expected per-state objective when states are sampled from interactions with the environment: DISPLAYFORM1 Most policy-based algorithms, including REINFORCE BID37 and actorcritic BID15 , aim to optimize O ER given a parameterized policy.

Path Consistency Learning (PCL).

Inspired by BID37 , BID21 augment the objective O ER in (5) with a discounted entropy regularizer to derive an objective, DISPLAYFORM2 where τ ≥ 0 is a user-specified temperature parameter that controls the degree of entropy regularization, and the discounted entropy H(s, π) is recursively defined as DISPLAYFORM3 Note that the objective O ENT (s, π) can then be re-expressed recursively as, BID21 show that the optimal policy π * for O ENT and V * (s) = O ENT (s, π * ) mutually satisfy a softmax temporal consistency constraint along any sequence of states s 0 , . . .

, s d starting at s 0 and a corresponding sequence of actions a 0 , . . .

, a d−1 : DISPLAYFORM4 DISPLAYFORM5 This observation led to the development of the PCL algorithm, which attempts to minimize squared error between the LHS and RHS of (10) to simultaneously optimize parameterized π θ and V φ .

Importantly, PCL is applicable to both on-policy and off-policy trajectories.

Trust Region Policy Optimization (TRPO).

As noted, standard policy-based algorithms for maximizing O ER can be unstable and require small learning rates for training.

To alleviate this issue, BID28 proposed to perform an iterative trust region optimization to maximize O ER .

At each step, a prior policyπ is used to sample a large batch of trajectories, then π is subsequently optimized to maximize O ER while remaining within a constraint defined by the average per-state KL-divergence withπ.

That is, at each iteration TRPO solves the constrained optimization problem, DISPLAYFORM6 The prior policy is then replaced with the new policy π, and the process is repeated.

To enable more stable training and better exploit the natural information geometry of the parameter space, we propose to augment the entropy regularized expected reward objective O ENT in (7) with a discounted relative entropy trust region around a prior policyπ, DISPLAYFORM0 where the discounted relative entropy is recursively defined as DISPLAYFORM1 This objective attempts to maximize entropy regularized expected reward while maintaining natural proximity to the previous policy.

Although previous work has separately proposed to use relative entropy and entropy regularization, we find that the two components serve different purposes, each of which is beneficial: entropy regularization helps improve exploration, while the relative entropy improves stability and allows for a faster learning rate.

This combination is a key novelty.

Using the method of Lagrange multipliers, we cast the constrained optimization problem in (13) into maximization of the following objective, DISPLAYFORM2 Again, the environment-wide objective is the expected per-state objective when states are sampled from interactions with the environment, DISPLAYFORM3

A key technical observation is that the O RELENT objective has a similar decomposition structure to O ENT , and one can cast O RELENT as an entropy regularized expected reward objective with a set of transformed rewards, i.e., DISPLAYFORM0 where DISPLAYFORM1 is an expected reward objective on a transformed reward distribution functioñ r(s, a) = r(s, a) + λ logπ(a|s).

Thus, in what follows, we derive a corresponding form of the multi-step path consistency in (10).Let π * denote the optimal policy, defined as π * = argmax π O RELENT (π).

As in PCL BID21 , this optimal policy may be expressed as DISPLAYFORM2 where V * are the softmax state values defined recursively as DISPLAYFORM3 We may re-arrange (17) to yield DISPLAYFORM4 This is a single-step temporal consistency which may be extended to multiple steps by further expanding V * (s t+1 ) on the RHS using the same identity.

Thus, in general we have the following softmax temporal consistency constraint along any sequence of states defined by a starting state s t and a sequence of actions a t , . . .

, a t+d−1 : DISPLAYFORM5

We propose to train a parameterized policy π θ and value estimate V φ to satisfy the multi-step consistencies in (21).

Thus, we define a consistency error for a sequence of states, actions, and rewards s t:t+d ≡ (s t , a t , r t , . . .

, s t+d−1 , a t+d−1 , r t+d−1 , s t+d ) sampled from the environment as DISPLAYFORM0 We aim to minimize the squared consistency error on every sub-trajectory of length d. That is, the loss for a given batch of episodes (or sub-episodes) S = {s DISPLAYFORM1 We perform gradient descent on θ and φ to minimize this loss.

In practice, we have found that it is beneficial to learn the parameter φ at least as fast as θ, and accordingly, given a mini-batch of episodes we perform a single gradient update on θ and possibly multiple gradient updates on φ (see Appendix for details).In principle, the mini-batch S may be taken from either on-policy or off-policy trajectories.

In our implementation, we utilized a replay buffer prioritized by recency.

As episodes (or sub-episodes) are sampled from the environment they are placed in a replay buffer and a priority p(s 0:T ) is given to a trajectory s 0:T equivalent to the current training step.

Then, to sample a batch for training, B episodes are sampled from the replay buffer proportional to exponentiated priority exp{βp(s 0:T )} for some hyperparameter β ≥ 0.For the prior policy πθ, we use a lagged geometric mean of the parameters.

At each training step, we updateθ ← αθ + (1 − α)θ.

Thus on average our training scheme attempts to maximize entropy regularized expected reward while penalizing divergence from a policy roughly 1/(1 − α) training steps in the past.

The use of a relative entropy regularizer as a penalty rather than a constraint introduces several difficulties.

The hyperparameter λ must necessarily adapt to the distribution of rewards.

Thus, λ must be tuned not only to each environment but also during training on a single environment, since the observed reward distribution changes as the agent's behavior policy improves.

Using a constraint form of the regularizer is more desirable, and others have advocated its use in practice BID28 specifically to robustly allow larger updates during training.

To this end, we propose to redirect the hyperparameter tuning from λ to .

Specifically, we present a method which, given a desired hard constraint on the relative entropy defined by , approximates the equivalent penalty coefficient λ( ).

This is a key novelty of our work and is distinct from previous attempts at automatically tuning a regularizing coefficient, which iteratively increase and decrease the coefficient based on observed training behavior BID31 BID11 .We restrict our analysis to the undiscounted setting γ = 1 with entropy regularizer τ = 0.

Additionally, we assume deterministic, finite-horizon environment dynamics.

An additional assumption we make is that the expected KL-divergence over states is well-approximated by the KL-divergence starting from the unique initial state s 0 .

Although in our experiments these restrictive assumptions are not met, we still found our method to perform well for adapting λ during training.

In this setting the optimal policy of FORMULA0 is proportional to exponentiated scaled reward.

Specifically, for a full episode s 0:T = (s 0 , a 0 , r 0 , . . . , s T −1 , a T −1 , r T −1 , s T ), we have DISPLAYFORM0 where π(s 0: DISPLAYFORM1 We would like to approximate the trajectory-wide KL-divergence between π * andπ.

We may express the KL-divergence analytically: DISPLAYFORM2 DISPLAYFORM3 Since all expectations are with respect toπ, this quantity is tractable to approximate given episodes sampled fromπ Therefore, in Trust-PCL, given a set of episodes sampled from the prior policy πθ and a desired maximum divergence , we can perform a simple line search to find a suitable λ( ) which yields KL(π * ||πθ) as close as possible to .The preceding analysis provided a method to determine λ( ) given a desired maximum divergence .

However, there is still a question of whether should change during training.

Indeed, as episodes may possibly increase in length, KL(π * ||π) naturally increases when compared to the average perstate KL(π * (−|s)||π(−|s)), and vice versa for decreasing length.

Thus, in practice, given an and a set of sampled episodes S = {s DISPLAYFORM4 , we approximate the best λ which yields a maximum divergence of N N k=1 T k .

This makes it so that corresponds more to a constraint on the lengthaveraged KL-divergence.

To avoid incurring a prohibitively large number of interactions with the environment for each parameter update, in practice we use the last 100 episodes as the set of sampled episodes S. While this is not exactly the same as sampling episodes from πθ, it is not too far off since πθ is a lagged version of the online policy π θ .

Moreover, we observed this protocol to work well in practice.

A more sophisticated and accurate protocol may be derived by weighting the episodes according to the importance weights corresponding to their true sampling distribution.

We evaluate Trust-PCL against TRPO on a number of benchmark tasks.

We choose TRPO as a baseline since it is a standard algorithm known to achieve state-of-the-art performance on the continuous control tasks we consider (see e.g., leaderboard results on the OpenAI Gym website BID5 ).

We find that Trust-PCL can match or improve upon TRPO's performance in terms of both average reward and sample efficiency.

We chose a number of control tasks available from OpenAI Gym BID5 .

The first task, Acrobot, is a discrete-control task, while the remaining tasks (HalfCheetah, Swimmer, Hopper, Walker2d, and Ant) are well-known continuous-control tasks utilizing the MuJoCo environment BID33 .For TRPO we trained using batches of Q = 25, 000 steps (12, 500 for Acrobot), which is the approximate batch size used by other implementations Schulman, 2017) .

Thus, at each training iteration, TRPO samples 25, 000 steps using the policy πθ and then takes a single step within a KL-ball to yield a new π θ .Trust-PCL is off-policy, so to evaluate its performance we alternate between collecting experience and training on batches of experience sampled from the replay buffer.

Specifically, we alternate between collecting P = 10 steps from the environment and performing a single gradient step based on a batch of size Q = 64 sub-episodes of length P from the replay buffer, with a recency weight of β = 0.001 on the sampling distribution of the replay buffer.

To maintain stability we use α = 0.99 and we modified the loss from squared loss to Huber loss on the consistency error.

Since our policy is parameterized by a unimodal Gaussian, it is impossible for it to satisfy all path consistencies, and so we found this crucial for stability.

For each of the variants and for each environment, we performed a hyperparameter search to find the best hyperparameters.

The plots presented here show the reward achieved during training on the best hyperparameters averaged over the best 4 seeds of 5 randomly seeded training runs.

Note that this reward is based on greedy actions (rather than random sampling).Experiments were performed using Tensorflow BID0 .

Although each training step of Trust-PCL (a simple gradient step) is considerably faster than TRPO, we found that this does not have an overall effect on the run time of our implementation, due to a combination of the fact that each environment step is used in multiple training steps of Trust-PCL and that a majority of the run time is spent interacting with the environment.

A detailed description of our implementation and hyperparameter search is available in the Appendix.

We present the reward over training of Trust-PCL and TRPO in FIG0 .

We find that Trust-PCL can match or beat the performance of TRPO across all environments in terms of both final reward and sample efficiency.

These results are especially significant on the harder tasks (Walker2d and Ant).

We additionally present our results compared to other published results in Table 1 .

We find that even when comparing across different implementations, Trust-PCL can match or beat the state-of-the-art.

The most important hyperparameter in our method is , which determines the size of the trust region and thus has a critical role in the stability of the algorithm.

To showcase this effect, we present the reward during training for several different values of in FIG1 .

As increases, instability increases as well, eventually having an adverse effect on the agent's ability to achieve optimal reward.

The results of Trust-PCL against a TRPO baseline.

Each plot shows average greedy reward with single standard deviation error intervals capped at the min and max across 4 best of 5 randomly seeded training runs after choosing best hyperparameters.

The x-axis shows millions of environment steps.

We observe that Trust-PCL is consistently able to match and, in many cases, beat TRPO's performance both in terms of reward and sample efficiency.

The results of Trust-PCL across several values of , defining the size of the trust region.

Each plot shows average greedy reward across 4 best of 5 randomly seeded training runs after choosing best hyperparameters.

The x-axis shows millions of environment steps.

We observe that instability increases with , thus concluding that the use of trust region is crucial.

Note that standard PCL BID21 corresponds to → ∞ (that is, λ = 0).

Therefore, standard PCL would fail in these environments, and the use of trust region is crucial.

The main advantage of Trust-PCL over existing trust region methods for continuous control is its ability to learn in an off-policy manner.

The degree to which Trust-PCL is off-policy is determined by a combination of the hyparparameters α, β, and P .

To evaluate the importance of training off-policy, we evaluate Trust-PCL with a hyperparameter setting that is more on-policy.

We set α = 0.95, β = 0.1, and P = 1, 000.

In this setting, we also use large batches of Q = 25 episodes of length P (a total of 25, 000 environment steps per batch).

Figure 3 shows the results of Trust-PCL with our original parameters and this new setting.

We note a dramatic advantage in sample efficiency when using off-policy training.

Although Trust-PCL (on-policy) can achieve state-of-the-art reward performance, it requires an exorbitant amount of experience.

On the other hand, Trust-PCL (off- Trust-PCL (on-policy) Trust-PCL (off-policy) Figure 3 : The results of Trust-PCL varying the degree of on/off-policy.

We see that Trust-PCL (on-policy) has a behavior similar to TRPO, achieving good final reward but requiring an exorbitant number of experience collection.

When collecting less experience per training step in Trust-PCL (off-policy), we are able to improve sample efficiency while still achieving a competitive final reward.

BID9 .

These results are each on different setups with different hyperparameter searches and in some cases different evaluation protocols (e.g.,TRPO (rllab) and IPG were run with a simple linear value network instead of the two-hidden layer network we use).

Thus, it is not possible to make any definitive claims based on this data.

However, we do conclude that our results are overall competitive with state-of-the-art external implementations.policy) can be competitive in terms of reward while providing a significant improvement in sample efficiency.

One last hyperparameter is τ , determining the degree of exploration.

Anecdotally, we found τ to not be of high importance for the tasks we evaluated.

Indeed many of our best results use τ = 0.

Including τ > 0 had a marginal effect, at best.

The reason for this is likely due to the tasks themselves.

Indeed, other works which focus on exploration in continuous control have found the need to propose exploration-advanageous variants of these standard benchmarks BID10 .

We have presented Trust-PCL, an off-policy algorithm employing a relative-entropy penalty to impose a trust region on a maximum reward objective.

We found that Trust-PCL can perform well on a set of standard control tasks, improving upon TRPO both in terms of average reward and sample efficiency.

Our best results on Trust-PCL are able to maintain the stability and solution quality of TRPO while approaching the sample-efficiency of value-based methods (see e.g., BID18 ).

This gives hope that the goal of achieving both stability and sample-efficiency without trading-off one for the other is attainable in a single unifying RL algorithm.

We thank Matthew Johnson, Luke Metz, Shane Gu, and the Google Brain team for insightful comments and discussions.

We have already highlighted the ability of Trust-PCL to use off-policy data to stably train both a parameterized policy and value estimate, which sets it apart from previous methods.

We have also noted the ease with which exploration can be incorporated through the entropy regularizer.

We elaborate on several additional benefits of Trust-PCL.Compared to TRPO, Trust-PCL is much easier to implement.

Standard TRPO implementations perform second-order gradient calculations on the KL-divergence to construct a Fisher information matrix (more specifically a vector product with the inverse Fisher information matrix).

This yields a vector direction for which a line search is subsequently employed to find the optimal step.

Compare this to Trust-PCL which employs simple gradient descent.

This makes implementation much more straightforward and easily realizable within standard deep learning frameworks.

Even if one replaces the constraint on the average KL-divergence of TRPO with a simple regularization penalty (as in proximal policy gradient methods BID31 BID35 ), optimizing the resulting objective requires computing the gradient of the KL-divergence.

In Trust-PCL, there is no such necessity.

The per-state KL-divergence need not have an analytically computable gradient.

In fact, the KL-divergence need not have a closed form at all.

The only requirement of Trust-PCL is that the log-density be analytically computable.

This opens up the possible policy parameterizations to a much wider class of functions.

While continuous control has traditionally used policies parameterized by unimodal Gaussians, with Trust-PCL the policy can be replaced with something much more expressive-for example, mixtures of Gaussians or autoregressive policies as in BID18 .We have yet to fully explore these additional benefits in this work, but we hope that future investigations can exploit the flexibility and ease of implementation of Trust-PCL to further the progress of RL in continuous control environments.

We describe in detail the experimental setup regarding implementation and hyperparameter search.

In Acrobot, episodes were cut-off at step 500.

For the remaining environments, episodes were cutoff at step 1, 000.Acrobot, HalfCheetah, and Swimmer are all non-terminating environments.

Thus, for these environments, each episode had equal length and each batch contained the same number of episodes.

Hopper, Walker2d, and Ant are environments that can terminate the agent.

Thus, for these environments, the batch size throughout training remained constant in terms of steps but not in terms of episodes.

There exists an additional common MuJoCo task called Humanoid.

We found that neither our implementation of TRPO nor Trust-PCL could make more than negligible headway on this task, and so omit it from the results.

We are aware that TRPO with the addition of GAE and enough finetuning can be made to achieve good results on Humanoid .

We decided to not pursue a GAE implementation to keep a fair comparison between variants.

Trust-PCL can also be made to incorporate an analogue to GAE (by maintaining consistencies at varying time scales), but we leave this to future work.

We use fully-connected feed-forward neural networks to represent both policy and value.

The policy π θ is represented by a neural network with two hidden layers of dimension 64 with tanh activations.

At time step t, the network is given the observation s t .

It produces a vector µ t , which is combined with a learnable (but t-agnostic) parameter ξ to parametrize a unimodal Gaussian with mean µ t and standard deviation exp(ξ).

The next action a t is sampled randomly from this Gaussian.

The value network V φ is represented by a neural network with two hidden layers of dimension 64 with tanh activations.

At time step t the network is given the observation s t and the component-wise squared observation s t s t .

It produces a single scalar value.

At each training iteration, both the policy and value parameters are updated.

The policy is trained by performing a trust region step according to the procedure described in BID28 .The value parameters at each step are solved using an LBFGS optimizer.

To avoid instability, the value parameters are solved to fit a mixture of the empirical values and the expected values.

That is, we determine φ to minimize s∈batch (V φ (s) − κVφ(s) − (1 − κ)Vφ(s)) 2 , where againφ is the previous value parameterization.

We use κ = 0.9.

This method for training φ is according to that used in Schulman (2017) .

At each training iteration, both the policy and value parameters are updated.

The specific updates are slightly different between Trust-PCL (on-policy) and Trust-PCL (off-policy).For Trust-PCL (on-policy), the policy is trained by taking a single gradient step using the Adam optimizer BID14 with learning rate 0.001.

The value network update is inspired by that used in TRPO we perform 5 gradients steps with learning rate 0.001, calculated with regards to a mix between the empirical values and the expected values according to the previousφ.

We use κ = 0.95.For Trust-PCL (off-policy), both the policy and value parameters are updated in a single step using the Adam optimizer with learning rate 0.0001.

For this variant, we also utilize a target value network (lagged at the same rate as the target policy network) to replace the value estimate at the final state for each path.

We do not mix between empirical and expected values.

We found the most crucial hyperparameters for effective learning in both TRPO and Trust-PCL to be (the constraint defining the size of the trust region) and d (the rollout determining how to evaluate the empirical value of a state).

For TRPO we performed a grid search over ∈ {0.01, 0.02, 0.05, 0.1}, d ∈ {10, 50}. For Trust-PCL we performed a grid search over ∈ {0.001, 0.002, 0.005, 0.01}, d ∈ {10, 50}. For Trust-PCL we also experimented with the value of τ , either keeping it at a constant 0 (thus, no exploration) or decaying it from 0.1 to 0.0 by a smoothed exponential rate of 0.1 every 2,500 training iterations.

We fix the discount to γ = 0.995 for all environments.

A simplified pseudocode for Trust-PCL is presented in Algorithm 1.

Input: Environment EN V , trust region constraint , learning rates η π , η v , discount factor γ, rollout d, batch size Q, collect steps per train step P , number of training steps N , replay buffer RB with exponential lag β, lag on prior policy α.function Gradients({s // Update auxiliary variables Updateθ = αθ + (1 − α)θ.

Update λ in terms of according to Section 4.3. end for

<|TLDR|>

@highlight

We extend recent insights related to softmax consistency to achieve state-of-the-art results in continuous control.
Reinforcement learning algorithms, though successful, tend to over-fit to training environments, thereby hampering their application to the real-world.

This paper proposes $\text{W}\text{R}^{2}\text{L}$ -- a robust reinforcement learning algorithm with significant robust performance on low and high-dimensional control tasks.

Our method formalises robust reinforcement learning as a novel min-max game with a Wasserstein constraint for a correct and convergent solver.

Apart from the formulation, we also propose an efficient and scalable solver following a novel zero-order optimisation method that we believe can be useful to numerical optimisation in general.

We empirically demonstrate significant gains compared to standard and robust state-of-the-art algorithms on high-dimensional MuJuCo environments

Reinforcement learning (RL) has become a standard tool for solving decision-making problems with feedback, and though significant progress has been made, algorithms often over-fit to training environments and fail to generalise across even slight variations of transition dynamics (Packer et al., 2018; Zhao et al., 2019) .

Robustness to changes in transition dynamics is a crucial component for adaptive and safe RL in real-world environments.

Motivated by real-world applications, recent literature has focused on the above problems, proposing a plethora of algorithms for robust decisionmaking (Morimoto & Doya, 2005; Pinto et al., 2017; Tessler et al., 2019) .

Most of these techniques borrow from game theory to analyse, typically in a discrete state and actions spaces, worst-case deviations of agents' policies and/or environments, see Sargent & Hansen (2001) ; Nilim & El Ghaoui (2005) ; Iyengar (2005); Namkoong & Duchi (2016) and references therein.

These methods have also been extended to linear function approximators (Chow et al., 2015) , and deep neural networks (Peng et al., 2017) showing (modest) improvements in performance gain across a variety of disturbances, e.g., action uncertainties, or dynamical model variations.

In this paper, we propose a generic framework for robust reinforcement learning that can cope with both discrete and continuous state and actions spaces.

Our algorithm, termed Wasserstein Robust Reinforcement Learning (WR 2 L), aims to find the best policy, where any given policy is judged by the worst-case dynamics amongst all candidate dynamics in a certain set.

This set is essentially the average Wasserstein ball around a reference dynamics P 0 .

The constraints makes the problem well-defined, as searching over arbitrary dynamics can only result in non-performing system.

The measure of performance is the standard RL objective, the expected return.

Both the policy and the dynamics are parameterised; the policy parameters θ k may be the weights of a deep neural network, and the dynamics parameters φ j the parameters of a simulator or differential equation solver.

The algorithm performs estimated descent steps in φ space and -after (almost) convergence -performs an update of policy parameters, i.e., in θ space.

Since φ j may be high-dimensional, we adapt a zero'th order sampling method based extending Salimans et al. (2017) to make estimations of gradients, and in order to define the constraint set which φ j is bounded by, we generalise the technique to estimate Hessians (Proposition 2).

We emphasise that although access to a simulator with parameterisable dynamics is required, the actual reference dynamics P 0 need not be known explicitly nor learnt by our algorithm.

Put another way, we are in the "RL setting", not the "MDP setting" where the transition probability matrix is known a priori.

The difference is made obvious, for example, in the fact that we cannot perform dynamic programming, and the determination of a particular probability transition can only be estimated from sampling, not retrieved explicitly.

Hence, our algorithm is not model-based in the traditional sense of learning a model to perform planning.

We believe our contribution is useful and novel for two main reasons.

Firstly, our framing of the robust learning problem is in terms of dynamics uncertainty sets defined by Wasserstein distance.

Whilst we are not the first to introduce the Wasserstein distance into the context of MDPs (see, e.g., Yang (2017) or Lecarpentier & Rachelson (2019) ), we believe our formulation is amongst the first suitable for application to the demanding application-space we desire, that being, high-dimensional, continuous state and action spaces.

Secondly, we believe our solution approach is both novel and effective (as evidenced by experiments below, see Section 5), and does not place a great demand on model or domain knowledge, merely access to a simulator or differentiable equation solver that allows for the parameterisation of dynamics.

Furthermore, it is not computationally demanding, in particular, because it does not attempt to build a model of the dynamics, and operations involving matrices are efficiently executable using the Jacobian-vector product facility of automatic differentiation engines.

A Markov decision process (MDP) 1 is denoted by M = S, A, P, R, γ , where S ⊆ R d denotes the state space, A ⊆ R n the action space, P : S × A × S → [0, 1] is a state transition probability describing the system's dynamics, R : S × A → R is the reward function measuring the agent's performance, and γ ∈ [0, 1) specifies the degree to which rewards are discounted over time.

At each time step t, the agent is in state s t ∈ S and must choose an action a t ∈ A, transitioning it to a new state s t+1 ∼ P (s t+1 |s t , a t ), and yielding a reward R(s t , a t ).

A policy π : S × A → [0, 1] is defined as a probability distribution over state-action pairs, where π(a t |s t ) represents the density of selecting action a t in state s t .

Upon subsequent interactions with the environment, the agent collects a trajectory τ of state-action pairs.

The goal is to determine an optimal policy π by solving:

where p π (τ ) denotes the trajectory density function, and R Total (τ ) the return, that is, the total accumulated reward:

with µ 0 (·) denoting the initial state distribution.

We make use of the Wasserstein distance to quantify variations from a reference transition density P 0 (·).

The latter being a probability distribution, one may consider other divergences, such as Kullback-Leibler (KL) or total variation (TV).

Our main intuition for choosing Wasserstein distance is explained below, but we note that it has a number of desirable properties: Firstly, it is symmetric (W p (µ, ν) = W p (ν, µ), a property K-L lacks).

Secondly, it is well-defined for measures with different supports (which K-L also lacks).

Indeed, the Wasserstein distance is flexible in the forms of the measures that can be compared -discrete, continuous or a mixture.

Finally, it takes into account the underlying geometry of the space the distributions are defined on, which can encode valuable information.

It is defined as follows: Let X be a metric space with metric d(·, ·).

Let C(X ) be the space of continuous functions on X and let M(X ) be the set of probability measures on X .

Let µ, ν ∈ M(X ).

Let K(µ, ν) be the set of couplings between µ, ν:

That is, the set of joint distributions κ ∈ M(X × X ) whose marginals agree with µ and ν respectively.

Given a metric (serving as a cost function) d(·, ·) for X , the p'th Wasserstein distance W p (µ, ν) for p ≥ 1 between µ and ν is defined as:

(in this paper, and mostly for computational convenience, we use p = 2, though other values of p are applicable).

The desirable properties of Wasserstein distance aside, our main intuition for choosing it is described thus: Per the definition, constraining the possible dynamics to be within an -Wasserstein ball of a reference dynamics P 0 (·) means constraining it in a certain way.

Wasserstein distance has the form mass × distance.

If this quantity is constrained to be less than a constant , then if the mass is large, the distance is small, and if the distance is large, the mass is small.

Intuitively, when modelling the dynamics of a system, it may be reasonable to concede that there could be a systemic erroror bias -in the model, but that bias should not be too large.

It is also reasonable to suppose that occasionally, the behaviour of the system may be wildly different to the model, but that this should be a low-probability event.

If the model is frequently wrong by a large amount, then there is no use in it.

In a sense, the Wasserstein ball formalises these assumptions.

Due to the continuous nature of the state and action spaces considered in this work, we resort to deep neural networks to parameterise policies, which we write as π θ (a t |s t ), where θ ∈ R d1 is a set of tunable hyper-parameters to optimise.

For instance, these policies can correspond to multilayer perceptrons for MuJoCo environments, or to convolutional neural networks in case of highdimensional states depicted as images.

Exact policy details are ultimately application dependent and, consequently, provided in the relevant experiment sections.

In principle, one can similarly parameterise dynamics models using deep networks (e.g., LSTMtype models) to provide one or more action-conditioned future state predictions.

Though appealing, going down this path led us to agents that discover worst-case transition models which minimise training data but lack any valid physical meaning.

For instance, original experiments we conducted on CartPole ended up involving transitions that alter angles without any change in angular velocities.

More importantly, these effects became more apparent in high-dimensional settings where the number of potential minimisers increases significantly.

It is worth noting that we are not the first to realise such an artifact when attempting to model physics-based dynamics using deep networks.

Authors in (Lutter et al., 2019 ) remedy these problems by introducing Lagrangian mechanics to deep networks, while others (Koller et al., 2018; argue the need to model dynamics given by differential equation structures directly.

Though incorporating physics-based priors to deep networks is an important and challenging task that holds the promise of scaling model-based reinforcement learning for efficient solvers, in this paper we rather study an alternative direction focusing on perturbing differential equation solvers and/or simulators with respect to the dynamic specification parameters φ ∈ R d2 .

Not only would such a consideration reduce the dimension of parameter spaces representing transition models, but would also guarantee valid dynamics due to the nature of the simulator.

Though tackling some of the above problems, such a direction arrives with a new set of challenges related to computing gradients and Hessians of black-box solvers.

In Section 4, we develop an efficient and scalable zero-order method for valid and accurate model updates.

Unconstrained Loss Function: Having equipped agents with the capability of representing policies and perturbing transition models, we are now ready to present an unconstrained version of WR 2 L's loss function.

Borrowing from robust optimal control, we define robust reinforcement learning as an algorithm that learns best-case policies under worst-case transitions:

where p φ θ (τ ) is a trajectory density function parameterised by both policies and transition models, i.e., θ and φ, respectively:

specs vector and diff.

solver

Under review as a conference paper at ICLR 2020

At this stage, it should be clear that our formulation, though inspired from robust optimal control, is, truthfully, more generic as it allows for parameterised classes of transition models without incorporating additional restrictions on the structure or the scope by which variations are executed 2 .

Constraints & Complete Problem Definition: Clearly, the problem in Equation 5 is ill-defined due to the arbitrary class of parameterised transitions.

To ensure well-behaved optimisation objectives, we next introduce constraints to bound search spaces and ensure convergence to feasible transition models.

For a valid constraint set, our method assumes access to samples from a reference dynamics model P 0 (·|s, a), and bounds learnt transitions in an -Wasserstein ball around P 0 (·|s, a), i.e., the set defined as:

where ∈ R + is a hyperparameter used to specify the "degree of robustness" in a similar spirit to maximum norm bounds in robust optimal control.

It is worth noting, that though we have access to samples from a reference simulator, our setting is by no means restricted to model-based reinforcement learning in an MDP setting.

That is, our algorithm operates successfully given only traces from P 0 accompanied with its specification parameters, e.g., pole lengths, torso masses, etc.

-a more flexible framework that does not require full model learners.

Though defining a better behaved optimisation objective, the set in Equation 6 introduces infinite number of constraints when considering continuous state and/or actions spaces.

To remedy this problem, we target a relaxed version that considers a constraint of average Wasserstein distance bounded by a hyperparameter :

The sampling (s, a) in the expectation is done as follows: We sample trajectories using reference dynamics P 0 and a policy π that chooses actions uniformly at random (uar).

Then (s, a) pairs are sampled uar from those collected trajectories.

For a given pair (s, a), W 2 2 (P φ (·|s, a), P 0 (·|s, a)) is approximated through the empirical distribution: we use the state that followed (s, a) in the collected trajectories as a data point.

Estimating Wasserstein distance using empirical data is standard, see, e.g., Peyré et al. (2019) .

One approach which worked well in our experiments, was to assume that the dynamics are given by deterministic functions plus Gaussian noise with diagonal convariance matrices.

This makes estimation easier in high dimensions since sampling in each dimension is independent of others, and the total samples needed is a constant factor of the number of dimensions.

Gaussian distributions also have closed-form expressions for Wasserstein distance, given in terms of mean and covariance.

We thus arrive at WR 2 L's optimisation problem allowing for best policies under worst-case yet bounded transition models:

Wasserstein Robust Reinforcement Learning Objective:

Our solution alternates between updates of θ and φ, keeping one fixed when updating the other.

Fixing dynamics parameters φ, policy parameters θ can be updated by solving

, which is the formulation of a standard RL problem.

Consequently, one can easily adapt any policy search method for updating policies under fixed dynamical models.

As described later in Section 4, we make use of proximal policy optimisation (Schulman et al., 2017) .

When updating φ given a set of fixed policy parameters θ, the Wasserstein constraints must be respected.

Unfortunately, even with the simplification introduced in Section 3.1 the constraint is still difficult to compute.

To alleviate this problem, we propose to approximate the constraint in (8) by its Taylor expansion up to second order.

That is, defining W (φ) := E (s,a) W 2 2 (P φ (·|s, a), P 0 (·|s, a)) , the above can be approximated around φ 0 by a second-order Taylor as:

Recognising that W (φ 0 ) = 0 (the distance between the same probability densities), and ∇ φ W (φ 0 ) = 0 since φ 0 minimises W (φ), we can simplify the Hessian approximation by writing:

.

Substituting our approximation back in the original problem in Equation 8, we reach the following optimisation problem for determining model parameter given fixed policies:

where

is the Hessian of the expected squared 2-Wasserstein distance evaluated at φ 0 .

Optimisation problems with quadratic constraints can be efficiently solved using interior-point methods.

To do so, one typically approximates the loss with a first-order expansion and determines a closed-form solution.

Consider a pair of parameters θ [k] and φ [j] (which will correspond to parameters of the j'th inner loop of the k'th outer loop in the algorithm we present).

To find φ

[j+1] , we solve:

It is easy to show that a minimiser to the above equation can derived in a closed-form as:

with g [k,j] denoting the gradient 3 evaluated at θ [k] and φ [j] , i.e., g

Generic Algorithm: Having described the two main steps needed for updating policies and models, we now summarise these findings in the pseudo-code in Algorithm 1.

As the Hessian 4 of the Wasserstein distance is evaluated based on reference dynamics and any policy π, we pass it, along with and φ 0 as inputs.

Then Algorithms 1 operates in a descent-ascent fashion in two main phases.

In the first, lines 5 to 10 in Algorithm 1, dynamics parameters are updated using the closed-form solution in Equation 10, while ensuring that learning rates abide by a step size condition (we used the Wolfe conditions (Wolfe, 1969) , though it can be some other method).

With this, the second phase (line 11) utilises any state-of-the-art reinforcement learning method to adapt policy parameters generating θ [k+1] .

Regarding the termination condition for the inner loop, we leave this as a decision for the user.

It could be, for example, a large finite time-out, or the norm of the gradient g [k,j] being below a threshold, or whichever happens first.

, and j ← 0

Phase I: Update model parameter while fixing the policy:

while termination condition not met do 7:

Compute descent direction for the model parameters as given by Equation 10:

Update candidate solution, while satisfying step size conditions (see discussion below) on the learning rate α:

end while 10:

Perform model update setting

11:

Phase II: Update policy given new model parameters:

Use any standard reinforcement learning algorithm for ascending in the gradient direction, e.g.,

, with β

[k] a learning rate.

13: end for

Consider a simulator (or differential equation solver) S φ for which the dynamics are parameterised by a real vector φ, and for which we can execute steps of a trajectory (i.e., the simulator takes as input an action a and gives back a successor state and reward).

For generating novel physics-grounded transitions, one can simply alter φ and execute the instruction in S φ from some a state s ∈ S, while applying an action a ∈ A. Not only does this ensure valid (under mechanics) transitions, but also promises scalability as specification parameters typically reside in lower dimensional spaces compared to the number of tuneable weights when using deep networks as transition models.

Gradient Estimation:

Recalling the update rule in Phase I of Algorithm 1, we realise the need for, estimating the gradient of the loss function with respect to the vector specifying the dynamics of the environment, i.e., g

at each iteration of the inner-loop j.

Handling simulators as black-box models, we estimate the gradients by sampling from a Gaussian distribution with mean 0 and σ 2 I co-variance matrix.

Our choice for such estimates is not arbitrary but rather theoretically grounded as one can easily prove the following proposition: Proposition 1 (Zero-Order Gradient Estimate).

For a fixed θ and φ, the gradient can be computed as:

Hessian Estimation: Having derived a zero-order gradient estimator, we now generalise these notions to a form allowing us to estimate the Hessian.

It is also worth reminding the reader that such a Hessian estimator needs to be performed one time only before executing the instructions in Algorithm 1 (i.e., H 0 is passed as an input).

Precisely, we prove the following proposition: Proposition 2 (Zero-Order Hessian Estimate).

The hessian of the Wasserstein distance around φ 0 can be estimated based on function evaluations.

Recalling that

, and defining W (s,a) (φ) := W 2 2 (P φ (·|s, a), P 0 (·|s, a)), we prove:

Proofs of these propositions are given in Appendix A.

They allow for a procedure where gradient and Hessian estimates can be simply based on simulator value evaluations while perturbing φ and φ 0 .

It is important to note that in order to apply the above, we are required to be able to evaluate

An empirical estimate of the p-Wasserstein distance between two measures µ and ν can be performed by computing the p-Wasserstein distance between the empirical distributions evaluated at sampled data.

That is, one can approximation µ by µ n = 1 n n i=1 δ xi where x i are identically and independently distributed according to µ. Approximating ν n similarly, we then realise that

We evaluate WR 2 L on a variety of continuous control benchmarks from the MuJoCo environment.

Dynamics in our benchmarks were parameterised by variables defining physical behaviour, e.g., density of the robot's torso, friction of the ground, and so on.

We consider both low and high dimensional dynamics and demonstrate that our algorithm outperforms state-of-the-art from both standard and robust reinforcement learning.

We are chiefly interested in policy generalisation across environments with varying dynamics, which we measure using average test returns on novel systems.

The comparison against standard reinforcement learning algorithms allows us to understand whether lack of robustness is a critical challenge for sequential decision making, while comparisons against robust algorithms test if we outperform state-of-the-art that considered a similar setting to ours.

From standard algorithms, we compare against proximal policy optimisation (PPO) (Schulman et al., 2017) , and trust region policy optimisation (TRPO) (Schulman et al., 2015b) ; an algorithm based on natural actor-crtic (Peters & Schaal, 2008; Pajarinen et al., 2019) .

From robust algorithms, we demonstrate how WR 2 L favours against robust adversarial reinforcement learning (RARL) (Pinto et al., 2017) , and action-perturbed Markov decision processes (PR-MDP) proposed in (Tessler et al., 2019) .

Due to space constraints, the results are presented in Appendix B.2.

It is worth noting that we attempted to include deep deterministic policy gradients (DDPG) (Silver et al., 2014) in our comparisons.

Results including DDPG were, however, omitted as it failed to show any significant robustness performance even on relatively simple systems, such as the inverted pendulum; see results reported in Appendix B.3.

During initial trials, we also performed experiments parameterising models using deep neural networks.

Results demonstrated that these models, though minimising training data error, fail to provide valid physics-grounded dynamics.

For instance, we arrived at inverted pendula models that vary pole angles without exerting any angular speeds.

This problem became even more apparent in high-dimensional systems, e.g., Hopper, Walker, etc due to the increased number of possible minima.

As such, results presented in this section make use of our zero-order method that can be regarded as a scalable alternative for robust solutions.

We evaluate our method both in low and high-dimensional MuJuCo tasks (Brockman et al., 2016) .

We consider a variety of systems including CartPole, Hopper, and Walker2D; all of which require direct joint-torque control.

Keeping with the generality of our method, we utilise these frameworks as-is with no additional alterations, that is, we use the exact setting of these benchmarks as that shipped with OpenAI gym without any reward shaping, state-space augmentation, feature extraction, or any other modifications of-that-sort.

Details are given in section B. Due to space constraints, results for one-dimensional parameter variations are given in Appendix B.2, where it can be seen that WR 2 L outperforms both robust and non-robust algorithms when onedimensional simulator variations are considered.

Figure 1 shows results for dynamics variations along two dimensions.

Here again, our methods demonstrates considerable robustness.

The fourth column, "PPO mean", refers to experiments where PPO is trained on a dynamics sampled uniformly at random from the Wasserstein constraint set.

It displayes more robustness than when trained on just the reference dynamics, however, as can be seen from Fig. 2 , our method performs noticably better in high dimensions, which is the main strength of our algorithm.

Results with High-Dimensional Model Variation: Though results above demonstrate robustness, an argument against a min-max objective can be made especially when only considering lowdimensional changes in the simulator.

Namely, one can argue the need for such an objective as opposed to simply sampling a set of systems and determining policies performing-well on average similar to the approach proposed in (Rajeswaran et al., 2017) .

A counter-argument to the above is that a gradient-based optimisation scheme is more efficient than a sampling-based one when high-dimensional changes are considered.

In other words, a sampling procedure is hardly applicable when more than a few parameters are altered, while WR 2 L can remain suitable.

To assess these claims, we conducted two additional experiments on the Hopper and HalfCheetah benchmarks.

In the first, we trained robustly while changing friction and torso densities, and tested on 1,000 systems generated by varying all 11 dimensions of the Hopper dynamics, and 21 dimensions of the HalfCheetah system.

The histogram Figures 2(b) and (f) demonstrate that the empirical densities of the average test returns are mostly centered around 3,000 for the Hopper, and around 4,500 for the Cheetah, which improves that of PPO trained on reference (Figures 2(a) and (e)) with return masses mostly accumulated at around 1,000 in the case of the Hopper and almost equally distributed when considering HalfCheetah.

Such improvements, however, can be an artifact of the careful choice of the low-dimensional degrees of freedom allowed to be modified during Phase I of Algorithm 1.

To get further insights, Figures 2(c) and (g) demonstrate the effectiveness of our method trained and tested while allowing to tune all 11 dimensional parameters of the Hopper sim- ulator, and the 21 dimensions of the HalfCheetah.

Indeed, our results are in accordance with these of the previous experiment depicting that most of the test returns' mass remains around 3,000 for the Hopper, and improves to accumulate around 4,500 for the HalfCheetah.

Interestingly, however, our algorithm is now capable of acquiring higher returns on all systems 6 since it is allowed to alter all parameters defining the simulator.

As such, we conclude that WR 2 L outperforms others when high-dimensional simulator variations are considered.

In Figures 2(d) and (h), we see the results for PPO trained with dynamics sampled uar from the Wasserstein constraint set.

We see that although in the two-dimensional variation case this training method worked well (see Figures 1(d) , (h), (l)), it does not scale well to high dimensions, and our method does better.

Previous work on robust MDPs, e.g., Iyengar (2005) Petrik & Russell (2019) , whilst valuable in its own right, is not sufficient for the RL setting due to the need in the latter case to give efficient solutions for large state and action spaces, and the fact that the dynamics are not known a priori.

Closer to our own work, Rajeswaran et al. (2017) approaches the robustness problem by training on an ensemble of dynamics in order to be deployed on a target environment.

The algorithm introduced, Ensemble Policy Optimisation (EPOpt), alternates between two phases: (i) given a distribution over dynamics for which simulators (or models) are available (the source domain), train a policy that performs well for the whole distribution; (ii) gather data from the deployed environment (target domain) to adapt the distribution.

The objective is not max-min, but a softer variation defined by conditional value-at-risk (CVaR).

The algorithm samples a set of dynamics {φ k } from a distribution over dynamics P ψ , and for each dynamics φ k , it samples a trajectory using the current policy parameter θ i .

It then selects the worst performing -fraction of the trajectories to use to update the policy parameter.

Clearly this process bears some resemblance to our algorithm, but there is a crucial difference: our algorithm takes descent steps in the φ space.

The difference if important when the dynamics parameters sit in a high-dimensional space, since in that case, optimisationfrom-sampling could demand a considerable number of samples.

In any case, our experiments demonstrate our algorithm performs well even in these high dimensions.

We note that we were were unable to find the code for this paper, and did not attempt to implement it ourselves.

The CVaR criterion is also adopted in Pinto et al. (2017) , in which, rather than sampling trajectories and finding a quantile in terms of performance, two policies are trained simultaneously: a "protagonist" which aims to optimise performance, and an adversary which aims to disrupt the protagonist.

The protagonist and adversary train alternatively, with one being fixed whilst the other adapts.

We made comparisons against this algorithm in our experiments.

More recently, Tessler et al. (2019) studies robustness with respect to action perturbations.

There are two forms of perturbation addressed: (i) Probabilistic Action Robust MDP (PR-MDP), and (ii) Noisy Action Robust MDP (NR-MDP).

In PR-MDP, when an action is taken by an agent, with probability α, a different, possibly adversarial action is taken instead.

In NR-MDP, when an action is taken, a perturbation is added to the action itself.

Like Rajeswaran et al. (2017) and Pinto et al. (2017) , the algorithm is suitable for applying deep neural networks, and the paper reports experiments on InvertedPendulum, Hopper, Walker2d and Humanoid.

We tested against PR-MDP in some of our experiments, and found it to be lacking in robustness (see Section 5).

In Lecarpentier & Rachelson (2019) a non-stationary Markov Decision Process model is considered, where the dynamics can change from one time step to another.

The constraint is based on Wasserstein distance, specifically, the Wasserstein distance between dynamics at time t and t is bounded by L|t − t |, i.e., is L-Lipschitz with respect to time, for some constant L. They approach the problem by treating nature as an adversary and implement a Minimax algorithm.

The basis of their algorithm is that due to the fact that the dynamics changes slowly (due to the Lipschitz constraint), a planning algorithm can project into the future the scope of possible future dynamics and plan for the worst.

The resulting algorithm, known as Risk Averse Tree Search, isas the name implies -a tree search algorithm.

It operates on a sequence "snapshots" of the evolving MDP, which are instances of the MDP at points in time.

The algorithm is tested on small grid world, and does not appear to be readily extendible to the continuous state and action scenarios our algorithm addresses.

To summarise, our paper uses the Wasserstein distance for quantifying variations in possible dynamics, in common with Lecarpentier & Rachelson (2019) , but is suited to applying deep neural networks for continuous state and action spaces.

Our algorithm does not require a full dynamics available to it, merely a parameterisable dynamics.

It competes well with the above papers, and operates well for high dimensional problems, as evidenced by the experiments.

In this paper, we proposed a robust reinforcement learning algorithm capable of outperforming others in terms of test returns on unseen dynamics.

The algorithm makes use of Wasserstein constraints for policies generalising across varying domains, and considers a zero-order method for scalable solutions.

Empirically, we demonstrated superior performance against state-of-the-art from both standard and robust reinforcement learning on low and high-dimensional MuJuCo environments.

In future work, we aim to consider robustness in terms of other components of MDPs, e.g., state representations, reward functions, and others.

Furthermore, we will implement WR 2 L on real hardware, considering sim-to-real experiments.

-Sub-Case III when indices are all distinct:

We have

Diagonal Elements Conclusion: Using the above results we conclude that

• Off-Diagonal Elements (i.e., when i = j): The above analysis is now repeated for computing the expectation of the off-diagonal elements of matrix B. Similarly, this can also be split into three sub-cases depending on indices:

-Sub-Case III when indices are all distinct:

We have

Off-Diagonal Elements Conclusion: Using the above results and due to the symmetric properties of H, we conclude that

Finally, analysing c, one can realise that

Substituting the above conclusions back in the original approximation in Equation 11 , and using the linearity of the expectation we can easily achieve the statement of the proposition.

For clarity, we summarise variables parameterising dynamics in Table 1 , and detail specifics next.

CartPole: The goal of this classic control benchmark is to balance a pole by driving a cart along a rail.

The state space is composed of the position x and velocityẋ of the cart, as well as the angle θ and angular velocities of the poleθ.

We consider two termination conditions in our experiments: 1) pole deviates from the upright position beyond a pre-specified threshold, or 2) cart deviates from its zeroth initial position beyond a certain threshold.

To conduct robustness experiments, we parameterise the dynamics of the CartPole by the pole length l p , and test by varying l p ∈ [0.3, 3].

Hopper: In this benchmark, the agent is required to control a hopper robot to move forward without falling.

The state of the hopper is represented by positions, {x, y, z}, and linear velocities, {ẋ,ẏ,ż}, of the torso in global coordinate, as well as angles, {θ i } 2 i=0 , and angular speeds, {θ i } 2 i=0 , of the three joints.

During training, we exploit an early-stopping scheme if "unhealthy" states of the robot were visited.

Parameters characterising dynamics included densities {ρ i } 3 i=0 of the four links, armature {a i } 2 i=0 and damping {ζ i } 2 i=0 of three joints, and the friction coefficient µ g .

To test for robustness, we varied both frictions and torso densities leading to significant variations in dynamics.

We further conducted additional experiments while varying all 11 dimensional specification parameters.

Walker2D: This benchmark is similar to Hopper except that the controlled system is a biped robot with seven bodies and six joints.

Dimensions for its dynamics are extended accordingly as reported in Table 1 .

Here, we again varied the torso density for performing robustness experiments in the range ρ 0 ∈ [500, 3000].

Halfcheetah: This benchmark is similar to the above except that the controlled system is a twodimensional slice of a three-dimensional cheetah robot.

Parameters specifying the simulator consist of 21 dimensions, with 7 representing densities.

In our two-dimensional experiments we varied the torso-density and floor friction, while in high-dimensional ones, we allowed the algorithm to control all 21 variables.

Our experiments included training and a testing phases.

During the training phase we applied Algorithm 1 for determining robust policies while updating transition model parameters according to the min-max formulation.

Training was performed independently for each of the algorithms on the relevant benchmarks while ensuring best operating conditions using hyper-parameter values reported elsewhere (Schulman et al., 2017; Pinto et al., 2017; Tessler et al., 2019) .

For all benchmarks, policies were represented using parametrised Gaussian distributions with their means given by a neural network and standard derivations by a group of free parameters.

The neural network consisted of two hidden layers with 64 units and hyperbolic tangent activations in each of the layers.

The final layer exploited linear activation so as to output a real number.

Following the actor-critic framework, we also trained a standalone critic network having the same structure as that of the policy.

For each policy update, we rolled-out in the current worst-case dynamics to collect a number of transitions.

The number associated to these transitions was application-dependent and varied between benchmarks in the range of 5,000 to 10,000.

The policy was then optimised (i.e., Phase II of Algorithm 1) using proximal policy optimization with a generalised advantage estimation.

To solve the minimisation problem in the inner loop of Algorithm 1, we sampled a number of dynamics from a diagonal Gaussian distribution that is centered at the current worst-case dynamics model.

The number of sampled dynamics and the variance of the sampled distributions depended on both the benchmark itself, and well as the dimensions of the dynamics.

Gradients needed for model updates were estimated using the results in Propositions 7 and 8.

Finally, we terminated training when the policy entropy dropped below an application-dependent threshold.

When testing, we evaluated policies on unseen dynamics that exhibited simulator variations as described earlier.

We measured performance using returns averaged over 20 episodes with a maximum length of 1,000 time steps on testing environments.

We note that we used non-discounted mean episode rewards to compute such averages.

Figure 3 shows the robustness of policies on various taks.

For a fair comparison, we trained two standard policy gradient methods (TRPO (Schulman et al., 2015b) and PPO (Schulman et al., 2017) ), and two robust RL algorithms (RARL (Pinto et al., 2017) , PR-MDP (Tessler et al., 2019) ) with the reference dynamics preset by our algorithm.

The range of evaluation parameters was intentionally designed to include dynamics outside of the -Wasserstein ball.

Clearly, WR 2 L outperforms all baselines in this benchmark.

As mentioned in the experiments section of the main paper, we refrained from presenting results involving deep deterministic policy gradients (DDPG) due to its lack in robustness even on simple systems, such as the CartPole.

In Section 3.2 we presented a closed form solution to the following optimisation problem:

which took the form of:

@highlight

An RL algorithm that learns to be robust to changes in dynamics
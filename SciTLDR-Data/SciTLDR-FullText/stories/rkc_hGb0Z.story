We present a method for evaluating the sensitivity of deep reinforcement learning (RL) policies.

We also formulate a zero-sum dynamic game for designing robust deep reinforcement learning policies.

Our approach mitigates the brittleness of policies when agents are trained in a simulated environment and are later exposed to the real world where it is hazardous to employ RL policies.

This framework for training deep RL policies involve a zero-sum  dynamic game against an adversarial agent, where the goal is to drive the system dynamics to a saddle region.

Using a variant of the guided policy search algorithm, our agent learns to adopt robust policies that require less samples for learning the dynamics and performs better than the GPS algorithm.

Without loss of generality, we demonstrate that deep RL policies trained in this fashion will be maximally robust to a ``worst" possible adversarial disturbances.

Deep reinforcement learning (RL) for complex agent behavior in realistic environments usually combines function approximation techniques with learning-based control.

A good RL controller should guarantee fulfillment of performance specifications under external disturbances, or modeling errors.

Quite often in practice, however, this is not the case -with deep RL policies not often generalizing well to real-world scenarios.

This can be attributed to the inherent differences between the training and testing environments.

Recently, there have been efforts at integrating function approximation techniques with learning-based control, in an end-to-end fashion, in order to have systems that optimize objectives while guaranteeing generalization to environmental uncertainties.

Examples include trajectory-based optimization for known dynamics ( BID16 BID25 ), or trajectory optimization for unknown dynamics such as guided policy search algorithms BID0 BID13 BID15 .While these methods produce performance efficiency for agent tasks in the real world, there are sensitivity questions of such policies that need to be addressed such as, how to guarantee maximally robust deep RL policies in the presence of external disturbances, or modeling errors.

A typical approach employed in minimizing sample inefficiency is to engineer an agent's policy in a simulated environment, and later transfer such policies to physical environments.

However, questions of robustness persist in such scenarios as the agent often has to cope with modeling errors and new sensory inputs from a different environment.

For continuous control tasks, learned policies may become brittle in the presence of external perturbations, or a slight change in the system dynamics may significantly affect the performance of the learned controller BID20 -defeating the purpose of having a robust policy that is learned through environmental interaction .The contribution of this paper is two-fold:• first, we provide a framework that demonstrates the brittleness of a state-of-the-art deep RL policy; specifically, given a trained RL policy, we pose an adversarial agent against the fixed trained policy; the goal is to perturb the parameter space of the learned policy.

We demonstrate that the most sophisticated deep policies fail in the presence of adversarial perturbations.• second, we formulate an iterative dynamic zero-sum, two player game, where each agent executes an opposite reaction to its pair: a concave-convex problem follows explicitly, and our goal is to achieve a saddle point equilibrium, where the state is everywhere defined but possibly infinite-valued).Noting that lack of generalization of learned reward functions to the real-world can be thought of as external disturbance that perturb the system dynamics, we formulate the learning of robust control policies as a zero-sum two player Markov game -an iterative dynamic game (iDG) -that pits an adversarial agent against a protagonist agent.

The controller aims to minimize a given cost while the second agent, an adversary aims to maximize the given cost in the presence of an additive disturbance.

We run the algorithm in finite episodic settings and show a dynamic game approach aimed at generating policies that are maximally robust.

The content of this paper is thus organized: we review relevant literature to our contribution in Sec. 2; we then provide an H ∞ background in Sec. BID2 .

This H ∞ technical introduction will be used in formulating the design of perturbation signals in Sec. BID3 .

Without loss of generality, we provide a formal treatment of the iDG algorithm within the guided policy search framework in Sec. 5.

Experimental evaluation on multiple robots is provided in Sec. 6 followed by conclusions in Sec. 7.

Robustness studies in classical control have witnessed the formalization of algorithms and computation necessary to carry out stable feedback control and dynamic game tasks (e.g. BID1 BID14 BID18 ).

There now exist closed-form and iterative-based algorithms to quantify the sensitivity of a control system and design robust feedback controllers.

These methods are well-studied in classical H ∞ control theory.

While questions of robustness of policies have existed for long in connectionist RL settings BID24 , only recently have researchers started addressing the question of incorporating robustness guarantees into deep RL controllers.

Heess et.

al BID8 posit that rich, robust performance will emerge if an agent is simulated in a sufficiently rich and diverse environment.

BID8 proposed a learning framework for agents in locomotion tasks which involved choosing simple reward functions, but exposing the agent to various levels of difficult environments as a way of achieving ostensibly sophisticated performance objectives.

Incorporating various levels of difficulty in obstacles, height and terrain smoothness to an agent's environment for every episodic task, they achieved robust behaviors for difficult locomotion tasks after many episodes ( ≈ 10 6 ) of training.

However, this strategy defeats one of the primary objectives of RL namely, to make an agent discover good policies with finite data based on little interaction with the environment.

An ideal robust RL controller must come from data-efficient samples or imitations.

Furthermore, this approach takes a qualitative measure at building robust signals into the reward function via means such as locomotion hurdles with variations in height, slopes, and slalom walls.

We reckon that building such physical barriers for an agent is expensive in the real-world and learning such emergent locomotion behaviors takes a long training time.

Pinto et.

al. BID19 , posed the learning of robust RL rewards in a zero-sum, two-player markov decision process (MDP) defined by the standard RL tuple {S, A 1 , A 2 , P, R, γ, s 0 }, where A 1 and A 2 denote the continuous action spaces of the two players.

Both players share a joint transition probability P and reward R. Pinto's approach assumed a knowledge of the underlying dynamics so that an adversarial policy, π adv θ (u t |x t ), can exploit the weakness in a protagonist agent's policy, π prot θ (u t |x t ).

This relied on a minimax alternating optimization process: optimizing for one set of actions while holding the other fixed, to the end of ensuring robustness of the learned reward function.

While it introduced H ∞ control as a robustness measure for classical RL problems, it falls short of adapting H ∞ for complex agent tasks and policy optimizations.

Moreover, there are no theoretical analyses of saddle-/pareto-point or Nash equilibrium guarantees and the global optimum that assures maximal robustness at π prot θ (u t |x t ) = π adv θ (u t |x t ) is left unaddressed.

Perhaps the closest formulation to this work is BID9 's neural fictitious self-play for large games with imperfect state information whereby players select best responses to their opponents' average strategies.

BID9 showed that with deep reinforcement learning, self-play in imperfect-information environments approached a Nash equilibrium where other reinforcement learning methods diverged.

From a methodical perspective, we formulate the robustness of RL controllers within an H ∞ framework (see BID18 ) for deep robot motor tasks.

Similar to a matrix game with two agents, we let both agents play a zero-sum, two person game where each agent's action strategy or security level never falls below that of the other.

The ordering according to which the players act so that each player acts optimally in a "min max" fashion does not affect the convergence to saddle point in our formulation.

We consider the case where the security levels of both players coincide so that the strategy pair of both agents constitute a saddle-point pure strategy [3, p. 19 ].

Reinforcement learning in robot control tasks consists of selecting control commands u from the space of a control policy π (often parameterized by θ) that act on a high-dimensional state x. The x typically composed of internal (e.g. joint angles and velocities) and external (e.g. object pose, positional information in the world) components.

For a stochastic policy π(u t |x t ) the commands influence the state of the robot based on the transition distribution π θ (u t |x t , t).

The state and action pairs constitute a trajectory distribution τ = ( DISPLAYFORM0 The performance of the robot on an episodic motor task is evaluated by an accumulated reward function R(τ ) defined as DISPLAYFORM1 for an instantaneous reward function, r t , and a final reward, r t f .

Many tasks in robot learning domains can be formulated as above, whereby we choose a locally optimal policy π θ that optimizes the expectation of the accumulated reward DISPLAYFORM2 where p π θ (τ ) denotes distribution over trajectories τ and is defined as DISPLAYFORM3 p(x t+1 |x t , u t ) above represents the robot's dynamics and its environment.

Given the inadequacy of value function approximation methods in managing high-dimensional continuous state and action spaces as well as the difficulty of carrying out arbitrary exploration given hardware constraints BID6 , we resolve to use policy search (PS) methods as they operate in the parameter space of parameterized policies.

However, direct PS are often specialized algorithms that produce optimal policies for a particular task (often using policy gradient methods), and they come with the negative effects of not generalizing well to flexible trajectory optimizations and large representations e.g. using neural network policies.

Guided policy search (GPS) algorithms BID0 BID13 BID15 are able to guide the search for parameterized policies from poor local minima using an alternating block coordinate ascent of the optimization problem, made up of a C-Step and an S-Step.

In the C-step, a well-posed cost function is minimized with respect to the trajectory samples, generating guiding distributions p i (u t |x t ); and in the Sstep, the locally learned time-varying control laws, p i (u t |x t ), are parameterized by a nonlinear, neural network policy using supervised learning.

The S-step fits policies of the form π θ (u t |x t ) = N (µ π (x t ), Σ π (x t )) to the local controllers p i (u t |x t ), where µ π (x t ), and Σ π (x t ) are functions that are estimated.

In order to ensure the learned policy for a dynamical system is robust to external uncertainties, modeling and transfer learning errors, we propose an iterative dynamic game consisting of an agent within an environment, and an adversarial agent, interacting with the original agent in the closed-loop environment E, over a finite horizon, T (it could also be extended to the infinite horizon case).

The adversary could represent a spoofing agent in the world or modeling errors between the plant and dynamics.

The states evolve according to the following stochastic dynamics p(x t+1 |x t , u t , v t ), ∀ t = 0, ..., T where x t ∈ X t is a markovian system state, u t ∈ U t is the action taken by the agent (henceforth called the protagonist), v t ∈ V t is the action taken by an adversarial agent.

The subscripts denote time steps t ∈ [1, T ], allowing for a simpler structuring of the individual policies per time step BID6 .

The problems we consider are control tasks with complex dynamics, having continuous state and action spaces, and with trajectories defined as DISPLAYFORM4 At time t, the system's controller visits states with high rewards while the adversarial agent wants to visit states with low rewards.

The solution to this zero-sum game follows with an equilibrium at the origin -a saddle-point solution ensues.

The policies that govern the behavior of the agents are defined as π θ (u t |x t ) and π θ (v k |x k ) respectively, and the learned local linear time-varying Gaussian controllers are defined as p(u k |x k ), p(v k |x k ).

In the next subsection, we show that learned motor policies, p(u k |x k ), are sensitive to minute additive disturbances; we later on propose how to build robustness to such trained neural network policies.

This is important in learning tasks where the robustness margins of a trained controller need to be known in advance before being introduced to a new execution environment.

Our goal is to select a suitable policy parameterization, so as to assure robustness and stability guarantees BID3 .

In this paper, we specifically use convex variant of the mirror descent version of GPS BID15 .

In this section, we show why guided policy search algorithms are non-robust to even the simplest form of perturbations -additive disturbance.

Before we proceed, we note that policy search algorithms are popular among the RL tools available because they have a "modest" level of robustness built into them e.g.• by requiring the learning controller to start the policy parameterization from multiple initial states, • adding a Kullback-Leibler (KL) constraint term to the reward function e.g. BID0 BID13 , • solving a motor task in multiple ways where more than one solution exist BID6 , • or by introducing additive noise into the system as white noise e.g. differential dynamic programming (DDP) methods BID10 or their iLQG variants BID22 .A fundamental drawback of these robustness mechanisms, however, is that the learned policy can only tolerate disturbance for slightly changing conditions; parametric uncertainty is not suitably modeled as white noise, and treating the error as an extra input might be relative to the size of the inputs (drawn from the environment) -necessitating the need for a formal treatment of robustness in PS algorithms.

A methodical way of solving the robustness problem in deep RL would be to consider techniques formalized in H ∞ control theory, where controller sensitivity and robustness are solved within the framework of a differential game.

We conjecture that the lack of robustness of a RL trained policy arises from the difference between a plant model and the real system, or the difference in learning environments.

If we can measure the sensitivity of a system, γ, then we can aim for policy robustness by ensuring that γ is sufficiently small to reject disturbance arising from the training environment or modeling errors if the gain of mapping from the error space to the disturbance is less than γ −1 BID26 .

The figure to the right depicts the standard H ∞ control problem.

Suppose G in the left inset is a plant for which we can find an internally stabilizing controller, Σ K , that ensures stable transfer of input u to measurement y, the H ∞ control objective is to find the "worst" possible disturbance, v, which produces an undesired output, z; we want to minimize the effect of z. In the right inset in the figure, we treat unmodeled dynamics, transfer errors and other uncertainty as an additional feedback to which we would like to adapt with respect to the worst possible disturbance in a prescribed range.

Σ ∆ in the right inset represents these uncertainties; our goal is to find the closed-loop optimal policy for which the plant, G, will satisfy performance requirements and maintain robustness for a large range of systems Σ ∆ .

We focus on conditions under Algorithm 1 Guided policy search: convex linear variant 1: for iteration k ∈ {1, . . .

, K} do 2: DISPLAYFORM0 ) (from supervised learning) BID3 : end for which we can make the H ∞ norm of the system less than a given prior, γ.

Specifically, we want to design a controller Σ K that minimizes the H ∞ norm of the closed-loop transfer function T zv from disturbance v to output z defined as DISPLAYFORM1 From the small-gain theorem, the system in the right figure above will be stable for any stable mapping ∆ : z → v for ∆ ∞ < γ −1 BID18 .

In a differential game setting, we can consider a minmax solution to the H ∞ problem for the plant G with dynamics given byẋ = f (x, u, w) so that we solve an H ∞ problem that satisfies the constraint DISPLAYFORM2 We consider a differential game for which the best control u that minimizes V , and the worst disturbance v that maximizes V are derived from DISPLAYFORM3 The optimal value function is determined from the Hamilton-Jacobi-Isaacs (HJI) equation, DISPLAYFORM4 from which the optimal u and v can be computed.

is adjusted based on the formulation in BID15 .

The dynamics DISPLAYFORM5 k } using a mixture of Gaussian models to the generated samples {x

This section offers guidance on testing the sensitivity of a deep neural network policy for an agent.

We consider additive disturbance to a deep RL policy.

Our goal is to study the degradation of performance of a trained neural network policy in the presence of the "worst" possible disturbance in the parameter space of the policy; if this disturbance cannot alter the performance of the trained policy, we have some value for the policy parameters in the prescribed range that the decision strategy is acceptable.

We follow the model described above, where Σ ∆ denotes the uncertainty injected by the adversary.

We arrive at the nominal system from u to y when the transfer matrix of Σ ∆ is zero.

We call Σ ∆ the adversary whose control, v's effect on the output z is to be minimized.

We quantify the effect of v on z in closed loop using a suitable cost function as a min-max criteria.

This can be seen as an H ∞ norm on the system.

Suppose the local actions, p(u k |x k ), of the controller belong to the policy space π = [π 0 , ..., π T ] that maximize the expected sum of rewards DISPLAYFORM0 Therefore, the augmented reward for the closed-loop protagonist-adversary system becomes DISPLAYFORM1 where DISPLAYFORM2 where α(·) can be chosen as a function of the adversarial disturbance v t 1 .

We chose α as the L 2 norm of the disturbance v t in our implementation.

γ is a sensitivity parameter that adjusts the strength of the adversary by increasing the penalty incurred by its actions.

In FORMULA11 , we carry out the optimization procedure by first learning the optimal policy for the controller; we then fix this optmal policy and carry out the minimization of the augmented reward function with the adversary in closed-loop as in (3).As γ → ∞ in (3), the optimal closed-loop policy is for the agent to do nothing, since any action will incur a large penalty; as γ decreases, however, the adversary's actions have a greater effect on the state of the closed-loop system.

The (inverse of the) lowest value of γ for which the adversary's policy causes unacceptable performance provides a measure of robustness of the control policy π θ (u t |x t ).

For various values of γ, the state-of-the-art robot learning policies are non-robust to small perturbations as we show in Sec. 6.

To learn robust policies, we run an alternating optimization algorithm that maximizes the cost function with respect to the adversarial controller (modeled with the worst possible disturbance) and minimizes the cost function with respect to the protagonist's policy.

We consider a two-player, zero-sum Markov game framework for simultaneously learning policies for the protagonist and the adversary.

We seek to learn saddle-point equilibrium strategies for the zero-sum game: DISPLAYFORM0 where we have overloaded notation such that π( DISPLAYFORM1 is the stage cost.π(x t ) denotes that the adversarial actions are drawn from outside of the action space of the protagonist's policy.

Fixing a value of γ is equivalent to an assumption on the capability of the adversary or the magnitude of a worst possible disturbance.

To validate this proposal, we develop locally robust controllers for a trajectory optimization problem from multiple initial states using (4) as a guiding cost; a neural network function approximator is then used to parameterize these local controllers using supervised learning.

We discuss this procedure in the next section.

GPS adds off-policy guiding samples to a sample set: this guides the policy toward spaces of high rewards.

If p(τ ) is the trajectory distribution induced by the locally linear Gaussian controller p(u t |x t ) andp(u t |x t ) denotes the previous local controller, GPS algorithms reduce the effect of visiting regions of low entropy by minimizing the KL divergence of the current local policy from the previous one as follows, DISPLAYFORM0 where H is the entropy term that favors broad distributions, η is a Lagrange multiplier and the first term forces the actions p to be high in regions of high reward.

The trajectory is optimized using optimal control principles under linear quadratic Gaussian assumptions BID22 .

GPS minimizes the expected cost, E π θ (xt,ut) r(x t , u t ) over the joint distribution of state and action pairs given by the marginals π θ (τ ) = p(x 1 ) T t=1 p(x t+1 |x t , u t ).

GPS algorithms optimize the cost J(θ) via a split process of trajectory optimization of local control laws and a standard supervised learning to generalize to high-dimensional policy space settings.

A generic GPS algorithm is shown in Algorithm 1.

During the C-step, multiple local control laws, p i (u t |x t ), are generated for different initial states x i 1 ∼ p(x 1 ).

The supervised learning stage (S-step) regresses the global policy π θ (u t |x t ) to all the local actions computed in the C-step.

For unknown dynamics, one can fit p(x t+1 |x t , u t ) to sampled trajectories from the trajectory distribution underp(τ ).

To avoid divergence in dynamics, the difference between the current and previous trajectories are constrained by the KL divergence as in step 2 in algorithm 1.The KL divergencep from p in (5) will not optimize for a robust policy in the presence of modeling errors, changes in environment settings or disturbance as we show in the sensitivity section in subsection 4.1.

To make the computed neural network policy robust to these uncertainties, we propose a zero-sum, two-person dynamic game scenario in the next section.

Algorithm 2 Robust guided policy search: unknown nonlinear dynamics 1: for iteration k ∈ {1, . . .

, K} do 2:Generate samples Di = {τi,j} by running pi(u k |x k ) and pi(v k |x k ) or π θi (u|x k ) and π θi (v|x k ) 3:Fit linear-Gaussian dynamics pi(x k+1 |x k , u k , v k ) using samples in Di 4:Fit linearized protagonist policy π θi (u k |x k ) using samples in Di 5:Regress global policiesπ θi (u k |x k ),π θi (v k |x k ) with samples in Di DISPLAYFORM1

To guarantee robust performance during the training of policies of a stochastic system, we introduce the "worst" disturbance in the H ∞ paradigm to the search for a good guiding distribution problem.

We begin by augmenting the reward function with a term that allows for withstanding a disturbing input DISPLAYFORM0 where γ 2 v T v allows us to introduce a quadratic weighting term in the disturbing input; γ denotes the robustness parameter.

A zero-sum game follows explicitly: the protagonist is guided toward regions of high reward regions while adversary pulls in its own favorite direction -yielding a saddle-point solution.

This framework facilitates learning control decision strategies that are robust in the presence of disturbances and modeling errors -improving upon the generic optimal control policies that GPS and indeed deep RL algorithms guarantee.

We propose repeatedly solving an MPC-based finite-horizon trajectory optimization problem within the framework of DDP.

Specifically, we generalize a DDP variant -the iLQG algorithm of BID21 , to a two-player, zero-sum dynamic game as follows:• we iteratively approximate the nonlinear dynamics,ẋ = f (x t , u t , v t ), starting with nominal control,ū t ; t ∈ [t 0 , t f ], and nominal adversarial inputv t ; t ∈ [t 0 , t f ] which are assumed to be available.• we run the passive dynamics withū t andv t to generate a trajectory (x t ,ū t ,v t )• discretizing time, we linearize the nonlinear system,ẋ t , about (x k ,ū k ,v k ), so that the new state and action pairs become DISPLAYFORM0 δx k , δu k , and δv k are measured w.r.t the nominal vectorsx k ,ū k ,v k and are not necessarily small.

The LQG approximation to the original optimal control problem and reward become DISPLAYFORM1 where single and double subscripts in the augmented reward denote first-order and second-order derivatives respectively, and f zk are the respective Jacobians e.g. f xk = ∂f (·) DISPLAYFORM2 ∂x | k at time k, E(w t ) is an additive random noise term (folded into v t in our implementation); the value function is the cost-to-go given by the min-max of the control sequence DISPLAYFORM3 , where k f is the final time step, the dynamic programming problem transforms the min-max over an entire control sequence to a series of optimizations over a single control, which proceeds backward in time as DISPLAYFORM4 The Hamiltonian, (·) + V (·), can be considered as a function of perturbations around the tuple {x k , u k , v k }.

Given the intractability of solving the Bellman partial differential equation above, we restrict our attention to the local neighborhood of the nominal trajectory by expanding a power series about the nominal, nonoptimal trajectory similar to BID17 .

We proceed as follows:• we maintain a second-order local model of the perturbed Q-coefficients of the LQR problem, (Q k , Q xk , Q uk , Q vk , Q xxk , Q uxk , Q vxk , Q uuk , Q vvk ) 2 , defined thus DISPLAYFORM5 • a second-order Taylor approximation of Q(δx k , δu k , δv k , k) in the preceding equation yields DISPLAYFORM6 • the best possible (protagonist) action and the worst possible (adversarial) action can be found by performing the respective arg min and arg max operations DISPLAYFORM7 so that we have the following linear controllers that minimize and maximize the quadratic Q-function respectively: vvk BID11 BID4 , by replacing the Hessian with an identity matrix (which gives the steepest descent) BID22 , or by multiplying by lowest eigenvalue of the matrix.

We find that the protagonist and adversary in the above-equations have a local action containing a state feedback term, G, and an open-loop term, g, given by DISPLAYFORM8 DISPLAYFORM9 respectively.

The tuple {g u k , G u k , g v k , G v k } can be computed efficiently as shown in BID16 .

We can construct linear Gaussian controllers with mean given by the deterministic optimal solutions and the covariance proportional to the curvatures of the respective Q functions: DISPLAYFORM10 vvk ).

BID12 has shown that these types of distributions optimize an objective function with maximum entropy given by arg min DISPLAYFORM11 is the system's trajectory evolution over all states, i, visited by both local controllers, and H is the differential entropy.

Equation (9) produces a trajectory that follows the widest, highest-entropy distribution while minimizing the expected cost under linearized dynamics and quadratic cost; (10) produces an opposing trajectory to what p(u k |x k ) does by maximizing the expected cost under locally linear quadratic assumptions about the dynamics.

Note that the open-loop control strategies in (8) depend on the action of the other player.

Therefore, equations FORMULA28 ensure we have a cooperative game in which the protagonist and the adversary alternate between taking best possible and worst possible local actions during the trajectory optimization phase.

This helps maintain equilibrium around the system's desired trajectory, while ensuring robustness in local policies.

Substituting FORMULA28 into (7) and equating coefficients of δx k , δu k , δv k to those of DISPLAYFORM12 we obtain a quadratic value function at time k, through the backward pass given by BID18 in the appendix.

Say, the protagonist first implements its strategy, then transmits its information to the adversary, who subsequently chooses its strategy; it follows that the adversary can choose a more favorable outcome since it knows what the protagonist's choice of strategy is.

It becomes obvious that the best action for the protagonist is to choose a control strategy that is an optimal response to the choice of the adversary determined from DISPLAYFORM13 Similarly, if the roles of the players are changed, the protagonist response to the adversary's worst choice will be DISPLAYFORM14 Therefore, it does not matter that the order of play is predetermined.

We end up with an iterative dynamic game, where each agent's strategy depends on its previous actions.

The update rules for the Q coefficients are determined using a Gauss-Newton approximation and is given in BID14 in the appendices.

In the forward pass, we integrate the state equation,ẋ, compute the protagonist's deterministic optimal policy and update the trajectory as follows: DISPLAYFORM15 Compared to previous GPS algorithms, the local controllers not only produce locally linear Gaussian controllers that favors the widest and highest entropy, they also have robustness to disturbance and modeling errors built into them in the H ∞ sense.

We arrive at a saddle point in the energy space of the cost function and we posit that the local controllers generated during the trajectory optimization phase become robust to external perturbations, modeling errors e.t.c.

We arrive at a saddle point in the energy space of the cost function and we posit that the local controllers generated during the trajectory optimization phase become robust to external perturbations, modeling errors e.t.c.

The next section shows how we generate the function V (x k ) that guarantees saddle-point equilibria for our examples.

The dynamics of the two player system is given by the tuple {x In order to avoid the GMM not being a good separator of boundaries of complex modes, we follow BID0 , and use the GMM to generate a prior for the regression phase.

This enables us to obtain different linear modes at separate time steps based on the observed transitions, even when the states are dissimilar.

The correct linear mode is obtained from the empirical covariance of {x t , u t , v t } with x t+1 in the current samples at time t. As in BID0 and BID13 , we improve sample efficiency by refitting the GMM at each iteration to all of the samples at all time steps from the current iteration and the previous 3 iterations and use this to construct a good prior for the dynamics.

We then obtain linear Gaussian dynamics by fitting Gaussian distributions to samples {x DISPLAYFORM0 T .

The prior allows us to build a normal-inverse Wishart prior on the conditioned Gaussians so that the maximum a posteriori estimates for mean µ and covariance Σ are given by DISPLAYFORM1 where Σ e and µ e are respectively the empirical covariance and mean of the dataset and Φ, µ 0 , m and n 0 are prior parameters so chosen: Φ = n 0Σ and µ 0 =μ.

As in BID13 , we set n 0 = m = 1 in order to fit the prior to many samples than what is available at each time step.

The trajectories from the previous subsection are used to generate training data for global policies for the controller and adversary.

The local policies pτ (u k |x k ) and pτ (v k |x k ) will ideally be generated for all possible initial states x i 1 ∼ p(x k ).

Since the iLQG-based linearized dynamics will only be valid within a finite region of the state space; we used the KL-divergence constraint proposed in BID15 to ensure the current protagonist policy does not diverge too much from the previous policy.

The learning problem involves imposing KL constraints on the cost function such that the protagonist controller distribution agree with the global policy π θ (u t |x t ) by performing the following alternating optimization between two steps at each iteration i: DISPLAYFORM0 Essentially, p(u k |x k ) above generates robust local policies; The first step in (12) solves for a robust local policy p(u k |x k ) via the min-max operation, by constraining p(u k |x k ) against its global policy π i using the given KL divergence constraint; the second step projects the local linear Gaussian controller distribution onto the constraint set Π Θ , with respect to the divergence D(p i , π).

The local policy that governs the agent's dynamics is given by DISPLAYFORM1 Notice that the state is linearly dependent on the mean of the distribution p(u k |x k ) and the covariance is independent of v k ; we therefore end up with a linear Gaussian controller for the robust guided policy search algorithm.

For linear Gaussian dynamics and policies, the iterative KL constraint during the S-step translates to minimizing the KL-divergence between policies i.e. , DISPLAYFORM2 For the nonlinear cases that we treat in this work, the KL-divergence term in the S-step above is flipped as proposed in BID15 so that DISPLAYFORM3 .

Therefore, the S-step minimizes, where x k,i,j is the j th sample from p i (x k ) obtained by running p i (u k |x k ) on the real system, and D i are the trajectory samples rolled out on the system.

Our robust GPS algorithm is thus given in algorithm 2.

We follow the prior works in BID0 BID12 BID13 BID15 in computing the KL divergence term and we refer readers to these works for a more detailed treatment.

DISPLAYFORM4

In this section, we present experiments to (i) confirm our hypothesis that guided policy search methods, with carefully engineered complex high-dimensional policies, fail when exposed to the simplest of all perturbation signals; and (ii) answer the question of robustness using the trajectory optimization scheme and the robust guided policy framework we have presented.

We solve this under unknown dynamics.

We answer both questions in this paper by using physics engines for policies that do not use visual features as feedback.

Our validation examples are implemented in the MuJoCo physics engine BID23 and the pybox2d game engine BID5 , aided by the publicly available GPS codebase BID7 .

Highdimensional policy experiments are implemented on a PR2 robot, while low-dimensional policy experiments are implemented using a 2-DOF cart-pole swing-up experiment in the pybox2d game engine.

The perturbation signal we consider are those that enter additively through the reward functions as described in subsection 4.1.

We conducted simulated experiments demonstrating that guided policy search policies are sensitive to disturbance introduced into the action space of their policies.

The 7-DoF robot result presented shortly previously appeared in our abstract that introduced robust GPS BID20 .

The states x k are the joint angles, joint velocities, pose and velocity of the end effector as 3 points in 3-D. We assume the initial velocity of the 2-link and robot arm are zero.

Experimental tasks.

We simulated a 3D peg insertion task by a robot into a hole at the bottom of the slot FIG3 .

The difficulty of this experiment stems from the discontinuity in dynamics from the contact between the peg and the walls.

The 2-link arm swing-up experiment involves learning to balance the arm vertically about the origin of the cart (see right inset of FIG3 ).

The diffculty lies in the discontinuity of the dynamics along the vertical axis of the arm when it is upright.

We initialized the linear-Gaussian controllers p i (u k |x k ), p i (v k |x k ) in the neighborhood of the initial state x 1 using a PD control law for both the inverted pendulum task and the peg insertion task.

Peg Insertion: We implement the sensitivity algorithm for the peg insertion task of BID7 with a robotic arm that requires dexterous manipulation.

The robot has 12 states consisting of joint angles and angular velocities with two controller states.

We train the protagonist's policy using the GPS algorithm.

We then pit an adversarial disturbance against the trained policy so that the adversary stays in closed-loop with the trained protagonist; The closed-loop cost function is given by DISPLAYFORM0 where γ represents the disturbance term, d x k denotes the end effector's (EE) position at state x k and d denotes the EE's position at the slot's base.

12 (ζ) is a term that makes the peg reach the target at the hole's base, precisely given by DISPLAYFORM1 .

We set w u and w p to 10 −6 and 1 respectively.

For various values of γ, we check the sensitivity of the trained policy and its effect on the task performance by maximizing the cost function above w.r.t v k .

We run each sensitivity experiment for a total of 10 iterations.

FIG4 shows that the adversary causes a sharp degradation in the protagonist's performance for values of γ < 1.5.

This corresponds to when the GPS-trained policy gets destabilized and the arm struggles to reach the desired target.

As values of γ ≥ 1.5, however, we find that the adversary has a reduced effect on task performance: the adversary's effect decreases as γ gets larger.

Video of this result is available at https://goo.gl/YmmdhC.Arm Swing-up: Similar to the peg insertion task, we carry out a sensitivity evaluation procedure as we did for the robot arm with the peg insertion experiment with a 2D arm.

The goal is to balance a 2D arm vertically about its origin.

This agent has 7 states made up of two joint angles, two joint angle velocities and a 3D end effector point.

The action space has two dimensions.

Contrary to the example in BID7 that uses the Bregman alternating direction method of multipliers algorithm, we implement this experiment using the mirror descent GPS algorithm.

We proceed as before: first, we optimize the optimal global policy using GPS on the agent; we then fix the agent's policy and pit various adversarial disturbances, controlled by the γ robustness term in order to evaluate its sensitivity.

We use a similar cost function as the one used for the peg insertion task.

FIG5 shows the evolution of the cost function as we vary the values of γ.

We notice that the augmented reward function gets larger as the adversary's torque increases in magnitude and for lower values of γ, the augmented cost is relatively low stays the same.

The values of γ < 10 12 in FIG5 represent the disturbance band where the protagonist's learned policy becomes unstable and the arm never reaches the vertical position (see videos here: https://goo.gl/52rKnt).

This experiment further confirms that the state-of-the-art reinforcement learning algorithms fail in the presence of additive disturbances to their parameter space making them brittle when used in situations that call for robustness.

To mitigate these sensitivity errors, we implement the robust two-player, zero-sum game framework provided in 5 in order to develop more robust deep RL controllers and mitigate modeling errors and uncertainty.

As proposed in section 5, our goal is to improve the robustness of the controller's policy in the presence of modeling errors and uncertainties and transfer errors.

We follow the formulation in section 5 and generate v k from zero-mean, unit variance noise samples in every iteration.

We employ various values of γ as a robustness parameter and we run the dynamic game during the trajectory optimization phase of the GPS algorithm.

Specifically, for the values of γ that the erstwhile policies in the previous subsection fail, we run the dynamic game algorithm to provide robustness in performnace at test time compared against the GPS algorithm.

We run experiments on the peg insertion task to verify the algorithm.

FIG5 shows the cost of running the robust GPS algorithm on the 7-DoF robot.

We see that the policies that show achieve optimal performance behavior are now less costly compared to vanilla GPS algorithm.

For values of the sensitivity term γ that the algorithm erstwhile fails in, we now see smoother execution of the trajectory in trying to achieve our goal.

The modeling phase of the algorithm is also much less data consuming as our GMM algorithm now takes less samples before generalizing to the global model.

We have evaluated the sensitivity of select deep reinforcement learning algorithms and shown that despite the most carefully designed policies, such policies implemented on real-world agents exhibit a potential for disastrous performances when unexpected such as when there exist modeling errors and discrepancy between training environment and real-world roll-outs (as evidenced by the results from the two dynamics the agent faces in our sensitivity experiment).

We then test the dynamic trajectory optimization two-player algorithm on a robot motor task using Levine et al's BID13 's guided policy search algorithm.

In our implementation of the dynamic game algorithm, we focus on the robustness parameters that cause the robot's policy to fail in the presence of the erstwhile sensysensitivity parameter.

We demonstrate that our two-player game framework allows agents operating under nonlinear dynamics to learn the underlying dynamics under significantly more finite samples than vanilla GPS algorithm does -thus improving upon the Gaussian model mixture method used in BID0 and BID13 .Having agents that are robust to unmodeled nonlinearities, dynamics, and high frequency modes in a nonlinear dynamical system has long been a fundamental question that control theory strives to achieve.

To the best of our knowledge, we are not aware of other works that addresses the robustness of deep policies that are trained end-to-end from a maximal robustness perspective.

In future work, we hope to replace the crude Gaussian Mixture Model for the dynamics with a more sophisticated nonlinear model, and evaluate how the agent behaves in the presence of unknown dynamics.

@highlight

This paper demonstrates how H-infinity control theory can help better design robust deep policies for robot motor taks

@highlight

Proposes to incorporate elements of robust control into guided policy research in order to devise a method that is resilient to perturbations and model mismatch.

@highlight

The paper presents a method for evaluating the sensitivity and robustness of deep RL policies, and proposes a dynamic game approach for learning robust policies.
Addressing uncertainty is critical for autonomous systems to robustly adapt to the real world.

We formulate the problem of model uncertainty as a continuous Bayes-Adaptive Markov Decision Process (BAMDP), where an agent maintains a posterior distribution over latent model parameters given a history of observations and maximizes its expected long-term reward with respect to this belief distribution.

Our algorithm, Bayesian Policy Optimization, builds on recent policy optimization algorithms to learn a universal policy that navigates the exploration-exploitation trade-off to maximize the Bayesian value function.

To address challenges from discretizing the continuous latent parameter space, we propose a new policy network architecture that encodes the belief distribution independently from the observable state.

Our method significantly outperforms algorithms that address model uncertainty without explicitly reasoning about belief distributions and is competitive with state-of-the-art Partially Observable Markov Decision Process solvers.

At its core, real-world robotics focuses on operating under uncertainty.

An autonomous car must drive alongside unpredictable human drivers under road conditions that change from day to day.

An assistive home robot must simultaneously infer users' intended goals as it helps them.

A robot arm must recognize and manipulate varied objects.

These examples share common themes: (1) an underlying dynamical system with unknown latent parameters (road conditions, human goals, object identities), (2) an agent that can probe the system via exploration, while ultimately (3) maximizing an expected long-term reward via exploitation.

The Bayes-Adaptive Markov Decision Process (BAMDP) framework BID9 elegantly captures the exploration-exploitation dilemma that the agent faces.

Here, the agent maintains a belief, which is a posterior distribution over the latent parameters φ given a history of observations.

A BAMDP can be cast as a Partially Observable Markov Decision Process (POMDP) BID7 whose state is (s, φ), where s corresponds to the observable world state.

By planning in the belief space of this POMDP, the agent balances explorative and exploitative actions.

In this paper, we focus on BAMDP problems in which the latent parameter space is either a discrete finite set or a bounded continuous set that can be approximated via discretization.

For this class of BAMDPs, the belief is a categorical distribution, allowing us to represent it using a vector of weights.

The core problem for BAMDPs with continuous state-action spaces is how to explore the reachable belief space.

In particular, discretizing the latent space can result in an arbitrarily large belief vector, which causes the belief space to grow exponentially.

Approximating the value function over the reachable belief space can be challenging: although point-based value approximations BID16 BID26 have been largely successful for approximating value functions of discrete POMDP problems, these approaches do not easily extend to continuous state-action spaces.

Monte-Carlo Tree Search approaches (Silver & Veness, 2010; BID10 are also prohibitively expensive in continuous state-action spaces: the width of the search tree after a single iteration is too large, preventing an adequate search depth from being reached.

Our key insight is that we can bypass learning the value function and directly learn a policy that maps beliefs to actions by leveraging the latest advancements in batch policy optimization algorithms BID32 .

Inspired by previous approaches that train learning algorithms with an ensemble of models BID30 BID43 , we examine model uncertainty through a BAMDP lens.

Although our approach provides only locally optimal policies, we believe that it offers a practical and scalable solution for continuous BAMDPs.

Our method, Bayesian Policy Optimization (BPO), is a batch policy optimization method which utilizes a black-box Bayesian filter and augmented state-belief representation.

During offline training, BPO simulates the policy on multiple latent models sampled from the source distribution FIG0 ).

At each simulation timestep, it computes the posterior belief using a Bayes filter and inputs the state-belief pair (s, b) to the policy.

Our algorithm only needs to update the posterior along the simulated trajectory in each sampled model, rather than branching at each possible action and observation as in MCTS-based approaches.

Our key contribution is the following.

We introduce a Bayesian policy optimization algorithm to learn policies that directly reason about model uncertainty while maximizing the expected long-term reward (Section 4).

To address the challenge of large belief representations, we introduce two encoder networks that balance the size of belief and state embeddings in the policy network FIG0 .

In addition, we show that our method, while designed for BAMDPs, can be applied to continuous POMDPs when a compact belief representation is available (Section 4.2).

Through experiments on classical POMDP problems and BAMDP variants of OpenAI Gym benchmarks, we show that BPO significantly outperforms algorithms that address model uncertainty without explicitly reasoning about beliefs and is competitive with state-of-the-art POMDP algorithms (Section 5).

The Bayes-Adaptive Markov Decision Process framework BID7 BID31 BID15 ) was originally proposed to address uncertainty in the transition function of an MDP.

The uncertainty is captured by a latent variable, φ ∈ Φ, which is either directly the transition function, e.g. φ sas = T (s, a, s ), or is a parameter of the transition, e.g. physical properties of the system.

The latent variable is either fixed or has a known transition function.

We extend the previous formulation of φ to address uncertainty in the reward function as well.

Formally, a BAMDP is defined by a tuple S, Φ, A, T, R, P 0 , γ , where S is the observable state space of the underlying MDP, Φ is the latent space, and A is the action space.

T and R are the parameterized transition and reward functions, respectively.

The transition function is defined as: T (s, φ, a , s , φ ) = P (s , φ |s, φ, a ) = P (s |s, φ, a )P (φ |s, φ, a , s ).

The initial distribution over (s, φ) is given by P 0 : S × Φ → R + , and γ is the discount.

Bayesian Reinforcement Learning (BRL) considers the long-term expected reward with respect to the uncertainty over φ rather than the true (unknown) value of φ.

The uncertainty is represented as a belief distribution b ∈ B over latent variables φ.

BRL maximizes the following Bayesian value function, which is the expected value given the uncertainty: DISPLAYFORM0 where the action is a = π(s, b).

The Bayesian reward and transition functions are defined in expectation with respect to DISPLAYFORM0 The belief distribution can be maintained recursively, with a black-box Bayes filter performing posterior updates given observations.

We describe how to implement such a Bayes filter in Section 4.1.The use of (s, b) casts the partially observable BAMDP as a fully observable MDP in belief space, which permits the use of any policy gradient method.

We highlight that a reactive Bayesian policy in belief space is equivalent to a policy with memory in observable space BID13 .

In our work, the complexity of memory is delegated to a Bayes filter that computes a sufficient statistic of the history.

In partially observable MDPs (POMDPs), the states can be observed only via a noisy observation function.

Mixed-observability MDPs (MOMDPs) BID21 are similar to BAMDPs: their states are (s, φ), where s is observable and φ is latent.

Although any BAMDP problem can be cast as a POMDP or a MOMDP problem BID7 , the source of uncertainty in a BAMDP usually comes from the transition function, not the unobservability of the state as it does with POMDPs and MOMDPs.

A long history of research addresses belief-space reinforcement learning and robust reinforcement learning.

Here, we highlight the most relevant work and refer the reader to BID9 , BID34 , and BID0 for more comprehensive reviews of the Bayes-Adaptive and Partially Observable MDP literatures.

Belief-Space Reinforcement Learning.

Planning in belief space, where part of the state representation is a belief distribution, is intractable BID23 .

This is a consequence of the curse of dimensionality: the dimensionality of belief space over a finite set of variables equals the size of that set, so the size of belief space grows exponentially.

Many approximate solvers focus on one or more of the following: 1) value function approximation, 2) compact, approximate belief representation, or 3) direct mapping of belief to an action.

QMDP BID17 assumes full observability after one step to approximate Q-value.

Point-based solvers, like SARSOP BID16 and PBVI BID26 , exploit the piecewise-linear-convex structure of POMDP value functions (under mild assumptions) to approximate the value of a belief state.

Samplingbased approaches, such as BAMCP BID10 and POMCP BID35 , combine Monte Carlo sampling and simple rollout policies to approximate Q-values at the root node in a search tree.

Except for QMDP, these approaches target discrete POMDPs and cannot be easily extended to continuous spaces.

BID38 extend POMCP to continuous spaces using double progressive widening.

Model-based trajectory optimization methods BID28 BID40 have also been successful for navigation on systems like unmanned aerial vehicles and other mobile robots.

Neural network variants of POMDP algorithms are well suited for compressing highdimensional belief states into compact representations.

For example, QMDP-Net BID14 jointly trains a Bayes-filter network and a policy network to approximate Q-value.

Deep Variational Reinforcement Learning BID12 learns to approximate the belief using variational inference and a particle filter, and it uses the belief to generate actions.

Our method is closely related to Exp-GPOMDP BID1 , a model-free policy gradient method for POMDPs, but we leverage model knowledge from the BAMDP and revisit the underlying policy optimization method with recent advancements.

BID25 use Long Short-Term Memory (LSTM) BID11 to encode a history of observations to generate an action.

The key difference between our method and Peng et al. FORMULA0 is that BPO explicitly utilizes the belief distribution, while in Peng et al.(2018) the LSTM must implicitly learn an embedding for the distribution.

We believe that explicitly using a Bayes filter improves data efficiency and interpretability.

Robust (Adversarial) Reinforcement Learning.

One can bypass the burden of maintaining belief and still find a robust policy by maximizing the return for worst-case scenarios.

Commonly referred to as Robust Reinforcement Learning BID19 , this approach uses a min-max objective and is conceptually equivalent to H-infinity control BID3 ) from classical robust control theory.

Recent works have adapted this objective to train agents against various external disturbances and adversarial scenarios BID27 BID2 BID24 .

Interestingly, instead of training against an adversary, an agent can also train to be robust against model uncertainty with an ensemble of models.

For example, Ensemble Policy Optimization (EPOpt) BID30 trains an agent on multiple MDPs and strives to improve worst-case performance by concentrating rollouts on MDPs where the current policy performs poorly.

Ensemble-CIO BID18 optimizes trajectories across a finite set of MDPs.

While adversarial and ensemble model approaches have proven to be robust even to unmodeled effects, they may result in overly conservative behavior when the worst-case scenario is extreme.

In addition, since these methods do not infer or utilize uncertainty, they perform poorly when explicit information-gathering actions are required.

Our approach is fundamentally different from them because it internally maintains a belief distribution.

As a result, its policies outperform robust policies in many scenarios.

Adaptive Policy Methods.

Some approaches can adapt to changing model estimates without operating in belief space.

Adaptive-EPOpt BID30 retrains an agent with an updated source distribution after real-world interactions.

PSRL BID22 samples from a source distribution, executes an optimal policy for the sample for a fixed horizon, and then re-samples from an updated source distribution.

These approaches can work well for scenarios in which the latent MDP is fixed throughout multiple episodes.

Universal Policy with Online System Identification (UP-OSI) BID43 learns to predict the maximum likelihood estimate φ M LE and trains a universal policy that maps (s, φ M LE ) to an action.

However, without a notion of belief, both PSRL and UP-OSI can over-confidently execute policies that are optimal for the single estimate, causing poor performance in expectation over different MDPs.

We propose Bayesian Policy Optimization, a simple policy gradient algorithm for BAMDPs (Algorithm 1).

The agent learns a stochastic Bayesian policy that maps a state-belief pair to a probability distribution over actions π : S × B → P (A).

During each training iteration, BPO collects trajectories by simulating the current policy on several MDPs sampled from the prior distribution.

During the simulation, the Bayes filter updates the posterior belief distribution at each timestep and sends the updated state-belief pair to the Bayesian policy.

By simulating on MDPs with different latent variables, BPO observes the evolution of the state-belief throughout multiple trajectories.

Since the state-belief representation makes the partially observable BAMDP a fully observable Belief-MDP, any batch policy optimization

Require: Bayes filter ψ(·), initial belief b 0 (φ), P 0 , policy π θ0 , horizon H, n itr , n sample 1: for i = 1, 2, · · · , n itr do 2:for n = 1, 2, · · · , n sample do 3: DISPLAYFORM0 Update policy: DISPLAYFORM1 Execute a t on M , observing r t , s t 11: DISPLAYFORM2 algorithm (e.g., BID32 ) can be used to maximize the Bayesian Bellman equation FORMULA0 ).One key challenge is how to represent the belief distribution over the latent state space.

To this end, we impose one mild requirement, i.e., that the belief can be represented with a fixed-size vector.

For example, if the latent space is discrete, we can represent the belief as a categorical distribution.

For continuous latent state spaces, we can use Gaussian or a mixture of Gaussian distributions.

When such specific representations are not appropriate, we can choose a more general uniform discretization of the latent space.

Discretizing the latent space introduces the curse of dimensionality.

An algorithm must be robust to the size of the belief representation.

To address the high-dimensionality of belief space, we introduce a new policy network structure that consists of two separate networks to independently encode state and belief FIG0 .

These encoders consist of multiple layers of nonlinear (e.g., ReLU) and linear operations, and they output a compact representation of state and belief.

We design the encoders to yield outputs of the same size, which we concatenate to form the input to the policy network.

The encoder networks and the policy network are jointly trained by the batch policy optimization.

Our belief encoder achieves the desired robustness by learning to compactly represent arbitrarily large belief representations.

In Section 5, we empirically verify that the separate belief encoder makes our algorithm more robust to large belief representations ( FIG1 ).As with most policy gradient algorithms, BPO provides only a locally optimal solution.

Nonetheless, it produces robust policies that scale to problems with high-dimensional observable states and beliefs (see Section 5).

Given an initial belief b 0 , a Bayes filter recursively performs the posterior update: DISPLAYFORM0 where η is the normalizing constant, and the transition function is defined as T (s, φ, a , s , φ ) = P (s , φ |s, φ, a ) = P (s |s, φ, a )P (φ |s, φ, a , s ).

At timestep t, the belief b t (φ t ) is the posterior distribution over Φ given the history of states and actions, (s 0 , a 1 , s 1 , ..., s t ).

When φ corresponds to physical parameters for an autonomous system, we often assume that the latent states are fixed.

Our algorithm utilizes a black-box Bayes filter to produce a posterior distribution over the latent states.

Any Bayes filter that outputs a fixed-size belief representation can be used; for example, we use an extended Kalman filter to maintain a Gaussian distribution over continuous latent variables in the LightDark environment in Section 5.

When such a specific representation is not appropriate, we can choose a more general discretization of the latent space to obtain a computationally tractable belief update.

For our algorithm, we found that uniformly discretizing the range of each latent parameter into K equal-sized bins is sufficient.

From each of the resulting K |Φ| bins, we form an MDP by selecting the mean bin value for each latent parameter.

Then, we approximate the belief distribution with a categorical distribution over the resulting MDPs.

We approximate the Bayes update in Equation 2 by computing the probability of observing s under each discretized φ ∈ {φ 1 , · · · , φ K |Φ| } as follows: DISPLAYFORM1 where the denominator corresponds to η.

As we verify in Section 5, our algorithm is robust to approximate beliefs, which allows the use of computationally efficient approximate Bayes filters without degrading performance.

A belief needs only to be accurate enough to inform the agent of its actions.

Although BPO is designed for BAMDP problems, it can naturally be applied to POMDPs.

In a general POMDP where state is unobservable, we need only b(s), so we can remove the state encoder network.

Knowing the transition and observation functions, we can construct a Bayes filter that computes the belief b over the hidden state: DISPLAYFORM0 where η is the normalization constant, and Z is the observation function, Z(s, a , o ) = P (o |s, a ), of observing o after taking action a at state s.

Then, BPO optimizes the following Bellman equation: DISPLAYFORM1 For general POMDPs with large state spaces, however, discretizing state space to form the belief state is impractical.

We believe that this generalization is best suited for beliefs with conjugate distributions, e.g., Gaussians.

We evaluate BPO on discrete and continuous POMDP benchmarks to highlight its use of information-gathering actions.

We also evaluate BPO on BAMDP problems constructed by varying physical model parameters on OpenAI benchmark problems BID4 .

For all BAMDP problems with continuous latent spaces (Chain, MuJoCo), latent parameters are sampled in the continuous space in Step 3 of Algorithm 1, regardless of discretization.

We compare BPO to EPOpt and UP-MLE, robust and adaptive policy gradient algorithms, respectively.

We also include BPO-, a version of our algorithm without the belief and state encoders; this version directly feeds the original state and belief to the policy network.

Comparing with BPO-allows us to better understand the effect of the encoders.

For UP-MLE, we use the maximum likelihood estimate (MLE) from the same Bayes filter used for BPO, instead of learning an additional online system identification (OSI) network as originally proposed by UP-OSI.

This lets us directly compare performance when a full belief distribution is used (BPO) rather than a point estimate (UP-MLE).

For the OpenAI BAMDP problems, we also compare to a policy trained with TRPO in an environment with the mean values of the latent parameters.

All policy gradient algorithms (BPO, BPO-, EPOpt, UP-MLE) use TRPO as the underlying batch policy optimization subroutine.

We refer the reader to Appendix A.1 for parameter details.

For all algorithms, we compare the results from the seed with the highest mean reward across multiple random seeds.

Although EPOpt and UP-MLE are the most relevant algorithms that use batch policy optimization to address model uncertainty, we emphasize that neither formulates the problems as BAMDPs.

As shown in FIG0 , the BPO network's state and belief encoder components are identical, consisting of two fully connected layers with N h hidden units each and tanh activations (N h = 32 for Tiger, Chain, and LightDark; N h = 64 for MuJoCo).

The policy network also consists of two fully connected layers with N h hidden units each and tanh activations.

For discrete action spaces (Tiger, Chain), the output activation is a softmax, resulting in a categorical distribution over the discrete actions.

For continuous action spaces (LightDark, MuJoCo), we represent the policy as a Gaussian distribution.

FIG1 illustrates the normalized performance for all algorithms and experiments.

We normalize by dividing the total reward by the reward of BPO.

For LightDark, which has negative reward, we first shift the total reward to be positive and then normalize.

Appendix A.2 shows the unnormalized rewards.

Tiger (Discrete POMDP).

In the Tiger problem, originally proposed by BID13 , a tiger is hiding behind one of two doors.

An agent must choose among three actions:listen, or open one of the two doors; when the agent listens, it receives a noisy observation of the tiger's position.

If the agent opens the door and reveals the tiger, it receives a penalty of -100.

Opening the door without the tiger results in a reward of 10.

Listening incurs a penalty of -1.

In this problem, the optimal agent listens until its belief about which door the tiger is behind is substantially higher for one door vs. the other.

frame Tiger as a BAMDP problem with two latent states, one for each position of the tiger.

FIG1 demonstrates the benefit of operating in state-belief space when information gathering is required to reduce model uncertainty.

Since the EPOpt policy does not maintain a belief distribution, it sees only the most recent observation.

Without the full history of observations, EPOpt learns only that opening doors is risky; because it expects the worst-case scenario, it always chooses to listen.

UP-MLE leverages all past observations to estimate the tiger's position.

However, without the full belief distribution, the policy cannot account for the confidence of the estimate.

Once there is a higher probability of the tiger being on one side, the UP-MLE policy prematurely chooses to open the safer door.

BPO significantly outperforms both of these algorithms, learning to listen until it is extremely confident about the tiger's location.

In fact, BPO achieves close to the approximately optimal return found by SARSOP (19.0 ± 0.6), a state-of-the-art offline POMDP solver that approximates the optimal value function rather than performing policy optimization BID16 .

Chain-10 (tied) 364.5 ± 0.5 365.0 ± 0.4 366.1 ± 0.2 Chain-10 (semitied) 364.9 ± 0.8 364.8 ± 0.3 365.1 ± 0.3 321.6 ± 6.4 The BPO policy moves toward the light to obtain a better state estimate before moving toward the goal.

To evaluate the usefulness of the independent encoder networks, we consider a variant of the Chain problem BID37 .

The original problem is a discrete MDP with five states {s i } 5 i=1 and two actions {A, B}. Taking action A in state s i transitions to s i+1 with no reward; taking action A in state s 5 transitions to s 5 with a reward of 10.

Action B transitions from any state to s 1 with a reward of 2.

However, these actions are noisy: in the canonical version of Chain, the opposite action is taken with slip probability 0.2.

In our variant, the slip probability is uniformly sampled from [0, 1.0] at the beginning of each episode.2 In this problem, either action provides equal information about the latent parameter.

Since active information-gathering actions do not exist, BPO and UP-MLE achieve similar performance.

FIG1 shows that our algorithm is robust to the size of latent space discretization.

We discretize the parameter space with 3, 10, 100, 500, and 1000 uniformly spaced samples.

At coarser discretizations (3, 10), we see little difference between BPO and BPO-.

However, with a large discretization (500, 1000), the performance of BPO-degrades significantly, while BPO maintains comparable performance.

The performance of BPO also slightly degrades when the discretization is too fine, suggesting that this level of discretization makes the problem unnecessarily complex.

FIG1 shows the best discretization (10).In this discrete domain, we compare BPO to BEETLE BID29 and MCBRL BID41 , state-of-the-art discrete Bayesian reinforcement learning algorithms, as well as Perseus BID36 , a discrete POMDP solver.

In addition to our variant, we consider a more challenging version where the slip probabilities for both actions must be estimated independently.

BID29 refer to this as the "semi-tied" setting; our variant is "tied." BPO performs comparably to all of these benchmarks TAB0 .Light-Dark (Continuous POMDP).

We consider a variant of the LightDark problem proposed by BID28 , where an agent tries to reach a known goal location while being uncertain about its own position.

At each timestep, the agent receives a noisy observation of its location.

In our problem, the vertical dashed line is a light source; the farther the agent is from the light, the noisier its observations.

The agent must decide either to reduce uncertainty by moving closer to the light, or to exploit by moving from its estimated position to the goal.

We refer the reader to Appendix A.3 for details about the rewards and observation noise model.

This example demonstrates how to apply BPO to general continuous POMDPs (Section 4.2).

The latent state is the continuous pose of the agent.

For this example, we parameterize the belief as a Gaussian distribution and perform the posterior update with an Extended Kalman Filter, as in BID28 .Figure 3 compares sample trajectories from different algorithms on the LightDark environment.

Based on its initial belief, the BPO policy moves toward a light source to acquire less noisy observations.

As it becomes more confident in its position estimate, it changes direction toward the light and then moves straight to the goal.

Both EPOpt and UP-MLE move straight to the goal without initially reducing uncertainty.

MuJoCo (Continuous BAMDP).

Finally, we evaluate the algorithms on three simulated benchmarks from OpenAI Gym BID4 using the MuJoCo physics simulator BID39 : HalfCheetah, Swimmer, and Ant.

Each environment has several latent physical parameters that can be changed to form a BAMDP.

We refer the reader to Appendix A.4 for details regarding model variation and belief parameterization.

The MuJoCo benchmarks demonstrate the robustness of BPO to model uncertainty.

For each environment, BPO learns a universal policy that adapts to the changing belief over the latent parameters.

Figure 4 highlights the performance of BPO on Ant.

BPO can efficiently move to the right even when the model substantially differs from the nominal model (Figure 4a ).

It takes actions that reduce entropy more quickly than UP-MLE (Figure 4b ).

The belief over the possible MDPs quickly collapses into a single bin (Figure 4c ), which allows BPO to adapt the policy to the identified model.

Figure 5 provides a more in-depth comparison of the long-term expected reward achieved by each algorithm.

In particular, for the HalfCheetah environment, BPO has a higher average return than both EPOpt and UP-MLE for most MDPs.

Although BPO fares slightly worse than UP-MLE on Swimmer, we believe that this is largely due to random seeds, especially since BPO-matches UP-MLE's performance FIG1 ).Qualitatively, all three algorithms produced agents with reasonable gaits in most MDPs.

We postulate two reasons for this.

First, the environments do not require active informationgathering actions to achieve a high reward.

Furthermore, for deterministic systems with little noise, the belief collapses quickly ( Figure 4b) ; as a result, the MLE is as meaningful as the belief distribution.

As demonstrated by BID30 , a universally robust policy for these problems is capable of performing the task.

Therefore, even algorithms that do not maintain a history of observations can perform well.

Bayesian Policy Optimization is a practical and scalable approach for continuous BAMDP problems.

We demonstrate that BPO learns policies that achieve performance comparable to state-of-the-art discrete POMDP solvers.

They also outperform state-of-the-art robust policy gradient algorithms that address model uncertainty without formulating it as a BAMDP problem.

Our network architecture scales well with respect to the degree of latent parameter space discretization due to its independent encoding of state and belief.

We highlight that BPO is agnostic to the choice of batch policy optimization subroutine.

Although we used TRPO in this work, we can also use more recent policy optimization algorithms, such as PPO BID33 , and leverage improvements in variance-reduction techniques BID42 ).BPO outperforms algorithms that do not explicitly reason about belief distributions.

Our Bayesian approach is necessary for environments where uncertainty must actively be reduced, as shown in FIG1 and FIG2 .

If all actions are informative (as with MuJoCo, Chain) and the posterior belief distribution easily collapses into a unimodal distribution, UP-MLE provides a lightweight alternative.

BPO scales to fine-grained discretizations of latent space.

However, our experiments also suggest that each problem has an optimal discretization level, beyond which further discretization may degrade performance.

As a result, it may be preferable to perform variable-resolution discretization rather than an extremely fine, single-resolution discretization.

Adapting iterative densification ideas previously explored in motion planning BID8 and optimal control BID20 to the discretization of latent space may yield a more compact belief representation while enabling further improved performance.

An alternative to the model-based Bayes filter and belief encoder components of BPO is learning to directly map a history of observations to a lower-dimensional belief embedding, analogous to BID25 .

This would enable a policy to learn a meaningful belief embedding without losing information from our a priori choice of discretization.

Combining a recurrent policy for unidentified parameters with a Bayes filter for identified parameters offers an intriguing future direction for research efforts.

After each action, an agent receives a noisy observation of its location, which is sampled from a Gaussian distribution, o ∼ N ([x, y] , w(x)), where [x, y] is the true location.

The noise variance is a function of x and is minimized when x = 5: w(x) = 1 2 (x − 5) 2 + const.

There is no process noise.

The reward function is r(s, a) = − 1 2 ( s − g 2 + a 2 ), where s is the true agent position and g is the goal position.

A large penalty of −5000 s T − g 2 is incurred if the agent does not reach the goal by the end of the time horizon, analogous to the strict equality constraint in the original optimization problem BID28 .

For ease of analysis, we vary two parameters for each environment.

For HalfCheetah, the front and back leg lengths are varied.

For Ant, the two front leg lengths are varied.

Swimmer has four body links, so the first two link lengths vary together according to the first parameter, and the last two links vary together according to the second parameter.

We chose to vary link lengths rather than friction or the damping constant because a policy trained on a single nominal environment can perform well across large variations in those parameters.

All link lengths vary by up to 20% of the original length.

To construct a Bayes filter, the 2D-parameter space is discretized into a 5 × 5 grid with a uniform initial belief.

We assume Gaussian noise on the observation, i.e. o = f φ (s, a) + w with w ∼ N (0, σ 2 ), with φ being the parameter corresponding to the center of each grid cell.

It typically requires only a few steps for the belief to concentrate in a single cell of the grid, even when a large σ 2 is assumed.

@highlight

We formulate model uncertainty in Reinforcement Learning as a continuous Bayes-Adaptive Markov Decision Process and present a method for practical and scalable Bayesian policy optimization.

@highlight

Using a Bayesian approach, there is a better trade-off between exploration and exploitation in RL
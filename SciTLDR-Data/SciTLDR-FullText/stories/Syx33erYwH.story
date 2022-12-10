Imitation learning aims to inversely learn a policy from expert demonstrations, which has been extensively studied in the literature for both single-agent setting with Markov decision process (MDP) model, and multi-agent setting with Markov game (MG) model.

However, existing approaches for general multi-agent Markov games are not applicable to multi-agent extensive Markov games, where agents make asynchronous decisions following a certain order, rather than simultaneous decisions.

We propose a novel framework for asynchronous multi-agent generative adversarial imitation learning (AMAGAIL) under general extensive Markov game settings, and the learned expert policies are proven to guarantee subgame perfect equilibrium (SPE), a more general and stronger equilibrium than Nash equilibrium (NE).

The experiment results demonstrate that compared to state-of-the-art baselines, our AMAGAIL model can better infer the policy of each expert agent using their demonstration data collected from asynchronous decision-making scenarios (i.e., extensive Markov games).

Imitation learning (IL) also known as learning from demonstrations allows agents to imitate expert demonstrations to make optimal decisions without direct interactions with the environment.

Especially, inverse reinforcement learning (IRL) (Ng et al. (2000) ) recovers a reward function of an expert from collected demonstrations, where it assumes that the demonstrator follows an (near-)optimal policy that maximizes the underlying reward.

However, IRL is an ill-posed problem, because a number of reward functions match the demonstrated data (Ziebart et al. (2008; ; Ho & Ermon (2016) ; Boularias et al. (2011) ), where various principles, including maximum entropy, maximum causal entropy, and relative entropy principles, are employed to solve this ambiguity (Ziebart et al. (2008; ; Boularias et al. (2011); Ho & Ermon (2016) ; Zhang et al. (2019) ).

Going beyond imitation learning with single agents discussed above, recent works including Song et al. (2018) , Yu et al. (2019) , have investigated a more general and challenging scenario with demonstration data from multiple interacting agents.

Such interactions are modeled by extending Markov decision processes on individual agents to multi-agent Markov games (MGs) (Littman & Szepesvári (1996) ).

However, these works only work for synchronous MGs, with all agents making simultaneous decisions in each turn, and do not work for general MGs, allowing agents to make asynchronous decisions in different turns, which is common in many real world scenarios.

For example, in multiplayer games (Knutsson et al. (2004) ), such as Go game, and many card games, players take turns to play, thus influence each other's decision.

The order in which agents make decisions has a significant impact on the game equilibrium.

In this paper, we propose a novel framework, asynchronous multi-agent generative adversarial imitation learning (AMAGAIL): A group of experts provide demonstration data when playing a Markov game (MG) with an asynchronous decision-making process, and AMAGAIL inversely learns each expert's decision-making policy.

We introduce a player function governed by the environment to capture the participation order and dependency of agents when making decisions.

The participation order could be deterministic (i.e., agents take turns to act) or stochastic (i.e., agents need to take actions by chance).

A player function of an agent is a probability function: given the perfectly known agent participation history, i.e., at each previous round in the history, we know which agent(s) participated, it provides the probability of the agent participating in the next round.

With the general MG model, our framework generalizes MAGAIL (Song et al. (2018) ) from the synchronous Markov games to (asynchronous) Markov games, and the learned expert policies are proven to guarantee subgame perfect equilibrium (SPE) (Fudenberg & Levine (1983) ), a stronger equilibrium than the Nash equilibrium (NE) (guaranteed in MAGAIL Song et al. (2018) ).

The experiment results demonstrate that compared to GAIL (Ho & Ermon (2016) ) and MAGAIL (Song et al. (2018) ), our AMAGAIL model can better infer the policy of each expert agent using their demonstration data collected from asynchronous decision-making scenarios.

Markov games (MGs) (Littman (1994) ) are the cases of N interacting agents, with each agent making a sequence of decisions with strategies only depending on the current state.

A Markov game 1 is denoted as a tuple (N, S, A, Y, ζ, P, η, r, γ) with a set of states S and N sets of actions

.

At each time step t with a state s t ∈ S, if the indicator variable I i,t = 1, an agent i is allowed to take an action; otherwise, I i,t = 0, the agent i does not take an action.

As a result, the participation vector I t = [I 1,t , · · · , I N,t ] indicates active vs inactive agents at step t.

The set of all possible participation vectors is denoted as I, namely, I t ∈ I. Moreover, h t−1 = [I 0 , · · · , I t−1 ] represent the participation history from step 0 to t − 1.

The player function Y (governed by the environment) describes the probability of an agent i being allowed to make an action at a step t, given the participation history h t−1 , namely, Y (i|h t−1 ).

ζ defines the participation probability of an agent at the initial time step

Note that, the player function can be naturally extended to a higher-order form when the condition includes both previous participation history and previous state-action history; thus, it can be adapted to non-Markov processes.

The initial states are determined by a distribution η : S → [0, 1].

Let φ denotes no participation, determined by player function Y , the transition process to the next state follows a transition function: P : S × A 1 ∪ {φ} × · · · × A N ∪ {φ} → P(S).

Agent i obtains a (bounded) reward given by a function r i : S ×A i → R 2 .

Agent i aims to maximize its own total expected return R i = ∞ t=0 γ t r i,t , where γ ∈ [0, 1] is the discount factor.

Actions are chosen through a stationary and stochastic policy π i : S × A i → [0, 1].

In this paper, bold variables without subscript i denote the concatenation of variables for all the agents, e.g., all actions as a, the joint policy defined as π(a|s) = N i=1 π i (a i |s), r as all rewards.

Subscript −i denotes all agents except for i, then (a i , a −i ) represents the action of all N agents (a 1 , · · · , a N ).

We use expectation with respect to a policy π to denote an expectation with respect to the trajectories it generates.

For

, denotes the following sample process as s 0 ∼ η, I 0 ∼ ζ, I t ∼ Y , a ∼ π(·|s t ), s t+1 ∼ P (s t+1 |s t , a), for ∀i ∈ [N ].

Clearly, when the player function Y (i|h t−1 ) = 1 for all agents i's at any time step t, a general Markov game boils down to a synchronous Markov game (Littman (1994); Song et al. (2018) ), where all agents take actions at all steps.

To distinguish our work from MAGAIL and be consistent with the literature Chatterjee et al. (2004) and Hansen et al. (2013) , we refer the game setting discussed in MAGAIL as synchronous Markov games (SMGs), and that of our work as Markov games (MGs).

In synchronous Markov games (SMGs), all agents make simultaneous decisions at any time step t, with the same goal of maximizing its own total expected return.

Thus, agents' optimal policies are interrelated and mutually influenced.

Nash equilibrium (NE) has been employed as a solution concept to resolve the dependency across agents, where no agents can achieve a higher expected reward by unilaterally changing its own policy (Song et al. (2018) ).

However, in Markov games (MGs) allowing asynchronous decisions, there exist situations where agents encounter states (subgames) resulted from other agents' "trembling-hand" actions.

Since the NE does not consider the "trembling-hand" resulted states and subgames, when trapped in these situations, agents are not able to make optimal decisions based on their polices under NE.

To address this problem, Selten firstly proposed subgame perfect equilibrium (SPE) (Selten (1965) ).

SPE ensures NE for every possible subgame of the original game.

It has been shown that in a finite or infinite extensive-form game with either discrete or continuous time, best-response strategies all converge to SPE, rather than NE (Selten (1965) ; Abramsky & Winschel (2017); Xu (2016) ).

In synchronous Markov games, MAGAIL (Song et al. (2018) ) was proposed to learn experts' policies constrained by Nash equilibrium.

Since there may exist multiple Nash equilibrium solutions, a maximum causal entropy regularizer is employed to resolve the ambiguity.

Thus, the optimal policies can be found by solving the following multi-agent reinforcement learning problem.

where

, and β is a weight to the entropy regularization term.

In practice, the reward function is unknown.

MAGAIL applies multi-agent IRL (MAIRL) below to recover experts' reward functions, with ψ as a convex regularizer,

Moreover, MAGAIL solves MARL • MAIRL ψ (π E ) to inversely learn each expert's policy via applying generative adversarial imitation learning (Ho & Ermon (2016)

D wi is a discriminator for agent i that classifies the experts' vs policy trajectories.

π θ represent the learned experts' parameterized policies, which generate trajectories with maximized the scores from

Extending multi-agent imitation learning to general Markov games is challenging, because of the asynchronous decision making and dynamic state (subgame) participating.

In this section, we will tackle this problem using subgame perfect equilibrium (SPE) solution concept.

In a Markov game (MG), the Nash equilibrium needs to be guaranteed at each state s ∈ S 3 , namely, we apply subgame perfect equilibrium (SPE) solution concept instead.

Formally, a set of agent

is an SPE if at each state s ∈ S (also considered as a root node of a subgame), no agent can achieve a higher reward by unilaterally changing its policy on the root node or any other descendant nodes of the root node, i.e., ∀i

Therefore, our constrained optimization problem is (Filar & Vrieze (2012) , Theorem 3.7.2)

For an agent i with a probability of taking action a at state s t given a history h t−1 , its Q-function is

where P r(I t |h t−1 ) = i:Ii,t=1 Y (i|h t−1 ) j:Ij,t=0 (1 − Y (j|h t−1 )) is the probability of participation vector I t given history h t−1 .

The constraints in eq. (5) guarantee an SPE, i.e.,

.

Consistent with MAGAIL (Song et al. (2018) ) the objective has a global minimum of zero under SPE, and π forms SPE if and only if f r (π, v) reaches zero while being a feasible solution.

We use AMA-RL(r) to denote the set of policies that form a sSPE under reward function r, and can maximize γ-discounted causal entropy of policies:

where q i is defined in eq. (7).

Our objective is to define a suitable inverse operator AMAIRL in analogy to MAIRL in eq. (2).

The key idea of MAIRL is to choose a reward that creates a margin between a set of experts and every other set of policies.

However, the constraints in SPE optimization eq. (8) can make this challenging.

To that end, we derive an equivalent Lagrangian formulation of eq. (8) to defined a margin between the expected rewards of two sets of policies to capture the "difference".

The SPE constraints in eq. (9) state that no agent i can obtain a higher expected reward via 1-step temporal (TD) difference learning.

We replace 1-step constraints with (t+1)-step constraints with the solution remaining the same as AMARL.

The general idea is consistent with MAGAIL (Song et al. (2018) ).

The detailed derivation is in Appx A.1.

The updated (t+1)-step constraints arê

By implementing the (t+1)-step formulation eq. (10), we aim to construct the Lagrangian dual of the primal in eq. (8).

Since for any policy π, f r (π,v) = 0 givenv i defined as in Theorem 1 in Appx A.1 (proved in Lemma 1 in Appx A.2), we just focus on the constraints in eq. (10) to get the dual problem

where T t i is the set of all length-t trajectories of the form {s (j) , a

i | · |H| Lagrange multipliers, andv i is defined as in Theorem 1 in Appx A.1.

Theorem 2 illustrates that a specific λ is able to recover the difference of the sum of expected rewards between not all optimal and all optimal policies.

Theorem 2 For any two policies π * and π, let

be the probability of generating the sequence τ i using policy π i and π * −i and h t−1 , where P r(h t−1 ) = P r(I 0 ) t−1 k=1 P r(I k |h k−1 ) is the probability of history h t−1 .

Then

where the dual function is L (t+1) r (π * , λ * π ) and each multiplier can be considered as the probability of generating a trajectory of agent i ∈ N , τ i ∈ T t i , and h t−1 ∈ H.

Theorem 2 (proved in Appx A.3) provides a horizon to establish AMAIRL objective function with regularizer ψ.

is the discounted causal entropy for policy π i when other agents follow π E−i , and β is a hyper-parameter controlling the strength of the entropy regularization term as in GAIL (Ho & Ermon (2016) ).

We first define the asynchronous occupancy measure in Markov games:

with a policy π i ∈ Π, define its asynchronous occupancy measure ρ

The occupancy measure can be interpreted as the distribution of state-action pairs that an agent i encounters under the participating and nonparticipating situations.

Notably, when ζ(i) = 1, Y (i|h t−1 ) = 1 for all t ∈ {1, ..., ∞}, h t−1 ∈ H, asynchronous occupancy measure in MG turns to the occupancy measure defined in MAGAIL and GAIL, i.e., ρ p πi = ρ πi .

With the additively separable regularization ψ, for each agent i, π Ei is the unique optimal response to other experts π E−i .

Therefore we obtain the following theorem (see proof of Theorem 3 in Appendix A.4):

, and that AMA-RL(r) has a unique solution 4 for all r ∈ AMA-IRL ψ (π E ), then

where π i , E −i denotes π i for agent i, and π E−i for other agents.

In practice, we are only able to calculate ρ p π E and ρ p π .

As following MAGAIL (Song et al. (2018)

In this section, we propose practical algorithms for asynchronous multi-agent imitation learning, and introduce three representative scenarios with different player function structures.

The selected ψ i in Proposition 1 (in Appx A.5) contributes to the corresponding generative adversarial model where each agent i has a generator π θi and a discriminator, D wi .

When the generator is allowed to behave, the produced behavior will receive a score from discriminator.

The generator attempts to train the agent to maximize its score and fool the discriminator.

We optimize the following Three agents all have stochastic player functions (i.e., yellow boxes), thus, each agent has a certain probability to make an action w.r.t the player function given the participation history h t−1 ; in this example, only agents #2 and #3 happen to make actions, and agent #1 does not.

objective:

In practice, the input of AMAGAIL is Z, the demonstration data from N expert agents in the same environment, where the demonstration data Z = {(s t , a)} T t=0 are collected by sampling s 0 ∼ η,

The assumptions include knowledge of N, γ, S, A. Transition P , initial state distribution η, agent distribution ζ, player function Y are all considered as black boxes, and no additional expert interactions with environment during training process are allowed.

In the RL process of finding each agent's policy π θi , we follow MAGAIL (Song et al. (2018) ) to apply Multi-agent Actor-Critic with Kronecker-factors (MACK) and use the advantage function with the baseline V ν for variance reduction.

The summarized algorithm is presented in Algorithm 1 in Appx B.

In MGs, the order in which agents make decisions is determined by the player function Y .

Below, we discuss three representative structures of player function Y , including synchronous participation, deterministic participation, and stochastic participation.

Synchronous participation.

When Y (i|h t−1 ) = 1 holds for all agents i ∈ [N ] at every step t (as shown in Figure 1a ), agents make simultaneous actions, and a general Markov game boils down to a simple synchronous Markov game.

Deterministic participation.

When the player function Y (i|h t−1 ) is deterministic for all agents i ∈ [N ], it can only output 1 or 0 at each step t. Many board games, e.g., Go, and Chess, have deterministic player functions, where agents take turns to play.

Figure 1b shows an example of deterministic participation structure.

Stochastic participation.

When the player function is stochastic, namely, Y (i|h t−1 ) ∈ [0, 1] for some agent i ∈ [N ] at certain time step t, the agent i will make an action by chance.

As illustrated in Figure 1c , three agents all have stochastic player functions at step t, and agent #1 does not take an action at step t, while agent #2 and #3 happen to take actions.

We evaluate AMAGAIL with both stochastic and deterministic player function structures under cooperative and competitive games, respectively.

We compared our AMAGAIL with two baselines, including Behavior Cloning (BC) by OpenAI (Dhariwal et al. (2017) ) and decentralized Multi-agent generative adversarial imitation learning (MAGAIL) (Song et al. (2018) ).

The results are collected by averaging over 5 random seeds (refer to Appx C for implementation details).

We use the particle environment (Lowe et al. (2017) ) as a basic setting, and customize it into four games to allow different asynchronous player function structures.

Deterministic Cooperative Navigation: Three agents (agent #1, #2 and #3) need to cooperate to get close to three randomly placed landmarks through physical actions.

They get high rewards if they are close to the landmarks and Figure 2 : Average true reward from cooperative tasks.

Performance of experts and random policies are normalized to one and zero respectively.

We use inverse log scale for better comparison.

are penalized for any collision with each other.

Ideally, each agent should cover a single distinct landmark.

In this process, the agents must follow a deterministic participation order to take actions, i.e., in the first round all three agents act, in the second round only agent #1 and #2 act, in the third round only agent #1 acts, and repeat these rounds until the game is completed.

Stochastic Cooperative Navigation: This game is the same with deterministic cooperative navigation except that all three agents have a stochastic player function.

Each agent has 50% chance to act at each round t. Deterministic Cooperative Reaching: This game has three agents with their goals as cooperatively reaching a single landmark with minimum collision.

In this game, agents follow a deterministic player function, same as that in deterministic cooperative navigation game, to make actions.

Stochastic Predator-Prey: Three slower cooperating agents (referred to as adversaries) chase a faster agent in an environment of two landmarks; the faster agent acts first, then each adversary with a stochastic player function of 50% chance to act with the same goal of catching the faster agent.

The adversaries and the agent need to avoid two randomly placed landmarks.

The adversaries collect rewards when touching the agent, where the agent is penalized.

Note that, an agent that does not participate in a round of a game does not get a reward.

In these four game environments, agents are first trained with Multi-agent ACKTR ; Song et al. (2018) ), thus the true reward functions are available, which enable us to evaluate the quality of recovered policies.

When generating demonstrations from well-trained expert agents, a "null" (no-participation) as a placeholder action is recorded for each no-participation round in the trajectory.

The quality of a recovered policy is evaluated by calculating agents' average true reward of a set of generated trajectories.

We compare our AMAGAIL with two baselines -behavior cloning (BC) (Pomerleau (1991)) and decentralized Multi-agent generative adversarial imitation learning (MAGAIL) (Song et al. (2018) ).

Behavior cloning (BC) utilizes the maximum likelihood estimation for each agent independently to approach their policies.

Decentralized multi-agent generative adversarial imitation learning (MAGAIL) treats each agent with a unique discriminator working as the agent's reward signal and a unique generator as the agent's policy.

It follows the maximum entropy principle to match agents' occupancy measures from recovered policies to demonstration data.

We compare AMAGAIL with baselines under three particle environment games, namely, deterministic cooperative navigation, stochastic cooperative navigation, and deterministic cooperative reaching games.

Figure 2 show the normalized rewards, when learning policies with BC, MAGAIL and AMAGAIL, respectively.

When there is only a small amount of expert demonstrations, the normalized rewards of BC and AMAGAIL increase, especially, when less demonstration data are used, i.e., less than 400 demonstrations.

After a sufficient amount of demonstrations are used, i.e., more than 400, AMAGAIL has higher rewards than BC and MAGAIL.

This makes sense since at certain time steps there exist non-participating agents (based on the player functions), but BC and MAGAIL models consider the no-participation as an action the agent can choose, where in reality it is governed by the environment.

On the other hand, with the introduced player function Y , AMAGAIL characterizes such no participation events correctly, thus more accurately learns the expert policies.

The normalized awards of BC are roughly unchanged in Figure 2 (a)&(c), and in Figure 2 (b) after 400 demonstrations, which seems contradictory to that of Ross & Bagnell (2010) ; Song et al. (2018) , and can be explained as follows.

In Figure 2 (b) (stochastic cooperative navigation), the performance of BC is low when using less demonstrations, but increases rapidly as more demonstrations are used, and finally converges to the "best" performance around 0.65 with 300 demonstrations.

In Figure 2 (a) (resp.

Figure 2(c) ), deterministic cooperative navigation (resp.

reaching) is easier to learn compared Table 1 : Average agent rewards in stochastic predator-prey.

We compare behavior cloning (BC) and multi-agent GAIL (MAGAIL) methods.

Best results are marked in bold.

Note that high vs low rewards are preferred, when running BC for agent vs adversaries, respectively).

Task Stochastic Predator-Prey Agent Behavior Cloning MAGAIL AMAGAIL Adversaries BC MAGAIL AMAGAIL Behavior Cloning Rewards −5.0 ± 10.8 −9.0 ± 13.1 −14.0 ± 19.4 −3.6 ± 8.5 −2.1 ± 6.9

with the stochastic cooperative navigation game shown in Figure 2 (b), since there is no randomness in the player function.

The performance with only 200 demonstrations is already stabilized at 0.7 (resp.

0.94).

In the stochastic cooperative navigation game (Figure 2(b) ), AMAGAIL performs consistently better than MAGAIL and BC.

However, in the deterministic cooperative navigation game (Figure 2(b) ), with 200 demonstration, AMAGAIL does not perform as well as MAGAIL.

This is due to the game setting, namely, two players actively searching for landmarks are sufficient to gain a high reward in this game.

The last agent, player #3, learned to be "lazy", without any motivation to promote the total shared reward among all agents.

In this case, it is hard for AMAGAIL to learn a good policy of player #3 with small amount of demonstration data, because player #3's has 2 3 absence rate, given the pre-defined deterministic participation function.

Hence, AMAGAIL does not have enough state-action pairs to learn player #3.

This gets improved when there are sufficient data, say, more than 400 demonstrations.

When we adjust the game setting from 3 landmarks to 1 landmark, i.e., all agents need to act actively to reach the landmark.

This is captured in the deterministic cooperative reaching game.

In this scenario, an inactive player will lower down the overall reward.

As shown in Figure 2 (c), AMAGAIL outperforms BC and MAGAIL consistently, even with a small amount of demonstration data.

5.2 PERFORMANCE WITH MIXED GAME MODE Now, we further evaluate the performance of AMAGAIL under a mixed game mode with both cooperative and adversarial players, i.e., stochastic predator-prey game.

Since there are two competing sides in this game, we cannot directly compare each methods' performance via expected reward.

Therefore, we use the Song et al. (2018)'s evaluation paradigm and compare with baselines by letting (agents trained by) BC play against (adversaries trained by) other methods, and vice versa.

From Table 1 , AMAGAIL consistently performs better than MAGAIL and BC.

Imitation learning (IL) aims to learn a policy from expert demonstrations, which has been extensively studied in the literature for single agent scenarios (Finn et al. (2016) ; Ho & Ermon (2016) ).

Behavioral cloning (BC) uses the observed demonstrations to directly learn a policy (Pomerleau (1991); Torabi et al. (2018) ).

Apprenticeship learning and inverse reinforcement learning (IRL) ((Ng et al. (2000) ; Syed & Schapire (2008); Ziebart et al. (2008; Boularias et al. (2011) )) seek for recovering the underlying reward based on expert trajectories in order to further learn a good policy via reinforcement learning.

The assumption is that expert trajectories generated by the optimal policy maximize the unknown reward.

Generative adversarial imitation learning (GAIL) and conditional GAIL (cGAIL) incorporate maximum casual entropy IRL (Ziebart et al. (2010) ) and the generative adversarial networks (Goodfellow et al. (2014) ) to simultaneously learn non-linear policy and reward functions (Ho & Ermon (2016); Zhang et al. (2019) ; Baram et al. (2017) ).

A few recent studies on multi-agent imitation learning, such as MAGAIL (Song et al. (2018) and MAAIRL (Yu et al. (2019) ), model the interactions among agents as synchronous Markov games, where all agents make simultaneous actions at each step t. These works fail to characterize a more general and practical interaction scenario, i.e., Markov games including turn-based games (Chatterjee et al. (2004) ), where agents make asynchronous decisions over steps.

In this paper, we make the first attempt to propose an asynchronous multi-agent generative adversarial imitation learning (AMAGAIL) framework, which models the asynchronous decision-making process as a Markov game and develops a player function to capture the participation dynamics of agents.

Experimental results demonstrate that our proposed AMAGAIL can accurately learn the experts' policies from their asynchronous trajectory data, comparing to state-of-the-art baselines.

Beyond capturing the dynamics of participation vs no-participation (as only two participation choices), our proposed player function Y (and AMAGAIL framework) can also capture a more general case 5 , where Y determines how the agent participates in a particular round, i.e., which action set A

A.1 TIME DIFFERENCE LEARNING Theorem 1.

For a certain policy π and reward r, letv i (s (t) ; π, r, h t−1 ) be the unique solution to the Bellman equation:

i ; π, r, h t−1 ) as the discounted expected return for the i-th agent conditioned on visiting the trajectory {s (j) , a (j) } t−1 j=0 , s (t) in the first t − 1 steps and choosing action a (t)

i at the t-th step, when other agents using policy π −i :

i ; π, r, h t−1 )

Then π is subgame perfect equilibrium if and only if:

Theorem 1 illustrates that if we replace the 1-step constraints with (t + 1)-step constraints, we still get the same solution as AMA-RL(r) in terms of a subgame perfect equilibrium solution.

A.2 EXISTENCE AND EQUIVALENCE OF V AND SUBGAME PERFECT EQUILIBRIUM Lemma 1 By definition ofv i (s (t) ; π, r, h t−1 ) in Theorem 1 andq i (s (t) , a i ; π, r, h t−1 ) in eq. 7.

Then for any π, f r (π,v) = 0.

Furthermore, π is subgame perfect equilibrium under r if and only if v i (s; π, r, h t−1 ) ≥q i (s, a i ; π, r, h t−1 ) for all i ∈ [N ], s ∈ S, a i ∈ A i and h t−1 ∈ H.

Proof We havê

i ; π, r, h t−1 )].

which utilizes the fact that a i and a −i are independent at s. Therefore, we can easily get f r (π,v) = 0.

If π is a subgame perfect equilibrium, and existing one or more of the constrains does not hold, so agent i can receive a strictly higher expected reward for rest of the states, which is against the subgame perfect equilibrium assumption.

If the constraints hold, i.e., for all i and (s, a i ),v i (s; π, r, h t−1 ) ≥q i (s, a i ; π, r, h t−1 ) then

Value iteration, thus, overv i (s; π, r, h t−1 ) converges.

If one can find another policy π so that v i (s; π, r, h t−1 ) < E πi [q i (s, a i ; π, r, h t−1 )], then at least one violation exists in the constraints since π i is a convex combination over action a i .

Therefore, for any policy π i and action a i for any agent i, E πi [q i (s, a i ; π, r, h t−1 )]

≥ E π i [q i (s, a i ; π, r, h t−1 )] always hold, so π i is the optimal reply to π −i , and π constitutes a subgame perfect equilibrium once it repeats this argument for all agents.

Notably, by assuming f r (π, v) = 0 for some v; if v satisfies the assumptions, then v =v.

Proof We use Q * ,q * ,v * to denote the Q,q andv quantities defined for policy π * .

For the two

For agent i, τ i and h t−1 we have, λ * π (τ i ; h t−1 ) · Q * i (τ i ; π * , r, h t−1 ) = P r(τ i ; h t−1 ) · Q * i (τ i ; π * , r, h t−1 ).

Therefore, the proof of AMA-RL•AMA-IRL can be derived in a similar fashion with GAIL (Ho & Ermon (2016) ) and MAGAIL (Song et al. (2018) Theorem 3 and Proposition 1 discuss the differences from the single agent scenario similar in Song et al. (2018) .

On the one hand, in Theorem 3 we make the assumption that AMA-RL(r) has a unique solution, which is always true in the single agent case due to convexity of the space of the optimal policies.

On the other hand, in Proposition 1 we remove the entropy regularizer because here the causal entropy for π i may depend on the policies of the other agents, so the entropy regularizer on two sides are not the same quantity.

Specifically, the entropy for the left hand side conditions on π E−i and the entropy for the right hand side conditions on π −i (which would disappear in the single-agent case).

B APPENDIX B. ALGORITHM Generate state-action pairs of batch size B from π u through the process: s 0 ∼ η, I 0 ∼ ζ, I t ∼ Y, a ∼ π u (·|s t ), s t+1 ∼ P (s t+1 |s t , a); φ is recorded as a placeholder action when an agent does not participate in a round; denote the generated state-action pair set as X .

Sample state-action pairs from Z with batch size B; denote the demonstrated state-action pair set as X E .

for each agent i = 1, · · · , N do

Filter out state-action pairs (s, φ) from X and X E .

Filter out state-action pairs (s, φ) from X and X E .

Update ν i to decrease the objective: E X ,Y [(V νi (s) − V * (s)) 2 ].

Update θ i by policy gradient with the setting step sizes: E X ,Y [ θi π θi (a i |s i )A i (s, a)].

C APPENDIX C. EXPERIMENT DETAILS C.1 HYPERPARAMETERS For the particle environment, we follow the setting of MAGAIL (Song et al. (2018) ) to use two layer multiple layer perceptrons with 128 cells in each layer for the policy generator network, value

@highlight

This paper extends the multi-agent generative adversarial imitation learning to extensive-form Markov games.
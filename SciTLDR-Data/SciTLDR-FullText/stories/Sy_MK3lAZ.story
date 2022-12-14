Most existing deep reinforcement learning (DRL) frameworks consider action spaces that are either discrete or continuous space.

Motivated by the project of design Game AI for King of Glory (KOG), one the world’s most popular mobile game, we consider the scenario with the discrete-continuous hybrid action space.

To directly apply existing DLR frameworks, existing approaches either approximate the hybrid space by a discrete set or relaxing it into a continuous set, which is usually less efficient and robust.

In this paper, we propose a parametrized deep Q-network (P-DQN) for the hybrid action space without approximation or relaxation.

Our algorithm combines DQN and DDPG and can be viewed as an extension of the DQN to hybrid actions.

The empirical study on the game KOG validates the efficiency and effectiveness of our method.

In recent years, the exciting field of deep reinforcement learning (DRL) have witnessed striking empirical achievements in complicated sequential decision making problems that are once believed unsolvable.

One active area of the application of DRL methods is to design artificial intelligence (AI) for games.

The success of DRL in the game of Go provides a promising methodology for game AI.

In addition to the game of Go, DRL has been widely used in other games such as Atari BID19 , Robot Soccer BID8 BID17 , and Torcs ) to achieve super-human performances.

However, most existing DRL methods only handle the environments with actions chosen from a set which is either finite and discrete (e.g., Go and Atari) or continuous (e.g. MuJoCo and Torcs) For example, the algorithms for discrete action space include deep Q-network (DQN) BID18 , Double DQN (Hasselt et al., 2016) , A3C BID20 ; the algorithms for continuous action space include deterministic policy gradients (DPG) BID29 and its deep version DDPG .Motivated by the applications in Real Time Strategic (RTS) games, we consider the reinforcement learning problem with a discrete-continuous hybrid action space.

Different from completely discrete or continuous actions that are widely studied in the existing literature, in our setting, the action is defined by the following hierarchical structure.

We first choose a high level action k from a discrete set {1, 2, · · · , K}; upon choosing k, we further choose a low level parameter x k ∈ X k which is associated with the k-th high level action.

Here X k is a continuous set for all k ∈ {1, . . . , K}.1 Therefore, we focus on a discrete-continuous hybrid action space A = (k, x k ) x k ∈ X k for all 1 ≤ k ≤ K .To apply existing DRL approaches on this hybrid action space, two straightforward ideas include:• Approximate A by an finite discrete set.

We could approximate each X k by a discrete subset, which, however, might lose the natural structure of X k .

Moreover, when X k is a region in the Euclidean space, establishing a good approximation usually requires a huge number discrete actions.• Relax A into a continuous set.

To apply existing DRL framework with continuous action spaces, BID8 define the following approximate space DISPLAYFORM0 where F k ⊆ R. Here f 1 , f 2 , . . . , f K is used to select the discrete action either deterministically (by picking arg max i f i ) or randomly (with probability softmax(f )).

Compared with the original action space A, A might significantly increases the complexity of the action space.

Furthermore, continuous relaxation can also lead to unnecessary confusion by over-parametrization.

For example, (1, 0, · · · , 0, x 1 , x 2 , x 3 , · · · , x K ) ∈ A and (1, 0, · · · , 0, x 1 , x 2 , x 3 , · · · , x K ) ∈ A indeed represent the same action (1, x 1 ) in the original space A.In this paper, we propose a novel DRL framework, namely parametrized deep Q-network learning (P-DQN), which directly work on the discrete-continuous hybrid action space without approximation or relaxation.

Our method can be viewed as an extension of the famous DQN algorithm to hybrid action spaces.

Similar to deterministic policy gradient methods, to handle the continuous parameters within actions, we first define a deterministic function which maps the state and each discrete action to its corresponding continuous parameter.

Then we define a action-value function which maps the state and finite hybrid actions to real values, where the continuous parameters are obtained from the deterministic function in the first step.

With the merits of both DQN and DDPG, we expect our algorithm to find the optimal discrete action as well as avoid exhaustive search over continuous action parameters.

To evaluate the empirical performances, we apply our algorithm to King of Glory (KOG), which is one of the most popular online games worldwide, with over 200 million active users per month.

KOG is a multi-agent online battle arena (MOBA) game on mobile devices, which requires players to take hybrid actions to interact with other players in real-time.

Empirical study indicates that P-DQN is more efficient and robust than BID8 's method that relaxes A into a continuous set and applies DDPG.

In reinforcement learning, the environment is usually modeled by a Markov decision process (MDP) M = {S, A, p, p 0 , γ, r}, where S is the state space, A is the action space, p is the Markov transition probability distribution, p 0 is the probability distribution of the initial state, r(s, a) is the reward function, and γ ∈ [0, 1] is the discount factor.

An agent interacts with the MDP sequentially as follows.

At the t-th step, suppose the MDP is at state s t ∈ S and the agent selects an action a t ∈ A, then the agent observe an immediate reward r(s t , a t ) and the next state s t+1 ∼ p(s t+1 |s t , a t ).

A stochastic policy π maps each state to a probability distribution over A, that is, π(a|s) is defined as the probability of selecting action a at state s.

Whereas a deterministic µ : S → A maps each state to a particular action in A. Let R t = j≥t γ j−t r(s j , a j ) be the cumulative discounted reward starting from time-step t. We define the state-value function and the action-value function of policy π as V π = E(R t |S t = s; π) and Q π (s, a) = E(R t |S 0 = s, A 0 = a; π), respectively.

Moreover, we define the optimal state-and action-value functions as V π = sup π V π and Q * = sup π Q π , respectively, where the supremum is taken over all possible policies.

The goal of the agent is to find a policy the maximizes the expected total discounted reward J(π) = E(R 0 |π), which is can be achieved by estimating Q * .

Broadly speaking, reinforcement learning algorithms can be categorized into two classes: value-based methods and policy-based methods.

Value-based methods first estimate Q * and then output the greedy policy with respect to that estimate.

Whereas policy-based methods directly optimizes J(π) as a functional of π.

The Q-learning algorithm BID36 ) is based on the Bellman equation DISPLAYFORM0 which has Q * as the unique solution.

In the tabular setting, the algorithm updates the Q-function by iteratively applying the sample counterpart of the Bellman equation DISPLAYFORM1 where α > 0 is the stepsize and s is the next state observed given the current state s and action a. However, when the state space S is so large that it is impossible to store all the states in memory, function approximation for Q * is applied.

Deep Q-Networks (DQN) BID19 approximates Q * using a neural network Q(s, a; w) ≈ Q(s, a), where w is the network weights.

In the t-th iteration, the DQN updates the parameter using the gradient of the least squares loss function DISPLAYFORM2 In practice, DQN is trained with techniques such as experience replay and asynchronous stochastic gradient descent methods BID20 ) which enjoy great empirical success.

In addition to the value-based methods, the policy-based methods directly models the optimal policy.

In specific, let π be any policy.

We write p t (·|s; π) as the distribution of S t given S 1 = s with actions executed according to policy π.

We define the discounted probability distribution ρ π by DISPLAYFORM3 Then the objective of policy-based methods is to find a policy that maximizes the expected reward DISPLAYFORM4 Let π θ be a stochastic policy parametrized by θ ∈ Θ. For example, π θ could be a neural network in which the last layer is a softmax layer with |A| neurons.

The stochastic gradient methods aims at finding a parameter θ that maximizes J(π θ ) via gradient descent.

The stochastic policy gradient theorem BID33 states that DISPLAYFORM5 3)The policy gradient algorithm iteratively updates θ using estimates of (2.3).

For example, the REINFORCE algorithm BID37 updates θ using ∇ θ log π θ (a t |s t ) · r t .

Moreover, the actor-critic methods use another neural network Q(s, a; w) to estimate the value function Q π θ (s, a) associated to policy π θ .

This algorithm combines the value-based and policy-based perspectives together, and is recently used to achieve superhuman performance in the game of Go BID31 .

When the action space is continuous, value-based methods will no longer be computationally tractable because of taking maximum over the action space A in (2.2), which in general cannot be computed efficiently.

The reason is that the neural network Q(s, a; w) is nonconvex when viewed as a function of a; max a∈A Q(s, a; w) is the global minima of a nonconvex function, which is NP-hard to obtain in the worst case.

To resolve this issue, the continuous Q-learning BID6 rewrite the action value function as Q(s, a) = V (s) + A(s, a), where V (s) is the state value function and A(s, a) is the advantage function that encodes the relative advantage of each action.

These functions are approximated by neural networks V (s; θ V ) and A(s, a; θ A ), respectively, where θ V and θ A are network weights.

The action value function is given by DISPLAYFORM0 Then in the t-th iteration, the continuous Q-learning updates θ v and θ a by taking a gradient step using the least squares loss function DISPLAYFORM1 Moreover, it is also possible to adapt policy-based methods to continuous action spaces by considering deterministic policies.

Let µ θ : S → A be a deterministic policy.

Similar to (2.3), the deterministic policy gradient (DPG) theorem BID29 states that DISPLAYFORM2 Furthermore, this deterministic version of the policy gradient theorem can be viewed as the limit of (2.3) with the variance of π θ going to zero.

Based on (2.4), the DPG algorithm BID29 and the deep deterministic policy gradient (DDPG) algorithm are proposed.

General reinforcement learning There is a huge body of literature in reinforcement learning, we refer readers to textbooks by Sutton & Barto (1998); Szepesvári FORMULA6 for detailed introduction.

Combined with the recent advancement of deep learning BID5 , deep reinforcement learning becomes a blossoming field of research with a plethora of new algorithms which achieve surprising empirical success in a variety of applications that are previously considered extremely difficult and challenging.

Finite discrete action space methods For reinforcement learning problems with finite action spaces, BID18 propose the DQN algorithm, which first combines the deep neural networks with the classical Q-learning algorithm BID36 .

A variety of extensions are proposed to improve DQN, including Double DQN , dueling DQN BID35 , bootstrap DQN BID23 , asynchronous DQN BID20 , and averaged- DQN Anschel et al. (2017) .In terms of policy-based methods, BID33 propose the REINFORCE algorithm, which is the basic form of policy gradient.

An important extension is the actor-critic method BID13 , whose asynchronous deep version A3C BID20 produces the stateof-the-art performances on the Arcade Learning Environment (ALE) benchmark BID1 .Continuous action space methods Moreover, for DRL on continuous action spaces, BID29 proposes the deterministic policy gradient algorithm and deterministic actor-critic algorithms.

This work is further extended by , which propose the DDPG algorithm, which is an model-free actor critic algorithm using deep neural networks to parametrize the policies.

A related line of work is policy optimization methods, which improve the policy gradient method using novel optimization techniques.

These methods include natural gradient descent BID12 , trust region optimization BID27 , proximal gradient descent BID28 , mirror descent BID21 , and entropy regularization BID22 .Hybrid actions A related body of literature is the recent work on reinforcement learning with a structured action space, which contains finite actions each parametrized by a continuous parameter.

To handle such parametrized actions, BID8 applies the DDPG algorithm on the relaxed action space directly, and BID17 proposes a learning framework updating the parameters for discrete actions and continuous parameters alternately.

Game AI Recently remarkable advances have been made in building AI bots for computer games using deep reinforcement learning.

These games include Atari Games, a collection of video games, Texas Hold'em, a multi-player poker game, and Doom, a first-person shooter game.

See BID18 ; BID9 ; BID15 ; BID2 for details and see BID11 for a comprehensive survey.

More notably, the computer Go agent AlphaGo achieves super-human performances by defeating the human world champion Lee Sedol.

Two more complicated class of games are the real-time strategy (RTS) games and MOBA games.

These are multi-agent games which involves searching within huge state and action spaces that are possibly continuous.

Due to the difficulty of these problems, current research for these games are rather inadequate with most existing work consider specific scenarios instead of the full-fledged RTS or MOBA games.

See, e.g., BID4 ; BID24 for an recent attempt on applying DRL methods to RTS games.

This section introduces the proposed framework to handle the application with hybrid discretecontinuous action space.

We consider a MDP with a parametrized action space A, which consists of K discrete actions each associated with a continuous parameter.

In specific, we assume that any action a ∈ A can be written as a = (k, x k ), where k ∈ {1, . . .

, K} is the discrete action, and x k ∈ X k is a continuous parameter associated with the k-th discrete action.

Thus action a is a hybrid of discrete and continuous components with the value of the continuous action determined after the discrete action is chosen.

Then the parametrized action space A can be written as DISPLAYFORM0 In the sequel, we denote {1, . . .

, K} by [K] for short.

For the action space A in (4.1), we denote the action value function by Q(s, a) = Q(s, k, x k ) where s ∈ S, 1 ≤ k ≤ K, and x k ∈ X k .

Let k t be the discrete action selected at time t and let x kt be the associated continuous parameter.

Then the Bellman equation becomes DISPLAYFORM1 Here inside the conditional expectation on the right-hand side of (4.2), we first solve DISPLAYFORM2 , and then take the largest Q(s t+1 , k, x * k ).

Note that taking supremum over continuous space X k is computationally intractable.

However, the right-hand side of (4.2) can be evaluated efficiently providing x * k is given.

To elaborate this idea, first note that, when the function Q is fixed, for any s ∈ S and k ∈ [K], we can view x Q k (s) = argsup DISPLAYFORM3 as a function of state s. That is, we identify (4.3) as a function x Q k : S → X k .

Then we can rewrite the Bellman equation in (4.2) as DISPLAYFORM4 Note that this new Bellman equation resembles the classical Bellman equation in (2.1) with A = [K].

Similar to the deep Q-networks, we use a deep neural network Q(s, k, x k ; ω) to approximate Q(s, k, x k ), where ω denotes the network weights.

Moreover, for such a Q(s, k, x k ; ω), we approximate x Q k (s) in (4.3) with a deterministic policy network x k (·; θ) : S → X k , where θ denotes the network weights of the policy network.

That is, when ω is fixed, we want to find θ such that Q s, k, x k (s; θ); ω ≈ sup DISPLAYFORM5 Remark 4.1.

Readers who are familiar with the work by BID8 , that also claims to handle discrete-continuous hybrid action spaces, may be curious of its difference from the proposed P-DQN.

The key differences are as follows.• In BID8 , the discrete action types are parametrized as some continuous values, say f .

And the discrete action that is actually executed is chosen via k = arg max i f (i).

Such a trick actually turns the hybrid action space into a continuous action space, upon which the classical DDPG algorithm can be applied.

However, in our framework, the discrete action type is chosen directly by maximizing the action's Q value explicitly.• The Q network in BID8 uses the artificial parameters f as input, which makes it an action-value function estimator of current policy (Q π ).

While in our framework, the Q network is actually an approximate estimator of the optimal policy's action-value function (Q ).•

We note that P-DQN is an off-policy method that can use historical data, while it is hard to use historical data in BID8 because there is only discrete action k without parameters f .(a) Network of P-DQN (b) Network of DDPG Figure 1 : Illustration of the networks of P-DQN and DDPG BID8 .

P-DQN selects the discrete action type by maximizing Q values explicitly; while in DDPG, the discrete action with largest f , which can be seen as a continuous parameterization of K discrete action types, is chosen.

Also in P-DQN the state and action parameters are feed into the Q-network which outputs K action values for each action type; while in DDPG, the continuous parameterization f , instead of the actual action k taken, is feed into the Q-network.

Suppose that θ satisfies (4.4), then similar to DQN, we could estimate ω by minimizing the meansquared Bellman error via gradient descent.

In specific, in the t-th step, let ω t and θ t be the weights of the value network and the deterministic policy network, respectively.

To incorporate multi-step algorithms, for a fixed n ≥ 1, we define the n-step target y t by DISPLAYFORM0 We define the least squares loss function for ω by DISPLAYFORM1 Moreover, since we aim to find θ that minimizes Q[s, k, x k (s; θ); ω] with ω fixed, we define the loss function for θ by DISPLAYFORM2 Then we update ω t and θ t by gradient-based optimization methods.

Moreover, the gradients are given by DISPLAYFORM3 Here ∇ x Q(s, k, x k ; ω) and ∇ ω Q(s, k, x k ; ω) are the gradients of the Q-network with respect to its third argument and fourth argument, respectively.

By (5.5) and (5.4) we update the parameters using stochastic gradient methods.

In addition, note that in the ideal case, we would minimize the loss function Θ t (θ) in (5.3) when ω t is fixed.

From the results in stochastic approximation methods BID14 , we could approximately achieve such a goal in an online fashion via a two-timescale update rule BID3 .

In specific, we update ω with a stepsize α t that is asymptotically negligible compared with the stepsize β t for θ.

In addition, for the validity of

Input: Stepsizes {αt, βt} t≥0 , exploration parameter , minibatch size B, the replay memory D, and a probability distribution µ over the action space A for exploration.

Initialize network weights ω1 and θ1.

for t = 1, 2, . . .

, T doCompute action parameters x k ← x k (s , θt).

Select action at = (kt, x k t ) according to the -greedy policy at = a sample from distribution µ with probability , (kt, x k t ) such that kt = arg max k∈[K] Q(s , k, x k ; ωt) with probability 1 − .Take action at, observe reward rt and the next state st+1.

DISPLAYFORM0 Use data {y b , s b , a b } b∈ [B] to compute the stochastic gradient ∇ω Q t (ω) and ∇ θ Θ t (θ) defined in (5.5) and (5.4).

Update the parameters by ωt+1 ← ωt − αt · ∇ω Q t (ωt) and θt+1 ← θt − βt · ∇ θ Θ t (θt). end for stochastic approximation, we require {α t , β t } to satisfy the Robbins-Moron condition BID25 .

We present the P-DQN algorithm with experienced replay in Algorithm 1.Note that this algorithm requires a distribution µ defined on the action space A for exploration.

In each step, with probability , the agent sample an random action from µ; otherwise, it takes the greedy action with respect to the current value function.

In practice, if each X k is a compact set in the Euclidean space (as in our case), µ could be defined as the uniform distribution over A. In addition, as in the DDPG algorithm (Lillicrap et al., 2016), we can also add additive noise to the continuous part of the actions for exploration.

Moreover, we use experience replay BID18 to reduce the dependencies among the samples, which can be replaced by more sample-efficient methods such as prioritized replay .Moreover, we note that our P-DQN algorithm can easily incorporate asynchronous gradient descent to speed up the training process.

Similar to the asynchronous n-step DQN in BID20 , we consider a centralized distributed training framework where each process can compute its local gradient and synchronize with a global parameter server.

In specific, each local process runs an independent game environment to generate transition trajectories and use its own transitions to compute gradients with respect to ω and θ.

These local gradients are then aggregated across multiple processes to update the global parameters.

Note that these local stochastic gradients are independent.

Thus tricks such as experience replay can be avoided in the distributed setting.

Moreover, aggregating independent stochastic gradient decrease the variance of gradient estimation, which yields better algorithmic stability.

We present the asynchronous P-DQN algorithm in Algorithm 2.

For simplicity, here we only lay out the algorithm for each local process, which fetches ω and θ from the parameter server and computes the gradient.

The parameter server stores the global parameters ω, θ .

It updates the global parameters using the gradients sent from the local processes .

In addition we use the RMSProp BID10 to update the network parameters, which is shown to be more stable in practice.

The game King of Glory is a MOBA game, which is a special form of the RTS game where the players are divided into two opposing teams fighting against each other.

Each team has a team base located in either the bottom-left or the top-right corner which are guarded by three towers on each of the three lanes.

The towers can attack the enemies when they are within its attack range.

Each player controls one hero, which is a powerful unit that is able to move, kill, perform skills, and purchase

Input: exploration parameter , a probability distribution µ over the action space A for exploration, the max length of multi step return tmax, and maximum number of iterations Nstep.

Initialize global shared parameter ω and θ Set global shared counter Nstep = 0 Initialize local step counter t ← 1.

repeat Clear local gradients dω ← 0, dθ ← 0.

tstart ← t Synchronize local parameters ω ← ω and θ ← θ from the parameter server.

repeat Observe state st and let x k ← x k (st, θ ) Select action at = (kt, x k t ) according to the -greedy policy at = a sample from distribution µ with probability , (kt, x k t ) such that kt = arg max k∈[K] Q(st, k, x k ; ω ) with probability 1 − .Take action at, observe reward rt and the next state st+1.

t ← t + 1 Nstep ← Nstep + 1 until st is the terminal state or t − tstart = tmax Define the target y = 0 for terminal st DISPLAYFORM0 Update global θ and ω using dθ and dω with RMSProp BID10 ).

until Nstep > Nmax equipments.

The goal of the heroes is to destroy the base of the opposing team.

In addition, for both teams, there are computer-controlled units spawned periodically that march towards the opposing base in all the three lanes.

These units can attack the enemies but cannot perform skills or purchase equipments.

An illustration of the map is in FIG2 , where the blue or red circles on each lane are the towers.

During game play, the heroes advance their levels and obtain gold by killing units and destroying the towers.

With gold, the heros are able to purchase equipments such as weapons and armors to enhance their power.

In addition, by upgrading to the new level, a hero is able to improve its unique skills.

Whereas when a hero is killed by the enemy, it will wait for some time to reborn.

In this game, each team contains one, three, or five players.

The five-versus-five model is the most complicated mode which requires strategic collaboration among the five players.

In contrast, the one-versus-one mode, which is called solo, only depends on the player's control of a single hero.

In a solo game, only the middle lane is active; both the two players move along the middle lane to fight against each other.

The map and a screenshot of a solo game are given in FIG2 -(b) and (c), respectively.

In our experiments, we play focus on the solo mode.

We emphasize that a typical solo game lasts about 10 to 20 minutes where each player must make instantaneous decisions.

Moreover, the players have to make different types of actions including attack, move and purchasing.

Thus, as a reinforcement learning problem, it has four main difficulties: first, the state space has huge capacity; second, since there are various kinds of actions, the action space is complicated; third, the reward function is not well defined; and fourth, heuristic search algorithms are not feasible since the game is in real-time.

Therefore, although we consider the simplest mode of King of Glory, it is still a challenging game for artificial intelligence.

In this section, we applied the P-DQN algorithm to the solo mode of King of Glory.

In our experiments, we play against the default AI hero Lu Ban provided by the game, which is a shooter with long attack range.

To evaluate the performances, we compared our algorithm with the DDPG algorithm BID8 under fair condition.

In our experiment, the state of the game is represented by a 179-dimensional feature vector which is manually constructed using the output from the game engine.

These features consist of two parts.

The first part is the basic attributes of the two heroes, the computer-controlled units, and buildings such as the towers and the bases of the two teams.

For example, the attributes of the heroes include Health Point, Magic Point, Attack Damage, Armor, Magic Power, Physical Penetration/Resistance, and Magic Penetration/Resistance, and the attributes of the towers include Health Point and Attack Damage.

The second component of the features is the relative positions of other units and buildings with respect to the hero controlled by the P-DQN player as well as the attacking relations between other units.

We note that these features are directly extracted from the game engine without sophisticated feature engineering.

We conjecture that the overall performances could be improved with a more careful engineered set of features.

We simplify the actions of a hero into K = 6 discrete action types: Move, Attack, UseSkill1, UseSkill2, UseSkill3, and Retreat.

Some of the actions may have additional continuous parameters to specify the precise behavior.

For example, when the action type is k = Move, the direction of movement is given by the parameter x k = α, where α ∈ [0, 2π].

Recall that each hero's skills are unique.

For Lu Ban, the first skill is to throw a grenade at some specified location, the second skill is to launch a missile in a particular direction, and the last skill is to call an airship to fly in a specified direction.

A complete list of actions as well as the associated parameters are given in TAB0 .

The ultimate goal of a solo game is to destroy the opponent's base.

However, the final result is only available when the game terminates.

Using such kind of information as the reward for training might not be very effective, as it is very sparse and delayed.

In practice, we manually design the rewards using information from each frame.

Specifically, we define a variety of statistics as follows. (In the sequel, we use subscript 0 to represent the attributes of our side and 1 to represent those of the opponent.)• Gold difference GD = Gold 0 − Gold 1 .

This statistic measures the difference of gold gained from killing hero, soldiers and destroying towers of the opposing team.

The gold can be used to buy weapons and armors, which enhance the offending and defending attributes of the hero.

Using this value as the reward encourages the hero to gain more gold.• Health Point difference (HPD = HeroRelativeHP 0 − HeroRelativeHP 1 ): This statistic measures the difference of Health Point of the two competing heroes.

A hero with higher Health Point can bear more severe damages while hero with lower Health Point is more likely to be killed.

Using this value as the reward encourages the hero to avoid attacks and last longer before being killed by the enemy.• Kill/Death KD = Kills 0 − Kills 1 .

This statistic measures the historical performance of the two heroes.

If a hero is killed multiple times, it is usually considered more likely to lose the game.

Using this value as the reward can encourage the hero to kill the opponent and avoid death.• Tower/Base HP difference THP = TowerRelativeHP 0 − TowerRelativeHP 1 , BHP = BaseRelativeHP 0 − BaseRelativeHP 1 .

These two statistics measures the health difference of the towers and bases of the two teams.

Incorporating these two statistic in the reward encourages our hero to attack towers of the opposing team and defend its own towers.• Tower Destroyed TD = AliveTower 0 − AliveTower 1 .

This counts the number of destroyed towers, which rewards the hero when it successfully destroy the opponent's towers.• Winning Game W = AliveBase 0 − AliveBase 1 .

This value indicates the winning or losing of the game.• Moving forward reward: MF = x + y, where (x, y) is the coordinate of Hero 0 : This value is used as part of the reward to guide our hero to move forward and compete actively in the battle field.

The overall reward is calculated as a weighted sum of the time differentiated statistics defined above.

In specific, the exact formula is r t = 0.5 × 10 −5 (MF t − MF t−1 ) + 0.001(GD t − GD t−1 ) + 0.5(HPD t − HPD t−1 DISPLAYFORM0 The coefficients are set roughly inversely proportional to the scale of each statistic.

We note that our algorithm is not very sensitive to the change of these coefficients in a reasonable range.

In the experiments, we use the default parameters of skills provided by the game environment (usually pointing to the opponent hero's location).

We found such kind of simplification does not affect to the overall performance of our agent.

In addition, to deal with the periodic problem of the direction of movement, we use (cos(α), sin(α)) to represent the direction and learn a normalized two-dimensional vector instead of a degree (in practice, we add a normalize layer at the end to ensure this).

In addition, the 6 discrete actions are not always usable, due to skills level up, lack of Magic Point (MP), or skills Cool Down(CD).

In order to deal with this problem, we replace the max k∈ [K] with max k∈[K] and k is usable when selecting the action to perform, and calculating multi-step target as in Equation 5.1.For the network structure, recall that we use a feature vector of 179 dimensions as the state.

We set both the value-network and the policy network as multi-layer fully-connected deep networks.

The networks are in the same size of 256-128-64 nodes in each hidden layer, with the Relu activation function.

During the training and testing processes, we set the frame skipping parameter to 2.

This means that we take actions every 3 frames or equivalently, 0.2 second, which adapts to the human reaction time, 0.1 second.

We set t max = 20 (4 seconds) to alleviate the delayed reward.

In order to encourage exploration, we use -greedy sampling in training with = 0.255.

In specific, the first 5 type actions We further smooth the original noisy curves (plotted in light colors) to their running average (plotted in dark colors).

In the 3 rows, we plot the average of episode lengths, reward sum averaged for each episode in training, and reward sum averaged for each episode in validation, for the two algorithms respectively.

Usually a positive reward sum indicates a winning game, and vice versa.

We can see that the proposed algorithm P-DQN learns much faster than its precedent work in our setting.

(a) Performance of P-DQN.

(b) Performance of DDPG are sampled with probability of 0.05 each and the action "Retreat" with probability 0.005.

For actions with additional parameters, since the parameters are in bounded sets, we draw these parameters from a uniform distribution.

Moreover, if the sampled action is infeasible, we execute the greedy policy from the feasible ones, so the effective exploration rate is less than .

We uses 48 parallel workers with constant learning rate 0.001 in training and 1 worker with deterministic sampling in validation.

The training and validating performances are plotted in Figure 3 .We implemented the DDPG BID8 ) algorithm within our learning environment to have a fair comparison.

The exact network structure is plotted in Figure 1 .

Each algorithm is allowed to run for 15 million steps, which corresponds to roughly 140 minutes of wall clock time when paralleled with 48 workers.

From the experiments results, we can see that our algorithm P-DQN can learn the value network and the policy network much faster comparing to the other algorithm.

In (a1), we see that the average length of games increases at first, reaches its peak when the two player's strength are close, and decreases when our player can easily defeat the opponent.

In addition, in (a2) and (a3), we see that the total rewards in an episode increase consistently in training as well as in test settings.

The DDPG algorithm may not be suitable for hybrid actions with both a discrete part and a continuous part.

The major difference is that maximization over k when we need to select a action is computed explicitly in P-DQN, instead of approximated implicitly with the policy network as in DDPG.

Moreover, with a deterministic policy network, we extend the DQN algorithm to hybrid action spaces of discrete and continuous types, which makes the P-DQN algorithm more suitable for realistic scenarios.

Previous deep reinforcement learning algorithms mostly can work with either discrete or continuous action space.

In this work, we consider the scenario with discrete-continuous hybrid action space.

In contrast of existing approaches of approximating the hybrid space by a discrete set or relaxing it into a continuous set, we propose the parameterized deep Q-network (P-DQN), which extends the classical DQN with deterministic policy for the continuous part of actions.

Empirical experiments of training AI for King of Glory, one of the most popular games, demonstrate the efficiency and effectiveness of P-DQN.

@highlight

A DQN and DDPG hybrid algorithm is proposed to deal with the discrete-continuous hybrid action space.
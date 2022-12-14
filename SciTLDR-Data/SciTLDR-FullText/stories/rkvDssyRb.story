We consider tackling a single-agent RL problem by distributing it to $n$ learners.

These learners, called advisors, endeavour to solve the problem from a different focus.

Their advice, taking the form of action values, is then communicated to an aggregator, which is in control of the system.

We show that the local planning method for the advisors is critical and that none of the ones found in the literature is flawless: the \textit{egocentric} planning overestimates values of states where the other advisors disagree, and the \textit{agnostic} planning is inefficient around danger zones.

We introduce a novel approach called \textit{empathic} and discuss its theoretical aspects.

We empirically examine and validate our theoretical findings on a fruit collection task.

When a person faces a complex and important problem, his individual problem solving abilities might not suffice.

He has to actively seek for advice around him: he might consult his relatives, browse different sources on the internet, and/or hire one or several people that are specialised in some aspects of the problem.

He then aggregates the technical, ethical and emotional advice in order to build an informed plan and to hopefully make the best possible decision.

A large number of papers tackle the decomposition of a single Reinforcement Learning task (RL, Sutton & Barto, 1998) into several simpler ones.

They generally follow a method where agents are trained independently and generally greedily to their local optimality, and are aggregated into a global policy by voting or averaging.

Recent works BID12 BID30 prove their ability to solve problems that are intractable otherwise.

Section 2 provides a survey of approaches and algorithms in this field.

Formalised in Section 3, the Multi-Advisor RL (MAd-RL) partitions a single-agent RL into a MultiAgent RL problem BID22 , under the widespread divide & conquer paradigm.

Unlike Hierarchical RL BID2 BID19 BID4 , this approach gives them the role of advisor: providing an aggregator with the local Q-values for all actions.

The advisors are said to have a focus: reward function, state space, learning technique, etc.

The MAd-RL approach allows therefore to tackle the RL task from different focuses.

When a person is consulted for an advice by a enquirer, he may answer egocentrically: as if he was in charge of next actions, agnostically: anticipating any future actions equally, or empathically: by considering the next actions of the enquirer.

The same approaches are modelled in the local advisors' planning methods.

Section 4 shows that the egocentric planning presents the severe theoretical shortcoming of inverting a max into a max in the global Bellman equation.

It leads to an overestimation of the values of states where the advisors disagree, and creates an attractor phenomenon, causing the system to remain static without any tie-breaking possibilities.

It is shown on a navigation task that attractors can be avoided by lowering the discount factor ?? under a given value.

The agnostic planning BID30 has the drawback to be inefficient in dangerous environments, because it gets easily afraid of the controller performing a bad sequence of actions.

Finally, we introduce our novel empathic planning and show that it converges to the global optimal Bellman equation when all advisors are training on the full state space.van BID29 demonstrate on a fruit collection task that a distributed architecture significantly speeds up learning and converges to a better solution than non distributed baselines.

Section 5.2 extends those results and empirically validates our theoretical analysis: the egocentric planning gets stuck in attractors with high ?? values; with low ?? values, it gets high scores but is also very unstable as soon as some noise is introduced; the agnostic planning fails at efficiently gathering the fruits near the ghosts; despite lack of convergence guarantees with partial information in advisors' state space, our novel empathic planning also achieves high scores while being robust to noise.

Task decomposition -Literature features numerous ways to distribute a single-agent RL problem over several specialised advisors: state space approximation/reduction BID0 , reward segmentation BID2 BID8 BID31 BID30 , algorithm diversification BID32 BID14 , algorithm randomization BID1 BID10 , sequencing of actions BID27 , or factorisation of actions BID15 .

In this paper, we mainly focus on reward segmentation and state space reduction but the findings are applicable to any family of advisors.

BID23 are the first to propose to merge Markov decision Processes through their value functions.

It makes the following strong assumptions: positive rewards, model-based RL, and local optimality is supposed to be known.

Finally, the algorithm simply accompanies a classical RL algorithm by pruning actions that are known to be suboptimal.

BID24 propose to use a local SARSA online learning algorithm for training the advisors, but they elude the fact that the online policy cannot be locally accurately estimated with partial state space, and that this endangers the convergence properties.

BID21 study more in depth the theoretical guaranties of convergence to optimality with the local Q-learning, and the local SARSA algorithms.

However, their work is limited in the fact that they do not allow the local advisors to be trained on local state space.

van BID30 relax this assumption at the expense of optimality guarantees and beat one of the hardest Atari games: Ms. Pac-Man, by decomposing the task into hundreds of subtasks that are trained in parallel.

MAd-RL can also be interpreted as a generalisation of Ensemble Learning BID5 for RL.

As such, BID25 use a boosting algorithm in a RL framework, but the boosting is performed upon policies, not RL algorithms.

In this sense, this article can be seen as a precursor to the policy reuse algorithm BID7 rather than a multi-advisor framework.

BID32 combine five online RL algorithms on several simple RL problems and show that some mixture models of the five experts performs generally better than any single one alone.

Each algorithm tackles the whole task.

Their algorithms were off-policy, on-policy, actor-critics, etc.

BID6 continue this effort in a very specific setting where actions are explicit and deterministic transitions.

We show in Section 4 that the planning method choice is critical and that some recommendations can be made in accordance to the task definition.

In BID11 , while all advisors are trained on different reward functions, these are potential based reward shaping variants of the same reward function.

They are therefore embedding the same goals.

As a consequence, it can be related to a bagging procedure.

The advisors recommendation are then aggregated under the HORDE architecture BID28 , with egocentric planning.

Two aggregator functions are tried out: majority voting and ranked voting.

BID14 follow a different approach in which, instead of boosting the weak advisors performances by aggregating their recommendation, they select the best advisor.

This approach is beneficial for staggered learning, or when one or several advisors may not find good policies, but not for variance reduction brought by the committee, and it does not apply to compositional RL.The UNREAL architecture BID12 improves the state-of-the art on Atari and Labyrinth domains by training their deep network on auxiliary tasks.

They do it in an unsupervised manner and do not consider each learner as a direct contributor to the main task.

The bootstrapped DQN architecture BID17 also exploits the idea of multiplying the Q-value estimations to favour deep exploration.

As a result, UNREAL and bootstrapped DQN do not allow to break down a task into smaller, tractable pieces.

As a summary, a large variety of papers are published on these subjects, differing by the way they factorise the task into subtasks.

Theoretical obstacles are identified in BID23 and BID21 , but their analysis does not go further than the non-optimality observation in the general case.

In this article, we accept the non-optimality of the approach, because it naturally comes from the simplification of the task brought by the decomposition and we analyse the pros and cons of each planning methods encountered in the literature.

But first, Section 3 lays the theoretical foundation for Multi-Advisor RL.Domains -Related works apply their distributed models to diverse domains: racing BID21 , scheduling BID21 , dialogue BID14 , and fruit collection BID23 BID24 BID29 b) .

Pac-Boy is yellow, the corridors are in black, the walls in grey, the fruits are the white dots and the ghosts are in red.

The fruit collection task being at the centre of attention, it is natural that we empirically validate our theoretical findings on this domain: the Pac-Boy game (see FIG0 , borrowed from van BID29 .

Pac-Boy navigates in a 11x11 maze with a total of 76 possible positions and 4 possible actions in each state: A = {N, W, S, E}, respectively for North, West, South and East.

Bumping into a wall simply causes the player not to move without penalty.

Since Pac-Boy always starts in the same position, there are 75 potential fruit positions.

The fruit distribution is randomised: at the start of each new episode, there is a 50% probability for each position to have a fruit.

A game lasts until the last fruit has been eaten, or after the 300 th time step.

During an episode, fruits remain fixed until they get eaten by Pac-Boy.

Two randomly-moving ghosts are preventing Pac-Boy from eating all the fruits.

The state of the game consists of the positions of PacBoy, fruits, and ghosts: 76 ?? 2 75 ?? 76 2 ??? 10 28 states.

Hence, no global representation system can be implemented without using function approximation.

Pac-Boy gets a +1 reward for every eaten fruit and a ???10 penalty when it is touched by a ghost.

Markov Decision Process -The Reinforcement Learning (RL) framework is formalised as a Markov Decision Process (MDP).

An MDP is a tuple X , A, P, R, ?? where X is the state space, A is the action space, P : X ?? A ??? X is the Markovian transition stochastic function, R : X ?? A ??? R is the immediate reward stochastic function, and ?? is the discount factor.

A trajectory x(t), a(t), x(t + 1), r(t) t??? 0,T ???1 is the projection into the MDP of the task episode.

The goal is to generate trajectories with high discounted cumulative reward, also called more succinctly return: DISPLAYFORM0 To do so, one needs to find a policy ?? : DISPLAYFORM1 Advisor Advisor ...

Figure 2: The MAd-RL architectureMAd-RL structure -This section defines the Multi-Advisor RL (MAd-RL) framework for solving a single-agent RL problem.

The n advisors are regarded as specialised, possibly weak, learners that are concerned with a sub part of the problem.

Then, an aggregator is responsible for merging the advisors' recommendations into a global policy.

The overall architecture is illustrated in Figure 2 .

At each time step, an advisor j sends to the aggregator its local Q-values for all actions in the current state.

Aggregating advisors' recommendations -In Figure 2 , the f function's role is to aggregate the advisors' recommendations into a policy.

More formally, the aggregator is defined with f : R n??|A| ??? A, which maps the received q j = Q j (x j , ??? ) values into an action of A. This article focuses on the analysis of the way the local Q j -functions are computed.

From the values q j , one can design a f function that implements any aggregator function encountered in the Ensemble methods literature BID5 : voting schemes BID9 , Boltzmann policy mixtures BID32 and of course linear value-function combinations BID25 BID21 BID30 .

For the further analysis, we restrict ourselves to the linear decomposition of the rewards: R(x, a) = j w j R j (x j , a), which implies the same decomposition of return if they share the same ??.

The advisor's state representation may be locally defined by ?? j : X ??? X j , and its local state is denoted by x j = ?? j (x) ??? X j .

We define the aggregator function f ?? (x) as being greedy over the Q j -functions aggregation Q ?? (x, a): DISPLAYFORM0 We recall hereunder the main theoretical result of BID30 : a theorem ensuring, under conditions, that the advisors' training eventually converges.

Note that by assigning a stationary behaviour to each of the advisors, the sequence of random variables X 0 , X 1 , X 2 , . . . , with X t ??? X is a Markov chain.

For later analysis, we assume the following.

Assumption 1.

All the advisors' environments are Markov: DISPLAYFORM1 Theorem 1 (van BID30 ).

Under Assumption 1 and given any fixed aggregator, global convergence occurs if all advisors use off-policy algorithms that converge in the single-agent setting.

Although Theorem 1 guarantees convergence, it does not guarantee the optimality of the converged solution.

Moreover, this fixed point only depends on each advisor model and on their planning methods (see Section 4), but not on the particular optimisation algorithms that are used by them.

This section present three planning methods at the advisor's level.

They differ in the policy they evaluate: egocentric planning evaluates the local greedy policy, agnostic planning evaluates the random policy, and empathic planning evaluates the aggregator's greedy policy.

The most common approach in the literature is to learn off-policy by bootstrapping on the locally greedy action: the advisor evaluates the local greedy policy.

This planning, referred to in this paper as egocentric, has already been employed in BID23 , BID21 , BID11 , and van BID29 .

Theorem 1 guarantees for each advisor j the convergence to the local optimal value function, denoted by Q ego j , which satisfies the Bellman optimality equation: DISPLAYFORM0 where the local immediate reward r j is sampled according to R j (x j , a), and the next local state x j is sampled according to P j (x j , a).

In the aggregator global view, we get: By construction, r = j w j r j , and therefore we get: DISPLAYFORM1 DISPLAYFORM2 Egocentric planning suffers from an inversion between the max and operators and, as a consequence, it overestimate the state-action values when the advisors disagree on the optimal action.

This flaw has critical consequences in practice: it creates attractor situations.

Before we define and study them formally, let us explain attractors with an illustrative example based on the simple MDP depicted in FIG1 .

In initial state x 0 , the system has three possible actions: stay put (action a 0 ), perform advisor 1's goal (action a 1 ), or perform advisor 2's goal (action a 2 ).

Once achieving a goal, the trajectory ends.

The Q-values for each action are easy to compute: DISPLAYFORM3 As a consequence, if ?? > r 1 /(r 1 + r 2 ) and ?? > r 2 /(r 1 + r 2 ), the local egocentric planning commands to execute action a 0 sine die.

This may have some apparent similarity with the Buridan's ass paradox BID33 BID20 : a donkey is equally thirsty and hungry and cannot decide to eat or to drink and dies of its inability to make a decision.

The determinism of judgement is identified as the source of the problem in antic philosophy.

Nevertheless, the egocentric sub-optimality does not come from actions that are equally good, nor from the determinism of the policy, since adding randomness to the system will not help.

Let us define more generally the concept of attractors.

Definition 1.

An attractor x is a state where the following strict inequality holds: DISPLAYFORM4 Theorem 2.

State x is attractor, if and only if the optimal egocentric policy is to stay in x if possible.

Note that there is no condition in Theorem 2 (proved in appendix, Section A) on the existence of actions allowing the system to be actually static.

Indeed, the system might be stuck in an attractor set, keep moving, but opt to never achieve its goals.

To understand how this may happen, just replace state x 0 in FIG1 with an attractor set of similar states: where action a 0 performs a random transition in the attractor set, and actions a 1 and a 2 respectively achieve tasks of advisors 1 and 2.

Also, it may happen that an attractor set is escapable by the lack of actions keeping the system in an attractor set.

For instance, in FIG1 , if action a 0 is not available, x 0 remains an attractor, but an unstable one.

Definition 2.

An advisor j is said to be progressive if the following condition is satisfied: DISPLAYFORM5 The intuition behind the progressive property is that no action is worse than losing one turn to do nothing.

In other words, only progress can be made towards this task, and therefore non-progressing actions are regarded by this advisor as the worst possible ones.

Theorem 3.

If all the advisors are progressive, there cannot be any attractor.

The condition stated in Theorem 3 (proved in appendix, Section A) is very restrictive.

Still, there exist some RL problems where Theorem 3 can be applied, such as resource scheduling where each advisor is responsible for the progression of a given task.

Note that a MAd-RL setting without any attractors does not guarantee optimality for the egocentric planning.

Most of RL problems do not fall into this category.

Theorem 3 neither applies to RL problems with states that terminate the trajectory while some goals are still incomplete, nor to navigation tasks: when the system goes into a direction that is opposite to some goal, it gets into a state that is worse than staying in the same position.

Navigation problem attractors -We consider the three-fruit attractor illustrated in FIG2 : moving towards a fruit, makes it closer to one of the fruits, but further from the two other fruits (diagonal moves are not allowed).

The expression each action Q-value is as follows: DISPLAYFORM6 As a result, the aggregator would opt to go South and hit the wall indefinitely.

More generally in a deterministic 1 task where each action a in a state x can be cancelled by a new action a -1 x , it can be shown that the condition on ?? is a function of the size of the action set A. Theorem 4 is proved in Section A of the appendix.

Theorem 4.

State x ??? X is guaranteed not to be an attractor if ???a ??? A, ???a -1x ??? A, such that P (P (x, a), a

The agnostic planning does not make any prior on future actions and therefore evaluates the random policy.

Once again, Theorem 1 guarantees the convergence of the local optimisation process to its local optimal value, denoted by Q agn j , which satisfies the following Bellman equation: DISPLAYFORM0 Local agnostic planning is equivalent to the global agnostic planning.

Additionally, as opposed to the egocentric case, r.h.s.

of the above equation does not suffer from max-inversion.

It then follows that no attractor are present in agnostic planning.

Nevertheless, acting greedily with respect to Q agn ?? (x, a) only guarantees to be better than a random policy and in general may be far from being optimal.

Still, agnostic planning has proved its usefulness on Ms. Pac-Man (van Seijen et al., 2017b).

A novel approach, inspired from the online algorithm found in BID24 ; BID21 is to locally predict the aggregator's policy.

In this method, referred to as empathic, the aggregator is in control, and the advisors are evaluating the current aggregator's greedy policy f with respect to their local focus.

More formally, the local Bellman equilibrium equation is the following one: Q ap j (x j , a) = E r j + ??Q ap j (x j , f ?? (x )) .

Theorem 5.

Assuming that all advisors are defined on the full state space, MAd-RL with empathic planning converges to the global optimal policy.

Proof.

DISPLAYFORM0 ?? is the unique solution to the global Bellman optimality equation, and therefore equals the optimal value function, quod erat demonstrandum.

However, most MAd-RL settings involve taking advantage of state space reduction to speed up learning, and in this case, there is no guarantee of convergence because function f ?? (x ) can only be approximated from the local state space scope.

As a result the local estimatef j (x ) is used instead of f ?? (x ) and the reconstruction of max a ???A Q ap ?? (x , a ) is not longer possible in the global Bellman equation: DISPLAYFORM1

For this experiment, we intend to show that the value function is easier to learn with the MAd-RL architecture.

We consider a fruit collection task where the agent has to navigate through a 5 ?? 5 grid and receives a +1 reward when visiting a fruit cell (5 are randomly placed at the beginning of each episode).

A deep neural network (DNN) is fitted to the ground-truth value function V ?? * ?? for various objective functions: TSP is the optimal number of turns to gather all the fruits, RL is the optimal return, and egocentric is the optimal MAd-RL return.

This learning problem is fully supervised on 1000 samples, allowing us to show how fast a DNN can capture V ?? * ?? while ignoring the burden of finding the optimal policy and estimating its value functions through TD-backups or value iteration.

To evaluate the DNN's performance, actions are selected greedily by moving the agent up, down, left, or right to the neighbouring grid cell of highest value.

Section B.1 of the appendix gives the details.

FIG6 displays the performance of the theoretical optimal policy for each objective function in dashed lines.

Here TSP and RL targets largely surpass MAd-RL one.

But FIG6 also displays the performances of the networks trained on the limited data of 1000 samples, for which the results are completely different.

The TSP objective target is the hardest to train on.

The RL objective target follows as the second hardest to train on.

The egocentric planning MAd-RL objective is easier to train on, even without any state space reduction, or even without any reward/return decomposition (summed version).

Additionally, if the target value is decomposed (vector version), the training is further accelerated.

Finally, we found that the MAd-RL performance tends to dramatically decrease when ?? gets close to 1, because of attractors' presence.

We consider this small experiment to show that the complexity of objective function is critical and that decomposing it in the fashion of MAd-RL may make it simpler and therefore easier to train, even without any state space reduction.

In this section, we empirically validate the findings of Section 4 in the Pac-Boy domain, presented in Section 2.

The MAd-RL settings are associating one advisor per potential fruit location.

The local state space consists in the agent position and the existence -or not-of the fruit.

Six different settings are compared: the two baselines linear Q-learning and DQN-clipped, and four MAd-RL settings: egocentric with ?? = 0.4, egocentric with ?? = 0.9, agnostic with ?? = 0.9, and empathic with ?? = 0.9.

The implementation and experimentation details are available in the appendix, at Section B.2.We provide links to 5 video files (click on the blue links) representing a trajectory generated at the 50 th epoch for various settings.

egocentric-?? = 0.4 adopts a near optimal policy coming close to the ghosts without taking any risk.

The fruit collection problem is similar to the travelling salesman problem, which is known to be NP-complete BID18 .

However, the suboptimal small-?? policy consisting of moving towards the closest fruits is in fact a near optimal one.

Regarding the ghost avoidance, egocentric with small ?? gets an advantage over other settings: the local optimisation guarantees a perfect control of the system near the ghosts.

The most interesting outcome is the presence of the attractor phenomenon in egocentric-?? = 0.9: Pac-Boy goes straight to the centre area of the grid and does not move until a ghost comes too close, which it still knows to avoid perfectly.

This is the empirical confirmation that the attractors, studied in Section 4.1, present a real practical issue.

empathic is almost as good as egocentric-?? = 0.4.

agnostic proves to be unable to reliably finish the last fruits because it is overwhelmed by the fear of the ghosts, even when they are still far away.

This feature of the agnostic planning led van BID30 to use a dynamic normalisation depending on the number of fruits left on the board.

Finally, we observe that DQN-clipped also struggles to eat the last fruits.

The quantitative analysis displayed in FIG6 confirms our qualitative video-based impressions.

egocentric-?? = 0.9 barely performs better than linear Q-learning, DQN-clipped is still far from the optimal performance, and gets hit by ghosts from time to time.

agnostic is closer but only rarely eats all the fruits.

Finally, egocentric-?? = 0.4 and empathic are near-optimal.

Only egocentric-?? = 0.4 trains a bit faster, and tends to finish the game 20 turns faster too (not shown).Results with noisy rewards -Using a very small ?? may distort the objective function and perhaps even more importantly the reward signal diminishes exponentially as a function of the distance to the number of turns network trained on vector egocentric objective network trained on summed egocentric objective network trained on RL objective network trained on TSP objective optimal policy given egocentric objective optimal policy given RL objective optimal policy given TSP objective (a) Value function training.

goal, which might have critical consequences in noisy environments, hence the following experiment: several levels of Gaussian centred white noise ?? ?? with standard deviation ?? ??? {0.01, 0.1} have been applied to the reward signal: at each turn, each advisor now receivesr j = r j + ?? ?? instead.

Since the noise is centred and white, the ground truth Q-functions remain the same, but their estimators obtained during sampling is corrupted by noise variance.

Empirical results displayed in FIG6 shows that the empathic planning performs better than the egocentric one even under noise with a 100 times larger variance.

Indeed, because of the noise, the fruit advisors are only able to consistently perceive the fruits that are in a radius dependent on ?? and ??.

The egocentric planning, incompatible with high ?? values, is therefore myopic and cannot perceive distant fruits.

The same kind of limitations are expected to be encountered for small ?? values when the local advisors rely on state approximations, and/or when the transitions are stochastic.

This also supports the superiority of the empathic planning in the general case.

This article presented MAd-RL, a common ground for the many recent and successful works decomposing a single-agent RL problem into simpler problems tackled by independent learners.

It focuses more specifically on the local planning performed by the advisors.

Three of them -two found in the literature and one novel -are discussed, analysed and empirically compared: egocentric, agnostic, and empathic.

The lessons to be learnt from the article are the following ones.

The egocentric planning has convergence guarantees but overestimates the values of states where the advisors disagree.

As a consequence, it suffers from attractors: states where the no-op action is preferred to actions making progress on a subset of subtasks.

Some domains, such as resource scheduling, are identified as attractor-free, and some other domains, such as navigation, are set conditions on ?? to guarantee the absence of attractor.

It is necessary to recall that an attractor-free setting means that the system will continue making progress towards goals as long as there are any opportunity to do so, not that the egocentric MAd-RL system will converge to the optimal solution.

The agnostic planning also has convergence guarantees, and the local agnostic planning is equivalent to the global agnostic planning.

However, it may converge to bad solutions.

For instance, in dangerous environments, it considers all actions equally likely, it favours staying away from situation where a random sequence of actions has a significant chance of ending bad: crossing a bridge would be avoided.

Still, the agnostic planning simplicity enables the use of general value functions BID28 BID30 .The empathic planning optimises the system according to the global Bellman optimality equation, but without any guarantee of convergence, if the advisor state space is smaller than the global state.

In our experiments, we never encountered a case where the convergence was not obtained, and on the Pac-Boy domain, it robustly learns a near optimal policy after only 10 epochs.

It can also be safely applied to Ensemble RL tasks where all learners are given the full state space.

Theorem 1 BID30 ).

Under Assumption 1 and given any fixed aggregator, global convergence occurs if all advisors use off-policy algorithms that converge in the single-agent setting.

Proof.

Each advisor can be seen as an independent learner training from trajectories controlled by an arbitrary behavioural policy.

If Assumption 1 holds, each advisor's environment is Markov and off-policy algorithms can be applied with convergence guarantees.

Theorem 2.

State x is attractor, if and only if the optimal egocentric policy is to stay in x if possible.

Proof.

The logical biconditional will be demonstrated by successively proving the two converse conditionals.

First, the sufficient condition: let us assume that state x is an attractor.

By Definition 1, if state x is an attractor, we have: DISPLAYFORM0 Let a 0 denote the potential action to stay in x, and consider the MDP augmented with a 0 in x. Then, we have : DISPLAYFORM1 which proves that a 0 , if possible, will be preferred to any other action a ??? A.Second, the reciprocal condition: let us assume that, in state x, action a 0 would be preferred by an optimal policy under the egocentric planning.

Then: DISPLAYFORM2 which proves that x is an attractor.

Theorem 3.

If all the advisors are progressive, there cannot be any attractor.

Proof.

Let sum Definition 2 over advisors: DISPLAYFORM3 which proves the theorem.

Theorem 4.

State x ??? X is guaranteed not to be an attractor if:??? ???a ??? A, ???a -1x ??? A, such that P (P (x, a), a -1 DISPLAYFORM4 Under review as a conference paper at ICLR 2018Proof.

Let us denote J x a as the set of advisors for which action a is optimal in state x. Q ego a (x) is defined as the sum of perceived value of performing a in state x by the advisors that would choose it: DISPLAYFORM5 Let a + be the action that maximises this Q ego a (x) function: DISPLAYFORM6 We now consider the left hand side of the inequality characterising the attractors in Definition 1: DISPLAYFORM7 Since R(x, a + ) ??? 0, and since the a maximising Q ego j (x j , a ) is at least as good as the cancelling action (a + )-1x , we can follow with: DISPLAYFORM8 By comparing this last result with the right hand side of Definition 1, the condition for x not being an attractor becomes: DISPLAYFORM9 It follows directly from the inequality Q Similarly to the Taxi Domain Dietterich (1998), we incorporate the location of the fruits into the state representation by using a 50 dimensional bit vector, where the first 25 entries are used for fruit positions, and the last 25 entries are used for the agent's position.

The DNN feeds this bit-vector as the input layer into two dense hidden layers with 100 and then 50 units.

The output is a single linear head representing the state-value, or a multiple in the case of the vector MAd-RL target.

In order to assess the value function complexity, we train for each discount factor setting a DNN of fixed size on 1000 random states with their ground truth values.

Each DNN is trained over 500 epochs using the Adam optimizer Kingma & Ba (2014) with default parameters.

Four different objective function targets are considered:??? The TSP objective function target is the natural objective function, as defined by the Travelling Salesman Problem: the number of turns to gather all the fruits: DISPLAYFORM0 where k is the number of fruits remaining in statex, where ?? k is the ensemble of all permutations of integers between 1 and k, where ?? is one of those permutations, where x 0 is the position of the agent in x, where x i for 1 ??? i ??? k is the position of fruit with index i, where d(x i , x j ) is the distance (||??|| 1 in our gridworld) between positions x i and x j .???

The RL objective function target is the objective function defined for an RL setting, which depends on the discount factor ??: DISPLAYFORM1 with the same notations as for TSP.??? The summed egocentric planning MAd-RL objective function target does not involve the search into the set of permutations and can be considered simpler to this extent: DISPLAYFORM2 with the same notations as for TSP.??? The vector egocentric planning MAd-RL objective function target is the same as the summed one, except that the target is now a vector, separated into as many channels as potential fruit position: x0,xi) if there is a fruit in x i , 0 otherwise.

DISPLAYFORM3

MAd-RL Setup -Each advisor is responsible for a specific source of reward (or penalty).

More precisely, we assign an advisor to each possible fruit location.

This advisor sees a +1 reward only if a fruit at its assigned position gets eaten.

Its state space consists of Pac-Boy's position, resulting in 76 states.

In addition, we assign an advisor to each ghost.

This advisor receives a -10 reward if Pac-Boy bumps into its assigned ghost.

Its state space consists of Pac-Boy's and ghost's positions, resulting in 76 2 states.

A fruit advisor is only active when there is a fruit at its assigned position.

Because there are on average 37.5 fruits, the average number of advisors running at the beginning of each episode is 39.5.

Each fruit advisor is set inactive when its fruit is eaten.

The learning was performed through Temporal Difference updates.

Due to the small state spaces for the advisors, we can use a tabular representation.

We train all learners in parallel with off-policy learning, with Bellman residuals computed as presented in Section 4 and a constant ?? = 0.1 parameter.

The aggregator function sums the Q-values for each action a ??? A: Q ?? (x, a) := j Q j (x j , a), and uses -greedy action selection with respect to these summed values.

Because ghost agents have exactly identical MDP, we also benefit from direct knowledge transfer by sharing their Q-tables.

One can notice that Assumption 1 holds in this setting and that, as a consequence, Theorem 1 applies for the egocentric and agnostic planning methods.

Theorem 4 determines sufficient conditions for not having any attractor in the MDP.

In the Pac-Boy domain, the cancelling action condition is satisfied for every x ??? X .

As for the ?? condition, it is not only sufficient but also necessary, since being surrounded by goals of equal value is an attractor if ?? > 1/3.

In practice, an attractor becomes stable only when there is an action enabling it to remain in the attraction set.

Thus, the condition for not being stuck in an attractor set can be relaxed to ?? ??? 1/(|A|???2).

Hence, the result of ?? > 1/2 in the example illustrated by FIG2 .Baselines -The first baseline is the standard DQN algorithm BID16 with reward clipping (referred to as DQN-clipped).

Its input is a 4-channel binary image with the following features: the walls, the ghosts, the fruits, or Pac-Boy.

The second baseline is a system that uses the exact same input features as the MAd-RL model.

Specifically, the state of each advisor of the MAd-RL model is encoded with a one-hot vector and all these vectors are concatenated, resulting in a sparse binary feature vector of size 17, 252.

This vector is used for linear function approximation with Q-learning.

We refer to this setting with linear Q-learning.

We also tried to train deep architectures from these features with no success.

Experimental setting -Time scale is divided into 50 epochs lasting 20,000 transitions each.

At the end of each epoch an evaluation phase is launched for 80 games.

The theoretical expected maximum score is 37.5 and the random policy average score is around -80.Explicit links to the videos (www.streamable.com website was used to ensure anonymity, if accepted the videos will be linked to a more sustainable website):??? egocentric-?? = 0.4: https://streamable.com/6tian??? egocentric-?? = 0.9: https://streamable.com/sgjkq??? empathic: https://streamable.com/h6gey??? agnostic: https://streamable.com/grswh??? DQN-clipped: https://streamable.com/emh6y

@highlight

We consider tackling a single-agent RL problem by distributing it to $n$ learners.
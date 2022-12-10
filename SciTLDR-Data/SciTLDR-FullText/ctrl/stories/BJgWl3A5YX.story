Reinforcement learning (RL) has proven to be a powerful paradigm for deriving complex behaviors from simple reward signals in a wide range of environments.

When applying RL to continuous control agents in simulated physics environments, the body is usually considered to be part of the environment.

However, during evolution the physical body of biological organisms and their controlling brains are co-evolved, thus exploring a much larger space of actuator/controller configurations.

Put differently, the intelligence does not reside only in the agent's mind, but also in the design of their body.

We propose a method for uncovering strong agents, consisting of a good combination of a body and policy, based on combining RL with an evolutionary procedure.

Given the resulting agent, we also propose an approach for identifying the body changes that contributed the most to the agent performance.

We use the Shapley value from cooperative game theory to find the fair contribution of individual components, taking into account synergies between components.

We evaluate our methods in an environment similar to the the recently proposed Robo-Sumo task, where agents in a 3D environment with simulated physics compete in tipping over their opponent or pushing them out of the arena.

Our results show that the proposed methods are indeed capable of generating strong agents, significantly outperforming baselines that focus on optimizing the agent policy alone.



A video is available at: www.youtube.com/watch?v=eei6Rgom3YY

Reinforcement Learning (RL) uses a simple reward signal to derive complex agent policies, with recent progress on representing the policy using deep neural networks leading to strong results in game playing BID29 Silver et al., 2016) , robotics BID22 BID23 and dialog systems BID24 .

Such algorithms were designed for stationary environments, but having multiple learning agents interact yields a non-stationary environment (Littman, 1994; BID2 .

Various approaches were proposed for continuous control, required for locomotion in an physics simulator environment BID16 BID31 BID0 .

Although very successful, such approaches consider the body of the agent to be fixed, simply a part of the environment.

However, during evolution the physical body of biological organisms is constantly changing; thus, the controlling brain and physical body are jointly optimized, exploring a larger space of actuator-controller configurations.

The interaction of evolution with learning by individual animals over their lifetime can result in superior performance (Simpson, 1953) .

Researchers refer to how individual learning can enhance evolution at the species level as the "Baldwin Effect" (Weber & Depew, 2003) , where learning guides evolution by smoothing the fitness landscape.

In learning agents, the physical shape of the body plays a double role.

First, a good body has the capability of effectively exerting many forces in the environment.

Second, a well-configured body is easier to learn to control, by making it simpler to identify good policies for exerting the forces.

Consider a physical task which requires exerting certain forces at the right time, such as locomotion.

Some bodies can exert the required forces, while others cannot.

Further, some bodies exert the required forces only under a small set of exactly correct policies, whereas others have a wide range of policies under which they exert the required forces (at least approximately).

In other words, some bodies have a wide "basin of attraction" where a learner can find a policy that exerts at least a part of the required forces; once discovering a policy in this wide basin, the learner can optimize the policy to exert the required forces.

This indicates that the intelligence of agents resides not only in their mind (the controller), but also in the design of their body.

Our contribution is proposing a method for uncovering strong agents, consisting of a good combination of a body and policy.

This stands in contrast to the traditional paradigm, which takes the body as a given (i.e. a fixed part of the environment), as shown in FIG0 .

Our technique combines RL with an evolutionary procedure.

We also show how to identify the body changes that contributed the most to agent performance, taking into account synergies between them.

We demonstrate our method in an environment similar to the Robo-Sumo task (AlShedivat et al., 2017) , where agents in a 3D environment with simulated physics compete in pushing the opponent out of the arena or tipping it over.

This environment is based on the MuJoCo physics simulator (Todorov et al., 2012) , allowing us to easily modify the agent's body.

Our results show that the proposed methods are indeed capable of generating superior agents, significantly outperforming baselines that focus on optimizing the agent policy alone.

Related Work Evolving virtual creatures (EVCs) work uses genetic algorithms to evolve the structure and controllers of virtual creatures in physically simulated environments, without learning (Sims, 1994) .

EVCs have a genetically defined morphology and control system that are co-evolved to perform locomotion tasks (Sims, 1994; BID7 BID4 , with some methods using a voxel-based "soft-body" BID4 BID17 BID28 .

Most such attempts have yielded relatively simple behaviors and morphologies BID7 BID3 .

One approach to enable continually increasing complex behavior is using a curriculum BID6 .

Researchers hypothesized that embodied cognition, where a controller expresses its behavior through a body, may cause morphological changes to have an immediate detrimental impact on a behavior BID3 .

For example, a mutation generating longer legs may harm performance with a controller optimized for shorter legs.

This results in pressure to converge on a body design early in evolution, to give the controller a stable platform to optimize.

This interdependence can be mitigated by giving the controller time to adapt to morphological changes, so bodies that are easier to learn to control would have an evolutionary advantage, and learning would smooth the fitness landscape; this may speed up body evolution, with the extent of learning required for new bodies decreasing over time (Simpson, 1953; Weber & Depew, 2003) .Scenarios where learning is used only in the evaluation phase of evolved agents are referred to as Baldwinian evolution (Weber & Depew, 2003) , where the results of learning are discarded when an offspring is generated.

This is in contrast to "Lamarkian evolution" (Whitley et al., 1994; BID20 , where the result of learning is passed on to offspring.

Typically the adaption stage uses a genetic algorithm operating to evolve the controller BID3 BID20 .

In contrast, we use an RL algorithm to learn to control an evolving body.

RL has achieved complex behaviours in continuous control tasks with fixed morphology BID16 BID31 BID0 , and has the potential to adapt to morphological changes.

Our experiments evaluate the potential of evolving the bodies of a population of learning agents.

We leverage Population Based Training BID18 (PBT), originally proposed to evolve parameters of the controller.

To our knowledge, this is the first attempt at evolving the body of continuously controlled RL agents in a physically simulated environment.

Preliminaries We apply multi-agent reinforcement learning in partially-observable Markov games (i.e. partially-observable stochastic games) BID34 Littman, 1994; BID13 .

In every state, agents take actions given partial observations of the true world state, and each obtains an individual reward.

Agents learn an appropriate behavior policy from past interactions.

In our case, given the physics simulator state, agents observe an egocentric view consisting of the positions and velocities of their and their opponent's bodies (end effectors and joints) and distances from the edges of the pitch.

Our agents have actuated hinges (one at the "knee" and one at the "hip" of every limb).

The full specification for our environment (observations and actions) are given in the Appendix, and are similar to other simulated physics locomotion tasks BID16 .

Every agent has its own experience in the environment, and independently learns a policy, attempting to maximize its long term γ-discounted utility, so learning is decentralized BID2 .Our analysis of the relative importance of the body changes uses cooperative game theory.

We view the set of body changes as a "team" of players, and quantify the impact of individual components, taking into account synergies between them.

Game theory studies players who can form teams, looking for fair ways of estimating the impact of individual players in a team.

A cooperative game consists of a set A of n players, and a characteristic function v : 2 A → R which maps any team C ⊆ A of players to a real value, showing the performance of the team as a whole.

In our case, A consists of all changes to body components resulting in the final body configuration.

The marginal contribution of a player i in a team C that includes it (i.e. i ∈ C) is the change in performance resulting from excluding i: DISPLAYFORM0 We define a similar concept for permutations.

Denote by π a permutation of the players (i.e. π : {1, 2, . . .

, n} → {1, 2, . . . , n} where π is a bijection), and by Π the set of all player permutations.

We refer to the players occurring before i in the permutation π as the predecessors of i in π, and denote by S π (i) the predecessors of i in π, i.e. S π (i) = {j|π(j) < π(i)}. The marginal contribution of a player i in a permutation is the change in performance between i's predecessors and including i, and the performance of i's predecessors alone: BID35 ) is considered a fair allocation of the overall reward achieved by a team to the individual players in a team, reflecting the contribution of each individual player to the team's success BID12 Straffin, 1988) .

It is the unique value exhibiting various fairness axioms BID9 BID11 , taking into account synergies between agents (see Section 8 in the Appendix for a detailed discussion and examples of how the Shapley value captures such synergies between body components).

The Shapley value is the marginal contribution of a player, averaged across all player permutations, given by the vector DISPLAYFORM1 DISPLAYFORM2

We consider agents who compete with one another in a physical environment, and propose a method for optimizing both the agent's policy and its physical characteristics.

We refer to the agent's policy as the controller, and to the configuration of its physical structure as the body.

Our method begins with an initial agent body and a random policy and repeatedly competes agents with each other, identifying beneficial changes to both the agent's policy and the agent's body.

Finally, the procedure outputs high performing agents, consisting of both a body configuration and a controller.

Our high level approach combines a reinforcement learning procedure that optimizes each agent's controller with an evolutionary procedure that optimizes the agent's body (as well as the learner's parameters).

We thus simultaneously improve the agents' bodies, while improving and fitting each agent's controller to its current body.

Specifically, we employ a variant of Population Based Training BID18 (PBT), which maintains a population of RL agents and leverages an evolutionary procedure to improve their controllers, except we apply evolution to not only the policy learner, but also to the physical agent body.

Further, given a final agent body, we decompose the overall agent performance to the contribution of each individual body change.

POEM maintains a population of agents and lets them participate in contests with each other.

It uses the data from the contests in two ways: first, it uses the experience from these episodes to improve the controller by applying RL; second, it analyzes the outcomes of the contests to rank agents by their performance, and uses this ranking to apply an evolutionary process to improve the agents' bodies (and controllers).

POEM retains two sub-populations of agents, a body-fixed population where only the agent policy is optimized, and a body-improving population, where the agent body as well as the controller are improved over time.

The individual agents, in both sub-populations, are continuous policy agents.

For the evolutionary procedure, POEM uses a variant of PBT which improves model parameters and learner hyper-parameters (in both body-fixed and body-improving sub-populations), and also evolves the agent bodies in the body-improving population.

We examine continuous control RL agents, based on Stochastic Value Gradients (SVG) BID15 and employing off-policy Retrace-correction BID30 .

SVG is a policy gradient algorithm that learns continuous control policies, allowing for stochastic control by modelling stochasticity in the Bellman equation as a deterministic function of external noise.

Our implementation augments SVG with an experience replay buffer for learning the action-value function with k-step returns, applying off-policy Retrace-corrections BID30 (several papers cover this in detail BID14 ).

POEM uses an evolutionary procedure jointly with policy learners to evolve agent bodies and learner parameters, adapting PBT.

In PBT, agents play against each other in multiple contests, and Elo ratings BID10 are used to measure agents' performance, and "evolve" them, with low-ranked agents copying the parameters of highly-ranked agents.

Both the episode traces fed to the policy learner and the agent performance ratings that drive the evolution depend on the tournaments played.

The original PBT procedure is designed for "monolithic" agents, but we maintain two subpopulations with an important asymmetry between them; the action space is the same for all agents, but the outcome of taking the same action depends on the body of the agent (agents in the body-fixed population are identical, but the body-improving population agents have different bodies, yielding different outcomes for the same action).

POEM ensures that agents from both sub-populations constantly encounter one another, by applying random match-making: each agent faces another agent chosen uniformly at random from the whole population in a contest yielding a single episode.

However, the evolution procedure differs between the two sub-populations; both use the evolution procedure to periodically copy policy parameters and copy and perturb learner hyperparameters, but only the body-improving agents also evolve the body parameters during evolution.

Evolution Agents in our framework face their peers in multiple contests, so each agent faces opponents that learn and change their policy, and also evolve and change their body.

This makes the environment an agent faces highly non-stationary.

Further, changes in the agent's own body also contribute to the non-stationary nature of the environment.

The optimal body configuration and controller thus depend on the configuration of the other agents in the population (i.e. their bodies and policies, which are the result of the learning and evolution procedure).

We apply PBT to optimize different parts of the agent configuration, including policy parameters, hyperparameters of the learner, and the shape of the agent's body.

We now provide a short description of the PBT procedure we use (full details are given in the Appendix).

The high-level pseudocode is given in Algorithm 1, and the subroutines are described below.

Fitness: PBT uses ratings that quantify agents' relative performance based on the outcomes of previous contests.

Following the original PBT work, we use the Elo rating system BID10 which was originally introduced to rate chess players (the specific update details of the Elo procedure are given in the Appendix).

Evolution eligibility: Agents are examined using evolution eligibility criteria, designed to avoid early conversion of the agent population.

Initially there is a warm-up period, during which only the RL learner is used and no evolution steps are performed.

Following the warm-up period, agents are only considered for evolution if they meet these criteria: a certain number of steps must have passed since they last became eligible for evolution, and a certain number of steps must have passed since their parameters were modified by the evolution.

Selection: Not every agent eligible for evolution immediately modifies its parameters: eligible agents are examined using a selection procedure to determine whether the agent would modify its parameters.

Each eligible agent i is compared to another agent j sampled uniformly at random from the sub-population, and the ratings are used to compute s i,j , the probability of agent i to win in a contest against j. If this probability (win-rate) is lower than a certain threshold, an agent undergoes inheritance and mutation.

Inheritance: PEOM uses an inheritance procedure to mod- ify i's configuration to be more similar to j's, affecting three types of parameters: neural network weights, learner hyper-parameters, and body configuration parameters.

Neural network parameters and body configuration parameters are set to the target j's configuration.

Each hyper-parameter is taken either from the evolving agent i or from the target j depending on a (possibly-biased) coinflip.

The inheritance procedure is given in Algorithm 2.

Mutation: After inheritance, parameters undergo mutation, which modifies each parameter p by a factor f p following a uniform distribution f p ∼ U(1 − m, 1 + m).

After mutation, body parameters are capped at their individual upper and lower bounds (see Algorithm 4 in the Appendix for full details).Algorithm 1 POEM PBT Procedure DISPLAYFORM0

We now describe our experiments for evaluating the performance of the POEM framework.

We examine several research questions.

First, does POEM allow obtaining high performing agents (in controller and body)?

Second, is the advantage achieved by the resulting agent due solely to their improved body, or does the process allow us to obtain superior controllers even for the original agent body?

Finally, which body changes are most influential in achieving an improved performance?Environment Our experiments involve contests between agents, conducted using the MuJoCo physics simulator (Todorov et al., 2012) .

We focus on the Robo-Sumo task BID0 , where ant shaped robots must tip their opponent over or force them out of the arena.

Agent Body We use a quadruped body, which we refer to as the "ant body", an example of which is shown in FIG1 .

The body is composed of a root sphere and 8 capsules (cylinders capped by hemispheres) connected via hinges (single DoF joints), each of which are actuated.

All the rigid bodies have unit density.

A schematic configuration is shown in FIG1 .

In our experiments, the morphology is represented as a directed graph-based genotype where the nodes represent physical component specifications and the edges describe relationships between them (Sims, 1994 ).

An agent's morphology is expressed by parsing its genotype.

Each node describes the shape of a 3D rigid body (sphere or capsule), and the limits of the hinge joint attaching it to its parent (see Figure 3) .

Edges contain parameters used to position, orient and Figure 3 .

Edges have a "reflection" parameter to facilitate body symmetry; when enabled, the body of the child node is created twice: once in its specified position and orientation, and a second time reflected across the parent's Z-Y plane.

No control systems are specified by the genotype, instead all the actuators are made available to the RL algoithm as its actions.

The genotype constructed for our ant consists of three nodes and three edges connected as shown in FIG1 .

The root node specifies the spherical torso, with two edges connected to an "upper leg" node, one for the upper segment of the rear legs, and one for the front legs.

The lower segments of the ant's legs are all specified by the same "lower leg" node.

The full physical structure of the body is determined by 25 parameters in these nodes and edges (detailed in the Appendix).Population Configuration Our experiments are based on two sub-populations, a body-fixed and body-improving subpopulations.

Each of these consists of n = 64 agents.

In the body-fixed subpopulation all agents have the same body configuration, but have different policy parameters and learner hyper-parameters, whereas the body-improving sub-population has agents with different bodies.

Both sub-populations are initialized with random policy parameters and the same hyperparameters.

The body-fixed agents are all initialized to the same standard body-configuration (as shown in FIG1 , with full details given in Appendix).

The body-improving agents are each initialized with a different body by sampling body configurations around the original body configuration as detailed in the Appendix.

Figure 4 shows example initial bodies for the body-improving population (more examples appear in Figure 7 in the Appendix).

Our experiment is based on data from k = 50 runs of the POEM method of Section 2.1, with two sub-populations (a body-fixed and a body-improving sub-population), each with n = 64 agents.

POEM matches agents for contests uniformly at random across the entire population, so the bodyfixed agents to adapt the controller so as to best match the body-improving agents, making them increasingly stronger opponents.

Note that finding a good controller for the body-improving population is challenging, as the controller must cope with having many different possible bodies it may control (i.e. it must be robust to changes in the physical body of the agent).

We examine agent performance, as reflected by agent Elo scores.

FIG4 shows agent Elo ratings over time, in one run, where agents of the body-improving sub-population outperform the body-fixed agents (body-fixed agents are shown in red, and body-improving agents in blue).

Both populations start with similar Elo scores, but even early in training there is a gap in favor of the body-improving agents.

To determine whether POEM results in a significant advantage over optimizing only the controller, we study outcomes in all k = 50 runs.

We run POEM for a fixed number of training steps, 10 training hours (equivalently, 2e9 training steps), and analyze agent performance.

At the evaluation time, each agent (in either sub-population) has its own Elo rating, reflecting its win-rate against others.

As our goal is to identify the strongest agents, we examine the highest Elo agent in each sub-population.

FIG4 shows a histogram of Elo scores on the run of FIG4 , at evaluation time, showing that in this run all body-improving agents outperform all body-fixed agents.

We examine the proportion of runs where the highest Elo agent is a body-improving agent (rather than a body-fixed one).

In over 70% of the runs, the top body-improving agent outperforms the top body-fixed agent.

A binomial test shows this to be significant at the p < 0.001 level.

When one can change physical traits of agents, this shows that POEM can find the configuration of strong agents (a body and matching controller), typically outperforming agents with the original body and a controller optimized for that body.

FIG1 shows an example evolved ant from the body-improving population.

On average the evolved body is 15% wider and 6% higher (thus 25% heavier), and has a lower center of gravity; the caps on parameters during evolution allow the body to evolve to be much heavier, so the advantage is not only due to mass.

FIG4 shows how some body parameters evolve over time within the population.

Initially the variance is high, but by 1.5e9 steps it is negligible.

This shows the population converges early in training, to a possibly sub-optimal body.

Using Body-Improving Controllers in the Original Body POEM uncovers good combinations of a body and controller.

One might conjecture that the advantage stems from the agent's modified body, rather than from the controller.

As the overall structure of the ant remains the same, with only sizes, locations and angles of joints modified, we can use any controller in any body.

Thus we can test the performance of the controller discovered for the evolved body in the original, unevolved body.

We compared the win-rate of the body-fixed population against that of the body-improving controllers fit into the unevolved body.

Controllers were taken after 10 hours of training.

The results show that in 64% of the runs, the controllers taken from the body-improving population outperform those of the body-fixed population, when used in the unevolved body (similar to recent observations in EVCs by BID21 ).

This shows POEM can find strong controllers even for the original body, and is thus useful even when we cannot modify the physical body of agents.

Section 3.1 shows we can find a set of changes to the original body, that allow significantly improving its performance (given an appropriate controller).

However, which of these changes had the most impact on agent's performance?

This question is not completely well-defined, as the performance of agent is not a simple sum of the "strengths" of individual body changes.

The different body components depend on each other, and may exhibit synergies between components.

For instance, changing the orientation of the leg may only be helpful when changing its length.

We thus view the set of body changes as a "team" of components, and attempt to fairly attribute the improvement in performance to each of the parts, taking into account synergies between components, using the Shapley value.

We thus define a cooperative game where "players" are the changes to body parts.

As we have 25 body configuration parameters, we obtain a cooperative game with 25 players.

2 2 Our approach is akin to using the Shapley value to measure the importance of features in predictive models BID5 BID8 .

Appendix Section 8 contains a discussion and a motivating example.

We define the value v(S) of a subset S of body changes as follows.

Given the original body b and the evolved body b and a set S of body parts, we define the body b(S) as the body where each body part p ∈ S takes the configuration as in the evolved body, and where each part p / ∈ S takes the configuration as in the unevolved body b. The body b(S) is a hybrid body with some parameters configured as in the original body and some as in the evolved body.

To evaluate the performance of a hybrid body we use the evolved controller discussed in Section 3.1.

This controller has been learned over many different bodies, and can thus likely handle a hybrid body well.

Given an (evolved) controller c and a fixed baseline agent d (consisting of a fixed body and a fixed policy), we define the value v(S) of a set S of body changes as the win probability of an agent with the body b(S) and controller (policy) c against the baseline agent d. v(S) defines a cooperative game over the body parts, so we can use the formula in Section 1 to compute the Shapley value of each body part.

For computational reasons, we settle for an approximation BID1 ) (see Appendix for details).

FIG6 shows Shapley values measuring relative importance of body changes (for the top body-improving agent from Section 3.1), showing that body changes are unequal in their contribution to agent performance.

The high impact body parameters are the orientation of the front and rear upper leg, whereas changing the body radius and scale parameters have a lower impact.

We conduct another experiment to confirm that high Shapley components indeed yield a bigger performance boost than low Shapley ones.

We rank body parameters by their Shapley value and use the ranking to incrementally apply evolved-body parameter values to an unevolved body-fixed body.

We do this twice; once descending through the ranking starting with the highest Shapley-valued parameters, and a second time in an ascending order.

This process generates 26 body variants, where the first variant has no evolved body parameters and the last has all 25.

Each body variant competes against a fixed baseline agent (with fixed body and policy) in 25, 000 matches to get the proportion of won matches, used as a performance measure.

FIG6 depicts the resulting agent performance.

The curves show that introducing the highest Shapley valued parameters first has a large impact on performance.

The figure also shows that the Shapley ranking also outperforms another baseline heuristic, which ranks parameters by the magnitude of their change from the unevolved body.

We proposed a framework for jointly optimizing agent body and policy, combining continuous control RL agents with an evolutionary procedure for modifying agent bodies.

Our analysis shows that this technique can achieve stronger agents than obtained by optimizing the controller alone.

We also used game theoretic solutions to identify the most influential body changes.

Several questions remain open.

First, can we augment our procedure to also modify the neural network architecture of the controller, similarly to recent neural architecture optimizers BID27 ?

Second, can we use similar game theoretic methods to guide the evolutionary process?

Finally, How can we ensure the diversity of agents' bodies so as to improve the final performance?

Darrell Whitley, V Scott Gordon, and Keith Mathias.

Lamarckian evolution, the baldwin effect and function optimization.

In International Conference on Parallel Problem Solving from Nature, pp.

5-15.

Springer, 1994.

We now describe the full details regarding the structure of the ant, and the genotype containing the parameters governing the shape of the body.

Section 3 in the main text describes the configuration of the ant body as a graph-based genotype consisting of three nodes and three edges (shown in FIG1 ), with a root node specifying the spherical torso and two edges specifying the rear and front sets of legs.

The parameters for the body-fixed ant nodes and edges are shown in TAB3 respectively.

In the body-improving ant all these parameters are mutable with the exception of a node's Shape and the edge's Reflection.

Our analysis is based on independent multi-agent RL, in a physics simulator environment.

At a high level, we use multiple independent learners in a partially-observable Markov game, often called partially-observable stochastic games BID13 .

In every state, agents take actions given partial observations of the true world state, and each agent obtains an individual reward.

Agents receive a shaping reward to encourage them to approach their opponent, +100 if they win and −100 if they lose.

Through their individual experiences interacting with one another in the environment, agents learn an appropriate behavior policy.

More formally, consider an N -player partially observable Markov game M BID34 Littman, 1994) defined on a finite state set S. An observation function O : S × {1, . . .

, N } → R d gives each agent's d-dimensional restricted view of the true state space.

On any state, the agents may apply an action from A 1 , . . .

, A N (one per agent).

Given the joint action a 1 , . . . , a N ∈ A 1 , . . . , A N the state changes, following a transition function T : S × A 1 × · · · × A N → ∆(S) (this is a stochastic transition, and we denote the set of discrete probability distributions over S as ∆(S)).

We use DISPLAYFORM0 } to denote the observation space of agent i. Every agent gets an individual reward r i : DISPLAYFORM1 Every agent has its own experience in the environment, and independently learns a policy N ) ).

Every agent attempts to maximize its long term γ-discounted utility: DISPLAYFORM2 DISPLAYFORM3

Given the above definitions, to fully describe the environment in which our agents interact, we must define the partial observations they receive of the true world state, and the actions they may take.

As discussed in Section 2 and Section 3, the physics simulator holds the true world state, but agents only receive partial observations in the form of an egocentric view.

The full list of observed variables include the 3D positions of each end effector of the body, the 3D positions of each joint, the velocities and acceleration of the joints, distances (on 3 axes) to the corners of the pitch.

The agents observe all of these variables for both their own body, and the relative ones of the opponents body.

The action space of the agents relates to the actuated hinges.

Each limb of the ant body has two hinges, one at the "hip" (attaching it to the spherical torso), and one at the "knee".

Each of these is a single degree of freedom (DoF) joint, responding to a continuous control signal.

The full action space is thus the Cartesian product of the allowed action for each of the hinges (8 hinges in total, with a single DoF each).

Our POEM framework uses population based training BID18 (PBT) to evolve agents.

Algorithm 1 in Section 2 presents a high level view of the procedure, applying 5 subprocedures: measuring fitness, checking whether an agent is eligible for evolution, selection, inheritance and mutation.

As discussed in Section 2, we use Elo ranking BID10 for fitness, based on the outcomes of previous contests.

The specific Elo computation we use is given in Algorithm 3.Algorithm 3 Iterative Elo rating update.

DISPLAYFORM0 r j ←

r j − K(s − s elo ) 9: end procedure Periodically evolutionary events take place, whereby selected agents are potentially updated with the parameters of the other agents.

Each evolutionary event consists of a pairwise comparison between agents within each sub-population.

Each pair consist of an eligible recipient and an eligible donor.

To be eligible both agents need to have processed 1 × 10 8 frames for learning since the beginning of training or the last time they received parameters (whichever is most recent).

Further, a recipient agent needs to have also processed 4 × 10 6 frames for learning since the last time it was involved in an evolutionary event.

POEM's pairwise-matching procedure is simple.

For each sub-population, recipient-donor pairs (i, j) are uniformly sampling from the eligible recipients and eligible donors within that subpopulation.

The Elo ratings are used to compute s i,j , the probability of agent i to win in a contest against j. If this probability (win-rate) is lower than a certain threshold t then the recipient agent i will be receive an update based on the donor j. We use a threshold of t = 45% (i.e. an agent i inherits only if its probability of winning against the target is 45% or less).When the win-rate of a recipient i against a donor j is lower than the threshold, we change i to be more similar to j by applying two sub-procedures: an inheritance procedure and a mutation procedure.

The inheritance procedure is given in Algorithm 2.

We have opted to simply copy all of the donor's body configuration parameters to the recipient although other possible variants can be considered, such as taking only some parameters at random, or modifying the parameters to be closer to those of the donor.

Following the inheritance procedure, we apply random mutations to the parameters.

The mutation procedure is given in Algorithm 4.

It multiplies each parameter by a factor sampled uniformly at random from the range [1 − m, 1 + m] (we use m = 0.01), but maintains caps for each of the parameters.

The caps avoid the body-improving morphology from diverging too far from the body-fixed morphology, and we impose upper and lower bounds on each mutable parameter at ±10% of the parameter's value in the body-fixed configuration.ple compute the Shapley value of this game using the formula in Section 1, to obtain the vector φ = (φ 1 , . . .

, φ n ), reflecting the fair contribution of each body change, taking into account the interdependence and synergies between components.

To motivate the use of the Shapley consider a simple example, where there are three possible changes that can be made to the body: a) increase the length of the leg, b) change the angle of the leg, and c) change the size of the torso.

Further suppose that the changes a and b in isolation each increase the agent win-rate against a baseline from 50% to 56% (an increase of 6%), while applying c in isolation increases the win-rate from 50% to 54% (an increase of 4%).

Based solely on this, one might claim that a and b are more impactful changes (as they increase the performance more, in isolation).

However, suppose that a and b are substitutes so that applying both changes a and b only increases the win-rate to 56% (i.e. once one of these changes has been applied, applying the other change does not further improve the win-rate).

In contrast, while applying c in isolation only increases performance by 4%, it is synergetic with a and b, so when combined with either a or b, it improves performance by 5%; for instance, applying both a and c (or both b and c) result in a win rate of 50% + 6% + 5% = 61%.

Finally, applying all three changes (a,b,c) still achieves a win-rate of 61%.

As a and b are substitutes, their fair contribution should be lower than one would assume based on applying changes in isolation.

Similarly, as c complements the other changes, it contribution should be higher than one would assume based on applying changes in isolation.

The Shapley value examines the average marginal contribution of components in all permutations, as given in the table below.

It would thus be 3% for a and b, and 4.67% for c (i.e. the fair impact of c is higher than of a or b, when taking synergies into account).

Table 3 : Shapley computation for hypothetical example Section 3.2 analyzes the contribution of individual body changes using the Shapley value.

It is based on computing the Shapley value similarly to the computation in Table 3 for the simple hypothetical example.

Such a direct computation simply averages the marginal contribution of each component across all permutations (as given in the formula for the Shapley value in Section 1).

Although this direct computation is straighforward, it is intractable in for two reasons.

First, our definition of the cooperative game in Section 3.2 uses the probability of a certain agent (with a hybrid body and the evolved controller) beating another agent (with the original body and a baseline controller).

However, given a set S of body changes, we do not know the win probability of the agent with body b(S) against the baseline agent.

Second, the direct formula for the Shapley value in Section 1 is a sum of r!

components (each being a difference between v(A), v(B) for some two subsets of body parts A and B) where r is the number of body changes.

As we have 25 body components, as opposed to the three components in the hypothetical example, this requires going over 25! permutations, which is clearly computationally intractable.

To overcome the above problems we settle for approximating the Shapley value in the above game, rather than computing it exactly.

We estimate v(S) by generating m episodes where agent b(S) competes against the baseline agent using our simulator, and use the proportion of these where b(S) wins as an estimate of its win-rate (in our experiments we use m = 1000 episodes).

We then compute the Shapley value using a simple approximation method BID1 , which samples component permutations, rather than iterating over all such permutations.

<|TLDR|>

@highlight

Evolving the shape of the body in RL controlled agents improves their performance (and help learning)

@highlight

PEOM algorithm that incorporates Shapley value to accelerate the evolution by identifying contribution of each body part
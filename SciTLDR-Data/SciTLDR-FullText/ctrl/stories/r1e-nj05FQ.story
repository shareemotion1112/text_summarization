Multi-agent cooperation is an important feature of the natural world.

Many tasks involve individual incentives that are misaligned with the common good, yet a wide range of organisms from bacteria to insects and humans are able to overcome their differences and collaborate.

Therefore, the emergence of cooperative behavior amongst self-interested individuals is an important question for the fields of multi-agent reinforcement learning (MARL) and evolutionary theory.

Here, we study a particular class of multi-agent problems called intertemporal social dilemmas (ISDs), where the conflict between the individual and the group is particularly sharp.

By combining MARL with appropriately structured natural selection, we demonstrate that individual inductive biases for cooperation can be learned in a model-free way.

To achieve this, we introduce an innovative modular architecture for deep reinforcement learning agents which supports multi-level selection.

We present results in two challenging environments, and interpret these in the context of cultural and ecological evolution.

Nature shows a substantial amount of cooperation at all scales, from microscopic interactions of genomes and bacteria to species-wide societies of insects and humans BID36 .

This is in spite of natural selection pushing for short-term individual selfish interests (Darwin, 1859) .

In its purest form, altruism can be favored by selection when cooperating individuals preferentially interact with other cooperators, thus realising the rewards of cooperation without being exploited by defectors BID19 BID31 BID9 BID48 BID12 ).

However, many other possibilities exist, including kin selection, reciprocity and group selection BID40 Úbeda & Duéñez-Guzmán, 2011; BID52 BID41 BID56 BID50 .Lately the emergence of cooperation among self-interested agents has become an important topic in multi-agent deep reinforcement learning (MARL).

and BID25 formalize the problem domain as an intertemporal social dilemma (ISD), which generalizes matrix game social dilemmas to Markov settings.

Social dilemmas are characterized by a trade-off between collective welfare and individual utility.

As predicted by evolutionary theory, self-interested reinforcement-learning agents are typically unable to achieve the collectively optimal outcome, converging instead to defecting strategies BID45 .

The goal is to find multi-agent training regimes in which individuals resolve social dilemmas, i.e., cooperation emerges.

Previous work has found several solutions, belonging to three broad categories: 1) opponent modelling BID13 BID31 , 2) long-term planning using perfect knowledge of the game's rules BID33 BID46 ) and 3) a specific intrinsic motivation function drawn from behavioral economics BID25 .

These hand-crafted approaches run at odds with more recent end-to-end model-free learning algorithms, which have been shown to have a greater ability to generalize (e.g. BID10 ).

We propose that evolution can be applied to remove the hand-crafting of intrinsic motivation, similar to other applications of evolution in deep learning.

Evolution has been used to optimize single-agent hyperparameters BID26 , implement black-box optimization BID55 , and to evolve neuroarchitectures BID38 BID51 , regularization BID3 , loss functions BID27 BID24 , behavioral diversity BID6 , and entire reward functions BID49 .

These principles tend to be driven by single-agent search and optimization or competitive multi-agent tasks.

Therefore there is no guarantee of success when applying them in the ISD setting.

More closely related to our domain are evolutionary simulations of predator-prey dynamics BID57 , which used enforced subpopulations to evolve populations of neurons which are sampled to form the hidden layer of a neural network.

To address the specific challenges of ISDs, the system we propose distinguishes between optimization processes that unfold over two distinct time-scales: (1) the fast time-scale of learning and (2) the slow time-scale of evolution (similar to BID23 .

In the former, individual agents repeatedly participate in an intertemporal social dilemma using a fixed intrinsic motivation.

In the latter, that motivation is itself subject to natural selection in a population.

We model this intrinsic motivation as an additional additive term in the reward of each agent BID5 .

We implement the intrinsic reward function as a two-layer fully-connected feed-forward neural network, whose weights define the genotype for evolution.

We propose that evolution can help mitigate this intertemporal dilemma by bridging between these two timescales via an intrinsic reward function.

Evolutionary theory predicts that evolving individual intrinsic reward weights across a population who interact uniformly at random does not lead to altruistic behavior BID0 .

Thus, to achieve our goal, we must structure the evolutionary dynamics BID40 .

We first implement a "Greenbeard" strategy BID9 BID28 in which agents choose interaction partners based on an honest, real-time signal of cooperativeness.

We term this process assortative matchmaking.

Although there is ecological evidence of assortative matchmaking BID30 , it cannot explain cooperation in all taxa BID15 BID22 BID14 .

Moreover it isn't a general method for multi-agent reinforcement learning, since honest signals of cooperativeness are not normally observable in the ISD models typically studied in deep reinforcement learning.

To address the limitations of the assortative matchmaking approach, we introduce an alternative modular training scheme loosely inspired by ideas from the theory of multi-level (group) selection BID56 BID22 , which we term shared reward network evolution.

Here, agents are composed of two neural network modules: a policy network and a reward network.

On the fast timescale of reinforcement learning, the policy network is trained using the modified rewards specified by the reward network.

On the slow timescale of evolution, the policy network and reward network modules evolve separately from one another.

In each episode every agent has a distinct policy network but the same reward network.

As before, the fitness for the policy network is the individual's reward.

In contrast, the fitness for the reward network is the collective return for the entire group of co-players.

In terms of multi-level selection theory, the policy networks are the lower level units of evolution and the reward networks are the higher level units.

Evolving the two modules separately in this manner prevents evolved reward networks from overfitting to specific policies.

This evolutionary paradigm not only resolves difficult ISDs without handcrafting but also points to a potential mechanism for the evolutionary origin of social inductive biases.

We varied and explored different combinations of parameters, namely: (1) environments {Harvest, Cleanup}, (2) reward network features {prospective, retrospective}, (3) matchmaking {random, assortative}, and (4) reward network evolution {individual, shared, none}. We describe these in the following sections.

In this paper, we consider Markov games (Littman, 1994) within a MARL setting.

Specifically we study intertemporal social dilemmas BID25 , defined as games in which individually selfish actions produce individual benefit on short timescales but have negative impacts on the group over a longer time horizon.

This conflict between the two timescales characterizes the intertemporal nature of these games.

The tension between individual and group-level rationality identifies them as social dilemmas (e.g. the famous Prisoner's Dilemma).

We consider two dilemmas, each implemented as a partially observable Markov game on a 2D grid (see FIG0 ).

In the Cleanup game, agents tried to collect apples (reward +1) that spawned in a field at a rate inversely related to the cleanliness of a geographically separate aquifer.

Over time, this aquifer filled up with waste, lowering the respawn rate of apples linearly, until a critical point past which no apples could spawn.

Episodes were initialized with no apples present and zero spawning, thus necessitating cleaning.

The dilemma occurred because in order for apples to spawn, agents must leave the apple field and clean, which conferred no reward.

However if all agents declined to clean (defect), then no rewards would be received by any.

In the Harvest game, again agents collected rewarding apples.

The apple spawn rate at a particular point on the map depended on the number of nearby apples, falling to zero once there were no apples in a certain radius.

There is a dilemma between the short-term individual temptation to harvest all the apples quickly and the consequential rapid depletion of apples, leading to a lower total yield for the group in the long-term.

For more details, see the Appendix.

In our model, there are three components to the reward that enter into agents' loss functions (1) total reward, which is used for the policy loss, (2) extrinsic reward, which is used for the extrinsic value function loss and (3) intrinsic reward, which is used for the intrinsic value function loss.

The total reward for player i is the sum of the extrinsic reward and an intrinsic reward as follows: DISPLAYFORM0 (The extrinsic reward r E i (s, a) is the environment reward obtained by player i when it takes action a i from state s i , sometimes also written with a time index t. The intrinsic reward u(f ) is an aggregate social preference across features f and is calculated according to the formula, DISPLAYFORM1 where σ is the ReLU activation function, and θ = {W, v, b} are the parameters of a 2-layer neural network with 2 hidden nodes.

These parameters are evolved based on fitness (see Section 2.3).

The elements of v = (v 1 , v 2 ) can be seen to approximately correspond to a linear combination of the coefficients related to advantagenous and disadvantagenous inequity aversion mentioned in BID25 , which were found via grid search in this previous work, but are here evolved.

The feature vector f i is a player-specific quantity that other agents can transform into intrinsic reward via their reward network.

Each agent has access to the same set of features, with the exception that its own feature is demarcated specially.

The features themselves are a function of recently received or expected future (extrinsic) reward for each agent.

In Markov games the rewards received by different players may not be aligned in time.

Thus, any model of social preferences should not be overly influenced by the precise temporal alignment of different players' rewards.

Intuitively, they ought to depend on comparing temporally averaged reward estimates between players, rather than instantaneous values.

Therefore, we considered two different ways of temporally aggregating the rewards.

Figure 2: (a) Agent A j adjusts policy π j (s, a|φ) using off-policy importance weighted actor-critic (V-Trace) BID10 by sampling from a queue with (possibly stale) trajectories recorded from 500 actors acting in parallel arenas.

(b) The architecture includes intrinsic and extrinsic value heads, a policy head, and evolution of the reward network.

The retrospective method derives intrinsic reward from whether an agent judges that other agents have been actually (extrinsically) rewarded in the recent past.

The prospective variant derives intrinsic reward from whether other agents are expecting to be (extrinsically) rewarded in the near future.2 For the retrospective variant, f ij = e t j , where the temporally decayed reward e t j for the agents j = 1, . . .

, N are updated at each timestep t according to DISPLAYFORM2 and η = 0.975.

The prospective variant uses the value estimates V est j for f ij and has a stop-gradient before the reward network module so that gradients don't flow back into other agents.

We used the same training framework as in BID27 , which performs distributed asynchronous training in multi-agent environments, including population-based training (PBT) BID26 .

We trained a population of 50 agents 3 with policies {π i }, from which we sampled 5 players in order to populate each of 500 arenas running in parallel.

Within each arena, an episode of the environment was played with the sampled agents, before resampling new ones.

Agents were sampled using one of two matchmaking processes (described in more detail below).

Episode trajectories lasted 1000 steps and were written to queues for learning, from which weights were updated using V-Trace (Figure 2(a) ).

More details are in the Appendix.

The set of weights evolved included learning rate, entropy cost weight, and reward network weights θ 4 .

The parameters of the policy network φ were inherited in a Lamarckian fashion as in BID26 .

Furthermore, we allowed agents to observe their last actions a i,t−1 , last intrinsic rewards (r E i,t−1 (s i , a i )), and last extrinsic rewards (u i,t−1 (f i )) as input to the LSTM in the agent's neural network.

The objective function was identical to that presented in BID10 and comprised three components: (1) the value function gradient, (2) policy gradient, and (3) entropy regularization, weighted according to hyperparameters baseline cost and entropy cost (see Figure 2(b) ).Evolution was based on a fitness measure calculated as a moving average of total episode return, which was a sum of apples collected minus penalties due to tagging, smoothed as follows: DISPLAYFORM0 where ν = 0.001 and R i j is the return obtained on episode i by agent j (or reward network j in the case of the shared reward network evolution (see Section 2.5 and Appendix for details).

Matches were determined according to two methods: (1) random matchmaking and (2) assortative matchmaking.

Random matchmaking simply selected uniformly at random from the pool of agents to populate the game, while cooperative matchmaking first ranked agents within the pool according to a metric of recent cooperativeness, and then grouped agents such that players of similar rank played with each other.

This ensured that highly cooperative agents played only with other cooperative agents, while defecting agents played only with other defectors.

For Cleanup, cooperativeness was calculated based on the amount of steps in the last episode the agent chose to clean.

For Harvest, it was calculated based on the difference between the the agent's return and the mean return of all players, so that having less return than average yielded a high cooperativeness ranking.

Cooperative metric-based matchmaking was only done with either individual reward networks or no reward networks FIG2 ).

We did not use cooperative metric-based matchmaking for our multi-level selection model, since these are theoretically separate approaches.

Building on previous work that evolved either the intrinsic reward BID27 or the entire loss function BID24 , we separately evolved the reward network within its own population, thereby allowing different modules of the agent to compete only with like components.

This allowed for independent exploration of hyperparameters via separate credit assignment of fitness and thus considerably more of the hyperparameter landscape could be explored compared with using only a single pool.

In addition, reward networks could be randomly assigned to any policy network, and so were forced to generalize to a wide range of policies.

In a given episode, 5 separate policy networks were paired with the same reward network, which we term a shared reward network.

In line with BID26 , the fitness determining the copying of policy network weights and evolution of optimization-related hyperparameters (entropy cost and learning rate) were based on individual agent return.

By contrast, the reward network parameters were evolved according to fitness based on total episode return across the group of co-players FIG2 ).This contribution is distinct from previous work which evolved intrinsic rewards (e.g. BID27 because FORMULA1 we evolve over social features rather than a remapping of environmental events, and (2) reward network evolution is motivated by dealing with the inherent tension in ISDs, rather than merely providing a denser reward signal.

In this sense it's closer to evolving a form of communication for social cooperation, rather than learning reward-shaping in a sparse-reward environment.

We allow for multiple agents to share the same components, and as we shall see, in a social setting, this winds up being critical.

Shared reward networks provide a biologically principled method that mixes between group fitness on a long timescale and individual reward on a short timescale.

This contrasts with hand-crafted means of aggregation, as in previous work BID4 BID35 .

As shown in FIG3 , PBT without using an intrinsic reward network performs poorly on both games, where it asymptotes to 0 total episode reward in Cleanup and 400 for Harvest (the number of apples gained if all agents collect as quickly as they can).

Figures 4(a) and (b) compare random and assortative matchmaking with PBT and reward networks using retrospective social features.

When using random matchmaking, individual reward network agents perform no better than PBT on Cleanup, and only moderately better at Harvest.

Hence there is little benefit to adding reward networks over social features if players have separate networks, evolved selfishly.

The assortative matchmaking experiments used either no reward network (u(f ) = 0) or individual reward networks.

Without a reward network, performance was the same as the PBT baseline.

With individual reward networks, performance was very high, indicating that both conditioning the internal rewards on social features and a preference for cooperative agents to play together were key to resolving the dilemma.

On the other hand, shared reward network agents perform as well as assortative matchmaking and the handcrafted inequity aversion intrinsic reward from BID25 , even using random matchmaking.

This implies that agents didn't necessarily need to have immediate access to honest signals of other agents' cooperativeness to resolve the dilemma; it was enough to simply have the same intrinsic reward function, evolved according to collective episode return.

Figures 4(c) and (d) compare the retrospective and prospective variants of reward network evolution.

The prospective variant, although better than PBT when using a shared reward network, generally results in worse performance and more instability.

This is likely because the prospective variant depends on agents learning good value estimates before the reward networks become useful, whereas the retrospective variant only depends on environmentally provided reward and thus does not suffer from this issue.

We next plot various social outcome metrics in order to better capture the complexities of agent behavior (see FIG4 for Harvest, see Appendix for Cleanup).

Sustainability measures the average time step on which agents received positive reward, averaged over the episode and over agents.

Figure 5(a) shows that having no reward network results in players collecting apples extremely quickly, compared with much more sustainable behavior with reward networks.

Equality is calculated as E(1 − G(R)), where G(R) is the Gini coefficient over individual returns.

FIG4 (b) demonstrates that having the prospective version of reward networks tends to lead to lower equality, while retrospective variant has very high equality.

Tagging measures the average number of times a player fined another player throughout the episode.

FIG4 (c) shows that there is a higher propensity for tagging when using either a prospective reward network or an individual reward network, compared to the retrospective shared reward network.

This explains the performance shown in FIG3 .Finally, we can directly examine the weights of the final retrospective shared reward networks which were best at resolving the ISDs.

Interestingly, the final weights evolved in the second layer suggest that resolving each game might require a different set of social preferences.

In Cleanup, one of the final layer weights v 2 evolved to be close to 0, whereas in Harvest, v 1 and v 2 evolved to be of large magnitude but opposite sign.

We can see a similar pattern with the biases b. We interpret this to mean that Cleanup required a less complex reward network: it was enough to simply find other agents' being rewarded as intrinsically rewarding.

In Harvest, however, a more complex reward function was perhaps needed in order to ensure that other agents were not over-exploiting the apples.

We found that the first layer weights W tended to take on arbitrary (but positive) values.

This is because of random matchmaking: co-players were randomly selected and thus there was little evolutionary pressure to specialize these weights.

Real environments don't provide scalar reward signals to learn from.

Instead, organisms have developed various internal drives based on either primary or secondary goals BID1 .

Here we examined intrinsic rewards based on features derived from other agents in the environment.

In accord with evolutionary theory BID0 BID40 , we found that naïvely implementing natural selection via genetic algorithms did not lead to the emergence of cooperation.

Furthermore, assortative matchmaking was sufficient to generate cooperative behavior in cases where honest signals were available.

Finally, we proposed a new multi-level evolutionary paradigm based on shared reward networks that achieves cooperation in more general situations.

Why does evolving intrinsic social preferences promote cooperation?

Firstly, evolution ameliorates the intertemporal choice problem by distilling the long timescale of collective fitness into the short timescale of individual reinforcement learning, thereby improving credit assignment between selfish acts and their temporally displaced negative group outcomes BID25 .

Secondly, it mitigates the social dilemma itself by allowing evolution to expose social signals that correlate with, for example, an agent's current level of selfishness.

Such information powers a range of mechanisms for achieving mutual cooperation like competitive altruism BID21 , other-regarding preferences BID7 , and inequity aversion BID11 .

In accord, laboratory experiments show that humans cooperate more readily when they can communicate BID43 BID29 .The shared reward network evolution model was inspired by multi-level selection; yet it does not correspond to the prototypical case of that theory since its lower level units of evolution (the policy networks) are constantly swapping which higher level unit (reward network) they are paired with.

Nevertheless, there are a variety of ways in which we see this form of modularity arise in nature.

For example, free-living microorganisms occasionally form multi-cellular structures to solve a higher order adaptive problem, like slime mold forming a spore-producing stalk for dispersal BID54 , and many prokaryotes can incorporate plasmids (modules) found in their environment or received from other individuals as functional parts of their genome, thereby achieving cooperation in social dilemmas BID17 BID37 .

Alternatively, in humans a reward network may represent a shared "cultural norm", with its fitness based on cultural information accumulated from the groups in which it holds sway.

In this way, the spread of norms can occur independently of the success of individual agents BID2 ).For future work, we suggest investigating alternative evolutionary mechanisms for the emergence of cooperation, such as kin selection BID16 and reciprocity BID52 .

It would be interesting to see whether these lead to different weights in a reward network, potentially hinting at the evolutionary origins of different social biases.

Along these lines, one might consider studying an emergent version of the assortative matchmaking model along the lines suggested by BID22 , adding further generality and power to our setup.

Finally, it would be fascinating to determine how an evolutionary approach can be combined with multi-agent communication to produce that most paradoxical of cooperative behaviors: cheap talk.

All episodes last 1000 steps, and the total size of the playable area is 25×18 for Cleanup and 36×16 for Harvest.

Games are partially observable in that agents can only observe via a 15×15 RGB window, centered on their current location.

The action space consists of moving left, right, up, and down, rotating left and right, and the ability to tag each other.

This action has a reward cost of 1 to use, and causes the player tagged to lose 50 reward points, thus allowing for the possibility of punishing free-riders BID42 BID18 .

The Cleanup game has an additional action for cleaning waste.

Training was done via joint optimization of network parameters via SGD and hyperparameters/reward network parameters via evolution in the standard PBT setup.

Gradient updates were applied for every trajectory up to a maximum length of 100 steps, using a batch size of 32.

Optimization was via RMSProp with epsilon=10 −5 , momentum=0, decay rate=0.99, and an RL discount factor of 0.99.

The baseline cost weight (see BID39 ) was fixed at 0.25, and the entropy cost was sampled from LogUniform(2 × 10 −4 ,0.01) and evolved throughout training using PBT.

The learning rates were all initially set to 4 × 10 −4 and then allowed to evolve.

PBT uses evolution (specifically genetic algorithms) to search over a space of hyperparameters rather than manually tuning or performing a random search, resulting in an adaptive schedule of hyperparameters and joint optimization with network parameters learned through gradient descent BID26 .There was a mutation rate of 0.1 when evolving hyperparameters, using either multiplicative perturbations of ±20% for entropy cost and learning rate, and additive perturbation of ±0.1 for reward network parameters.

We implemented a burn-in period for evolution of 4 × 10 6 agent steps, to allow network parameters and hyperparameters to be used in enough episodes for an accurate assessment of fitness before evolution.

<|TLDR|>

@highlight

We introduce a biologically-inspired modular evolutionary algorithm in which deep RL agents learn to cooperate in a difficult multi-agent social game, which could help to explain the evolution of altruism.
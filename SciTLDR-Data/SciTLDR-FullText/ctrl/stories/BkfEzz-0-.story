Existing multi-agent reinforcement learning (MARL) communication methods have relied on a trusted third party (TTP) to distribute reward to agents, leaving them inapplicable in peer-to-peer environments.

This paper proposes reward distribution using {\em Neuron as an Agent} (NaaA) in MARL without a TTP with two key ideas: (i) inter-agent reward distribution and (ii) auction theory.

Auction theory is introduced because inter-agent reward distribution is insufficient for optimization.

Agents in NaaA maximize their profits (the difference between reward and cost) and, as a theoretical result, the auction mechanism is shown to have agents autonomously evaluate counterfactual returns as the values of other agents.

NaaA enables representation trades in peer-to-peer environments, ultimately regarding unit in neural networks as agents.

Finally, numerical experiments (a single-agent environment from OpenAI Gym and a multi-agent environment from ViZDoom) confirm that NaaA framework optimization leads to better performance in reinforcement learning.

To the best of our knowledge, no existing literature discusses reward distributions in the configuration described above.

Because CommNet assumes an environment that distributes a uniform reward to all the agents, if the distributed reward is in limited supply (such as money), it causes the Tragedy of the Commons BID15 , where the reward of contributing agents will be reduced due to the participation of free riders.

Although there are several MARL methods for distributing rewards ac- BID0 BID23 BID6 BID7 .

They should suppose TTP to distribute the optimal reward to the agents.

(b) Inter-agent reward distribution model (our model).

Some agents receive reward from the environment directly, and redistribute to other agents.

The idea to determine the optimal reward without TTP is playing auction game among the agents.cording to agents' contribution such as QUICR BID0 and COMA BID23 , they suppose the existence of TTP and hence cannot be applied to the situation investigated here.

The proposed method, Neuron as an Agent (NaaA), extends CommNet to actualize reward distributions in MARL without TTP based on two key ideas: (i) inter-agent reward distribution and (ii) auction theory.

Auction theory was introduced because inter-agent reward distributions were insufficient for optimization.

Agents in NaaA maximize profit, the difference between their received rewards and the costs which they redistribute to other agents.

If the framework is naively optimized, a trivial solution is obtained where agents reduce their costs to zero to maximize profits.

Then, NaaA employs the auction theory in game design to prevent costs from dropping below their necessary level.

As a theoretical result, we show that agents autonomously evaluate the counterfactual return as values of other agents.

The counterfactual return is equal to the discounted cumulative sum of counterfactual reward BID0 distributed by QUICR and COMA.NaaA enables representation trades in peer-to-peer environments and, ultimately, regards neural network units as agents.

As NaaA is capable of regarding units as agents without losing generality, this setting was utilized in the current study.

The concept of the proposed method is illustrated in FIG0 .An environment extending ViZDoom BID11 , a POMDP environment, to MARL was used for the experiment.

Two agents, a cameraman sending information and a main player defeating enemies with a gun, were placed in the environment.

Results confirmed that the cameraman learned cooperative actions for sending information from dead angles (behind the main player) and outperformed CommNet in score.

Interestingly, NaaA can apply to single-and multi-agent settings, since it learns optimal topology between the units.

Adaptive DropConnect (ADC), which combines DropConnect (Wan et al., 2013) (randomly masking topology) with an adaptive algorithm (which has a higher probability of pruning connections with lower counterfactual returns) was proposed as a further application for NaaA. Experimental classification and reinforcement learning task results showed ADC outperformed DropConnect.

The remainder of this paper is organized as follows.

In the next section, we show the problem setting.

Then, we show proposed method with two key ideas: inter-agent reward distribution and auction theory in Section 3.

After related works are introduced in Section 4, the experimental results are shown in classification, single-agent RL and MARL in Section 5.

Finally, a conclusion ends the paper.

Suppose there is an N -agent system in an environment.

The goal of this paper was to maximize the discounted cumulative reward the system obtained from the environment.

This was calculated as: DISPLAYFORM0 where R ex t is a reward which the system obtains from the environment at t, and γ ∈ [0, 1] is the discount rate and T is the terminal time.

Reward distribution is distributing R t to all the agents under the following constraint.

DISPLAYFORM1 where R it is a reward which is distributed to i-th agent at time t. For instance, in robot soccer, the environment give a reward 1 when an agent shoot a ball to the goal.

Each agent should receive the reward along to their contribution.

In most of MARL communication methods, the policy of reward distribution is determined by a centralized agent.

For example, QUICR BID0 and COMA BID7 distribute R it according to counterfactal reward, difference of reward between an agent made an action and not.

The value of counterfactual reward is calculated by centralized agent, called trusted third party (TTP).In a peer-to-peer environment such as inter-industry and -country trade, they cannot place a TTP.

Hence, another framework required to actualize reward distribution without TTP.

The proposed method, Neuron as an Agent (NaaA), extends CommNet BID23 to actualize reward distributions in MARL without TTP based on two key ideas: (i) inter-agent reward distribution and (ii) auction theory.

As we show in Section 3.3, NaaA actualizes that we can regard even a unit as an agent.

Some agents receive rewards from the environment directly R ex it , and they distribute these to other agents as incentives for giving precise information.

Rewards are limited, so if an agent distributes ρ rewards, the reward of that agents is reduced by ρ to satisfy the constraint of Eq:(2).

For this reason, agents other than a specified agent of interest can be considered a secondary environment for the agent giving rewards of −ρ instead of an observation x. This secondary environment was termed the internal environment, whereas the original environment was called the external environment.

Similarly to CommNet BID23 , the communication protocol between agents was assumed to be a continuous quantity (such as a vector), the content of which could be trained by backpropagation.

A communication network among the agents is represented as a directed graph G = (V, E) between agents, where V = {v 1 , . . .

, v N } is a set of the agents and E ⊂ V 2 is a set of edges representing the connections between two agents.

If (v i , v j ) ∈ E, then connection v i → v j holds, indicating that v j observes the representation of v i .

Here, the representation of agent v i at time t was denoted as x it ∈ R. Additionally, the set of agents that agent i connects to was designated to be N out i = {j|(v i , v j ) ∈ E} and the set of agents that agent i is connected from was DISPLAYFORM0 The following assumptions were added to the v i characteristics: N1: (Selfishness) The utility each agent v i wants to maximize is its own return (cumulative discounted reward): DISPLAYFORM1 N2: (Conservation) The summation of internal rewards over all V equals 0.

Hence, the summation of rewards which V (receive both internal and external environment R it ) are equivalent to the reward R ex t , which the entire multi-agent system receives from the external environment: DISPLAYFORM2 representation signal x i before transferring this signal to the agent.

Simultaneously, ρ jit will be subtracted from the reward of v j .

N4: (NOOP) v i can select NOOP (no operation), for which the return is δ > 0, as an action.

In NOOP, the agent inputs and outputs nothing.

The social welfare function (total utility of the agents) G all is equivalent to the objective function G. That is, DISPLAYFORM3 From N2, G all = G holds.

From N3, the reward R it received by v i at t can be written as: DISPLAYFORM0 which can be divided into positive and negative terms, where the former is defined as revenue, and the latter as cost.

These are respectively denoted as DISPLAYFORM1 Here, R it represents profit, difference between revenue and cost.

The agent v i maximizes its cumulative discounted profit, G it , represented as: DISPLAYFORM2 G it could not be observed until the end of an episode (the final time).

Because predictions based on current values were needed to select optimal actions, G it was approximated with the value function DISPLAYFORM3 where s it is a observation for i-th agent at time t. Under these conditions, the following equation holds: DISPLAYFORM4 Thus, herein, we only consider maximization of revenue, the value function, and cost minimization.

The inequality R it > 0 (i.e., r it > c it ) indicates that the agent in question gave additional value to the obtained data.

The agent selected the NOOP action because DISPLAYFORM5

If we directly optimize Eq:(6), a trivial solution is obtained in which the internal rewards converge at 0, and all agents (excepting agents which directly receive reward from the external environment) select NOOP as their action.

This phenomenon occurs regardless of the network topology G, as no nodes are incentivized to send payments ρ ijt to other agents.

With this in mind, multi-agent systems must select actions with no information, achieving the equivalent of taking random actions.

For that reason, the total external reward R ex t shrinks markedly.

This phenomenon also known as social dilemma in MARL, which is caused from a problem that each agent does not evaluate other agents' value truthfully to maximize their own profit.

We are trying to solve this problem with auction theory in Section 3.2.

To make the agents to evaluate other agents' value truthfully, the proposed objective function borrows its idea from the digital goods auction theory BID10 .

In general, an auction theory is a part of mechanism design intended to unveil the true price of goods.

Digital goods auctions are one mechanism developed from auction theory, specifically targeting goods that may be copied without cost such as digital books and music.

Although several variations of digital goods auctions exist, an envy-free auction BID10 was used here because it required only a simple assumption: equivalent goods have a single simultaneous price.

In NaaA, this can be represented by the following assumption: DISPLAYFORM0 The assumption above indicates that ρ jit takes either 0 or a positive value, depending on i at an equal selected time t.

Therefore, the positive side was named v i 's price, and denoted as q it .The envy-free auction process is shown in the left section of FIG2 , displaying the negotiation process between one agent sending an representation (defined as a seller), and a group of agents buying the representation (defined as buyers).

First, a buyer places a bid with the agent at a bidding price b jit (1).

Next, the seller selects the optimal priceq it and allocates the representation (2).

Payment occurs if b ijt exceeds q jt .

In this case, ρ jit = H(b jit − q it )q it holds where H(·) is a step function.

For this transaction, the definition g jit = H(b jit − q it ) holds, and is named allocation.

After allocation, buyers perform payment: ρ jit = g jitqit (3).

The seller sends the representation x i only to the allocated buyers (4).

Buyers who do not receive the representation approximate DISPLAYFORM1 This negotiation is performed at each time step in reinforcement learning.

The sections below discuss the revenue, cost, and value functions based on Eq:(6).Revenue: The revenue of an agent is given as DISPLAYFORM0 where DISPLAYFORM1 g jit is the demand, the number of agents for which the bidding price b jit is greater than or equal to q it .

Because R ex i is independent of q it , the optimal priceq it maximizing r it is given as:q DISPLAYFORM2 The r it curve is shown on the right side of FIG2 .Cost: Cost is defined as an internal reward that one agent pays to other agents.

It is represented as: DISPLAYFORM3 where The effects of the value function were considered for both successful and unsuccessful v j purchasing cases.

The value function was approximated as a linear function g it : DISPLAYFORM4 DISPLAYFORM5 where o it is equivalent to the cumulative discount value of the counterfactual reward BID0 , it was named counterfactual return.

As V 0 it (a constant independent of g it ) is equal to the value function when v i takes an action without observing x 1 , . . .

, x N .The optimization problem is, therefore, presented below using a state-action value function for i-th where DISPLAYFORM6 DISPLAYFORM7 was taken because the asking priceq t was unknown for v i , except whenq it and g iit = 0.Then, to identify the bidding price that b it maximizes returns, the following theorem holds.

This proof is shown in the Appendix A.This implies agents should only consider their counterfactual returns!

When γ = 0 it is equivalent to a case without auction.

Hence, the bidding value is raised if each agent considers their long-time rewards.

Consequently, when the NaaA mechanism is used agents behave as if performing valuation for other agents, and declare values truthfully.

Under these conditions, the following corollary holds:Corollary 3.1.

The Nash equilibrium of an envy-free auction is DISPLAYFORM0 The remaining problem is how to predict o t .

Q-learning was used to predict o t in this paper as the same way as QUICR BID0 .

As o it represented the difference between two Qs, each Q was approximated.

The state was parameterized using the vector s t , which contained input and weight.

The ϵ-greedy policy with Q-learning typically supposed that discrete actions Thus the allocation g ijt was employed as an action rather than b it and q it .Algorithm The overall algorithm is shown in Algorithm 1.

One benefit of NaaA is that it can be used not only for MARL, but also for network training.

Typical neural network training algorithms such as RMSProp (Tieleman & Hinton, 2012) and Adam BID13 are based on sequential algorithms such as the stochastic gradient descent (SGD).

Therefore, the problem they solve can be interpreted as a problem of updating a state (i.e., weight) to a goal (the minimization of the expected likelihood).

Learning can be accelerated by applying NaaA to the optimizer.

In this paper, the application of NaaA to SGD was named Adaptive DropConnect (ADC), the finalization of which can be interpreted as a combination of DropConnect (Wan et al., 2013) and Adaptive DropOut BID2 .

In the subsequent section, ADC is introduced as a potential NaaA application.

Algorithm 1 NaaA: inter-agent reward distribution with envy-free auction 1: for t = 1 to T do 2:Compute a bidding price for every edge: DISPLAYFORM0 Compute an asking price for every node: DISPLAYFORM1 qd it (q).

for DISPLAYFORM0 Compute allocation: DISPLAYFORM1 Compute the price the agent should pay:

ρ jit ← g jitqit 7:end for 8:Make a payment: DISPLAYFORM2 Make a shipment: DISPLAYFORM3 for v i ∈ V do Compute a bidding price for every edge: DISPLAYFORM4 Compute an asking price for every node: DISPLAYFORM5 qd it (q).

for DISPLAYFORM0 Compute allocation: DISPLAYFORM1 end for

Sample a switching matrix U t from a Bernoulli distribution: DISPLAYFORM0 Sample the random mask M t from a Bernoulli distribution: DISPLAYFORM1 Generate the adaptive mask: DISPLAYFORM2 Compute h t for making a shipment: DISPLAYFORM3 Update W t and b t by backpropagation.

12: end for ADC uses NaaA for supervised optimization problems with multiple revisions.

In such problems, the first step is the presentation of an input state (such as an image) by the environment.

Agents are expected to update their parameters to maximize the rewards presented by a criterion calculator.

Criterion calculators gives batch-likelihoods to agents, representing rewards.

Each agent, a classifier, updates its weights to maximize the reward from the criterion calculator.

These weights are recorded as an internal state.

A heuristic utilizing the absolute value of weight |w ijt | (the technique used by Adaptive DropOut) was applied as the counterfactual return o ijt .

The absolute value of weights was used because it represented the updated amounts for which the magnitude of error of unit outputs was proportional to |w ijt |.This algorithm is presented as Algorithm 2.

Because the algorithm is quite simple, it can be easily implemented and, thus, applied to most general deep learning problems such as image recognition, sound recognition, and even deep reinforcement learning.

Existing multi-agent reinforcement learning (MARL) communication methods have relied on a trusted third party (TTP) to distribute reward to agents, leaving them inapplicable in peer-to-peer environments.

R/DIAL BID6 ) is a communication method for deep reinforcement learning, which train the optimal communication among the agent with Q-learning.

It focuses on that paradigm of centralized planning.

CommNet BID23 , which exploits the characteristics of a unit that is agnostic to the topology of other units, employs backpropagation to train multi-agent communication.

Instead of reward R(a t ) of an agent i for actions at t a t , QUICR-learning BID0 ) maximizes counterfactual reward R(a t ) − R(a t − a it ), the difference in the case of the agent i takes an action a it (a t ) and not (a t − a it ).

COMA BID7 ) also maximizes counterfactual rewards in an actor-critic setting.

CommNet, QUICR and COMA have a centralized environment for distributing rewards through a TTP.

In contrast, NaaA does not rely on a TTP, and hence, each agent calculates its reward.

While inter-agent reward distribution has not been considered in the context of communication, trading agents have been considered in other contexts.

Trading agent competition (TACs), competitions for trading agent design, have been held in various locations regarding topics such as smart grids BID12 , wholesale BID14 , and supply chains BID19 , yielding innumerable trading algorithms such as Tesauro's bidding algorithm BID26 and TacTex'13 (Urieli & Stone, 2014) .

Since several competitions employed an auction as optimal price determination mechanism (Wellman et al., 2001; BID22 , using auctions to determine optimal prices is now a natural approach.

Unfortunately, these existing methods cannot be applied to the present situation.

First, their agents did not communicate because the typical purpose of a TAC is to create market competition between agents in a zero-sum game.

Secondly, the traded goods are not digital goods but instead goods in limited supply, such as power and tariffs.

Hence, this is the first paper to introduce inter-agent reward distribution to MARL communications.

Auction theory is discussed in terms of mechanism design BID17 , also known as inverse game theory.

Second-price auctions (Vickrey, 1961) are auctions including a single product and several buyers.

In this paper, a digital goods auction BID10 was used as an auction with an infinite supply.

Several methods extend digital goods auction to address collusion, including the consensus estimate BID8 and random sample auction BID9 , which can be used to improve our method.

This paper is also related to DropConnect in terms of controlling connections between units.

Adaptive DropConnect (ADC), proposed in a later section of this paper as a further application, extends the DropConnect (Wan et al., 2013) regularization technique.

The finalized idea of ADC (which uses a skew probability correlated to the absolute value of weights rather than dropping each connection between units by a constant probability) is closer to Adaptive DropOut BID2 , although their derivation differs.

The adjective "adaptive" is added with respect to the method.

Neural network optimizing using RL was investigated by BID1 ; however, their methods used a recurrent neural network (RNN) and are therefore difficult to implement, whereas the proposed method is RNN-free and forms as a layer.

For these reasons, its implementation is simple and fast and it also has a wide area of applicability.

To confirm that NaaA works widely with machine learning tasks, we confirm our method of supervised learning tasks as well as reinforcement learning tasks.

As supervised learning tasks, we use typical machine learning tasks such as image classification using MNIST, CIFAR-10, and SVHN.As reinforcement tasks, we confirm single-and multi-agent environment.

The single-agent environment is from OpenAI Gym.

We confirm the result using a simple reinforcement task: CartPole.

In multi-agent, we use ViZDoom, a 3D environment for reinforcement learning.

For classification, three types of datasets were used: MNIST, CIFAR-10, and STL-10.

The given task was to predict the label of each image, and each dataset had a class number of 10.

The first dataset, MNIST, was a collection of black and white images of handwritten digits sized 28 28.

The training and test sets contained 60,000 and 10,000 example images, respectively.

The CIFAR-10 dataset images were colored and sized 32 32, and the assigned task was to predict what was shown in each picture.

This dataset contained 6,000 images per class (5,000 for training and 1,000 for testing).

The STL-10 dataset was used for image recognition, and had 1,300 images for each class (500 training, 800 testing).

Each image was sized 96 96; however, for the experiment, the images were resized to 48 48 because the greater resolution of this dataset (relative to the above datasets) required far more computing time and resources.

Two models were compared in this experiment: DropConnect and Adaptive DropConnect (the model proposed in this paper).

The baseline model was composed of two convolutional layers and two fully connected layers whose outputs are dropped out (we set the possibility as 0.5).

The labels of input data were predicted using log-softmaxed values from the last fully connected layer.

In the DropConnect and Adaptive DropConnect models, the first fully connected layer was replaced by a DropConnected and Adaptive DropConnected layer, respectively.

It should be noted that the DropConnect model corresponded to the proposed method when ε = 1.0, meaning agents did not perform their auctions but instead randomly masked their weights.

The models were trained over ten epochs using the MNIST datasets, and were then evaluated using the test data.

The CIFAR-10 and STL-10 epoch numbers were 20 and 40, respectively.

Experiments were repeated 20 times for each condition, and the averages and standard deviations of error rates were calculated.

Results are shown in TAB1 .

As expected, the Adaptive DropConnect model performed with a lower classification error rate than either the baseline or DropConnect models regardless of the given experimental datasets.

Next, the single-agent reinforcement learning task was set as the CartPole task from OpenAI Gym BID3 with visual inputs.

In this setting, the agent was required to balance a pole while moving a cart.

The images contained a large amount of non-useful information, making pixel pruning important.

The result in TAB1 demonstrates that our method improves the standard RL.

The proposed reward distribution method was confirmed to work as expected by a validation experiment using the multi-agent setting in ViZDoom BID11 , an emulator of Doom containing a map editor where additional agents complement the main player.

A main player in the ViZDoom environment aims to seek the enemy in the map and then defeat the enemy.

A defend the center (DtC)-based scenario, provided by ViZDoom platform, was used for this experiment.

Two players, a main player and a cameraman, were placed in the DtC, where they started in the center of a circular field and then attacked enemies that came from the surrounding wall.

Although the main player could attack the enemy with bullets, the cameraman had no way to attack, only scouting for the enemy.

The action space for the main player was the combination of { attack, turn left, turn right }, giving a total number of actions 2 3 = 8.

The cameraman had two possible actions: { turn left, turn right }.

Although the players could change direction, they could not move on the field.

Enemies died after receiving one attack (bullet) from the main player, and then player received a score of +1 for each successful attack.

The main player received 26 bullets by default at the beginning of each episode.

The main player died if they received attacks from the enemy to the extent that their health dropped to 0, and received a score of -1 for each death.

The cameraman did not die if attacked by an enemy.

Episodes terminated either when the maim player died or after 525 steps elapsed.

Figure 4: NaaA leads agents to enter a cooperative relationship.

First, the two agents face different directions, and the cameraman sells their information to the main player (1).

The main player (information buyer) starts to turn right to find the enemy.

The cameraman (information seller) starts to turn left to seek new information by finding the blind area of the main player (2 and 3).

After turning, the main player attacks the first, having already identified enemy (4 and 5).

Once the main player finds the enemy, he attacks and obtains the reward (6 and 7).

Both agents then return to watching the dead area of the other until the next enemy appears (8).

Three models, described below, were compared: the proposed method and two comparison targets.

Baseline: DQN without communication.

The main player learned standard DQN with the perspective that the player is viewing.

Because the cameraman did not learn, this player continued to move randomly.

Comm: DQN with communication, inspired by Commnet.

The main player learns DQN with two perspectives: theirs and that of the cameraman.

The communication vector is learned with a feedforward neural network.

NaaA: The proposed method.

The main player learned DQN with two perspectives: theirs and that of the cameraman.

Transmissions of rewards and communications were performed using the proposed method.

Training was performed over the course of 10 million steps.

FIG3 Left demonstrates the proposed NaaA model outperformed the other two methods.

Improvement was achieved by Adaptive DropConnect.

It was confirmed that the cameraman observed the enemy through an episode, which could be interpreted as the cameraman reporting enemy positions.

In addition to seeing the enemy, the cameraman observed the area behind the main player several times.

This enabled the cameraman to observe enemy attacks while taking a better relative position.

To further interpret this result, a heatmap visualization of revenue earned by the agent is presented in FIG3 Right.

The background picture is a screen from Doom, recorded at the moment when the CNN filter was most activated.

Figure 4 shows an example of learnt sequence of actions by our method.

This paper proposed a NaaA model to address communication in MARL without a TTP based on two key ideas: inter-agent reward distribution and auction theory.

Existing MARL communication methods have assumed the existence of a TTP, and hence could not be applied in peer-to-peer environments.

The inter-agent reward distribution, making agents redistribute the rewards they received from the internal/external environment, was reviewed first.

When an envy-free auction was introduced using auction theory, it was shown that agents would evaluate the counterfactual returns of other agents.

The experimental results demonstrated that NaaA outperformed a baseline method and a CommNet-based method.

Furthermore, a Q-learning based algorithm, termed Adaptive DropConnect, was proposed to dynamically optimize neural network topology with counterfactual return evaluation as a further application.

To evaluate this application, experiments were performed based on a single-agent platform, demonstrating that the proposed method produced improved experimental results relative to existing methods.

Future research may also be directed toward considering the connection between NaaA and neuroscience or neuroevolution.

Edeleman propounded the concept of neural Darwinism BID5 , in which group selection occurs in the brain.

Inter-agent rewards, which were assumed in this paper, correspond to NTFs and could be used as a fitness function in genetic algorithms for neuroevolution such as hyperparameter tuning.

As NaaA can be applied in peer-to-peer environments, the implementation of NaaA in blockchain BID24 is under consideration.

This implementation would extend the areas where deep reinforcement learning could be applied.

Bitcoin BID18 could be used for inter-agent reward distribution, and the auction mechanism could be implemented by smart contracts BID4 .

Using the NaaA reward design, it is hoped that the world may be united, allowing people to share their own representations on a global scale.

The optimization problem in Eq:11 is made of two terms except of the constant, and the only second term is depends on b. Hence, we consider to optimize the second term.

The optimal bidding priceŝ q t is given by the following equation.

DISPLAYFORM0 From independence, the equation is solved if we solve the following problem.

DISPLAYFORM1 Hence,b ijt can be derived as the solution which satisfies the following equation.

DISPLAYFORM2 For simplicity, we let q = q jt and o = o ij,t+1 .

Then, the following equation holds.

DISPLAYFORM3 DISPLAYFORM4

<|TLDR|>

@highlight

Neuron as an Agent (NaaA) enable us to train multi-agent communication without a trusted third party.
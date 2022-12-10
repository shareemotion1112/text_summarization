We explore the collaborative multi-agent setting where a team of deep reinforcement learning agents attempt to solve a shared task in partially observable environments.

In this scenario, learning an effective communication protocol is key.

We propose a communication protocol that allows for targeted communication, where agents learn \emph{what} messages to send and \emph{who} to send them to.

Additionally, we introduce a multi-stage communication approach where the agents co-ordinate via several rounds of communication before taking an action in the environment.

We evaluate our approach on several cooperative multi-agent tasks, of varying difficulties with varying number of agents, in a variety of environments ranging from 2D grid layouts of shapes and simulated traffic junctions to complex 3D indoor environments.

We demonstrate the benefits of targeted as well as multi-stage communication.

Moreover, we show that the targeted communication strategies learned by the agents are quite interpretable and intuitive.

Effective communication is a key ability for collaborative multi-agents systems.

Indeed, intelligent agents (humans or artificial) in real-world scenarios can significantly benefit from exchanging information that enables them to coordinate, strategize, and utilize their combined sensory experiences to act in the physical world.

The ability to communicate has wide-ranging applications for artificial agents -from multi-player gameplay in simulated games (e.g. DoTA, Quake, StarCraft) or physical worlds (e.g. robot soccer), to networks of self-driving cars communicating with each other to achieve safe and swift transport, to teams of robots on search-and-rescue missions deployed in hostile and fast-evolving environments.

A salient property of human communication is the ability to hold targeted interactions.

Rather than the 'one-size-fits-all' approach of broadcasting messages to all participating agents, as has been previously explored BID19 BID4 , it can be useful to direct certain messages to specific recipients.

This enables a more flexible collaboration strategy in complex environments.

For example, within a team of search-and-rescue robots with a diverse set of roles and goals, a message for a fire-fighter ("smoke is coming from the kitchen") is largely meaningless for a bomb-defuser.

In this work we develop a collaborative multi-agent deep reinforcement learning approach that supports targeted communication.

Crucially, each individual agent actively selects which other agents to send messages to.

This targeted communication behavior is operationalized via a simple signaturebased soft attention mechanism: along with the message, the sender broadcasts a key which encodes properties of agents the message is intended for, and is used by receivers to gauge the relevance of the message.

This communication mechanism is learned implicitly, without any attention supervision, as a result of end-to-end training using a downstream task-specific team reward.

The inductive bias provided by soft attention in the communication architecture is sufficient to enable agents to 1) communicate agent-goal-specific messages (e.g. guide fire-fighter towards fire, bomb-defuser towards bomb, etc.), 2) be adaptive to variable team sizes (e.g. the size of the local neighborhood a self-driving car can communicate with changes as it moves), and 3) be interpretable through predicted attention probabilities that allow for inspection of which agent is communicating what message and to whom.

Multi-agent systems fall at the intersection of game theory, distributed systems, and Artificial Intelligence in general BID18 , and thus have a rich and diverse literature.

Our work builds on and is related to prior work in deep multi-agent reinforcement learning, the centralized training and decentralized execution paradigm, and emergent communication protocols.

Multi-Agent Reinforcement Learning (MARL).

Within MARL (see BID1 for a survey), our work is related to recent efforts on using recurrent neural networks to approximate agent policies BID6 , algorithms stabilizing multi-agent training BID13 BID5 , and tasks in novel application domains such as coordination and navigation in 3D simulated environments BID17 BID16 BID8 .Centralized Training & Decentralized Execution.

Both BID19 and Hoshen (2017) adopt a fully centralized framework at both training and test time -a central controller processes local observations from all agents and outputs a probability distribution over joint actions.

In this setting, any controller (e.g. a fully-connected network) can be viewed as implicitly encoding communication.

BID19 present an efficient architecture to learn a centralized controller invariant to agent permutations -by sharing weights and averaging as in BID23 .

Meanwhile Hoshen (2017) proposes to replace averaging by an attentional mechanism to allow targeted interactions between agents.

While closely related to our communication architecture, his work only considers fully supervised one-next-step prediction tasks, while we tackle the full reinforcement learning problem with tasks requiring planning over long time horizons.

Moreover, a centralized controller quickly becomes intractable in real-world tasks with many agents and high-dimensional observation spaces (e.g. navigation in House3D BID22 ).

To address these weaknesses, we adopt the framework of centralized learning but decentralized execution (following BID4 ; BID13 BID19 No No Yes Yes (REINFORCE) VAIN (Hoshen, 2017) No Yes Yes No (Supervised) ATOC BID9 Yes sonable trade-off between allowing agents to globally coordinate while retaining tractability (since the communicated messages are much lower-dimensional than the observation space).Emergent Communication Protocols.

Our work is also related to recent work on learning communication protocols in a completely end-to-end manner with reinforcement learning -from perceptual input (e.g. pixels) to communication symbols (discrete or continuous) to actions (e.g. navigating in an environment).

While BID4 BID10 BID3 BID14 BID12 constrain agents to communicate with discrete symbols with the explicit goal to study emergence of language, our work operates in the paradigm of learning a continuous communication protocol in order to solve a downstream task BID19 Hoshen, 2017; BID9 .

While BID9 ) also operate in a decentralized execution setting and use an attentional communication mechanism, their setup is significantly different from ours as they use attention to decide when to communicate, not who to communicate with ('who' depends on a hand-tuned neighborhood parameter in their work).

TAB2 summarizes the main axes of comparison between our work and previous efforts in this exciting space.

Decentralized Partially Observable Markov Decision Processes (Dec-POMDPs).

A Dec-POMDP is a cooperative multi-agent extension of a partially observable Markov decision process BID15 ).

For N agents, it is defined by a set of states S describing possible configurations of all agents, a global reward function R, a transition probability function T , and for each agent i P 1, ..., N a set of allowed actions A i , a set of possible observations Ω i and an observation function O i .

Operationally, at each time step every agent picks an action a i based on its local observation ω i following its own stochastic policy π θi pa i |ω i q. The system randomly transitions to the next state s 1 given the current state and joint action T ps 1 |s, a 1 , ..., a N q. The agent team receives a global reward r " Rps, a 1 , ..., a N q while each agent receives a local observation of the new state O i pω i |s 1 q. Agents aim to maximize the total expected return J " ř T t"0 γ t r t where γ is a discount factor and T is the episode time horizon.

Actor-Critic Algorithms.

Policy gradient methods directly adjust the parameters θ of the policy in order to maximize the objective Jpθq " E s"pπ,a"π θ psq rRps, aqs by taking steps in the direction of ∇Jpθq.

We can write the gradient with respect to the policy parameters as DISPLAYFORM0 where Q π ps, aq is called the action-value, it is the expected remaining discounted reward if we take action a in state s and follow policy π thereafter.

Actor-Critic algorithms learn an approximation of the unknown true action-value functionQps, aq by e.g. temporal-difference learning BID20 .

ThisQps, aq is called the Critic while the policy π θ is called the Actor.

Multi-Agent Actor-Critic.

BID13 propose a multi-agent Actor-Critic algorithm adapted to centralized learning and decentralized execution.

Each agent learns its own individual policy π θi pa i |ω i q conditioned on local observation ω i , using a centralized Critic which estimates the joint action-valueQps, a 1 , ..., a N q.

We now describe our multi-agent communication architecture in detail.

Recall that we have N agents with policies tπ 1 , ..., π N u, respectively parameterized by tθ 1 , ..., θ N u, jointly performing a cooperative task.

At every timestep t, the ith agent for all i P t1, ..., N u sees a local observation ω t i , and must select a discrete environment action a t i " π θi and a continuous communication message m t i , received by other agents at the next timestep, in order to maximize global reward r t " R. Since no agent has access to the underlying state of the environment s t , there is incentive in communicating with each other and being mutually helpful to do better as a team.

Policies and Decentralized Execution.

Each agent is essentially modeled as a Dec-POMDP augmented with communication.

Each agent's policy π θi is implemented as a 1-layer Gated Recurrent Unit BID2 .

At every timestep, the local observation ω t i and a vector c t i aggregating messages sent by all agents at the previous timestep (described in more detail below) are used to update the hidden state h t i of the GRU, which encodes the entire message-action-observation history up to time t. From this internal state representation, the agent's policy π θi pa t i | h t i q predicts a categorical distribution over the space of actions, and another output head produces an outgoing message vector m t i .

Note that for all our experiments, agents are symmetric and policies are instantiated from the same set of shared parameters; i.e. θ 1 " ... " θ N .

This considerably speeds up learning.

Centralized Critic.

Following prior work BID13 BID5 , we operate under the centralized learning and decentralized execution paradigm wherein during training, a centralized critic guides the optimization of individual agent policies.

The centralized Critic takes as input predicted actions ta t 1 , ..., a t N u and internal state representations th t 1 , ..., h t N u from all agents to estimate the joint action-valueQ t at every timestep.

The centralized Critic is learned by temporal difference BID20 and the gradient of the expected return Jpθ i q " ErRs with respect to policy parameters is approximated by: DISPLAYFORM0 Note that compared to an individual criticQ i ph t i , a t i q for each agent, having a centralized critic leads to considerably lower variance in policy gradient estimates since it takes into account actions from all agents.

At test time, the critic is not needed anymore and policy execution is fully decentralized.

At the receiving end, each agent (indexed by j) predicts a query vector q DISPLAYFORM1 and uses it to compute a dot product with signatures of all N messages.

This is scaled by 1{ ?

d k followed by a softmax to obtain attention weight α ji for each message value vector: DISPLAYFORM2 Note that equation 2 also includes α ii corresponding to the ability to self-attend BID21 , which we empirically found to improve performance, especially in situations when an agent has found the goal in a coordinated navigation task and all it is required to do is stay at the goal, so others benefit from attending to this agent's message but return communication is not needed.

and internal state h t j are first used to predict the next internal state h 1 t j taking into account a first round of communication: DISPLAYFORM0 Next, h 1 t j is used to predict signature, query, value followed by repeating Eqns 1-4 for multiple rounds until we get a final aggregated message vector c t`1 j to be used as input at the next timestep.

We evaluate our targeted multi-agent communication architecture on a variety of tasks and environments.

All our models were trained with a batched synchronous version of the multi-agent ActorCritic described above, using RMSProp with a learning rate of 7ˆ10´4 and α " 0.99, batch size 16, discount factor γ " 0.99 and entropy regularization coefficient 0.01 for agent policies.

All our agent policies are instantiated from the same set of shared parameters; i.e. θ 1 " ... " θ N .

Each agent's GRU hidden state is 128-d, message signature/query is 16-d, and message value is 32-d (unless specified otherwise).

All results are averaged over 5 independent runs with different seeds.

The SHAPES dataset was introduced by BID0 1 , and originally created for testing compositional visual reasoning for the task of visual question answering.

It consists of synthetic images of 2D colored shapes arranged in a grid (3ˆ3 cells in the original dataset) along with corresponding question-answer pairs.

There are 3 shapes (circle, square, triangle), 3 colors (red, green, blue), and 2 sizes (small, big) in total (see FIG1 .We convert each image from SHAPES into an active environment where agents can now be spawned at different regions of the image, observe a 5ˆ5 local patch around them and their coordinates, and take actions to move around -tup, down, left, right, stayu.

Each agent is tasked with navigating to a specified goal state in the environment -t'red', 'blue square', 'small green circle', etc.

u -and the reward for each agent at every timestep is based on team performance i.e. r t " # agents on goal # agents .

1 github.com/jacobandreas/nmn2/tree/shapes (a) 4 agents have to find rred, red, green, blues respectively.

t " 1: inital spawn locations; t " 2: 4 was on red at t " 1 so 1 and 2 attend to messages from 4 since they have to find red.

3 has found its goal (green) and is self-attending; t " 6: 4 attends to messages from 2 as 2 is on 4's target -blue; t " 8: 1 finds red, so 1 and 2 shift attention to 1; t " 21: all agents are at their respective goal locations and primarily self-attending.(b) 8 agents have to find red on a large 100ˆ100 environment.

t " 7: Agent 2 finds red and signals all other agents; t " 7 to t " 150: All agents make their way to 2's location and eventually converge around red.

Having a symmetric, team-based reward incentivizes agents to cooperate with each other in finding each agent's goal.

For example, as shown in FIG1 , if agent 2's goal is to find red and agent 4's goal is to find blue, it is in agent 4's interest to let agent 2 know if it passes by red (t " 2) during its exploration / quest for blue and vice versa (t " 6).

SHAPES serves as a flexible testbed for carefully controlling and analyzing the effect of changing the size of the environment, no. of agents, goal configurations, etc.

How does targeting work in the communication learnt by TarMAC?

Recall that each agent predicts a signature and value vector as part of the message it sends, and a query vector to attend to incoming messages.

The communication is targeted because the attention probabilities are a function of both the sender's signature and receiver's query vectors.

So it is not just the receiver deciding how much of each message to listen to.

The sender also sends out signatures that affects how much of each message is sent to each receiver.

The sender's signature could encode parts of its observation most relevant to other agents' goals (for example, it would be futile to convey coordinates in the signature), and the message value could contain the agent's own location.

For example, in FIG1 , at t " 6, we see that when agent 2 passes by blue, agent 4 starts attending to agent 2.

Here, agent 2's signature encodes the color it observes (which is blue), and agent 4's query encodes its goal (which is also blue) leading to high attention probability.

Agent 2's message value encodes coordinates agent 4 has to navigate to, as can be seen at t " 21 when agent 4 reaches there.

Environment and Task.

The simulated traffic junction environments from BID19 consist of cars moving along pre-assigned, potentially intersecting routes on one or more road junctions.

The total number of cars is fixed at N max and at every timestep, new cars get added to the environment with probability p arrive .

Once a car completes its route, it becomes available to be sampled and added back to the environment with a different route assignment.

Each car has a limited visibility of a 3ˆ3 region around it, but is free to communicate with all other cars.

The action space for each car at every timestep is gas and brake, and the reward consists of a linear time penalty´0.01τ , where τ is the number of timesteps since car has been active, and a collision penalty r collision "´10.

Quantitative Results.

We compare our approach with CommNets BID19 on the easy and hard difficulties of the traffic junction environment.

The easy task has one junction of two one-way roads on a 7ˆ7 grid with N max " 5 and p arrive " 0.30, while the hard task has four connected junctions of two-way roads on a 18ˆ18 grid with N max " 20 and p arrive " 0.05.See FIG5 , 4b for an example of the four two-way junctions in the hard task.

As shown in Model Interpretation.

Interpreting the learned policies, FIG5 shows braking probabilities at different locations: cars tend to brake close to or right before entering traffic junctions, which is reasonable since junctions have the highest chances for collisions.

Turning our attention to attention probabilities FIG5 ), we can see that cars are most-attended to when in the 'internal grid' -right after crossing the 1st junction and before hitting the 2nd junction.

These attention probabilities are intuitive: cars learn to attentively attend to specific sensitive locations with the most relevant local observations to avoid collisions.

Finally, FIG5 compares total number of cars in the environment vs. number of cars being attended to with probability ą 0.1 at any time.

Interestingly, these are (loosely) positively correlated, with Spearman's σ " 0.49, which shows that TarMAC is able to adapt to variable number of agents.

Crucially, agents learn this dynamic targeting behavior purely from task rewards with no handcoding!

Note that the right shift between the two curves is expected, as it takes a few timesteps of communication for team size changes to propagate.

At a relative time shift of 3, the Spearman's rank correlation between the two curves goes up to 0.53.Message size vs. multi-stage communication.

We study performance of TarMAC with varying message value size and number of rounds of communication on the 'hard' variant of the traffic junction task.

As can be seen in FIG3 , multiple rounds of communication leads to significantly higher performance than simply increasing message size, demonstrating the advantage of multistage communication.

In fact, decreasing message size to a single scalar performs almost as well as 64-d, perhaps because even a single real number can be sufficiently partitioned to cover the space of meanings/messages that need to be conveyed for this task.

Finally, we benchmark TarMAC on a cooperative point-goal navigation task in House3D BID22 .

House3D provides a rich and diverse set of publicly-available 2 3D indoor environments, wherein agents do not have access to the top-down map and must navigate purely from first-person vision.

Similar to SHAPES, the agents are tasked with finding a specified goal (such as 'fireplace'), spawned at random locations in the environment and allowed to communicate with each other and move around.

Each agent gets a shaped reward based on progress towards the specified target.

An episode is successful if all agents end within 0.5m of the target object in 50 navigation steps.

TAB10 shows success rates on a find[fireplace] task in House3D.

A no-communication navigation policy trained with the same reward structure gets a success rate of 62.1%.

Mean-pooled communication (no attention) performs slightly better with a success rate of 64.3%, and TarMAC achieves the best success rate at 68.9%.

FIG6 visualizes predicted navigation trajectories of 4 agents.

Note that the communication vectors are significantly more compact (32-d) than the high-dimensional observation space, making our approach particularly attractive for scaling to large teams.

We introduced TarMAC, an architecture for multi-agent reinforcement learning which allows targeted interactions between agents and multiple stages of collaborative reasoning at every timestep.

Evaluation on three diverse environments show that our model is able to learn intuitive attention behavior and improves performance, with downstream task-specific team reward as sole supervision.

While multi-agent navigation experiments in House3D show promising performance, we aim to exhaustively benchmark TarMAC on more challenging 3D navigation tasks because we believe this is where decentralized targeted communication can have the most impact -as it allows scaling to a large number of agents with large observation spaces.

Given that the 3D navigation problem is hard in and of itself, it would be particularly interesting to investigate combinations with recent advances orthogonal to our approach (e.g. spatial memory, planning networks) with the TarMAC framework.

@highlight

Targeted communication in multi-agent cooperative reinforcement learning
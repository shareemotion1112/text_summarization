Learning communication via deep reinforcement learning has recently been shown to be an effective way to solve cooperative multi-agent tasks.

However, learning which communicated information is beneficial for each agent's decision-making remains a challenging task.

In order to address this problem, we introduce a fully differentiable framework for communication and reasoning, enabling agents to solve cooperative tasks in partially-observable environments.

The framework is designed to facilitate explicit reasoning between agents, through a novel memory-based attention network that can learn selectively from its past memories.

The model communicates through a series of reasoning steps that decompose each agent's intentions into learned representations that are used first to compute the relevance of communicated information, and second to extract information from memories given newly received information.

By selectively interacting with new information, the model effectively learns a communication protocol directly, in an end-to-end manner.

We empirically demonstrate the strength of our model in cooperative multi-agent tasks, where inter-agent communication and reasoning over prior information substantially improves performance compared to baselines.

Communication is one of the fundamental building blocks for cooperation in multi-agent systems.

The ability to effectively represent and communicate information valuable to a task is especially important in multi-agent reinforcement learning (MARL).

Apart from learning what to communicate, it is critical that agents learn to reason based on the information communicated to them by their teammates.

Such a capability enables agents to develop sophisticated coordination strategies that would be invaluable in application scenarios such as search-and-rescue for multi-robot systems (Li et al., 2002) , swarming and flocking with adversaries (Kitano et al., 1999) , multiplayer games (e.g., StarCraft, (Vinyals et al., 2017 ), DoTA, (OpenAI, 2018 ), and autonomous vehicle planning, (Petrillo et al., 2018) Building agents that can solve complex cooperative tasks requires us to answer the question: how do agents learn to communicate in support of intelligent cooperation?

Indeed, humans inspire this question as they exhibit highly complex collaboration strategies, via communication and reasoning, allowing them to recognize important task information through a structured reasoning process, (De Ruiter et al., 2010; Garrod et al., 2010; Fusaroli et al., 2012) .

Significant progress in multiagent deep reinforcement learning (MADRL) has been made in learning effective communication (protocols), through the following methods: (i) broadcasting a vector representation of each agent's private observations to all agents (Sukhbaatar et al., 2016; Foerster et al., 2016) , (ii) selective and targeted communication through the use of soft-attention networks, (Vaswani et al., 2017) , that compute the importance of each agent and its information, (Jiang & Lu, 2018; Das et al., 2018) , and (iii) communication through a shared memory channel (Pesce & Montana, 2019; Foerster et al., 2018) , which allows agents to collectively learn and contribute information at every time instant.

The architecture of (Jiang & Lu, 2018) implements communication by enabling agents to communicate intention as a learned representation of private observations, which are then integrated in the hidden state of a recurrent neural network as a form of agent memory.

One downside to this approach is that as the communication is constrained in the neighborhood of each agent, communicated information does not enrich the actions of all agents, even if certain agent communications may be critical for a task.

For example, if an agent from afar has covered a landmark, this information would be beneficial to another agent that has a trajectory planned towards the same landmark.

In contrast, Memory Driven Multi-Agent Deep Deterministic Policy Gradient (MD-MADDPG), (Pesce & Montana, 2019) , implements a shared memory state between all agents that is updated sequentially after each agent selects an action.

However, the importance of each agent's update to the memory in MD-MADDPG is solely decided by its interactions with the memory channel.

In addition, the sequential nature of updating the memory channel restricts the architecture's performance to 2-agent systems.

Targeted Multi-Agent Communication (TarMAC), (Das et al., 2018) , uses soft-attention (Vaswani et al., 2017) for the communication mechanism to infer the importance of each agent's information, however without the use of memory in the communication step.

The paradigm of using relations in agent-based reinforcement learning was proposed by (Zambaldi et al., 2018) through multi-headed dot-product attention (MHDPA) (Vaswani et al., 2017) .

The core idea of relational reinforcement learning (RRL) combines inductive logic programming (Lavrac & Dzeroski, 1994; D??eroski et al., 2001 ) and reinforcement learning to perform reasoning steps iterated over entities in the environment.

Attention is a widely adopted framework in Natural Language Processing (NLP) and Visual Question Answering (VQA) tasks (Andreas et al., 2016b; a; Hudson & Manning, 2018) for computing these relations and interactions between entities.

The mechanism (Vaswani et al., 2017) generates an attention distribution over the entities, or more simply a weighted value vector based on importance for the task at hand.

This method has been adopted successfully in state-of-the-art results for Visual Question Answering (VQA) tasks (Andreas et al., 2016b) , (Andreas et al., 2016a) , and more recently (Hudson & Manning, 2018) , demonstrating the robustness and generalization capacity of reasoning methods in neural networks.

In the context of multi-agent cooperation, we draw inspiration from work in soft-attention (Vaswani et al., 2017) to implement a method for computing relations between agents, coupled with a memory based attention network from Compositional Attention Networks (MAC) (Hudson & Manning, 2018) , yielding a framework for a memory-based communication that performs attentive reasoning over new information and past memories.

Concretely, we develop a communication architecture in MADRL by leveraging the approach of RRL and the capacity to learn from past experiences.

Our architecture is guided by the belief that a structured and iterative reasoning between non-local entities should enable agents to capture higherorder relations that are necessary for complex problem-solving.

To seek a balance between computational efficiency and adaptivity to variable team sizes, we exploit the soft-attention (Vaswani et al., 2017) as the base operation for selectively attending to an entity or information.

To capture the information and histories of other entities, and to better equip agents to make a deliberate decision, we separate out the attention and reasoning steps.

The attention unit informs the agent of which entities are most important for the current time-step, while the reasoning steps use previous memories and the information guided by the attention step to extract the shared information that is most relevant.

This explicit separation in communication enables agents to not only place importance on new information from other agents, but to selectively choose information from its past memories given new information.

This communication framework is learned in an end-to-end fashion, without resorting to any supervision, as a result of task-specific rewards.

Our empirical study demonstrates the effectiveness of our novel architecture to solve cooperative multi-agent tasks, with varying team sizes and environments.

By leveraging the paradigm of centralized learning and decentralized execution, alongside communication, we demonstrate the efficacy of the learned cooperative strategies.

We consider a team of N agents and model it as a cooperative multi-agent extension of a partially observable Markov decision process (POMDP) (Oliehoek, 2012) .

We characterize this POMDP by the set of state values, S, describing all the possible configurations of the agents in the environment, control actions {A 1 , A 2 , ..., A N }, where each agent i performs an action A i , and set of observations {O 1 , O 2 , ..., O N }, where each agent i's local observation, O i is not shared globally.

Actions are selected through a stochastic policy ?? ?? ?? ??i : (Silver et al., 2014) with policy parameters ?? i , and a new state is generated by the environment according to the transition function T :

S ?? A 1 ?? ... ?? A N ??? S .

At every step, the environment generates a reward, r i :

S ?? A i ??? R, for each agent i and a new local observation o i : S ??? O i .

The goal is to learn a policy such that each agent maximizes the total expected return

where T is the time horizon and ?? is the discount factor.

We choose the deterministic policy gradient algorithms for all our training.

In this framework, the parameters ?? of the policy, ?? ?? ?? ?? , are updated such that the objective J(??) = E s???p ?? ?? ?? ,a????? ?? [R(s, a)] and the policy gradient, (see Appendix A), is given by:

Deep Deterministic Policy Gradient (DDPG) is an adaptation of DPG where the policy ?? ?? ?? and critic Q ?? ?? ?? are approximated as neural networks.

DDPG is an off-policy method, where experience replay buffers, D, are used to sample system trajectories which are collected throughout the training process.

These samples are then used to calculate gradients for the policy and critic networks to stabilize training.

In addition, DDPG makes use of a target network, similar to Deep Q-Networks (DQN) (Mnih et al., 2015) , such that the parameters of the primary network are updated every few steps, reducing the variance in learning.

Recent work (Lowe et al., 2017) proposes a multi-agent extension to the DDPG algorithm, socalled MADDPG, adapted through the use of centralized learning and decentralized execution.

Each agent's policy is instantiated similar to DDPG, as ?? ?? ?? ??i (a i |o i ) conditioned on its local observation o i .

The major underlying difference is that the critic is centralized such that it estimates the joint action-

We operate under this scheme of centralized learning and decentralized execution of MADDPG, (Lowe et al., 2017) , as the critics are not needed during the execution phase.

We introduce a communication architecture that is an adaptation of the attention mechanism of the Transformer network, (Vaswani et al., 2017) , and the structured reasoning process used in the MAC Cell, (Hudson & Manning, 2018) .

The framework holds memories from previous time-steps separately for each agent, to be used for reasoning on new information received by communicating teammates.

Through a structured reasoning, the model interacts with memories and communications received from other agents to produce a memory for the agent that contains the most valuable information for the task at hand.

An agent's memory is then used to predict the action of the agent, such that policy is given by ?? ?? ?? ?? : O i ?? M i ??? A i , where M i is the memory of agent i.

To summarize, before any agent takes an action, the agent performs four operations via the following architectural features: (1) Thought Unit, where each agent encodes its local observations into appropriate representations for communication and action selection, (2) Question Unit, which is used to generate the importance of all information communicated to the agent, (3) Memory Unit, which controls the final message to be used for predicting actions by combining new information from other agents with its own memory, through the attention vector generated in the Question unit, and (4) Action Unit, that predicts the action.

In Figure 1 we illustrate our proposed Structured Attentive Reasoning Network (SARNet).

The thought unit at each time-step t transforms an agent's private observations into three separate vector representations: query, q Figure 1 : SARNet consists of a thought unit, question unit, memory unit and action unit.

The thought unit operates over the current observation, to represent the information for the communication and action step.

The question unit attends to the information from communicating agents, representing the importance of each neighboring agent's information.

The memory unit, guided by the Question Unit, performs a reasoning operation to extract relevant information from communicated information, which is then integrated into memory.

The Action Unit, processes the information from the memory and the observation encoding to predict an action for the current time-step.

The query is used by each agent i to inform the Question Unit which aspects of the communicated information are relevant to the current step.

The key and value are broadcast to all communicating agents.

The key vector is used in the Question Unit to infer the relevance of the broadcasting agent to the current reasoning step, and the value vector is subsequently used to update the information into the memory of agent i.

The resulting information broadcasted by each agent i to all the cooperating agents is then:

This component is designed to capture the importance of each agent in the environment, including the reasoning agent i, similar to the self-attention mechanism in (Vaswani et al., 2017) .

In the attention mechanism used in (Vaswani et al., 2017) , the attention computes a weight for each entity through the use of the sof tmax.

However, we generate the attention mechanism over all individual representations in the vector for each entity, using Eq. 5.

This allows the agent to compute the importance of each individual communicated information from other agents for a particular timestep.

This is performed through a soft attention-based weighted average using the query generated by agent i, and the set of keys, K, that contain the keys, {k

} from all agents.

The recipient agent, i, upon receiving the set of keys, K, from all agents, computes the interaction with every agent through a Hadamard product, , of its own query vector, q i and all the keys, k j , in the set K. qh

, is then applied to every interaction, qh t ij , that defines the query targeted for each communicating agent j, including self, to produce a scalar defining the weight of the particular agent.

A sof tmax operation is then used over the new scalars for each agent to generate the weights specific to each agent.

The use of the linear transformation in Eq. 5 allows the model to specify an importance not only for each individual agent, but more specifically it learns to assign an importance to each element in the information vector, as compared to the approach used in standard soft-attention based networks, such as Transformer (Vaswani et al., 2017) , which only perform a dot-product computation between the query and keys.

The memory unit is responsible for decomposing the set of new values, V , that contain, {v

into relevant information for the current time-step.

Specifically, it computes the interaction of this new knowledge with the memory aggregated from the preceding time-step.

The new retrieved information, from the memory and the values, is then measured in terms of relevance based on the importance of each agent generated in the Question unit.

As a first step, an agent computes a direct interaction between the new values from other agents, v j ??? V , and its own memory, m i .

This step performs a relative reasoning between newly received information and the memory from the previous step.

This allows the model to potentially highlight information from new communications, given information from prior memory.

The new interaction per agent j is evaluated relative to the memory, mi t ij , and current knowledge, V , is then used to compute a new representation for the final attention stage, through a feed-forward network, W .

This enables the model to reason independently on the interaction between new information and previous memory, and new information alone.

Finally, we aggregate the important information, mr ij , based on the weighting calculated in the Question unit, in (6).

This step generates a weighted average of the new information, mr ij , gathered from the reasoning process, based on the attention values computed in (6).

A linear transformation, W

, is applied to the result of the reasoning operation to prepare the information for input to the action cell.

The action unit, as the name implies, predicts the final action of the agent, i, based on the new memory, Eq. 10, computed from the Memory unit and an encoding, e t i , Eq. 2, of its local observation, o i , from the Thought unit.

where ?? a ?? a i is a multi-layer perceptron (MLP) parameterised by ?? a i .

We incorporate the centralized learning-decentralized execution framework from (Lowe et al., 2017) to implement an actor-critic model to learn the policy parameters for each agent.

We use parameter sharing across agents to ease the training process.

The policy network ?? ??i produces actions a i , which is then evaluated by the critic Q ?? ?? i , which aims to minimize the loss function.

, are the observations when actions {a 1 , a 2 , ..., a N } are performed, the experience replay buffer, D, contains the tuples (x, x , m 1 , m 2 , ..., m N , a 1 , a 2 , ...a N , r 1 , r 2 , ...r N ), and the target Q-value is defined as y. To keep the training stable, delayed updates to the target network Q ?? ?? i is implemented, such that current parameters of Q ?? ?? i , are only copied periodically.

The goal of the loss function, L(?? i ) is to minimize the expectation of the difference between the current and the target action-state function.

The gradient of the resulting policy, with communication, to maximize the expectation of the rewards, J(?? i ) = E[R i ], can be written as:

We evaluate our communication architecture on OpenAI's multi-agent particle environment, (Lowe et al., 2017) , a two-dimensional environment consisting of agents and landmarks with cooperative tasks.

Each agent receives a private observation that includes only partial observations of the environment depending on the task.

The agents act independently and collect a shared global reward for cooperative tasks.

We consider different experimental scenarios where a team of agents cooperate to complete tasks against static goals, or compete against non-communicating agents.

We compare our communication architecture, SARNet, to communication mechanisms of CommNet, (Sukhbaatar et al., 2016) , TarMAC (Das et al., 2018) and the non-communicating policy of MADDPG, (Lowe et al., 2017) .

In this environment, N agents need to cooperate to reach L landmarks.

Each agent observes the relative positions of the neighboring agents and landmarks.

The agents are penalized if they collide with each other, and positively rewarded based on the proximity to the nearest landmark.

At each time-step, the agent receives a reward of ???d, where d is the distance to the closest landmark, and penalized a reward of ???1 if a collision occurs with another agent.

In this cooperative task, all agents strive to maximize a shared global reward.

Performance is evaluated per episode by average reward, number of collisions, and occupied landmarks.

Our model outperforms all the baselines achieving a higher reward through lower metrics of average distance to the landmark and collisions for N = L = 3 and N = L = 6 agents as shown in Table 1 .

We hypothesize that in an environment with more agents, the effect of retaining a memory of previous communications from other agents allows the policy to make a more informed decision.

This leads to a significant reduction in collisions, and lower distance to the landmark.

Our architecture outperforms TarMAC, which uses a similar implementation of soft-attention, albeit without a memory, and computing the communication for the next time-step, unlike SARNet, where the communication mechanism in time t, shapes the action, a t i for the current time-step.

We also show the attention metrics for agent 0 at a single time-step during the execution of the tasks with N = 6 agents in Fig. 3a .

Table 1 : Partially observable cooperative navigation.

For N = L = 3, the agents can observe the nearest 1 agent and 2 landmarks, and for N = L = 6, the agents can observe, 2 agents and 3 landmarks.

Number of collisions between agents, and average distance at the end of the episode are measured.

This task involves a slower moving team of N cooperating agents chasing M faster moving agents in an environment with L static landmarks.

Each agent receives its own local observation, where it can observe the nearest prey, predator, and landmarks.

Predators are rewarded by +10 every time they collide with a prey, and subsequently the prey is penalized ???10.

Since the environment is unbounded, the prey are also penalized for moving out of the environment.

Predators are trained using the SARNet, TarMAC, CommNet, and MADDPG and prey are trained using DDPG.

Due to the nature of dynamic intelligent agents competing with the communicating agents, the complexity of the task increases substantially.

As shown in Fig. 3b , we observe that agent 0 learns to place a higher importance on agent 1's information over itself.

This dynamic nature of the agent, in selecting which information is beneficial, coupled with extracting relevant information from the memory, enables our architecture to substantially outperform the baseline methods, Table 2a .

Table 2 : In 2a, N = L = 3, M = 1, the agents can observe the nearest 1 predator, 1 prey and 2 landmarks, and for N = L = 6, M = 2, the agents can observe, 3 predators, 1 prey and 3 landmarks.

Number of prey captures by the predators per episode is measured.

For Table 2b , we measure the avg.

success rate of the communicating agents N , to reach the target landmark, and the same for the adversary M .

Larger values for N are desired, and lower for M .

(a) Partially observable predatory-prey Table 2b , both TarMAC and CommNet agents choose to stay far away from the target landmark, such that the adversarial agent, follow them.

In contrast, SARNet and MADDPG, learn to spread out over all the landmarks, with a higher adversarial score, but achieving an overall higher mean reward for the complete task.

(a) Attention for agent 0, at a single time-step for cooperative navigation environment for N = L = 6.

(b) Attention for agent 0, at a single time-step in a predator-prey environment, for N = 6, and M = 2.

Figure 3: Visualizing attention predicted by agent i over all agents during the initial stages of the task.

We observe that agents learn to devalue their own information if it is more advantageous to place importance on information from other agents when reading and writing to the memory unit.

We perform additional benchmarks showcasing the importance of memory in our communication architecture.

By introducing noise in the memory at each time-step, we evaluate the performance of SARNet on partially-observable cooperative navigation and predator-prey environments.

We find a general worsening of the results for both tasks as demonstrated by the metrics in Table 3 .

However, we note that in spite of the corrupted memory, the agent's policy is robust to adversarial noise, as it learns to infer the important information from the thorough reasoning process in the communication.

Table 3 : Performance metrics when a Gaussian Noise of mean 0 and variance 1 is introduced in the memory (MC-SARNet) during execution of the tasks for predator-prey and cooperative navigation.

We have introduced a novel framework, SARNet, for communication in multi-agent deep RL which performs a structured attentive reasoning between agents to improve coordination skills.

Through a decomposition of the representations of communication into reasoning steps, our agents exceed baseline methods in overall performance.

Our experiments demonstrate key benefits of gathering insights from (1) its own memories, and (2) the internal representations of the information available to agent.

The communication architecture is learned end-to-end, and is capable of computing taskrelevant importance of each piece of computed information from cooperating agents.

While this multi-agent communication mechanism shows promising results, we believe that we can further adapt this method to scale to a larger number of agents, through a gating mechanism to initiate communication, and decentralized learning.

Policy gradient (PG) methods are the popular choice for a variety of reinforcement learning (RL) tasks in the context described above.

In the PG framework, the parameters ?? of the policy are directly adjusted to maximize the objective J(??) = E s???p ?? ?? ?? ,a????? ?? ?? ?? [R], by taking steps in the direction of ??? ?? J(??), where p ?? ?? ?? , is the state distribution, s is the sampled state and a is the action sampled from the stochastic policy.

Through learning a value function for the state-action pair, Q ?? ?? ?? (s, a), which estimates how good an optimal action a is for an agent in state s, the policy gradient is then written as, (Sutton et al., 2000) :

Several variations of PG have been developed, primarily focused on techniques for estimating Q ?? ?? ?? .

For example, the REINFORCE algorithm (Williams, 1992) uses a rather simplistic method of sample return calculated as a cumulative expected reward for an episode with a discount factor ??, R t = T i=t ?? i???t r i .

When temporal-difference learning (Sutton et al., 1998 ) is used, the learned function Q ?? ?? ?? (s, a) is described as the critic, which leads to several different actor-critic algorithms (Sutton et al., 1998), (Konda & Tsitsiklis, 2000) , where the actor could be a stochastic ?? ?? ?? ?? or deterministic policy ?? ?? ?? ?? for predicting actions.

Hyperparameters We use batch synchronous method for off-policy gradient methods, (Nair et al., 2015; Stooke & Abbeel, 2018) , for all the experiments.

Each environment instance has a separate replay buffer of size 10 6 .

All policies are trained using the Adam optimizer with a learning rate of 5 ?? 10 ???4 , a discount factor, ?? = 0.96.

and ?? = 0.001, for the soft update of the target network.

The query, key and values, share a common first layer of size 128, and subsequently are linearly projected to 32-d.

Batch normalization and dropouts in the communication channel are implemented.

The observation, and action units have two hidden layers of size 128 units with ReLU as the activation function.

The critic is implemented as a 2-layer MLP, with 1024 and 512 units.

We use a batch-size of 128, and updates are initiated after accumulating experiences for 1280 episodes.

Exploration noise is implemented through Orhnstein-Uhlenbeck process with ?? = 0.15 and ?? = 0.3.

All experimental results are averaged over 3 separate runs, with different random seeds.

Baseline Hyperparameters Policies for TarMAC, CommNet and MADDPG are instantiated as an MLP, similar to SARNet.

All layers in the policy network are of size 128 units, while the critic is a 2-layer network with 1024, 512 units.

Both TarMAC and CommNet are implemented with 1-stage communication.

For TarMAC, the query's, and key's have 16 units, and values are 32 units as described in (Das et al., 2018) .

All other training parameters are kept similar to the SARNet implementation.

@highlight

Novel architecture of memory based attention mechanism for multi-agent communication.
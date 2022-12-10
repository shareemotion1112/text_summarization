We address the problem of learning structured policies for continuous control.

In traditional reinforcement learning, policies of agents are learned by MLPs which take the concatenation of all observations from the environment as input for predicting actions.

In this work, we propose NerveNet to explicitly model the structure of an agent, which naturally takes the form of a graph.

Specifically, serving as the agent's policy network, NerveNet first propagates information over the structure of the agent and then predict actions for different parts of the agent.

In the experiments, we first show that our NerveNet is comparable to state-of-the-art methods on standard MuJoCo environments.

We further propose our customized reinforcement learning environments for benchmarking two types of structure transfer learning tasks, i.e., size and disability transfer.

We demonstrate that policies learned by NerveNet are significantly better than policies learned by other models and are able to transfer even in a zero-shot setting.

Deep reinforcement learning (RL) has received increasing attention over the past few years, with the recent success of applications such as playing Atari Games, Mnih et al. (2015) , and Go, BID26 .

Significant advances have also been made in robotics using the latest RL techniques, e.g., BID1 ; BID19 .Many RL problems feature agents with multiple dependent controllers.

For example, humanoid robots consist of multiple physically linked joints.

Action to be taken by each joint or the body should thus not only depend on its own observations but also on actions of other joints.

Previous approaches in RL typically use MLP to learn the agent's policy.

In particular, MLP takes the concatenation of observations from the environment as input, which may be measurements like positions, velocities of body and joints in the current time instance.

The MLP policy then predicts actions to be taken by every joint and body.

Thus the task of the MLP policy is to discover the latent relationships between observations.

This typically leads to longer training times, requiring more exposure of the agent to the environment.

In our work, we aim to exploit the body structure of an agent, and physical dependencies that naturally exist in such agents.

We rely on the fact that bodies of most robots and animals have a discrete graph structure.

Nodes of the graph may represent the joints, and edges represent the (physical) dependencies between them.

In particular, we define the agent's policy using a Graph Neural Network, Scarselli et al. (2009) , which is a neural network that operates over graph structures.

We refer to our model as NerveNet due to the resemblance of the neural nervous system to a graph.

NerveNet propagates information between different parts of the body based on the underlying graph structure before outputting the action for each part.

By doing so, NerveNet can leverage the structure information encoded by the agent's body which is advantageous in learning the correct inductive bias, and thus is less prone to Figure 1: Visualization of the graph structure of CentipedeEight in our environment.

We use this agent for testing the ability of transfer learning of our model.

Since for this agent, each body node is paired with at least one joint node, we omit the body nodes and fill up the position with the corresponding joint nodes.

By omitting the body nodes, a more compact graph is constructed, the details of which are illustrated in the experimental section.overfitting.

Moreover, NerveNet is naturally suitable for structure transferring tasks as most of the model weights are shared across the nodes and edges, respectively.

We first evaluate our NerveNet on standard RL benchmarks such as the OpenAI Gym, BID5 which stem from MuJoCo.

We show that our model achieves comparable results to state-of-the-art MLP based methods.

To verify our claim regarding the structure transfer, we further introduce our customized RL environments which are based on the ones of Gym.

Two types of structure transfer tasks are designed, size transfer and disability transfer.

In particular, size transfer focuses on the scenario in which policies are learned for small-sized agents (simpler body structure) and applied directly to large-sized agents which are composed by some repetitive components shared with the small-sized agent.

Secondly, disability transfer investigates scenarios in which policies are learned for one agent and applied to the same agent with some components disabled.

Our experiments demonstrate that for structure transfer tasks our NerveNet is significantly better than all other competitors, and can even achieve zero-shot learning for some agents.

For the multi-task learning tasks, NerveNet is also able to learn policies that are more robust with better efficiency.

The main contribution of this paper is the following: We explore the problem of learning transferable and generalized features by incorporating a prior on the structure via graph neural networks.

NerveNet permits powerful transfer learning from one structure to another, which goes well beyond the ability of previous models.

NerveNet is also more robust and has more potential in performing multi-task learning.

The demo and code for this project are released, under the project page of http://www.cs.toronto.edu/˜tingwuwang/nervenet.html.

In this section, we first introduce the notation.

We then explain how to construct the graph for each of our agents, followed by the description of the NerveNet.

Finally, we describe the learning algorithm for our model.

We formulate the locomotion control problems as an infinite-horizon discounted Markov decision process (MDP).

To fully describe the MDP for continuous control problems which include locomotion control, we define the state space or observation space as S and action space as A. To interact with the environments, the agent generates its stochastic policy π θ (a τ |s τ ) based on the current state s τ ∈ S, where a τ ∈ A is the action and θ are the parameters of the policy function.

The environment on the other hand, produces a reward r(s τ , a τ ) for the agent, and the agent's objective is to find a policy that maximizes the expected reward.

Controller's Output Vector

Figure 2: In this figure, we use Walker-Ostrich as an example of NerveNet.

In the input model, for each node, NerveNet fetches the corresponding elements from the observation vector.

NerveNet then computes the messages between neighbors in the graph, and updates the hidden state of each node.

This process is repeated for a certain number of propagation steps.

In the output model, the policy is produced by collecting the output from each controller.

In real life, skeletons of most robots and animals have a discrete graph structure, and are most often trees.

Simulators such as the MuJoCo engine by BID34 , organize the agents using an XML-based kinematic tree.

In our experiments, we will use the tree graphs as per MuJoCo.

Note that our model can be directly applied to arbitrary graphs.

In particular, we assume two types of nodes in our tree: body and joint.

The body nodes are abstract nodes used to construct the kinematic tree via nesting, which is similar to the coordinate frame system used in robotics, BID29 .

The joint node represents the degrees of freedom of motion between the two body nodes.

Take a simple humanoid as an example; the body nodes Thigh and Shin are connected via the Knee, where Knee is a hinge joint.

We further add a root node which observes additional information about the agent.

For example, in the Reacher agent in MuJoCo, the root node has access to the target position of the agent.

We build edges to form a tree graph.

Fig. 1 illustrates the graph structure of an example agent, CentipedeEight.

We sketch the agent and its corresponding graph in the left and right part of the figure, respectively.

Note that for better visualization, we omit the joint nodes and use edges to represent the physical connections of joint nodes.

Different elements of the agent are parsed into nodes with different colors.

Further details are provided in the experimental section.

We now turn to NerveNet which parametrizes the policy with a Graph Neural Network.

Before delving into details, we first introduce our notation.

We then specify the input model which helps to initialize the hidden state of each node.

We further introduce the propagation model that updates these hidden states.

Finally, we describe the output model.

We denote the graph structure of the agent as G = (V, E) where V and E are the sets of nodes and edges, respectively.

We focus on the directed graphs as the undirected case can be easily addressed by splitting one undirected edge into two directed edges.

We denote the out-going neighborhood of node u as N out (u) which contains all endpoints v with (u, v) being an edge in the graph.

Similarly, we denote the in-coming neighborhood of node u as N in (u).

Every node u has an associated node type p u ∈ {1, 2, . . .

, P }, which in our case corresponds to body, joint and root.

We also associate each edge (u, v) with an edge type c (u,v) ∈ {1, 2, . . .

, C}. Node type can help in capturing different importances across nodes.

Edge type can be used to describe different relationships between nodes, and thus propagate information between them differently.

One can also add more than one edge type to the same edge which results in a multi-graph.

We stick to simple graphs for simplicity.

One interesting fact is that we have two notions of "time" in our model.

One is the time step in the environment which is the typical time coordinate for RL problems.

The other corresponds to the internal propagation step of NerveNet.

These two coordinates work as follows.

At each time step of the environment, NerveNet receives observation from the environment and performs a few internal propagation steps in order to decide on the action to be taken by each node.

To avoid confusion, throughout this paper, we use τ to describe the time step in the environment and t for the propagation step.

For each time step τ in the environment, the agent receives an observation s τ ∈ S. The observation vector s τ is the concatenation of observations of each node.

We denote the elements of observation vector s τ corresponding to node u with x u .

From now on, we drop the time step in the environment to derive the model for simplicity.

The observation vector goes through an input network to obtain a fixed-size state vector as follows: DISPLAYFORM0 where the subscript and superscript denote the node index and propagation step, respectively.

Here, F in may be a MLP and h 0 u is the state vector of node u at propagation step 0.

Note that we may need to pad zeros to the observation vectors if different nodes have observations of different sizes.

We now describe the propagation model of our NerveNet which mimics a synchronous message passing system studied in distributed computing, BID2 .

We will show how the state vector of each node is updated from one propagation step to the next.

This update process is recurrently applied during the whole propagation.

We leave the details to the appendix.

In particular, at propagation step t, for every node u, we have access to a state vector h t u .

For every edge (u, v) ∈ N out (u), node u computes a message vector as below, m DISPLAYFORM0 where M c (u,v) is the message function which may be an identity mapping or a MLP.

Note that the subscript c (u,v) indicates that edges of the same edge type share the same instance of the message function.

For example, the second torso in Fig. 1 sends a message to the first and third torso, as well as the LeftHip and RightHip.

Message Aggregation Once every node finishes computing messages, we aggregate messages sent from all in-coming neighbors of each node.

Specifically, for every node u, we perform the following aggregation:m DISPLAYFORM1 where A is the aggregation function which may be a summation, average or max-pooling function.

Here,mStates Update We now update every node's state vector based on both the aggregated message and its current state vector.

In particular, for every node u, we perform the following update: DISPLAYFORM2 where U is the update function which may be a gated recurrent unit (GRU), a long short term memory (LSTM) unit or a MLP.

From the subscript p u of U , we can see that nodes of the same node type share the same instance of the update function.

The above propagation model is then recurrently applied for a fixed number of time steps T to get the final state vectors of all nodes, i.e., {h DISPLAYFORM3

In RL, agents typically use a MLP policy, where the network outputs the mean of the Gaussian distribution for each of the actions, while the standard deviation is a trainable vector, BID24 .

In our output model, we also treat standard deviation in the same way.

However, instead of predicting the action distribution of all nodes by a single network, we make predictions for each individual node.

We denote the set of nodes which are assigned controllers for the actuators as O. For each such node, a MLP takes its final state vectors h T u∈O as input and produces the mean of the action of the Gaussian policy for the corresponding actuator.

For each output node u ∈ O, we define its output type as q u .

Different sharing schemes are available for the instance of MLP O qu , for example, we can force the nodes with similar physical structure to share the instance of MLP.

For example, in Fig. 1 , two LeftHip nodes have a shared controller.

Therefore, we have the following output model: DISPLAYFORM0 where µ u∈O is the mean value for action applied on each actuator.

In practice, we found that we can force controllers of different output types to share one unified controller, while not hurting the performance.

By integrating the produced Gaussian policy for each action, the probability density of the stochastic policy is calculated as DISPLAYFORM1 where a τ ∈ A is the output action, and σ u is the variable standard deviation for each action.

Here, θ represents the parameters of the policy function.

To interact with the environments, the agent generates its stochastic policy π θ (a τ |s τ ) after several propagation steps.

The environment on the other hand, produces a reward r(s τ , a τ ) for the agent, and transits to the next state with transition probability P (s τ +1 |s τ ).

The target of the agent is to maximize its cumulative return DISPLAYFORM0 To optimize the expected reward, we use the proximal policy optimization (PPO) by BID24 .

In PPO, the agents alternate between sampling trajectories with the latest policy and performing optimization on surrogate objective using the sampled trajectories.

The algorithm tries to keep the KL-divergence of the new policy and the old policy within the trust region.

To achieve that, PPO clips the probability ratio and adds an KL-divergence penalty term to the loss.

The likelihood ratio is defined as r τ (θ; θ old ) = π θ (a τ |s τ )/π θ old (a τ |s τ ).

Following the notation and the algorithm of PPO, our NerveNet tries to minimize the summation of the original loss in Eq. FORMULA7 , KL-penalty and the value function loss which is defined as: DISPLAYFORM1 whereÂ t is the generalized advantage estimation (GAE) calculated using algorithm from BID23 , and is the clip value, which we choose to be 0.2.

Here, β is a dynamical coefficient adjusted to keep the KL-divergence constraints, and α is used to balance the value loss.

Note that in Eq. (8), V (s t ) target is the target state value in accordance with the GAE method.

To optimize thẽ J(θ), PPO make use of the policy gradient in BID30 to do first-order gradient descent optimization.

Value Network To produce the state value V θ (s τ ) for given observation s τ , we have several alternatives: (1) using one GNN as the policy network and using one MLP as the value network (NerveNet-MLP); (2) using one GNN as policy network and using another GNN as value network (NerveNet-2) (without sharing the parameters of the two GNNs); (3) using one GNN as both policy network and value network (NerveNet-1).

The GNN for value network is very similar to the GNN for policy network.

The output for value GNN is a scalar instead of a vector of mean action.

We will compare these variants in the experimental section.

Reinforcement Learning Reinforcement learning (RL) has recently achieved huge success in a variety of applications.

Powered by the progress of deep neural networks, Krizhevsky et al. (2012) , agents are now able to successfully play Atari Games and beat the world's best (human) players in the game of Go (Mnih et al., 2015; BID26 .

Based on simulation engines like MuJoCo, BID34 , numerous algorithms have been proposed to train agents also in continuous control problems BID24 BID22 Metz et al., 2017) .Structure in RL Most approaches that exploit priors on structure of the problem fall in the domain of hierarchical RL, (Kulkarni et al., 2016; BID35 , which mainly focus on modeling intrinsic motivation of agents.

In BID21 , the authors extend the deep RL algorithms to MDPS with parameterized action space by exploiting the structure of action space and bounding the action space gradients.

Graphs have been used in RL problems prior to our work.

In Metzen (2013); Mabu et al. FORMULA1 ; BID25 ; Mahadevan & Maggioni (2007) , the authors use graphs to learn a representation of the environment.

However, these methods are limited to problems with simple dynamical models like for example the task of 2d-navigation, and thus these problems are usually solved via model-based RL.

However, for complex multi-joint agents, learning the dynamical model as well as predicting the transition of states is time consuming and biased.

For problems of training model-free multi-joint agents in complex physical environments, relatively little attention has been devoted to modeling the physical structure of the agents.

Graph Neural Networks There have been many efforts to generalize neural networks to graphstructured data.

One line of work is based on convolutional neural networks (CNNs).

In BID6 BID11 Kipf & Welling, 2017) , CNNs are employed in the spectral domain relying on the graph Laplacian matrix.

BID17 BID13 used hash functions in order to apply CNNs to graphs.

Another popular direction is based on recurrent neural networks (RNNs) BID17 BID18 Scarselli et al., 2009; BID28 Li et al., 2015; BID31 .

Among RNN based methods, many are only applicable to special structured graph, e.g., sequences or trees, BID28 BID31 .

One class of models which are applicable to general graphs are so-called graph neural networks (GNNs), Scarselli et al. (2009) .

The inference procedure is a forward pass that exploits a fixed-length propagation process which resembles synchronous message passing system in the theory of distributed computing, BID2 .

Nodes in the graph have state vectors which are recurrently updated based on their history and received messages.

One of the representative work of GNNs, i.e., gated graph neural networks (GGNNs) by Li et al. (2015) , uses gated recurrent unit to update the state vectors.

Learning such a model can be achieved by the back-propagation through time (BPTT) algorithm or recurrent back-propagation, BID8 .

It has been shown that GNNs, (Li et al., 2015; Qi et al., 2017; BID16 have a high capacity and achieve state-of-the-art performance in many applications which involve graph-structured data.

In this paper, we model the structure of the reinforcement learning agents using GNNs.

Transfer and Multi-task Learning in RL Recently, there has been increased interest in transfer learning tasks for RL, Taylor & Stone (2009), which mainly focus on transferring the policy learned from one environment to another.

In Rajeswaran et al. (2017b; a) , the authors show that agents in reinforcement learning are prone to over-fitting, and that the learned policies generalize poorly across environments.

In model-based RL, traditional control has been well studied for generalization properties, BID9 .

BID20 try to increase the transferability via learning invariant visual features.

Efforts have also been made from the meta-learning perspective BID12 BID15 a) .

In BID37 , the authors propose a method of transfer learning by using imitation learning.

Transferability comes naturally in our model by exploiting the (shared) graph structure of the agents.

Multi-task learning has also received a lot of attention, BID36 .

In BID33 , the authors use a distilled policy that captures common behaviour across tasks.

BID36

In this section, we first verify the effectiveness of NerveNet on standard MuJoCo environments in OpenAI Gym.

We then investigate the transfer abilities of NerveNet and other competitors by customizing some of those environments, as well as the multi-task learning ability and robustness.

Baselines We compare NerveNet with the standard MLP models utilized by BID24 and another baseline which is constructed as follows.

We first remove the physical graph structure and introduce an additional super node which connects to all nodes in the graph.

This results in a singly rooted depth-1 tree.

We refer to this baseline as TreeNet.

The propagation model of TreeNet is similar to NerveNet where, however, the policy first aggregates the information from all children and then feeds the state vector of the root to the output model.

This simpler model serves as a baseline to verify the importance of the graph structure.

We run experiments on 8 simulated continuous control benchmarks from the Gym, BID5 , which is based on MuJoCo, BID34 .

In particular, we use Reacher, InvertedPendulum, InvertedDoublePendulum, Swimmer, and four walking or running tasks: HalfCheetah, Hopper, Walker2d, Ant.

We set the maximum number of training steps to be 1 million for all environments as it is enough to solve them.

Note that for InvertedPendulum, different from the original one in Gym, we add the distance penalty of the cart and velocity penalty so that the reward is more consistent to the InvertedDoublePendulum.

This change of design also makes the task more challenging.

Results We do grid search to find the best hyperparameters and leave the details in the Appendix 6.3.

As the randomness might have a big impact on the performance, for each environment, we run 3 experiments with different random seeds and plot the average curves and the standard deviations.

We show the results in FIG2 .

From the figures, we can see that MLP with the same setup as in BID24 works the best in most of tasks.

1 NerveNet basically matches the performance of MLP in terms of sample efficiency as well as the performance after it converges.

In most cases, the TreeNet is worse than NerveNet which highlights the importance of keeping the physical graph structure.

We now benchmark our model in the task of structure transfer learning by creating customized environments based on the existing ones from MuJoCo.

We mainly investigate two types of structure transfer learning tasks.

The first one is to train a model with an agent of small size (small graph) and apply the learned model to an agent with a larger size, i.e., size transfer.

When increasing the size of the agent, observation and action space also increase which makes learning more challenging.

Another type of structure transfer learning is disability transfer where we first learn a model for the original agent and then apply it to the same agent with some components disabled.

If one model overfits the environment, disabling some components of the agent might bring catastrophic performance degradation.

Note that for both transfer tasks, all factors of environments do not change except the structure of the agent.

Centipede We create the first environment in which the agent has a similar structure to a centipede.

The goal of the agent is to run as fast as possible along the y-direction in the MuJoCo environment.

The agent consists of repetitive torso bodies where each one has two legs attached.

For two consecutive bodies, we add two actuators which control the rotation between them.

Furthermore, each leg consists of a thigh and shin, which are controlled by two hinge actuators.

By linking copies of torso bodies and corresponding legs, we create agents with different lengths.

Specifically, the shortest Centipede is CentipedeFour and the longest one is CentipedeFourty due to the limit of supported resource of MuJoCo.

For each time step, the total reward is the speed reward minus the energy cost and force feedback from the ground.

Note that in practice, we found that training a CentipedeEight from scratch is already very difficult.

For size transfer experiments, we create many instances which are listed in Figure 4 , like "4to06", "6to10".

For disability transfer, we create CrippleCentipede agents of which two back legs are disabled.

In Figure 4 , CrippleCentipede is specified as "Cp".Snakes We also create a snake-like agent which is common in robotics, BID10 .

We design the Snake environment based on the Swimmer model in Gym.

The goal of the agent is to move as fast as possible.

For details of the environment, please see the schematic figure 16.

To fully investigate the performance of NerveNet, we build several baseline models for structure transfer learning which are explained below.

NerveNet For the NerveNet, since all the weights are exactly the same for the small and the largeagent models, we directly use the old weights trained on the small-agent model.

When the large agent has repetitive structure, we further re-use the weights of the corresponding joints from the small-agent model.

For the MLP based model, while transferring from one structure to another, the size of the input layer changes since the size of the observation changes.

One straight- forward idea is to reuse the weights from the first hidden layer to the output layer and randomly initialize the weights of the new input layer.

MLP Activation Assigning (MLPAA) Another way of making MLP transferable is assigning the weights of the small-agent model to the corresponding partial weights of the large-agent model and setting the remaining weights to be zero.

Note that we do not add or remove any layers from the small-agent model to the large-agent except for changing the size of the layers.

By doing so, we can keep the output of the large-agent model to be same as the small-agent in the beginning, i.e., keeping the same initial policy.

TreeNet TreeNet is similar as the model described before.

We apply the same way of assigning weights as MLPAA to TreeNet for the transfer learning task.

Random We also include the random policy which is uniformly sampled from the action space.

Centipedes For the Centipedes environment, we first run experiments of all models on CentipedeSix and CentipedeFour to get the pre-trained models for transfer learning.

We train different models until these agents run equally well as possible, which is reported in TAB0 .

Note that, in practice, we train TreeNet on CentipedeFour for more than 8 million time steps.

However, due to the difficulty of optimizing TreeNet on CentipedeFour, the performance is still lower.

But visually, the TreeNet agent is able to run in CentipedeFour.

We then examine the zero-shot performance where zero-shot means directly applying the model trained with one setting to the other without any fine-tuning.

To better visualize the results, we linearly normalize the performance to get a performance score, and color the results accordingly.

The normalization scheme is recorded in Appendix 11.

The performance score is less than 1, and is shown in the parentheses behind the original results.

As we can see from Figure 4 (full chart in Appendix 6.5), NerveNet outperforms all competitors on all settings, except in the 4toCp06 scenario.

Note that transferring from CentipedeFour is more difficult than from CentipedeSix since the situation where one torso connects to two neighboring torsos only happens beyond 4 bodies.

TreeNet has a surprisingly good performance on tasks from CentipedeFour.

However, by checking the videos, the learned agent is actually not able to "move" as good as other methods.

The high reward is mainly due to the fact that TreeNet policy is better at standing still and gaining alive bonus.

We argue that the average running-length in each episode is also a very important metric.

By including the results of running-length, we notice that NerveNet is the only model able to walk in the zero-shot setting.

In fact, the performance of NerveNet is orders-of-magnitude better, and most of the time, agents from other methods cannot even move forward.

We also notice that if transferred from CentipedeSix, NerveNet is able to provide walkable pre-trained models on all new agents.

We fine-tune for both size transfer and disability transfer experiments and show the training curves in Figure 5 .

From the figure, we can see that by using the pre-trained model, NerveNet significantly decreases the number of episodes required to reach the level of reward which is considered as solved.

By looking at the videos, we notice that the bottleneck of learning for the agent is "how to stand".

When training from scratch, it can be seen that almost 0.5 million time steps are spent on a very flat reward surface.

Therefore, the MLPAA agents, which copy the learned policy, are able to stand and bypass this time-consuming process and reach to a good performance in the end.

Moreover, by examining the result videos, we noticed that the "walk-cycle" behavior is observed for NerveNet but is not common for others.

Walk-cycle are adopted for many insects in the world, BID4 .

For example, six-legged ants use a tripedal gait, where the legs are used in two separate triangles alternatively touching the ground.

We give more details of walk-cycle in Section 4.5.One possible reason is that the agent of MLP based method (MLPAA, MLPP) learns a policy that does not utilize all legs.

From CentipedeEight and up, we do not observe any MLP agents to be able to coordinate all legs whereas almost all policies learned by NerveNet use all legs.

Therefore, NerveNet is better at utilizing structure information and not over-fitting the environments.

Snakes The zero-shot performance for snakes is summarized in Figure 6 .

As we can see, NerveNet has the best performance on all transfer learning tasks.

In most cases, NerveNet has a starting reward value of more than 300, which is a pretty good policy since 350 is considered as solved for snakeThree.

By looking at the videos, we found that agents of other competitors are not able to control the new actuators in the zero-shot setting.

They either overfit to the original models, where the policy is completely useless in the new setting (e.g., the MLPAA is worse than random policy in SnakeThree2SnakeFour), or the new actuators are not able to coordinate with the old actuators trained before.

While for NerveNet, the actuators are able to coordinate to its neighbors, regardless of whether they are new to the agents.

We also summarize the training curves of fine-tuning in Fig. 7 .

We can observe that NerveNet has a very good initialization with the pre-trained model, and the performance increases with fine-tuning.

When training from scratch, NerveNet is less sample efficient compared to the MLP model which might be caused by the fact that optimizing our model is more challenging than MLP.

Fine-tuning helps to improve the sample efficiency of our model by a large margin.

At the same time, although the MLPAA has a very good initialization, its performance progresses slowly with the increasing number of episodes.

In most experiments, the MLPAA and TreeNet did not match the performance of its non-pretrained MLP baseline.

In this section, we show that NerveNet has a good potential of multi-task learning by incorporating structure prior into the network structure.

It is important to point out that multi-task learning represents a very difficult, and more often the case, unsolved problem in RL.

Most multi-task learning algorithms, BID33 ; BID1 Oh et al. (2017); BID38 have not been applied to domains as difficult as locomotion for complex physical models, not to mention multi-task learning among different agents with different dynamics.

In this work, we constrain our problem domain, and design the Walker multi-task learning taskset, which contains five 2d-walkers.

We aim to test the model's ability of multi-task learning, in particular, the ability to control multiple agents using one unified network.

Table 3 : Results of robustness evaluations.

Note that we show the average results for each type of parameters after perturbation.

And the results are columned by the agent type.

The ratio of the average performance of perturbed agents and the original performance is shown in the figure.

Details are listed in 6.6.

To show the ability of multi-task learning of NerveNet, we design several baselines.

We use a vanilla multi-task policy update for all models.

More specifically, for each sub-task in the multi-task learning task-set, we use an equal number of time steps for each policy's update and calculate the gradients separately.

Gradients are then aggregated and the mean value of gradients is applied to update the network.

To compensate for the additional difficulty in training more agents and tasks, we linearly increase the number of update epochs during each update in training, as well as the total number of time steps generated before the training is terminated.

The hyper-parameter setting is summarized in Appendix 6.7.NerveNet For NerveNet, the weights are naturally shared among different agents.

More specifically, for different agents, the weight matrices for propagation and output are shared.

For the MLP method, we shared the weight matrices between hidden layers.

In the MLP Sharing approach, the total size of the weight matrices grows with the number of tasks.

For different agents, whose dimension of the observations are usually different, weights from observation to the first hidden layer cannot be reused in the MLP Sharing approach.

Therefore in the MLP Aggregation method, we multiply each element of the observation vector separately by one matrix, and aggregate the resulting vectors from each element.

The size of this multiplying matrix is (1, dimension of the first hidden layer).TreeNet Similarly, TreeNet also has the benefits that its weights are naturally shared among different agents.

However, TreeNet has no knowledge of the agents' physical structure, where the information of each node is aggregated into the root node.

We also include the baseline of training single-task MLP for each agent.

We train the single-task MLP baselines for 1 million time steps per agent.

In the FIG6 , we align the results of single-task MLP baseline and the results of multi-task models by the number of episodes of one task.

As can be seen from FIG6 , NerveNet achieves the best performance in all the sub-tasks.

In Walker-HalfHumanoid, Walker-Hopper, Walker-Ostrich, Walker-Wolf our NerveNet is able to out-perform other agents by a large margin.

In Walker-Horse, the performance of NerveNet and MLP Sharing are relatively similar.

For MLP Sharing, the performance on other four agents are relatively limited, while for Walker-Hopper, the improvement of performance is limited from half of the experiment onwards.

The MLP Aggregation and TreeNet methods are not able to solve the multi-task learning problem, with both of them stuck at a very low reward level.

In the vanilla optimization setting, we show that NerveNet has a bigger potential than the baselines.

From Table 2 , one can observe that the performance of MLP drops drastically (42% performance drop) when switching from single-task to multi-task learning, while for NerveNet, there is no obvious drop in performance.

Our intuition is that NerveNet is better at learning generalized features, and learning of different agents can help in training other agents, while for MLP methods, the performance decreases due the competition of different agents.

Figure 10: Results of visualization of feature distribution and trajectory density.

As can be seen from the figure, NerveNet agent is able to learn shareable features for its legs, and certain walk-cycle is learnt during training.

In this section, we also report the robustness of our policy by perturbing the agent parameters.

In reality, the parameters simulated might be different from the actual parameters of the agents.

Therefore, it is important that the agent is robust to parameters perturbation.

The model that has the better ability to learn generalized features are likely more robust.

We perturb the mass of the geometries (rigid bodies) in MuJoCo as well as the scale of the forces of the joints.

We use the pre-trained models with similar performance on the original task for both the MLP and NerveNet.

The performance is tested in five agents from Walker task set.

The average performance is recorded in Table 3 , and the specific details are summarized in Appendix 6.6.

The robustness of NerveNet' policy is likely due to the structure prior of the agent instilled in the network, which facilitates overfitting.

In this section, we try to visualize and interpret the learned representations.

We extract the final state vectors of nodes of NerveNet trained on CentipedeEight.

We then apply 1-D and 2-D PCA on the node representations.

In Figure 10 , we notice that each pair of legs is able to learn invariant representations, despite their different position in the agent.

We further plot the trajectory density map in the feature map.

By recording the period of the walk-cycle, we plot the transformed features of the 6 legs on FIG8 .

As we can see, there is a clear periodic behavior of our hidden representations learned by our model.

Furthermore, the representations of adjacent left legs and the adjacent right legs demonstrate a phase shift, which further proves that our agents are able to learn the walk-cycle without any additional supervision.

We have several variants of NerveNet, based on the type of network we use for the policy/value representation.

We here compare all variants.

Again, we run experiments for each task three times.

The details of hyper-parameters are given in the Appendix.

For each environment, we train the network for one million time steps, with batch size 2050 for one update.

As we can see from FIG9 , the NerveNet-MLP and NerveNet-2 variants perform better than NerveNet-1.

One potential reason is that sharing the weights of the value and policy networks makes the trust-region based optimization methods, like PPO, more sensitive to the weight α of the value function in equation 8.

Based on the figure, choosing α to be 1 is not giving good performance on the tasks we experimented on.

In this paper, we aimed to exploit the body structure of Reinforcement Learning agents in the form of graphs.

We introduced a novel model called NerveNet which uses a Graph Neural Network to represent the agent's policy.

At each time instance of the environment, NerveNet takes observations for each of the body joints, and propagates information between them using non-linear messages computed with a neural network.

Propagation is done through the edges which represent natural dependencies between joints, such as physical connectivity.

We experimentally showed that our NerveNet achieves comparable performance to state-of-the-art methods on standard MuJoCo environments.

We further propose our customized reinforcement learning environments for benchmarking two types of structure transfer learning tasks, i.e., size and disability transfer.

We demonstrate that policies learned by NerveNet are significantly better than policies learned by other models and are able to transfer even in a zero-shot setting.

We use MLP to compute the messages which uses tanh nonlinearities as the activation function.

We do a grid search on the size of the MLP to compute the messages, the details of which are listed in TAB7 , 4.Throughout all of our experiments, we use average aggregation and GRU as the update function.

In MuJoCo, we observe that most body nodes are paired with one and only one joint node.

Thus, we simply merge the two paired nodes into one.

We point out that this model is very compact, and is the standard graph we use in our experiments.

In the Gym environments, observation for the joint nodes normally includes the angular velocity, twist angle and optionally the torque for the hinge joint, and position information for the positional joint.

For the body nodes, velocity, inertia, and force are common observations.

For example in the centipede environment 1, the LeftHip node will receive the angular velocity j , and the twist angle θ j .

For MLP, we run grid search with the hidden size from two layers to three layers, and with hidden size from 32 to 256.

For NerveNet, to reduce the time spent on grid search, we constrain the propagation network and output network to be the same shape.

Similarly, we run grid search with the network's hidden size, and at the same time, we run a grid search on the size of node's hidden states from 32 to 64.

For the TreeNet, we run similar grid search on the node's hidden states and output network's shape.

For details of hyperparameter search, please see TAB7 , 7, 6.

In the MLP-Bind method, we bind the weights of MLP.

By doing this, the weights of the agent from the similar structures will be shared.

For example, in the centipede environment, the weights from observation to action of all the LeftHips are constrained to be same.

As the scale of zero-shot results is very different, we normalize the results across different models for each transfer learning task.

For each task, we record the worst value of results from different models and the pre-set worst value V min .

we set the normalization minimun value as this worst value.

We calculate the normalization maximum value by max(V )/IntLen * IntLen.

@highlight

using graph neural network to model structural information of the agents to improve policy and transferability 

@highlight

A method for representing and learning structured policy for continuous control tasks using Graph Neural Networks

@highlight

The submission proposes incorporation of additional structure into reinforcement learning problems, particularly the structure of the agent's morphology

@highlight

Propose an application of Graph Neural Networks to learning policies for controlling "centipede" robots of different lengths.
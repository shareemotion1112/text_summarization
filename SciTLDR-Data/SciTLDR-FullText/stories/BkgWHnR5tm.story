Despite the recent successes in robotic locomotion control, the design of robot relies heavily on human engineering.

Automatic robot design has been a long studied subject, but the recent progress has been slowed due to the large combinatorial search space and the difficulty in evaluating the found candidates.

To address the two challenges, we formulate automatic robot design as a graph search problem and perform evolution search in graph space.

We propose Neural Graph Evolution (NGE), which performs selection on current candidates and evolves new ones iteratively.

Different from previous approaches, NGE uses graph neural networks to parameterize the control policies, which reduces evaluation cost on new candidates with the help of skill transfer from previously evaluated designs.

In addition, NGE applies Graph Mutation with Uncertainty (GM-UC) by incorporating model uncertainty, which reduces the search space by balancing exploration and exploitation.

We show that NGE significantly outperforms previous methods by an order of magnitude.

As shown in experiments, NGE is the first algorithm that can automatically discover kinematically preferred robotic graph structures, such as a fish with two symmetrical flat side-fins and a tail, or a cheetah with athletic front and back legs.

Instead of using thousands of cores for weeks, NGE efficiently solves searching problem within a day on a single 64 CPU-core Amazon EC2 machine.

The goal of robot design is to find an optimal body structure and its means of locomotion to best achieve a given objective in an environment.

Robot design often relies on careful human-engineering and expert knowledge.

The field of automatic robot design aims to search for these structures automatically.

This has been a long-studied subject, however, with limited success.

There are two major challenges: 1) the search space of all possible designs is large and combinatorial, and 2) the evaluation of each design requires learning or testing a separate optimal controller that is often expensive to obtain.

In BID28 , the authors evolved creatures with 3D-blocks.

Recently, soft robots have been studied in BID13 , which were evolved by adding small cells connected to the old ones.

In BID3 , the 3D voxels were treated as the minimum element of the robot.

Most evolutionary robots BID8 BID24 require heavy engineering of the initial structures, evolving rules and careful human-guidance.

Due to the combinatorial nature of the problem, evolutionary, genetic or random structure search have been the de facto algorithms of automatic robot design in the pioneering works BID28 BID31 BID20 BID16 BID17 BID34 BID2 .

In terms of the underlying algorithm, most of these works have a similar population-based optimization loop to the one used in BID28 .

None of these algorithms are able to evolve kinematically reasonable structures, as a result of large search space and the inefficient evaluation of candidates.

Similar in vein to automatic robot design, automatic neural architecture search also faces a large combinatorial search space and difficulty in evaluation.

There have been several approaches to tackle these problems.

Bayesian optimization approaches BID29 primarily focus on fine-tuning the number of hidden units and layers from a predefined set.

Reinforcement learning BID38 and genetic algorithms BID19 are studied to evolve recurrent neural networks (RNNs) and convolutional neural networks (CNNs) from scratch in order to maximize the validation accuracy.

These approaches are computationally expensive because a large number of candidate networks have to be trained from grounds up.

BID25 and BID30 propose weight sharing among all possible candidates in the search space to effectively amortize the inner loop training time and thus speed up the architecture search.

A typical neural architecture search on ImageNet BID15 ) takes 1.5 days using 200 GPUs BID19 .In this paper, we propose an efficient search method for automatic robot design, Neural Graph Evolution (NGE), that co-evolves both, the robot design and the control policy.

Unlike the recent reinforcement learning work, where the control policies are learnt on specific robots carefully designed by human experts BID21 BID0 BID11 , NGE aims to adapt the robot design along with policy learning to maximize the agent's performance.

NGE formulates automatic robot design as a graph search problem.

It uses a graph as the main backbone of rich design representation and graph neural networks (GNN) as the controller.

This is key in order to achieve efficiency of candidate structure evaluation during evolutionary graph search.

Similar to previous algorithms like BID28 , NGE iteratively evolves new graphs and removes graphs based on the performance guided by the learnt GNN controller.

The specific contributions of this paper are as follows:• We formulate the automatic robot design as a graph search problem.• We utilize graph neural networks (GNNs) to share the weights between the controllers, which greatly reduces the computation time needed to evaluate each new robot design.• To balance exploration and exploitation during the search, we developed a mutation scheme that incorporates model uncertainty of the graphs.

We show that NGE automatically discovers robot designs that are comparable to the ones designed by human experts in MuJoCo , while random graph search or naive evolutionary structure search BID28 fail to discover meaningful results on these tasks.

In reinforcement learning (RL), the problem is usually formulated as a Markov Decision Process (MDP).

The infinite-horizon discounted MDP consists of a tuple of (S, A, γ, P, R), respectively the state space, action space, discount factor, transition function, and reward function.

The objective of the agent is to maximize the total expected reward DISPLAYFORM0 , where the state transition follows the distribution P (s t+1 |s t , a t ).

Here, s t and a t denotes the state and action at time step t, and r(s t , a t ) is the reward function.

In this paper, to evaluate each robot structure, we use PPO to train RL agents BID27 BID11 .

PPO uses a neural network parameterized as π θ (a t |s t ) to represent the policy, and adds a penalty for the KL-divergence between the new and old policy to prevent over-optimistic updates.

PPO optimizes the following surrogate objective function instead: DISPLAYFORM1 We denote the estimate of the expected total reward given the current state-action pair, the value and the advantage functions, as Q t (s t , a t ), V (s t ) and A t (s t , a t ) respectively.

PPO solves the problem by iteratively generating samples and optimizing J PPO BID27 .

Graph Neural Networks (GNNs) are suitable for processing data in the form of graph BID1 BID6 BID18 BID14 BID9 BID12 .

Recently, the use of GNNs in locomotion control has greatly increased the transferability of controllers BID37 .

A GNN operates on a graph whose nodes and edges are denoted respectively as u ∈ V and e ∈ E. We consider the following GNN, where at timestep t each node in GNN receives an input feature and is supposed to produce an output at a node level.

Input Model:

The input feature for node u is denoted as x Propagation Model:

Within each timestep t, the GNN performs T internal propagations, so that each node has global (neighbourhood) information.

In each propagation, every node communicates with its neighbours, and updates its hidden state by absorbing the input feature and message.

We denote the hidden state at the internal propagation step τ (τ ≤ T ) as h t,τ u .

Note that h t,0 u is usually initialized as h t−1,T u , i.e., the final hidden state in the previous time step.

h 0,0 is usually initialized to zeros.

The message that u sends to its neighbors is computed as DISPLAYFORM0 where M is the message function.

To compute the updated h t,τ u , we use the following equations: r DISPLAYFORM1 where R and U are the message aggregation function and the update function respectively, and N G (u) denotes the neighbors of u.

Output Model: Output function F takes input the node's hidden states after the last internal propagation.

The node-level output for node u is therefore defined as µ DISPLAYFORM2 .

Functions M, R, U, F in GNNs can be trainable neural networks or linear functions.

For details of GNN controllers, we refer readers to BID37 .

In robotics design, every component, including the robot arms, finger and foot, can be regarded as a node.

The connections between the components can be represented as edges.

In locomotion control, the robotic simulators like MuJoCo use an XML file to record the graph of the robot.

As we can see, robot design is naturally represented by a graph.

To better illustrate Neural Graph Evolution (NGE), we first introduce the terminology and summarize the algorithm.

Graph and Species.

We use an undirected graph G = (V, E, A) to represent each robotic design.

V and E are the collection of physical body nodes and edges in the graph, respectively.

The mapping DISPLAYFORM0 2: while Evolving jth generation do Evolution outer loop 3:for ith species (θ DISPLAYFORM1 end for 7: DISPLAYFORM2 Mutate from survivors 9: DISPLAYFORM3 Pruning 10: end while A : V → Λ maps the node u ∈ V to its structural attributes A(u) ∈ Λ, where Λ is the attributes space.

For example, the fish in FIG0 consists of a set of ellipsoid nodes, and vector A(u) describes the configurations of each ellipsoid.

The controller is a policy network parameterized by weights θ.

The tuple formed by the graph and the policy is defined as a species, denoted as Ω = (G, θ).Generation and Policy Sharing.

In the j-th iteration, NGE evaluates a pool of species called a generation, denoted as DISPLAYFORM4 where N is the size of the generation.

In NGE, the search space includes not only the graph space, but also the weight or parameter space of the policy network.

For better efficiency of NGE, we design a process called Policy Sharing (PS), where weights are reused from parent to child species.

The details of PS is described in Section 3.4.Our model can be summarized as follows.

NGE performs population-based optimization by iterating among mutation, evaluation and selection.

The objective and performance metric of NGE are introduced in Section 3.1.

In NGE, we randomly initialize the generation with N species.

For each generation, NGE trains each species and evaluates their fitness separately, the policy of which is described in Section 3.2.

During the selection, we eliminate K species with the worst fitness.

To mutate K new species from surviving species, we develop a novel mutation scheme called Graph Mutation with Uncertainty (GM-UC), described in Section 3.3, and efficiently inherit policies from the parent species by Policy Sharing, described in Section 3.4.

Our method is outlined in Algorithm 1.

Fitness represents the performance of a given G using the optimal controller parameterized with θ * (G).

However, θ * (G) is impractical or impossible to obtain for the following reasons.

First, each design is computationally expensive to evaluate.

To evaluate one graph, the controller needs to be trained and tested.

Model-free (MF) algorithms could take more than one million in-game timesteps to train a simple 6-degree-of-freedom cheetah BID27 , while model-based (MB) controllers usually require much more execution time, without the guarantee of having higher performance than MF controllers BID23 BID7 BID4 .

Second, the search in robotic graph space can easily get stuck in local-optima.

In robotic design, local-optima are difficult to detect as it is hard to tell whether the controller has converged or has reached a temporary optimization plateau.

Learning the controllers is a computation bottleneck in optimization.

In population-based robot graph search, spending more computation resources on evaluating each species means that fewer different species can be explored.

In our work, we enable transferablity between different topologies of NGE (described in Section 3.2 and 3.4).

This allows us to introduce amortized fitness (AF) as the objective function across generations for NGE.

AF is defined in the following equation as, DISPLAYFORM0 In NGE, the mutated species continues the optimization by initializing the parameters with the parameters inherited from its parent species.

In past work BID28 , species in one generation are trained separately for a fixed number of updates, which is biased and potentially undertrained or overtrained.

In next generations, new species have to discard old controllers if the graph topology is different, which might waste valuable computation resources.

Given a species with graph G, we train the parameters θ of policy network π θ (a t |s t ) using reinforcement learning.

Similar to BID37 , we use a GNN as the policy network of the controller.

A graphical representation of our model is shown in FIG0 .

We follow notation in Section 2.2.For the input model, we parse the input state vector s t obtained from the environment into a graph, where each node u ∈ V fetches the corresponding observation o(u, t) from s t , and extracts the feature x O,t u with an embedding function Φ. We also encode the attribute information A(u) into x A u with an embedding function denoted as ζ.

The input feature x t u is thus calculated as: DISPLAYFORM0 where [.]

denotes concatenation.

We use θ Φ , θ ζ to denote the weights of embedding functions.

The propagation model is described in Section 2.2.

We recap the propagation model here briefly: Initial hidden state for node u is denoted as h t,0 u , which are initialized from hidden states from the last timestep h t−1,T u or simply zeros.

T internal propagation steps are performed for each timestep, during each step (denoted as τ ≤ T ) of which, every node sends messages to its neighboring nodes, and aggregates the received messages.

h t,τ +1 u is calculated by an update function that takes in h t,τ u , node input feature x t u and aggregated message m t,τ u .

We use summation as the aggregation function and a GRU BID5 as the update function.

For the output model, we define the collection of controller nodes as F, and define Gaussian distributions on each node's controller as follows: DISPLAYFORM1 DISPLAYFORM2 where µ u and σ u are the mean and the standard deviation of the action distribution.

The weights of output function are denoted as θ F .

By combining all the actions produced by each node controller, we have the policy distribution of the agent: DISPLAYFORM3 We optimize π(a t |s t ) with PPO, the details of which are provided in Appendix A.

Between generations, the graphs evolve from parents to children.

We allow the following basic operations as the mutation primitives on the parent's graph G: DISPLAYFORM0 In the M 1 (Add-Node) operation, the growing of a new body part is done by sampling a node v ∈ V from the parent, and append a new node u to it.

We randomly initialize u's attributes from an uniform distribution in the attribute space.

M 2 , Add-Graph: The M 2 (Add-Graph) operation allows for faster evolution by reusing the subtrees in the graph with good functionality.

We sample a sub-graph or leaf node G = (V , E , A ) from the current graph, and a placement node u ∈ V (G) to which to append G .

We randomly mirror the attributes of the root node in G to incorporate a symmetry prior.

M 3 , Del-Graph: The process of removing body parts is defined as M 3 (Del-Graph) operation.

In this operation, a sub-graph G from G is sampled and removed from G.M 4 , Pert-Graph: In the M 4 (Pert-Graph) operation, we randomly sample a sub-graph G and recursively perturb the parameter of each node u ∈ V (G ) by adding Gaussian noise to A(u).We visualize a pair of example fish in FIG0 .

The fish in the top-right is mutated from the fish in the top-left by applying M 1 .

The new node FORMULA2 is colored magenta in the figure.

To mutate each new candidate graph, we sample the operation M and apply M on G as DISPLAYFORM1 p l m is the probability of sampling each operation with l p l m = 1.

To facilitate evolution, we want to avoid wasting computation resources on species with low expected fitness, while encouraging NGE to test species with high uncertainty.

We again employ a GNN to predict the fitness of the graph G, denoted as ξ P (G).

The weights of this GNN are denoted as ψ.

In particular, we predict the AF score with a similar propagation model as our policy network, but the observation feature is only x A u , i.e., the embedding of the attributes.

The output model is a graph-level output (as opposed to node-level used in our policy), regressing to the score ξ.

After each generation, we train the regression model using the L2 loss.

However, pruning the species greedily may easily overfit the model to the existing species since there is no modeling of uncertainty.

We thus propose Graph Mutation with Uncertainty (GM-UC) based on Thompson Sampling to balance between exploration and exploitation.

We denote the dataset of past species and their AF score as D. GM-UC selects the best graph candidates by considering the posterior distribution of the surrogate P (ψ| D): DISPLAYFORM2 Instead of sampling the full model with ψ ∼ P (ψ|D), we follow BID10 and perform dropout during inference, which can be viewed as an approximate sampling from the model posterior.

At the end of each generation, we randomly mutate C ≥ N new species from surviving species.

We then sample a single dropout mask for the surrogate model and only keep N species with highest ξ P .

The details of GM-UC are given in Appendix F.

To leverage the transferability of GNNs across different graphs, we propose Policy Sharing (PS) to reuse old weights from parent species.

The weights of a species in NGE are as follows: DISPLAYFORM0 where θ Φ , θ ζ , θ M , θ U , θ F are the weights for the models we defined earlier in Section 3.2 and 2.2.

Since our policy network is based on GNNs, as we can see from FIG0 , model weights of different graphs share the same cardinality (shape).

A different graph will only alter the paths of message propagation.

With PS, new species are provided with a strong weight initialization, and the evolution will less likely be dominated by species that are more ancient in the genealogy tree.

Previous approaches including naive evolutionary structure search (ESS-Sims) BID28 or random graph search (RGS) utilize human-engineered one-layer neural network or a fully connected network, which cannot reuse controllers once the graph structure is changed, as the parameter space for θ might be different.

And even when the parameters happen to be of the same shape, transfer learning with unstructured policy controllers is still hardly successful BID26 .

We denote the old species in generation j, and its mutated species with different topologies as (θ G , G ) for NGE.

We also denote the network initialization scheme for fully-connected networks as B. We show the parameter reuse between generations in TAB0 .

Mutation Parameter Space Policy Initialization DISPLAYFORM0

In this section, we demonstrate the effectiveness of NGE on various evolution tasks.

In particular, we evaluate both, the most challenging problem of searching for the optimal body structure from scratch in Section 4.1, and also show a simpler yet useful problem where we aim to optimize humanengineered species in Section 4.2 using NGE.

We also provide an ablation study on GM-UC in Section 4.3, and an ablation study on computational cost or generation size in Section 4.4.Our experiments are simulated with MuJoCo.

We design the following environments to test the algorithms.

Fish Env:

In the fish environment, graph consists of ellipsoids.

The reward is the swimming-speed along the y-direction.

We denote the reference human-engineered graph BID33 as G F .

Walker Env: We also define a 2D environment walker constructed by cylinders, where the goal is to move along x-direction as fast as possible.

We denote the reference humanengineered walker as G W and cheetah as G C BID33 .

To validate the effectiveness of NGE, baselines including previous approaches are compared.

We do a grid search on the hyper-parameters as summarized in Appendix E, and show the averaged curve of each method.

The baselines are introduced as follows:

This method was proposed in BID28 , and applied in BID3 BID34 , which has been the most classical and successful algorithm in automatic robotic design.

In the original paper, the author uses evolutionary strategy to train a human-engineered one layer neural network, and randomly perturbs the graph after each generation.

With the recent progress of robotics and reinforcement learning, we replace the network with a 3-layer Multilayer perceptron and train it with PPO instead of evolutionary strategy.

In the original ESS-Sims, amortized fitness is not used.

Although amortized fitness could not be fully applied, it could be applied among species with the same topology.

We name this variant as ESS-Sims-AF.

The goal is to explore how GM-UC affects the performance without the use of a structured model like GNN.

We also want to answer the question of whether GNN is indeed needed.

We use both an unstructured models like MLP, as well as a structured model by removing the message propagation model.

In the Random Graph Search (RGS) baseline, a large amount of graphs are generated randomly.

RGS focuses on exploiting given structures, and does not utilize evolution to generate new graphs.

In this experiment, the task is to evolve the graph and the controller from scratch.

For both fish and walker, species are initialized as random (G, θ).

Computation cost is often a concern among structure search problems.

In our comparison results, for fairness, we allocate the same computation budget to all methods, which is approximately 12 hours on a EC2 m4.16xlarge cluster with 64 cores for one session.

A grid search over the hyper-parameters is performed (details in Appendix E).

The averaged curves from different runs are shown in FIG5 .

In both fish and walker environments, NGE is the best model.

We find RGS is not able to efficiently search the space of G even after evaluating 12, 800 different graphs.

Figure 3: The genealogy tree generated using NGE for fish.

The number next to the node is the reward (the averaged speed of the fish).

For better visualization, we down-sample genealogy sub-chain of the winning species.

NGE agents gradually grow symmetrical side-fins.

generations, but is significantly worse than our method in the end.

The use of AF and GM-UC on ESS-Sims can improve the performance by a large margin, which indicates that the sub-modules in NGE are effective.

By looking at the generated species, ESS-Sims and its variants overfit to local species that dominate the rest of generations.

The results of ESS-BodyShare indicates that, the use of structured graph models without message passing might be insufficient in environments that require global features, for example, walker.

To better understand the evolution process, we visualize the genealogy tree of fish using our model in Figure 3 .

Our fish species gradually generates three fins with preferred {A(u)}, with two side-fins symmetrical about the fish torso, and one tail-fin lying in the middle line.

We obtain similar results for walker, as shown in Appendix C. To the best of our knowledge, our algorithm is the first to automatically discover kinematically plausible robotic graph structures.

Evolving every species from scratch is costly in practice.

For many locomotion control tasks, we already have a decent human-engineered robot as a starting point.

In the fine-tuning task, we verify the ability of NGE to improve upon the human-engineered design.

We showcase both, unconstrained experiments with NGE where the graph (V, E, A) is fine-tuned, and constrained fine-tuning experiments where the topology of the graph is preserved and only the node attributes {A(u)} are fine-tuned.

In the baseline models, the graph (V, E, A) is fixed, and only the controllers are trained.

We can see in FIG7 that when given the same wall-clock time, it is better to co-evolve the attributes and controllers with NGE than only training the controllers.

The figure shows that with NGE, the cheetah gradually transforms the forefoot into a claw, the 3D-fish rotates the pose of the side-fins and tail, and the 2D-walker evolves bigger feet.

In general, unconstrained fine-tuning with NGE leads to better performance, but not necessarily preserves the initial structures.

We also investigate the performance of NGE with and without Graph Mutation with Uncertainty, whose hyper-parameters are summarized in Appendix E. In FIG9 , we applied GM-UC to the evolution graph search task.

The final performance of the GM-UC outperforms the baseline on both fish and walker environments.

The proposed GM-UC is able to better explore the graph space, showcasing its importance.

We also investigate how the generation size N affect the final performance of NGE.

We note that as we increase the generation size and the computing resources, NGE achieves marginal improvement on the simple Fish task.

A NGE session with 16-core m5.4xlarge ($0.768 per Hr) AWS machine can achieve almost the same performance with 64-core m4.16xlarge ($3.20 per Hr) in Fish environment in the same wall-clock time.

However, we do notice that there is a trade off between computational resources and performance for the more difficult task.

In general, NGE is effective even when the computing resources are limited and it significantly outperforms RGS and ES by using only a small generation size of 16.

In this paper, we introduced NGE, an efficient graph search algorithm for automatic robot design that co-evolves the robot design graph and its controllers.

NGE greatly reduces evaluation cost by transferring the learned GNN-based control policy from previous generations, and better explores the search space by incorporating model uncertainties.

Our experiments show that the search over the robotic body structures is challenging, where both random graph search and evolutionary strategy fail to discover meaning robot designs.

NGE significantly outperforms the naive approaches in both the final performance and computation time by an order of magnitude, and is the first algorithm that can discovers graphs similar to carefully hand-engineered design.

We believe this work is an important step towards automated robot design, and may show itself useful to other graph search problems.

A DETAILS OF NERVENET++ Similar to NerveNet, we parse the agent into a graph, where each node in the graph corresponds to the physical body part of the agents.

For example, the fish in FIG0 can be parsed into a graph of five nodes, namely the torso (0), left-fin (1), right-fin (2), and tail-fin bodies (3, 4).

By replacing MLP with NerveNet, the learnt policy has much better performance in terms of robustness and the transfer learning ability.

We here propose minor but effective modifications to BID37 , and refer to this model as NerveNet++.In the original NerveNet, at every timestep, several propagation steps need to be performed such that every node is able to receive global information before producing the control signal.

This is time and memory consuming, with the minimum number of propagation steps constrained by the depth of the graph.

Since the episode of each game usually lasts for several hundred timesteps, it is computationally expensive and ineffective to build the full back-propagation graph.

Inspired by BID22 , we employ the truncated graph back-propagation to optimize the policy.

NerveNet++ is suitable for an evolutionary search or population-based optimization, as it brings speed-up in wall-clock time, and decreases the amount of memory usage.

Therefore in NerveNet++, we propose a propagation model with the memory state, where each node updates its hidden state by absorbing the input feature and a message with time.

The number of propagation steps is no longer constrained by the depth of the graph, and in back-propagation, we save memory and time consumption with truncated computation graph.

The memory state h t+1,τ u depends on the previous actions, observations, and states.

Therefore, the full back-propagation graph will be the same length as the episode length, which is very computationally intensive.

The intuition from the authors in BID22 is that, for the RL agents, the dependency of the agents on timesteps that are far-away from the current timestep is limited.

Thus, negligible accuracy of the gradient estimator will be lost if we truncate the back-propagation graph.

We define a back-propagation length Γ, and optimize the following objective function instead: DISPLAYFORM0 DISPLAYFORM1 Essentially this optimization means that we only back-propagate up to Γ timesteps, namely at the places where κ = 0, we treat the hidden state as input to the network and stop the gradient.

To optimize the objective function, we follow same optimization procedure as in BID37 , which is a variant of PPO Schulman et al. (2017) , where a surrogate loss J ppo (θ) is optimized.

We refer the readers to these papers for algorithm details.

Similar to the fish genealogy tree, in FIG13 , the simple initial walking agent evolves into a cheetah-like structure, and is able to run with high speed.

We also show the species generated by NGE, ESS-Sims (ESS-Sims-AF to be more specific, which has the best performance among all ESS-Sims variants.) and RGS.

Although amortized fitness is a better estimation of the ground-truth fitness, it is still biased.

Species that appear earlier in the experiment will be trained for more updates if it survives.

Indeed, intuitively, it is possible that in real nature, species that appear earlier on will dominate the generation by number, and new species are eliminated even if the new species has better fitness.

Therefore, we design the experiment where we reset the weights for all species θ = (θ Φ , θ ζ , θ M , θ U , θ F ) randomly.

By doing this, we are forcing the species to compete fairly.

FIG0 we notice that this method helps exploration, which leads to a higher reward in the end.

However, it usually takes a longer time for the algorithm to converge.

Therefore for the graph search task in FIG5 we do not include the results with the controller-resetting.

Figure 9 : We present qualitative comparison between the three algorithms in the figure.

Specifically, the aligned comparison between our method and naive baseline are the representative creatures at the same generation (using same computation resources).

Our algorithm notably display stronger dominance in terms of its structure as well as reward.

E HYPER-PARAMETERS SEARCHED All methods are given equal amount of computation budget.

To be more specific, the number of total timesteps generated by all species for all generations is the same for all methods.

For example, if we use 10 training epochs in one generation, each of the epoch with 2000 sampled timesteps, then the computation budget allows NGE to evolve for 200 generations, where each generation has a species size of 64.

For NGE, RGS, ESS-Sims-AF models in FIG0 we run a grid search over the hyper-parameters recorded in TAB4 , and plot the curve with the best results respectively.

To test the performance with and without GM-UC, we use 64-core clusters (generations of size 64).

Here, the hyper-parameters are chosen to be the first value available in TAB4

Thompson Sampling is a simple heuristic search strategy that is typically applied to the multi-armed bandit problem.

The main idea is to select an action proportional to the probability of the action being optimal.

When applied to the graph search problem, Thompson Sampling allows the search to balance the trade-off between exploration and exploitation by maximizing the expected fitness under the posterior distribution of the surrogate model.

Formally, Thompson Sampling selects the best graph candidates at each round according to the expected estimated fitness ξ P using a surrogate model.

The expectation is taken under the posterior distribution of the surrogate P (model|data): DISPLAYFORM0 F.1 SURROGATE MODEL ON GRAPHS.Here we consider a graph neural network (GNN) surrogate model to predict the average fitness of a graph as a Gaussian distribution, namely P (f (G)) ∼ N ξ P (G), σ 2 (G) .

We use a simple architecture that predicts the mean of the Gaussian from the last hidden layer activations, h W (G) ∈ R D , of the GNN, where W are the weights in the GNN up to the last hidden layer.

Greedy search.

We denoted the size of dataset as N .

The GNN weights are trained to predict the average fitness of the graph as a standard regression task: DISPLAYFORM1 Algorithm FORMULA2 for k < maximum parameter updates do 10:Train policy π Gm

end for 12:Evaluate the fitness ξ(G m , θ m )

end for 14: end for Thompson Sampling In practice, Thompson Sampling is very similar to the previous greedy search algorithm.

Instead of picking the top action according to the best model parameters, at each generation, it draws a sample of the model and takes a greedy action under the sampled model.

Approximating Thompson Sampling using Dropout Performing dropout during inference can be viewed as an approximately sampling from the model posterior.

At each generation, we will sample a single dropout mask for the surrogate model and rank all the proposed graphs accordingly.

Sample a model from the posterior of the weights.

for k < maximum parameter updates do

Train policy π Gm

end for Train W and W out on {G n , ξ(G n )} N n=1 using dropout rate 0.5 on the inputs of the fc layers.

for k < maximum parameter updates do 10:Train policy π Gm

end for 12:Evaluate the fitness ξ(G m , θ m )

end for 14: end for

@highlight

Automatic robotic design search with graph neural networks

@highlight

Proposes an approach for automatic robot design based on Neural graph evolution. The experiments demonstrate that optimizing both controller and hardware is better than optimizing just the controller.

@highlight

The authors propose a scheme based on a graph representation of the robot structure, and a graph-neural-network as controllers to optimize robot structures, combined with their controllers.  
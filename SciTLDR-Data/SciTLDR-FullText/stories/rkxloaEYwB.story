Planning in high-dimensional space remains a challenging problem, even with recent advances in algorithms and computational power.

We are inspired by efference copy and sensory reafference theory from neuroscience.

Our aim is to allow agents to form mental models of their environments for planning.

The cerebellum is emulated with a two-stream, fully connected, predictor network.

The network receives as inputs the efference as well as the features of the current state.

Building on insights gained from knowledge distillation methods, we choose as our features the outputs of a pre-trained network,  yielding a compressed representation of the current state.

The representation is chosen such that it allows for fast search using classical graph search algorithms.

We display the effectiveness of our approach on a viewpoint-matching task using a modified best-first search algorithm.

As we manipulate an object in our hands, we can accurately predict how it looks after some action is performed.

Through our visual sensory system, we receive high-dimensional information about the object.

However, we do not hallucinate its full-dimensional representation as we estimate how it would look and feel after we act.

But we feel that we understood what happened if there is an agreement between the experience of the event and our predicted experience.

There has been much recent work on methods that take advantage of compact representations of states for search and exploration.

One of the advantages of this approach is that finding a good representation allows for faster and more efficient planning.

This holds in particular when the latent space is of a much lower dimensionality than the one where the states originally live in.

Our central nervous system (CNS) sends a command (efferent) to our motor system, as well as sending a copy of the efferent to our cerebellum, which is our key organ for predicting the sensory outcome of actions when we initiate a movement and is responsible for fine motor control.

The cerebellum then compares the result of the action (sensory reafference) with the intended consequences.

If they differ, then the cerebellum makes changes to its internal structure such that it does a better job next time -i.e., in no uncertain terms, it learns.

The cerebellum receives 40 times more information than it outputs, by a count of the number of axons.

This gives us a sense of the scale of the compression ratio between the high dimensional input and low dimensional output.

Thus, we constrain our attention to planning in a low-dimensional space, without necessarily reconstructing the high-dimensional one.

We apply this insight for reducing the complexity of tasks such that planning in high dimensionality space can be done by classical AI methods in low dimensionality space .

Our contributions are thus twofold: provide a link between efference theory and classical planning with a simple model and introduce a search method for applying the model to reduced state-space search.

We validate our approach experimentally on visual data associated with categorical actions that connect the images, for example taking an object and rotating it.

We create a simple manipulation task using the NORB dataset (LeCun et al., 2004) , where the agent is presented with a starting viewpoint of an object and the task is to produce a sequence of actions such that the agent ends up with the target viewpoint of the object.

As the NORB data set can be embedded on a cylinder (Schüler et al., 2018) (Hadsell et al., 2006) or a sphere (Wang et al., 2018) , we can visualize the actions as traversing the embedded manifold.

We essentially aim at approximating a Markov decision process' state transition function.

Thus, similarities abound in the reinforcement learning literature: many agents perform explicit latentspace planning computations (Tamar et al., 2016) (Srinivas et al., 2018) (Hafner et al., 2018) (Henaff et al., 2017) (Chua et al., 2018) (Gal et al., 2016) as part of learning and executing policies.

Gelada et al. (2019) train a reinforcement learning (RL) agent to simultaneously predict rewards as well as future latent states.

Our work is distinct from these as we are not assuming a reward signal and are not constrained to an RL setting during training.

Similar to our training setup, Oh et al. (2015) predict future frames in ATARI environments conditioned on actions.

The predicted frames are used for learning the dynamics of the environment, e.g. for improving exploration by informing agents of which actions are more likely to result in unseen states.

Our work differs as we are maneuvering within the latent space and not the full input space.

Vision, memory, and controller modules are combined for learning a model of the world before learning a decision model in Ha and Schmidhuber's World Models (Ha & Schmidhuber, 2018) .

A predictive model is trained in an unsupervised manner, allowing the agent to learn policies entirely within its learned latent space representation of the environment.

Instead of training the representation from scratch, we apply the machinery of transfer learning by using pre-trained networks.

This allows the method to be applied to a generic representation rather than a specifically trained representation.

Using the output of pre-trained networks as targets instead of the original pixels is not new.

The case where the output of a larger model is the target for a smaller model is known as knowledge distillation (Hinton et al., 2015) (Buciluǎ et al., 2006) .

This is used for compressing a model ensemble into a single model.

Vondrick et al. (Vondrick et al., 2016 ) learn to make high-level semantic predictions of future frames in video data.

Given a frame at a current time, a neural network is tasked with predicting the representation of a future frame, e.g. by AlexNet.

Our approach extends this general idea by admitting an action as the input to our predictor network as well.

Causal InfoGAN (Kurutach et al., 2018 ) is a method based on generative adversarial networks (GANs) (Goodfellow et al., 2014) , inspired by InfoGAN in particular (Chen et al., 2016) , for learning plannable representations.

A GAN is trained for encoding start and target states and plans a trajectory in the representation space as well as reconstructing intermediate states in the plans.

Our method differ from this by training a simple forward model and forgoing reconstruction, which is unnecessary for planning.

To motivate our approach, we will briefly describe the concept of efference copies from neuroscience.

As we act, our central nervous system (CNS) sends a signal, or an efference, to our peripheral nervous system (PNS).

An additional copy of the efference is created, which is sent to an internal forward model (Jeannerod & Arbib, 2003) .

This model makes a prediction of the sensory outcome and learns by minimizing the errors between its prediction and the actual sensory inputs that are experienced after the action is performed.

The sensory inputs that reach the CNS from the sensory receptors in the PNS are called afference.

They can be exafference, signals that are caused by the environment, or reafference, signals that are caused by ourself acting.

By creating an efference copy and training the forward model, we can tell apart exafference from reafference.

This is how we can be tickled when grass straws are brushed against our feet (exafference), but we can walk barefeet over a field of grass without feeling tickled (reafference).

We assume that the motor system and sensory system are fixed and not necessarily trainable.

The motor system could be the transition function of a Markov decision process.

The sensory system is assumed to be a visual feature extractor -in our experiments, a Laplacian Eigenmap or an intermediate layer of a pre-trained convolutional neural network (CNN) is used.

Pre-trained CNNs can provide powerful, generic descriptors (Sharif Razavian et al., 2014) while eliminating the computational load associated with predicting the pixel values of the full images (Vondrick et al., 2016) .

Gwijde, 2018) .

A motor command, or efference, is issued by the CNS.

A copy of the efference is made and is sent to an internal forward model, which predicts the sensory result of the efference.

The original efference is sent to the motor system to act.

The consequence of the action in the world is observed by the sensory system, and a reafference is created and sent to the CNS.

The forward model is then updated to minimize the discrepancy between predicted and actual reafference.

Suppose we have a feature map φ and a training set of N data points (

, where S t is the state at time step t and a t is an action resulting in a state S t+1 .

We train the predictor f , parameterized by θ, by minimizing the mean squared error loss over f 's parameters:

is our set of training data.

In our experiments, we construct f as a two-stream, fully connected, neural network (Fig. 2) .

Using this predictor we are able to carry out efficient planning in the latent space defined by φ.

By planning we mean that there is a start state S start and a goal state S goal and we are tasked with finding a path between them.

Assuming deterministic dynamics, we output the expected representation after performing the action.

This allows us to formulate planning as a classical AI pathfinding or graph traversal problem.

The environment is turned to a graph by considering each state as a node and each action as an edge.

We used a modified best-first search algorithm with the trained EfferenceNets for our experiments (Algorithm 1).

Each state is associated with a node as well as a score: the distance between its representation and the representation of the goal state.

The edges between nodes are the available actions at each state.

The output of the search is the sequence of actions that corresponds to the path connecting the start node to the proposed solution node, i.e. the node whose representation is the one closest to the goal.

To make the algorithm faster, we only consider paths that do not take us to a state that has already been evaluated, even if there might be a difference in the predictions from going this roundabout way.

That is, if a permutation of the actions in the next path to be considered is already in an evaluated path, it will be skipped.

This is akin to transposition tables used to speed up search in game trees.

This might yield paths with redundancies which can be amended with path-simplifying routines (e.g. take one step forward instead of one step left, one forward then one right).

Selecting the optimal action in each step of a temporally-extended Markov decision process in a task effectively is a hard problem that is solved by different fields in different ways.

For example, in Algorithm 1 Find a plan to reach S result = S j , where j = argmin i≤k ||Ri − φ(S goal )|| and R = (r 0 , . . . , r k ) are the representations of explored states.

Input: S start , S goal , max.

trials m, action set A, feature map φ and EfferenceNet f Output: A plan of actions (a 0 , . . . , a n ) reaching

# add j to the set of checked indices for k ← 0 to m do j = argmin i≤k,i / ∈D ||ri − φ(S goal )|| # take a new state, most similar to the goal for all a ∈ A do r k+1 ← f (rj, a) # try every action from current state R ← R ∪ {r k+1 } # save the estimated representation p k+1 ← pj ∪ a P ← P ∪ {p k+1 } # store path from goal state to current state

# add j to the set of checked indices end for return pj the area of reinforcement learning, it is addressed by estimating the accumulated reward signal for a given behavioral strategy and adapting the strategy to optimize this estimate.

This requires either a large number of actively generated samples or exact knowledge of the environment (in dynamic programming) to find a strategy that behaves optimally.

In this paper, we choose a different approach and utilize a one-step prediction model to allow decisions to be made based on the predicted outcome of a number of one-step lookaheads started from an initial state.

The actions are chosen so that each step greedily maximizes progress towards a known target.

This method, sometimes called hillclimber or best-first search, belongs to the family of informed search algorithms (Nilsson, 2014).

To be more efficient than random or exhaustive search, these kinds of algorithms rely on heuristics to provide sufficient evidence for a good -albeit not necessarily optimal -decision at every time step to reach the goal.

Here we use the Euclidean distance in representation space: An action is preferred if its predicted result is closest to the goal.

The usefulness of this heuristics depends on how well and how coherently the Euclidean distance encodes the actual distance to the goal state in terms of the number of actions.

Our experiments show that an easily attainable general purpose representation, such as a pre-trained VGG16 (Simonyan & Zisserman, 2014) , can already provide sufficient guidance to apply such this heuristic effectively.

One might, however, ask what a particularly suited representation might look like when attainability is ignored.

It would need to take the topological structure of the underlying data manifold into account, such that the Euclidean distance becomes a good proxy for the geodesic distance.

One class of methods that fulfill this are spectral embeddings, such as Laplacian Eigenmaps (LEMs) (Belkin & Niyogi, 2003) .

Since they do not readily allow for out-of-sample embedding, they will only be applied in an in-sample fashion to serve as a control experiment in Section 5.2.1.

In the experiments, we show that our method can be combined with simple, informed graph search (Russell & Norvig, 2016) algorithms to plan trajectories for manipulation tasks (Fig. 3, top row) .

We use the NORB dataset, which contains images of different objects each under 9 different elevations, 36 azimuths, and 6 lighting conditions.

We derive from the dataset an object manipulation environment.

The actions correspond to turning a turntable back and forth by 20

• , moving the camera up or down by 5

• or changing the illumination intensity.

After the EfferenceNet is trained, we apply Algorithm 1 to the viewpoint matching task.

The goal is to find a sequence of actions that transforms the start state to the goal state.

The two states differ in their azimuth and elevation configuration.

Given a feature map φ, we task the EfferenceNet f with predicting φ(S t+1 ) after the action a was performed in the state S t .

We train f by minimizing the mean-squared error between f (φ(S t ), a) and φ(S t+1 ).

The network (Fig. 2 is built with Keras (Chollet et al., 2015) and optimized with Nadam (Dozat, 2016) and converges in two hours on a Tesla P40.

It is a two-stream dense neural network.

Each stream consists of one dense layer followed by a batch normalization (BatchNorm) layer.

The outputs of these streams are then concatenated and passed through 3 dense layers, each one followed by a BatchNorm, and then an output dense layer.

Every dense layer, except the last, is followed by a rectified linear unit (ReLU) activation.

Figure 2 : EfferenceNet architecture.

The network takes as input the representation vector φ(S t ) of the state S t as determined by the feature map φ, as well as the one-hot encoded action a. It outputs the estimated feature vector of the resulting state S t+1 after action a is performed.

Our φ is chosen to be a Laplacian Eigenmap for in-sample search and the second-to-last layer output of VGG16 (Simonyan & Zisserman, 2014) , pre-trained on ImageNet (Deng et al., 2009) , for out-ofsample search.

As the representation made by φ do not change over the course of the training, they can be cached, expediting the training.

Of the 10 car toys in the NORB dataset, we randomly chose 9 for our training set and test on the remaining one.

Embedding a single toy's confgurations in three dimensions using Laplacian Eigenmaps will result in a cylindrical embedding that encodes both, elevation and azimuth angles, as visible in Figure 4 .

Three dimensions are needed so that the cyclic azimuth can be embedded correctly as sin(θ) and cos(θ).

If such a representation is now used to train the EfferenceNet which is subsequently applied in Algorithm 1, one would expect monotonically decreasing distance the closer the prediction comes to the target.

Figure 5 shows that this is the case and that this behavior can be very effectively used for a greedy heuristic.

While the monotinicity is not always exact due to imperfections in the approximate prediction, Figure 5 still qualitatively illustrates a best-case scenario.

The goal and start states are chosen randomly, with the constraint that their distances along each dimension (azimuth and elevation) are uniformly distributed.

As the states are searched, a heat map of similarities is produced (Fig. 3) .

To visualize the performance of the search we plot a histogram (Fig. 6) illustrating the accuracy of the search.

The result looks less accurate with respect to elevation than azimuth, but this is due to the elevation changes being more fine-grained than the azimuth changes, namely by a factor of 4.

The difference between the elevation of the goal and solution viewpoints in Figure 3 left, for example, is hardly perceptible.

If one would scale the histograms by angle and not by bins, the drop-off would be comparable.

The heat maps of the type shown in Figure 3 can be aggregated to reveal basins of attraction during the search.

Each heat map is shifted such that the goal position is at the bottom, middle row (Fig. 7,  a) .

Here it is apparent that the goal and the 180

• flipped (azimuth) version of the goal are attractor states.

This is due to the feature map being sensitive to the rough shape of the object, but being unable to distinguish finer details.

In (Fig. 7, b) we display an aggregate heat map when the agent can alter the lighting conditions as well.

In our work, we focus on learning a transition model.

Doing control after learning the model is an established approach, with the mature field of model-based RL dedicated to it.

This has the advantage of allowing for quick learning of new reward functions, since disentangling reward contingencies from the transition function is helpful when learning for multiple/changing reward functions and allows useful learning when there is no reward available at all.

Thus, it might also be useful in a sparse or late reward setting.

Another advantage of our approach is that it accommodates evaluations of reward trajectories with arbitrary discounts.

Standard RL methods are usually restricted in their optimization problems.

Often, there is a choice between optimizing discounted or undiscounted expected returns.

Simulation/rollout-based planning methods are not restricted in that sense: If you are able to predict reward trajectories, you can (greedily) optimize arbitrary functions of these -possibly allowing for behavior regularization.

For example, the risk-averse portfolio manager can prioritize smooth reward trajectories over volatile ones.

We use a pre-trained network because we believe that a flexible algorithm should be based rather on generic, multi-purpose representions and not on very specific representations.

This contributes to the flexibility of the system.

However, a drawback of using pre-trained networks is that features might be encoded that are irrelevant for the current task.

This has the effect that informed search methods, such as best-first search, are not guaranteed to output the accurate solution in the latent space, as there might be distracting pockets of erroneous local minima.

Our visualizations reveal gradient towards the goal state as well as a visually similar, far away states.

There is variation in the similarities, preventing the planning algorithm from finding the exact goal for every task, sometimes yielding solutions that are the polar-opposites of the goal, w.r.t.

the azimuth.

Pairing the EfferenceNet with a good but generic feature map allows us to perform an accurate search in the latent space of manipulating unseen objects.

This remarkably simple method, inspired by the neurology of the cerebellum, reveals a promising line of future work.

We validate our method by on a viewpoint-matching task derived from the NORB dataset.

In the case of deterministic environments, EfferenceNets calculate features of the current state and action, which in turn define the next state.

This opens up a future direction of research by combining EfferenceNets with successor features (Barreto et al., 2017) .

Furthermore, the study of effective feature maps strikes us as an important factor in this line of work to consider.

We utilize here Laplacian Eigenmaps and pre-trained deep networks.

It is probably possible to improve the performance of the system by end-to-end training but we believe that it is more promising to work on generic multi-purpose representations.

Possible further methods include Slow Feature Analysis (SFA) (Wiskott & Sejnowski, 2002) (Schüler et al., 2018) .

SFA has been previously shown (Sprekeler, 2011) to solve a special case of LEMs while it allows for natural out-of-sample embeddings.

@highlight

We present a neuroscience-inspired method based on neural networks for latent space search
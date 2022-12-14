Neural networks in the brain and in neuromorphic chips confer systems with the ability to perform multiple cognitive tasks.

However, both kinds of networks experience a wide range of physical perturbations, ranging from damage to edges of the network to complete node deletions, that ultimately could lead to network failure.

A critical question is to understand how the computational properties of neural networks change in response to node-damage and whether there exist strategies to repair these networks in order to compensate for performance degradation.

Here, we study the damage-response characteristics of two classes of neural networks, namely multilayer perceptrons (MLPs) and convolutional neural networks (CNNs) trained to classify images from MNIST and CIFAR-10 datasets respectively.

We also propose a new framework to discover efficient repair strategies to rescue damaged neural networks.

The framework involves defining damage and repair operators for dynamically traversing the neural networks loss landscape, with the goal of mapping its salient geometric features.

Using this strategy, we discover features that resemble path-connected attractor sets in the loss landscape.

We also identify that a dynamic recovery scheme, where networks are constantly damaged and repaired, produces a group of networks resilient to damage as it can be quickly rescued.

Broadly, our work shows that we can design fault-tolerant networks by applying on-line retraining consistently during damage for real-time applications in biology and machine learning.

In this paper, inspired by the powerful paradigms introduced by deep learning, we attempt to understand the computational and mathematical 23 principles that impact the ability of neural networks to tolerate damage and be repaired.

We characterize the response of two classes 24 of neural networks, namely multilayer perceptrons (MLP's) and convolutional neural nets (CNN's) to node-damage and propose a new 25 framework that identifies strategies to efficiently rescue damaged networks in a principled fashion.

Our key contribution is the introduction of a framework that conceptualizes damage and repair of networks as operators of a dynamical 27 system in the high-dimensional parameter space of a neural network.

The damage and repair operators are used to dynamically traverse the 28 landscape with the goal of mapping local geometric features [9, 10] (like, fixed points, limit-cycles or point/line-attractors) of the neural 29 networks' loss landscape.

The framework led us to discovering that the iterative application of damage and repair operators results in 30 networks that are highly resilient to node-deletions as well as guides us to uncover the presence of geometric features that resemble a 31 path-connected attractors set, in many respects, in the neural networks' loss landscape.

Attractor-like geometric features in the networks' 32 loss landscape explains why the iterative damage-repair strategy always results in the rescue of damaged networks within a small number of 33 training cycles.

2 Susceptibility of neural networks to damage

The first question we ask in this paper is how do neural networks respond to physical perturbations and how does it affect their functional 36 performance.

We characterize the impact of neural damage on 'cognitive' performance of neural networks by tracking the performance of 37 two classes of artificial neural networks, namely MLPs and CNNs, to deletion of neural units from the network.

The MLPs and CNNs were 38 trained to perform simple cognitive tasks like image classification on MNIST and CIFAR-10 datasets respectively before the networks were 39 perturbed.

To damage a node i in the hidden layer of an MLP or in the fully connected layer of a CNN, we zero all connections between node i and the 41 rest of the network.

And, to damage a node j in the convolutional layer of a CNN, we zero the entire feature map.

In this paper, we are 42 specifically interested in node-damage as our perturbation because of its similarity in phenomena to neuron death in biological networks 43 and node-failures in neuromorphic hardware.

We observe a steep increase in the rate of decline of functional performance as we incrementally delete nodes from either an MLP with 1

hidden layer (Fig-1a ), an MLP with 2 hidden layers (Fig-1b ) or a CNN with 2 convolutional layers, a pooling layer and 2 fully connected 46 layers (Fig-1c) .

We refer to this discrete jump in the rate of decline of performance as a phase transition.

The existence of a phase transition shows that neural nets (MLP's and CNN's) damaged above their respective critical thresholds are not 48 resilient to any further perturbation.

We are interested in deciphering strategies that enable the quick rescue of damaged neural nets and also

want to identify networks that are more resilient to perturbation.

3 Can we rescue these damaged networks?

We ask whether it is fundamentally possible to rescue damaged networks in order to compensate for their performance degradation.

To do The plots in figure-2 show that damaged neural networks can be rescued to regain their original functional performance when re-trained 57 via both strategies 1 and 2.

However, they require a large number of training cycles (epochs) to be effectively rescued ( figure-2c) .

The 58 requirement of a large number of training cycles for the effective rescue of a neural network reduces the feasibility of either strategy as it 59 isn't ideal for both, living neural networks in the brain or artificial networks implemented on neuromorphic hardware to be re-trained for 60 extended periods of time to recover from small damages to its network. 'space' in the networks' loss manifold that contains high performing, more resilient, sparser networks.

As the iterative process of damage and repair always enabled the fast recovery of a damaged network

(irrespective of the number of damaged units), this was surprising to us and we were interested in 74 determining if the loss landscape manifold had 'special' geometric features that enabled this rescue.

To map geometric features of a neural networks' loss landscape, we formally conceptualize the iterative 76 damage-repair paradigm as a dynamical system that involves the application of a damage and repair 77 operator (r) on a neural network (w).

We define w to be a feed-forward neural network with n nodes and N total connections.

Here, w i is the set of connections made by node i with the previous layer in the network.

By definition, w i = ??, if node i is in the first layer.

We also have:

Dim( w i ) = N and w ??? R N To damage a neural network, we define a damage operator D i , that damages node i in the network.

To repair a neural network, we define a rescue operator r {i,j} .

Here {i, j} refers to the set of damaged nodes.

The rescue operator forces the 83 network to descend the loss manifold, while fixing nodes within the set and their connections to zero.

Rescue of the network is achieved by 84 performing a constrained gradient descent on the networks' loss manifold.

where, ?? is the gradient step-size and ???L ??? w k is the gradient of the loss function of the neural network along w k

A damage-repair sequence involves the application of a damage operator followed by a repair operator.

A stochastic damage-repair sequence involves the random sampling of a damage operator from D, followed by the application of an 88 appropriate repair operator (ensuring that gradient descent is performed on remaining undamaged nodes).

We define a random variable D to sample an operator D i from the set of all possible damage operators = {D i : i ??? {1, ..., n}}. An iterative 90 damage-repair sequence is the repeated application of a random damage operator D coupled with a deterministic repair operator r {i,j,k,...} ,

that ensures all damaged nodes maintain a zero edge-weight, while other weights are plastic.

Here, we show the long-hand and short-hand 92 notation for the iterative application of damage-repair operators.

We hypothesize that ??? an open set of networks U , that constitutes an invariant set, where:

For any two points, w 1 and w 2 ????? : [0, 1] ?????? U, such that:

Our numerical results strongly suggests the presence of an invariant, path-connected topological space U in the neural networks' loss 97 manifold.

In our experiments, the invariant, path-connected set is a collection of trained networks, whose image corresponding to the 98 application of a damage and repair operator lies in the same set, visualized by the thick black arc (as shown in figure-4) obtained by tSNE 99 embeddings of the high-dimensional network (w).

We observe that iterative application of the damage-repair operator on a network sampled 100 from U results in a series of networks that belong to the same set U .

This is observed in fig-4b & fig-4d .

The red lines indicate damage of 101 network, while the green lines correspond to repair of damaged networks.

This hints at the possibility that U is an invariant set.

We also 102 interpolated between all pairs of networks sampled from U and observed that all the interpolated networks were present in U as well.

In this paper, we address a pertinent question of how neural networks in the brain, or in engineered systems respond to damage of their units 105 and whether there exists efficient strategies to repair damaged networks.

We observe a phase transition behavior as we incrementally delete 106 nodes from the neural network as the rate of decline of performance steeply increases after crossing a critical number of node deletions.

We discover that damaged networks can be rescued and the iterative damage-rescue strategy produces networks that are highly resilient 108 to perturbations, and can be rescued within a small number of training cycles.

This is enabled by the putative presence of an invariant,

path-connected set in the networks' loss manifold.

Although we have shown numerical results that strongly suggest the presence of invariant 110 sets in the loss manifold, our future work will focus on analytically proving the presence of these topological spaces in the loss manifold,

through the formalization presented in the paper, and the use of the Koopman operator machinery, amongst others.

@highlight

strategy to repair damaged neural networks
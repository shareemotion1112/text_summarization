We introduce an unsupervised structure learning algorithm for deep, feed-forward, neural networks.

We propose a new interpretation for depth and inter-layer connectivity where a hierarchy of independencies in the input distribution is encoded in the network structure.

This results in structures allowing neurons to connect to neurons in any deeper layer skipping intermediate layers.

Moreover, neurons in deeper layers encode low-order (small condition sets) independencies and have a wide scope of the input, whereas neurons in the first layers encode higher-order (larger condition sets) independencies and have a narrower scope.

Thus, the depth of the network is automatically determined---equal to the maximal order of independence in the input distribution, which is the recursion-depth of the algorithm.

The proposed algorithm constructs two main graphical models: 1) a generative latent graph (a deep belief network) learned from data and 2) a deep discriminative graph constructed from the generative latent graph.

We prove that conditional dependencies between the nodes in the learned generative latent graph are preserved in the class-conditional discriminative graph.

Finally, a deep neural network structure is constructed based on the discriminative graph.

We demonstrate on image classification benchmarks that the algorithm replaces the deepest layers (convolutional and dense layers) of common convolutional networks, achieving high classification accuracy, while constructing significantly smaller structures.

The proposed structure learning algorithm requires a small computational cost and runs efficiently on a standard desktop CPU.

Over the last decade, deep neural networks have proven their effectiveness in solving many challenging problems in various domains such as speech recognition BID17 , computer vision BID28 BID16 BID46 and machine translation BID9 .

As compute resources became more available, large scale models having millions of parameters could be trained on massive volumes of data, to achieve state-of-the-art solutions for these high dimensionality problems.

Building these models requires various design choices such as network topology, cost function, optimization technique, and the configuration of related hyper-parameters.

In this paper, we focus on the design of network topology-structure learning.

Generally, exploration of this design space is a time consuming iterative process that requires close supervision by a human expert.

Many studies provide guidelines for design choices such as network depth BID46 , layer width BID55 , building blocks , and connectivity BID20 BID23 .

Based on these guidelines, these studies propose several meta-architectures, trained on huge volumes of data.

These were applied to other tasks by leveraging the representational power of their convolutional layers and fine-tuning their deepest layers for the task at hand BID21 BID33 .

However, these meta-architecture may be unnecessarily large and require large computational power and memory for training and inference.

The problem of model structure learning has been widely researched for many years in the probabilistic graphical models domain.

Specifically, Bayesian networks for density estimation and causal discovery BID42 BID50 .

Two main approaches were studied: score-based (search-and-score) and constraint-based.

Score-based approaches combine a scoring function, such as BDe BID10 and BIC BID44 , with a strategy for searching through the space of structures, such as greedy equivalence search BID6 .

BID1 introduced an algorithm for sampling deep belief networks (generative model) and demonstrated its applicability to high-dimensional image datasets.

Constraint-based approaches BID42 BID50 find the optimal structures in the large sample limit by testing conditional independence (CI) between pairs of variables.

They are generally faster than score-based approaches BID54 ) and have a well-defined stopping criterion (e.g., maximal order of conditional independence).

However, these methods are sensitive to errors in the independence tests, especially in the case of high-order conditional-independence tests and small training sets.

Motivated by these methods, we propose a new interpretation for depth and inter-layer connectivity in deep neural networks.

We derive a structure learning algorithm such that a hierarchy of independencies in the input distribution is encoded in the network structure, where the first layers encode higher-order independencies than deeper layers.

Thus, the number of layers is automatically determined.

Moreover, a neuron in a layer is allowed to connect to neurons in deeper layers skipping intermediate layers.

An example of a learned structure, for MNIST, is given in Figure 1 .We describe our recursive algorithm in two steps.

In Section 2 we describe a base case-a singlelayer structure learning.

In Section 3 we describe multi-layer structure learning by applying the key concepts of the base case, recursively (proofs are provided in Appendix A).

In Section 4 we discuss related work.

We provide experimental results in Section 5, and conclude in Section 6.

DISPLAYFORM0 a set of latent variables, and Y a class variable.

Our algorithm constructs three graphical models and an auxiliary graph.

Each variable is represented by a single node and a single edge may connect two distinct nodes.

Graph G is a generative DAG defined over the observed and latent variables X ∪ H. Graph G Inv is called a stochastic inverse of G. Graph G D is a discriminative model defined over the observed, latent, and class variables X ∪ H ∪ Y .

An auxiliary graph G X is defined over X (a CPDAG; an equivalence class of a Bayesian network) and is generated and maintained as an internal state of the algorithm.

The parents set of a node X in G is denoted P a(X; G).

The order of an independence relation is defined to be the condition set size.

For example, if X 1 and X 2 are independent given X 3 and X 4 , denoted X 1 ⊥ ⊥ X 2 |{X 3 , X 4 }, then the independence order is two.

Figure 1 : An example of a structure learned by our algorithm (classifying MNIST digits).

Neurons in a layer may connect to neurons in any deeper layer.

Depth is determined automatically.

Each gather layer selects a subset of the input, where each input variable is gathered only once.

A neural route, starting with a gather layer, passes through densely connected layers where it may split (copy) and merge (concatenate) with other routes in correspondence with the hierarchy of independencies identified by the algorithm.

All routes merge into the final output layer (e.g., a softmax layer).

We start by describing the key concepts of our approach using a simple scenario: learning the connectivity of a single-layer neural network.

Assume the input joint distribution p(X) complies with the following property.

Assumption 1.

The joint distribution p(X) is faithful to a DAG G over observed X and latent nodes H, where for all X ∈ X and H ∈ H, P a(X; G) ⊆ H and P a(H; G) ⊆ H\H. DISPLAYFORM0 Note that the generative graphical model G can be described as a layered deep belief network where parents of a node in layer m can be in any deeper layer, indexes greater than m, and not restricted to the next layer m + 1.

This differs from the common definition of deep belief networks BID22 BID1 where the parents are restricted to layer m + 1.It is desired to learn an efficient graph G having small sets of parents and a simple factorization of p(H) while maintaining high expressive power.

We first construct an auxiliary graph, a CPDAG BID50 , G X over X (an equivalence class of a fully visible Bayesian network) encoding only marginal independencies 1 (empty condition sets) and then construct G such that it can mimic BID42 .

That is, preserving all conditional dependencies of X in G X .

DISPLAYFORM1 The simplest connected DAG that encodes statistical independence is the v-structure, a structure with three nodes X 1 → X 3 ← X 2 in which X 1 and X 2 are marginally independent X 1 ⊥ ⊥ X 2 and conditionally dependent X 1 ⊥ ⊥X 2 |X 3 .

In graphs encoding only marginal independencies, dependent nodes form a clique.

We follow the procedure described by BID54 and decompose X into autonomous sets (complying with the Markov property) where one set, denoted X D (descendants), is the common child of all other sets, denoted X A1 , . . .

, X AK (ancestor sets).

We select X D to be the set of nodes that have the lowest topological order in G X .

Then, by removing X D from G X (temporarily for this step), the resulting K disjoint sets of nodes (corresponding to K disjoint substructures) form the K ancestor sets DISPLAYFORM2 .

See an example in FIG1 .

Next, G is initialized to an empty graph over X. Then, for each ancestor set X Ai a latent variable H i is introduced and assigned to be a common parent of the pair (X Ai , X D ).

Thus, DISPLAYFORM3 Note that the parents of two ancestor sets are distinct, whereas the parents set of the descendant set is composed of all the latent variables.

In the auxiliary graph G X , for each of the resulting v-structures (X Ai → X D ← X Aj ), a link between a parent and a child can be replaced by a common latent parent without introducing new independencies.

For example, in Algorithm 1 summarizes the procedure of constructing G having a single latent layer.

Note that we do not claim to identify the presence of confounders and their inter-relations as in BID14 ; BID45 ; BID2 .

Instead, we augment a fully observed Bayesian network with latent variables, while preserving conditional dependence.

[a] DISPLAYFORM4 A stochastic inverse generated by the algorithm presented by .[c]

A stochastic inverse generated by our method where the graph is a projection of a latent structure.

A dependency induced by a latent Q is described using a bi-directional edge DISPLAYFORM5 having a class node Y that provides an explaining away relation for H A ↔ H B .

That is, the latent Q is replaced by an observed common child Y .

It is important to note that G represents a generative distribution of X and is constructed in an unsupervised manner (class variable Y is ignored).

Hence, we construct G Inv , a graphical model that preserves all conditional dependencies in G but has a different node ordering in which the observed variables, X, have the highest topological order (parentless)-a stochastic inverse of G. Note that conditional dependencies among X are not required to be preserved in the stochastic inverse as these are treated (simultaneously) as observed variables (highest topological order).

; BID41 presented a heuristic algorithm for constructing such stochastic inverses where the structure is a DAG (an example is given in Figure 3 -[b] ).

However, these DAGs, though preserving all conditional dependencies, may omit many independencies and add new edges between layers.

We avoid limiting G Inv to a DAG and instead limit it to be a projection of another latent structure BID42 .

That is, we assume the presence of additional hidden variables Q that are not in G Inv but induce dependency 2 among H. For clarity, we omit these variables from the graph and use bi-directional edges to represent the dependency induced by them.

An example is given in Figure 3 -[c] where a bi-directional edge represents the effect of some variable Q ∈ Q on H A and H B .

We construct G Inv in two steps:1.

Invert all G edges (invert inter-layer connectivity).2.

Connect each pair of latent variables, sharing a common child in G, with a bi-directional edge.

This simple procedure ensures G G Inv over X ∪ H while maintaining the exact same number of edges between the layers (Proposition 1, Appendix A).

XD ←− nodes having the lowest topological order identify autonomous sets DISPLAYFORM0 set each Hi to be a parent of {XA 1 ∪ XD} connect 12 return G

Recall that G encodes the generative distribution of X and G Inv is the stochastic inverse.

We further construct a discriminative graph G D by replacing bi-directional dependency relations in G Inv , induced by Q, with explaining-away relations by adding the observed class variable Y .

Node Y is set in G D to be the common child of the leaves in G Inv (latents introduced after testing marginal independencies) (see an example in Figure 3 - [d] ).

This preserves the conditional dependency relations of G Inv .

That is, G D can mimic G Inv over X and H given Y (Proposition 2, Appendix A).

It is interesting to note that the generative and discriminative graphs share the exact same inter-layer connectivity (inverted edge-directions).

Moreover, introducing node Y provides an "explaining away" relation between latents, uniquely for the classification task at hand.

We construct a neural network based on the connectivity in G D .

Sigmoid belief networks BID38 have been shown to be powerful neural network density estimators BID29 BID15 .

In these networks, conditional probabilities are defined as logistic regressors.

Similarly, for G D we may define for each latent variable H ∈ H, DISPLAYFORM0 where sigm(x) = 1/(1 + exp(−x)), X = P a(H ; G D ), and (W , b ) are the parameters of the neural network.

BID37 proposed replacing each binary stochastic node H by an infinite number of copies having the same weights but with decreasing bias offsets by one.

They showed that this infinite set can be approximated by DISPLAYFORM1 where v = W X + b .

They further approximate this function by max(0, v + ) where is a zerocentered Gaussian noise.

Following these approximations, they provide an approximate probabilistic interpretation for the ReLU function, max(0, v).

As demonstrated by BID25 and BID37 , these units are able to learn better features for object classification in images.

In order to further increase the representational power, we represent each H by a set of neurons having ReLU activation functions.

That is, each latent variable H in G D is represented in the neural network by a dense (fully-connected) layer.

Finally, the class node Y is represented by a softmax layer.

We now extend the method of learning the connectivity of a single layer into a method of learning multi-layered structures.

The key idea is to recursively introduce a new and deeper latent layer by testing n-th order conditional independence (n is the condition set size) and connect it to latent layers created by previous recursive calls that tested conditional independence of order n + 1.

The method is described in Algorithm 2.

It is important to note that conditional independence is tested only between input variables X and condition sets do not include latent variables.

Conditioning on latent variables or testing independence between them is not required as the algorithm adds these latent variables in a specific manner, preserving conditional dependencies between the input variables.

Algorithm 2: Recursive Latent Structure Learning (multi-layer)1 RecurLatStruct (GX , X, Xex, n) Input: an initial DAG GX over observed X & exogenous nodes Xex and a desired resolution n. Output: G, a latent structure over X and H 2 if the maximal indegree of GX (X) is below n + 1 then exit condition DISPLAYFORM0 to be a parent of {HA The algorithm maintains and recursively updates an auxiliary graph G X (a CPDAG) over X and utilizes it to construct G. BID54 introduced an efficient algorithm (RAI) for constructing a CPDAG over X by a recursive application of conditional independence tests with increasing condition set sizes (n).

Our algorithm is based on this framework for updating the auxiliary graph G X (Algorithm 2, lines 5 and 6).The algorithm starts with n = 0, G X a complete graph, and a set of exogenous nodes X ex = ∅. The set X ex is exogenous to G X and consists of parents of X.The function IncreaseResolution (Algorithm 2-line 5) disconnects (in G X ) conditionally independent variables in two steps.

First, it tests dependency between X ex and X, i.e., X ⊥ ⊥ X |S for every connected pair X ∈ X and X ∈ X ex given a condition set S ⊂ {X ex ∪ X} of size n. Next, it tests dependency within X, i.e., X i ⊥ ⊥ X j |S for every connected pair X i , X j ∈ X given a condition set S ⊂ {X ex ∪ X} of size n. After removing the corresponding edges, the remaining edges are directed by applying two rules BID42 BID50 .

First, v-structures are identified and directed.

Then, edges are continually directed, by avoiding the creation of new v-structures and directed cycles, until no more edges can be directed.

Following the terminology of BID54 , we say that this function increases the graph d-separation resolution from n − 1 to n.

The function SplitAutonomous (Algorithm 2-line 6) identifies autonomous sets in a graph in two steps, as described in Algorithm 1 lines 7 and 8.

An autonomous set in G X includes all its nodes' parents (complying with the Markov property) and therefore a corresponding latent structure can be constructed independently using a recursive call.

Thus, the algorithm is recursively and independently called for the ancestor sets (Algorithm 2 lines 7-8), and then called for the descendant set while treating the ancestor sets as exogenous (Algorithm 2 line 9).[a]

Each recursive call returns a latent structure for each autonomous set.

Recall that each latent structure encodes a generative distribution over the observed variables where layer H (n+1) , the last added layer (parentless nodes), is a representation of the input X ⊂ X. By considering only layer H (n+1) of each latent structure, we have the same simple scenario discussed in Section 2-learning the connectivity between H (n) , a new latent layer, and H (n+1) , treated as an "input" layer.

Thus, latent variables are introduced as parents of the H (n+1) layers, as described in Algorithm 2 lines 11-13.

A simplified example is given in Figure 4 .Next, a stochastic inverse G Inv is constructed as described in Section 2-all the edge directions are inverted and bi-directional edges are added between every pair of latents sharing a common child in G. An example graph G and a corresponding stochastic inverse G Inv are given in Figure 5 .

A discriminative structure G D is then constructed by removing all the bi-directional edges and adding the class node Y as a common child of layer H (0) , the last latent layer that is added ( Figure 5-[c] ).

Finally, a neural network is constructed based on the connectivity of G D .

That is, each latent node, H ∈ H (n) , is replaced by a set of neurons, and each edge between two latents, H ∈ H (n) and DISPLAYFORM1 , is replaced by a bipartite graph connecting the neurons corresponding to H and H .

Recent studies have focused on automating the exploration of the design space, posing it as a hyperparameter optimization problem and proposing various approaches to solve it.

BID34 learns the topology of an RNN network introducing structural parameters into the model and optimize them along with the model weights by the common gradient descent methods.

BID47 takes a similar approach incorporating the structure learning into the parameter learning scheme, gradually growing the network up to a maximum size.

A common approach is to define the design space in a way that enables a feasible exploration process and design an effective method for exploring it.

BID56 (NAS) first define a set of hyper-parameters characterizing a layer (number of filters, kernel size, stride).

Then they use a controller-RNN for finding the optimal sequence of layer configurations for a "trainee network".

This is done using policy gradients (REINFORCE) for optimizing the objective function that is based on the accuracy achieved by the "trainee" on a validation set.

Although this work demonstrates capabilities to solve large-scale problems (Imagenet), it comes with huge computational cost.

In a following work, BID57 address the same problem but apply a hierarchical approach.

They use NAS to design network modules on a small-scale dataset (CIFAR-10) and transfer this knowledge to a large-scale problem by learning the optimal topology composed of these modules.

BID3 use reinforcement learning as well and apply Q-learning with epsilon-greedy exploration strategy and experience replay.

BID39 propose a language that allows a human expert to compactly represent a complex search-space over architectures and hyperparameters as a tree and then use methods such as MCTS or SMBO to traverse this tree.

Smithson et al. FORMULA1 present a multi objective design space exploration, taking into account not only the classification accuracy but also the computational cost.

In order to reduce the cost involved in evaluating the network's accuracy, they train a Response Surface Model that predicts the accuracy at much lower cost, reducing the number of candidates that go through actual validation accuracy evaluation.

Another common approach for architecture search is based on evolutionary strategies to define and search the design space.

BID43 BID35 use evolutionary algorithm to evolve an initial model or blueprint based on its validation performance.

Common to all these recent studies is the fact that structure learning is done in a supervised manner, eventually learning a discriminative model.

Moreoever, these approaches require huge compute resources, rendering the solution unfeasible for most applications given limited compute and time resources.

We evaluate the quality of the learned structure in two experiments:• Classification accuracy as a function of network depth and size for a structure learned directly from MNIST pixels.• Classification accuracy as a function of network size on a range of benchmarks and compared to common topologies.

All the experiments were repeated five times where average and standard deviation of the classification accuracy were recorded.

In all of our experiments, we used a ReLU function for activation, ADAM BID26 for optimization, and applied batch normalization BID24 followed by dropout BID51 to all the dense layers.

All optimization hyperparameters that were tuned for the vanilla topologies were also used, without additional tuning, for the learned structures.

For the learned structures, all layers were allocated an equal number of neurons.

Threshold for independence tests, and the number of neurons-per-layer were selected by using a validation set.

Only test-set accuracy is reported.

Our structure learning algorithm was implemented using the Bayesian network toolbox BID36 and Matlab.

We used Torch7 BID8 and Keras BID7 with the TensorFlow BID0 back-end for optimizing the parameters of both the vanilla and learned structures.

We analyze the accuracy of structures learned by our algorithm as a function of the number of layers and parameters.

Although network depth is automatically determined by the algorithm, it is implicitly controlled by the threshold used to test conditional independence (partial-correlation test in our experiments).

For example, a high threshold may cause detection of many independencies leading to early termination of the algorithm and a shallow network (a low threshold has the opposite effect).

Thus, four different networks having 2, 3, 4, and 5 layers, using four different thresholds, are learned for MNIST.

We also select three configurations of network sizes: a baseline (normalized to 1.00), and two configurations in which the number of parameters is 0.5, and 0.375 of the baseline network (equal number of neurons are allocated for each layer).Classification accuracies are summarized in TAB0 .

When the number of neurons-per-layers is large enough (100%) a 3-layer network achieves the highest classification accuracy of 99.07% (standard deviation is 0.01) where a 2-layer dense network has only a slight degradation in accuracy, 99.04%.

For comparison, networks with 2 and 3 fully connected layers (structure is not learned) with similar number of parameters achieve 98.4% and 98.75%, respectively.

This demonstrates the efficiency of our algorithm when learning a structure having a small number of layers.

In addition, for a smaller neuron allocation (50%), deeper structures learned by our algorithm have higher accuracy than shallower ones.

However, a decrease in the neurons-per-layer allocation has a greater impact on accuracy for deeper structures.

MNIST images as a function of network depth and number of parameters (normalized).

For comparison, when a structure is not learned, networks with 2 and 3 dense layers, achieve 98.4% and 98.75% accuracy, respectively (having the same size as learned structures at configuration "100%").

We evaluate the quality of learned structures using five image classification benchmarks.

We compare the learned structures to common topologies (and simpler hand-crafted structures), which we call "vanilla topologies", with respect to network size and classification accuracy.

The benchmarks and vanilla topologies are described in Table 2 .

In preliminary experiments we found that, for SVHN and ImageNet, a small subset of the training data is sufficient for learning the structure (larger training set did not improve classification accuracy).

As a result, for SVHN only the basic training data is used (without the extra data), i.e., 13% of the available training data, and for ImageNet 5% of the training data is used.

Parameters were optimized using all of the training data.

Table 2 : Benchmarks and vanilla topologies.

MNIST-Man and SVHN-Man topologies were manually created by us.

MNIST-Man has two convolutional layer (32 and 64 filters each) and one dense layer with 128 neurons.

SVHN-Man was created as a small network reference having reasonable accuracy compared to Maxout-NiN. In the first row we indicate that in one experiment a structure for MNIST was learned from the pixels and feature extracting convolutional layers were not used.

Convolutional layers are powerful feature extractors for images exploiting domain knowledge, such as spatial smoothness, translational invariance, and symmetry.

We therefore evaluate our algorithm by using the first convolutional layers of the vanilla topologies as "feature extractors" (mostly below 50% of the vanilla network size) and learning a deep structure from their output.

That is, the deepest layers of the vanilla network (mostly over 50% of the network size) is removed and replaced by a structure learned by our algorithm in an unsupervised manner.

Finally, a softmax layer is added and the entire network parameters are optimized.

First, we demonstrate the effect of replacing a different amount of the deepest layers and the ability of the learned structure to replace feature extraction layers.

Table 3 describes classification accuracy achieved by replacing a different amount of the deepest layers in VGG-16.

For example, column "conv.10" represents learning a structure using the activations of conv.10 layer.

Accuracy and the normalized number of network parameters are reported for the overall network, e.g., up to conv.10 + the learned structure.

Column "vanilla" is the accuracy achieved by the VGG-16 network, after training under the exact same setting (a setting we found to maximize a validation-set accuracy for the vanilla topologies).

Table 3 : Classification accuracy (%) and overall network size (normalized number of parameters).

VGG-16 is the "vanilla" topology.

For both, CIFAR 10/100 benchmarks, the learned structure achieves the highest accuracy by replacing all the layers that are deeper than layer conv.10.

Moreover, accuracy is maintained when replacing the layers deeper than layer conv.7.One interesting phenomenon to note is that the highest accuracy is achieved at conv.

10 rather than at the "classifier" (the last dense layer).

This might imply that although convolutional layers are useful at extracting features directly from images, they might be redundant for deeper layers.

By using our structure learning algorithm to learn the deeper layers, accuracy of the overall structure increases with the benefit of having a compact network.

An accuracy, similar to that of "vanilla" VGG-16, is achieved with a structure having 85% less total parameters (conv.

7) than the vanilla network, where the learned structure is over 50X smaller than the replaced part.

Next, we evaluate the accuracy of the learned structure as a function of the number of parameters and compare it to a densely connected network (fully connected layers) having the same depth and size.

For SVHN, we used the Batch Normalized Maxout Network in Network topology BID4 and removed the deepest layers starting from the output of the second NiN block (MMLP-2-2).

For CIFAR-10, we used the VGG-16 and removed the deepest layers starting from the output of conv.10 layer.

For MNIST, a structure was learned directly from pixels.

Results are depicted in FIG6 .

It is evident that accuracy of the learned structures is significantly higher (error bars represent 2 standard deviations) than a set of fully connected layers, especially in cases where the network is limited to a small number of parameters.[a] Finally, in Table 4 we provide a summary of network sizes and classification accuracies, achieved by replacing the deepest layers of common topologies (vanilla) with a learned structure.

In the first row, a structure is learned directly from images; therefore, it does not have a "vanilla" topology as reference (a network with 3 fully-connected layers having similar size achieves 98.75% accuracy).

In all the cases, the size of the learned structure is significantly smaller than the vanilla topology, and generally has an increase in accuracy.

Comparison to other methods.

Our structure learning algorithm runs efficiently on a standard desktop CPU, while providing structures with competitive classification accuracies and network sizes.

For example, the lowest classification error rate achieved by our unsupervised algorithm for CIFAR 10 is 4.58% with a network of size 6M (WRN-40-4 row in Table 4 ).

For comparison, the NAS algorithm BID56 achieves error rates of 5.5% and 4.47% for networks of sizes 4.2M and 7.1M, respectively, and requires optimizing thousands of networks using hundreds of GPUs.

For AlexNet network, recent methods for reducing the size of a pre-trained network (pruning while maintaining classification accuracy) achieve 5× (Denton et al., 2014) and 9× BID18 BID34 Table 4 : A summary of network sizes and classification accuracies (and standard deviations), achieved by replacing the deepest layers of common topologies (vanilla) with a learned structure.

The number of parameters are reported for "feature extraction" (first layers of the vanilla topology), removed section (the deepest layers of the vanilla topology), and the learned structure that replaced the removed part.

The sum of parameters in the "feature extraction" and removed parts equals to the vanilla topology size.

The first row corresponds to learning a structure directly from image pixels.

We presented a principled approach for learning the structure of deep neural networks.

Our proposed algorithm learns in an unsupervised manner and requires small computational cost.

The resulting structures encode a hierarchy of independencies in the input distribution, where a node in one layer may connect another node in any deeper layer, and depth is determined automatically.

We demonstrated that our algorithm learns small structures, and maintains high classification accuracies for common image classification benchmarks.

It is also demonstrated that while convolution layers are very useful at exploiting domain knowledge, such as spatial smoothness, translational invariance, and symmetry, they are mostly outperformed by a learned structure for the deeper layers.

Moreover, while the use of common topologies (meta-architectures), for a variety of classification tasks is computationally inefficient, we would expect our approach to learn smaller and more accurate networks for each classification task, uniquely.

As only unlabeled data is required for learning the structure, we expect our approach to be practical for many domains, beyond image classification, such as knowledge discovery, and plan to explore the interpretability of the learned structures.

@highlight

A principled approach for structure learning of deep neural networks with a new interpretation for depth and inter-layer connectivity. 
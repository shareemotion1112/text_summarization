Graph networks have recently attracted considerable interest, and in particular in the context of semi-supervised learning.

These methods typically work by generating node representations that are propagated throughout a given weighted graph.



Here we argue that for semi-supervised learning, it is more natural to consider propagating labels in the graph instead.

Towards this end, we propose a differentiable neural version of the classic Label Propagation (LP) algorithm.

This formulation can be used for learning edge weights, unlike other methods where weights are set heuristically.

Starting from a layer implementing a single iteration of LP, we proceed by adding several important non-linear steps that significantly enhance the label-propagating mechanism.



Experiments in two distinct settings demonstrate the utility of our approach.

We study the problem of graph-based semi-supervised learning (SSL), where the goal is to correctly label all nodes of a graph, of which only a few are labeled.

Methods for this problem are often based on assumptions regarding the relation between the graph and the predicted labels.

One such assumption is smoothness, which states that adjacent nodes are likely to have similar labels.

Smoothness can be encouraged by optimizing an objective where a loss term L over the labeled nodes is augmented with a quadratic penalty over edges: (1) Here, y are the true labels, f are "soft" label predictions, S is the set of labeled nodes, and w are non-negative edge weights.

The quadratic term in Eq. (1) is often referred to as Laplacian Regularization since (for directed graphs) it can equivalently be expressed using the graph Laplacian BID5 .

Many early methods for SSL have adopted the general form of Eq. (1) BID51 BID50 BID4 BID6 BID0 BID42 BID47 .

Algorithms such as the seminal Label Propagation BID51 are simple, efficient, and theoretically grounded but are limited in two important ways.

First, predictions are parameterized either naïvely or not at all.

Second, edge weights are assumed to be given as input, and in practice are often set heuristically.

Recent deep learning methods address the first point by offering intricate predictive models that are trained discriminatively BID47 BID38 BID48 BID28 BID20 BID21 BID34 .

Nonetheless, many of them still require w as input, which may be surprising given the large body of work highlighting the importance of good weights BID51 BID24 BID46 BID4 BID25 .

While some methods consider some form of weight learning BID45 BID35 , to some extent they have drifted away from the original quadratic criterion.

Other works address the second point by proposing disciplined ways for learning w. However, these either assume specific simple parameterizations BID49 BID25 , or altogether consider weights disjointly from predictions BID46 BID32 .Our goal in this paper is to simultaneously addresses both issues.

We propose a framework that, given a graph, jointly learns both a parametric predictive model and the edge weights.

To do this, we begin by revisiting the Label Propagation (LP), and casting it as a differentiable neural network.

Each layer in the network corresponds to a single iterative update, making a forward pass equivalent to a full run of the algorithm.

Since the network is differentiable, we can then optimize the weights of the LP solution using gradient descent.

As we show, this can be done efficiently with a suitable loss function.

The key modeling point in our work is that labeled information is used as input to both the loss and the network.

In contrast to most current methods, our network's hidden layers directly propagate labeling information, rather than node or feature representations.

Each layer is therefore a self-map over the probability simplex; special care is therefore needed when introducing non-linearities.

To this end, we introduce two novel architectural components that are explicitly designed to operate on distributions.

The first is an information-gated attention mechanism, where attention is directed based on the informativeness and similarity of neighboring nodes' states.

The second is a novel "bifurcation" operator that dynamically controls label convergence, and acts as a balancing factor to the model's depth.

Our main guideline in designing our model was to tailor it to the semi-supervised setting.

The result is a slim model having relatively few parameters and only one model-specific hyper-parameter (depth), making it suitable for tasks where only few labeled nodes are available.

The final network provides a powerful generalization of the original propagation algorithm that can be trained efficiently.

Experiments on benchmark datasets in two distinct learning settings show that our model compares favorably against strong baselines.

Many SSL methods are based on Eq. (1) or on similar quadratic forms.

These differ in their assumed input, the optimization objective, and the parametric form of predictions.

Classic methods such as LP BID51 assume no parametric form for predictions, and require edge weights as inputs.

When node features are available, weights are often set heuristically based on some similarity measure (e.g., w ij = exp x i − x j 2 2 /σ 2 ).

LP constrains predictions on S to agree with their true labels.

Other propagation methods relax this assumption BID4 BID6 , add regularization terms BID0 , or use other Laplacian forms BID50 .Some methods aim to learn edge weights, but do not directly optimize for accuracy.

Instead, they either model the relations between the graph and features BID46 BID32 or simply require f as input BID14 BID23 BID17 .

Methods that focus on accuracy are often constrained to specific parameterizations or assumptions BID25 .

BID49 optimize the leave-one-out loss (as we do), but require a series of costly matrix inversions.

Several recent works in deep learning have been focused on graph inputs in general BID2 and specifically for inductive SSL.

The main idea behind these methods is to utilize a weighted graph to create meaningful vector representations of nodes, which are then fed into a classifier.

Methods are typically designed for one of two settings: when the input includes only a graph, and when node features are available.

When the input includes only a graph (and no features), node representations are generated using embedding techniques.

BID38 use a SkipGram model over random walks on the graph, which are used to define context.

BID20 further this idea by introducing expressive parameterized random walks, while BID43 focus on optimizing similarities between pairs of node embeddings.

Various methods have been proposed to utilize node features, when available.

Spectral methods, stemming from a CNN formulation for graphs BID11 , include different approximations of spectral graph convolutions BID16 BID28 adaptive convolution filters BID34 , or attention mechanisms BID45 Embedding approaches have been suggesting for handling bag-of-words representations BID48 and general node attributes such as text or continuous features BID18 .

Many of the above methods can be thought of as propagating features over the graph in various forms.

Our method, in contrast, propagates labels.

The main advantage label propagation is that labeled information is used not only to penalize predictions (in the loss), but also to generate predictions.

We begin by describing the learning setup and introducing notation.

The input includes a (possibly directed) graph G = (V, E), for which a subset of nodes S ⊂ V are labeled by y S = {y i } i∈S with y i ∈ {1, . . .

, C}. We refer to S as the "seed" set, and denote the unlabeled nodes by U = V \ S, and the set of i's (incoming) neighbors by N i = {j : (j, i) ∈ E}. We use n = |V |, m = |E|, = |S|, and u = |U | so that n = + u. In a typical task, we expect to be much smaller than n.

We focus on the transductive setting where the goal is to predict the labels of all i ∈ U .

Most methods (as well as ours) output "soft" labels f i ∈ ∆ C , where ∆ C is the C-dimensional probability simplex.

For convenience we treat "hard" labels y i as one-hot vectors in ∆ C .

All predictions are encoded as a matrix f with entries f ic = P[y i = c].

For any matrix M , we will use M A to denote the sub-matrix with rows corresponding to A. Under this notation, given G, S, y S , and possibly x, our goal is to predict soft labels f U that match y U .In some cases, the input may also include features for all nodes x = {x i } i∈V .

Importantly, however, we do not assume the input includes edge weights w = {w e } e∈E , nor do we construct these from x. We denote by W the weighted adjacency matrix of w, and useW andw for the respective (row)-normalized weights.

Many semi-supervised methods are based on the notion that predictions should be smooth across edges.

A popular way to encourage such smoothness is to optimize a (weighted) quadratic objective.

Intuitively, the objective encourages the predictions of all adjacent nodes to be similar.

There are many variations on this idea; here we adopt the formulation of BID51 where predictions are set to minimize a quadratic term subject to an agreement constraint on the labeled nodes: DISPLAYFORM0 In typical applications, w is assumed to be given as input.

In contrast, our goal here is to learn them in a discriminative manner.

A naïve approach would be to directly minimize the empirical loss.

For a loss function L, regularization term R, and regularization constant λ, the objective would be: DISPLAYFORM1 While appealing, this approach introduces two main difficulties.

First, f * is in itself the solution to an optimization problem (Eq. (2)), and so optimizing Eq. FORMULA1 is not straightforward.

Second, the constraints in Eq. (2) ensure that fIn what follows, we describe how to overcome these issues.

We begin by showing that a simple algorithm for approximating f * can be cast as a deep neural network.

Under this view, the weights (as well as the algorithm itself) can be parametrized and optimized using gradient descent.

We then propose a loss function suited to SSL, and show how the above network can be trained efficiently with it.

Recall that we would like to learn f * (w; S).

When w is symmetric, the objective in Eq. FORMULA10 is convex and has a closed form solution.

This solution, however, requires the inversion of a large matrix, which can be costly, does not preserve sparsity, and is non-trivial to optimize.

The LP algorithm BID51 circumvents this issue by approximating f * using simple iterative averaging updates.

Let f (t) be the set of soft labels at iteration t andw ij = w ij / k w ik , then for the following recursive relation: DISPLAYFORM0 it holds that lim t→∞ f (t) = f * for any initial f (0) BID51 .

In practice, the iterative algorithm is run up to some iteration T , and predictions are given using f (T ) .

This dynamic process can be thought of as labels propagating over the graph from labeled to unlabeled nodes over time.

Motivated by the above, the idea behind our method is to directly learn weights for f (T ) , rather than for f * .

In other words, instead of optimizing the quadratic solution, our goal is to learn weights under which LP preforms well.

This is achieved by first designing a neural architecture whose layers correspond to an "unrolling" of the iterative updates in Eq. (4), which we describe next.

The main building block of our model is the basic label-propagation layer, which takes in two main inputs: a set of (predicted) soft labels h = {h i } n i=1 for all nodes, and the set of true labels y A for some A ⊆ S. For clarity we use A = S throughout this section.

As output, the layer produces a new set of soft labels h = {h i } n i=1 for all nodes.

Note that both h i and h i are in ∆ C .

The layer's functional form borrows from the LP update rule in Eq. (4) where unlabeled nodes are assigned the weighted-average values of their neighbors, and labeled nodes are fixed to their true labels.

For a given w, the output is: DISPLAYFORM0 whereW is the row-normalized matrix of w. A basic network is obtained by composing T identical layers: DISPLAYFORM1 where the model's parameters w are shared across layers, and the depth T is the model's only hyper-parameter.

The input layer h (0) is initialized to y i for each i ∈ S and to some prior ρ i (e.g., uniform) for each i ∈ U .

Since each layer h (t) acts as a single iterative update, a forward pass unrolls the full algorithm, and hence H can be thought of as a parametrized and differentiable form of the LP algorithm.

In practice, rather than directly parameterizing H by w, it may be useful to use more sophisticated forms of parameterization.

We will denote such general networks by H(θ), where θ are learned parameters.

As a first step, given edge features {φ e } e∈E , we can further parametrize w using linear scores s ij and normalizing with softmax: DISPLAYFORM2 where θ φ ∈ R d are learned parameters.

We propose using three types of features (detailed in Appendix B): Node-feature similarities: when available, node features can be used to define edge features by incorporating various similarity measures such as cosine similarity (φ ij = x i x j / x i x j ) and Gaussian similarity DISPLAYFORM3 DISPLAYFORM4 where each similarity measure induces a distinct feature.

Graph measures: these include graph properties such as node attributes (e.g., source-node degree), edge centrality measures (e.g., edge betweenness), path-ensemble features (e.g., Katz distance), and graph-partitions (e.g., k-cores).

These allow generalization across nodes based on local and global edge properties.

Seed relations: relations between the edge (i, j) and nodes in S, such the minimal (unweighted) distance to i from some s ∈ S. Since closer nodes are more likely to be labeled correctly, these features are used to quantify the reliability of nodes as sources of information, and can be class-specific.

The label propagation layers in H pass distributions rather than node feature representations.

It is important to take this into account when adding non-linearities.

We therefore introduce two novel components that are explicitly designed to handle distributions, and can be used to generalize the basic layer in Eq. (5) The general layer (illustrated in FIG1 ) replaces weights and inputs with functions of the previous layer's output: DISPLAYFORM0 whereÃ(·) is a normalized weight matrix (replacingW ), µ(·) is a soft-label matrix (replacing h (t) ), and θ α and θ τ are corresponding learned parameters.

The edge-weight functionÃ offers an information-gated attention mechanism that dynamically allocates weights according to the "states" of a node and its neighbors.

The labeling function µ is a time-dependent bifurcation mechanism which controls the rate of label convergence.

We next describe our choice ofÃ and µ in detail.

The LP update (Eq. (4)) uses fixed weights w.

The importance of a neighbor j is hence predetermined, and is the same regardless of, for instance, whether h j is close to some y, or close to uniform.

Here we propose to relax this constraint and allow weights to change over time.

Thinking of h (t) i as the state of i at time t, we replace w ij with dynamic weights a (t) ij that depend on the states of i and j through an attention mechanism α: DISPLAYFORM0 where θ α are the attention parameters.

Ã in Eq. FORMULA8 is the corresponding row-normalized weight matrix. determined by e and d (boxed bars), which are computed using h (t) and θ.

Here θ directs attention at the informative and similar neighbor (thick arrow), and the update amplifies the value of the correct label. (Right) The bifurcation mechanism for C = 3 and various τ .

Arrows map each h ∈ ∆ C to bif(h) ∈ ∆ C .

DISPLAYFORM1 When designing α, one should take into account the nature of its inputs.

Since both h i and h j are label distributions, we have found it useful to let α depend on information theoretic measures and relations.

We use negative entropy e to quantify the certainty of a label, and negative KL-divergence d to measure cross-label similarity.

Both are parameterized by respective class-dependent weights θ e c and θ d c , which are learned: DISPLAYFORM2 where: DISPLAYFORM3 In a typical setting, unlabeled nodes start out with uniform labels, making the overall entropy high.

As distributions pass through the layers, labeled information propagates, and both entropy and divergence change.

The attention of node i is then directed according to the informativeness (entropy) and similarity (divergence) of the states of its neighbors.

As we show in the experiments (Sec. 5), this is especially useful when the data does not include node features (from which weights are typically derived).

FIG2 (left) exemplifies this.

Although the updates in Eq. (4) converge for any w, this can be slow.

Even with many updates, predictions are often close to uniform and thus sensitive to noise BID39 .

One effective solution is to dynamically bootstrap confident predictions as hard labels BID29 BID19 .

This process speeds up the rate of convergence by decreasing the entropy of low-entropy labels.

Here we generalize this idea, and propose a flexible bifurcation mechanism.

This mechanism allows for dynamically increasing or decreasing the entropy of labels.

For node i and some τ ∈ R, h ic is replaced with: DISPLAYFORM0 Note that since h i ∈ ∆ C , this definition ensures that, for any τ , we have that µ(h i ) ∈ ∆ C as well.

When τ > 1 and as τ increases, entropy decreases, and confident labels are amplified.

In contrast, when 0 < τ < 1 and as approaches 0, entropy decreases, and labels become uniform.

For τ < 0 the effects are reversed, and setting τ = 1 gives µ(h i ) = h i , showing that Eq. FORMULA8 DISPLAYFORM1 Thus, when θ τ = 0, Eq. FORMULA1

Recall that our goal is to learn the parameters θ of the network H(θ; S).

Note that by Eq. (5), for all i ∈ S it holds that H i (θ; S) = y i .

In other words, as in LP, predictions for all labeled nodes are constrained to their true value.

Due to this, the standard empirical loss becomes degenerate, as it penalizes H i (θ; S) only according to y i , and the loss becomes zero for all i ∈ S and for any choice of θ.

As an alternative, we propose to follow BID49 and minimize the leave-one-out loss: DISPLAYFORM0 where S −i = S \ {i}, L is a loss function, R is a regularization term with coefficient λ, and θ contains all model parameters (such as θ φ , θ α , and θ τ ).

Here, each true label y i is compared to the model's prediction given all labeled points except i. Thus, the model is encouraged to propagate the labels of all nodes but one in a way which is consistent with the held-out node.

In practice we have found it useful to weight examples in the loss by the inverse class ratio (estimated on S).The leave-one-out loss is a well-studied un-biased estimator of the expected loss with strong generalization guarantees BID26 BID8 BID22 .

In general settings, training the model on all sets {S −i } i∈S introduces a significant computational overhead.

However, in SSL, when is sufficiently small, this becomes feasible BID49 .

For larger values of , a possible solution is to instead minimize the leave-k-out loss, using any number of sets with k randomly removed examples.

When λ is small, θ is unconstrained, and the model can easily overfit.

Intuitively, this can happen when only a small subset of edges is sufficient for correctly propagating labels within S. This should result in noisy labels for all nodes in U .

In the other extreme, when λ is large, w approaches 0, and by Eq. (15)

w is uniform.

The current graph-SSL literature includes two distinct evaluation settings: one where the input includes a graph and node features, and one where a graph is available but features are not.

We evaluate our method in both settings, which we refer to as the "features setting" and "no-features setting", respectively.

We use benchmark datasets that include real networked data (citation networks, social networks, product networks, etc.).

Our evaluation scheme follows the standard SSL setup 1 BID51 BID50 BID12 BID49 BID38 BID20 BID43 BID34 .

First, we sample k labeled nodes uniformly at random, and ensure at least one per class.

Each method then uses the input (graph, labeled set, and features when available) to generate soft labels for all remaining nodes.

Hard labels are set using argmax.

We repeat this for 10 random splits using k = 1% labeled nodes.

For further details please see Appendices A and C.For each experimental setting we use different Label Propagation Network (LPN) variant that differ in how edge weights are determined.

Both variants use bifurcation with linear time-dependency (Sec. 3.2), and include a-symmetric bi-directional edge weights.

In all tasks, LPN was initialized to simulate vanilla LP with uniform weights by setting θ = 0.

Hence, we expect the learned model deviate from LP (by utilizing edge features, attention, or bifurcation) only if this results in more accurate predictions.

We choose T ∈ {10, 20, . . .

, 100} by running cross-validation on LP rather than LPN.

This process does not require learning and so is extremely fast, and due to bifurcation, quite robust (see FIG5 ).

For training we use a class-balanced cross-entropy loss with 2 regularization, and set λ by 5-fold cross-validation.

We optimize with Adam (Kingma & Ba, 2014) using a learning rate of 0.01.

We use all relevant datasets from the LINQS collection BID41 .

These include three citation graphs, where nodes are papers, edges link citing papers, and features are bag-of-words.

As described in Sec. 2.2, for this setting we use a model (LPN φ ) where w is parameterized using a linear function of roughly 30 edge features (φ).

These are based on the given "raw" node features, the graph, and the labeled set.

See Appendix B for more details.

Baselines include LP BID51 with uniform (LP U ) and RBF (LP RBF ) weights, the LP variant ADSORPTION BID0 , ICA BID33 , Graph Convolutional Networks (GCN, Kipf & Welling (2016)), and Graph Attention Networks (GAT, BID45 ).

2 We also add a features-only baseline (RIDGEREG) and a graph-only baseline (NODE2VEC).

We use the FLIP collection BID40 , which includes several types of real networks.

As no features are available, for generating meaningful weights we equip our model (LPN α ) with the attention mechanism (Sec. 3.1), letting weights vary according to node states.

Baselines include LP with uniform weights (LP U ), the spectral embedding LEM BID3 , and the deep embedding DEEPWALK BID38 , LINE BID43 , and NODE2VEC BID20 .Results: TAB4 includes accuracies for both features and no-features settings, each averaged over 10 random splits.

As can be seen, LPN outperforms other baselines on most datasetes, and consistently ranks high.

Since LPN generalizes LP, the comparison to LP U and LP RBF quantifies the gain achieved by learning weights (as opposed to setting them heuristically).

When weights are parameterized using features, accuracy improves by 13.4% on average.

When attention is used, accuracy improves by a similar 13.7%.While some deep methods perform well on some datasets, they fail on others, and their overall performance is volatile.

This is true for both learning settings.

A possible explanation is that, due to their large number of parameters and hyper-parameters, they require more labeled data.

Deep methods tend to perform well when more labeled nodes are available, and when tuning is done on a large validation set, or even on an entire dataset (see, e.g., BID48 BID28 ; BID34 BID45 ).

In contrast, LPN requires relatively few parameters (θ) and only a singe model-specific hyper-parameter (T ).

Analysis: FIG4 gives some insight as to why LPN learns good weights.

It is well known that the Laplacian's eigenvalues (and specifically λ 2 , the second smallest one) play an important role in the generalization of spectral methods BID4 .

The figure shows how λ 2 and accuracy change over the training process.

As can be seen, learning leads to weights with increasing λ 2 , followed by an increase in accuracy.

FIG5 demonstrates the effect of bifurcation for different depths T .

As can be seen, a model with bifurcation (LPN bif ) clearly outperforms the same model without it (LPN nobif ).

While adding depth generally improves LPN bif , it is quite robust across T .

This is mediated by larger values of τ that increase label convergence rate for smaller T .

Interestingly, LPN nobif degrades with large T , and even τ slightly above 1 makes a difference.

In this work we presented a deep network for graph-based SSL.

Our design process revolved around two main ideas: that edge weights should be learned, and that labeled data should be propagated.

We began by revisiting the classic LP algorithm, whose simple structure allowed us to encode it as a differentiable neural network.

We then proposed two novel ad-hoc components: information-gated attention and bifurcation, and kept our design slim and lightly parameterized.

The resulting model is a powerful generalization of the original algorithm, that can be trained efficiently using the leave-one-out loss using few labeled nodes.

We point out two avenues for future work.

First, despite its non-linearities, the current network still employs the same simple averaging updates that LP does.

An interesting challenge is to design general parametric update schemes, that can perhaps be learned.

Second, since the Laplacian's eigenvalues play an important role in both theory and in practice, an interesting question is whether these can be used as the basis for an explicit form of regularization.

We leave this for future work.

Dataset statistics are summarized in TAB1 .

As described in Sec. 5, there are two collections of data, LINQS 3 BID41 and FLIP 4 BID40

Although possible, parameterizing H directly by w will likely lead to overfitting.

Instead, we set edge weights to be a function of edge features φ ij ∈ R d and parameters θ φ ∈ R d , and normalize using softmax over scores: DISPLAYFORM0 Our main guideline in choosing features is to keep in line with the typical SSL settings where there are only few labeled nodes.

To this end, we use only a handful of features, thus keeping the number of model parameters to a minimum.

We propose three types of features suited for different settings.

Most works consider only "raw" node features (e.g., bag-of-words for papers in a citation network).

The model, however, requires edge features for parameterizing edge weights.

Edge features are therefore implicitly constructed from node features, typically by considering node-pair similarities in features space.

This has three limitations.

First, node feature spaces tend to be large and can thus lead to over-parameterization and eventually overfitting.

Second, edge features are inherently local, as they are based only on the features of corresponding nodes, and global graph-dependent properties of edges are not taken into account.

Third, parameterization is completely independent of the labeled set, meaning that edges are treated similarly regardless of whether they are in an informative region of the graph (e.g., close to labeled nodes) or not (e.g., far from any labeled node).In accordance, we propose three types of features that overcome these issues by leveraging raw features, the graph, and the labeled "seed" set.

Raw features (φ x ): When the data includes node features x i , a simple idea is to use a small set of uncorrelated (unparameterized) similarity measures.

Examples include feature similarity measures such as cosine (x i x j / x i x j ) or Gaussian (exp{− x i − x j 2 2 /σ 2 }) and the top components of dimensionality reduction methods.

Graph features (φ G ): When the graph is real (e.g., a social network), local attributes and global roles of nodes are likely to be informative features.

These can include node attributes (e.g., degree), centrality measures (e.g., edge betweenness), path-ensembles (e.g., Katz distance), and graph-partitions (e.g., k-cores).

These have been successfully used in other predictive tasks on networked data (e.g., BID13 ).Seed features (φ S ): Since labels propagate over the graph, nodes that are close to the labeled set typically have predictions that are both accurate and confident.

One way of utilizing this is to associate an incoming edge with the lengths of paths that originate in a labeled node and include it.

This acts as a proxy for the reliability of a neighbor as a source of label information.

In general, features should be used if they lead to good generalization; in our case, this depends on the available data (such as node features), the type of graph (e.g., real network vs. k-NN graph), and by the layout of the labeled set (e.g., randomly sampled vs. crawled).

TAB4 provides a list of some useful features of each of the above types.

These were used in our experiments.

BID30 |Γ(u) ∩ Γ(v)|/|Γ(u) ∪ Γ(v)| Graph Link Prediction Edge Adamic Adar Index BID30 w∈Γ FORMULA10 x, y denote feature vectors of two different nodes.

Γ(u) is the set of neighbors of u. σ(s, t) is the number of shortest paths from s to t and σ(s, t|e) is those that pass through e.• DEEPWALK: used source code provided by the authors.• NODE2VEC: used source code provided by the authors.• LINE: used source code provided by the authors.

@highlight

Neural net for graph-based semi-supervised learning; revisits the classics and propagates *labels* rather than feature representations
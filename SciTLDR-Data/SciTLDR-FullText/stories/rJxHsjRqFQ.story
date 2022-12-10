Recent approaches have successfully demonstrated the benefits of learning the parameters of shallow networks in hyperbolic space.

We extend this line of work by imposing hyperbolic geometry on the embeddings used to compute the ubiquitous attention mechanisms for different neural networks architectures.

By only changing the geometry of embedding of object representations, we can use the embedding space more efficiently without increasing the number of parameters of the model.

Mainly as the number of objects grows exponentially for any semantic distance from the query, hyperbolic geometry  --as opposed to Euclidean geometry-- can encode those objects without having any interference.

Our method shows improvements in generalization on neural machine translation on WMT'14 (English to German), learning on graphs (both on synthetic and real-world graph tasks) and visual question answering (CLEVR) tasks while keeping the neural representations compact.

The focus of this work is to endow neural network representations with suitable geometry to capture fundamental properties of data, including hierarchy and clustering behaviour.

These properties emerge in many real-world scenarios that approximately follow power-law distributions BID28 BID9 ).

This includes a wide range of natural phenomena in physics BID23 , biology BID26 , and even human-made structures such as metabolic-mass relationships BID4 , social networks , and frequencies of words BID33 BID32 BID38 ).Complex networks , which connect distinguishable heterogeneous sets of elements represented as nodes, provide us an intuitive way of understanding these structures.

They will also serve as our starting point for introducing hyperbolic geometry, which is by itself difficult to visualize.

Nodes in complex networks are referred to as heterogeneous, in the sense that they can be divided into sub-nodes which are themselves distinguishable from each other.

The scale-free structure of natural data manifests itself as a power law distribution on the node degrees of the complex network that describes it.

Complex networks can be approximated with tree-like structures, such as taxonomies and dendrograms, and as lucidly presented by , hyperbolic spaces can be thought of as smooth trees abstracting the hierarchical organization of complex networks.

Let us begin by recalling a simple property of n-ary trees that will help us understand hyperbolic space and why hyperbolic geometry is well suited to model relational data.

In an n-ary tree, the number of nodes at distance r from the root and the number of nodes at distance no more than r from the root both grow as n r .

Similarly, in a two-dimensional hyperbolic space with curvature −ζ 2 ,ζ > 0, the circumference and area of a disc of radius r grows as 2πsinh(ζr) and 2π(cosh(ζr)−1), respectively, both of are exponential in r BID20 .

The growth of volume in hyperbolic space should be contrasted with Euclidean space where the corresponding quantities expand polynomially, circumference as 2πr and area as πr 2 .In the two-dimensional example of Figure 1 , the expanding rings show examples at a fixed semantic distance from the central object ("pug").

The number of concepts grows quickly with semantic distance forcing each successive ring to be more crowded in order to maintain a fixed distance to the center.

In contrast, the extra volume of hyperbolic spheres (depicted by reducing the size of the examples) allows all of the examples to remain well separated from their semantic neighbours.

Figure 1: An intuitive depiction of how images might be embedded in 2D.

The location of the embeddings reflects the similarity between each image and that of a pug.

Since the number of instances within a given semantic distance from the central object grows exponentially, the Euclidean space is not able to compactly represent such structure (left).

In hyperbolic space (right) the volume grows exponentially, allowing for sufficient room to embed the images.

For visualization, we have shrunk the images in this Euclidean diagram, a trick also used by Escher.

Mechanically, the computed embeddings by a random network for objects at a given semantic distance might still seem epsilon distance away from each other (or crowded) as the ones obtained by using Euclidean geometry.

However, enforcing hyperbolic geometry intuitively means that all operations with these embeddings take into account, the density in that particular region of the space.

For example, any noise introduced in the system (e.g., in gradients) will also be corrected by the density.

In contrast to working in Euclidean space, this means that the embeddings will be equally distinguishable regardless of the density.

The intimate connection between hyperbolic space and scale free networks (where node degree follows a power law) is made more precise in .

In particular, there it is shown that the heterogeneous topology implies hyperbolic geometry, and conversely hyperbolic geometry yields heterogeneous topology.

Moreover, Sarkar (2011) describes a construction that embeds trees in two-dimensional hyperbolic space with arbitrarily low distortion, which is not possible in Euclidean space of any dimension BID24 .

Following this exciting line of research, recently the machine learning community has gained interest in learning non-Euclidean embeddings directly from data BID29 BID7 BID34 BID30 BID39 BID5 .Fuelled by the desire of increasing the capacity of neural networks without increasing the number of trainable parameters so as to match the complexity of data, we propose hyperbolic attention networks.

As opposed to previous approaches, which impose hyperbolic geometry on the parameters of shallow networks BID29 BID7 , we impose hyperbolic geometry on the activations of deep networks.

This allows us to exploit hyperbolic geometry to reason about embeddings produced by deep networks.

We introduce efficient hyperbolic operations to express the popular, ubiquitous mechanism of attention BID1 BID12 BID42 BID47 .

Our method shows improvements in terms of generalization on neural machine translation BID42 , learning on graphs and visual question answering BID0 BID25 BID16 tasks while keeping the representations compact.

Simultaneously to our work, BID8 proposed a method to learn SVMs in the hyperboloid model of hyperbolic space, and Nickel and Kiela (2018) proposed a method to learn shallow embeddings of graphs in hyperbolic space by using the hyperboloid model.

Hyperbolic space cannot be isometrically embedded into Euclidean space ; however, there are several ways to endow different subsets of Euclidean space with a hyperbolic metric, leading to different models of hyperbolic space.

This leads to the well known Poincaré ball model BID15 ) and many others.

The different models of hyperbolic space are all essentially the same, but different models define different coordinate systems, which offer different affordances for computation.

In this paper, we primarily make use of the hyperboloid (or Lorentz) model of the hyperbolic space.

Since the hyperboloid is unbounded, it a convenient target for projecting into hyperbolic space.

We also make use of the Klein model, because it admits an efficient expression for the hyperbolic aggregation operation we define in Section 4.2.We briefly review the definitions of the hyperboloid and Klein models and the relationship between them, in just enough detail to support the presentation in the remainder of the paper.

A more thorough treatment can be found in BID15 .

The geometric relationship between the Klein and hyperboloid models is diagrammed in FIG5 of the supplementary material.

Hyperboloid model: This model of n dimensional hyperbolic space is a manifold in the n + 1 dimensional Minkowski space.

The Minkowski space is R n+1 endowed with the indefinite Minkowski bilinear form DISPLAYFORM0 The hyperboloid model consists of the set DISPLAYFORM1 Klein model: This model of hyperbolic space is a subset of R n given by K n = {x ∈ R n | x < 1}, and a point in the Klein model can be obtained from the corresponding point in the hyperboloid model by projection DISPLAYFORM2 , with its inverse given by DISPLAYFORM3 Distance computations in the Klein model can be inherited from the hyperboloid, in the sense that DISPLAYFORM4

Learning relations in a graph by using neural networks to model the interactions or relations has shown promising results in visual question answering BID35 , modelling physical dynamics BID2 , and reasoning over graphs BID22 BID44 BID17 BID19 .

Graph neural networks BID22 BID2 incorporate a message passing as part of the architecture in order to capture the intrinsic relations between entities.

Graph convolution networks BID6 BID18 BID10 use convolutions to efficiently learn a continuous-space representation for a graph of interest.

Many of these relational reasoning models can be expressed in terms of an attentive read operation.

In the following subsection, we give a general description of the attentive read, and then discuss its specific instantiations in two relational reasoning models from the literature.

First introduced for translation in BID1 , attention has seen widespread use in deep learning, not only for applications in NLP but also for image processing BID47 imitation learning BID12 and memory BID14 .

The core computation is the attentive read operation, which has the following form: DISPLAYFORM0 Here q i is a vector called the query and the k j 's are the keys for the memory locations being read from.

The pairwise function f (·,·) computes a scalar matching score between a query and a key, and the vector v ij is a value to be read from location j by query i. Z > 0 is a normalization factor for the full sum.

Both v ij and Z are free to depend on arbitrary information, but we leave any dependencies here implicit.

It will be useful in the discussion to break this operation down into two parts.

The first is the matching, which computes attention weights α ij = f (q i ,k j ) and the second is the aggregation, which takes a weighted average of the values using these weights, DISPLAYFORM1 Instantiating a particular attentive read operation involves specifying both f (·,·) and v ij along with the normalization constant Z.If one performs an attentive read for each element of the set j then the resulting operation corresponds in a natural way to message passing on a graph, where each node i aggregates messages {v ij } j from its neighbours along edges of weight DISPLAYFORM2 We can express many (although not all) message passing neural network architectures BID13 using the attentive read operation of Equation 1 as a primitive.

In the following sections we do this for two architectures and then discuss how we can replace both the matching and aggregation steps with versions that leverage hyperbolic geometry.

Relation Networks (RNs) BID35 ) are a neural network architecture designed for reasoning about the relationships between objects.

An RN operates on a set of objects O by applying a shared operator to each pair of objects (o i , o j ) ∈ O×O. The pairs can be augmented by a global information, and the result of each relational operation is passed through a further global transformation.

Using the notation of the previous section, we can write the RN as DISPLAYFORM0 h is the global transformation, g is the local transformation and c is the global context, as described in BID35 .

We augment the basic RN to allow f (o i ,o j ) ∈ [0,1] to be a general learnable function.

Interpreting the RN as learned message passing on a graph over objects, the attention weights take on the semantics of edge weights, where α ij can be thought of as the probability of the (directed) edge o j → o i appearing in the underlying reasoning graph.

In the Transformer model of BID42 the authors define an all-to-all message passing operation on a set of vectors which they call scaled dot-product attention.

In the language of Section 3.1 the scaled dot-product attention operation performs several attentive reads in parallel, one for each element of the input set.

BID42 write scaled dot-product attention as R = softmax DISPLAYFORM0 V, where Q, K and V are referred to as the queries, keys, and values respectively, and d is the shared dimensionality of the queries and keys.

Using lowercase letters to denote rows of the corresponding matrices, we can write each row of R as the result of an attentive read with DISPLAYFORM1 We experiment with both softmax and sigmoid operations for computing the attention weights in our hyperbolic models.

The motivation for considering sigmoid attention weights is that in some applications (e.g. visual question answering), it makes sense for the attention weights over different entities to not compete with each other.

In this section we show how to redefine the attentive read operation of Section 3.1 as an operation on points in hyperbolic space.

The key for doing this is to define new matching and aggregation functions that operate on hyperbolic points and take advantage of the metric structure of the manifold they live on.

However, in order to apply these operations inside of a network we first we need a way to interpret network activations as points in hyperbolic space.

We describe how to map an arbitrary point in R n onto the hyperboloid, where we can interpret the result as a point in hyperbolic space.

The choice of mapping is important since we must ensure that the rapid scaling behavior of hyperbolic space is maintained.

Armed with an appropriate mapping we proceed to describe the hyperbolic matching and aggregation operations that operate on these points.

Mapping neural network activations into hyperbolic space requires care, since network activations might live anywhere in R n , but hyperbolic structure can only be imposed on special subsets of Euclidean space .

This means we need a way to map activations into an appropriate manifold.

We choose to map into the hyperboloid, which is convenient since it is the only unbounded model of hyperbolic space in common use.

In polar coordinates, we express an n-dimensional point as a scalar radius, and n−1 angles.

Pseudo-polar coordinates consist of a radius r, as in ordinary polar coordinates, and an n-dimensional vector d representing the direction of the point from the origin.

In the following discussion we assume that the coordinates are normalized, i.e. that d = 1.

DISPLAYFORM0 are the activations of a layer in the network, we map them onto the hyperbolid in R n+1 using π((d, r)) = (sinh(r)d, cosh(r)), which increases the scale by an exponential factor.

It is easily verified that the resulting point lies in the hyperboloid, and to verify that we maintain the appropriate scaling properties we compute the distance between a point and the origin using this projection: DISPLAYFORM1 which shows that this projection preserves exponential growth in volume for a linear increase in r. Without the exponential scaling factor the effective distance of π((d, r)) from the origin grows logarithmically in hyperbolic space.

In this section, we show how to build an attentive read operation that operates on points in hyperbolic space.

We consider how to exploit hyperbolic geometry in both the matching and the aggregation steps of the attentive read operation separately.

Hyperbolic matching: The most natural way to exploit hyperbolic geometry for matching pairs of points is to use the hyperbolic distance between them.

Given a query q i and a key k j both lying in hyperbolic space we take, DISPLAYFORM2 where d H (·, ·) is the hyperbolic distance, and β and c are parameters that can be set manually or learned along with the rest of the network.

Having the bias parameter c is useful because distances are non-negative.

We take the function f (·) to be either exp(·), in which case we set the normalization appropriately to obtain a softmax, or sigmoid(·).

Hyperbolic aggregation: The path to extend the weighted midpoint to hyperbolic space is much less obvious, but fortunately such a extension already exists as the Einstein midpoint.

The Einstein midpoint is straightforward to compute by adjusting the aggregation weights appropriately (see Ungar (2005, Definition 4 .21)) Fortunately the various models of hyperbolic space in common use are all isomorphic, so we can work in an arbitrary hyperbolic model and simply project to and from the Klein model to execute midpoint computations, as discussed in Section 2.

DISPLAYFORM3 The reason for using the Einstein midpoint for hyperbolic aggregation is that it obeys many of the properties that we expect from a weighted average in Euclidean space.

In particular, translating the v ij 's by a fixed distance in a common direction also translates the midpoint, and it is invariant to rotations of the constellation of points about the midpoint.

The derivation of this operation is quite involved, and beyond the scope of this paper.

We point the interested reader to BID40 BID41 for a full exposition.

We evaluate our models on synthetic and real-world tasks.

Experiments where the underlying graph structure is explicitly known clearly show the benefits of using hyperbolic geometry as an inductive bias.

At the same time, we show that real-world tasks within implicit graph structure such as a diagnostic visual question answering task BID16 , and neural machine translation, equally benefit from relying on hyperbolic geometry.

We provide experiments with feed-forward networks, the Transformer BID42 and Relation Networks BID35 endowed with hyperbolic attention.

Our results show the effectiveness of our approach on diverse tasks and architectures.

The benefit of our approach is particularly prominent in relatively small models, which supports our hypothesis that hyperbolic geometry induces compact representations and is therefore better able to represent complex functions in limited space.

We use the algorithm of von BID46 to efficiently generate large scale-free graphs, and define two predictive tasks that test our model's ability to represent different aspects of the structure of these networks.

For both tasks in this section, we train Recursive Transformer (RT) models, using hyperbolic and Euclidean attention.

A Recursive Transformer is identical to the original transformer, except that the weights of each self-attention layer are tied across depth.

Simultaneously to our work, BID11 have proposed the same model as a generalization of the Transformer model and they referred to it as "Universal Transformers".

We use models with 3 recursive self-attention layers, each of which has 4 heads with 4 units each for each of q, k, and v. This model has similarities to Graph Attention Networks BID43 BID19 .

Link prediction is a classical graph problem, where the task is to predict if an edge exists between two nodes in the graph.

We experimented with graphs of 1000 and 1200 nodes and observed that the hyperbolic RT performs better than the Euclidean RT on both tasks.

We report the results in FIG4 (middle).

In general, we observed that for graphs of size 1000 and 1200 the hyperbolic transformer performs better than the Euclidean transformer given the same amount of capacity.

In this task, the goal is to predict the length of the shortest path between a pair of nodes in the graph.

We treat this as a classification problem with a maximum pathlength of 25 which becomes naturally an unbalanced classification problem.

We use rejection sampling during training to ensure the network is trained on an approximately uniform distribution of path lengths.

At test time we sample paths uniformly at random, so the length distribution follows that of the underlying graphs.

We report the results in FIG4 (left).

In FIG4 (right), we visualize the distribution of the scale of the learned activations (r in the projection of Section 4.1) when training on graphs of size 100 and 400.

We observe that our model tends to use larger scales for the larger graphs.

As a baseline, we compare to the optimal constant predictor, which always predicts the most common expected path length.

This baseline does quite well since the path length distribution on the test set is quite skewed.

For both tasks, we generate training data online.

Each example is a new graph in which we query the connectivity of a randomly chosen pair of nodes.

To make training easier, we use a curriculum, whereby we start training on smaller graphs and gradually increase the number of vertices towards the final number.

More details on the dataset generation procedure and the curriculum scheme are found in the supplementary material.

Since we expect hyperbolic attention to be particularly well suited to relational modelling, we investigate our models on the relational variant of the Sort-of-CLEVR dataset BID35 .

This dataset consists of simple visual scenes allowing us to solely focus on the relational aspect of the problem.

Our models extend Relation Nets (RNs) with the attention mechanism in hyperbolic space (with the Euclidean or Einstein midpoint aggregation), but otherwise we follow the standard setup-up BID35 .

Our best method yields accuracy of 99.2% that significantly exceeds the accuracy of the original RN (96%).However, we are more interested in evaluating models on the low-capacity regime.

Indeed, as Figure 4 (left) shows, the attention mechanism computed in the hyperbolic space improves around 20 percent points over the standard RN, where all the models use only two units of the relational MLP.

We use two of the standard graph transduction benchmark datasets, Citeseer and Cora BID37 and used the same experimental protocol defined in BID43 .

We use graph attention 0 100 200 300 400 500 600 700Number of updates (x1000)

Hyperbolic RN (Sigmoid) Hyperbolic RN (Softmax) Euclidean RN (Softmax) Figure 4 : Left:

Comparison of our models with low-capacity on the Sort-of-CLEVR dataset.

The "EA" refers to the model that uses hyperbolic attention weights with Euclidean aggregation.

Right: Performance of Relation Network extended by attention mechanism in either Euclidean or hyperbolic space on the CLEVR dataset.

Method Cora Citeseer GCN BID18 81.5%70.3% GAT BID43 83.0% ± 0.14 72.5% ± 0.14 H-GAT 83.5% ± 0.12 72.9% ± 0.078 Table 1 : Results on graph transduction tasks.

We have used the same setup that is described in BID43 .

H-GAT refers to our graph attention network with hyperbolic attention mechanism.

Table shows the mean performance over 100 random seeds, along with 95% confidence intervals for this estimate.networks (GAT) as our baseline and developed a hyperbolic version of GAT (H-GAT) by replacing the original attention mechanism with the hyperbolic attention using softmax.

We report our results in Table 1 and compare against the GAT with the Euclidean attention mechanism.

We compute the standard deviations over 100 seeds and got improvements both on Citeseer and Cora datasets over the original GAT model.

We show the visualizations of the learned hyperbolic embeddings of q and k in A.4.

We train our Relation Network with various attention mechanisms on the CLEVR dataset BID16 .

CLEVR is a synthetic visual question answering datasets consisting of 3D rendered objects like spheres, cubes, or cylinders of various size, material, or color.

In contrast to other visual question answering datasets BID0 BID25 Zhu et al., 2016) , the focus of CLEVR is on relational reasoning.

In our experiments, we closely follow the procedure established in BID35 , both in terms of the model architecture, capacity, or the choice of the hyperparameters, and only differ by the attention mechanism (Euclidean or hyperbolic attention), or sigmoid activations.

Results are shown in Figure 4 (Right).

For each model, we vary the capacity of the relational part of the network and report the resulting test accuracy.

We find that hyperbolic attention with sigmoid consistently outperforms other models.

Our RN with hyperbolic attention and sigmoid achieves 95.7% accuracy on the test set at the same capacity level as RN, whereas the latter reportedly achieves 95.5% accuracy BID35 .

The Transformer BID42 ) is a recently introduced state of the art model for neural machine translation that relies heavily on attention as its core operation.

As described in Section 3.

BID42 .

Citations indicate results taken from the literature.

Latest is the result of training a new model using an unmodified version of the same code where we added hyperbolic attention (we have observed that the exact performance of the transformer on this task varies as the Tensor2tensor codebase evolves).we have extended the Transformer 2 by replacing its scaled dot-product attention operation with its hyperbolic counterpart.

We evaluate all the models on the WMT14 En-De dataset BID3 .We train several versions of the Transformer model with hyperbolic attention.

They use different coordinate systems (Cartesian or pseudo-polar), or different attention normalization functions (softmax or sigmoid).

We consider three model sizes, referred to here as tiny, base and big.

The tiny model has two layers of encoders and decoders, each with 128 units and 4 attention heads.

The base model has 6 layers of encoders and decoders, each with 512 units and 8 attention heads.

All hyperparameter configurations for the Euclidean versions of these models are available in the Tensor2tensor repository.

Results are shown in TAB2 .

We observe improvements over the Euclidean model by using hyperbolic attention, in particular when coupled with the sigmoid activation function for the attention weights.

The improvements are more significant when the model capacity is restricted.

In addition, our best model (with sigmoid activation function and without pseudo-polar coordinates) using the big architecture from Tensor2tensor, achieves 28.52 BLEU score, whereas BID42 report 28.4 BLEU score with the original version of this model.3 .

We have presented a novel way to impose the inductive biases from hyperbolic geometry on the activations of deep neural networks.

Our proposed hyperbolic attention operation makes use of hyperbolic geometry in both the computation of the attention weights, and in the aggregation operation over values.

We implemented our proposed hyperbolic attention mechanism in both Relation Networks and the Transformer and showed that we achieve improved performance on a diverse set of tasks.

We have shown improved performance on link prediction and shortest path length prediction in scale free graphs, on two visual question answering datasets, real-world graph transduction tasks and finally on English to German machine translation.

The gains are particularly prominent in relatively small models, which confirms our hypothesis that hyperbolic geometry induces more compact representations.

Yang and Rush (2016) have proposed to imposed the activations of the neural network to lie on a Lie-group manifold in the memory.

Similarly as a future work, an interesting potential future direction is to use hyperbolic geometry as an inductive bias for the activation of neural networks in the memory.

In FIG5 , we illustrate the relationship between different models of hyperbolic space.

There are one-to-one isometric transformations defined between each different models of the hyperbolic space.

Hyperboloid model is unbounded, whereas Klein and Poincare models are bounded in a disk.

We use the algorithm described by von BID46 .

In our experiments, we set the α to 0.95 and edge_radius_R_factor to 0.35.

We will release our code both for generating and the operations in the hyperbolic space along with the camera-ready version of our paper.

Curriculum was an essential part of our training on the scale-free graph tasks.

On LP and SPLP tasks, we use a curriculum where we extract the connected components from the graph by cutting the disk that the graphs generated on into slices by starting from a 30 degree angle and gradually increasing the size of the slice from the disk by increasing the angle during the training according to the number of lessons that are involved in the curriculum.

This process is also visualized in Figure 6 .

In FIG6 and 9, we visualize the embeddings of the query (q) and the keys (k) going into the hyperbolic matching function on the Poincare Ball model.

In FIG6 , the embeddings of a model trained with dropout are bounded in a ball with smaller volume than the model trained without dropout.

Also as clearly can be seen from the embedding visualizations k's and q's are clustered on different regions of the space.

We train an off-policy DQN-like agent BID27 with the HRT.

The graphs for the TSP is generated following the procedure introduced in BID45 .On this task, as an ablation we just compared the hyperbolic networks with and The results are provided in Figure 4 (Right) with and without implicit coordinates.

Overall, we found that the hyperbolic transformer networks performs better when using the implicit polar coordinates.

A.6 HYPERBOLIC RECURSIVE TRANSFORMER As shown in Figure ? ?

, the hyperbolic RT is an extension of transformer that ties the parameters of the self-attention layers.

The self-attention layer gets the representations of the nodes of the graph coming from the encoder and the decoder decodes that representation from the recursive self-attention layers for the prediction.

Figure 7: An illustration of how trees can be represented in hyperbolic (left) and Euclidean geometry (right) in a cone.

In hyperbolic space, as the tree grows the angles between the edges (θ) can be preserved from one level to the next.

In Euclidean space, since the number of nodes in the tree grows faster than the rate that the volume grows, angles may not be preserved (θ to α).

Lines in the left diagram are straight in hyperbolic space, but appear curved in this Euclidean diagram.

Hyperbolic Recursive Transformer Hyperbolic Recursive Transformer (+spherical) Figure 10 : The comparisons between a hyperbolic recursive transformer with and without pseudo-polar (denoted as +spherical in the legend) coordinates on the travelling salesman problem.

@highlight

We propose to incorporate inductive biases and operations coming from hyperbolic geometry to improve the attention mechanism of the neural networks.

@highlight

This paper replaces the dot-product similarity used in attention mechanisms with the negative hyperbolic distance, and applies it to the existing Transformer model, graph attention networks, and Relation Networks

@highlight

The authors propose a novel approach to improve relational-attention by changing the matching and aggregation functions to use hyperbolic geometric. 
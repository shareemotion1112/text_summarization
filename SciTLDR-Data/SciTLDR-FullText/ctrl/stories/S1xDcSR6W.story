Neural embeddings have been used with great success in Natural Language Processing (NLP) where they provide compact representations that encapsulate word similarity and attain state-of-the-art performance in a range of linguistic tasks.

The success of neural embeddings has prompted significant amounts of research into applications in domains other than language.

One such domain is graph-structured data, where embeddings of vertices can be learned that encapsulate vertex similarity and improve performance on tasks including edge prediction and vertex labelling.

For both NLP and graph-based tasks, embeddings in high-dimensional Euclidean spaces have been learned.

However, recent work has shown that the appropriate isometric space for embedding complex networks is not the flat Euclidean space, but a negatively curved hyperbolic space.

We present a new concept that exploits these recent insights and propose learning neural embeddings of graphs in hyperbolic space.

We provide experimental evidence that hyperbolic embeddings significantly outperform Euclidean embeddings on vertex classification tasks for several real-world public datasets.

Embeddings are used to represent complex high-dimensional data in lower-dimensional continuous spaces BID28 BID3 .

Embedded representations provide three principal benefits over sparse schemes: They encapsulate similarity, are compact, and perform better as inputs to machine learning models BID29 .

These benefits are particularly important for graph-structured data where the native representation is the adjacency matrix, which is typically a sparse matrix of connection weights.

Neural embedding models are a flavour of embedding where the embedded representation corresponds to a subset of the connection weights in a neural network (see FIG2 ), which are learned through backpropagation.

Neural embedding models have been shown to improve performance on many tasks across multiple domains, including word analogies (Mikolov et al., 2013a; BID20 , machine translation BID31 ), document comparison (Kusner et al., 2015 , missing edge prediction BID12 , vertex attribution BID26 , product recommendations BID10 BID1 , customer value prediction BID14 BID6 and item categorisation BID2 .

In all cases, the embeddings are learned without labels (unsupervised) from a sequence of tokens.

Previous work on neural embedding models has either either explicitly or implicitly (by using the Euclidean dot product) assumed that the embedding space is Euclidean.

However, recent work in the field of complex networks has found that many interesting networks, particularly those with a scale-free structure such as the Internet BID30 BID5 or academic citations BID8 BID7 can be well described with a geometry which is non-Euclidean, such as hyperbolic geometry.

Even more recently the problem of mapping graphs and datasets to a low-dimensional hyperbolic space has been addressed in BID24 and BID4 .

Here we use a neural embedding approach based on the Skipgram architecture to find hyperbolic embeddings.

There are two reasons why embedding complex networks in hyperbolic geometry can be expected to perform better than Euclidean geometry.

The first is that complex networks exhibit a hierarchical structure.

Hyperbolic geometry provides a continuous analogue of tree-like graphs, and even infinite trees have nearly isometric embeddings in hyperbolic space BID11 .

The second property is that complex networks have power-law degree distributions, resulting in high-degree hub vertices.

All tiles are of constant area in hyperbolic space, but shrink to zero area at the boundary of the disk in Euclidean space.

c Hub and spokes graph.

It is impossible to embed this graph in two-dimensional Euclidean space and preserve the properties that (1) all spokes are the same distance from the hub, (2) all spokes are the same distance from each other, and (3) the distance between spokes along the circumference is more than twice the distance to the hub.

In hyperbolic space such embeddings exist.

FIG1 shows a simple hub-and-spoke graph where each spoke is a distance R from the hub and 2R from each other.

For an embedding in two-dimensional Euclidean space it is impossible to reproduce this geometry for more than two spokes.

However, in hyperbolic space, large numbers of spokes that satisfy these geometrical constraints can be embedded because the circumference of a circle expands exponentially rather than polynomially with the radius.

The starting point for our model is the celebrated Skipgram architecture (Mikolov et al., 2013a; b) shown in FIG2 .

Skipgram is a shallow neural network with three layers: (1) An input projection layer that maps from a one-hot-encoded token to a distributed representation, (2) a hidden layer, and (3) an output softmax layer.

Skipgram is trained on a sequence of words that is decomposed into (input word, context word)-pairs.

The model uses two separate vector representations, one for the input words and another for the context words, with the input representation comprising the learned embedding.

The (input word, context word)-pairs are generated by running a fixed length sliding window over a word sequence.

Words are initially randomly allocated to vectors within the two vector spaces.

Then, for each training word pair, the vector representations of the observed input and context words are pushed towards each other and away from all other words (see FIG2 ).

The model can be extended to network structured data using random walks to create sequences of vertices.

Vertices are then treated exactly analogously to words in the NLP formulation.

This was originally proposed as DeepWalk BID26 .

Extensions varying the nature of the random walks have been explored in LINE BID32 and Node2vec BID12 .Contribution In this paper, we introduce the new concept of neural embeddings in hyperbolic space.

We formulate backpropagation in hyperbolic space and show that using the natural geometry of complex networks improves performance in vertex classification tasks across multiple networks.

At the same time, BID24 independently proposed a hyperbolic embedding algorithm that has similarities to ours.

The key differences are that BID24 try to fit the hyperbolic distance between nodes using cartesian coordinates in the Poincaré disk, whereas we use a modified cosine distance in a spherical hyperbolic coordinate system.

Our approach does not require a numerical constraint to prevent points from 'falling off' the edge of the disk and becoming infinitely distant from the others.

Hyperbolic geometry emerged through a relaxation of Euclid's fifth geometric postulate (the parallel postulate).

In hyperbolic space, there is not just one, but an infinite number of parallel lines that pass through a single point.

This is illustrated in FIG1 where every fine line is parallel to the bold, blue line, and all pass through the same point.

Hyperbolic space is one of only three types of isotropic space that can be defined entirely by their curvature.

The most familiar is flat Euclidean space.

Space with uniform positive curvature has an elliptic geometry (e.g. the surface of a sphere) and space with uniform negative curvature has a hyperbolic geometry, which is analogous to a saddle-like surface.

Unlike Euclidean space, in hyperbolic space even infinite trees have nearly isometric embeddings, making the space well suited to model complex networks with hierarchical structure.

Additionally, the defining features of complex networks, such as power-law degree distributions, strong clustering and community structure, emerge naturally when random graphs are embedded in hyperbolic space .One of the defining characteristics of hyperbolic space is that it is in some sense larger than Euclidean space; the 2D hyperbolic plane cannot be isometrically embedded into Euclidean space of any dimension, unlike elliptic geometry where a 2-sphere can be embedded into 3D Euclidean space etc.

The hyperbolic area of a circle or volume of a sphere grows exponentially with its radius, rather than polynomially.

This property allows low-dimensional hyperbolic spaces to provide effective representations of data in ways that low-dimensional Euclidean spaces cannot.

FIG1 shows a hub-and-spoke graph with four spokes embedded in a two-dimensional Euclidean plane so that each spoke sits on the circumference of a circle surrounding the hub.

Each spoke is a distance R from the hub and 2R from every other spoke, but in the embeddings the spokes are a distance of R from the hub, but only R √ 2 from each other.

Complex networks often have small numbers of vertices with degrees that are orders of magnitude greater than the median.

These vertices approximate hubs.

The distance between spokes tends to the distance along the circumference s = 2πR n as the number of spokes n increases, and so the shortest distance between two spokes is via the hub only when n < π.

However, for embeddings in hyperbolic space, we get n < sinh R R , such that an infinite number of spokes can satisfy the property that they are the same distance from a hub, and yet the path that connects them via the hub is shorter than along the arc of the circle.

As hyperbolic space can not be isometrically embedded in Euclidean space, there are many different representations that each conserve some geometric properties, but distort others.

In this paper, we use the Poincaré disk model of hyperbolic space.

The Poincaré disk models the infinite two-dimensional hyperbolic plane as a unit disk.

For simplicity we work with the two-dimensional disk, but it is easily generalised to the d-dimensional Poincaré ball, where hyperbolic space is represented as a unit d-ball.

Hyperbolic distances grow exponentially towards the edge of the disk.

The boundary of the disk represents infinitely distant points as the infinite hyperbolic plane is squashed inside the finite disk.

This property is illustrated in FIG1 where each tile is of constant area in hyperbolic space, but rapidly shrink to zero area in Euclidean space.

Although volumes and distances are warped, the Poincaré disk model is conformal.

Straight lines in hyperbolic space intersect the boundary of the disk orthogonally and appear either as diameters of the disk, or arcs of a circle.

FIG1 shows a collection of straight hyperbolic lines in the Poincaré disk.

Just as in spherical geometry, shortest paths appear curved on a flat map, hyperbolic geodesics also appear curved in the Poicaré disk.

This is because it is quicker to move close to the centre of the disk, where distances are shorter, than nearer the edge.

In our proposed approach, we will exploit both the conformal property and the circular symmetry of the Poincaré disk.

The geometric intuition motivating our approach is that vertices embedded near the middle of the disk can have more 'near' neighbours than they could in Euclidean space, whilst vertices nearer the edge of the disk can still be very far from each other.

The distance metric in Poincaré disk is a function only of the radius.

Exploiting the angular symmetries of the model using polar coordinates considerably simplifies the mathematical description of our approach and the efficiency of our optimiser.

Points in the disk are x = (r e , θ), with r e ∈ [0, 1) and θ ∈ [0, 2π).

The distance from the origin, r h is given by DISPLAYFORM0 and the circumference of a circle of hyperbolic radius R is C = 2π sinh R. Note that as points approach the edge of the disk, r e = 1, the hyperbolic distance from the origin r h tends to infinity.

In Euclidean neural embeddings, the inner product between vector representations of vertices is used to quantify their similarity.

However, unlike Euclidean space, hyperbolic space is not a vector space and there is no global inner product.

Instead, given points x 1 = (r 1 , θ 1 ) and x 2 = (r 2 , θ 2 ) we define a cosine similarity weighted by the hyperbolic distance from the origin as DISPLAYFORM1 DISPLAYFORM2 It is this function that we will use to quantify the similarity between points in the embedding.

We note that using a cosine distance in this way does lose some properties of hyperbolic space such as conformality.

Our goal is to learn embeddings that perform well on downstream tasks and the key properties of hyperbolic space that permit this are retained.

Trade-offs like this are common in the embeddings literature such as the use of negative sampling BID22 BID20 .

We adopt the notation of the original Skipgram paper (Mikolov et al., 2013a) whereby the input vertex is w I and the context / output vertex is w O .

The corresponding vector representations are v w I and v w O , which are elements of the two vector spaces shown in FIG2 , W and W respectively.

Skipgram has a geometric interpretation, shown in FIG2 for vectors in W .

Updates to v wj are performed by simply adding (if w j is the observed output vertex) or subtracting (otherwise) an error-weighted portion of the input vector.

Similar, though slightly more complicated, update rules apply to the vectors in W. Given this interpretation, it is natural to look for alternative geometries that improve on Euclidean geometry.

To embed a graph in hyperbolic space we replace Skipgram's two Euclidean vector spaces (W and W in FIG2 ) with two Poincaré disks.

We learn embeddings by optimising an objective function that predicts context vertices from an input vertex, but we replace the Euclidean dot products used in Skipgram with (2).

A softmax function is used for the conditional predictive distribution DISPLAYFORM0 where v wi is the vector representation of the i th vertex, primed indicates context vectors (see FIG2 ) and ·, · H is given in (2).

Directly optimising (3) is computationally demanding as the sum in the denominator is over every vertex in the graph.

Two commonly used techniques for efficient computation are replacing the softmax with a hierarchical softmax BID21 Mikolov et al., 2013a) and negative sampling BID22 BID20 .

We use negative sampling as it is faster.

We learn the model using backpropagation with Stochastic Gradient Descent (SGD).

Optimisation is conducted in polar native hyperbolic coordinates where r ∈ (0, ∞), θ ∈ (0, 2π].

For optimisation, this coordinate system has two advantages over the cartesian Euclidean system used by BID24 .

Firstly there is no need to constrain the optimiser s.t.

x < 1.

This is important as arbitrarily moving points a small Euclidean distance inside the disk equates to an infinite hyperbolic distance.

Secondly, polar coordinates result in update equations that are simple modifications of the Euclidean updates, which avoids evaluating the metric tensor for each data point.

The negative log likelihood using negative sampling is DISPLAYFORM0 where v w I , v w O are the vector representation of the input and context vertices, u j = v wj , v w I H , W neg is a set of samples drawn from the noise distribution and σ is the sigmoid function.

The first term represents the observed data and the second term the negative samples.

To draw W neg , we specify the noise distribution P n to be the unigram distribution of the vertices in the input sequence raised to 3/4 as in (Mikolov et al., 2013a) .

The gradient of the negative log-likelihood in (5) w.r.t.

u j is given by DISPLAYFORM1 The derivatives w.r.t.

the components of vectors in W (in natural polar hyperbolic coordinates) are DISPLAYFORM2 such that the Jacobian is ∇ r E = DISPLAYFORM3 where χ = w O ∪ W neg , η is the learning rate and j is the prediction error defined in (6).

Calculating the derivatives w.r.t.

the input embedding follows the same pattern, and we obtain ∂E ∂r I = j:wj ∈χ DISPLAYFORM4 The corresponding update equations are where t j is an indicator variable s.t.

t j = 1 iff w j = w O , and t j = 0 otherwise.

Following optimisation, the vectors are mapped back to Euclidean coordinates on the Poincaré disk through θ h → θ e and r h → tanh r h 2 .

The asymptotic runtimes of the update equations (8) - FORMULA7 and FORMULA0 are the same as Euclidean Skipgram, i.e., the hyperbolic embedding does not add computational burden.

In this section, we assess the quality of hyperbolic embeddings and compare them to embeddings in Euclidean spaces.

Firstly we perform a qualitative assessment of the embeddings on a synthetic fully connected tree graph and a small social network.

It is clear that embeddings in hyperbolic space exhibit a number of features that are superior to Euclidean embeddings.

Secondly we run experiments on a number of public benchmark networks, producing both Euclidean and hyperbolic embeddings and contrasting the performance of both on a downstream vertex classification task.

We provide a TensorFlow implementation and datasets to replicate our experiments in our github repository 1 .

To illustrate the usefulness of hyperbolic embeddings we visually compare hyperbolic embeddings with Euclidean plots.

In all cases, embeddings were generated using five training epochs on an intermediate dataset of ten-step random walks, one originating at each vertex.

FIG3 show hyperbolic embeddings in the 2D Poincaré model of hyperbolic space where the circles of radius 1 is the infinite boundary and Euclidean embeddings in R 2 .

FIG3 shows embeddings of a complete 4-ary tree with three levels.

The vertex numbering is breadth first with one for the root and 2, 3, 4, 5 for the second level etc.

The hyperbolic embedding has the root vertex close to the origin of the disk, which is the position with the shortest average path length.

The leaves are all located in close proximity to their parents, and there are clearly four clusters representing the tree's branching factor.

The Euclidean embedding is incapable of representing the tree structure with adjacent vertices at large distances (such as 1 and 3) and vertices that are maximally separated in the tree appearing close in the embedding (such as 19 and 7).

FIG6 shows the 34-vertex karate network, which is split into two factions.

FIG6 shows the hyperbolic embedding of this network where the two factions can be clearly separated.

In addition, the vertices (5, 6, 7, 11, 17) in FIG6 are the junior instructors, who are forbidden by the instructor (vertex 1) from socialising with other members of the karate club.

For this reason they form a community that is only connected through the instructor.

This community is clearly visible in FIG6 to the The results of our experiments together with the HyBed and 2D Deepwalk embeddings used to derive them are shown in FIG8 .

The vertex colours of the embedding plots indicate different values of the vertex labels.

The legend shown in FIG8 applies to all line graphs.

The line graphs show macro F1 scores against the percentage of labelled data used to train a logistic regression classifier with the embeddings as features.

Here we follow the method for generating multi-label F1 scores described in BID17 .

The error bars show one standard error from the mean over ten repetitions.

The blue lines show HyBed hyperbolic embeddings, the yellow lines give the 2D Poincaré embeddings of BID24 while the red lines depict Deepwalk embeddings at various dimensions.

As we use one-vs-all logistic regression with embedding coordinates as features, good embeddings are those that can linearly separate one class from all other classes.

FIG8 shows that HyBed embeddings tend to cluster together similar classes so that they are linearly separable from other classes, unlike the Euclidean embeddings.

<|TLDR|>

@highlight

We learn neural embeddings of graphs in hyperbolic instead of Euclidean space
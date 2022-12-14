Despite their prevalence, Euclidean embeddings of data are fundamentally limited in their ability to capture latent semantic structures, which need not conform to Euclidean spatial assumptions.

Here we consider an alternative, which embeds data as discrete probability distributions in a Wasserstein space, endowed with an optimal transport metric.

Wasserstein spaces are much larger and more flexible than Euclidean spaces, in that they can successfully embed a wider variety of metric structures.

We propose to exploit this flexibility by learning an embedding that captures the semantic information in the Wasserstein distance between embedded distributions.

We examine empirically the representational capacity of such learned Wasserstein embeddings, showing that they can embed a wide variety of complex metric structures with smaller distortion than an equivalent Euclidean embedding.

We also investigate an application to word embedding, demonstrating a unique advantage of Wasserstein embeddings: we can directly visualize the high-dimensional embedding, as it is a probability distribution on a low-dimensional space.

This obviates the need for dimensionality reduction techniques such as t-SNE for visualization.

Learned embeddings form the basis for many state-of-the-art learning systems.

Word embeddings like word2vec BID34 , GloVe BID42 , fastText BID5 , and ELMo BID43 are ubiquitous in natural language processing, where they are used for tasks like machine translation BID38 , while graph embeddings BID41 like node2vec BID21 are used to represent knowledge graphs and pre-trained image models BID47 appear in many computer vision pipelines.

An effective embedding should capture the semantic structure of the data with high fidelity, in a way that is amenable to downstream tasks.

This makes the choice of a target space for the embedding important, since different spaces can represent different types of semantic structure.

The most common choice is to embed data into Euclidean space, where distances and angles between vectors encode their levels of association BID34 BID56 BID27 BID36 .

Euclidean spaces, however, are limited in their ability to represent complex relationships between inputs, since they make restrictive assumptions about neighborhood sizes and connectivity.

This drawback has been documented recently for tree-structured data, for example, where spaces of negative curvature are required due to exponential scaling of neighborhood sizes BID39 BID49 .In this paper, we embed input data as probability distributions in a Wasserstein space.

Wasserstein spaces endow probability distributions with an optimal transport metric, which measures the distance traveled in transporting the mass in one distribution to match another.

Recent theory has shown that Wasserstein spaces are quite flexible-more so than Euclidean spaces-allowing a variety of other metric spaces to be embedded within them while preserving their original distance metrics.

As such, they make attractive targets for embeddings in machine learning, where this flexibility might capture complex relationships between objects when other embeddings fail to do so.

Unlike prior work on Wasserstein embeddings, which has focused on embedding into Gaussian distributions BID37 BID58 , we embed input data as discrete distributions supported at a fixed number of points.

In doing so, we attempt to access the full flexibility of Wasserstein spaces to represent a wide variety of structures.

Optimal transport metrics and their gradients are costly to compute, requiring the solution of a linear program.

For efficiency, we use an approximation to the Wasserstein distance called the Sinkhorn divergence BID15 , in which the underlying transport problem is regularized to make it more tractable.

While less well-characterized theoretically with respect to embedding capacity, the Sinkhorn divergence is computed efficiently by a fixed-point iteration.

Moreover, recent work has shown that it is suitable for gradient-based optimization via automatic differentiation BID20 .

To our knowledge, our work is the first to explore embedding properties of the Sinkhorn divergence.

We empirically investigate two settings for Wasserstein embeddings.

First, we demonstrate their representational capacity by embedding a variety of complex networks, for which Wasserstein embeddings achieve higher fidelity than both Euclidean and hyperbolic embeddings.

Second, we compute Wasserstein word embeddings, which show retrieval performance comparable to existing methods.

One major benefit of our embedding is that the distributions can be visualized directly, unlike most embeddings, which require a dimensionality reduction step such as t-SNE before visualization.

We demonstrate the power of this approach by visualizing the learned word embeddings.

The p-Wasserstein distance between probability distributions ?? and ?? over a metric space X is DISPLAYFORM0 where the infimum is taken over transport plans ?? that distribute the mass in ?? to match that in ??, with the p-th power of the ground metric d(x 1 , x 2 ) on X giving the cost of moving a unit of mass from support point x 1 ??? X underlying distribution ?? to point x 2 ??? X underlying ??.

The Wasserstein distance is the cost of the optimal transport plan matching ?? and ?? BID52 .In this paper, we are concerned with discrete distributions supported on finite sets of points in R n : DISPLAYFORM1 Here, u and v are vectors of nonnegative weights summing to 1, and {x DISPLAYFORM2 are the support points.

In this case, the transport plan ?? matching ?? and ?? in Equation 1 becomes discrete as well, supported on the product of the two support sets.

Define D ??? R M ??N + to be the matrix of pairwise ground metric distances, with DISPLAYFORM3 Then, for discrete distributions, Equation 1 is equivalent to solving the following: DISPLAYFORM4 with T ij giving the transported mass between x i and y j .

The power D p is taken elementwise.

Equation 3 is a linear program that can be challenging to solve in practice.

To improve efficiency, recent learning algorithms use an entropic regularizer proposed by BID15 .

The resulting Sinkhorn divergence solves a modified version of Equation 3: DISPLAYFORM0 where log(??) is applied elementwise and ?? ??? 0 is the regularization parameter.

For ?? > 0, the optimal solution takes the form DISPLAYFORM1 , where ???(r) and ???(c) are diagonal matrices with diagonal elements r and c, resp.

Rather than optimizing over matrices T , one can optimize for r and c, reducing the size of the problem to M + N .

This can be solved via matrix balancing, starting from an initial matrix K := exp( ???D p ?? ) and alternately projecting onto the marginal constraints until convergence: DISPLAYFORM2 Here, ./ denotes elementwise division for vectors.

Beyond simplicity of implementation, Equation 5 has an additional advantage for machine learning: The steps of this algorithm are differentiable.

With this observation in mind, BID20 incorporate entropic transport into learning pipelines by applying automatic differentiation (back propagation) to a fixed number of Sinkhorn iterations.

Given two metric spaces A and B, an embedding of A into B is a map ?? : A ??? B that approximately preserves distances, in the sense that the distortion is small: DISPLAYFORM0 for some uniform constants L > 0 and C ??? 1.

The distortion of the embedding ?? is the smallest C such that Equation 6 holds.

One can characterize how "large" a space is (its representational capacity) by the spaces that embed into it with low distortion.

In practical terms, this capacity determines the types of data (and relationships between them) that can be well-represented in the embedding space.

R n with the Euclidean metric, for example, embeds into the L 1 metric with low distortion, while the reverse is not true BID16 ).

We do not expect Manhattan-structured data to be well-represented in Euclidean space, no matter how clever the mapping.

Wasserstein spaces are very large: Many spaces can embed into Wasserstein spaces with low distortion, even when the converse is not true.

W p (A), for A an arbitrary metric space, embeds any product space A n , for example BID28 , via discrete distributions supported at n points.

Even more generally, certain Wasserstein spaces are universal, in the sense that they can embed arbitrary metrics on finite spaces.

W 1 ( 1 ) is one such space BID7 , and it is still an open problem to determine if W 1 (R k ) is universal for any k < +???. Recently it has been shown that every finite metric space embeds the 1 p power of its metric into W p (R 3 ), p > 1, with vanishing distortion BID1 .

A hopeful interpretation suggests that W 1 (R 3 ) may be a plausible target space for arbitrary metrics on symbolic data, with a finite set of symbols; we are unaware of similar universality results for L p or hyperbolic spaces, for example.

The reverse direction, embedding Wasserstein spaces into others, is well-studied in the case of discrete distributions.

Theoretical results in this domain are motivated by interest in efficient algorithms for approximating Wasserstein distances by embedding into spaces with easily-computed metrics.

In this direction, low-distortion embeddings are difficult to find.

W 2 (R 3 ), for example, is known not to embed into L 1 BID2 .

Some positive results exist, nevertheless.

For a Euclidean ground metric, for example, the 1-Wasserstein distance can be approximated in a wavelet domain BID46 or by high-dimensional embedding into L 1 BID25 .In ??4, we empirically investigate the embedding capacity of Wasserstein spaces, by attempting to learn low-distortion embeddings for a variety of input spaces.

For efficiency, we replace the Wasserstein distance by its entropically-regularized counterpart, the Sinkhorn divergence ( ??2.2).

The embedding capacity of Sinkhorn divergences is previously unstudied, to our knowledge, except in the weak sense that the approximation error with respect to the Wasserstein distance vanishes with the regularizer taken to zero BID10 BID19 .

While learned vector space embeddings have a long history, there is a recent trend in the representation learning community towards more complex target spaces, such as spaces of probability distributions BID54 BID4 , Euclidean norm balls BID36 BID35 , Poincar?? balls BID39 , and Riemannian manifolds BID40 .

From a modeling perspective, these more complex structures assist in representing uncertainty about the objects being embedded BID54 BID6 as well as complex relations such as inclusion, exclusion, hierarchy, and ordering BID36 BID51 BID4 .

In the same vein, our work takes probability distributions in a Wasserstein space as our embedding targets.

The distance or discrepancy measure between target structures is a major defining factor for a representation learning model.

L p distances as well as angle-based discrepancies are fairly common BID34 , as is the KL divergence BID31 , when embedding into probability distributions.

For distributions, however, the KL divergence and L p distances are problematic, in the sense that they ignore the geometry of the domain of the distributions being compared.

For distributions with disjoint support, for example, these divergences do not depend on the separation between the supports.

Optimal transport distances BID53 BID15 BID44 BID49 , on the other hand, explicitly account for the geometry of the domain.

Hence, models based on optimal transport are gaining popularity in machine learning; see (Rubner et al., 1998; BID13 BID18 BID32 BID3 BID20 BID11 for some examples.

Learned embeddings into Wasserstein spaces are relatively unexplored.

Recent research proposes embedding into Gaussian distributions BID37 BID58 .

Restricting to parametric distributions enables closed-form expressions for transport distances, but the resulting representation space may lose expressiveness.

We note that BID14 study embedding in the opposite direction, from Wasserstein into Euclidean space.

In contrast, we learn to embed into the space of discrete probability distributions endowed with the Wasserstein distance.

Discrete distributions are dense in W 2 BID29 BID8 ).

We consider the task of recovering a pairwise distance or similarity relationship that may be only partially observed.

We are given a collection of objects C-these can be words, symbols, images, or any other data-as well as samples DISPLAYFORM0 ) of a target relationship r : C ??C ??? R that tells us the degree to which pairs of objects are related.

Our objective is to find a map ?? : C ??? W p (X ) such that the relationship r(u, v) can be recovered from the Wasserstein distance between ??(u) and ??(v), for any u, v ??? C. Examples include:1.

METRIC EMBEDDING: r is a distance metric, and we want W p (??(u), ??(v)) ??? r(u, v) for all u, v ??? C. 2.

GRAPH EMBEDDING: C contains the vertices of a graph and r : C??C ??? {0, 1} is the adjacency relation; we would like the neighborhood of each ??(u) in W p to coincide with graph adjacency.

3.

WORD EMBEDDING: C contains individual words and r is a semantic similarity between words.

We want distances in W p to predict this semantic similarity.

Although the details of each task require some adjustment to the learning architecture, our basic representation and training procedure detailed below applies to all three examples.

Given a set of training samples DISPLAYFORM0 ??? C ?? C ?? R, we want to learn a map ?? : C ??? W p (X ).

We must address two issues.

First we must define the range of our map ??.

The whole of W p (X ) is infinite-dimensional, and for a tractable problem we need a finite-dimensional output.

We restrict ourselves to discrete distributions with an a priori fixed number of support points M , reducing optimal transport to the linear program in Equation 3.

Such a distribution is parameterized by the locations of its support points {x DISPLAYFORM1 , forming a point cloud in the ground metric space X .

For simplicity, we restrict to uniform weights u, v ??? 1, although it is possible to optimize simultaneously over weights and locations.

As noted in BID8 BID29 BID11 , however, when constructing a discrete M -point approximation to a fixed target distribution, allowing non-uniform weights does not improve the asymptotic approximation error.

The second issue is that, as noted in ??2.2, exact computation of W p in general is costly, requiring the solution of a linear program.

As in BID20 , we replace W p with the Sinkhorn divergence W ?? p , which is solvable by a the fixed-point iteration in Equation 5.

Learning then takes the form of empirical loss minimization: DISPLAYFORM0 over a hypothesis space of maps H. The loss L is problem-specific and scores the similarity between the regularized Wasserstein distance W

We first demonstrate the representational power of learned Wasserstein embeddings.

As discussed in ??2.3, theory suggests that Wasserstein spaces are quite flexible, in that they can embed a wide variety of metrics with low distortion.

We show that this is true in practice as well.

To generate a variety of metrics to embed, we take networks with various patterns of connectivity and compute the shortest-path distances between vertices.

The collection of vertices for each network serves as the input space C for our embedding, and our goal is to learn a map ?? : C ??? W p (R k ) such that the 1-Wasserstein distance W 1 (??(u), ??(v)) matches as closely as possible the shortest path distance between vertices u and v, for all pairs of vertices.

We learn a minimum-distortion embedding: Given a fully-observed distance metric d C : C ??C ??? R in the input space, we minimize the mean distortion: DISPLAYFORM0 ?? is parameterized as in ??3.2, directly specifying the support points of the output distribution.

We examine the performance of Wasserstein embedding using both random networks and real networks.

The random networks in particular allow us systematically to test robustness of the Wasserstein embedding to particular properties of the metric we are attempting to embed.

Note that these experiments do not explore generalization performance: We are purely concerned with the representational capacity of the learned Wasserstein embeddings.

For random networks, we use three standard generative models: Barab??si-Albert BID0 , BID55 , and the stochastic block model BID23 .

Random scale-free networks are generated from the Barab??si-Albert model, and possess the property that distances are on average much shorter than in a Euclidean spatial graph, scaling like the logarithm of the number of vertices.

Random small-world networks are generated from the Watts-Strogatz model; in addition to log-scaling of the average path length, the vertices of Watts-Strogatz graphs cluster into distinct neighborhoods.

Random communitystructured networks are generated from the stochastic block model, which places vertices within densely-connected communities, with sparse connections between the different communities.

We additionally generate random trees by choosing a random number of children 2 for each node, progressing in breadth-first order until a specified total number of nodes is reached.

In all cases, we generate networks with 128 vertices.

1 In both the non-uniform and uniform cases, the order of convergence in Wp of the nearest weighted point cloud to the target measure, as we add more points, is O(M ???1/d ), for a d-dimensional ground metric space.

This assumes the underlying measure is absolutely continuous and compactly-supported.2 Each non-leaf node has a number of children drawn uniformly from {2, 3, 4}. We compare against two baselines, trained using the same distortion criterion and optimization method: Euclidean embeddings, and hyperbolic embeddings.

Euclidean embeddings we expect to perform poorly on all of the chosen graph types, since they are limited to spatial relationships with zero curvature.

Hyperbolic embeddings model tree-structured metrics, capturing the exponential scaling of graph neighborhoods; they have been suggested for a variety of other graph families as well BID57 .

FIG3 shows the result of embedding random networks.3 As the total embedding dimension increases, the distortion decreases for all methods.

Importantly, Wasserstein embeddings achieve lower distortion than Euclidean and hyperbolic embeddings, establishing their flexibility under the varying conditions represented by the different network models.

In some cases, the Wasserstein distortion continues to decrease long after the other embeddings have saturated their capacity.

As expected, hyperbolic space significantly outperforms both Euclidean and Wasserstein specifically on tree-structured metrics.

We test R 2 , R 3 , and R 4 as ground metric spaces.

For all of the random networks we examined, the performance between R 3 and R 4 is nearly indistinguishable.

This observation is consistent with theoretical results ( ??2.3) suggesting that R 3 is sufficient to embed a wide variety of metrics.

We also examine fragments of real networks: an ArXiv co-authorship network, an Amazon product co-purchasing network, and a Google web graph BID33 .

For each graph fragment, we choose uniformly at random a starting vertex and then extract the subgraph on 128 vertices taken in breadth-first order from that starting vertex.

Distortion results are shown in Figure 2 .

Again the Wasserstein embeddings achieve lower distortion than Euclidean or hyperbolic embeddings.

Figure 2: Real networks: Learned Wasserstein embeddings achieve lower distortion than Euclidean and hyperbolic embeddings of real network fragments.one: f, two, i, after, four W ?? 1 (R 2 ) united: series, professional, team, east, central algebra: skin, specified, equation, hilbert, reducing one: two, three, s, four, after W ?? 1 (R 3 ) united: kingdom, australia, official, justice, officially algebra: binary, distributions, reviews, ear, combination one: six, eight, zero, two, three W ?? 1 (R 4 ) united: army, union, era, treaty, federal algebra: tables, transform, equations, infinite, differential Table 1 : Change in the 5-nearest neighbors when increasing dimensionality of each point cloud with fixed total length of representation.

In this section, we embed words as point clouds.

In a sentence s = (x 0 , . . .

, x n ), a word x i is associated with word x j if x j is in the context of x i , which is a symmetric window around x i .

This association is encoded as a label r; r xi,xj = 1 if and only if |i ??? j| ??? l where l is the window size.

For word embedding, we use a contrastive loss function BID22 ?? * = arg min ?? s xi,xj ???s DISPLAYFORM0 which tries to embed words x i , x j near each other in terms of 1-Wasserstein distance (here W ?? 1 ) if they co-occur in the context; otherwise, it prefers moving them at least distance m away from one another.

This approach is similar to that suggested by BID34 , up to the loss and distance functions.

We use a Siamese architecture BID9 for our model, with negative sampling (as in BID34 ) for selecting words outside the context.

The network architecture in each branch consists of a linear layer with 64 nodes followed by our point cloud embedding layer.

The two branches of the Siamese network connect via the Wasserstein distance, computed as in ??2.2.

The training dataset is Text8 4 , which consists of a corpus with 17M tokens from Wikipedia and is commonly used as a language modeling benchmark.

We choose a vocabulary of 8000 words and a context window size of l = 2 (i.e., 2 words on each side), ?? = 0.05, number of epochs of 3, negative sampling rate of 1 per positive and Adam BID26 for optimization.

We first study the effect of dimensionality of the point cloud on the quality of the semantic neighborhood captured by the embedding.

We fix the total number of output parameters, being the product of the number of support points and the dimension of the support space, to 63 or 64 parameters.

Table 1 shows the 5 nearest neighbors in the embedding space.

Notably, increasing the dimensionality directly improves the quality of the learned representation.

Interestingly, it is more effective to use a budget of 64 parameters in a 16-point, 4-dimensional cloud than in a 32-point, 2-dimensional cloud.

Next we evaluate these models on a number of benchmark retrieval tasks from BID17 , which score a method by the correlation of its output similarity scores with human similarity Table 2 : Performance on a number of similarity benchmarks when dimensionality of point clouds increase given a fixed total number of parameters.

The middle block shows the performance of the proposed models.

The right block shows the performance of baselines.

The training corpus size when known appears below each model name.

DISPLAYFORM1 judgments, for various pairs of words.

Results are shown in Table 2 .

The results of our method, which use Sinkhorn distance to compute the point cloud (dis)similarities, appear in the middle block of Table 2 .

Again, we mainly see gradual improvement with increasing dimensionality of the point clouds.

The right block in Table 2 shows baselines: Respectively, RNN(80D) BID30 , Metaoptimize (50D) BID50 , SENNA (50D) BID12 Global Context (50D) BID24 and word2vec (80D) BID34 .

In the right block, as in BID17 , the cosine similarity is used for point embeddings.

The reported performance measure is the correlation with ground-truth rankings, computed as in BID17 .

Note there are many ways to improve the performance: increasing the vocabulary/window size/number of epochs/negative sampling rate, using larger texts, and accelerating performance.

We defer this tuning to future work focused specifically on NLP.

Wasserstein embeddings over low-dimensional ground metric spaces have a unique property: We can directly visualize the embedding, which is a point cloud in the low-dimensional ground space.

This is not true for most existing embedding methods, which rely on dimensionality reduction techniques such as t-SNE for visualization.

Whereas dimensionality reduction only approximately captures proximity of points in the embedding space, with Wasserstein embeddings we can display the exact embedding of each input, by visualizing the point cloud.

We demonstrate this property by visualizing the learned word representations.

Importantly, each point cloud is strongly clustered, which leads to apparent, distinct modes in its density.

We therefore use kernel density estimation to visualize the densities.

In FIG5 , we visualize three distinct words, thresholding each density at a low value and showing its upper level set to reveal the modes.

These level sets are overlaid, with each color in the figure corresponding to a distinct embedded word.

The density for each word is depicted by the opacity of the color within each level set.

It is easy to visualize multiple sets of words in aggregate, by assigning all words in a set a single color.

This immediately reveals how well-separated the sets are, as shown in FIG5 : As expected, military and political terms overlap, while names of sports are more distant.

Examining the embeddings in more detail, we can dissect relationships (and confusion) between different sets of words.

We observe that each word tends to concentrate its mass in two or more distinct regions.

This multimodal shape allows for multifaceted relationships between words, since a word can partially overlap with many distinct groups of words simultaneously.

FIG5 shows the embedding for a word that has multiple distinct meanings (kind), alongside synonyms for both senses of the word (nice, friendly, type).

We see that kind has two primary modes, which overlap separately with friendly and type.

nice is included to show a failure of the embedding to capture the full semantics: FIG5 shows that the network has learned that nice is a city in France, ignoring its interpretation as an adjective.

This demonstrates the potential of this visualization for debugging, helping identify and attribute an error.

Several characteristics determine the value and effectiveness of an embedding space for representation learning.

The space must be large enough to embed a variety of metrics, while admitting a mathematical description compatible with learning algorithms; additional features, including direct interpretability, make it easier to understand, analyze, and potentially debug the output of a representation learning procedure.

Based on their theoretical properties, Wasserstein spaces are strong candidates for representing complex semantic structures, when the capacity of Euclidean space does not suffice.

Empirically, entropy-regularized Wasserstein distances are effective for embedding a wide variety of semantic structures, while enabling direct visualization of the embedding.

Our work suggests several directions for additional research.

Beyond simple extensions like weighting points in the point cloud, one observation is that we can lift nearly any representation space X to distributions over that space W(X ) represented as point clouds; in this paper we focused on the case X = R n .

Since X embeds within W(X ) using ??-functions, this might be viewed as a general "lifting" procedure increasing the capacity of a representation.

We can also consider other tasks, such as co-embedding of different modalities into the same transport space.

Additionally, our empirical results suggest that theoretical study of the embedding capacity of Sinkhorn divergences may be profitable.

Finally, following recent work on computing geodesics in Wasserstein space BID45 , it may be interesting to invert the learned mappings and use them for interpolation.

@highlight

We show that Wasserstein spaces are good targets for embedding data with complex semantic structure.

@highlight

Learns embeddings in a discrete space of probability distributions, using a minimized, regularised version of Wasserstein distances.

@highlight

The paper describes a new embedding method that embeds data to the space of probability measures endowed with the Wasserstein distance. 

@highlight

The paper proposes embedding the data into low-dimensional Wasserstein spaces, which can capture the underlying structure of the data more accurately.
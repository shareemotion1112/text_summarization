The quality of the representations achieved by embeddings is determined by how well the geometry of the embedding space matches the structure of the data.

Euclidean space has been the workhorse for embeddings; recently hyperbolic and spherical spaces have gained popularity due to their ability to better embed new types of structured data---such as hierarchical data---but most data is not structured so uniformly.

We address this problem by proposing learning embeddings in a product manifold combining multiple copies of these model spaces (spherical, hyperbolic, Euclidean), providing a space of heterogeneous curvature suitable for a wide variety of structures.

We introduce a heuristic to estimate the sectional curvature of graph data and directly determine an appropriate signature---the number of component spaces and their dimensions---of the product manifold.

Empirically, we jointly learn the curvature and the embedding in the product space via Riemannian optimization.

We discuss how to define and compute intrinsic quantities such as means---a challenging notion for product manifolds---and provably learnable optimization functions.

On a range of datasets and reconstruction tasks, our product space embeddings outperform single Euclidean or hyperbolic spaces used in previous works, reducing distortion by 32.55% on a Facebook social network dataset.

We learn word embeddings and find that a product of hyperbolic spaces in 50 dimensions consistently improves on baseline Euclidean and hyperbolic embeddings, by 2.6 points in Spearman rank correlation on similarity tasks and 3.4 points on analogy accuracy.

With four decades of use, Euclidean space is the venerable elder of embedding spaces.

Recently, non-Euclidean spaces-hyperbolic BID27 BID33 and spherical BID42 BID24 )-have gained attention by providing better representations for certain types of structured data.

The resulting embeddings offer better reconstruction metrics: higher mean average precision (mAP) and lower distortion compared to their Euclidean counterparts.

These three spaces are the model spaces of constant curvature BID21 , and this improvement in representation fidelity arises from the correspondence between the structure of the data (hierarchical, cyclical) and the geometry of non-Euclidean space (hyperbolic: negatively curved, spherical: positively curved).

The notion of curvature plays the key role.

To improve representations for a variety of types of data-beyond hierarchical or cyclical-we seek spaces with heterogeneous curvature.

The motivation for such mixed spaces is intuitive: our data may have complicated, varying structure, in some regions tree-like, in others cyclical, and we seek the best of all worlds.

We expect mixed spaces to match the geometry of the data and thus provide higher quality representations.

However, to employ these spaces, we face several key obstacles.

We must perform a challenging manifold optimization to learn both the curvature and the embedding.

Afterwards, we also wish to operate on the embedded points.

For example, analogy operations for word embeddings in Euclidean vector space (e.g., a − b + c) must be lifted to manifolds.

2 , Euclidean plane E 2 , and hyperboloid H 2 .

Thick lines are geodesics; these get closer in positively curved (K = +1) space S 2 , remain equidistant in flat (K = 0) space E 2 , and get farther apart in negatively curved (K = −1) space H 2 .We propose embedding into product spaces in which each component has constant curvature.

As we show, this allows us to capture a wider range of curvatures than traditional embeddings, while retaining the ability to globally optimize and operate on the resulting embeddings.

Specifically, we form a Riemannian product manifold combining hyperbolic, spherical, and Euclidean components and equip it with a decomposable Riemannian metric.

While each component space in the product has constant curvature (positive for spherical, negative for hyperbolic, and zero for Euclidean), the resulting mixed space has non-constant curvature.

However, selecting appropriate curvatures for the embedding space is a potential challenge.

We directly learn the curvature for each component space along with the embedding (via Riemannian optimization), recovering the correct curvature, and thus the matching geometry, directly from data.

We show empirically that we can indeed recover non-uniform curvatures and improve performance on reconstruction metrics.

Another technical challenge is to select the underlying number of components and dimensions of the product space; we call this the signature.

This concept is vacuous in Euclidean space: the product of E r1 , . . .

, E rn is identical to the single space E r1+...+rn .

However, this is not the case with spherical and hyperbolic spaces.

For example, the product of the spherical space S 1 (the circle) with itself is the torus S 1 × S 1 , which is topologically distinct from the sphere S 2 .

We address this challenge by introducing a theory-guided heuristic estimator for the signature.

We do so by matching an empirical notion of discrete curvature in our data with the theoretical distribution of the sectional curvature, a fine-grained measure of curvature on Riemannian manifolds that is amenable to analysis in products.

We verify that this approach recovers the correct signature on reconstruction tasks.

Standard techniques such as PCA require centering so that the embedded directions capture components of variation.

Centering in turn needs an appropriate generalization of the mean.

We develop a formulation of mean for embedded points that exploits the decomposability of the distance and has theoretical guarantees.

For T = {p 1 , . . .

, p n } in a manifold M with dimension r, the mean is µ(T ) := arg min p i d 2 M (p, p i ).

We give a global existence result: under symmetry conditions on the distribution of the points in T on the spherical components, gradient descent recovers µ(T ) with error ε in time O(nr log ε −1 ).We demonstrate the advantages of product space embeddings through a variety of experiments; products are at least as good as single spaces, but can offer significant improvements when applied to structures not suitable for single spaces.

We measure reconstruction quality (via mAP and distortion) for synthetic and real datasets over various allocations of embedding spaces.

We observe a 32.55% improvement in distortion versus any single space on a Facebook social network graph.

Beyond reconstruction, we apply product spaces to skip-gram word embeddings, a popular technique with numerous downstream applications, which crucially require the use of the manifold structure.

We find that products of hyperbolic spaces improve performance on benchmark evaluations-suggesting that words form multiple smaller hierarchies rather than one larger one.

We see an improvement of 3.4 points over baseline single spaces on the Google word analogy benchmark and of 2.6 points in Spearman rank correlation on a word similarity task using the WS-353 corpus.

Our results and initial exploration suggest that mixed product spaces are a promising area for future study.

Embeddings For metric spaces 1 U, V equipped with distances d U , d V , an embedding is a mapping f : U → V .

The quality of an embedding is measured by various fidelity measures.

A standard measure is average distortion D avg .

The distortion of a pair of points a, b is ( a, b) , and D avg is the average over all pairs of points.

DISPLAYFORM0 Distortion is a global metric; it considers the explicit value of all distances.

At the other end of the global-local spectrum of fidelity measures is mean average precision (mAP), which applies to unweighted graphs.

Let G = (V, E) be a graph and node a ∈ V have neighborhood N a = {b 1 , . . .

, b deg(a) }, where deg(a) is the degree of a. In the embedding f , define R a,bi to be the smallest ball around f (a) that contains b i (that is, R a,bi is the smallest set of nearest points required to retrieve the ith neighbor of a in f ).

Then, mAP(f ) = DISPLAYFORM1 Note that mAP does not track explicit distances; it is a ranking-based measure for local neighborhoods.

Observe that mAP(f ) ≤ 1 (higher is better) while d avg ≥ 0 (lower is better).

We briefly review some notions from manifolds and Riemannian geometry.

A more in-depth treatment can be found in standard texts BID22 BID10 .

Let M be a smooth manifold, p ∈ M be a point, and T p M be the tangent space to the point p.

If M is equipped with a Riemannian metric g, then the pair (M, g) is called a Riemannian manifold.

The shortest-distance paths on manifolds are called geodesics.

To compute distance functions on a Riemannian manifold, the metric tensor g is integrated along the geodesic.

This is a smoothly varying function (in p) g : T p M × T p M → R that induces geometric notions such as length and angle by defining an inner product on the tangent space.

For example, the norm of v ∈ T p M is defined as DISPLAYFORM0 , and the metric tensor g E is simply the normal inner product.

Product Manifolds Consider a sequence of smooth manifolds M 1 , M 2 , . . .

, M k .

The product manifold is defined as the Cartesian product DISPLAYFORM1 Notationally, we write points p ∈ M through their coordinates p = (p 1 , . . .

, p k ) : p i ∈ M i , and similarly a tangent vector DISPLAYFORM2 If the M i are equipped with metric tensor g i , then the product M is also Riemannian with metric tensor DISPLAYFORM3 That is, the product metric decomposes into the sum of the constituent metrics.

Geodesics and Distances Optimization on manifolds requires a notion of taking a step.

This step can be performed in the tangent space and transferred to the manifold via the exponential map Exp p : T p M → M .

In a product manifold P, for tangent vectors v = (v 1 , . . .

, v k ) at p = (p 1 , . . . , p k ) ∈ M , the exponential map simply decomposes, as do squared distances BID12 BID36 : DISPLAYFORM4 In other words, the shortest path between points in the product travels along the shortest paths in each component simultaneously.

Note the analogy to Euclidean products DISPLAYFORM5

We use the hyperboloid model of hyperbolic space, with points in R d+1 .

Let J ∈ R (d+1)×(d+1) be the diagonal matrix with J 00 = −1 and DISPLAYFORM0 the corresponding norm is p * = p, p DISPLAYFORM1 When the subscript K is omitted, it is taken to be 1.

DISPLAYFORM2 Similarly, spherical space S d K is most easily defined when embedded in R d+1 .

The manifold is defined on the subset {p ∈ R d+1 : DISPLAYFORM3

We now tackle the challenges of mixed spaces.

First, we introduce a product manifold embedding space P composed of multiple copies of simple model spaces, providing heterogeneous curvature.

Next, in Section 3.1, given the signature of P (the number of components of each type and their dimensions), we describe how to simultaneously learn an embedding and the curvature for each component through optimization.

In Section 3.2, we provide a heuristic to choose the signature by estimating a discrete notion of curvature for given data.

Finally, in Section 3.3, given an embedding in P, we introduce a Karcher-style mean which can be recovered efficiently.

DISPLAYFORM0 a product manifold with m + n + 1 component spaces and total dimension i s i + j h j + e.

We refer to each S si , H hi , E e as components or factors.

We refer to the decomposition, e.g., (H 2 ) 2 = H 2 ×H 2 , as the signature.

For convenience, let M 1 , . . .

, M m+n+1 refer to the factors in the product.

Distances on P As discussed in Section 2, the product P is a Riemannian manifold defined by the structure of its components.

For p, q ∈ P, we write d Mi (p, q) for the distance d Mi restricted to the appropriate components of p and q in the product.

In particular, the squared distance in the product decomposes via (1).

In other words, d P is simply the 2 norm of the component distances d Mi .We note that P can also be equipped with different distances (ignoring the Riemannian structure), leading to a different embedding space.

Without the underlying manifold structure, we cannot freely operate on the embedded points such as taking geodesics and means, but some simple applications only interact through distances.

For such settings, we consider the 1 distance DISPLAYFORM1 These distances provide simple and interpretable embedding spaces using P, enabling us to introduce combinatorial constructions that allow for embeddings without the need for optimization.

We give an example below and discuss further in the Appendix.

We then focus on the Riemannian distance, which allows Riemannian optimization directly on the manifold, and enables full use of the manifold structure in generic downstream applications.

Example Consider the graph G shown on the right of FIG1 .

This graph has a backbone cycle with 9 nodes, each attached to a tree; such topologies are common in networking.

If a single edge (a, b) is removed from the cycle, the result is a tree embeddable arbitrarily well into hyperbolic space BID33 .

However, a, b (and their subtrees) would then incur an additional distance of 8 − 1 = 7, being forced to go the other way around the cycle.

But using the 1 distance, we can embed G tree into H 2 and G cycle into S 1 , yielding arbitrarily low distortion for G. We give the full details and another combinatorial construction for the min-distance in the Appendix.

To compute embeddings, we optimize the placement of points through an auxiliary loss function.

Given graph distances {d G (X i , X j )} ij , our loss function of choice is DISPLAYFORM0 Algorithm 1 R-SGD in products 1: Input: Loss function L : P → R 2: Initialize x (0) ∈ P randomly 3: for t = 0, . . .

, T − 1 do

h ← ∇L(x (t) )

for i = 1, . . .

, m do 6: DISPLAYFORM0 for i = m + 1, . . .

, m + n do 8: DISPLAYFORM1 11:for i = 1, . . .

, m + n + 1 do 12: which captures the average distortion.

(2) depends on hyperbolic distance d H (for which the gradient is unstable) only through the square d 2 H , which is continuously differentiable BID33 .

In any Riemannian manifold, a loss function can be optimized through standard Riemannian optimization methods such as RSGD BID1 and RSVRG .

We write down the full RSGD specialized to our product spaces in Algorithm 1.

This proceeds by first computing the Euclidean gradient ∇L(x) with respect to the ambient space of the embedding (Step 4), and then converting it to the Riemannian gradient by applying the Riemannian correction (multiply by the inverse of the metric tensor g −1 P ).

This overall strategy has been detailed in previous work in the hyperboloid model BID28 , and the same calculations apply to our hyperbolic components.

DISPLAYFORM2 Since g P is block diagonal on a product manifold, it suffices to apply the correction and perform the gradient step in each component M i independently.

In the spherical and hyperboloid models, which have smaller dimension than the ambient space, this is performed by first projecting the gradient vector h onto the tangent space T x M via proj S x (h) = h − h, x x (Step 6) and proj DISPLAYFORM3 Step 8).

In the hyperboloid model, a final rescaling by the inverse of the metric J is needed (Step 9).

This is not required in the spherical model since it inherits the same metric from the ambient Euclidean space.

Learning the Curvature There exists a spherical model for every curvature K > 0 (for example, the sphere S d K of radius K −1/2 ) and a hyperbolic model for every K < 0 (the hyperboloid H d −K ).

We jointly optimize the curvature K i of every non-Euclidean factor M i along with the embeddings.

The idea is that distances on the spherical and hyperboloid models of arbitrary curvature can be emulated through distances on the standard models S, H of curvature 1.

For example, given p, q on the sphere S 1/R 2 of radius R, then d(p, q) = R · d S1 (p/R, q/R) where p/R, q/R lie on the unit sphere.

Therefore the radius R, which is monotone in the curvature K, can be treated as a parameter as well, so that we can optimize K and implicitly represent points lying on the manifold of curvature K, while explicitly only needing to store and optimize points in the standard model of curvature 1 via Algorithm 1.

The hyperboloid model is analogous.

Moreover, the loss (2) depends only on squared distances on the product manifold, which are simple functions of distances in the components through (1), so we can optimize the curvature of each factor in P.

To choose the signature of an appropriate space P corresponding to given data, we again turn to curvature.

We use the sectional curvature, a finer-grained notion defined over all two-dimensional subspaces passing through a point.

Unlike coarser notions like scalar curvature, this is not constant in a product of basic spaces.

Given linearly independent u, v ∈ T p M spanning a two-dimensional subspace U , the sectional curvature DISPLAYFORM0 is defined as the Gaussian curvature of the surface Exp(U ) ⊆ M .

Intuitively, this captures the rate that geodesics on the surface emanating from p spread apart, which relates to volume growth.

In Appendix C.2, we show that the sectional curvature of P interpolates between the sectional curvatures of the factors, enabling us to better capture a wider range of structures in our embeddings: DISPLAYFORM1 Our estimation technique employs a triangle comparison theorem following from Toponogov's theorem and the law of cosines, which characterizes sectional curvature through the behavior of small triangles (note that a triangle determines a 2-dimensional submanifold).

Let abc be a geodesic triangle in manifold (or metric space) M and m be the (geodesic) midpoint of bc, and consider the quantity DISPLAYFORM2 This is non-negative (resp.

non-positive) when the curvature is non-negative (resp.

non-positive).

Note that consequently the equality case occurs exactly when the curvature is 0, and equation 3 becomes the parallelogram law of Euclidean geometry ( FIG2 ).Analogous to sectional curvature, which is a function of a point p and two directions x, y from p, in an undirected graph G we define an analog for every node m and two neighbors b, c. Given a reference node a we set: DISPLAYFORM3 .

This is exactly the expression from equation 3, normalized suitably so as to yield the correct scaling for trees and cycles.

Our curvature estimation is then a simple average ξ G (m; b, c) = , c; a) .

Importantly, ξ G recovers the right curvature for graph atoms such as lines, cycles, and trees (Appendix C.2, Lemma 4,5), and the correct sign for other special discrete objects like polyhedra (Thurston, 1998).

The curvature is zero for lines, positive for cycles, and negative for trees.

DISPLAYFORM4 For a generic graph G, we use this to generate a potential product manifold to embed in.

An empirical sectional curvature of G is estimated via Algorithm 3, which is based off the homogeneity of product manifolds (i.e. isometries act transitively), implying that it suffices to analyze the curvature at a random point.

In particular, we moment-match the distributions of sectional curvature through uniformly random 2-planes in the graph and in the manifold through Algorithms 3,2 (Appendix C.2).

A critical operation on manifolds is that of taking the mean; it is necessary for many downstream applications, including, for example, analogy tasks with word embeddings, for clustering, and for centering before applying PCA.

Even in simple settings like the circle S 1 , defining a mean is nontrivial.

A classic approach is to take the Euclidean mean (in E 2 ) of the points and to project back onto S 1 -but this operation fails in the case where the points are uniformly spaced on S 1 .

A further roadblock is the varying curvature of P. Fortunately, we can exploit the decomposability of the distance on P, reducing the challenge to breaking symmetries in the component spaces.

To do so, we introduce the following Karcher-style weighted mean.

Let T = {p 1 , p 2 , . . .

, p n } be a set of points in P and w 1 , . . .

, w n be positive weights satisfying DISPLAYFORM0 In special cases, this matches commonly used means (the centroid in the Euclidean case E d , the spherical average for S 2 in BID5 ).

We further note that when w i ≥ 0, the squared-distance components above are individually convex: this is trivial in the Euclidean term, holds in the hyperbolic case (cf.

Theorem 4.1 BID0 ), and holds in the spherical case under certain restrictions, e.g., when the points in T lie entirely in one hemisphere of S r BID5 .

Moreover, in this case, peforming the optimization on the mean with gradient descent via the exponential map offers linear rate convergence:Lemma 2.

Let P be a product of model spaces of total dimension r, T = {p 1 , . . .

, p n } points in P and w 1 , . . . , w n weights satisfying w i ≥ 0 and n i=1 w i = 1.

Moreover, let the components of the points in P, p i|S j restricted to each spherical component space S j fall in one hemisphere of S j .

Then, Riemannian gradient descent recovers the mean µ(T ) within distance in time O(nr log −1 ).This is a global result; with weaker assumptions, we can derive local results; for example, in the case where some of the w i are negative, which is useful for analogy operations.

In summary, we offer the following key takeaways of our development:• Product manifolds of model spaces capture heterogeneous curvature while providing tractable optimization,• Each component's curvature can be learned empirically through a reparametrization,• A signature for the product can be found by matching discrete notions of curvature on graphs with sectional curvature on manifolds,• There exists an easily-computed formulation of mean with theoretical guarantees.

We evaluate the proposed approach, comparing the representation quality of synthetic graphs and real datasets among different embedding spaces by measuring the reconstruction fidelity (through average distortion and mAP).

We expect that mixed product spaces perform better for nonhomogeneous data.

We consider the curvature of graphs, reporting the curvatures learned through optimization as well as the theoretical allocation from Section 3.2.

Beyond reconstruction, we evaluate the intrinsic performance of product space embeddings in a skip-gram word embedding model, by defining tasks with generic manifold operations such as means.

Datasets We examine synthetic datasets-trees, cycles, the ring of trees shown in FIG0 , confirming that each matches its theoretically optimal embedding space.

We then compare on several real-world datasets with describable structure, including the USCA312 dataset of distances between North American cities (Burkardt); a tree-like graph of computer science Ph.D. advisor-advisee relationships BID8 reported in previous hyperbolics work BID33 ; a powergrid distribution network with backbone structure BID39 ; and a dense social network from Facebook BID25 .

For the former two graphs with well-defined structure, we expect optimal embeddings in spaces of positive and negative curvature, respectively.

We hypothesize that the backbone network embeds well into simple products of hyperbolic and spherical spaces as in FIG1 , and the dense graph also benefits from a mixture of spaces.

Approaches We minimize the loss (2) using Algorithm 1.

We fix a total dimension d and consider the most natural ways to construct product manifolds of the given dimension, through iteratively 3 For a given signature, the curvatures are initialized to the appropriate value in {−1, 0, 1} and then learned using the technique in Section 3.1.

We additionally compare to the outputs of Algorithms 2,3 for heuristically selecting a combination of spaces in which to embed these datasets.

Quality We focus on the average distortion-which our loss function (2) optimizes-as our main metric for reconstruction, and additionally report the mAP metric for the unweighted graphs.

As expected, for the synthetic graphs (tree, cycle, ring of trees), the matching geometries (hyperbolic, spherical, product of hyperbolic and spherical) yield the best distortion TAB1 .

Next, we report in TAB2 the quality of embedding different graphs across a variety of allocations of spaces, fixing total dimension d = 10 following previous work BID28 .

We confirm that the structure of each graph informs the best allocation of spaces.

In particular, the cities graph-which has intrinsic structure close to S 2 -embeds well into any space with a spherical component, and the treelike Ph.

D.s graph embeds well into hyperbolic products.

We emphasize that even for such datasets that theoretically match a single constant-curvature space, the products thereof perform no worse.

In general, the product construction achieves high quality reconstruction: the traditional Euclidean approach is often well below several other signatures.

We additionally report the learned curvatures associated with the optimal signature, finding that the resulting curvatures are non-uniform even for products of identical spaces (cf.

Ph.

D.s).

Finally, Table 3 reports the signature estimations of Algorithms 2, 3 for the unweighted graphs.

Among the signatures over two components, the estimated curvature signs agree with best distortion results from Table 2.

To investigate the performance of product space embeddings in applications requiring the underlying manifold structure, we learned word embeddings and evaluated them on benchmark datasets for word similarity and analogy.

In particular, we extend results on hyperbolic skip-gram embeddings from (LW), who found that hyperbolic embeddings perform favorably against Euclidean word vectors in low dimensions (d = 5, 20) , but less so in higher dimensions (d = 50, 100).

Building on these results, we hypothesize that in high dimensions, a product of multiple smaller-dimension hyperbolic spaces will substantially improve performance.

Setup We use the standard skip-gram model BID26 and extend the loss function to a generic objective suitable for arbitrary manifolds, which is a variant of the objective proposed by LW.

Concretely, given a word u and target w, with label y = 1 if w is a context word for u and y = 0 if it is a negative sample, the model is P (y|w, u) = σ (−1) 1−y (− cosh(d(α u , γ w )) + θ) .Training followed the setup of LW, building on the fastText skip-gram implementation.

Euclidean results are reported directly from fastText.

Aside from choice of model, the training setup including hyperparameters (window size, negative samples, etc.) is identical to LW for all models.

Word Similarity We measure the Spearman rank correlation ρ between our scores and annotated ratings on the word similarity datasets WS-353 BID13 ), Simlex-999 (Hill et al., 3 Note that S 1 and H 1 are metrically equivalent to R, so these are not considered.

Table 3 : Heuristic allocation: estimated signatures for embedding unweighted graphs from TAB2 into two factors, using Algorithms 2,3 to match the empirical distribution of graph curvature.

The resulting curvature signs agree with results from TAB2 for choosing among two-component spaces.

Estimated Signature H 2015) and MEN BID3 .

The results are in TAB3 .

Notably, we find that hyperbolic word embeddings are consistently competitive with or better than the Euclidean embeddings, and the improvement increases with more factors in the product.

This suggests that word embeddings implicitly contain multiple distinct but smaller hierarchies rather than forming a single larger one.

Analogies In manifolds, there is no exact analog of the "word arithmetic" of conventional word embeddings arising from vector space structure.

However, analogies can still be defined via intrinsic product manifold operations.

In particular, note that the loss function depends on the embeddings solely through their pairwise distances.

We thus define analogies a : DISPLAYFORM0 through constructing an analog of the parallelogram, by geodesically reflecting a through the geodesic midpoint (i.e. mean) m of b, c. Note that this defines both the loss function and the intrinsic tasks purely in terms of distances and manifold operations.

Hence, unlike traditional word embeddings, this formulation is generic to any space.

Our evaluation, shown in Table 5 , uses the standard Google word analogy benchmark BID26 .

We observe a 22% accuracy improvement over single-space hyperbolic embeddings in 50 dimensions and similar improvements over a single hyperbolic space in 100 dimensions.

As with similarity, accuracy on the analogy task consistently improves as the number of factors increases.

Product spaces enable improved representations by better matching the geometry of the embedding space to the structure of the data.

We introduced a tractable Riemannian product manifold class that combines Euclidean, spherical, and hyperbolic spaces.

We showed how to learn embeddings and curvatures, estimate the product signature, and defined a tractable formulation of mean.

We hope that our techniques encourage further research on non-Euclidean embedding spaces.

Table 5 : Accuracy on the Google word analogy dataset.

Taking products of smaller hyperbolic spaces significantly improves performance.

Unlike conventional embeddings, the operations in hyperbolic and product spaces are defined solely through distances and manifold operations.

The Appendix starts with a glossary of symbols and a discussion of related work.

Afterwards, we provide the proof of Lemma 2.

We continue with a more in-depth treatment of the curvature estimation algorithm.

We then introduce two combinatorial constructions-embedding techniques that do not require optimization-that rely on the alternative product distances.

We give additional details on our experimental setup.

Finally, we additionally evaluate the interpretability of these embeddings (i.e., do the separate components in the embedding manifold capture intrinsic qualities of the data?) through visualizations of the synthetic example from FIG0 .

DISPLAYFORM0

We provide a glossary of commonly-used terms in our paper.

Used for mAP(f ) the mean average precision fidelity measure of the embedding f D(f ) the distortion fidelity measure of the embedding f D wc (f ) the worst-case distortion fidelity measure of the embedding f G a graph, typically with node set V and edge set E T a tree a, b, c nodes in a graph or tree f an embedding N a neighborhood around node a in a graph R a,b the smallest set of closest points to node a in an embedding f that contains node b M a manifold; when equipped with a metric g, M is Riemannian p a point in a manifold, p ∈ M T p M the tangent space of point p in M (a vector space) g a Riemannian metric defining an inner product on DISPLAYFORM0 product manifold consisting of spherical, Euclidean, hyperbolic factors Exp x (v) the exponential map for tangent vector v at point x R the Riemannian curvature tensor K(x, y) the sectional curvature for a subspace spanned by linearly independent x, y ∈ T p M d E metric distance between two points in Euclidean space d S metric distance between two points in spherical space d H metric distance between two points in hyperbolic space d U metric distance between two points in metric space U d G metric distance between two points in a graph G = (V, E) µ(T ) mean of a set of points T = {p 1 , . . .

, p n } in P I n the n × n identity matrix Table 6 : Glossary of variables and symbols used in this paper.

Hyperbolic space has recently been proposed as an alternative to Euclidean space to learn embeddings in cases where there is a (possibly latent) hierarchical structure.

In fact, many types of data (from various domains) such as social networks, word frequencies, metabolic-mass relationships, and phylogenetic trees of DNA sequences exhibit a non-Euclidean latent structure, as shown in BID2 .Initial works on hyperbolic embeddings include BID27 and BID6 .

In BID6 , neural graph embeddings are performed in hyperbolic space and used to classify the vertices of complex networks.

A similar application is link prediction in BID27 for the lexical database WordNet; this work also measured predicted lexical entailment on the HyperLex benchmark dataset.

The follow-up work BID28 performs optimizations in the hyperboloid (i.e. Lorentz) model instead of the Poincaré model.

BID34 proposed a neural ranking based question answering (Q/A) system in hyperbolic space that outperformed many state-of-the-art models using fewer parameters compared to competitor learning models.

BID15 proposed hyperbolic embeddings of entailment relations, described by directed acyclic graphs by applying hyperbolic cones as a heuristic and showed improvements over baselines in terms of representational capacity and generalization.

BID33 developed a combinatorial construction for efficiently embedding trees and tree-like graphs without optimization, studied the fundamental tradeoffs of hyperbolic embeddings, and explored PCA-like algorithms in hyperbolic space.

Unlike Euclidean space, most Riemannian manifolds are not vector spaces, and thus even basic operations such as vector addition, vector translation and matrix multiplication do not have universal interpretations.

In more complex geometries, closed form expressions for basic objects like distances, geodesics, and parallel transport do not exist.

As a result, standard machine learning or deep learning tools, such as convolutional neural networks, long short term memory networks (LSTMs), logistic regression, support vector machines, and attention mechanisms, do not have exact correspondences in these complex geometries.

A pair of recent approaches seek to formulate standard machine learning methods in hyperbolic space.

BID17 introduces a hyperbolic version of the attention mechanism using the hyperboloid model.

This work shows improvements in terms of generalization on several downstream applications including neural machine translation, learning on graphs and visual question answering tasks, while having compact neural representations.

BID16 formulates basic machine learning tools in hyperbolic space including multinomial logistic regression, feed-forward and recurrent neural networks like gated recurrent units and LSTMs in order to embed sequential data and perform classification in hyperbolic space.

They demonstrate empirical improvements on textual entailment and noisy-prefix recognition tasks using hyperbolic sentence embeddings.

BID7 introduced a hyperbolic formulation for support vector machine classifiers and demonstrated performance improvements for multi-class prediction tasks on real-world complex networks as well as simulated datasets.

Zipf's law states that word-frequency distributions obey a power law, which defines a hierarchy based on semantic specificities.

Concretely, semantically general words that occur in a wider range of contexts are closer to the root of the hierarchy while rarer words are further down in the hierarchy.

In order to capture the latent hierarchy in the natural language, there has been several proposals for training word embeddings in hyperbolic space.

BID9 trains word embeddings using the algorithm from BID27 .

They show that resulting hyperbolic word embeddings perform better on inferring lexical entailment relation than Euclidean embeddings trained with skip-gram model which is a standard method for training word embeddings, initially proposed by BID26 .

formulated the skip-gram loss function in hyperboloid model of hyperbolic space and evaluated on the standard the intrinsic evaluation tasks for word embeddings such as similarity and analogy in hyperbolic space.

Finally, the popularity of hyperbolic embeddings has stimulated interest in descent methods suitable for hyperbolic space optimization.

In addition to tools like BID1 and , offers convergence rate analysis for a variety of algorithms and settings for Hadamard manifolds.

BID11 proposes an explicit update rule along geodesics in a hyperbolic space with a theoretical guarantee on convergence, and BID44 introduces an accelerated Riemannian gradient methods.

Our work also touches on previous work on maximum distance scaling (MDS) and PCA-like algorithms in hyperbolic, spherical, and more general manifolds.

MDS-like algorithms in hyperbolic space are developed for visualization in BID38 and BID20 .

Embeddings into spherical or into hyperbolic space with a PCA-like loss function were developed in BID42 .

General forms of PCA include Geodesic PCA BID19 and principal geodesics analysis (PGA) BID14 .

A very general study of PCA-like algorithms is found in BID32 .

Below, we include proofs of our results and further discuss manifold notions such as curvature.

We begin with Lemma 2, restated below for convenience.

Lemma 2.

Let P be a product of model spaces of total dimension r, T = {p 1 , . . .

, p n } points in P and w 1 , . . . , w n weights satisfying w i ≥ 0 and n i=1 w i = 1.

Moreover, let the components of the points in P, p i|S j restricted to each spherical component space S j fall in one hemisphere of S j .

Then, Riemannian gradient descent recovers the mean µ(T ) within distance in time O(nr log −1 ).Proof.

Consider the squared distance d 2 (p, q) for p, q ∈ M for a manifold M .

Fix q. We denote the Hessian in p by H p,M (q).

Then, we have the following expressions for the Hessian of the squared distance of a sphere, derived in Pennec DISPLAYFORM0 where I r is the identity matrix, θ = acos(p · q) is the distance d S r (p, q) and u = (I r − pp T )q/ sin θ.

In Pennec, it is shown that the eigenvectors of H p,S r (q) are 0, 1, and θ cot(θ); thus the Hessian is bounded and if θ ∈ [0, π/2], it is also positive definite (PD).For hyperbolic space (under the hyperboloid model), the Hessian is DISPLAYFORM1 Here θ = acosh(− p, q * )

, J is the matrix associated with the Minkowski inner product, i.e., p, q * = p T Jq, and u = log p (q)/θ.

The log here refers to the logarithmic map.

That is, if q = exp p (v), then v = log p (q).

Moreover, exact expressions for the eigenvalues of H p,H r (q) in terms of θ imply that it is always bounded and PD.The Hessian for Euclidean space is H p,E r (q) = 2I r , which is also PD.

Now we can express the Hessian of the weighted mean.

We write H p,P for the Hessian of the weighted variance DISPLAYFORM2 We have, by the decomposability of the distance, that DISPLAYFORM3 Taking the Hessian, DISPLAYFORM4 Now, by assumption, the spherical components for our points in each of the spheres, p i|S j , fall within one hemisphere, and we may initialize our gradient descent (that is, our p 0 ) within this hemisphere.

Then, the angle θ in each of the spherical distances is in [0, π/2], so that the corresponding Hessians are PD.Since each term in the sum is PD and the weights satisfy w i ≥ 0, with at least one positive weight, H p,P is also PD.

Moreover, these Hessians are bounded.

Then we apply Theorem 4.2 (Chap.

7.4) in BID37 ), which shows linear rate convergence, as desired.

We discuss the notions of curvature relevant to our product manifold in more depth.

We start with a high-level overview of various definitions of curvature.

Afterwards, we introduce the formal definitions for curvature and apply them to the product construction.

Definitions of Curvature There are multiple notions of curvature, with varying granularity.

Some of these notions are suitable for working with manifolds abstractly (without reference to an ambient space, that is, intrinsic).

Others, in particular older definitions pre-dating the development of the formal mechanisms underpinning differential geometry, require the use of the ambient space.

Gauss defined the first intrinsic notion of curvature, Gaussian curvature.

It is the product of the principal curvatures, which can be thought of as the smallest and largest curvature in different directions.

4 Below we consider several such notions.

Scalar curvature is a single value associated with a point p ∈ M and intuitively relates to the area of geodesic balls.

Negative curvature means volumes grow faster than in Euclidean space, positive means volumes grow slower.

A more fine-grained notion of curvature is that of sectional curvature: it varies over all "sheets" passing through p. Note that curvature is inherently a notion of two-dimensional surfaces, and the sectional curvature fully captures the most general notion of curvature (the Riemannian curvature tensor).

More formally, for every two dimensional subspace U of the tangent space T p M , the sectional curvature K(U ) is equal to the Gaussian curvature of the sheet Exp p (U ).

Intuitively, it measures how far apart two geodesics emanating from p diverge.

In positively curved spaces like the sphere, they diverge more slowly than in flat Euclidean space.

The Ricci curvature of a tangent vector v at p is the average of the sectional curvature K(U ) over all planes U containing v. Geometrically the Ricci curvature measures how much the volume of a small cone around direction v compares to the corresponding Euclidean cone.

Positive curvature implies smaller volumes, and negative implies larger.

Note that this is natural from the way geodesics bend in various curvatures.

The scalar curvature is in fact defined as an average over the Ricci curvature, giving the intuitive relation between scalar curvature and volume.

It is thus also an average over the sectional curvature.

Discrete Analogs of Curvature Discrete data such as graphs do not have manifold structure.

The goal of curvature analogs such as ξ is to provide a discrete analog of curvature which satisfies similar properties to curvature; we use this to facilitate choosing an appropriate Riemannian manifold to embed discrete data into.

In this work, we focus on the sectional curvature, but discrete versions of other curvatures have been proposed such as the the Forman-Ricci BID40 and OllivierRicci (Ollivier, 2009) curvatures.

The input to the discrete curvature estimation from Section 3.2 is analogous to other discrete curvature analogs.

For example, the Ricci curvature is defined for a point p and a tangent vector u, and the coarse Ricci curvature is defined for a node p and neighbor x BID30 .

Similarly, the sectional curvature is defined for a point and two tangent vectors, and ξ is defined for a a node and two neighbors.

Sectional Curvature in Product Spaces Now we are ready to tackle the question of curvature in our proposed product space.

Let M be our Riemannian manifold and X (M ) be the set of vector fields on M .

The curvature R of M assigns a function R(X, Y ) : X (M ) →

X (M ) to each pair of vector fields (X, Y ) ∈ X × X .

For a vector field Z in X (M ), the function R(X, Y ) can be written DISPLAYFORM0 Here ∇ is the Riemannian connection for the manifold M , and [X, Y ]

is the Lie bracket of the vector fields X, Y .For convenience, we shall write the inner product R(X, Y )Z, T as (X, Y, Z, T ); this is the Riemannian curvature tensor.

Then, the sectional curvature is defined as follows.

Let us take V to be a two-dimensional subspace of T p M and x, y ∈ V be linearly independent (so that they span V).

Then, the sectional curvature at p for subspace V is DISPLAYFORM1 The model spaces S, H, E are the spaces of constant curvature, where K is constant for all points p and 2-subspaces V.For simplicity, suppose we are working with M = M 1 × M 2 ; the approach extends easily for larger products.

We write x = (x 1 , x 2 ) for x ∈ T p M .

Similarly, let R 1 , R 2 be the curvatures and K 1 , K 2 be the sectional curvatures of M 1 , M 2 at p, respectively.

Then the curvature tensor decomposes as DISPLAYFORM2 Our goal is to evaluate the sectional curvature K((x 1 , x 2 ), (y 1 , y 2 )) for the product manifold M .

We show the following, re-stated for convenience: DISPLAYFORM3 Proof.

Let us start with the numerator of equation FORMULA34 : DISPLAYFORM4 Here, we used equation 5 in the third line.

Note that when x 1 , y 1 are linearly independent, then R 1 (x 1 , y 1 ) DISPLAYFORM5 2 ) by (4).

Otherwise, this still holds since it is 0.

So we can relate the above to K 1 , K 2 : DISPLAYFORM6 For convenience, we write α i = x i 2 y i 2 −

x i , y i 2 for i = 1, 2.

Then the numerator is simply K 1 α 1 + K 2 α 2 .

Next, we consider the denominator of (equation 4): DISPLAYFORM7 where we set β = x 1 2 y 2 2 + x 2 2 y 1 2 .

Thus, we have that DISPLAYFORM8 Now, note that β > 0, since we assumed that x 1 , y 1 and x 2 , y 2 are linearly independent.

By CauchySchwarz, DISPLAYFORM9 Thus, we relate the product sectional curvature to a convex combination of the factor sectional curvatures K 1 , K 2 .

We have for non-negative K 1 , K 2 (e.g., Euclidean and spherical spaces) that DISPLAYFORM10 A similar result holds for the non-positive (Euclidean and hyperbolic) case.

The last case (one negative, one positive space) follows along the same lines.

Distribution of K The range of curvatures from Lemma 1 can be easily extended to a more refined distributional analysis.

In particular, consider sampling any point p and a random plane V ⊆ T p M .

By homogeneity, we can equivalently fix p.

The 2-subspaces of T p M R d forms the Grassmannian manifold Gr(2, T p M ).

The uniform measure on this (i.e. invariant to multiplication by an orthogonal matrix) can be recovered from the Haar measure on the orthogonal group O(d), which 1.

Embed the subgraph induced by B into P by any method; let the resulting worst-case distortion be 1 + δ.

Embed every node in T i into the embedded image of node i, 2.

Form the tree T by connecting each of the T 1 , . . .

, T |B| to a single node (equivalent to crushing all the nodes in B into a single node), and embed T into H r by using the combinatorial construction.

Additionally, all of the nodes in B are embedded into the image of the single central node in H r .We can check the distortion.

For nodes x a , y b in subtrees hanging off a, b ∈ B, the distance is d G (x a , y b ) = d T (x a , y b ) + d B (x, y).

Since the distortion for the two embeddings are given by 1 + δ and 1 + ε, it is easy to check that the overall distortion is at most max{1 + δ, 1 + ε}.As a concrete example, consider the ring of trees in FIG0 .

Then, B = C r , the cycle on r nodes.

In this case, we can embed B into P = S 1 .

Let the nodes of B be indexed a 1 , . . .

, a r .

We embed a i into A i = (cos( Thus indeed, the embedding has worst-case distortion 1.

Thus, the overall distortion for the ring of trees is 1 + ε.

Since we control ε, we can achieve arbitrarily good distortion for the ring of trees.

The complexity of this algorithm is linear in the number of nodes, since embedding the trees and ring is linear time.

General Graph Construction Now we use the min distance space to construct an embedding of any graph G on r nodes with arbitrarily low distortion via the space P = H 2 × H 2 × . . .

× H 2 with r − 1 copies.

As we shall see, this construction is ideal (arbitrarily low distortion, any graph) other than requiring O(r) spaces.

Let the nodes of G be V = {a 1 , . . .

, a r }.

Now, for each a i , 1 ≤ i ≤ r − 1, form the minimum distance tree T i rooted at a i .

Then, embed T i into the ith copy of H 2 via the combinatorial construction.

Then, for any nodes a i , a j ∈ V , the distance d G (a i , a j ) is attained by d Ti (a i , a j ), or d Tj (a j , a i ) in the case i = r. Since at least one of T i or T j , say T i , is embedded in H 2 with distortion 1 + ε, if we make ε small enough, the smallest distance among the embedded copies is indeed that for T i , so our overall distortion is still 1 + ε.

The combinatorial construction using the 1 distance (Section C.3) can embed the hanging tree graph arbitrarily well, unlike any single type of space.

Unlike a single space, this also lends more interpretability to the embedding, as each component displays different qualitative aspects of the underlying graph structure.

FIG5 shows that this phenomenon does in fact happen empirically, even using the optimization approach over the 2 (Riemannian) instead of 1 distance.

We provide some additional details for our experimental setups.

The optimization framework was implemented in PyTorch.

The loss function (2) was optimized with SGD using minibatches of 65536 edges for the real-world datasets, and ran for 2000 epochs.

For the Cities graph, the learning rate was chosen among {0.001, 0.003, 0.01}. For the rest of the datasets, the learning rate was chosen from a grid search among {10, 30, 100, 300, 1000} for each method.

6 Each point in the embedding is initialized randomly according to a uniform or Normal distribution in each coordinate with standard deviation 10 −3 .

(In the hyperboloid and spherical models, all but the first coordinate is chosen randomly, and the first coordinate is a function of the rest.)

TAB2 uses only Algorithm 1, and initializes the curvatures to −1 for hyperbolic components and 1 for spherical components.

These curvatures are learned using the method described in Section 3.1, and the "Best model" row reports the final curvatures of the best signature.

Word Embeddings Following LW, the input corpus is a 2013 dump of Wikipedia that has been preprocessed by lower casing and removing punctuation, and filtered to remove articles few page views.

All other hyperparameters are chosen exactly as in as LW, including their numbers for Euclidean embeddings from fastText.

The datasets used for similarity MEN) and analogy (Google) are also identical to the previous setup.

@highlight

Product manifold embedding spaces with heterogenous curvature yield improved representations compared to traditional embedding spaces for a variety of structures.

@highlight

Proposes a dimensionality reduction method that embeds data into a product manifold of spherical, Euclidean, and hyperbolic manifolds. The algorithm is based on matching the geodesic distances on the product manifold to graph distances.
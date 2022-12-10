In this work we propose a novel approach for learning graph representation of the data using gradients obtained via backpropagation.

Next we build a neural network architecture compatible with our optimization approach and motivated by graph filtering in the vertex domain.

We demonstrate that the learned graph has richer structure than often used nearest neighbors graphs constructed based on features similarity.

Our experiments demonstrate that we can improve prediction quality for several convolution on graphs architectures, while others appeared to be insensitive to the input graph.

Recently we have seen a rise in deep learning models, which can account for non-linearities and fit a wide range of functions.

Multilayer perceptron (MLP), a general purpose neural network, is a powerful predictor, but requires too many parameters to be estimated and often faces the problem of over-fitting, i.e. learns to almost exactly match training data and unable to generalize when it comes to testing.

While MLPs treat all features equally, which partially is the cause of excessive number of parameters, Convolutional Neural Networks (CNNs) have significantly fewer parameters and demonstrate groundbreaking results when it comes to object recognition in images BID11 .

The parameter reduction is due to utilizing convolutional operation: a window is sliding through the image and applying same linear transformation of the pixels.

The number of parameters then is proportional to the size of the window rather than polynomial of the number of data features as in the case of the MLPs.

Indeed images posses a specific structure, which can be encoded as a lattice graph, that makes the sliding window procedure meaningful, but inapplicable outside of the image domain.

In recent years there have been multiple works (cf.

Bronstein et al. (2017) for an overview) on generalizing convolution operation to a general domain, where graph is not a lattice.

Citing BID3 -"classification performance critically depends on the quality of the graph", nonetheless the problem of learning the graph useful for prediction has not been addressed so far and the graph was either known or pre-estimated only based on feature similarity in all of the prior work.

There are two major challenges when estimating the graph inside the neural network architecture.

First is the architecture itself -majority of the neural networks rely on gradient optimization methods, but the graph is often used in such ways that it is not possible to obtain its gradient.

In Section 3 we define a novel neural network architecture which is differentiable with respect to the graph adjacency matrix and built upon graph filtering in the vertex domain, extending the linear polynomial filters of BID20 .

Second problem is the series of constraints that are often imposed on the graph and therefore its adjacency.

In Section 2 we show how the three common graph properties, undirected sparse edges with positive weights, can be enforced by only utilizing the gradient obtained through backpropagation, therefore allowing us to utilize any of the modern deep learning libraries for graph estimation.

In Section 4 we discuss other graph based neural networks and evaluate them from the perspective of graph estimation.

In Section 5 we analyze graph estimation and interpretation for text categorization and time series forecasting.

We conclude with a discussion in Section 6 2 GRAPH OPTIMIZATION BASED ON BACKPROPAGATION In this section we provide an optimization procedure for learning adjacency matrix of a graph with various properties of interest, assuming that we can obtain its derivative via backpropagation.

In a subsequent section we will present novel neural network architecture that will allow us to get the derivative and utilize the graph in meaningful way.

Let data X ∈ R N ×D with N observation, D features and response Y ∈ R (or Y ∈ N for classification).

Graph G among data features can be encoded as its adjacency matrix A ∈ R D×D .

Our goal is to estimate functionŶ := f W (X, A), where W are weight parameters, that minimize some loss L := L(Ŷ , Y ).

We assume that we are able to evaluate partial derivative ∂L ∂A .

In the most general case, when edges of G can be directed, have negative weights and G can be fully connected, we perform the update A := A − γG ∂L ∂A , where G(·) depends on the optimizer (e.g., identity function for vanilla gradient descent) and γ is the step size.

Nonetheless, in the majority of the applications, G is desired to have some (or all) of the following properties:• Undirected graph, in which case A is restricted to be symmetric.• Have Positive edge weights, in which case A ∈ R D×D + .• Be Sparsely connected, in which case A should contain small proportion of non-zero entries.

First two properties are necessary for the existence of the graph Laplacian, crucial for the vast amount of neural networks on graphs architectures (e.g., BID2 ; BID8 ; BID3 ).

Third property greatly reduces computational complexity, helps to avoid overfitting and improves interpretability of the learned graph.

We proceed to present the Undirected Positive Sparse UPS optimizer, that can deliver each of the three properties and can be easily implemented as part of modern deep learning libraries.

Remark When node classification is of interest, our approach can be applied to graph between observations (e.g. social networks), then A ∈ R N ×N .

When G is desired to be undirected, its adjacency A is a symmetric matrix, hence A ij and A ji are the same parameters.

When backprop is used for gradient computation, this fact is not accounted for, but can be adjusted via the gradient correction DISPLAYFORM0 Correctness of this procedure can be easily verified.

Note that for modern stochastic optimization methods (e.g., Adam BID9 ) the corrected gradient U ∂L ∂A should be used for moment computations.

Restricting edge weights of the graph to be positive is necessary for the existence of the graph Laplacian and can help with interpretability of the resulting graph.

To achieve positive weights we need to add an inequality constraint of the form A ij ≥ 0 for i, j = 1, . . .

, D to our optimization task.

Constrained optimization has been widely studied and multiple techniques are available.

Given that we are building our optimization on top of the backprop, the most natural solution is the projected gradient descent.

This method has been previously shown to be effective even in the non-convex setups (e.g., Nonnegative Matrix Factorization BID12 ).

Projected gradient for positive weights constraint acts as follows: DISPLAYFORM0 Projection operator P is applied elementwise.

Another option is to consider adding a barrier function (i.e. elementwise logarithm of A) to the objective function, but we found projected gradient to be simpler and better aligning with out next step.

Sparsity is perhaps the most crucial property for several reasons.

In modern high dimensional problems, it is almost never the case that graph is fully connected, hence adjacency A should contain some zero entries.

Sparsity greatly reduces computational complexity of any neural network relying on the graph, especially when graph optimization is considered as in our work.

Finally, sparse graphs are much more interpretable.

Variable selection is an active research area in a variety of domains (cf.

BID4 ) and one of the dominant approaches is due to the L 1 penalty on the object that is desired to be sparse, in our case penalty is g λ (A) = λ i,j |A ij |, which is combined with the loss L(X, A, W ) to form a new objective function.

It is known that L 1 norm is not differentiable at 0, although, similar to gradient, subgradient descent optimization can be used.

They key disadvantage of such approach is that it does not actually obtain sparse solution.

Instead we propose to use proximal gradient method (cf.

Section 4 of Parikh et al. FORMULA0 ), which again aligns well with backprop based optimization.

Proximal operator of the L 1 penalty g λ (A) is the soft thresholding operation: DISPLAYFORM0 (3) Then our final, sparsifying, step is: DISPLAYFORM1 Notice that we threshold by λ scaled by the step size γ and that soft thresholding operation can be simplified for positive weights S γλ (x) = (x − γλ) + .Remark Another graph property that is sometimes of interest is the presence of self connections.

If one wants to prohibit self connections, it can trivially be done by setting the diagonal of A to 0 and not performing any updates on the diagonal.

We do not enforce this since UPS can estimate what self connections, if any, should be present in the graph via the proximal 4 step.

The key assumption of the UPS optimizer is that we can evaluate the partial gradient of the adjacency matrix.

We propose a novel neural network architecture arising from the Graph Signal Processing (GSP) literature that satisfies the assumption.

A prominent way to improve the quality of features of X ∈ R D (we consider a single observation in this section to simplify the notations) is to process it as a signal on graph among data features with adjacency matrix A ∈ R D×D .

GSP (see BID22 for an overview) then allows us to do filtering in the vertex domain as follows: DISPLAYFORM0 DISPLAYFORM1 . .

, D are filtering coefficients.

Equation 5 modifies signal at the ith feature by taking into account signals at features reachable in exactly k steps.

By varying k = 1, . . .

, K we can extract new features that account for the graph structure in the data and combine them into filtered graph signal f (X, A) ∈ R D used for prediction.

BID20 proposed linear polynomial graph filter of the form: DISPLAYFORM2 Then the filtered signal is obtained via matrix multiplication f (X, A) = XH(A) and is a special case of filtering in the vertex domain: DISPLAYFORM3 Filtering in Eq. 7 can be used for graph optimization with UPS, but it possesses several limitations.

Filtered signals f (k) (X i ) are combined in linear way and choice of filtering coefficients b (k) i,j = (A k ) ij might lack flexibility.

Neural networks are known to be much more effective than linear models and hence we address the two shortcomings of graph filter 7 with the following Graph Polynomial Signal (GPS) neural network: DISPLAYFORM0 GPS directly utilizes filtered signals based on vertex neighborhoods of varying degree k and allows for non-linear feature mappings.

GPS example is given in FIG0 .

Last step is to build a mapping from the GPS features into the response Y .

This can be done via linear fully connected layer or a more complex neural network can be built on top of the GPS features.

Our architecture can be easily implemented using modern deep learning libraries and backprop can be used to obtain the partial derivative of the adjacency A, required by the UPS optimizer.

Role of weights w (k) j , j = 1, . . .

, D, k = 1, . . . , K is two fold -firstly, they scale the graph adjacency, which is crucial for proximal optimization 4.

For inducing sparsity in the adjacency A ideally we would penalize number of nonzero elements (L 0 norm) in A, but such penalty is known to be NP-hard for optimization, hence the L 1 is always used instead.

The drawback of this choice is that we penalize nonzero edge weights A ij by their magnitude |A ij | which might be detrimental for the prediction.

To avoid disagreement between L 1 penalty term and prediction quality, re-scaling with weights is helpful.

Second role of weights has the nature of weight sharing of classical CNN on images.

For image data, objects are often considered to be location invariant, hence CNN shares same set of weights across the whole image.

For a general data type considered in our work, there is no reason to make location invariance assumption.

Instead we assume that weights should be shared among neighboring graph regions.

In particular, observe that UPS can decide to partition the graph into multiple connected components, then GPS will enforce weight sharing inside each component by construction.

GPS was designed to perform graph filtering based on its adjacency matrix in a way, that UPS optimizer can be used to learn the adjacency.

There are few other architectures that require a graph be given, but can be combined with the UPS for graph learning.

BID21 proposed Graph Neural Network (GNN) -a rather general framework for utilizing graph neighborhoods information.

Some of the recent works on neural networks on graphs can be viewed as a special case of it BID1 .

GNN does not utilize adjacency directly in the architecture, but its modern variation Graph Convolutional Network (GCN) BID10 does so for the case of graph among observations.

Their architecture can be modified for graph among features via stacking layers of the form: DISPLAYFORM0 where DISPLAYFORM1 ∈ R for j = 1, . . .

, D, k = 0, . . . , K − 1 are the trainable parameters.

Notice that since data in the applications we consider does not have multiple input channels, we modified the architecture to have different weights across the graph for every layer.

BID10 show that this architecture does 1-hop filtering inside each layer.

When multiple layers are stacked, resulting expression gets very complex due to non-linearities and can not be considered as filtering of higher degree in the sense of Graph Signal Processing as in Eq. 5.

It is possible to use UPS with GCN for graph learning if we use A instead ofÃ in Eq. 9, but the connection to graph filtering would be lost.

Deep learning on graphs has recently become an active area of research, but all of the prior work assumes the graph be given or estimated prior to model fitting, using, for example, kNN graph based on a feature similarity metric of choice.

BID8 formulated a supervised graph estimation procedure, but this again is done prior to their model fitting by training an MLP and utilizing first layer features for kNN graph construction.

Another popular direction, motivated by the success of word embedding BID13 in the NLP domain, is learning latent feature representations of the graph BID18 BID5 BID19 .

Here graph is also required as an input.

When it comes to architecture design involving graphs two approaches are usually distinguishedspatial and spectral (cf.

Bronstein et al. (2017) for an overview).

GPS is a spatial approach utilizing graph adjacency as building block in a suitable for graph optimization manner.

We have already discussed two existing spatial approaches that can be combined with graph optimization.

Next we discuss some other spatial and spectral approaches from the perspective of graph learning.

BID14 proposed creating receptive fields using graph labeling and then applying a 1D convolution.

BID7 suggested building CNN type architecture by considering neighborhoods of fixed degree using powers of the transition matrix.

Graph node sequencing and neighborhood search are not differentiable and hence not compatible with the UPS optimization.

Diffusion-Convolutional Neural Networks (DCNN) BID0 use power series of the transition matrix and combine it with a set of trainable weights to do graph, node and edge classifications.

Transition matrix requires positive weights and is restricted to be stochastic, which would complicate the optimization.

DCNN pre-computes powers of the transition matrix and stores them as a tensor, which is not suitable for graph learning.

Idea behind spectral architectures is to utilize eigenvector basis of the graph Laplacian to do filtering in the Fourier domain: DISPLAYFORM0 where u l , λ l l = 0, . . .

, D − 1 are the eigenvectors and eigenvalues of the graph Laplacian L of A. Key choice one has to make when using spectral approach is the functional form of filter h(λ l ).

BID2 ; BID8 proposed to use nonparametric filtersĥ(λ l ) = w l , where w l , l = 0, . . .

, D − 1 are trainable parameters.

When graph learning comes into play, such approach becomes inefficient since we would need to optimize for the eigenvectors of the graph Laplacian, which are not sparse even for sparse graphs and have to be constrained to be orthonormal.

Additionally, they proposed to use hierarchical graph clustering for pooling, which one would have to redo on every iteration of the graph optimization as the graph changes.

BID3 utilized another filtering functionĥ( DISPLAYFORM1 , which is appealing as it can be shown BID22 to perform filtering in the vertex domain 5 with filtering coefficients b DISPLAYFORM2 They also utilized a Chebyshev polynomial approximation to the graph filter to bypass the necessity for computing the eigen decomposition of the Laplacian BID6 .

In the architecture design, they used graph coarsening and pooling based on node rearrangement, which, as discussed before, are non-differentiable operations.

As in the case of transition matrix, optimizing for graph Laplacian would complicate the optimization, especially in the presence of Chebyshev polynomials.

In the experimental section our goal is to show that graph learned using UPS graph optimizer based on the GPS architecture can give additional insights about the data and can be utilized by other graph based neural networks.

We fit the GPS architecture using UPS optimizer for varying degree of the neighborhood of the graph.

Resulting graphs are then used to train ChebNet BID3 , ConvNet BID8 , GCN (Kipf & Welling, 2016) as in Eq. 9 and Graph CNN BID7 .

We also consider standard graph initialization using kNN graph, random graph and kNN graph based on the MLP features as in BID8 for all the above architectures and for the GPS without graph optimization.

We used Adam BID9 for weight optimization and as a stochastic optimizer G for the UPS in Eq. 1, 2, 4.

In this experiment we provide thorough evaluation of various graph convolutional architectures using 20news groups data set.

We keep the 1000 most common words, remove documents with less than 5 words and use bag-of-words as features.

Training size is 9924 and testing is 6695.

Results are presented in TAB0 .

We see that GPS with optimized graph can achieve good results, but fails when the graph is random or pre-estimated.

Interestingly, ChebNet and ConvNet do not appear to be sensitive to the graph being used.

This might be due to high number of trainable parameters in the respective architectures.

GCN did poorly overall, but showed a relative improvement when the estimated graphs were used.

Next we compare the learned graph versus a kNN graph often used for initialization of various graph neural networks.

We estimated the graph using UPS and GPS 4 architecture and used nested stochastic block model for visualization.

Note that nested stochastic block model selects number of levels and blocks on each level of the hierarchy to minimize the description length of the graph (cf.

Peixoto (2014a) and references therein).

GPS 4 utilizes graph neighborhoods up to 4th degree and we see in FIG1 that there are 4 levels in between the input dimensions and the endpoint.

Additionally note that intermediate levels have 18 and 5 blocks, which is roughly similar to 20 classes and 6 super classes (more general groupings of the news categories) of the 20news groups data, which is possibly due to the supervised nature of the graph.

For comparison, we also provide a hierarchical structure of the 100NN graph in FIG1 , which is very poorly structured and only has two intermediate levels.

We use a dataset consisting of time series of visits to a popular website across 100 cities in the US.

Visits counts were normalized by standard deviation.

The task is to predict the average number of visits across cities for tomorrow.

3 years of daily data is used for training and testing is done on consecutive 200 days.

Results are reported in TAB1 .

GPS demonstrated very good performance and we can see noticeable improvement of the GCN 3 result when graph was learned for the neighborhood degree of 3 and higher.

For ChebNet we report the best score instead of final one as it was overfitting severely.

Nonetheless the best score appears to improve when the trained graph is used.

In this work, motivated by the rising attention to convolution on graphs neural networks, we developed a procedure and a novel architecture for graph estimation that can account for neighborhoods of varying degree in a graph.

We showed that resulting graph has more structure then a commonly used kNN graph and demonstrated good performance of our architecture for text categorization and time series forecasting.

The worrisome observation is the insensitivity of some of the modern deep convolution networks on graphs to the graph being used.

Out of the considered architectures, only GPS and GCN showed noticeable performance improvement when a better graph was used.

These architectures stand out as they have much fewer trainable parameters and are more likely to suffer from a badly chosen graph.

We think that a deep network utilizing the graph should not be able to produce any sensible result when the random graph is used.

When doing convolution on images, pooling is one of the important steps that helps to reduce the resolution of filters.

It is unclear so far how to incorporate pooling into the GPS, while maintaining the ability to extract the gradient.

This is one of the limitations of our approach and is of interest for further investigation.

<|TLDR|>

@highlight

Graph Optimization with signal filtering in the vertex domain.

@highlight

The paper investigates learning adjacency matrix of a graph with sparsely connected undirected graph with nonnegative edge weights uses a projected sub-gradient descent algorithm.

@highlight

Develops a novel scheme for backpropogating on the adjacency matrix of a neural network graph
Discretizing floating-point vectors is a fundamental step of modern indexing methods.

State-of-the-art techniques learn parameters of the quantizers on training data for optimal performance, thus adapting quantizers to the data.

In this work, we propose to reverse this paradigm and adapt the data to the quantizer: we train a neural net whose last layers form a fixed parameter-free quantizer, such as pre-defined points of a sphere.

As a proxy objective, we design and train a neural network that favors uniformity in the spherical latent space, while preserving the neighborhood structure after the mapping.

For this purpose, we propose a new regularizer derived from the Kozachenko-Leonenko differential entropy estimator and combine it with a locality-aware triplet loss.

Experiments show that our end-to-end approach outperforms most learned quantization methods, and is competitive with the state of the art on widely adopted benchmarks.

Further more, we show that training without the quantization step results in almost no difference in accuracy, but yields a generic catalyser that can be applied with any subsequent quantization technique.

Recent work BID27 proposed to leverage the pattern-matching ability of machine learning algorithms to improve traditional index structures such as B-trees or Bloom filters, with encouraging results.

In their one-dimensional case, an optimal B-Tree can be constructed if the cumulative density function (CDF) of the indexed value is known, and thus they approximate this CDF using a neural network.

We emphasize that the CDF itself is a mapping between the indexed value and a uniform distribution in [0, 1] .

In this work, we wish to generalize such an approach to multi-dimensional spaces.

More precisely, as illustrated by FIG0 , we aim at learning a function that maps real-valued vectors to a uniform distribution over a d-dimensional sphere, such that a fixed discretizing structure, for example a fixed binary encoding (sign of components) or a regular lattice quantizer, offers competitive coding performance.

Our approach is evaluated in the context of similarity search, where methods often rely on various forms of learning machinery BID12 BID45 ; in particular there is a substantial body of literature on methods producing compact codes BID20 ).

Yet the problem of jointly optimizing a coding stage and a neural network remains essentially unsolved, partly because .

It is learned end-to-end, yet the part of the network in charge of the discretization operation is fixed in advance, thereby avoiding optimization problems.

The learnable function f , namely the "catalyzer", is optimized to increase the quality of the subsequent coding stage.

input ?? = 0 ?? = 0.01 ?? = 0.1 ?? ??? ??? FIG1 : Illustration of our method, which takes as input a set of samples from an unknown distribution.

We learn a neural network that aims at preserving the neighborhood structure in the input space while best covering the output space (uniformly).

This trade-off is controlled by a parameter ??.

The case ?? = 0 keeps the locality of the neighbors but does not cover the output space.

On the opposite, when the loss degenerates to the differential entropic regularizer (?? ??? ???), the neighbors are not maintained by the mapping.

Intermediate values offer different trade-offs between neighbor fidelity and uniformity, which is proper input for an efficient lattice quantizer (depicted here by the hexagonal lattice A 2 ).it is difficult to optimize through a discretization function.

For this reason, most efforts have been devoted to networks producing binary codes, for which optimization tricks exist, such as soft binarization or stochastic relaxation, which are used in conjunction with neural networks BID28 BID18 .

However it is difficult to improve over more powerful codes such as those produced by product quantization BID20 , and recent solutions addressing product quantization require complex optimization procedures BID24 BID34 .In order to circumvent this problem, we propose a drastic simplification of learning algorithms for indexing.

We learn a mapping such that the output follows the distribution under which the subsequent discretization method, either binary or a more general quantizer, performs better.

In other terms, instead of trying to adapt an indexing structure to the data, we adapt the data to the index.

Our technique requires to jointly optimize two antithetical criteria.

First, we need to ensure that neighbors are preserved by the mapping, using a vanilla ranking loss BID40 BID6 BID44 .

Second, the training must favor a uniform output.

This suggests a regularization similar to maximum entropy BID36 , except that in our case we consider a continuous output space.

We therefore propose to cast an existing differential entropy estimator into a regularization term, which plays the same "distribution-matching" role as the Kullback-Leiber term of variational auto-encoders BID9 .As a side note, many similarity search methods are implicitly designed for the range search problem (or near neighbor, as opposed to nearest neighbor BID15 BID0 ), that aims at finding all vectors whose distance to the query vector is below a fixed threshold.

For real-world high-dimensional data, range search usually returns either no neighbors or too many.

The discrepancy between near-and nearest-neighbors is significantly reduced by our technique, see Section 3.3 and Appendix C for details.

Our method is illustrated by FIG1 .

We summarize our contributions as follows:??? We introduce an approach for multi-dimensional indexing that maps the input data to an output space in which indexing is easier.

It learns a neural network that plays the role of an adapter for subsequent similarity search methods.??? For this purpose we introduce a loss derived from the Kozachenko-Leonenko differential entropy estimator to favor uniformity in the spherical output space.??? Our learned mapping makes it possible to leverage spherical lattice quantizers with competitive quantization properties and efficient algebraic encoding.??? Our ablation study shows that our network can be trained without the quantization layer and used as a plug-in for processing features before using standard quantizers.

We show quantitatively that our catalyzer improves performance by a significant margin for quantization-based (OPQ BID10 ) and binary (LSH BID5 ) method.

This paper is organized as follows.

Section 2 discusses related works.

Section 3 introduces our neural network model and the optimization scheme.

Section 4 details how we combine this strategy with lattice assignment to produce compact codes.

The experimental section 5 evaluates our approach.

Generative modeling.

Recent models such as Generative Adversarial Networks (GANs) BID13 or Variational Auto-Encoders (VAEs) BID23 ) learn a mapping between an isotropic Gaussian distribution and the empirical distribution of a training set.

Our approach maps an empirical input distribution to a uniform distribution on the spherical output space.

Another distinction is that GANs learn a unidirectional mapping from the latent code to an image (decoder), whereas VAEs learn a bidirectional mapping (encoder -decoder).

In our work, we focus on learning the encoder, whose goal is to pre-process input vectors for subsequent indexing.

Dimensionality reduction and representation learning.

There is a large body of literature on the topic of dimensionality reduction, see for instance the review by BID43 .

Relevant work includes self-organizing maps BID26 , the stochastic neighbor embedding BID14 and the subsequent t-SNE approach BID42 , which is tailored to low-dimensional spaces for visualisation purposes.

Both works are non-linear dimensionality reduction aiming at preserving the neighborhood in the output space.

Learning to index and quantize.

The literature on product compact codes for indexing is most relevant to our work, see BID45 BID9 for an overview of the topic.

Early popular highdimensional approximate neighbor methods, such as Locality Sensitive Hashing BID15 BID11 BID5 BID0 , were mostly relying on statistical guarantees without any learning stage.

This lack of data adaptation was subsequently addressed by several works.

The Iterative quantization (ITQ) BID12 modifies the coordinate system to improve binarization, while methods inspired by Vector Quantization and compression BID20 BID1 BID47 BID17 have gradually emerged as strong competitors for estimating distances or similarities with compact codes.

While most of these works aim at reproducing target (dis-)similarity, some recent works directly leverage semantic information in a supervised manner with neural networks BID28 BID18 BID24 BID38 .Lattices, also known as Euclidean networks, are discrete subsets of the Euclidean space that are of particular interest due to their space covering and sphere packing properties BID7 .

They also have excellent discretization properties under some assumptions about the distribution, and most interestingly the closest point of a lattice is determined efficiently thanks to algebraic properties BID37 .

This is why lattices have been proposed BID0 BID19 as hash functions in LSH.

However, for real-world data, lattices waste capacity because they assume that all regions of the space have the same density BID35 .

In this paper, we are interested in spherical lattices because of their bounded support.

Entropy regularization appears in many areas of machine learning and indexing.

For instance, BID36 argue that penalizing confident output distributions is an effective regularization.

BID8 use entropy regularization to speed up computation of optimal transport distances.

Another proposal by BID4 in an unsupervised learning context, is to spread the output by enforcing input images to map to points drawn uniformly on a sphere.

Interestingly, most recent works on binary hashing introduce some form of entropic regularization.

Deep hashing BID28 employs a regularization term that increases the marginal entropy of each bit.

SUBIC BID18 extends this idea to one-hot codes.

Our proposal is inspired by prior work for one-dimensional indexing BID27 .

However their approach based on unidimensional density estimation can not be directly translated to the multidimensional case.

Our strategy is to train a neural network f that maps vectors from a d in -dimensional space to the hypersphere of a d out -dimensional space S dout .

Let us first introduce our regularizer, which we design to spread out points uniformly across S dout .

With the knowledge of the density of points p, we could directly maximize the differential entropy Figure 3 : Histograms of the distance between a query point and its 1st (resp.

100 th ) nearest neighbors, in the original space (left) and after our catalyzer (right).

In the original space, the two histograms have a significant overlap, which means that a 100-th nearest neighbor for a query has often a distance lower that the 1st neighbor for another query.

This gap is significantly reduced by our catalyzer.

differential entropy as a proxy.

It was shown by Kozachenko and Leononenko (see e.g. BID3 ) that defining ?? n,i = min j =i f (x i ) ??? f (x j ) , the differential entropy of the distribution can be estimated by DISPLAYFORM0 DISPLAYFORM1 where ?? n and ?? n are two constants that depend on the number of samples n and the dimensionality of the data d out .

Ignoring the affine components, we define our entropic regularizer as DISPLAYFORM2 This loss also has a satisfactory geometric interpretation: closest points are pushed away, with a strength that is non-decreasing and concave.

This ensures diminishing returns: as points get away from each other, the marginal impact of increasing the distance becomes smaller.

We enforce the outputs of the neural network to follow the same neighborhood structure as in the input space by adopting the triplet loss BID6 BID44 ) DISPLAYFORM0 where x is a query, x + a positive match, x ??? a negative match.

The positive matches are obtained by computing the k pos nearest neighbors of each point x in the training set in the input space.

The negative matches are generated by taking the k neg -th nearest neighbor of f (x) in (f (x 1 ), ..., f (x n )).

In order to speed up the learning, we compute the k neg -th nearest neighbor of every point in the dataset at the beginning of each epoch and use these throughout the epoch.

Note that we do not need to use a margin, as its effect is essentially superseded by our regularizer.

Our overall loss combines the triplet loss and the entropy regularizer, as DISPLAYFORM1 where the parameter ?? ??? 0 controls the trade-off between ranking quality and uniformity.

Choice of ??.

The marginal distributions for these two views are much more uniform with our KoLeo regularizer, which is a consequence of the higher uniformity in the high-dimensional latent space.

Qualitative evaluation of the uniformity.

Figure 3 shows the histogram of the distance to the nearest (resp.

100 th nearest) neighbor, before applying the catalyzer (left) and after (right).

The overlap between the two distributions is significantly reduced by the catalyzer.

We evaluate this quantitatively by measuring the probability that the distance between a point and its nearest neighbor is larger than the distance between another point and its 100 th nearest neighbor.

In a very imbalanced space, this value is 50%, whereas in a uniform space it should approach 0%.

In the input space, this probability is 20.8%, and it goes down to 5.0% in the output space thanks to our catalyzer.

Visualization of the output distribution.

While FIG1 illustrates our method with the 2D disk as an output space, we are interested in mapping input samples to a higher dimensional hyper-sphere.

FIG2 proposes a visualization of the high-dimensional density from a different viewpoint, with the Deep1M dataset mapped in 8 dimensions.

We sample 2 planes randomly in R dout and project the dataset points (f (x 1 ), ..., f (x n )) on them.

For each column, the 2 figures are the angular histograms of the points with a polar parametrization of this plane.

The area inside the curve is constant and proportional to the number of samples n. A uniform angular distribution produces a centered disk, and less uniform distributions look like unbalanced potatoes.

The densities we represent are marginalized, so if the distribution looks non-uniform then it is non-uniform in d out -dimensional space, but the reverse is not true.

Yet one can compare the results obtained for different regularization coefficients, which shows that our regularizer has a strong uniformizing effect on the mapping, ultimately resembling that of a uniform distribution for ?? = 1.

In this section we describe how our method interplays with discretization, at training and at search time.

We consider two parameter-free coding methods: binarization and defining a fixed set of points on the unit sphere provided by a lattice spherical quantizer.

A key advantage of a fixed coding structure like ours is that compressed-domain distance computations between codes do not depend on external meta-data.

This is in contrast with quantization-based methods like product quantization, which require centroids to be available at search time.

Binary features are obtained by applying the sign function to the coordinates.

We relax this constraint at train time by replacing the sign with the identity function, and the binarization is used only to cross-validate the regularization parameter on the validation set.

As discussed by BID35 , lattices impose a rigid partitioning of the feature space, which is suboptimal for arbitrary distributions, see FIG1 .

In contrast, lattices offer excellent quantization properties for a uniform distribution BID7 .

Thanks to our regularizer, we are closer to uniformity in the output space, making lattices an attractive choice.

We consider the simplest spherical lattice, integer points of norm r, a set we denote S r d .

Given a vector x ??? R din , we compute its catalyzed features f (x), and find the nearest lattice point on S r d using the assignment operation, which formally minimizes q(f (x)) = min c???S r DISPLAYFORM0 This assignment can be computed very efficiently (see Appendix B for details).

Given a query y and its representation f (y), we approximate the similarity between y and x using the code: DISPLAYFORM1 , This is an asymmetric comparison, because the query vectors are not quantized BID20 .When used as a layer, it takes a vector in R d and returns the quantized version of this vector in the forward pass, and passes the gradient to the previous layer in the backward pass.

This heuristic is referred to as the straight-through estimator in the literature, and is often used for discretization steps, see e.g., van den Oord et al. (2017) .

This section presents our experimental results.

We focus on the class of similarity search methods that represents the database vectors with a compressed representation BID5 BID20 BID12 BID10 , which enables to store very large dataset in memory BID30 BID39 .

All experiments have two phases.

In the first phase (encoding), all vectors of a database are encoded into a representation (e.g. 32, 64 bits).

Encoding consists in a vector transformation followed by a quantization or binarization stage.

The second phase is the search phase: a set of query vectors is transformed, then the codes are scanned exhaustively and compared with the transformed query vector, and the top-k nearest vectors are returned.

Datasets and metrics.

We use two benchmark datasets Deep1M and BigAnn1M.

Deep1M consists of the first million vectors of the Deep1B dataset BID2 .

The vectors were obtained by running a convnet on an image collection, reduced to 96 dimensions by principal component analysis and subsequently 2 -normalized.

We also experiment with the BigAnn1M BID21 , which consists of SIFT descriptors BID29 .

Both datasets contain 1M vectors that serve as a reference set, 10k query vectors and a very large training set of which we use 500k elements for training, and 1M vectors that we use a base to cross-validate the hyperparameters d out and ??.

We also experiment on the full Deep1B and BigAnn datasets, that contain 1 billion elements.

We evaluate methods with the recall at k performance measure, which is the proportion of results that contain the ground truth nearest neighbor when returning the top k candidates (for k ??? {1, 10, 100}).Training.

For all methods, we train our neural network on the training set, cross-validate d out and ?? on the validation set, and use a different set of vectors for evaluation.

In contrast, some works carry out training on the database vectors themselves BID33 BID31 BID12 , in which case the index is tailored to a particular fixed set of database vectors.

Our model is a 3 -layer perceptron, with ReLU non-linearity and hidden dimension 1024.

The final linear layer projects the dataset to the desired output dimension d out , along with 2 -normalization.

We use batch normalization BID16 ) and train our model for 300 epochs with Stochastic Gradient Descent, with an initial learning rate of 0.1 and a momentum of 0.9.

The learning rate is decayed to 0.05 (resp.

0.01) at the 80-th epoch (resp.

120-th).

We evaluate the lattice-based indexing proposed in Section 4, and compare it to more conventional methods based on quantization, namely PQ BID20 and Optimized Product Quantization (OPQ) BID10 .

We use the Faiss BID22 implementation of PQ and OPQ and assign one byte per sub-vector (each individual quantizer has 256 centroids).

For our lattice, we vary the value of r to increase the quantizer size, hence generating curves for each value of d out .

Figure 5 provides a comparison of these methods.

On both datasets, the lattice quantizer strongly outperforms PQ and OPQ for most code sizes.

Figure 5 : Comparison of the performance of the product lattice vs OPQ on Deep1M (left) and BigAnn1M (right).

Our method maps the input vectors to a d out -dimensional space, that is then quantized with a lattice of radius r. We obtain the curves by varying the radius r.

Impact of the hyperparameters.

Varying the rank parameters k pos and k neg did not impact significantly the performance, so we fixed them respectively to k pos = 10 and k neg = 50.

For a fixed number of bits, varying the dimension d out is a trade-off between a good representation and an easily compressible one.

When d out is small, we can use a large r for a very small quantization error, but there are not enough dimensions to represent the degrees of freedom of the underlying data.

A larger d out allows for better representations but suffers from a coarser approximation.

Figure 5 shows that for low bitrates, small dimensions perform better because the approximation quality dominates, whereas for higher bitrates, larger dimensions are better because the representation quality dominates.

Similarly, the regularizer ?? needs to be set to a large value for small dimensions and low bitrates, but higher dimensions and higher bitrates require lower values of ?? (cf.

Appendix A for details).Large-scale experiments.

We experiment with the full Deep1B (resp.

BigAnn) dataset, that contains 1 billion vectors, with 64 bits codes.

At that scale, the recall at 10 drops to 26.1% for OPQ and to 37.8% for the lattice quantizer (resp.

21.3% and 36.5%).

As expected, the recall performance is lower than for the 1 million vectors database, but the precision advantage of the lattice quantizer is maintained at large scale.

Comparison to the state of the art.

Additive quantization variants BID1 BID32 BID34 are currently state-of-the art encodings for vectors in terms of accuracy.

However, their encoding stage involves an iterative optimization process that is prohibitively slow for practical use cases.

For example, Competitive quantization's reported complexity is 15?? Table 2 : Performance (1-recall at 10, %) with LSH, on Deep1M and BigAnn1M, as a function of the number of bits per index vector.

All results are averaged over 5 runs with different random seeds.

Our catalyzer gets a large improvement in binary codes over LSH and ITQ.slower than OPQ.

Table 1 compares our results with LSQ BID32 , a recent variant that is close to the state of the art and for which open-source code is available.

We show that our Catalyst + Lattice variant method is 14?? times faster for an accuracy that is competitive or well above that of LSQ.

To our knowledge, this is the first time that such competitive results are reported for a method that can be used in practice at a large scale.

Our search time is a bit slower: computing 1M asymmetric distances takes 7.5 ms with the Catalyzer+Lattice instead of 4.9 ms with PQ.

This is due to our decoding procedure, which does not rely on precomputed tables as used in PQ.

Ablation study.

As a sanity check, we first replace our catalyzer by a PCA that reduces the dimensionality to the same size as our catalyzer, followed by 2 -normalization.

This significantly decreases the performance of the lattice quantizer, as can be seen in Table 1 .We also evaluate the impact of training end-to-end, compared to training without the quantization layer.

Table 1 shows that end-to-end training has a limited impact on the overall performance for 64 bits, sometimes even decreasing performance.

This may be partly due to the approximation induced by the straight-through estimation, which handicaps end-to-end training.

Another reason is that the KoLeo regularizer narrows the performance gap induced by discretization.

In other terms, our method trained without the discretization layer trains a general-purpose network (hence the name catalyzer), on which we can apply any binarization or quantization method.

Table 1 shows that OPQ is improved when applied on top of catalyzed features, for example increasing the recall@10 from 63.6 to 71.1.Binary hashing.

We also show the interest of our method as a catalyzer for binary hashing, compared to two popular methods BID5 BID12 :LSH maps Euclidean vectors to binary codes that are then compared with Hamming distance.

A set of m fixed projection directions are drawn randomly and isotropically in d in , and each vector is encoded into m bits by taking the sign of the dot product with each direction.

ITQ is another popular hashing method, that improves LSH by using an orthogonal projection that is optimized to maximize correlation between the original vectors and the bits.

Table 2 compares our catalyzer to LSH and ITQ.

Note that a simple sign function is applied to the catalyzed features.

The catalyzer improves the performance by 2-9 percentage points in all settings, from 32 to 128 bits.

We train a neural network that maps input features to a uniform output distribution on a unit hypersphere, making high-dimensional indexing more accurate, in particular with fast and rigid lattice quantizers or a trivial binary encoding.

To the best of our knowledge, this is the first work on multi-dimensional data that demonstrates that it is competitive to adapt the data distribution to a rigid quantizer, instead of adapting the quantizer to the input data.

This has several benefits: rigid quantizers are fast at encoding time; and vectors can be decoded without carrying around codebooks or auxiliary tables.

We open-sourced the code corresponding to the experiments at https://github.com/facebookresearch/spreadingvectors.

The optimal value of the regularizer ?? decreases with the dimension, as shown by TAB2 : Optimal values of the regularization parameter ?? for Deep1M, using a fixed radius of r = 10.

We consider the set of integer points DISPLAYFORM0 Atoms.

We define a "normalization" function N of vectors: it consists in taking the absolute value of their coordinates, and sorting them by decreasing coordinates.

We call "atoms" the set of vectors that can be obtained by normalizing the vectors of S Encoding and enumerating.

To solve Equation 5, we apply the following steps:1.

normalize y with N , store the permutation ?? that sorts coordinates of |y| 2.

exhaustively search the atom z that maximizes N (y) z 3.

apply the inverse permutation ?? ???1 that sorts y to z to obtain z 4.

the nearest vector (z 1 , .., z d ) is z i = sign(y i )z i ???i = 1..d.

To encode a vector of z ??? S r d we proceed from N (z):1.

each atom is assigned a range of codes, so z is encoded relative to the start of N (z)'s range 2. encode the permutation using combinatorial number systems BID25 .

There are d! permutations, but the permutation of equal components is irrelevant, which divides the number combinations.

For example atom (2, 2, 1, 1, 0, 0, 0, 0) is the normalized form of 8!/(2!2!4!) = 240 vectors of S ??? 10 8 .3.

encode the sign of non-zero elements.

In the example above, there are 4 sign bits.

Decoding proceeds in the reverse order.

Encoding 1M vectors takes about 0.5 s on our reference machine, which is faster than PQ (1.9 s).

In other terms, he quantization time is negligible w.r.t.

the preprocessing by the catalyzer.

FIG7 shows how our method achieves a better agreement between range search and k-nearest neighbors search on Deep1M.

In this experiment, we consider different thresholds ?? for the range search and perform a set of queries for each ??.

Then we measure how many vectors we must return, on average, to achieve a certain recall in terms of the nearest neighbors in the original space.

Without our mapping, there is a large variance on the number of results for a given ??.

In contrast, after the mapping it is possible to use a unique threshold to find most neighbors.

For example: to obtain 80% recall, the search in the original space requires to set ?? = 0.54, which returns 700 results per query on average, while in the transformed space ?? = 0.38 returns just 200 results.

Observe the much better agreement in the latent spherical space.

@highlight

We learn a neural network that uniformizes the input distribution, which leads to competitive indexing performance in high-dimensional space
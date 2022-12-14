Deep representation learning has become one of the most widely adopted approaches for visual search, recommendation, and identification.

Retrieval of such  representations from a large database is however computationally challenging.

Approximate methods based on learning compact representations, have been widely explored for this problem, such as locality sensitive hashing, product quantization, and PCA.

In this work, in contrast to learning compact representations, we propose to learn high dimensional and sparse representations that have similar representational capacity as dense embeddings while being more efficient due to sparse matrix multiplication operations which can be much faster than dense multiplication.

Following the key insight that the number of operations decreases quadratically with the sparsity of embeddings provided the non-zero entries are distributed uniformly across dimensions, we propose a novel approach to learn such distributed sparse embeddings via the use of a carefully constructed regularization function that directly minimizes a continuous relaxation of the number of floating-point operations (FLOPs) incurred during retrieval.

Our experiments show that our approach is competitive to the other baselines and yields a similar or better speed-vs-accuracy tradeoff on practical datasets.

Learning semantic representations using deep neural networks (DNN) is now a fundamental facet of applications ranging from visual search (Jing et al., 2015; Hadi Kiapour et al., 2015) , semantic text matching (Neculoiu et al., 2016) , oneshot classification (Koch et al., 2015) , clustering (

Oh Song et al., 2017) , and recommendation (Shankar et al., 2017) .

The high-dimensional dense embeddings generated from DNNs however pose a computational challenge for performing nearest neighbor search in large-scale problems with millions of instances.

In particular, when the embedding dimension is high, evaluating the distance of any query to all the instances in a large database is expensive, so that efficient search without sacrificing accuracy is difficult.

Representations generated using DNNs typically have a higher dimension compared to hand-crafted features such as SIFT (Lowe, 2004) , and moreover are dense.

The key caveat with dense features is that unlike bag-of-words features they cannot be efficiently searched through an inverted index, without approximations.

Since accurate search in high dimensions is prohibitively expensive in practice (Wang, 2011) , one has to typically sacrifice accuracy for efficiency by resorting to approximate methods.

Addressing the problem of efficient approximate Nearest-Neighbor Search (NNS) (Jegou et al., 2011) or Maximum Inner-Product Search (MIPS) (Shrivastava and Li, 2014) is thus an active area of research, which we review in brief in the related work section.

Most approaches (Charikar, 2002; Jegou et al., 2011) aim to learn compact lower-dimensional representations that preserve distance information.

While there has been ample work on learning compact representations, learning sparse higher dimensional representations have been addressed only recently (Jeong and Song, 2018; Cao et al., 2018) .

As a seminal instance, Jeong and Song (2018) propose an end-to-end approach to learn sparse and high-dimensional hashes, showing significant speed-up in retrieval time on benchmark datasets compared to dense embeddings.

This approach has also been motivated from a biological viewpoint (Li et al., 2018) by relating to a fruit fly's olfactory circuit, thus suggesting the possibility of hashing using higher dimensions instead of reducing the dimensionality.

Furthermore, as suggested by Glorot et al. (2011) , sparsity can have additional advantages of linear separability and information disentanglement.

In a similar vein, in this work, we propose to learn high dimensional embeddings that are sparse and hence efficient to retrieve using sparse matrix multiplication operations.

In contrast to compact lowerdimensional ANN-esque representations that typically lead to decreased representational power, a key facet of our higher dimensional sparse embeddings is that they can have the same representational capacity as the initial dense embeddings.

The core idea behind our approach is inspired by two key observations: (i) retrieval of d (high) dimensional sparse embeddings with fraction p of non-zero values on an average, can be sped up by a factor of 1/p. (ii) The speed up can be further improved to a factor of 1/p 2 by ensuring that the non-zero values are evenly distributed across all the dimensions.

This indicates that sparsity alone is not sufficient to ensure maximal speedup; the distribution of the non-zero values plays a significant role as well.

This motivates us to consider the effect of sparsity on the number of floating point operations (FLOPs) required for retrieval with an inverted index.

We propose a penalty function on the embedding vectors that is a continuous relaxation of the exact number of FLOPs, and encourages an even distribution of the non-zeros across the dimensions.

We apply our approach to the large scale metric learning problem of learning embeddings for facial images.

Our training loss consists of a metric learning (Weinberger and Saul, 2009 ) loss aimed at learning embeddings that mimic a desired metric, and a FLOPs loss to minimize the number of operations.

We perform an empirical evaluation of our approach on the Megaface dataset (Kemelmacher-Shlizerman et al., 2016) , and show that our proposed method successfully learns high-dimensional sparse embeddings that are orders-of-magnitude faster.

We compare our approach to multiple baselines demonstrating an improved or similar speed-vs-accuracy trade-off.

The rest of the paper is organized as follows.

In Section 3 we analyze the expected number of FLOPs, for which we derive an exact expression.

In Section 4 we derive a continuous relaxation that can be used as a regularizer, and optimized using gradient descent.

We also provide some analytical justifications for our relaxation.

In Section 5 we then compare our method on a large metric learning task showing an improved speed-accuracy trade-off compared to the baselines.

Learning Compact Representations, ANN.

Exact retrieval of the top-k nearest neighbours is expensive in practice for high-dimensional dense embeddings learned from deep neural networks, with practitioners often resorting to approximate nearest neighbours (ANN) for efficient retrieval.

Popular approaches for ANN include Locality sensitive hashing (LSH) (Gionis et al., 1999; Andoni et al., 2015; Raginsky and Lazebnik, 2009 ) relying on random projections, Navigable small world graphs (NSW) (Malkov et al., 2014) and hierarchical NSW (HNSW) (Malkov and Yashunin, 2018) based on constructing efficient search graphs by finding clusters in the data, Product Quantization (PQ) (Ge et al., 2013; Jegou et al., 2011) approaches which decompose the original space into a cartesian product of low-dimensional subspaces and quantize each of them separately, and Spectral hashing (Weiss et al., 2009 ) which involves an NP hard problem of computing an optimal binary hash, which is relaxed to continuous valued hashes, admitting a simple solution in terms of the spectrum of the similarity matrix.

Overall, for compact representations and to speed up query times, most of these approaches use a variety of carefully chosen data structures, such as hashes (Neyshabur and Srebro, 2015; Wang et al., 2018) , locality sensitive hashes (Andoni et al., 2015) , inverted file structure (Jegou et al., 2011; Baranchuk et al., 2018) , trees (Ram and Gray, 2012) , clustering (Auvolat et al., 2015) , quantization sketches (Jegou et al., 2011; Ning et al., 2016) , as well as dimensionality reductions based on principal component analysis and t-SNE (Maaten and Hinton, 2008) .

End to End ANN.

Learning the ANN structure end-to-end is another thread of work that has gained popularity recently.

Norouzi et al. (2012) propose to learn binary representations for the Hamming metric by minimizing a margin based triplet loss.

Erin Liong et al. (2015) use the signed output of a deep neural network as hashes, while imposing independence and orthogonality conditions on the hash bits.

Other end-to-end learning approaches for learning hashes include (Cao et al., 2016; Li et al., 2017 ).

An advantage of end-to-end methods is that they learn hash codes that are optimally compatible to the feature representations.

Sparse Representations.

Sparse representations have been previously studied from various viewpoints.

Glorot et al. (2011) explore sparse neural networks in modeling biological neural networks and show improved performance, along with additional advantages such as better linear separability and information disentangling.

Ranzato et al. (2008; ; Lee et al. (2008) propose learning sparse features using deep belief networks.

Olshausen and Field (1997) explore sparse coding with an overcomplete basis, from a neurobiological viewpoint.

Sparsity in auto-encoders have been explored by Ng et al. (2011); Kavukcuoglu et al. (2010) .

Arpit et al. (2015) provide sufficient conditions to learn sparse representations, and also further provide an excellent review of sparse autoencoders.

Dropout (Srivastava et al., 2014 ) and a number of its variants (Molchanov et al., 2017; Park et al., 2018; Ba and Frey, 2013) have also been shown to impose sparsity in neural networks.

High-Dimensional Sparse Representations.

Sparse deep hashing (SDH) (Jeong and Song, 2018) is an end-to-end approach that involves starting with a pre-trained network and then performing alternate minimization consisting of two minimization steps, one for training the binary hashes and the other for training the continuous dense embeddings.

The first involves computing an optimal hash best compatible with the dense embedding using a min-cost-max-flow approach.

The second step is a gradient descent step to learn a dense embedding by minimizing a metric learning loss.

A related approach, k-sparse autoencoders (Makhzani and Frey, 2013) learn representations in an unsupervised manner with at most k non-zero activations.

The idea of high dimensional sparse embeddings is also reinforced by the sparse-lifting approach (Li et al., 2018) where sparse high dimensional embeddings are learned from dense features.

The idea is motivated by the biologically inspired fly algorithm (Dasgupta et al., 2017) .

Experimental results indicated that sparse-lifting is an improvement both in terms of precision and speed, when compared to traditional techniques like LSH that rely on dimensionality reduction.

1 regularization, Lasso.

The Lasso (Tibshirani, 1996) is the most popular approach to impose sparsity and has been used in a variety of applications including sparsifying and compressing neural networks .

The group lasso (Meier et al., 2008) is an extension of lasso that encourages all features in a specified group to be selected together.

Another extension, the exclusive lasso (Kong et al., 2014; Zhou et al., 2010) , on the other hand, is designed to select a single feature in a group.

Our proposed regularizer, originally motivated by idea of minimizing FLOPs closely resembles exclusive lasso.

Our focus however is on sparsifying the produced embeddings rather than sparsifying the parameters.

Sparse Matrix Vector Product (SpMV).

Existing work for SpMV computations include (Haffner, 2006; Kudo and Matsumoto, 2003) , proposing algorithms based on inverted indices.

Inverted indices are however known to suffer from severe cache misses.

Linear algebra back-ends such as BLAS (Blackford et al., 2002) rely on efficient cache accesses to achieve speedup.

Haffner (2006); Mellor-Crummey and Garvin (2004) ; Krotkiewski and Dabrowski (2010) propose cache efficient algorithms for sparse matrix vector products.

There has also been substantial interest in speeding up SpMV computations using specialized hardware such as GPUs (Vazquez et al., 2010; V??zquez et al., 2011) , FPGAs (Zhuo and Prasanna, 2005; Zhang et al., 2009) , and custom hardware (Prasanna and Morris, 2007) .

Metric Learning.

While there exist many settings for learning embeddings (Hinton and Salakhutdinov, 2006; Kingma and Welling, 2013; Kiela and Bottou, 2014) in this paper we restrict our attention to the context of metric learning (Weinberger and Saul, 2009) .

Some examples of metric learning losses include large margin softmax loss for CNNs (Liu et al., 2016) , triplet loss (Schroff et al., 2015) , and proxy based metric loss (Movshovitz-Attias et al., 2017).

In this section we study the effect of sparsity on the expected number of FLOPs required for retrieval and derive an exact expression for the expected number of FLOPs.

The main idea in this paper is based on the key insight that if each of the dimensions of the embedding are non-zero with a probability p (not necessarily independently), then it is possible to achieve a speedup up to an order of 1/p 2 using an inverted index on the set of embeddings.

Consider two embedding vectors u, v. Computing u T v requires computing only the pointwise product at the indices k where both u k and v k are non-zero.

This is the main motivation behind using inverted indices and leads to the aforementioned speedup.

Before we analyze it more formally, we introduce some notation.

be a set of n independent training samples drawn from Z = X ?? Y according to a distribution P, where X , Y denote the input and label spaces respectively.

Let F = {f ?? : X ??? R d | ?? ??? ??} be a class of functions parameterized by ?? ??? ??, mapping input instances to d-dimensional embeddings.

Typically, for image tasks, the function is chosen to be a suitable CNN (Krizhevsky et al., 2012) .

Suppose X, Y ??? P, then define the activation probability p j = P(f ?? (X) j = 0), and its empirical versionp j =

We now show that sparse embeddings can lead to a quadratic speedup.

Consider a d-dimensional sparse query vector u q = f ?? (x q ) ??? R d and a database of n sparse vectors

. .

, n) are sampled independently from P. Computing the vector matrix product Du q requires looking at only the columns of D corresponding to the non-zero entries of u q given by

1 Furthermore, in each of those columns we only need to look at the non-zero entries.

This can be implemented efficiently in practice by storing the non-zero indices for each column in independent lists, as depicted in Figure 1a .

The number of FLOPs incurred is given by,

Taking the expectation on both sides w.r.t.

x q , x (i) and using the independence of the data, we get

where X ??? P is an independent random sample.

Since the expected number of FLOPs scales linearly with the number of vectors in the database, a more suitable quantity is the mean-FLOPs-per-row defined as

Note that for a fixed amount of sparsity

this is minimized when each of the dimensions are non-zero with equal probability

2 (so that as a regularizer, F(f ?? , P) will in turn encourage such a uniform distribution across dimensions).

Given such a uniform distribution, compared to dense multiplication which has a complexity of O(d) per row, we thus get an improvement by a factor of 1/p 2 (p < 1).

Thus when only p fraction of all the entries is non-zero, and evenly distributed across all the columns, we achieve a speedup of 1/p 2 .

Note that independence of the non-zero indices is not necessary due to the linearity of expectation.

FLOPs versus Speedup.

While FLOPs reduction is a reasonable measure of speedup on primitive processors of limited parallelization and cache memory.

FLOPs is not an accurate measure of actual speedup when it comes to mainstream commercial processors such as Intel's CPUs and Nvidia's GPUs, as the latter have cache and SIMD (Single-Instruction Multiple Data) mechanism highly optimized for dense matrix multiplication, while sparse matrix multiplication are inherently less tailored to their cache and SIMD design (Sohoni et al., 2019) .

On the other hand, there have been threads of research on hardwares with cache and parallelization tailored to sparse operations that show speedup proportional to the FLOPs reduction (Han et al., 2016; Parashar et al., 2017) .

Modeling the cache and other hardware aspects can potentially lead to better performance but less generality and is left to our future works.

(a) The colored cells denote non-zero entries, and the arrows indicate the list structure for each of the columns, with solid arrows denoting links that were traversed for the given query.

The green and grey cells denote the non-zero entries that were accessed and not accessed, respectively.

The non-zero values in Duq (blue) can be computed using only the common non-zero values (green).

1: (Build Index) 2: Input: Sparse matrix D.

(stores the non-zero values and their indices as a list) 6: end for 7:

8: (Query) 9: Input: Sparse query u q .

10: Init score vector

end for 15: end for 16: return s

The 1 regularization is the most common approach to induce sparsity.

However, as we will also verify experimentally, it does not ensure an uniform distribution of the non-zeros in all the dimensions that is required for the optimal speed-up.

Therefore, we resort to incorporating the actual FLOPs incurred, directly into the loss function which will lead to an optimal trade-off between the search time and accuracy.

The FLOPs F(f ?? , P) being a discontinuous function of model parameters, is hard to optimize, and hence we will instead optimize using a continuous relaxation of it.

Denote by (f ?? , D), any metric loss on D for the embedding function f ?? .

The goal in this paper is to minimize the loss while controlling the expected FLOPs F(f ?? , P) defined in Eqn.

2.

Since the distribution P is unknown, we use the samples to get an estimate of F(f ?? , P).

Recall the empirical fraction of non-zero

, which converges in probability to p j .

Therefore, a consistent estimator for F(f ?? , P) based on the samples D is given by F(f ?? , D) = d j=1p 2 j .

Note that F denotes either the empirical or population quantities depending on whether the functional argument is P or D. We now consider the following regularized loss.

for some parameter ?? that controls the FLOPs-accuracy tradeoff.

The regularized loss poses a further hurdle, asp j and consequently F(f ?? , D) are not continuous due the presence of the indicator functions.

We thus compute the following continuous relaxation.

Define the mean absolute activation a j = E[|f ?? (X) j |] and its empirical version?? j = 1 n n i=1 |f ?? (x i ) j |, which is the 1 norm of the activations (scaled by 1/n) in contrast to the 0 quasi norm in the FLOPs calculation.

Define the relaxations, F(f ?? , P) = .

We propose to minimize the following relaxation, which can be optimized using any off-the-shelf stochastic gradient descent optimizer.

min

Sparse Retrieval.

During inference, the sparse vector of a query image is first obtained from the learned model and the nearest neighbour is searched in a database of sparse vectors forming a sparse matrix.

An efficient algorithm to compute the dot product of the sparse query vector with the sparse matrix is presented in Figure 1b .

This consists of first building a list of the non-zero values and their positions in each column.

As motivated in Section 3, given a sparse query vector, it is sufficient to only iterate through the non-zero values and the corresponding columns.

Using the scores from the above step, a shortlist of candidates having the top scores is first constructed.

The shortlisted candidates are further re-ranked using the dense embeddings.

The number of candidates is chosen such that the dense re-ranking time does not dominate the sparse ranking time.

Comparison to SDH (Jeong and Song, 2018) .

It is instructive to contrast our approach with that of SDH (Jeong and Song, 2018) .

In contrast to the binary hashes in SDH, our approach learns sparse real valued representations.

SDH uses a min-cost-max-flow approach in one of the training steps, while we train ours only using SGD.

During inference in SDH, a shortlist of candidates is first created by considering the examples in the database that have hashes with non-empty intersections with the query hash.

The candidates are further re-ranked using the dense embeddings.

The shortlist in our approach on the other hand is constituted of the examples with the top scores from the sparse embeddings.

Comparison to unrelaxed FLOPs regularizer.

We provide an experimental comparison of our continuous relaxation based FLOPs regularizer to its unrelaxed variant, showing that the performance of the two are markedly similar.

Setting up this experiment requires some analytical simplifications based on recent DNN analyses.

We first recall recent results that indicate that the output of a batch norm layer nearly follows a Gaussian distribution (Santurkar et al., 2018) , so that in our context, we could make the simplifying approximation that

, ?? is the ReLU activation, and where we suppress the dependency of ?? j and ?? j on P. We experimentally verify that this assumption holds by minimizing the KS distance (Massey Jr, 1951) between the CDF of ??(X) with X ??? N (??, ?? 2 ) and the empirical CDF of the activations, with respect to ??, ??.

Figure 2a shows the empirical CDF and the fitted CDF of ??(X) for two different architectures.

While ?? j , ?? j cannot be tuned independently for j ??? [d] due to their dependence on ??, consider a further simplification where the parameters are independent of each other.

Suppose for j ??? {1, 2}, f ?? (X) j = ReLU(X) where X ??? N (?? j , ?? 2 j ), and ?? = (?? 1 , ?? 2 , ?? 1 , ?? 2 ).

We analyze how minimizing F(f ?? , P) compares to minimizing F(f ?? , P).

Note that we consider the population quantities here instead of the empirical quantities, as they are more amenable to theoretical analyses.

We also consider the 1 regularizer as a baseline.

We initialize with ?? = (?? 1 , ?? 2 , ?? 1 , ?? 2 ) = (???1/4, ???1.3, 1, 1), and minimize the three quantities w.r.t.

?? via gradient descent with infinitesimally small learning rates.

For this contrastive analysis, we do not consider the effect of the metric loss.

Note that while the empirical quantity F(f ?? , D) cannot be optimized via gradient descent, it is possible to do so for its population counterpart F(f ?? , P) since it is available in closed form when making Gaussian assumptions.

The details of computing the gradients can be found in Appendix A. Figure 2b shows the trajectory of the activation probabilities (p 1 , p 2 ) during optimization.

We start with (p 1 , p 2 ) = (0.4, 0.1), and plot the trajectory taken when performing gradient descent.

Without the effect of the metric loss, the probabilities are expected to go to zero as observed in the figures.

It can be seen that, in contrast to the 1 -regularizer, F and F tend to sparsify the less sparse activation (p 1 ) at a faster rate, which corroborates the fact that they encourage an even distribution of non-zeros.

F promotes orthogonality.

We next show that when the embeddings are normalized to have a unit norm, as typically done in metric learning, then minimizing F(f ?? , D) is equivalent to promoting orthogonality on the absolute values of the embedding vectors.

Let f ?? (x) 2 = 1, ???x ??? X , we then have the following:

is minimized when the vectors {|f ?? (x i )|} n i=1 are orthogonal.

Metric learning losses aim at minimizing the interclass dot product, whereas the FLOPs regularizer aims at minimizing pairwise dot products irrespective of the class, leading to a tradeoff between sparsity and accuracy.

This approach of pushing the embeddings apart, bears some resemblance to the idea of spreading vectors (Sablayrolles et al., 2019) where an entropy based regularizer is used to uniformly distribute the embeddings on the unit sphere, albeit without considering any sparsity.

Maximizing the pairwise dot product helps in reducing FLOPs as is illustrated by the following toy example.

Consider a set of d

Then p,q??? [1:d] |v p |, |v q | is minimized when v p = e p , where e p is an one-hot vector with the p th entry equal to 1 and the rest Figure (a) shows that the CDF of the activations (red) closely resembles the CDF of ??(X) (blue) where X is a Gaussian random variable.

Figure (b) shows that F and F behave similarly by sparsifying the less sparser activation at a faster rate when compared to the 1 regularizer.

0.

The FLOPs regularizer thus tends to spread out the non-zero activations in all the dimensions, thus producing balanced embeddings.

This simple example also demonstrates that when the number of classes in the training set is smaller or equal to the number of dimensions d, a trivial embedding that minimizes the metric loss and also achieves a small number of FLOPs is f ?? (x) = e y where y is true label for x. This is equivalent to predicting the class of the input instance.

The caveat with such embeddings is that they might not be semantically meaningful beyond the specific supervised task, and will naturally hurt performance on unseen classes, and tasks where the representation itself is of interest.

In order to avoid such a collapse in our experiments, we ensure that the embedding dimension is smaller than the number of training classes.

Furthermore, as recommended by Sablayrolles et al. (2017) , we perform all our evaluations on unseen classes.

We evaluate our proposed approach on a large scale metric learning dataset: the Megaface (Kemelmacher-Shlizerman et al., 2016) used for face recognition.

This is a much more fine grained retrieval tasks (with 85k classes for training) compared to the datasets used by Jeong and Song (2018) .

This dataset also satisfies our requirement of the number of classes being orders of magnitude higher than the dimensions of the sparse embedding.

As discussed in Section 4, a few number of classes during training can lead the model to simply learn an encoding of the training classes and thus not generalize to unseen classes.

Face recognition datasets avoid this situation by virtue of the huge number of training classes and a balanced distribution of examples across all the classes. (2018) consisting of 1 million images spanning 85k classes.

We evaluate with 1 million distractors from the Megaface dataset and 3.5k query images from the Facescrub dataset (Ng and Winkler, 2014) , which were not seen during training.

Network Architecture.

We experiment with two architectures: MobileFaceNet (Chen et al., 2018), and ResNet-101 (He et al., 2016) .

We use ReLU activations in the embedding layer for MobileFaceNet, and SThresh activations for ResNet.

The activations are 2 -normalized to produce an embedding on the unit sphere, and used to compute the Arcface loss (Deng et al., 2018) .

We learn 1024 dimensional sparse embeddings for the 1 and F regularizers; and 128, 512 dimensional dense embeddings as baselines.

All models were implemented in Tensorflow (Abadi et al., 2016) with the sparse retrieval algorithm implemented in C++.

2 The re-ranking step used 512-d dense embeddings.

Activation Function.

In practice, having a non-linear activation at the embedding layer is crucial for sparsification.

Layers with activations such as ReLU are easier to sparsify due to the bias parameter in the layer before the activation (linear or batch norm) which acts as a direct control knob to the sparsity.

More specifically, ReLU(x ??? ??) can be made more (less) sparse by increasing (decreasing) the components of ??, where ?? is the bias parameter of the previous linear layer.

In this paper we consider two types of activations: ReLU(x) = max(x, 0), and the soft thresholding operator SThresh(x) = sgn(x) max(|x|???1/2, 0) (Boyd and Vandenberghe, 2004) .

ReLU activations always produce positive values, whereas soft thresholding can produce negative values as well.

Practical Considerations.

In practice, setting a large regularization weight ?? from the beginning is harmful for training.

Sparsifying too quickly using a large ?? leads to many dead activations (saturated to zero) in the embedding layer and the model getting stuck in a local minima.

Therefore, we use an annealing procedure and gradually increase ?? throughout the training using a regularization weight schedule ??(t) : N ??? R that maps the training step to a real valued regularization weight.

In our experiments we choose a ??(t) that increases quadratically as ??(t) = ??(t/T ) 2 , until step t = T , where T is the threshold step beyond which ??(t) = ??.

Baselines.

We compare our proposed F-regularizer, with multiple baselines: exhaustive search with dense embeddings, sparse embeddings using 1 regularization, Sparse Deep Hashing (SDH) (Jeong and Song, 2018) , and PCA, LSH, PQ applied to the 512 dimensional dense embeddings from both the architectures.

We train the SDH model using the aforementioned architectures for 512 dimensional embeddings, with number of active hash bits k = 3.

We use numpy (using efficient MKL optimizations in the backend) for matrix multiplication required for exhaustive search in the dense and PCA baselines.

We use the CPU version of the Faiss (Johnson et al., 2017) library for LSH and PQ (we use the IVF-PQ index from Faiss).

Further details on the training hyperparameters and the hardware used can be found in Appendix B.

We report the recall and the time-per-query for various hyperparameters of our proposed approach and the baselines, yielding trade-off curves.

The reported times include the time required for re-ranking.

The trade-off curves for MobileNet and ResNet are shown in Figures 3a and 3c respectively.

We observe that while vanilla 1 regularization is an improvement by itself for some hyperparameter settings, the F regularizer is a further improvement, and yields the most optimal trade-off curve.

SDH has a very poor speed-accuracy trade-off, which is mainly due to the explosion in the number of shortlisted candidates with increasing number of active bits leading to an increase in the retrieval time.

On the other hand, while having a small number of active bits is faster, it leads to a smaller recall.

For the other baselines we notice the usual order of performance, with PQ having the best speed-up compared to LSH and PCA.

While dimensionality reduction using PCA leads to some speed-up for relatively high dimensions, it quickly wanes off as the dimension is reduced even further.

We also report the sub-optimality ratio R sub = F(f ?? , D)/dp 2 computed over the dataset D, wher?? p = 1 d d j=1p j is the mean activation probability estimated on the test data.

Notice that R ??? 1, and the optimal R = 1 is achieved whenp j =p, ???j ??? [1 : d] , that is when the non-zeros are evenly distributed across the dimensions.

The sparsity-vs-suboptimality plots for MobileNet and ResNet are shown in Figures 3a and 3c respectively.

We notice that the F-regularizer yields values of R closer to 1 when compared to the 1 -regularizer.

For the MobileNet architecture we notice that the 1 regularizer is able to achieve values of R close to that of F in the less sparser region.

However, the gap increases substantially with increasing sparsity.

For the ResNet architecture on the other hand the 1 regularizer yields extremely sub-optimal embeddings in all regimes.

The F regularizer is therefore able to produce more balanced distribution of non-zeros.

The sub-optimality is also reflected in the recall values.

The gap in the recall values of the 1 and F models is much higher when the sub-optimality gap is higher, as in the case of ResNet, while it is small when the sub-optimality gap is smaller as in the case of MobileNet.

This shows the significance of having a balanced distribution of non-zeros.

In this paper we proposed a novel approach to learn high dimensional embeddings with the goal of improving efficiency of retrieval tasks.

Our approach integrates the FLOPs incurred during retrieval into the loss function as a regularizer and optimizes it directly through a continuous relaxation.

We provide further insight into our approach by showing that the proposed approach favors an even distribution of the non-zero activations across all the dimensions.

We experimentally showed that our approach indeed leads to a more even distribution when compared to the 1 regularizer.

We compared our approach to a number of other baselines and showed that it has a better speed-vs-accuracy trade-off.

Overall we were able to show that sparse embeddings can be around 50?? faster compared to dense embeddings without a significant loss of accuracy.

Proof.

Follows directly from Lemma 3.

Lemma 5.

Proof.

Follows directly from Lemma 2.

All images were resized to size 112 ?? 112 and aligned using a pre-trained aligner 3 .

For the Arcloss function, we used the recommended parameters of margin m = 0.5 and temperature s = 64.

We trained our models on 4 NVIDIA Tesla V-100 GPUs using SGD with a learning rate of 0.001, momentum of 0.9.

Both the architectures were trained for a total of 230k steps, with the learning rate being decayed by a factor of 10 after 170k steps.

We use a batch size of 256 and 64 per GPU for MobileFaceNet for ResNet respectively.

Pre-training in SDH is performed in the same way as described above.

The hash learning step is trained on a single GPU with a learning rate of 0.001.

The ResNet model is trained for 200k steps with a batch size of 64, and the MobileFaceNet model is trained for 150k steps with a batch size of 256.

We set the number of active bits k = 3 and a pairwise cost of p = 0.1.

Hyper-parameters for MobileNet models.

Re-ranking.

We use the following heuristic to create the shortlist of candidates after the sparse ranking step.

We first shortlist all candidates with a score greater than some confidence threshold.

For our experiments we set the confidence threshold to be equal to 0.25.

If the size of this shortlist is larger than k, it is further shrunk by consider the top k scorers.

For all our experiments we set k = 1000.

This heuristic avoids sorting the whole array, which can be a bottleneck in this case.

The parameters are chosen such that the time required for the re-ranking step does not dominate the total retrieval time.

1.

All models were trained on 4 NVIDIA Tesla V-100 GPUs with 16G of memory.

2.

System Memory: 256G.

3. CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz.

4. Number of threads: 32.

5.

Cache: L1d cache 32K, L1i cache 32K, L2 cache 256K, L3 cache 46080K.

All timing experiments were performed on a single thread in isolation.

C ADDITIONAL RESULTS C.1 RESULTS WITHOUT RE-RANKING Figure 4 shows the comparison of the approaches with and without re-ranking.

We notice that there is a significant dip in the performance without re-ranking with the gap being smaller for ResNet with FLOPs regularization.

We also notice that the FLOPs regularizers has a better trade-off curve for the no re-ranking setting as well.

In the main text we have reported the recall@1 which is a standard face recognition metric.

This however is not sufficient to ensure good face verification performance.

The goal in face verification is to predict whether two faces are similar or dissimilar.

A natural metric in such a scenario is the FPR-TPR curve.

Standard face verification datasets include LFW (Huang et al., 2008) and AgeDB (Moschoglou et al., 2017) .

We produce embeddings using our trained models, and use them to compute similarity scores (dot product) for pairs of images.

The similarity scores are used to compute the FPR-TPR curves which are shown in Figure 5 .

We notice that for curves with similar probability of activation p, the FLOPs regularizer performs better compared to 1 .

This demonstrates the efficient Figure 5 : FPR-TPR curves.

The 1 curves are all shown in shades of red, where as the FLOPs curves are all shown in shades of blue.

The probability of activation is provided in the legend for comparison.

For curves with similar probability of activation p, the FLOPs regularizer performs better compared to 1 , thus demonstrating that the FLOPs regularizer learns richer representations for the same sparsity.

utilization of all the dimensions in the case of the FLOPs regularizer that helps in learning richer representations for the same sparsity.

We also observe that the gap between sparse and dense models is smaller for ResNet, thus suggesting that the ResNet model learns better representations due to increased model capacity.

Lastly, we also note that the gap between the dense and sparse models is smaller for LFW compared to AgeDB, thus corroborating the general consensus that LFW is a relatively easier dataset.

We also experimented with the Cifar-100 dataset (Krizhevsky et al., 2009 ) consisting of 60000 examples and 100 classes.

Each class consists of 500 train and 100 test examples.

We compare the 1 and FLOPs regularized approaches with the sparse deep hashing approach.

All models were trained using the triplet loss (Schroff et al., 2015) and embedding dim d = 64.

For the dense and DH baselines, no activation was used on the embeddings.

For the 1 and FLOPs regularized models we used the SThresh activation.

Similar to Jeong and Song (2018) , the train-test and test-test precision values have been reported in Table 1 .

Furthermore, the reported results are without re-ranking.

Cifar-100 being a small dataset, we only report the FLOPs-per-row, as time measurements can be misleading.

In our experiments, we achieved slightly higher precisions for the dense model compared to (Jeong and Song, 2018) .

We notice that our models use less than 50% of the computation compared to SDH, albeit with a slightly lower precision.

Table 1 : Cifar-100 results using triplet loss and embedding size d = 64.

For 1 and F models, no re-ranking was used.

F is used to denote the FLOPs-per-row (lower is better).

<|TLDR|>

@highlight

We propose an approach to learn sparse high dimensional representations that are fast to search, by incorporating a surrogate of the number of operations directly into the loss function.
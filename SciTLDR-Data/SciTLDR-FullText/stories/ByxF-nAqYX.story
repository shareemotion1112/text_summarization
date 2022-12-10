The paper, interested in unsupervised feature selection, aims to retain the features best accounting for the local patterns in the data.

The proposed approach, called Locally Linear Unsupervised Feature Selection, relies on a dimensionality reduction method to characterize such patterns; each feature is thereafter assessed according to its compliance w.r.t.

the local patterns, taking inspiration from Locally Linear Embedding (Roweis and Saul, 2000).

The experimental validation of the approach on the scikit-feature benchmark suite demonstrates its effectiveness compared to the state of the art.

Machine Learning faces statistical and computational challenges due to the increasing dimension of modern datasets.

Dimensionality reduction aims at addressing such challenges through embedding the data in a lower dimensionality space, in an unsupervised BID17 BID28 BID34 BID41 or supervised BID9 BID8 BID25 way.

The requirement for understandable Machine Learning BID36 BID5 however makes it desirable to achieve interpretable dimensionality reduction.

In order to do so, the simplest way is to select a subset of the initial features, i.e. to achieve feature selection (FS), as opposed to generating compound new features from the initial ones, a.k.a.

feature construction.

For instance, determining the genes most important w.r.t.

a given disease or the underlying generative model of the data can be viewed as the mother goal in bioinformatics BID13 BID23 .In the supervised ML setting, features are assessed and selected based on their relevance to the prediction goal BID11 BID31 BID3 .

Unsupervised learning, aimed at making sense of the data, however constitutes a primary and most important task of ML, as emphasized by BID20 , while supervised ML intervenes at a later stage of the data exploitation process.

Unsupervised FS approaches BID14 BID47 BID2 BID22 BID48 ) (more in section 2) essentially rely on the assumption that the data samples are structured in clusters, and use the cluster partition in lieu of labels, making it possible to fall down on supervised FS, and select the features most amenable to characterize and separate the clusters.

A main limitation of this methodology is that clusters are bound to rely on some metric defined from the initial features (with the notable exception of BID22 ), although this metric can be arbitrarily corrupted based on irrelevant or random features.

On the other hand, as far as one considers the unsupervised setting, a feature can hardly be considered irrelevant per se.

The main contribution of the paper is to address both limitations: the proposed approach, called Locally Linear Unsupervised Feature Selection (LLUFS) jointly determines patterns in the data, and features relevant to characterize these patterns.

LLUFS is a 2-step process (Sec. 3): In a first step, a compressed representation of the data is built using Auto-Encoders BID37 BID7 .

In a second step, viewing the initial dataset as a high-dimensional embedding of the compressed dataset, each feature is scored according to its contribution to the reconstruction error of the embedding, taking inspiration from Locally Linear Embedding BID28 BID29 BID40 .After describing the goals of experiments and the experimental setting used to validate the approach, extensively relying on the scikit-feature project BID21 SKf, 2018) (Sec. 4) , the empirical validation is presented and discussed (Sec. 5), establishing the merits and discussing the weaknesses of the approach.

The paper concludes with a discussion and some perspectives for further research.

X denotes the m × D data matrix.

Row X[i, :], also noted x i when no confusion is to fear, is the i-th sample (x i in R D ).

Column X[:, j] is the j-th feature.

For j in [[1, D] ], µ j and σ j respectively denote the mean and the standard deviation of the j-th feature on the dataset.

1 denote the m-dimensional constant vector [1, ..., 1] t .Let S denote an m × m similarity matrix (S i,i > 0), with ∆ the associated diagonal degree matrix DISPLAYFORM0 2 ).

Two particular similarities will be considered in the following; the supervised similarity SU P , with SU P i,j = 0 iff x i and x j do not belong to the same class, and 1/|C i | if they both belong to class C i , and the unsupervised similarity RBF , with DISPLAYFORM1 2 }, and δ a hyper-parameter of the method.

Supervised FS aims to select a subset of features such that it maximizes the eventual classifier accuracy.

Supervised FS algorithms divide into filters, wrappers and embedded methods.

Filter methods BID43 BID30 operate at the data pre-processing stage, and are agnostic to the classifier algorithm.

Wrappers methods BID15 BID27 aim to determine the feature subset yielding a best accuracy when used within a specific classifier, through solving a black-box optimization problem.

Embedded methods BID12 BID46 alternatively learn and use the learned hypothesis to prune/select the unpromising/promising features.

Admittedly, wrapper and embedded approaches might produce a candidate feature set with moderate generality (being linked to a particular classifier) and moderate interpretability (with the retained features being good as a gang).

Since this paper focuses on unsupervised and interpretable FS, only filter methods will thus be considered in the following.

An early supervised filter method based on the so-called Fisher score was introduced by BID6 , independently ranking features according to their correlation with the labels.1 A general limitation of such scores is that they achieve a myopic feature selection, with XOR problems − where all relevant features need to be taken into account simultaneously − as typical failure cases.

A prominent unsupervised filter approach is based on spectral clustering: data clusters are first built using some metric or similarity; thereafter supervised FS approaches are used with these clusters in lieu of classes BID39 BID1 .

BID14 introduce the Laplacian score 1 , where each feature score measures how well this feature accounts for the sample similarity.

Interestingly, while the Fisher score is a particular case of Laplacian score using the SU P similarity, the Laplacian score overcomes the myopic limitations of the Fisher score when using the RBF similarity.

The Laplacian score is also remotely related to the MaxVariance FS method BID16 , selecting features with large variance for the sake of their higher representative power.

Also relying on spectral clustering is the SPEC approach BID47 , proposing three scores respectively noted φ 1 , φ 2 and φ 3 .

SPEC relies on the core idea that relevant features be smooth w.r.t.

the graph, i.e. slowly varying among samples close to each other.

After the spectral clustering theory BID32 BID26 , considering eigenvectors ξ 0 , ..., ξ m−1 of the normalized Laplacian L (respectively associated with eigenvalues λ 0 < λ 1 < ...

< λ m−1 ), smooth features are aligned with the first eigenvectors, hence the score φ 1 : DISPLAYFORM0 1 For the sake of space limitations, formal definitions are reminded in Appendix 1.Eigenvectors ξ 0 , ..., ξ m−1 of L define soft cluster indicators, and eigenvalues λ 0 < λ 1 < ...

< λ m−1 measure the separability of the clusters.

The smaller φ 1j , the more efficient the j-th feature is to separate the clusters.

As the first eigenvector ξ 0 = ∆ 1 2 1 does not carry any information, with λ 0 = 0, one might rather consider the projection of the feature vector X[:, j] on the orthogonal space of ξ 0 : DISPLAYFORM1 Finally, in the case where the target number of clusters κ is known, only the top-κ eigenvectors are considered, and score φ 3 is defined as: DISPLAYFORM2 Features are ranked in ascending order for φ 1 and φ 2 , and in descending order for φ 3 .The above three scores measure the overall capacity of a feature to separate clusters, which might prove inefficient in multi-classes/multi clusters settings: a feature most efficient to separate a pair of clusters might have a mediocre general score.

The Multi-Cluster Feature Selection (MCFS) BID2 addresses this limitation by defining a score per cluster.

Formally, the capacity of X[:, j] to separate clusters is estimated through fitting the eigenvectors (reminding that ξ k is a soft indicator of the k-th cluster) up to a regularization term.

Letting A[k,:] denote a vector in R D , with β a regularization weight: DISPLAYFORM3 The L 1 regularization term enforces the sparsity of A[k, :], retaining only the features most relevant to this cluster.

The overall MCFS score simply takes the maximum over all clusters of the absolute value of A k,j : DISPLAYFORM4 A general limitation of the above scores is to rely on a similarity metric that can be arbitrarily corrupted by noisy features, potentially leading to irrelevant clusters and scores.

This limitation is addressed by BID22 , introducing the Nonnegative Discriminative Feature Selection (NDFS) approach.

NDFS jointly optimizes the D × D feature importance matrix A together with a cluster indicator matrix ξ, with an L 2 regularization: DISPLAYFORM5 subject to ξ orthogonal and semi-positive definite (ξ t ξ = I D , ξ ≥ 0), with α, β regularization weights.

Following BID44 , the first term is rewritten as : DISPLAYFORM6 The minimization of Eq. (6) tends to enforce the intra-cluster similarity and the inter-cluster dissimilarity.

The orthogonality and nonnegativity constraints on ξ further enforce that each sample belong in exactly one cluster.

This section first presents LLE for the sake of self-containedness, before presenting and discussing LLUFS.

LLUFS takes inspiration from the Locally Linear Embedding (LLE) defined by BID28 ; BID29 .

LLE relies on the so-called Johnson Lindenstrauss lemma BID19 : DISPLAYFORM0 with f being a linear mapping (composed only of translations, rotations and rescalings).As this lemma guarantees the existence of a low-dimensional embedding approximately preserving the pairwise distances among the points, LLE BID28 : i) defines the local structure of the m data points x i ∈ R D , through approximating each point as the barycenter of its n W nearest neighbors; ii) finds points z 1 , . . .

z m in R d , with d D, such that the z i satisfy the same local relationships as the x i s. Formally, let N (i) denote the set of indices of the n W nearest neighbors of x i ; weights W i,j such that they minimize the Euclidean distance DISPLAYFORM1 Note that W is invariant under rotation, translation or homothety on the dataset X: it captures the local structure of the x i s. The LLE dimensionality reduction thus proceeds by finding another set of points z i s in R d , such that they satisfy the local relationships expressed by W : DISPLAYFORM2

While LLE is performed as a dimensionality reduction technique, its principle is general: after the local structure of the data has been captured through matrix W , this matrix can be used to transport the data from any source to target representation.

As our goal is to achieve feature selection, we implicitly assume that the X data live in a lowdimension space.

Accordingly, the proposed LLUFS approach proceeds by: i) finding a lowdimension representation of the data in R d ; ii) characterizing the matrix W capturing the data structure in this low-dimension representation; iii) using W to assess the initial features, as detailed below.

This step can be achieved using linear or non-linear approaches, ranging from PCA BID42 and SVD BID4 to Isomap BID34 or t-SNE BID24 .

For the sake of generality and robustness, LLUFS uses the non-linear Stacked Denoising AutoEncoder neural networks (SDAE) BID38 BID7 , meant to achieve a non-linear compression robust w.r.t.

input noise.

Let us consider the Z as the "true" data, with the X as an inflated and corrupted image of the Z. The overall loss of information from Z to X is measured as X − W X 2 F .

Most interestingly, this overall loss of information can be decomposed with respect to examples, with: DISPLAYFORM0 and with respect to the initial features, with: DISPLAYFORM1 The distorsion associated to the j-th feature is thus interpreted in terms of how much this feature is corrupted with respect to the "true" local structure of the data, defined from the Zs.

The features with lowest distorsion thus are deemed the most representative of the data.

Note that, although the distorsion score is defined for each initial feature, it might implicitly take into account the global structure of the data, captured by the W .

One weakness of the method is that the distorsion scores depend on the latent representation produced by the auto-encoder, which might be biased due to the redundancy of the initial features; typically, duplicating an initial feature will entail that the latent representation is more able to express this feature, mechanically reducing its distorsion score.

For this reason, a preliminary step is to detect and reduce the redundancy of the initial features.

In order to do so, LLUFS i) normalizes the initial features (with zero mean and unit variance); ii) uses Agglomerative Hierarchical feature clustering BID18 BID35 , using a high number of clusters n c (n c = 3 4 D in the experiments); iii) selects one feature per cluster (the nearest one to the cluster mean); iv) apply the auto-encoder on the pruned data.

Further work is concerned with taking into account the feature redundancy within the AE loss.

A second limitation is due to the sensitivity of the distorsion score to the feature distribution.

Typically, while a constant feature carries no information, its distorsion is null.

Likewise, the distorsion of discrete features depends on their being balanced.

In order to alleviate this issue, the reliability of the distorsion associated to each feature is measured through an empirical p-value BID33 .

Given a p-value threshold τ , 1/τ copies of each feature are generated and independently shuffled.

The feature distorsion is deemed relevant iff it is lower than the distorsion of all shuffled copies.

1 LLUFS (empirical p-value threshold τ , number of clusters n c , embedding dimension d, number of neighbors n W ); Input : X, n c , τ , d, n W 2 Normalize features to zero mean and unit variance.

3 Perform Agglomerative Hierarchical Feature Clustering, producing X f iltered ∈ R m×nc .4 Train a SDAE on X f iltered to produce compressed representation Z. 5 Solve W = arg min Z − W Z 2 F subject to the positivity and sum-to-1 constraints.

DISPLAYFORM0 Initialize set of candidates Cand to all features, ranked in ascending order w.r.t.

distorsion.

Initialize selection subset Sel = Ø. 10 while |Cand| > 0 do for k ∈ [[1, DISPLAYFORM1

The main goal of the experimental validation is to assess LLUFS compared to state of the art unsupervised feature selection approaches.

The performance assessment commonly falls back on the supervised setting, where the indicator is the predictive accuracy of a classifier trained from the selected features, where the number of selected features ranges from 1 to D BID3 .

2 For the sake of clarity and to sidestep issues related to classifier hyper-parameter tuning, the classifier considered in the following is the 1-nearest neighbor classifier.

Secondly (Q2), the respective impacts of both LLUFS ingredients, the feature clustering pre-processing, and the proper FS mechanism, are assessed.

Thirdly (Q3), experiments will investigate the robustness of the proposed approach specifically w.r.t.

XOR concepts (Sec. 2).

The experimental setting extensively relies on the scikit-feature project BID21 SKf, 2018) , defining a de facto standard for FS approaches through algorithm implementations and datasets BID3 BID45 .

Five baseline algorithms are considered: Laplacian Score (LAP) BID14 , Spectral Feature Selection (SPEC, considering the φ 1 score, Sec. 2) BID47 , Multi-Cluster Feature Selection (MCFS) BID2 , and Non-Negative Discriminative Feature Selection (NDFS) BID22 ; the fifth and last baseline (RANDOM), aimed to assess the impact of the only feature pre-processing in LLUFS (Q2), is defined by uniformly selecting the features after the feature clustering process (Alg.

1).

7 benchmark datasets from SKf (2018) are considered (Appendix 2): 6 datasets in the domains of image and bioinformatics, and the Madelon XOR problem BID11 to empirically investigate (Q3).

The number of features range from 500 to 10,000; the number of classes range from 2 to 11, and the number of examples is less than 200 (except for Madelon, with 2,000 examples).

As discussed in Sec. 3.3, only datasets with continuous features are considered.

All features are normalized (zero mean, unit variance).

LLUFS involves three hyper-parameters: the number n c of clusters used in the featurepreprocessing; the methodology used to build the latent representation; and the number of neighbors n W considered in Sec. 3 to build the W matrix.

n c is set to 3 4 D for all datasets.

The methodology used to build the latent representation is a 5-layer stacked denoising auto-encoder BID38 DISPLAYFORM0 with tanh activation function, trained to minimize the MSE loss for 10 2 epochs with a 10 −3 learning rate.

The denoising process uniformly selects 20% of the features and sets them to 0 for each example.

n W is set to 6 for all datasets, considering the small number of samples.

This section reports and discusses the comparative performance of all unsupervised FS methods, in view of the experiment goals.

3 On datasets ALLAML and TOX171 ( FIG4 and (b) ), LLUFS dominates all other methods over the whole learning curve.

On ALLAML, both LAP and NDFS do much better than MCFS, suggesting that the feature clusters are not much relevant to the classification task.

SPEC shows a robust performance after sufficiently many features have been selected (d > 50).

On TOX171, both LAP and MCFS do much better than NDFS, suggesting that the feature set presents a complex cluster structure (captured by NDFS) misleading to the classification task.

Likewise, SPEC yields good results after the beginning of the curve (d > 20).

In both cases, RANDOM is significantly outperformed, suggesting that quite a few feature clusters are irrelevant to the prediction task.

On datasets PIXRAW10P and ORLRAWS10P FIG4 ), LLUFS is dominated by NDFS at the beginning of the curve (d < 5 for PIXRAW10P and d < 10 for ORLRAWS10P); it thereafter dominates all other algorithms on ORLRAWS10P (resp.

dominates the others then reaches the same nearly maximal performance as all other algorithms on PIXRAW10P).

The relative comparative weakness of LLUFS at the very beginning of the curve is interpreted as LLUFS capturing patterns related to subsets of features (as the latent representation is bound to globally account for the initial features).

The importance of a feature standalone thus can hardly be accounted for, contrasting with NDFS.

On both datasets, RANDOM yields a decent performance (ranking 2nd or 3rd, especially for high values of d), which suggests that a main issue with those image datasets is the redundancy of the features.

On CARCINOM FIG4 ), LLUFS is dominated by all algorithms but MCFS at the beginning of the curve (d < 20); it then catches up and dominates the other algorithms for d > 60.

The best algorithm on this dataset is LAP, suggesting that the cluster structure defined from all features is relevant to the prediction task, which might explain why LLUFS and MCFS, more sensitive to local patterns, are outperformed.

On LUNG FIG4 ), LLUFS is consistently dominated by NDFS and SPEC over the whole learning curve and performs similarly as RANDOM, suggesting that the compressed representation learned by the neural network does not accurately represent data structure.

3 The variance of the predictive accuracy over 25 independent runs of the RANDOM baseline is circa 0.02 for d < 5, 5 * 10 −3 for d = 20); the confidence bars are omitted in the figures for the sake of readability.

As said, the Madelon problem (Appendix 2) is chosen to investigate the comparative performances of unsupervised FS methods in the case where, by construction, FS based on independent feature scores are bound to fail.

Two clusters of methods are clearly seen on this problem ( FIG4 ).Most methods fail to do better than random selection, due to the low signal to noise ratio (96% of the features being pure noise), adversely affecting spectral clustering methods and clusters based on Laplacian eigenvectors.

NDFS does much better, as it simultaneously learns cluster indicators and feature relevance.

LLUFS does even better as the latent representation tends to highlight the data patterns, and the distorsion score measures whether a feature is relevant to these patterns.

The drop of performance of NDFS and LLUFS, after the number of selected features goes beyond the number of relevant features FORMULA3 , is blamed on the addition of noise features perturbing the metric and misleading the 1-nn classifier.

A novel approach to unsupervised feature selection has been proposed in this paper, with a proof of concept of its empirical merits.

The core idea is to find an "oracle" representation of the data, and to consider the actual data as an inflated and corrupted image of the oracle data.

The quality of each feature is thereafter assessed depending on how it contributes to the loss of information between the "oracle" and the actual data.

A first perspective for further research, taking inspiration from NDFS, is to allow a feature to be partially relevant, e.g. through considering the quantiles of its distorsion.

A second perspective is to integrate the feature redundancy in the auto-encoder loss, to decrease the bias in favor of redundant features.

The approach will also be extended to supervised feature selection.

Denoting c the number of classes, n i the number of samples in the i-th class, µ ij and σ ij respectively the mean and the standard deviation of the j-th feature on the i-th class, the Fisher score is defined as: DISPLAYFORM0 LAPLACIAN SCORE With same notations, DISPLAYFORM1 with σ j the standard deviation of the j-th feature and S i,k the similarity of the i-th and k-th examples.

Note that L j can be rewritten using the Laplacian matrix: DISPLAYFORM2 1 t ∆1 1.

Furthermore, the Fisher score is a particular case of the Laplacian score: Madelon, an artificial XOR-like dataset, was created for the Feature Selection Challenge of NIPS2003 BID10 .

It involves 5 relevant features, the values of which are combined along two XOR concepts to define the positive and negative classes.

Specifically, each class includes 2 DISPLAYFORM3

Gaussian clusters, placed on the vertices of a hypercube.

The relevant 5 features are duplicated, combined and perturbed to obtain 15 "distractor" features.

480 standard normal noise features are then added.

Most classifiers (SVMs with linear, polynomial or Gaussian kernels; NNs with up to 10 layers) using all features yields the same 50% accuracy as random prediction, indicating that efficient FS is compulsory.

@highlight

Unsupervised feature selection through capturing the local linear structure of the data

@highlight

Proposes locally linear unsupervised feature selection.

@highlight

The paper proposes the LLUFS method for feature selection.
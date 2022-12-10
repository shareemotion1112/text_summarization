Profiling cellular phenotypes from microscopic imaging can provide meaningful biological information resulting from various factors affecting the cells.

One motivating application is drug development: morphological cell features can be captured from images, from which similarities between different drugs applied at different dosages can be quantified.

The general approach is to find a function mapping the images to an embedding space of manageable dimensionality whose geometry captures relevant features of the input images.

An important known issue for such methods is separating relevant biological signal from nuisance variation.

For example, the embedding vectors tend to be more correlated for cells that were cultured and imaged during the same week than for cells from a different week, despite having identical drug compounds applied in both cases.

In this case, the particular batch a set of experiments were conducted in constitutes the domain of the data; an ideal set of image embeddings should contain only the relevant biological information (e.g. drug effects).

We develop a general framework for adjusting the image embeddings in order to `forget' domain-specific information while preserving relevant biological information.

To do this, we minimize a loss function based on distances between marginal distributions (such as the Wasserstein distance) of embeddings across domains for each replicated treatment.

For the dataset presented, the replicated treatment is the negative control.

We find that for our transformed embeddings (1) the underlying geometric structure is not only preserved but the embeddings also carry improved biological signal (2) less domain-specific information is present.

In the framework where our approach is applicable, there are some inputs (e.g. images) and a map F sending the inputs to vectors in a low-dimensional space which summarizes information about the inputs.

F could either be engineered using specific image features, or learned (e.g. using deep neural networks).

We will call these vectors 'embeddings' and the space to which they belong the 'embedding space'.

Each input may also have corresponding semantic labels and domains, and for inputs with each label and domain pair, F produces some distribution of embeddings.

Semantically meaningful similarities between pairs of inputs can then be assessed by the distance between their corresponding embeddings, using some chosen distance metric.

Ideally, the embedding distribution of a group of inputs depends only on their label, but often the domain can influence the embedding distribution as well.

We wish to find an additional map to adjust the embeddings produced by F so that the distribution of adjusted embeddings for a given label is independent of the domain, while still preserving semantically meaningful distances between distributions of inputs with different labels.

The map F can be used for phenotypic profiling of cells.

In this application, images of biological cells perturbed by one of several possible biological stimuli (e.g. various drug compounds at different doses, some of which may have unknown effects) are mapped to embeddings, which are used to reveal similarities among the applied perturbations.

There are a number of ways to extract embeddings from images of cells.

One class of methods such as that used by BID10 relies on extracting specifically engineered features.

In the recent work by BID1 , a Deep Metric Network pre-trained on consumer photographic images (not microscope images of cells) described in BID16 was used to generate embedding vectors from cellular images, and it was shown that these clustered drug compounds by their mechanisms of action (MOA) more effectively.

See Figure 1 for example images of the different MOAs.

Currently one of the most important issues with using image embeddings to discriminate the effects of each treatment (i.e. a particular dose of a drug, the 'label' in the general problem described above) on morphological cell features is nuisance factors related to slight uncontrollable variations in each biological experiment.

Many cell imaging experiments are organized into a number of batches of experiments occurring over time, each of which contains a number of sample plates (typically 3-6), each of which contains individual wells in which thousands of cells are grown and treatments are applied (typically around 96 wells per plate).

For this application, the 'domain' is an instance of one of these hierarchical levels, and embeddings for cells with a given treatment tend to be closer to each other within the same domain than from a different one.

For example, the experimentalist may apply slightly different concentrations or amounts of a drug compound in two wells in which the same treatment was anticipated.

Another example is the location of a particular well within a plate or the order of the plate within a batch, which may influence the rate of evaporation, and hence, the appearance of the cells.

Finally, 'batch' effects may result from differences in experiment conditions (temperature, humidity) from week to week; they are various instances of this hierarchical level that we will consider as 'domains' in this work.

Our approach addresses the issue of nuisance variation in embeddings by transforming the embedding space in a possibly domain-specific way in order to minimize the variation across domains for a given treatment.

We remark that our main goal is to introduce a general flexible framework to address this problem.

In this framework, we use a metric function measuring the distances among pairs of probability distributions to construct an optimization problem whose solution yields appropriate transformations on each domain.

In our present implementation, the Wasserstein distance is used as a demonstration of a specific choice of the metric that can yield substantial improvements.

The Wasserstein distance makes few assumptions about the probability distributions of the embedding vectors.

Our approach is fundamentally different than those which explicitly identify a fixed 'target' and 'source' distributions.

Instead, we incorporate information from all domains on an equal footing, transforming all the embeddings.

This potentially allows our method to incorporate several replicates of a treatment across different domains to learn the transformations, and not only the controls.

We highlight that other distances may be used in our framework, such as the Cramer distance.

This may be preferable since the Cramer distance has unbiased sample gradients BID3 .

This could reduce the number of steps required to adjust the Wasserstein distance approximation for each step of training the embedding transformation.

Additionally we propose several other extensions and variations in Section 4.1.

Denote the embedding vectors x t,d,p for t ∈ T , d ∈ D, and p ∈ I t,d , where T and D are the treatment and domain labels respectively, and I t,d is the set of indices for embeddings belonging to treatment t and domain d. Suppose that x t,d,p were sampled from a probability distribution ν t,d .

our goal is to 'forget' the nuisance variation in the embeddings, which we formalize in the following way.

We wish to find maps A d transforming the embedding vectors such that the transformed marginals ν t,d have the property that for each t ∈ T and d i , d j ∈ D,ν t,di ≈ν t,dj (for some suitable metric between distributions).

Intuitively, the transformations A d can be thought of as correcting a domainspecific perturbation.

We do not have 'source' and 'target' distributions, and instead perturb all the embedding distributions simultaneously.

The transformations A d should be small to avoid distorting the underlying geometry of the embedding space, since we do not expect nuisance variation to be very large.

The 1-Wasserstein distance (hereafter will be simply referred to as the Wasserstein distance) between two probability distributions ν r and ν g on a compact metric space χ with metric δ is given by DISPLAYFORM0 γ∈Π(νr,νg)E (x,y)∼γ δ(x, y).Here Π(ν r , ν g ) is the set of all joint distributions γ(x, y) whose marginals are ν r and ν g .

This can be intuitively interpreted as the minimal cost of a transportation plan between the probability masses of ν r and ν g .

In our application, the metric space was R n and δ was the Euclidean metric.

If the Wasserstein distance between two distributions is zero, then it becomes impossible to discern the origin of a sample from one of these two distributions.

In addition, the Wasserstein distance (as well as other related metrics for probability distributions) are more appropriate to use than classifiers.

This is because classifiers are more sensitive to the distinguishability between probability distributions than other potentially meaningful features.

For instance, two otherwise identical Gaussian distributions displaced from one another would have Wasserstein distance equal to the displacement between them.

On the contrary, a classifier would yield a function that has vanishing gradients for sufficiently large displacement.

Given two or more probability distributions, their mean can be defined under the Wasserstein distance, known as the 'Wasserstein barycenter'.

Explicitly, the Wasserstein barycenter of N distributions ν 1 , ..., ν N is defined as the distribution µ that minimizes DISPLAYFORM1 The Wasserstein barycenter and its computation have been studied in many contexts, such as optimal transport theory BID5 BID0 .

In BID15 , the Wasserstein barycenter has been suggested as a method to remove nuisance variation in highthroughput biological experiments.

Two key ingredients of the Wasserstein barycenter are that (i) the nuisance variation is removed in the sense that a number of distinct distributions are transformed into a common distribution, and hence become indistinguishable; and (ii) the distributions are minimally perturbed by the transformtions.

Our method is based on these two requirements, where a separate map is associated with each domain.

For each treatment, the average Wasserstein distance among all pairs of transformed distributions across domains is included in the loss function.

Specifically, the average Wasserstein distance is formulated as DISPLAYFORM2 where the coefficient is the normalizing constant.

When multiple treatments are considered, the same number of average Wasserstein distances corresponding to the treatments are included in the loss function.

Thus, (i) is achieved by minimizing a loss function containing pairwise Wasserstein distances.

Compared with the ResNet used in BID12 , we achieve (ii) by early stopping or adding a regularization term to the loss function.

In Section 4.1, we will present another possible formulation that aligns more closely with the idea of the Wasserstein barycenter.

One distinct advantage of the Wasserstein distance is that this metric avoids problematic vanishing gradients during training, which are known to occur for metrics based on the KL-divergence, such as the cross entropy .

This is important from a practical point of view because vanishing gradients may halt the solving of the resulting minimax problem in our method.

The Wasserstein distance does not have a closed form except for a few special cases, and must be approximated in some way.

The Wasserstein distance is closely related to the maximum mean discrepancy (MMD) approximated in BID12 using an empirical estimator based on the kernel method.

This method requires selecting a kernel and relevant parameters.

In our application, we do not have a fixed 'target' distribution, so the kernel parameters would have to be updated during training.

We choose instead to use a method based on the ideas in and BID8 to train a neural network to estimate the Wasserstein distance.

A similar approach has been proposed in BID13 for domain adaptation.

To do this, first apply the Kantorovich-Rubinstein duality: DISPLAYFORM3 Here, ν r and ν g are two probability distributions.

The function f is in the space of Lipschitz functions with Lipschitz constant at most 1.

To estimate the Wasserstein distance, a function f can be optimized while keeping the norm of its gradient to be less than one.

We will call f the 'Wasserstein function' throughout this manuscript.

As a preprocessing step, we transform the embeddings for the dataset of interest such that the embeddings for the negative controls have mean zero and an identity covariance matrix (see Section 3.1 for details).

We observe that the embeddings for wells corresponding to different dosages of each compound are all shifted away from the origin in roughly the same direction by an amount that generally increases with dosage.

The variances of embeddings along the largest principal axes also increase in a manner consistent with the drugs inducing an affine transformation of the embeddings.

Given these observations, we choose to model the impact of nuisance variation by affine transformations, the intuition being that we can treat nuisance variations as small, random, drug-like perturbations resulting from unobserved covariates.

It is worth mentioning that we do not expect this assumption to hold generally.

In the current implementation, the domain-specific transformations A d map input embeddings to transformed embeddings of the same dimension.

Each A d is formulated as an affine transformation DISPLAYFORM0

is a penalized approximation to the Wasserstein distance between domains d i and d j , the function R(θ T ) is a regularization term for the learned transformation whose purpose is to preserve the geometry of the original embeddings, M t denotes the number of domains in which treatment t appears, and | · | represents the cardinality of a set.

In this paper, we explore either (i) neglecting R entirely and relying on early stopping instead, and (ii) specifying R as described below and in (eq. 6).

Using one of these methods is necessary since otherwise optimizing L may result in embeddings which contain no treatment information (for example, if all embeddings are transformed to a single point).There may be many possible forms for R, and these could involve multiple parameter choices for different components of the transformation that θ T determines.

In our case, θ T parameterizes an affine transformation, and hence we choose DISPLAYFORM0 where · F denotes the Frobenius norm, · 2 denotes the 2 norm, and q denotes the embedding dimensionality.

Moreover, there are two regularization weights λ M and λ b .In (eq. 5), W t,di,dj is used to approximate the Wasserstein distance between the transformed embeddings of domains d i and d j for treatment t upon optimization over θ W .

The Wasserstein distance is given by DISPLAYFORM1 Each Wasserstein function f t,di,dj in (eq. 7) depends on the parameters θ W , while each transformation A d depends on the parameters θ T .

For simplicity, we assume that N = |I t,di | = |I t,dj |, where | · | represents the cardinality of a set.

This is a reasonable assumption because in practice, the sets I t,d are chosen as minibatches in stochastic gradient descent.

Each of the terms g t,di,dj is a gradient penalty defined in (eq. 8-10).Each Wasserstein function should be Lipschitz with Lipschitz constant 1.

For differentiable functions, this is equivalent to the norm being bounded by 1 everywhere.

We use an approach based on BID8 to impose a soft constraint on the norm of the gradient.

In this approach, the hard constraint is replaced by a penalty, which is a function of the gradient of the Wasserstein function evaluated at some set of points.

The penalty term is weighted by an additional parameter γ.

We find that the value of γ = 10 used in BID8 works well in our application, and fix it throughout.

We remark this is an appropriate choice since it is large enough so that the approximation error in the Wasserstein function is small, while not causing numerical difficulties in the optimization routine.

Since it is impossible to check the gradient everywhere, we use the same strategy as BID8 : choose the intermediate points DISPLAYFORM2 randomly, where ∈ U [0, 1] and p k and q k denote the k th element of I t,di and I t,dj , respectively.

Denote the set of intermediate points by J t,di,dj .

Intuitively, the reason for sampling along these paths is that the Wasserstein function f whose gradient must be constrained has the interpretation of characterizing the optimal transport between the two probability distributions, and therefore it is most important for the gradient constraint to hold in the intermediate region between the distributions.

This is motivated more formally by Proposition 1 in BID8 , which shows that an optimal transport plan occurs along straight lines with gradient norm 1 connecting coupled points between the probability distributions.

Unlike BID8 , we impose the gradient penalty only if the gradient norm is greater than 1.

Doing so works better in practice for our application.

Explicitly, we define each gradient penalty g t,di,dj as DISPLAYFORM3 where DISPLAYFORM4 To approximate the Wasserstein distance we must maximize over θ W .

Thus, our objective is to find DISPLAYFORM5 We use the approach of BID6 to transform our minimax problem to a minimization problem by adding a 'gradient reversal' between the transformed embeddings and the approximated Wasserstein distances.

The gradient reversal is the identity in the forward direction, but negates the gradients used for backpropagation.

The embeddings under consideration are generated using the method described in BID1 , and summarized in Section 3.1.

We use the image set BBBC021v1 BID4 available from the Broad Bioimage Benchmark Collection BID9 .

This dataset corresponds to cells prepared on 55 plates across 10 separate batches, and imaged in three color channels (i.e. stains); for a population of control cells, a compound (DMSO) with no anticipated drug effect was applied, while various other drug compounds were applied to the remaining cells.

We compute the corresponding embeddings for each cell image using the method in Ando et al. FORMULA1 We use the same subset of treatments (concentration of a particular compound) evaluated in BID10 and BID1 .

This subset has 103 treatments from 38 compounds, each belonging to one of 12 known mechanism of action (MOA) groups.

Sample cell images from the 12 MOA groups are shown in Figure 1 .

In Figure 6 , we show a heatmap of the cosine similarity matrix between pairs of the selected treatments for the TVN embeddings.

This figure shows how embeddings of the same compound, and embeddings of the compounds with the same MOA have a tendency to cluster closer to each other in terms of the cosine distance.

Figure 1: A flowchart describing the procedure we use to generate and remove nuisance variation from image embeddings.

The embedding generation is described in Section 3.1 is characterized by F, which maps each 128 by 128 color image into a 192-dimensional embedding vector.

The nuisance variation removal by our method is denoted by WDN (Wasserstein Distance Network).

The 12 images on the right side show representative images of cells treated with drug compounds with one of the 12 known mechanisms of action (MOA), from the BBBC021 dataset BID9 .

Our method is evaluated by three metrics, the first two of which measure how much biological signal is preserved in the transformed embeddings, and the last one of which measures how much nuisance variation has been removed.

Each compound in the BBBC021 dataset has a known MOA.

A desirable property of embedding vectors is that compounds with the same MOA should group closely in the embedding space.

This property can be assessed in the following way using the ground truth MOA labels for each treatment.

First, compute the mean m X of the embeddings for each treatment X in each domain.

Find the nearest k neighbors n X,1 , n X,2 , ..., n X,k of m X either (i) not belonging to the same compound or (ii) not belonging to the same compound or batch (domain), and compute the portion of them having the same MOA as m X .

Our metric is defined as the average of this quantity across all treatment instances X in all domains.

If nuisance variation is corrected by transforming the embeddings, we may expect this metric to increase.

The reason for excluding same-domain nearest neighbors is to avoid the in-domain correlations from interfering with the metric.

The nearest k neighbors are found based on the cosine distance, which is more natural for the embedding space than the Euclidean distance, and can be directly compared with methods in existing literature.

Moreover, our k-NN metrics are generalizations of the 1-NN metrics used in BID10 and BID1 .

Cluster validation measures provide another way of characterizing how well compounds from the same MOA group together in embedding space.

In our application, each 'cluster' is a chosen MOA containing a group of treatments, and each point in a cluster is the mean of embeddings for a particular treatment (i.e. compound and concentration) and domain.

The Silhouette index is one such measure that compares each point's distance from points in its own cluster to its distance from points in other clusters.

It is defined as DISPLAYFORM0 where a(i) is the average distance from point i to all other points in its cluster, and b(i) is the minimum of all average distances from i to all other clusters (i.e. the distance to the closest neighboring cluster) BID11 .

The Silhouette index ranges between -1 and 1, with higher values indicating better clustering results.

Another metric measures how well domain-specific nuisance information has been 'forgotten'.

To do this, for each treatment we train a classifier to predict for each embedding the batch (domain) from the set of possible batches (domains) for that treatment.

We evaluate both a linear classifier (i.e. logistic regression) and a random forest with 3-fold cross validation.

If nuisance variation is being corrected, the batch (domain) classification accuracy should decrease significantly.

Because only the negative control (i.e., DMSO) has replicates across experiment batches in our dataset, we train and evaluate these two batch classifiers on this compound only.

For the model with either early stopping or a regularization term, the hyperparameters (i.e., the stopping time step or the regularization weights) can be selected by a cross-validation procedure to avoid overfitting (see BID7 for an example).

In particular, we apply this procedure to the case of early stopping.

Each time, an individual compound is held out, and the stopping time step is determined by maximizing the average k-NN MOA assignment metric for k = 1, ..., 4 on the remaining compounds.

FIG0 illustrates the k-NN MOA assignment metrics as a function of time steps in the case when early stopping is used with compound mitoxantrone held out.

For the embeddings transformed at the optimal time step, we evaluate the k-NN MOA assignment metrics for the held-out compound.

The procedure is repeated for all the compounds, and the k-NN MOA assignment metrics are aggregated across all the compounds.

Intuitively, for each fold of this leave-one-compound-out cross-validation procedure, the held-out compound can be treated as a new compound with unknown MOA, and the hyperparameters are optimized over the compounds with known MOAs.

In our case, we find that the optimal time step remains the same, i.e., 28000, regardless of the held-out compound.

To assess whether the improvements in the k-NN MOA assignment metric and the Silhouette index are statistically significant, we estimate the standard errors of the metrics using a nonparametric bootstrap method.

Each time, the bootstrap samples are generated by sampling with replacement the embeddings preprocessed by TVN in each well, and the metrics are evaluated using the bootstrap samples.

We repeat the procedure for 200 times, and obtain the standard errors of the 200 bootstrap estimates of the metrics, which are summarized in TAB2 .

The embedding transformations DISPLAYFORM0 , since we wish for the learned transformations to be not too far from the identity transformation.

To approximate each of the Wasserstein functions f t,di,dj in (eq. 7), we use a network consisting of softplus layer followed by a scalar-valued affine transformation.

The softplus loss is chosen because the Wasserstein distance estimates it produces are less noisy than other kinds of losses and it avoids the issue of all neurons becoming deactivated (which can occur for example when using RELU activations).The dimension of the softplus layer used to approximate each Wasserstein function is 2.

Optimization is done using stochastic gradient instead of the sums in (eq. 7).

For simplicity, the minibatch size for each treatment per iteration step is fixed throughout.

In the results presented, the minibatch size is 50.

Optimization for both classes of parameters θ T and θ W is done using separate RMSProp optimizers.

Prior to training θ T , we use a 'pre-training' period of 20000 time steps to obtain a good approximation for the Wasserstein distances.

After this, we alternate between adjusting θ T for 40 time steps and optimizing over θ W for a single time step.

We compare our results to either using no transformation other than normalization (TVN) and CORAL.

CORAL applies a domain-specific affine transformation to the embeddings represented as the rows of a matrix X d from domain d in the following way.

On the negative controls only, the covariance matrix across the entire experiment C as well as the covariance C d in each domain d are computed.

Notice that since TVN had already been applied (see Section 3.1), C = I. Then, all embedding coordinates in domain d are aligned by matching the covariance structures.

Alignment is done by computing the new embeddings DISPLAYFORM0 and R = C + ηI are regularized covariance matrices, with the regularizer η = 1, which is the same as that in BID1 .Other variations of the training procedure are discussed in Sections 3.4.3 and 3.5.

FIG1 shows the first two principal components of the embeddings transformed by WDN, compared with the embeddings preprocessed by TVN (see Section 3.1) and the embeddings generated by the CORAL method proposed in BID14 and applied by BID1 .

FIG5 shows the dosage response for each compound based on each set of transformed embeddings.

WDN is seen to better preserve the geometry of the embeddings than CORAL.

TAB2 shows the k-NN MOA assignment metrics of our transformed embeddings (early stopping and some particular choices of the regularization weights) compared to the original embeddings as well as the estimated standard errors.

We also include the values of this metric for CORAL.

We find that our WDN method performs better than CORAL in terms of the k-NN MOA assignment metrics.

Finally, TAB3 compares the average batch classification accuracy for a linear classifier (i.e., logistic regression) and a random forest classifier for the original TVN embeddings, WDN embeddings (early stopping and some particular choices of the regularization weights), CORAL embeddings, and for reference, a trivial transformation for which all embeddings are set to zero.

For each run, given a classifier and a transformed set of embeddings, we compute the mean accuracy for that classifier using 3-fold cross validation.

We see that the batch classification accuracy for the embeddings using our method is substantially smaller than that using TVN or CORAL, indicating our method is removing nuisance variation.

We have tried regularizing the network either with a regularization term or early stopping.

When using a regularization term, the loss function and the evaluation metrics converge for a chosen set of regularization weights.

We present the resulting k-NN MOA assignment metrics in TAB2 for several values of λ = λ M = λ b , as well as for the early stopping at the optimal time step 28000.

We see that the smaller regularization (λ = 40) results in a greater removal of nuisance variation.

However, removing more nuisance variation may be counterbalanced by also removing relevant biological signal, as suggested by the k-NN MOA assignment metrics in TAB2 .

In addition, using a non-optimal choice of the regularization weight may result in a lower Silhouette index, as shown in Table 3 .Each approach has advantages and disadvantages.

Using early stopping is simpler and does not require a computationally intensive grid search over all parameters to obtain optimal results, but on the other hand this may be a limiting factor in performance because of the smaller selection of parameters.

If the transformed embedding vectors do not follow an approximately direct path throughout the optimization, early stopping may miss the optimal solution.

This development is likely not a problem in our applications, since the transformation is small.

This explains why early stopping does not seem to produce negative side effects.

We find that early stopping produces a better result in terms of the k-NN MOA assignment metrics than the values of λ we have tried, but we anticipate using a more thorough search over the regularization weights would yield similar results between the two methods.

The learning curves for both the early stopping case and some regularization weights are shown in FIG4 .

To assess how the hyperparameters of the model affect its performance, we conduct additional experiments by varying the hyperparameters.

For example, the minibatch size is increased from 50 to 100.

The results are similar except that the learning curve in the case of 100 appears less noisy.

Moreover, the architecture of the network that estimates the pairwise Wasserstein distances is made more complicated by increasing the number of hidden layers from two to three and four, and the number of nodes per layer from two to four and eight, respectively.

Again, there is no significant difference in the results except that the curves of the k-NN MOA assignment metrics over the number of time steps appeared more stable.

We have shown how a neural network can be used to transform embedding vectors to 'forget' specifically chosen domain information as indicated by our proposed domain classification metric.

The transformed embeddings still preserve the underlying geometry of the space and improve the k-NN MOA metrics.

Our approach uses the Wasserstein distance and can in principle handle fairly general distributions of embeddings (as long as the neural network used to approximate the Wasserstein function is general enough).

Importantly, we do not have to assume that the distributions are Gaussian.

The framework itself is quite general and extendible (see Section 4.1).

Unlike methods that use only the controls for adjusting the embeddings, our method can also utilize information from replicates of a treatment across different domains.

However, the dataset used did not have treatment replicates across batches, so we only relied on aligning based on the controls.

Thus we implicitly assume that the transformation for the controls matches that of the various compounds.

We expect our method to be more useful in the context of experiments where many replicates are present, so that they can all be aligned simultaneously.

We expect transformations learned for such experiments to have better generalizability since it would be using available knowledge from a greater portion of the embedding space.

Our approach requires a choice of free parameters, either for regularization or early stopping, which we address by cross validation across compounds.

We discuss potential future directions below, as well as other limiting issues.

63.6 ± 1% 39.8 ± 0.6% 66.4 ± 0.7% 28.0 ± 0.8% 46.8 ± 0.9% 56.2 ± 0.9% 16.6% RF 45.9 ± 0.2% 34.4 ± 0.7% 46.8 ± 0.6% 26.7 ± 0.7% 33.3 ± 0.7% 39.5 ± 0.1% 16.6% Table 3 : We show the silhouette index for TVN only, TVN + WDN, and TVN + CORAL, as discussed in Section 3.2.2.

Here WDN refers to the the result using early stopping, and λ = 40, 80, 160 refers to the result when using a regularization with λ = λ M = λ b .

Both WDN and CORAL appear to increase the cohesion, as measured by this index.

The estimated error denoted by ± was determined by the bootstrapping procedure described in Section 3.

One possible modification we considered would be to replace the form of the cost function by the following, which would more closely resemble finding the Wasserstein barycenter: DISPLAYFORM0 The difference is that instead of comparing the pairwise transformed distributions, we instead compare the transformed distributions to the original distributions.

One advantage for this approach is that it avoids the 'shrinking to a point' problem, and therefore does not require a regularization term or early stopping to converge to a meaningful solution.

However, we did not find better performance for the new form of the cost function (eq. 13) for our specific dataset.

An alternative regularization term to the one we used penalizing how much the transformation differs from the identity may be used.

One interesting choice might be to penalize the change of pairwise distances between treatments within a specific domain.

Intuitively, in-domain variations carry biological signal that we would like to preserve, and using such a regularization term does so explicitly.

The Wasserstein functions were approximated with very simple nonlinear functions, and it is possible better results would be obtained using more sophisticated functions capturing the Wasserstein distance and its gradients more accurately.

Similarly, The transformations A d could be generalized from affine to a more general class of functions.

As in BID12 , we expect residual networks would make natural candidates for these transformations.

One possibility is to fine-tune the Deep Metric Network used to generate the embeddings instead of training a separate network on its outputs (or perhaps several such networks for the separate image stains used).Another issue is how to weigh the various Wasserstein distances against each other.

This might improve the results if there are many more points from some distributions than others (which happens in the real data).

Further, it is unclear how a regularization term should be weighed against the Wasserstein loss terms.

Another extension may involve applying our method hierarchically on the various domains of the experiment.

However, this would require replicates on multiple hierarchical levels.

Since the k-NN MOA assignment metric is based on the cosine distance, it is possible better results could be obtained by modifying the metric used to compute the Wasserstein distance accordingly, e.g. finding an optimal transportation plan only in non-radial directions.

A LEARNING CURVES WITH AND WITHOUT REGULARIZATION (a) No regularization term (i.e. λ = 0).

This is the training routine that was used together with early stopping.

These plots show that WDN better preserves the geometry of the embedding space than CORAL, where the latter can magnify the scale of the response.

WDN regularized by early stopping and a regularization term both slightly alter the embeddings.

C COMPOUND SIMILARITY MATRIX Figure 6 : A heatmap showing the cosine similarity matrix between pairs of treatments for the TVN embeddings.

Same-MOA compounds are grouped together, and the blue lines show distinctions between different MOAs.

The block diagonal terms correspond to the similarity matrices for same-MOA compounds.

This plot shows how same-MOA compounds tend to be more closely clustered together in the embedding space.

<|TLDR|>

@highlight

We correct nuisance variation for image embeddings across different domains, preserving only relevant information.

@highlight

Discusses a method for adjusting image embeddings in order tease apart technical variation from biological signal.

@highlight

The authors present a method to remove domain-specific information while preserving the relevant biological information by training a network that minimizes the Wasserstein distance between distrbutions.
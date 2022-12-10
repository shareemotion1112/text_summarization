Deep neural networks (DNNs) usually contain millions, maybe billions, of parameters/weights, making both storage and computation very expensive.

This has motivated a large body of work to reduce the complexity of the neural network by using sparsity-inducing regularizers.

Another well-known approach for controlling the complexity of DNNs is parameter sharing/tying, where certain sets of weights are forced to share a common value.

Some forms of weight sharing are hard-wired to express certain in- variances, with a notable example being the shift-invariance of convolutional layers.

However, there may be other groups of weights that may be tied together during the learning process, thus further re- ducing the complexity of the network.

In this paper, we adopt a recently proposed sparsity-inducing regularizer, named GrOWL (group ordered weighted l1), which encourages sparsity and, simulta- neously, learns which groups of parameters should share a common value.

GrOWL has been proven effective in linear regression, being able to identify and cope with strongly correlated covariates.

Unlike standard sparsity-inducing regularizers (e.g., l1 a.k.a.

Lasso), GrOWL not only eliminates unimportant neurons by setting all the corresponding weights to zero, but also explicitly identifies strongly correlated neurons by tying the corresponding weights to a common value.

This ability of GrOWL motivates the following two-stage procedure: (i) use GrOWL regularization in the training process to simultaneously identify significant neurons and groups of parameter that should be tied together; (ii) retrain the network, enforcing the structure that was unveiled in the previous phase, i.e., keeping only the significant neurons and enforcing the learned tying structure.

We evaluate the proposed approach on several benchmark datasets, showing that it can dramatically compress the network with slight or even no loss on generalization performance.

Deep neural networks (DNNs) have recently revolutionized machine learning by dramatically advancing the state-of-the-art in several applications, ranging from speech and image recognition to playing video games BID20 .

A typical DNN consists of a sequence of concatenated layers, potentially involving millions or billions of parameters; by using very large training sets, DNNs are able to learn extremely complex non-linear mappings, features, and dependencies.

A large amount of research has focused on the use of regularization in DNN learning BID20 , as a means of reducing the generalization error.

It has been shown that the parametrization of many DNNs is very redundant, with a large fraction of the parameters being predictable from the remaining ones, with no accuracy loss BID14 .

Several regularization methods have been proposed to tackle the potential over-fitting due to this redundancy.

Arguably, the earliest and simplest choice is the classical 2 norm, known as weight decay in the early neural networks literature , and as ridge regression in statistics.

In the past two decades, sparsity-inducing regularization based on the 1 norm (often known as Lasso) BID35 , and variants thereof, became standard tools in statistics and machine learning, including in deep learning BID20 .

Recently, BID32 used group-Lasso (a variant of Lasso that assumes that parameters are organized in groups and encourages sparsity at the group level BID37 ) in deep learning.

One of the effects of Lasso or group-Lasso regularization in learning a DNN is that many of the parameters may become exactly zero, thus reducing the amount of memory needed to store the model, and lowering the computational cost of applying it.

Figure 1: A DNN is first trained with GrOWL regularization to simultaneously identify the sparse but significant connectivities and the correlated cluster information of the selected features.

We then retrain the neural network only in terms of the selected connectivities while enforcing parameter sharing within each cluster.

It has been pointed out by several authors that a major drawback of Lasso (or group-Lasso) regularization is that in the presence of groups of highly correlated covariates/features, it tends to select only one or an arbitrary convex combination of features from each group BID6 BID7 BID17 BID28 BID42 .

Moreover, the learning process tends to be unstable, in the sense that subsets of parameters that end up being selected may change dramatically with minor changes in the data or algorithmic procedure.

In DNNs, it is almost unavoidable to encounter correlated features, not only due to the high dimensionality of the input to each layer, but also because neurons tend to co-adapt, yielding strongly correlated features that are passed as input to the subsequent layer BID34 .In this work, we propose using, as a regularizer for learning DNNs, the group version of the ordered weighted 1 (OWL) norm BID17 , termed group-OWL (GrOWL), which was recently proposed by BID28 .

In a linear regression context, GrOWL regularization has been shown to avoid the above mentioned deficiency of group-Lasso regularization.

In addition to being a sparsity-inducing regularizer, GrOWL is able to explicitly identify groups of correlated features and set the corresponding parameters/weights to be very close or exactly equal to each other, thus taking advantage of correlated features, rather than being negatively affected by them.

In deep learning parlance, this corresponds to adaptive parameter sharing/tying, where instead of having to define a priori which sets of parameters are forced to share a common value, these sets are learned during the training process.

We exploit this ability of GrOWL regularization to encourage parameter sparsity and group-clustering in a two-stage procedure depicted in Fig. 1 : we first use GrOWL to identify the significant parameters/weights of the network and, simultaneously, the correlated cluster information of the selected features; then, we retrain the network only in terms of the selected features, while enforcing the weights within the same cluster to share a common value.

The experiments reported below confirm that using GrOWL regularization in learning DNNs encourages sparsity and also yields parameter sharing, by forcing groups of weights to share a common absolute value.

We test the proposed approach on two benchmark datasets, MNIST and CIFAR-10, comparing it with weight decay and group-Lasso regularization, and exploring the accuracy-memory trade-off.

Our results indicate that GrOWL is able to reduce the number of free parameters in the network without degrading the accuracy, as compared to other approaches.

In order to relieve the burden on both required memory and data for training and storing DNNs, a substantial amount of work has focused on reducing the number of free parameters to be estimated, namely by enforcing weight sharing.

The classical instance of sharing is found in the convolutional layers of DNNs BID20 .

In fact, weight-sharing as a simplifying technique for NNs can be traced back to more than 30 years ago BID24 BID30 .Recently, there has been a surge of interest in compressing the description of DNNs, with the aim of reducing their storage and communication costs.

Various methods have been proposed to approximate or quantize the learned weights after the training process.

BID15 have shown that, in some cases, it is possible to replace the original weight matrix with a low-rank approximation.

Alternatively, BID1 propose retraining the network layer by layer, keeping the layer inputs and outputs close to the originally trained model, while seeking a sparse transform matrix, whereas BID19 propose using vector quantization to compress the parameters of DNNs.

Network pruning is another relevant line of work.

In early work, BID25 and BID22 use the information provided by the Hessian of the loss function to remove less important weights; however, this requires expensive computation of second order derivatives.

Recently, BID21 reduce the number of parameters by up to an order of magnitude by alternating between learning the parameters and removing those below a certain threshold.

propose to prune filters, which seeks sparsity with respect to neurons, rather than connections; that approach relieves the burden on requiring sparse libraries or special hardware to deploy the network.

All those methods either require multiple training/retraining iterations or a careful choice of thresholds.

There is a large body of work on sparsity-inducing regularization in deep learning.

For example, BID12 exploit 1 and 0 regularization to encourage weight sparsity; however, the sparsity level achieved is typically modest, making that approach not competitive for DNN compression.

Group-Lasso has also been used in training DNNs; it allows seeking sparsity in terms of neurons BID32 BID2 BID41 BID27 or other structures, e.g., filters, channels, filter shapes, and layer depth BID36 .

However, as mentioned above, both Lasso and group-Lasso can fail in the presence of strongly correlated features (as illustrated in Section 4, with both synthetic data and real data.

A recent stream of work has focused on using further parameter sharing in convolutional DNNs.

By tying weights in an appropriate way, BID16 obtain a convolutional DNN with rotation invariance.

On the task of analyzing positions in the game Go, BID10 showed improved performance by constraining features to be invariant to reflections along the x-axis, y-axis, and diagonal-axis.

Finally, BID9 used a hash function to randomly group the weights such that those in a hash bucket share the same value.

In contrast, with GrOWL regularization, we aim to learn weight sharing from the data itself, rather than specifying it a priori.

Dropout-type methods have been proposed to fight over-fitting and are very popular, arguably due to their simplicity of implementation BID34 .

Dropout has been shown to effectively reduce over-fitting and prevent different neurons from co-adapting.

Decorrelation is another popular technique in deep learning pipelines BID4 BID11 BID29 ; unlike sparsity-inducing regularizers, these methods try to make full use of the model's capacity by decorrelating the neurons.

Although dropout and decorrelation can reduce over-fitting, they do not compress the network, hence do not address the issue of high memory cost.

It should also be mentioned that our proposal can be seen as complementary to dropout and decorrelation: whereas dropout and decorrelation can reduce co-adaption of nodes during training, GrOWL regularization copes with co-adaptation by tying together the weights associated to co-adapted nodes.

We start by recalling the definition of the group-OWL (GrOWL) regularizer and very briefly reviewing some of its relevant properties BID28 .

Definition 1.

Given a matrix W ∈ R n×m , let w [i]· denote the row of W with the i-th largest 2 norm.

Let λ ∈ R n + , with 0 < λ 1 ≥ λ 2 ≥ · · · ≥ λ n ≥ 0.

The GrOWL regularizer (which is a norm) DISPLAYFORM0 This is a group version of the OWL regularizer BID17 , also known as WSL1 (weighted sorted 1 BID40 ) and SLOPE BID5 , where the groups are the rows of its matrix argument.

It is clear that GrOWL includes group-Lasso as a special case when λ 1 = λ n .

As a regularizer for multiple/multi-task linear regression, each row of W contains the regression coefficients of a given feature, for the m tasks.

It has been shown that by adding the GrOWL regularizer to a standard squared-error loss function, the resulting estimate of W has the following property: rows associated with highly correlated covariates are very close or even exactly equal to each other BID28 .

In the linear case, GrOWL encourages correlated features to form predictive clusters corresponding to the groups of rows that are nearly or exactly equal.

The rationale underlying this paper is that when used as a regularizer for DNN learning, GrOWL will induce both sparsity and parameters tying, as illustrated in Fig. 2 and explained below in detail.

A typical feed-forward DNN with L layers can be treated as a function f of the following form: DISPLAYFORM0 L denotes the set of parameters of the network, and each f i is a componentwise nonlinear activation function, with the rectified linear unit (ReLU), the sigmoid, and the hyperbolic tangent being common choices for this function BID20 .

DISPLAYFORM1 , DNN learning may be formalized as an optimization problem, DISPLAYFORM2 where L y,ŷ is the loss incurred when the DNN predictsŷ for y, and R is a regularizer.

Here, we adopt as regularizer a sum of GrOWL penalties, each for each layer of the neural network, i.e., DISPLAYFORM3 where N l denotes the number of neurons in the l-th layer and 0 < λ DISPLAYFORM4 .., b L , the biases are not regularized, as is common practice.

As indicated in Eq. (3), the number of groups in each GrOWL regularizer is the number of neurons in the previous layer, i.e., λ (l) ∈ R N l−1 .

In other words, we treat the weights associated with each input feature as a group.

For fully connected layers, where W l ∈ R N l−1 ×N l , each group is a row of the weight matrix.

In convolutional layers, where W l ∈ R Fw×F h ×N l−1 ×N l , with F w and F h denoting the width and height, respectively, of each filter, we first reshape W l to a 2-dimensional array, i.e., DISPLAYFORM5 , and then apply GrOWL on the reshaped matrix.

That is, if the l-th layer is convolutional, then DISPLAYFORM6 Each row of W 2D l represents the operation on an input channel.

The rationale to apply the GrOWL regularizer to each row of the reshaped weight matrix is that GrOWL can select the relevant features of the network, while encouraging the coefficient rows of each layer associated with strongly correlated features from the previous layer to be nearly or exactly equal, as depicted in Fig. 2 .

The goal is to significantly reduce the complexity by: (i) pruning unimportant neurons of the previous layer that correspond to zero rows of the (reshaped) weight matrix of the current layer; (ii) grouping the rows associated with highly correlated features of the previous layer, thus encouraging the coefficient rows in each of these groups to be very close to each other.

As a consequence, in the retraining process, we can further compress the neural network by enforcing the parameters within each neuron that belong to the same cluster to share same values.

In the work of BID2 , each group is predefined as the set of parameters associated to a neuron, and group-Lasso regularization is applied to seek group sparsity, which corresponds to zeroing out redundant neurons of each layer.

In contrast, we treat the filters corresponding Figure 2: GrOWL's regularization effect on DNNs.

Fully connected layers (Left): for layer l, GrOWL clusters the input features from the previous layer, l − 1, into different groups, e.g., blue and green.

Within each neuron of layer l, the weights associated with the input features from the same cluster (input arrows marked with the same color) share the same parameter value.

The neurons in layer l − 1 corresponding to zero-valued rows of W l have zero input to layer l, hence get removed automatically.

Convolutional layers (right): each group (row) is predefined as the filters associated with the same input channel; parameter sharing is enforced among the filters within each neuron that corresponds with the same cluster (marked as blue with different effects) of input channels.to the same input channel as a group, and GrOWL is applied to prune the redundant groups and thus remove the associated unimportant neurons of the previous layer, while grouping associated parameters of the current layer that correspond with highly correlated input features to different clusters.

Moreover, as shown in Section 4, group-Lasso can fail at selecting all relevant features of previous layers, and for the selected ones the corresponding coefficient groups are quite dissimilar from each other, making it impossible to further compress the DNN by enforcing parameter tying.

To solve (2), we use a proximal gradient algorithm BID3 , which has the following general form: at the t-th iteration, the parameter estimates are updated according to DISPLAYFORM0 where, for some convex function Q, prox Q denotes its proximity operator (or simply "prox") BID3 , defined as prox DISPLAYFORM1 2 denotes the sum of the squares of the differences between the corresponding components of ν and ξ, regardless of their organization (here, a collection of matrices and vectors).Since R(θ), as defined in FORMULA4 , is separable across the weight matrices of different layers and zero for b 1 , ..., b L , the corresponding prox is also separable, thus DISPLAYFORM2 DISPLAYFORM3 It was shown by BID28 that the prox of GrOWL can be computed as follows.

For some matrix V ∈ R N ×M , let U = prox Ω λ (V ), and v i and u i denote the corresponding i-th rows.

Then, DISPLAYFORM4 DISPLAYFORM5 For vectors in R N (in which case GrOWL coincides with OWL), prox Ω λ (l) can be computed with O(n log n) cost, where the core computation is the socalled pool adjacent violators algorithm (PAVA BID13 )) for isotonic regression.

We provide one of the existing algorithms in Appendix A; for details, the reader is referred to the work of BID5 and BID39 .

In this paper, we apply the proximal gradient algorithm per epoch, which generally performs better.

The training method is summarized in Algorithm 1.

GrOWL is a family of regularizers, with different variants obtained by choosing different weight sequences λ 1 , . . .

, λ n .

In this paper, we propose the following choice: DISPLAYFORM6 where p ∈ {1, ...n} is a parameter.

The first p weights follow a linear decay, while the remaining ones are all equal to Λ 1 .

Notice that, if p = n, the above setting is equivalent to OSCAR BID6 .

Roughly speaking, Λ 1 controls the sparsifying strength of the regularizer, while Λ 2 controls the clustering property (correlation identification ability) of GrOWL BID28 .

Moreover, by setting the weights to a common constant beyond index p means that clustering is only encouraged among the p largest coefficients, i.e., only among relevant coefficient groups.

Finding adequate choices for p, Λ 1 , and Λ 2 is crucial for jointly selecting the relevant features and identifying the underlying correlations.

In practice, we find that with properly chosen p, GrOWL is able to find more correlations than OSCAR.

We explore different choices of p in Section 4.1.

After the initial training phase, at each layer l, rows of W l that corresponds to highly correlated outputs of layer l − 1 have been made similar or even exactly equal.

To further compress the DNN, we force rows that are close to each other to be identical.

We first group the rows into different clusters 1 according to the pairwise similarity metric DISPLAYFORM0 where W l,i and W l,j denote the i-th and j-th rows of W l , respectively.

With the cluster information obtained by using GrOWL, we enforce parameter sharing for the rows that belong to a same cluster by replacing their values with the averages (centroid) of the rows in that cluster.

In the subsequent retraining process , let Gk denote the k-th cluster of the l-th layer, then centroid g DISPLAYFORM1

We assess the performance of the proposed method on two benchmark datasets: MNIST and CIFAR-10.

We consider two different networks and compare GrOWL with group-Lasso and weight decay, in terms of the compression vs accuracy trade-off.

For fair comparison, the training-retraining pipeline is used with the different regularizers.

After the initial training phase, the rows that are close to each other are clustered together and forced to share common values in the retraining phase.

We implement all models using Tensorflow BID0 .

We evaluate the effect of the different regularizers using the following quantities: sparsity = (#zero params)/(# total params), compression rate = (# total params)/(# unique params), and parameter sharing = (# nonzero params)/(# unique params).

First, we consider a synthetic data matrix X with block-diagonal covariance matrix Σ, where each block corresponds to a cluster of correlated features, and there is a gap g between two blocks.

Within each cluster, the covariance between two features X i and X j is cov(X i , X j ) = 0.96 |i−j| , while features from different clusters are generated independently of each other.

We set n = 784, K = 10, block size 50, and gap g = 28.

We generate 10000 training and 1000 testing examples.

FORMULA14 ).We train a NN with a single fully-connected layer of 300 hidden units.

FIG0 the first 25000 entries of the sorted pairwise similarity matrices (Eq 10) obtained by applying GrOWL with different p (Eq 9) values.

By setting the weights beyond index p to a common constant implies that clustering is only encouraged among the p largest coefficients, i.e., relevant coefficient groups; however, FIG0 shows that, with properly chosen p, GrOWL yields more parameter tying than OSCAR (p = n).

On the other hand, smaller p values allow using large Λ 2 , encouraging parameter tying among relatively loose correlations.

In practice, we find that for p around the target fraction of nonzero parameters leads to good performance in general.

The intuition is that we only need to identify correlations among the selected important features.

FIG0 shows that weight decay (denoted as 2 ) also pushes parameters together, though the parameter-tying effect is not as clear as that of GrOWL.

As has been observed in the literature BID6 , weight decay often achieves better generalization than sparsity-inducing regularizers.

It achieves this via parameter shrinkage, especially in the highly correlated region, but it does not yield sparse models.

In the following section, we explore the compression performance of GrOWL by comparing it with both group-Lasso and weight decay.

We also explore how to further improve the accuracy vs compression trade-off by using sparsity-inducing regularization together with weight decay ( 2 ).

For each case, the baseline performance is provided as the best performance obtained by running the original neural network (without compression) after sweeping the hyper-parameter on the weight decay regularizer over a range of values.

The MNIST dataset contains centered images of handwritten digits (0-9), of size 28×28 (784) pixels FIG2 shows the (784 × 784) correlation matrix of the dataset (the margins are zero due to the redundant background of the images).

We use a network with a single fully connected layer of 300 hidden units.

The network is trained for 300 epochs and then retrained for an additional 100 epochs, both with momentum.

The initial learning rate is set to 0.001, for both training and retraining, and is reduced by a factor of 0.96 every 10 epochs.

We set p = 0.5, and Λ 1 , Λ 2 are selected by grid search.

Pairwise similarities (see Eq. FORMULA15 ) between the rows of the weight matrices learned with different regularizers are shown in FIG2 .

As we can see, GrOWL (+ 2 ) identifies more correlations than group-Lasso (+ 2 ), and the similarity patterns in FIG2 are very close to that of the data FIG2 ).

On the other hand, weight decay also identifies correlations between parameter rows, but it does not induce sparsity.

Moreover, as shown in Table 1 FORMULA15 ) of the parameter rows obtained by training the neural network with GrOWL, GrOWL+ 2 , group-Lasso, group-Lasso+ 2 and weight decay ( 2 ).

Table 1 : Sparsity, parameter sharing, and compression rate results on MNIST.

Baseline model is trained with weight decay and we do not enforce parameter sharing for baseline model.

We train each model for 5 times and report the average values together with their standard deviations.

Sparsity Parameter Sharing Compression ratio Accuracy none 0.0 ± 0% 1.0 ± 0 1.0 ± 0 98.3 ± 0.1% weight decay 0.0 ± 0% 1.6 ± 0 1.6 ± 0 98.4 ± 0.0% group-Lasso 87.6 ± 0.1% 1.9 ± 0.1 15.8 ± 1.0 98.1 ± 0.1% group-Lasso+ 2 93.2 ± 0.4%1.6 ± 0.1 23.7 ± 2.1 98.0 ± 0.1% GrOWL 80.4 ± 1.0% 3.2 ± 0.1 16.7 ± 1.3 98.1 ± 0.1% GrOWL+ 2 83.6 ± 0.5% 3.9 ± 0.1 24.1 ± 0.8 98.1 ± 0.1%The compression vs accuracy trade-off of the different regularizers is summarized in Table 1 , where we see that applying 2 regularization together with group-Lasso or GrOWL leads to a higher compression ratio, with negligible effect on the accuracy.

Table 1 also shows that, even with lower sparsity after the initial training phase, GrOWL (+ 2 ) compresses the network more than group-Lasso (+ 2 ), due to the significant amount of correlation it identifies; this also implies that group-Lasso only selects a subset of the correlated features, while GrOWL selects all of them.

On the other hand, group-Lasso suffers from randomly selecting a subset of correlated features; this effect is illustrated in FIG4 , which plots the indices of nonzero rows, showing that GrOWL (+ 2 ) stably selects relevant features while group-Lasso (+ 2 ) does not.

The mean ratios of changed indices 2 are 11.09%, 0.59%, 32.07%, and 0.62% for group-Lasso, GrOWL, group-Lasso+ 2 , and GrOWL+ 2 , respectively.

To evaluate the proposed method on large DNNs, we consider a VGG-like BID33 architecture proposed by BID38 on the CIFAR-10 dataset.

The network architecture is summarized in Appendix C; comparing with the original VGG of BID33 , their fully connected layers are replaced with two much smaller ones.

A batch normalization layer is added after each convolutional layer and the first fully connected layer.

Unlike BID38 , we don't use dropout.

We first train the network under different regularizers for 150 epochs, then retrain it for another 50 epochs, using the learning rate decay scheme described by He et al. 2 The mean ratio of changed indices is defined as: Table 2 : Sparsity (S1) and Parameter Sharing (S2) of VGG-16 on CIFAR-10.

Layers marked by * are regularized.

We report the averaged results over 5 runs.

Weight Decay group-Lasso group-Lasso + 2 GrOWL GrOWL + 2 (S1, S2) (S1, S2) (S1, S2) (S1, S2) (S1, S2) conv1 0%, 1. (2016): the initial rates for the training and retraining phases are set to 0.01 and 0.001, respectively; the learning rate is multiplied by 0.1 every 60 epochs of the training phase, and every 20 epochs of the retraining phase.

For GrOWL (+ 2 ), we set p = 0.1 n (see Eq. FORMULA14 ) for all layers, where n denotes the number of rows of the (reshaped) weight matrices of each layer.

The results are summarized in Table 2 .

For all of the regularizers, we use the affinity propagation algorithm (with preference value 3 set to 0.8) to cluster the rows at the end of initial training process.

Our experiments showed that it is hard to encourage parameter tying in the first 7 convolutional layers; this may be because the filters of these first 7 convolutional layers have comparatively large feature maps (from 32 × 32 to 8 × 8), which are only loosely correlated.

We illustrate this reasoning in Fig. 6 , showing the cosine similarity between the vectorized output channels of layers 1, 6, 10, and 11, at the end of the training phase; it can be seen that the outputs of layers 10 and 11 have many more significant similarities than that of layer 6.

Although the output channels of layer 1 also Figure 6 : Output channel cosine similarity histogram obtained with different regularizers.

Labels: GO:GrOWL, GOL:GrOWL+ 2 , GL:group-Lasso, GLL:group-Lasso+ 2 , WD:weight decay.have certain similarities, as seen in Table 2 , neither GrOWL (+ 2 ) nor weight decay tends to tie the associated weights.

This may mean that the network is maintaining the diversity of the inputs in the first few convolutional layers.

Although GrOWL and weight decay both encourage parameter tying in layers 9-13, weight decay does it with less intensity and does not yield a sparse model, thus it cannot significantly compress the network.

propose to prune small weights after the initial training phase with weight decay, then retrain the reduced network; however, this type of method only achieves compression 4 ratios around 3.

As mentioned by , layers 3-7 can be very sensitive to pruning; however, both GrOWL (+ 2 ) and group-Lasso (+ 2 ) effectively compress them, with minor accuracy loss.

On the other hand, similar to what we observed by running the simple fully-connected network on MNIST, the accuracy-memory trade-off improves significantly by applying GrOWL or group-Lasso together with 2 .

However, Table 2 also shows that the trade-off achieved by GrOWL (+ 2 ) and group-Lasso (+ 2 ) are almost the same.

We suspect that this is caused by the fact that CIFAR-10 is simple enough that one could still expect a good performance after strong network compression.

We believe this gap in the compression vs accuracy trade-off can be further increased in larger networks on more complex datasets.

We leave this question for future research.

We have proposed using the recent GrOWL regularizer for simultaneous parameter sparsity and tying in DNN learning.

By leveraging on GrOWL's capability of simultaneously pruning redundant parameters and tying parameters associated with highly correlated features, we achieve significant reduction of model complexity, with a slight or even no loss in generalization accuracy.

We evaluate the proposed method on both a fully connected neural network and a deep convolutional neural network.

The results show that GrOWL can compress large DNNs by factors ranging from 11.4 to 14.5, with negligible loss on accuracy.

The correlation patterns identified by GrOWL are close to those of the input features to each layer.

This may be important to reveal the structure of the features, contributing to the interpretability of deep learning models.

On the other hand, by automatically tying together the parameters corresponding to highly correlated features, GrOWL alleviates the negative effect of strong correlations that might be induced by the noisy input or the co-adaption tendency of DNNs.

The gap in the accuracy vs memory trade-off obtained by applying GrOWL and group-Lasso decreases as we move to large DNNs.

Although we suspect this can be caused by running a much larger network on a simple dataset, it motivates us to explore different ways to apply GrOWL to compress neural networks.

One possible approach is to apply GrOWL within each neuron by predefining each 2D convolutional filter as a group (instead all 2D convolutional filters corresponding to the same input features).

By doing so, we encourage parameter sharing among much smaller units, which in turn would further improve the diversity vs parameter sharing trade-off.

We leave this for future work.

Various methods have been proposed to compute the proximal mapping of OWL (ProxOWL) .

It has been proven that the computation complexity of these methods is O(n log n) which is just slightly worse than the soft thresholding method for solving 1 norm regularization.

In this paper, we use Algorithm 2 that was originally proposed in BID5 .Unlike k-means or agglomerative algorithm, Affinity Propagation does not require the number of clusters as an input.

We deem this as a desired property for enforcing parameter sharing in neural network compression because it's impossible to have the exact number of clusters as a prior information.

In practice, the input preference of Affinity Propagation determines how likely each sample will be chosen as an exemplar and its value will influence the number of clusters created.

APPENDIX C VGG-16 ON CIFAR-10 Table 4 : VGG: Clustering rows over different preference values for running the affinity propagation algorithm (Algorithm 3).

For each experiment, we report clustering accuracy (A), compression rate (C), and parameter sharing (S) of layers 9-14.

For each regularizer, we use different preference values to run Algorithm 3 to cluster the rows at the end of initial training process.

Then we retrain the neural network correspondingly.

The results are reported as the averages over 5 training and retraining runs.

Preference Value 0.6 0.7 0.8 0.9 (A, C, S) (A, C, S) (A, C, S) (A, C, S) GrOWL 92.2%, 13.6, 3.5 92.2%, 12.5, 2.6 92.2%, 11.4, 2.1 92.2%, 10.9, 1.

<|TLDR|>

@highlight

We have proposed using the recent GrOWL regularizer for simultaneous parameter sparsity and tying in DNN learning. 
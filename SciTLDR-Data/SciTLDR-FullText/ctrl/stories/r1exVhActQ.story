State-of-the-art deep neural networks (DNNs) typically have tens of millions of parameters, which might not fit into the upper levels of the memory hierarchy, thus increasing the inference time and energy consumption significantly, and prohibiting their use on edge devices such as mobile phones.

The compression of DNN models has therefore become an active area of research recently, with \emph{connection pruning} emerging as one of the most successful strategies.

A very natural approach is to prune connections of DNNs via $\ell_1$ regularization, but recent empirical investigations have suggested that this does not work as well in the context of DNN compression.

In this work, we revisit this simple strategy and analyze it rigorously, to show that: (a) any \emph{stationary point} of an $\ell_1$-regularized layerwise-pruning objective has its number of non-zero elements bounded by the number of penalized prediction logits, regardless of the strength of the regularization; (b) successful pruning highly relies on an accurate optimization solver, and there is a trade-off between compression speed and distortion of prediction accuracy, controlled by the strength of regularization.

Our theoretical results thus suggest that $\ell_1$ pruning could be successful provided we use an accurate optimization solver.

We corroborate this in our experiments, where we show that simple $\ell_1$ regularization with an Adamax-L1(cumulative) solver gives pruning ratio competitive to the state-of-the-art.

State-of-the-art Deep Neural Networks (DNNs) typically have millions of parameters.

For example, the VGG-16 network BID0 ), from the winning team of ILSVRC-2014, contains more than one hundred million parameters; inference with this network on a single image takes tens of billions of operations, prohibiting its use on edge devices such as mobile phones or in real-time applications.

In addition, the huge size of DNNs often precludes them from being placed at the upper level of the memory hierarchy, with resulting slow access times and expensive energy consumption.

A recent thread of research has thus focused on the question of how to compress DNNs.

One successful approach that has emerged is to trim the connections between neurons, which reduces the number of non-zero parameters and thus the model size BID1 b) ; BID3 ; BID4 ; BID5 ; BID6 ; BID7 ).

However, there has been a gap between the theory and practice: the trimming algorithms that have been practically successful BID1 b) ; BID3 ) do not have theoretical guarantees, while theoretically-motivated approaches have been less competitive compared to the heuristics-based approaches BID5 , and often relies on stringent distributional assumption such as Gaussian-distributed matrices which might not hold in practice.

With a better theoretical understanding, we might be able to answer how much pruning one can achieve via different approaches on different tasks, and moreover when a given pruning approach might or might not work.

Indeed, as we discuss in our experiments, even the generally practically successful approaches are subject to certain failure cases.

Beyond simple connection pruning, there have been other works on structured pruning that prune a whole filter, whole row, or whole column at a time BID8 ; BID9 ; BID10 ; ; BID12 ).

The structured pruning strategy can often speed up inference speed at prediction time more than simple connection pruning, but the pruning ratios are typically not as high as non-structured connection pruning; so that the storage complexity is still too high, so that the caveats we noted earlier largely remain.

A very natural strategy is to use 1 regularized training to prune DNNs, due to their considerable practical success in general sparse estimation in shallow model settings.

However, many recent investigations seemed to suggest that such 1 regularization does not work as well with non-shallow DNNs, especially compared to other proposed methods.

Does 1 regularization not work as well in non-shallow models?

In this work, we theoretically analyze this question, revisit the trimming of DNNs through 1 regularization.

Our analysis provides two interesting findings: (a) for any stationary point under 1 regularization, the number of non-zero parameters in each layer of a DNN is bounded by the number of penalized prediction logits-an upper bound typically several orders of magnitude smaller than the total number of DNN parameters, and (b) it is critical to employ an 1 -friendly optimization solver with a high precision in order to find the stationary point of sparse support.

Our theoretical findings thus suggest that one could achieve high pruning ratios even via 1 regularization provided one uses high-precision solvers (which we emphasize are typically not required if we only care about prediction error rather than sparsity).

We corroborate these findings in our experiments, where we show that solving the 1 -regularized objective by the combination of SGD pretraining and Adamax-L1(cumulative) yields competitive pruning results compared to the state-ofthe-art.

Let DISPLAYFORM0 p × K 0 be an input tensor where N is the number of samples (or batch size).

We are interested in DNNs of the form DISPLAYFORM1 where σ W (j) (X (j−1) ) are piecewise-linear functions of both the parameter tensor W (j) : DISPLAYFORM2 Examples of such piecewise-linear function include:(a) convolution layers with Relu activation (using • to denote the p-dimensional convolution operator) DISPLAYFORM3 fully-connected layers with Relu activation DISPLAYFORM4 commonly used operations such as max-pooling, zero-padding and reshaping.

Note X (J) : N × K provide K scores (i.e. logits) of each sample that relate to the labels of our DISPLAYFORM5 as the task-specific loss function.

We define Support Labels of a DNN X (J) as indices (i, k) of non-zero loss subgradient w.r.t.

the prediction logit:Definition 1 (Support Labels).

Let L(X, Y ) be a convex loss function w.r.t.

the prediction logits X. The Support Labels regarding DNN outputs X (J) (W ) are defined as DISPLAYFORM6 We will denote k S (W ) := |S(W )| N ≤ K as the average number of support labels per sample.

We illustrate these concepts in the context of some standard machine learning tasks.

Multiple Regression.

In multiple regression, we are interested in multiple real-valued labels, such as the location and orientation of objects in an image, which over the set of N samples, can be expressed as an N × K real-valued matrix Y .

A popular loss function for such tasks is: which is convex and differentiable, and in general we have [∇ X L] i,k = 0, therefore all labels are support labels (i.e. k S = K).

DISPLAYFORM7 Binary Classification.

In binary classification, the labels are binary-valued, and over the set of N samples, can be represented as a binary vector y ∈ {−1, 1} N .

Popular loss functions include the logistic loss: DISPLAYFORM8 and the hinge loss: DISPLAYFORM9 For the logistic loss (1), we have DISPLAYFORM10 On the other hand, the hinge loss (2) typically has only a small portion of samples DISPLAYFORM11 Vectors, and which coincides with our definition of Support Labels in this context.

In applications with unbalanced positive and negative examples, such as object detection, we have k S 1.

Multiclass/Multilabel Classification.

In multiclass or multilabel classification, the labels of each sample can be represented as a K-dimensional binary vector {0, 1}K where 1/0 denotes the presence/absence of a class in the sample.

Let P i := {k | y ik = 1} and N i := {k | y ik = 0} denote the positive and negative label sets.

Popular loss functions include the cross-entropy loss: DISPLAYFORM12 and the maximum margin loss: DISPLAYFORM13 Although the cross-entropy loss (3) has number of support labels k S = K, it has been shown that the maximum-margin loss (4) typically has k S K in recent studies of classification problems with extremely large number of classes BID13 ).

In this section, we aim to solve the following DNN compression problem.

Definition 2 (Deep-Trim ( )).

Suppose we are given a target loss function L(X, Y ) between prediction X and training labels Y : N × K, and a pre-trained DNN X (J) parameterized by weights DISPLAYFORM0 The Deep-Trim ( ) task is to find a compressed DNN with weights W such that its number of non-zero parameters nnz( W ) ≤ τ , for some τ nnz(W ) and where DISPLAYFORM1 In the following, we show that the Deep-Trim problem with budget τ = (N k S ) × J can be solved via simple 1 regularization under a couple of mild conditions, with the caveat that with suitable optimization algorithms be used, and where k S is the maximum number of support labels for any DISPLAYFORM2 Trimming Objective Given a loss function L(., Y ) and a pre-trained DNN parameterized by DISPLAYFORM3 , we initialize the iterate with W * and apply an optimization algorithm that guarantees descent of the following layerwise 1 -regularized objective DISPLAYFORM4 for all j ∈ [J], where vec(W (j) ) denotes the vectorized version of the tensor W (j) .The following theorem states that most of stationary points of FORMULA18 have the number of non-zero parameters per layer bounded by the total number of support labels in the training set.

Theorem 1 (Deep-Trim with 1 penalty).

Let W (j) be any stationary point of objective (5) with dim(W (j) )=d that lies on a single linear piece of the piecewise-linear function X (J) (W ).

Let V : (N K) × d be the Jacobian matrix of that corresponding linear piece of the linear (vector-valued) function vec(X (J) )(vec(W (j) )).

For any regularization parameter λ > 0 and V in general position we have DISPLAYFORM5 where k S ( W ) is the average number of support labels of the stationary point W (j) .Proof.

Any stationary point of FORMULA18 should satisfy the condition DISPLAYFORM6 where A ∈ ∂L is an N × K subgradient matrix of the loss function w.r.t.

the prediction logits, and DISPLAYFORM7 be the set of indices of non-zero parameters, we have [ρ] r ∈ {−1, 1} and thus the linear system DISPLAYFORM8 cannot be satisfied if nnz(A) < nnz(W (j) ) for V is in general position (as defined in, for example, BID15 ).

Therefore, we have nnz DISPLAYFORM9 Note the concept of general position is studied widely in the literature of LASSO and sparse recovery, and it is a weak assumption in the sense that any matrix drawn from a continuous probability distribution is in general position BID15 ).

Figure 1 illustrates an example of a regression task where, no matter how small λ > 0 is, the second coordinate is always 0 at the stationary point.

Note since Theorem 1 holds for any λ > 0, one can guarantee to trim a DNN without hurting the training loss by choosing an appropriately small λ, as stated by the following corollary.

Corollary 1 (Deep-Trim without Distortion).

Given a DNN with weights W and with loss DISPLAYFORM10 where k S is a bound on the number of support labels of parameters W with loss no more than L * + .Proof.

By choosing λ ≤ /(J vec(W (j) ) 1 ), any descent optimization algorithm can guarantee to findŴ (j) with DISPLAYFORM11 Then by applying the procedure for each layer j ∈ [J], one can obtainŴ with DISPLAYFORM12 In practice, however, the smaller λ, the harder for the optimization algorithm to get close to the stationary point, as illustrated in the figure 1.

Therefore, it is crucial to choose optimization algorithms targeting for high precision for the convergence to the stationary point of (5) with sparse support, while the widely-used Stochastic Gradient Descent (SGD) method is notorious for being inaccurate in terms of the optimization precision.

Although our analysis is conducted on the layerwise pruning objective (5), in practice we have observed joint pruning of all layers to be as effective as layerwise pruning.

For ease of presentation of this section, we will denote our objective function min DISPLAYFORM0 in the following form min DISPLAYFORM1 where w := vec(W ) and f (w) := L(X (J) (W ), Y ).

Note the same formulation (9) can be also used to represent the layerwise pruning objective (5) by simply replacing their definitions as DISPLAYFORM2 As mentioned previously, even when the stationary point of an objective has sparse support, if the optimization algorithm does not converge close enough to the stationary point, the iterates would still have very dense support.

In this section, we propose a two-phase strategy for the non-convex optimization problem (9).

In the first phase, we initialize with the given model and use a simple Stochastic Gradient Descent (SGD) algorithm to optimize (9).

During this phase, we do not aim to reduce the number of non-zero parameters but only to reduce the 1 norm of the model parameters.

We run the SGD till both the training loss and 1 norm of model parameters have converged.

Then in the second phase, we employ an Adamax-L1 (cumulative) method to reduce the total number of non-zero parameters, and achieves pruning result on-par with state-of-the-art methods.

SGD with L1 Penalty For a simple optimization problem min w∈R d f (w), the SGD update follows the form w t+1 = w t − η t ∂f (w)∂w .

We consider general SGD-like algorithms which update in the form w t+1 = w t − η t g( DISPLAYFORM3 ∂w , θ), where θ is a set of parameters specific to the SGD-like update procedure.

This includes the commonly used Momentum (Qian (1999)), Adamax, Adam (Kingma and Ba FORMULA0 ), and RMSProp (Tieleman and Hinton FORMULA0 ) optimization algorithms.

When employing SGD-like optimizers, (9) can be rewritten as the following: DISPLAYFORM4 where j denotes one mini-batch of data and N b is the number of mini-batches.

The weight updated by the SGD-like optimizers can then be performed as DISPLAYFORM5 where sign(w i ) = 0 when w i = 0.

We note that after the update in (11), the weight does not become 0 unless w DISPLAYFORM6 , which rarely happens.

Therefore, adding the L1 penalty term to SGD-like optimizers only minimizes the L1-norm but does not induce a sparse weight matrix.

To achieve a sparse solution, we combine the L1 friendly update trick SGD-L1 (cumulative) BID19 ) along with SGD-like optimization algorithms.

Adamax-L1 (cumulative) SGD-L1 (clipping) is an alternative to perform L1 regularizing along with SGD to obtain a sparse w BID20 ).

Different to (11), SGD-L1 (clipping) divides the update into two steps.

The first step is updated without considering the L1 penalty term, and the second step updates the L1 penalty separately.

In the second step, any weight that has changed its sign during the update will be set to 0.

In other words, when the L1 penalty is larger than the weight value, it will be truncated to the weight value.

Therefore, SGD-L1 (clipping) can be seen as a special case of truncated gradient.

With a learning rate η k , the update algorithm can be written as DISPLAYFORM7 SGD-L1 (cumulative) is a modification of the SGD-L1 (clipping) algorithm proposed by BID19 , but uses the cumulative L1 penalty instead of the standard L1 penalty.

The intuition is that the cumulative L1 penalty is the amount of penalty that would be applied on the weight if true gradient is applied instead of stochastic gradient.

By applying the cumulative L1 penalty, the weight would not be moved away from zero by the noise of the stochastic gradient.

When applied to SGD-like optimization algorithms, the update rule can be written as DISPLAYFORM8 where q k i is the total amount of L1 penalty received until now q DISPLAYFORM9 By updating with (13) and adopting the Adamax optimization algorithm (Kingma and Ba FORMULA0 , we obtain Adamax-L1 (cumulative).

Originally, SGD-L1 (cumulative) was proposed to be used with the vanilla SGD optimizer, where we generalize it to be used with any SGD-like optimizer by separating the update on objective f (w) and the l1-cumulative update on λ w 1 .

In this section, we compare the -regularized pruning method discussed in section 4 with other state-of-the-art approaches.

In section 5.1, we evaluate different pruning methods on the convolution network LeNet-5 1 on the Mnist data set.

In section 5.2, we compare our method to VD on pruning VGG-16 network BID0 on the CIFAR-10 data set.

In section 5.3, we then conduct experiments with Resnet on CIFAR-10.

Finally, we show the trade-off for pruning Resnet-50 on the ILSVRC dataset.

Acc.% nnz per Layer% Table 1 : Compression Results with LeNet-5 model on MNIST.

We first compare our methods with other compression methods on the standard MNIST dataset with the LeNet-5 architecture.

We consider the following methods: Prune: The pruning algorithm proposed in BID1 , which iterates between pruning the network after training with L2 regularization and retraining.

DNS: Dynamic Network Surgery pruning algorithm proposed in BID3 , which was reported to improve upon the iterative pruning method proposed in BID1 by dynamically pruning and splicing variables during the training process.

VD: Variational Dropout method introduced by BID4 , a variant of dropout that induces sparsity during the training process with unbounded dropout rates.

L1 Naive: Ablation study of our method by training the 1 -regularized objective with SGD.

Ours: Our method which optimizes the 1 -regularized objective (8) in two phases (SGD and Adamax-L1(cumulative)).

The LeNet-5 network is trained from a random initialization and without data augmentation which achieves 99.2% accuracy.

We report the per layer sparsity and the total reduction of weights and Table 2 : Compression Results with VGG-like model on CIFAR-10 for VD and our method.

FLOP in Table 1 .

For LeNet-5, our method achieves comparable sparsity performance against other methods, with a slight accuracy drop.

Nevertheless, our compressed model still achieves over 99 percent testing accuracy, while achieving 260× weight reduction and 84× FLOP reduction.

We also observe that the L1 Naive does not induce any sparsity, even when the L1-norm is significantly reduced.

This demonstrates the effectiveness of adopting a L1-friendly optimization algorithm.

To test how our method works on large scale modern architecture, we perform experiments on the VGG-like network with CIFAR-10 dataset, which is used in BID4 .

The network contains 13 convolution layers and 2 fully connected layers and achieves 92.9% accuracy with pretraining.

We report the per layer weights and FLOP reduction for our Deep-Trim algorithm and VD BID4 ) in Table 5 .

Our model achieves a weight pruning ratio of 57× and reduces FLOP by 7.7× with a negligible accuracy drop, and VD achieves 48× weight pruning ratio and reduces FLOP by 6.4×.

2 Compared to VD, our model achieved sparser weights from Conv1_1 to Conv5_2 and VD achieved sparser weights from Conv5_2 to FC layers.

Interestingly, we observe that in both pruning methods, most remaining nnz and FLOPs lie in block2 and block3, where originally block4 and block5 have dominating amount of weights and equal amount of FLOPs.

The layer with the most non-zero parameters after pruning is conv3_2 with 65.9K.

In the experiments we employ the cross-entropy loss (3) which has a number of support labels N K = 500K on the CIFAR-10 data set.

We suspect a more careful analysis could improve our Theorem 1 to give a tighter bound for loss with entries of gradient close to 0 but not exactly 0, making the bound for cross-entropy loss (3) closer to that of maximum-margin loss (4).

2 We ran the experiments based on authors' code and tuned the coefficient of dropout regularization loss within the interval [10 2 , 10 −3 ] with binary search.

We note that although we are able to reproduce the 48× weight reduction ratio in the VD paper, we are only able to achieve Acc.

92.2% instead of 92.7% as reported in their paper.

While VGG-network are notorious for its large parameter size, it is not surprising that a large compression rate can be achieved.

Therefore, we evaluate the compression performance of our Deep-Trim algorithm on a smaller Resnet-32 model trained on CIFAR-10 data.

The Resnet-32 model contains 3 main blocks.

The first block contains the first 11 convolution layers with 64 filters in each layer, the second block contains the next 10 convolution layers with 128 filters each, and the last block contains 10 convolution layers with 256 filters and a fully connected layer.

We list the detailed architecture in the supplementary.

The pretrained Resnet-32 model reaches 94.0% accuracy.

We evaluate our Deep-Trim algorithm and compare it to variational dropout BID4 ) and report the results in TAB2 .

We report the pruning results for each main block of the resnet-32 model.

Our model achieves a 33× overall pruning ratio and 21× reduced FLOP with an accuracy drop of 1.4%, where VD has attained 28× overall pruning ratio and 13.5× reduction with similar accuracy.

We further observe that nnz(W) increases much gentler from the first block to the third block compared to the total number of parameters in each block.

This is not surprising since the upper bound of nnz(W) per layer given by Corollary 1 does not depend on the total number of unpruned parameters.

Acc

In this section, we compare the pruning results of our method on VGG-16 with different number of samples.

The pruning ratio and number of non-zero parameters are shown in TAB4 , we can see that the number of non-zero parameters after pruning clearly grows with the number of samples.

This can be understood intuitively, as the number of constraints to be satisfied grows in the training set, the more degree of freedom the model needs to fit the data.

This shows that our theory analysis matches our empirical results well.

In this work, we revisit the simple idea of pruning connections of DNNs through 1 regularization.

While recent empirical investigations suggested that this might not necessarily achieve high sparsity levels in the context of DNNs, we provide a rigorous theoretical analysis that does provide small upper bounds on the number of non-zero elements, but with the caveat that one needs to use a high-precision optimization solver (which is typically not needed if we care only about prediction error rather than sparsity).

When using such an accurate optimization solver, we can converge closer to stationary points than traditional SGD, and achieve much better pruning ratios than SGD, which might explain the poorer performance of 1 regularization in recent investigations.

We perform experiments across different datasets and networks and demonstrate state-of-the-art result with such simple 1 regularization.

Table 5 : Per-layer Resnet-32 architecture.

There are 3 main convolutional blocks with downsampling through stride=2 for the first layer of each block.

After the convloutional layers, global pooling is applied on the spatial axes and a fully-connected layer is appended for the output.

Each set of rows is a residual block.

<|TLDR|>

@highlight

We revisit the simple idea of pruning connections of DNNs through $\ell_1$ regularization achieving state-of-the-art results on multiple datasets with theoretic guarantees.
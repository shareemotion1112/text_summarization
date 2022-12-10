Magnitude-based pruning is one of the simplest methods for pruning neural networks.

Despite its simplicity, magnitude-based pruning and its variants demonstrated remarkable performances for pruning modern architectures.

Based on the observation that the magnitude-based pruning indeed minimizes the Frobenius distortion of a linear operator corresponding to a single layer, we develop a simple pruning method, coined lookahead pruning, by extending the single layer optimization to a multi-layer optimization.

Our experimental results demonstrate that the proposed method consistently outperforms the magnitude pruning on various networks including VGG and ResNet, particularly in the high-sparsity regime.

The "magnitude-equals-saliency" approach has been long underlooked as an overly simplistic baseline among all imaginable techniques to eliminate unnecessary weights from over-parametrized neural networks.

Since the early works of LeCun et al. (1989) ; Hassibi & Stork (1993) which provided more theoretically grounded alternative of magnitude-based pruning (MP) based on second derivatives of loss function, a wide range of methods including Bayesian / information-theoretic approaches (Neal, 1996; Louizos et al., 2017; Molchanov et al., 2017; Dai et al., 2018) , pregularization (Wen et al., 2016; Liu et al., 2017; Louizos et al., 2018) , sharing redundant channels (Zhang et al., 2018; Ding et al., 2019) , and reinforcement learning approaches (Lin et al., 2017; Bellec et al., 2018; He et al., 2018) has been proposed as more sophisticated alternatives.

On the other hand, the capabilities of MP heuristics are gaining attention once more.

Combined with minimalistic techniques including iterative pruning (Han et al., 2015) and dynamic reestablishment of connections (Zhu & Gupta, 2017) , a recent large-scale study by Gale et al. (2019) claims that MP can achieve a state-of-the-art trade-off of sparsity and accuracy on ResNet-50.

The unreasonable effectiveness of magnitude scores often extends beyond the strict domain of network pruning; a recent experiment by Frankle & Carbin (2019) suggests an existence of an automatic subnetwork discovery mechanism underlying the standard gradient-based optimization procedures of deep, overparametrized neural networks by showing that the MP algorithm finds an efficient trainable subnetwork.

These observations constitute a call to revisit the "magnitude-equals-saliency" approach for a better understanding of deep neural network itself.

As an attempt to better understand the nature of MP methods, we study a generalization of magnitude scores under a functional approximation framework; by viewing MP as a relaxed minimization of distortion in layerwise operators introduced by zeroing out parameters, we consider a multi-layer extension of the distortion minimization problem.

Minimization of the newly suggested distortion measure which 'looks ahead' the impact of pruning on neighboring layers gives birth to a novel pruning strategy, coined lookahead pruning (LAP).

In this paper, we focus on comparison of the proposed LAP scheme to its MP counterpart.

We empirically demonstrate that LAP consistently outperforms the MP under various setups including linear networks, fully-connected networks, and deep convolutional and residual networks.

In particular, the LAP consistently enables more than ×2 gain in the compression rate of the considered models, with increasing benefits under the high-sparsity regime.

Apart from its performance, the lookahead pruning method enjoys additional attractive properties: • Easy-to-use: Like magnitude-based pruning, the proposed LAP is a simple score-based approach agnostic to model and data, which can be implemented by computationally light elementary tensor operations.

Unlike most Hessian-based methods, LAP does not rely on an availability of training data except for the retraining phase.

It also has no hyper-parameter to tune, in contrast to other sophisticated training-based and optimization-based schemes.

• Versatility: As our method simply replaces the "magnitude-as-saliency" criterion with a lookahead alternative, it can be deployed jointly with algorithmic tweaks developed for magnitudebased pruning, such as iterative pruning and retraining (Han et al., 2015) or joint pruning and training with dynamic reconnections (Zhu & Gupta, 2017; Gale et al., 2019) .

The remainder of this manuscript is structured as follows: In Section 2, we introduce a functional approximation perspective toward MP and motivate LAP and its variants as a generalization of MP for multiple layer setups; in Section 3 we explore the capabilities of LAP and its variants with simple models, then move on to apply LAP to larger-scale models.

We begin by a more formal description of the magnitude-based pruning (MP) algorithm (Han et al., 2015) .

Given an L-layer neural network associated with weight tensors W 1 , . . .

, W L , the MP algorithm removes connections with smallest absolute weights from each weight tensor, until the desired level of sparsity has been achieved.

This layerwise procedure is equivalent to finding a mask M whose entries are either 0 or 1, incurring a smallest Frobenius distortion, measured by min

where denotes the Hadamard product, · 0 denotes the entry-wise 0 -norm, and s is a sparsity constraint imposed by some operational criteria.

Aiming to minimize the Frobenius distortion (Eq. (1)), the MP algorithm naturally admits a functional approximation interpretation.

For the case of a fully-connected layer, the maximal difference between the output from a pruned and an unpruned layer can be bounded as

(2) Namely, the product of the layerwise Frobenius distortion upper bounds the output distortion of the network incurred by pruning weights.

Note that this perspective on MP as a worst-case distortion minimization was already made in Dong et al. (2017) , which inspired an advent of the layerwise optimal brain surgery (L-OBS) procedure.

A similar idea holds for convolutional layers.

For the case of a two-dimensional convolution with a single input and a single output channel, the corresponding linear operator takes a form of a doubly block circulant matrix constructed from the associated kernel tensor (see, e.g., Goodfellow et al. (2016) ).

Here, the Frobenius distortion of doubly block circulant matrices can be controlled by the Frobenius distortion of the weight tensor of the convolutional layer.

The case of multiple input/output channel or non-circular convolution can be dealt with similarly using channel-wise circulant matrices as a block.

We refer the interested readers to Sedghi et al. (2019

The myopic optimization (Eq. (1)) based on the per-layer Frobenius distortion falls short even in the simplest case of the two-layer linear neural network with one-dimensional output, where we consider predictors of a form Y = u W x and try to minimize the Frobenius distortion of u W (equivalent to 2 distortion in this case).

Here, if u i is extremely large, pruning any nonzero element in the i-th row of W may incur a significant Frobenius distortion.

Motivated by this observation, we consider a block approximation analogue of the magnitude-based pruning objective Eq. (1).

Consider an L-layer neural network with associated weight tensors W 1 , . . .

, W L , and assume linear activation for simplicity (will be extended to nonlinear cases later in this section).

Let J (W i ) denote the Jacobian matrix corresponding to the linear operator characterized by W i .

For pruning the i-th layer, we take into account the weight tensors of neighboring layers W i−1 , W i+1 in addition to the original weight tensor W i .

In particular, we propose to minimize the Frobenius distortion of the operator block

An explicit minimization of the block distortion (Eq. (3)), however, is computationally intractable in general (see Appendix D for a more detailed discussion).

To avoid an excessive computational overhead, we propose to use the following score-based pruning algorithm, coined lookahead pruning (LAP), for approximating Eq. (3): For each entry w of W i , we prune away the weights with the smallest value of lookahead distortion (in a single step), defined as

where W i | w=0 denotes the tensor whose entries are equal to the entries of W i except for having zeroed out w.

We let both W 0 and W L+1 to be tensors consisting of ones.

In other words, lookahead distortion (Eq. (4)) measures the distortion (in Frobenius norm) induced by pruning w while all other weights remain intact.

For three-layer blocks consisting only of fully-connected layers and convolutional layers, Eq. (4) reduces to the following compact formula: for an edge w connected to the j-th input neuron/channel and the k-th output neuron/channel of the i-th layer, where its formal derivation is presented in Appendix E.

where |w| denotes the weight of w, W [j, :] denotes the slice of W composed of weights connected to the j-th output neuron/channel, and W [:, k] denotes the same for the k-th input neuron/channel.

In LAP, we compute the lookahead distortion for all weights, and then remove weights with smallest distortions in a single step (as done in MP).

A formal description of LAP is presented in Algorithm 1.

We also note the running time of LAP is comparable with that of MP (see Appendix G).

LAP on linear networks.

To illustrate the benefit of lookahead, we evaluate the performance of MP and LAP on a linear fully-connected network with a single hidden layer of 1,000 nodes, trained with MNIST image classification dataset.

Fig. 2a and Fig. 2b depict the test accuracy of models pruned with each methods, before and after retraining steps.

As can be expected from the discrepancy between the minimization objectives (Eqs.

(1) and (3)), networks pruned with LAP outperform networks pruned with MP at every sparsity level, in terms of its performance before a retraining phase.

Remarkably, we observe that test accuracy of models pruned with LAP monotonically increases from 91.2% to 92.3% as the sparsity level increases, until the fraction of surviving weights reaches 1.28%.

At the same sparsity level, models pruned with MP achieves only 71.9% test accuracy.

We also observe that LAP leads MP at every sparsity level even after a retraining phase, with an increasing margin as we consider a higher level of sparsity.

Understanding LAP with nonlinear activations.

Most neural network models in practice deploy nonlinear activation functions, e.g., rectified linear units (ReLU).

Although the lookahead distortion was originally derived using linear activation functions, LAP can also be used for nonlinear networks, as the quantity L i (w) remains relevant to the original block approximation point of view.

This is especially true when the network is severely over-parametrized.

To see this, consider a case where one aims to prune a connection in the first layer of a two-layer fully-connected network with ReLU, i.e.,

where σ(x) = max{0, x} is applied entrywise.

Under the over-parametrized scenario, zeroing out a single weight may alter the activation pattern of connected neurons with only negligible probability, which allows one to decouple the probability of activation of each neuron from the act of pruning each connection.

This enables us to approximate the root mean square distortion of the network output introduced by pruning w of W 1 by √ p k L 1 (w), where k is the index of the output neuron that w is connected to, and p k denotes the probability of activation for the k-th neuron.

In this sense, LAP (Algorithm 1) can be understood as assuming i.i.d.

activations of neurons, due to a lack of an additional access to training data.

In other word, LAP admits a natural extension to the regime where we assume an additional access to training data during the pruning phase.

This variant, coined LAP-act, will be formally described in Appendix F, with experimental comparisons to another datadependent baseline of optimal brain damage (OBD) (LeCun et al., 1989) .

Another theoretical justification of using the lookahead distortion (Eq. (5)) for neural networks with nonlinear activation functions comes from recent discoveries regarding the implicit bias imposed by training via stochastic gradient descent (Du et al., 2018) .

See Appendix M for a detailed discussion.

As will be empiricically shown in Section 3.1, LAP is an effective pruning strategy for sigmoids and tanh activations, that are not piece-wise linear as ReLU.

Batch normalization (BN), introduced by Ioffe & Szegedy (2015) , aims to normalize the output of a layer per batch by scaling and shifting the outputs with trainable parameters.

Based on our functional approximation perspective, having batch normalization layers in a neural network is not an issue for MP, which relies on the magnitudes of weights; batch normalization only affects the distribution of the input for each layer, not the layer itself.

On the other hand, as the lookahead distortion (Eq. (3)) characterizes the distortion of the multi-layer block, one must take into account batch normalization when assessing the abstract importance of each connection.

The revision of lookahead pruning under the presence of batch normalization can be done fairly simply.

Note that such a normalization process can be expressed as

for some a, b ∈ R dim(x) .

Hence, we revise the lookahead pruning to prune the connections with a minimum value of

where a i [k] denotes the k-th index scaling factor for the BN layer placed at the output of the i-th fully-connected or convolutional layer (if BN layer does not exist, let a i [k] = 1).

This modification of LAP makes it an efficient pruning strategy, as will be empirically verified in Section 3.3.

As the LAP algorithm (Algorithm 1) takes into account current states of the neighboring layers, LAP admits several variants in terms of lookahead direction, order of pruning, and sequential pruning methods; these methods are extensively studied in Section 3.2 Along with "vanilla" LAP, we consider in total, six variants, which we now describe below:

Mono-directional LAPs.

To prune a layer, LAP considers both preceding and succeeding layers.

Looking forward, i.e., only considering the succeeding layer, can be viewed as an educated modification of the internal representation the present layer produces.

Looking backward, on the other hand, can be interpreted as only taking into account the expected structure of input coming into the present layer.

The corresponding variants, coined LFP and LBP, are tested.

Order of pruning.

Instead of using the unpruned tensors of preceding/succeeding layers, we also consider performing LAP on the basis of already-pruned layers.

This observation brings up a question of the order of pruning; an option is to prune in a forward direction, i.e., prune the preceding layer first and use the pruned weight to prune the succeeding, and the other is to prune backward.

Both methods are tested, which are referred to as LAP-forward and LAP-backward, respectively.

Sequential pruning.

We also consider a sequential version of LAP-forward/backward methods.

More specifically, if we aim to prune total p% of weights from each layer, we divide the pruning budget into five pruning steps and gradually prune (p/5)% of the weights per step in forward/backward direction.

Sequential variants will be marked with a suffix "-seq".

In this section, we compare the empirical performance of LAP with that of MP.

More specifically, we validate the applicability of LAP to nonlinear activation functions in Section 3.1.

In Section 3.2, we test LAP variants from Section 2.3.

In Section 3.3, we test LAP on VGG (Simonyan & Zisserman, 2015) , ResNet (He et al., 2016) , and Wide ResNet (WRN, Zagoruyko & Komodakis (2016) ).

Experiment setup.

We consider five neural network architectures: (1) The fully-connected network (FCN) under consideration is consist of four hidden layers, each with 500 neurons.

(2) The convolutional network (Conv-6) consists of six convolutional layers, followed by a fully-connected classifier with two hidden layers with 256 neurons each; this model is identical to that appearing in the work of Frankle & Carbin (2019) suggested as a scaled-down variant of VGG.

2 (3) VGG-19 is used, with an addition of batch normalization layers after each convolutional layers, and a reduced number of fully-connected layers from three to one.

3 (4) ResNets of depths {18, 50} are used.

(5) WRN of 16 convolutional layers and widening factor 8 (WRN-16-8) is used.

All networks used ReLU activation function, except for the experiments in Section 3.1.

We mainly consider image classification tasks.

In particular, FCN is trained on MNIST dataset (Lecun et al., 1998) , Conv-6, VGG, and ResNet are trained on CIFAR-10 dataset (Krizhevsky & Hinton, 2009) , and VGG, ResNet, and WRN are trained on Tiny-ImageNet.

4 We focus on the one-shot pruning of MP and LAP, i.e., models are trained with a single training-pruning-retraining cycle.

All results in this section are averaged over five independent trials.

We provide more details on setups in Appendix A.

We first compare the performance of LAP with that of MP on FCN using three different types of activation functions: sigmoid, and tanh, and ReLU.

Figs. 3a to 3c depict the performance of models pruned with LAP (Green) and MP (Red) under various levels of sparsity.

Although LAP was motivated primarily from linear networks and partially justified for positivehomogenous activation functions such as ReLU, the experimental results show that LAP consistently outperforms MP even on networks using sigmoidal activation functions.

We remark that LAP outperforms MP by a larger margin as fewer weights survive (less than 1%).

Such a pattern will be observed repeatedly in the remaining experiments of this paper.

In addition, we also check whether LAP still exhibits better test accuracy before retraining under the usage of nonlinear activation functions, as in the linear network case (Fig. 2b) .

Fig. 3d illustrates the test accuracy of pruned FCN using ReLU on MNIST dataset before retraining.

We observe that the network pruned by LAP continues to perform better than MP in this case; the network pruned by LAP retains the original test accuracy until only 38% of the weights survive, and shows less than 1% performance drop with only 20% of the weights remaining.

On the other hand, MP requires 54% and 30% to achieve the same level of performance, respectively.

In other words, the models pruned with MP requires about 50% more survived parameters than the models pruned with LAP to achieve a similar level of performance before being retrained using additional training batches.

Now we evaluate LAP and its variants introduced in Section 2.3 on FCN and Conv-6, each trained on MNIST and CIFAR-10, respectively.

Table 1 summarizes the experimental results on FCN and  Table 2 summarizes the results on Conv-6.

In addition to the baseline comparison with MP, we also compare with random pruning (RP), where the connection to be pruned was decided completely independently.

We observe that LAP performs consistently better than MP and RP with similar or smaller variance in any case.

In the case of an extreme sparsity, LAP enjoys a significant performance gain; over 75% gain on FCN and 14% on Conv-6.

This performance gain comes from a better training accuracy, instead of a better generalization; see Appendix L for more information.

Comparing mono-directional lookahead variants, we observe that LFP performs better than LBP in the low-sparsity regime, while LBP performs better in the high-sparsity regime; in any case, LAP performed better than both methods.

Intriguingly, the same pattern appeared in the case of the ordered pruning.

Here, LAP-forward can be considered an analogue of LBP in the sense that they both consider layers closer to the input to be more important.

Likewise, LAP-backward can be considered an analogue of LFP.

We observe that LAP-forward performs better than LAP-backward in the high-sparsity regime, and vice versa in the low-sparsity regime.

Our interpretation is as follows:

Whenever the sparsity level is low, the importance of a carefully curating the input signal is not significant due to high redundancies in natural image signal.

This causes a relatively low margin of increment by looking backward in comparison to looking forward.

When the sparsity level is high, the input signal is scarce, and the relative importance of preserving the input signal is higher.

Finally, we observe that employing forward/backward ordering and sequential methods leads to a better performance, especially in the high-sparsity regime.

There is no clear benefit of adopting directional methods in the low-sparsity regime.

The relative gain in performance with respect to LAP is either marginal, or unreliable. (Tables 3 and 4) , and VGG-19, ResNet-50, and WRN-16-8 on TinyImageNet (Tables 5 to 7 ).

For models trained on CIFAR-10, we also test LAP-forward to verify the observation that it outperforms LAP in the high-sparsity regime on such deeper models.

We also report additional experimental results on VGG-{11, 16} trained on CIFAR-10 in Appendix B. For models trained on Tiny-ImageNet, top-1 error rates are reported in Appendix C.

From Tables 3 to 7 , we make the following two observations: First, as in Section 3.2, the models pruned with LAP consistently achieve a higher or similar level of accuracy compared to models pruned with MP, at all sparsity levels.

In particular, test accuracies tend to decay at a much slower rate with LAP.

In Table 3 , for instance, we observe that the models pruned by LAP retain test accuracies of 70∼80% even with less than 2% of weights remaining.

In contrast, the performance of models pruned with MP falls drastically, to below 30% accuracy.

This observation is consistent on both CIFAR-10 and Tiny-ImageNet datasets.

Second, the advantages of considering an ordered pruning method (LAP-forward) over LAP is limited.

While we observe from Table 3 that LAP-forward outperforms both MP and LAP in the highsparsity regime, the gain is marginal considering standard deviations.

LAP-forward is consistently worse than LAP (by at most 1% in absolute scale) in the low-sparsity regime.

In this work, we interpret magnitude-based pruning as a solution to the minimization of the Frobenius distortion of a single layer operation incurred by pruning.

Based on this framework, we consider the minimization of the Frobenius distortion of multi-layer operation, and propose a novel lookahead pruning (LAP) scheme as a computationally efficient algorithm to solve the optimization.

Although LAP was motivated from linear networks, it extends to nonlinear networks which indeed minimizes the root mean square lookahead distortion assuming i. τ fraction in all fully-connected layers, except for the last layer where we use (1 + q)/2 instead.

For FCN, we use (p, q) = (0, 0.5).

For Conv-6, VGGs ResNets, and WRN, we use (0.85, 0.8).

For ResNet-{18, 50}, we do not prune the first convolutional layer.

The range of sparsity for reported figures in all tables is decided as follows: we start from τ where test error rate starts falling below that of an unpruned model and report the results at τ, τ + 1, τ + 2, . . .

for FCN and Conv-6, τ, τ + 2, τ + 4, . . .

for VGGs, ResNet-50, and WRN, and τ, τ + 3, τ + 6, . . .

for ResNet-18.

In this section, we show that the optimization in Eq. (3) is NP-hard by showing the reduction from the following binary quadratic programming which is NP-hard (Murty & Kabadi, 1987) :

for some symmetric matrix A ∈ R n×n .

Without loss of generality, we assume that the minimum eigenvalue of A (denoted with λ) is negative; if not, Eq. (9) admits a trivial solution x = (0, . . .

, 0).

Assuming λ < 0, Eq. (9) can be reformulated as:

where H = A − λI. Here, one can easily observe that the above optimization can be solved by solving the below optimization for s = 1, . . .

, n min x∈{0,1} n : i xi=s

Finally, we introduce the below equality

where 1 denotes a vector of ones, U is a matrix consisting of the eigenvectors of H as its column vectors, and Λ is a diagonal matrix with corresponding (positive) eigenvalues of H as its diagonal elements.

The above equality shows that Eq. (11) is a special case of Eq. (3) by choosing W 1 = √ ΛU , W 2 = 1, W 3 = 1 and M = 1 − x. This completes the reduction from Eq. (9) to Eq. (3).

In this section, we provide a derivation of Eq. (5) for the fully-connected layers.

The convolutional layers can be handled similarly by substituting the multiplications in Eqs. (16) and (17) by the convolutions.

The Jacobian matrix of the linear operator correponding to a fully-connected layer is the weight matrix itself, i.e. J (W i ) = W i .

From this, lookahead distortion can be reformulated as

Now, we decompose the matrix product W i+1 W i W i−1 in terms of entries of W i as below:

where

, and j-th row of W i−1 , respectively.

The contribution of a single entry w :

].

Therefore, in terms of the Frobenius distortion, we conclude that

which completes the derivation of Eq. (5) for fully-connected layers.

F LAP-ACT: IMPROVING LAP USING TRAINING DATA Recall two observations made from the example of two-layer fully connected network with ReLU activation appearing in Section 2.1: LAP is designed to reflect the lack of knowledge about the training data at the pruning phase; once the activation probability of each neuron can be estimated, it is possible to refine LAP to account for this information.

In this section, we continue our discussion on the second observation.

In particular, we study an extension of LAP called lookahead pruning with activation (LAP-act) which prunes the weight with smallest value of

Here, W i is a scaled version of W i and w is the corresponding scaled value of w, defined by

where I ij denotes the set of output indices in the j-th output neuron/channel of i-th layer (for fully connected layers, this is a singleton).

Also, p k denotes the neuron's probability of activation, which can be estimated by passing the training data.

We derive LAP-act (Eq. (18)) in Appendix F.1 and perform preliminary empirical validations in Appendix F.2 with using optimal brain damage (OBD) as a baseline.

We also evaluate a variant of LAP using Hessian scores of OBD instead of magnitude scores.

It turns out that in the small networks (FCN, Conv-6), LAP-act outperforms OBD.

Consider a case where one aims to prune a connection of a network with ReLU, i.e.,

where σ(x) = max{0, x} is applied entrywise.

Under the over-parametrized scenario, zeroing out a single weight may alter the activation pattern of connected neurons with only negligible probability, which allows one to decouple the probability of activation of each neuron from the act of pruning each connection.

From this observation, we first construct the below random distortion, following the philosophy of the linear lookahead distortion Eq. (4)

where J (W i ) denotes a random matrix where

]

and g i [k] is a 0-1 random variable corresponding to the activation, i.e., g i [k] = 1 if and only if the k-th output of the i-th layer is activated.

However, directly computing the expected distortion with respect to the real activation distribution might be computationally expensive.

To resolve this issue, we approximate the root mean-squared lookahead distortion by applying the mean-field approximation to the activation probability of neurons, i.e., all activations are assumed to be independent, as

denotes the mean-field approximation of p(g).

Indeed, the lookahead distortion with ReLU nonlinearity (Eq. (22)) or three-layer blocks consisting only of the fully-connected layers and the convolutional layers can be easily computed by using the rescaled weight matrix W i :

where I i,j denotes a set of output indices in the j-th output neuron/channel of the i-th layer.

Finally, for an edge w connected to the j-th input neuron/channel and the k-th output neuron/channel of the i-th layer, Eq. (22) reduces to

where w denotes the rescaled value of w. This completes the derivation of Eq. (18).

We compare the performance of three algorithms utilizing training data at the pruning phase: optimal brain damage (OBD) which approximates the loss via second order Taylor seris approximation with the Hessian diagonal (LeCun et al., 1989) , LAP using OBD instead of weight magnitudes (OBD+LAP), and LAP-act as described in this section.

We compare the performances of three algorithms under the same experimental setup as in Section 3.2.

To compute the Hessian diagonal for OBD and OBD+LAP, we use a recently introduced software package called "BackPACK," (Anonymous, 2020), which is the only open-source package supporting an efficient of Hessians, up to our knowledge.

Note that the algorithms evaluated in this section are also evaluated for global pruning experiments in Appendix I.

The experimental results for FCN and Conv-6 are presented in Tables 13 and 14 .

Comparing to algorithms relying solely on the model parameters for pruning (MP/LAP in Tables 1 and 2 ), we observe that OBD performs better in general, especially in the high sparsity regime.

This observation is coherent to the findings of LeCun et al. (1989) .

Intriguingly, however, we observe that applying lookahead critertion to OBD (OBD+LAP) significantly enhances to OBD significantly enhances the performance in the high sparsity regime.

We hypothesize that LAP helps capturing a correlation among scores (magnitude or Hessian-based) of adjacent layers.

Also, we observe that LAP-act consistently exhibits a better performance compared to OBD.

This result is somewhat surprising, in the sense that LAP-act only utilizes (easier-to-estimate) information about activation probabilities of each neuron to correct lookahead distortion.

The average running time of OBD, OBD+LAP, and LAP-act is summarized in Table 15 .

We use Xeon E5-2630v4 2.20GHz for pruning edges, and additionally used a single NVidia GeForce GTX-1080 for the computation of Hessian diagonals (used for OBD, OBD+LAP) and activation probabiility (for LAP-act).

We observe that LAP-act runs in a significantly less running time than OBD/OBD+LAP, and the gap widens as the number of parameters and the dimensionality of the dataset increases (from MNIST to CIFAR-10).

MP comprises of three steps: (1) computing the absolute value of the tensor, (2) sorting the absolute values, and (3) selecting the cut-off threshold and zero-ing out the weights under the threshold.

Steps (2) and (3) remain the same in LAP, and typically takes O(n log n) steps (n denotes the number of parameters in a layer).

On the other hand,

Step (1) is replaced by computing the lookahead distortion

for each parameter w. Fortunately, this need not be computed separately for each parameter.

Indeed, one can perform tensor operations to compute the squared lookahead distortion, which has the same ordering with lookahead distortion.

For fully-connected layers with 2-dimensional Jacobians, the squared lookahead distortion for

where 1 i denotes all-one matrix of size d i−2 × d i ; multiplying 1 i denotes summing operation along an axis and duplicating summed results into the axis, and 2 denotes the element-wise square operation.

The case of convolutional layers can be handled similarly.

We note that an implementation of Eq. (25) Table 16 , where we fixed the layerwise pruning rate to be uniformly 90%.

The codes are implemented with PyTorch, and the computations have taken place on 40 CPUs of Intel Xeon E5-2630v4 @ 2.20GHz.

All figures are averaged over 100 trials.

We make two observations from Table 16 .

First, the time required for LAP did not exceed 150% of the time required for MP, confirming our claim on the computational benefits of LAP.

Second, most of the added computation comes from considering the factors from batch normalization, without which the added computation load is ≈5%.

In the main text, LAP is compared to the MP in the context of unstructured pruning, where we do not impose any structural constraints on the set of connections to be pruned together.

On the other hand, the magnitude-based pruning methods are also being used popularly as a baseline for channel pruning (Ye et al., 2018) , which falls under the category of structured pruning.

MP in channel pruning is typically done by removing channels with smallest aggregated weight magnitudes; this aggregation can be done by either taking 1 -norm or 2 -norm of magnitudes.

Similarly, we can consider channel pruning scheme based on an 1 or 2 aggregation of LAP distortions, which we will call LAP-1 and LAP-2 (as opposed to MP-1 and MP-2 ).

We compare the performances of LAP-based channel pruning methods to MP-based channel pruning methods, along with another baseline of random channel pruning (denoted with RP).

We test with Conv-6 (Table 17) and VGG-19 (Table 18 ) networks on CIFAR-10 dataset.

All reported figures are averaged over five trials, experimental settings are identical to the unstructure pruning experiments unless noted otherwise.

Similar to the case of unstructured pruning, we observe that LAP-based methods consistently outperform MP-based methods.

Comparing 1 with 2 aggregation, we note that LAP-2 performs better than LAP-1 in both experiments, by a small margin.

Among MP-based methods, we do not observe any similar dominance.

Table 19 and Table 20 .

In this methods, we prune a fraction of weights with smallest scores (e.g. weight magnitude, lookahead distortion, Hessian-based scores) among all weights in the whole network.

The suffix "-normalize" in the tables denotes that the score is normalized by the Frobenius norm of the corresponding layer's score.

For MP, LAP, OBD+LAP and LAP-act, we only report the results for global pruning with normalization, as the normalized versions outperform the unnormalized ones.

In the case of OBD, whose score is already globally designed, we report the results for both unnormalized and normalized versions.

As demonstrated in Section 3.2 for fixed layerwise pruning rates, we observe that LAP and its variants perform better than their global pruning baselines, i.e. MP-normalize and OBD.

We also note that LAP-normalize performs better than MP with pre-specified layerwise pruning rates (appeared in Section 3.2), with a larger gap for higher levels of sparsity.

We test LAP-all on FCN under the same setup as in Section 3.2, and report the results in Table 21 .

All figures are averaged over five trials.

We observe that LAP-all achieves a similar level of performance to LAP, while LAP-all underperforms under a high-sparsity regime.

We suspect that such shortfall originates from the accumulation of error terms incurred by ignoring the effect of activation functions, by which the benefits of looking further fades.

An in-depth theoretical analysis for the determination of an optimal "sight range" of LAP would be an interesting future direction.

As a sanity check, we compare the performance of large neural networks pruned via MP and LAP to the performance of a small network.

In particular, we prune VGG-16, VGG-19, and ResNet-18 trained on CIFAR-10 dataset, to have a similar number of parameters to MobileNetV2 (Sandler et al., 2018) .

For training and pruning VGGs and ResNet, we follows the prior setup in Appendix A while we use the same setup for training MobileNetV2 (Adam optimizer with learning rate of 3 · 10 −4 with batch size 60, and trained 60k steps).

We observe that models pruned via LAP (and MP) exhibit better performance compared to MobileNetV2, even when pruned to have a smaller number of parameters.

In this section, we briefly discuss where the benefits of the sub-network discovered by LAP comes from; does LAP subnetwork have a better generalizability or expressibility?

For this purpose, we look into the generalization gap, i.e., the gap between the training and test accuracies, of the hypothesis learned via LAP procedure.

Below we present a plot of test accuracies (Fig. 4a ) and a plot of generalization gap (Fig. 4b ) for FCN trained with MNIST dataset.

The plot hints us that the network structure learned by LAP may not necessarily have a smaller generalizability.

Remarkably, the generalization gap of the MP-pruned models and the LAP-pruned models are very similar to each other; the benefits of LAP subnetwork compared to MP would be that it can express a better-performing architecture with a network of similar sparsity and generalizability.

remains constant for any hidden neuron j over training via gradient flow.

In other words, the total outward flow of weights is tied to the inward flow of weights for each neuron.

This observation hints at the possibility of a relative undergrowth of weight magnitude of an 'important' connection, in the case where the connection shares the same input/output neuron with other 'important' connections.

From this viewpoint, the multiplicative factors in Eq. (5) take into account the abstract notion of neuronal importance score, assigning significance to connections to the neuron through which more gradient signals have flowed through.

Without considering such factors, LAP reduces to the ordinary magnitude-based pruning.

<|TLDR|>

@highlight

We study a multi-layer generalization of the magnitude-based pruning.
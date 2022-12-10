In this paper, the preparation of a neural network for pruning and few-bit quantization is formulated as a variational inference problem.

To this end, a quantizing prior that leads to a multi-modal, sparse posterior distribution over weights, is introduced and a differentiable Kullback-Leibler divergence approximation for this prior is derived.

After training with Variational Network Quantization, weights can be replaced by deterministic quantization values with small to negligible loss of task accuracy (including pruning by setting weights to 0).

The method does not require fine-tuning after quantization.

Results are shown for ternary quantization on LeNet-5 (MNIST) and DenseNet (CIFAR-10).

Parameters of a trained neural network commonly exhibit high degrees of redundancy BID4 which implies an over-parametrization of the network.

Network compression methods implicitly or explicitly aim at the systematic reduction of redundancy in neural network models while at the same time retaining a high level of task accuracy.

Besides architectural approaches, such as SqueezeNet BID25 or MobileNets (Howard et al., 2017) , many compression methods perform some form of pruning or quantization.

Pruning is the removal of irrelevant units (weights, neurons or convolutional filters) BID31 .

Relevance of weights is often determined by the absolute value ("magnitude based pruning" BID15 ), but more sophisticated methods have been known for decades, e.g., based on second-order derivatives (Optimal Brain Damage (LeCun et al., 1990) and Optimal Brain Surgeon BID18 ) or ARD (automatic relevance determination, a Bayesian framework for determining the relevance of weights, BID35 BID39 BID27 ).

Quantization is the reduction of the bit-precision of weights, activations or even gradients, which is particularly desirable from a hardware perspective BID46 .

Methods range from fixed bit-width computation (e.g., 12-bit fixed point) to aggressive quantization such as binarization of weights and activations BID41 BID52 .

Few-bit quantization (2 to 6 bits) is often performed by k-means clustering of trained weights with subsequent fine-tuning of the cluster centers .

Pruning and quantization methods have been shown to work well in conjunction .

In so-called "ternary" networks, weights can have one out of three possible values (negative, zero or positive) which also allows for simultaneous pruning and few-bit quantization BID33 BID53 ).This work is closely related to some recent Bayesian methods for network compression BID34 BID40 that learn a posterior distribution over network weights under a sparsity-inducing prior.

The posterior distribution over network parameters allows identifying redundancies through three means: weights with (1) an expected value very close to zero and (2) weights with a large variance can be pruned as they do not contribute much to the overall computation.

(3) the posterior variance over non-pruned parameters can be used to determine the required bit-precision (quantization noise can be made as large as implied by the posterior uncertainty).

Additionally, Bayesian inference over modelparameters is known to automatically reduce parameter redundancy by penalizing overly complex models BID36 .In this paper we present Variational Network Quantization (VNQ), a Bayesian network compression method for simultaneous pruning and few-bit quantization of weights.

We extend previous Bayesian pruning methods by introducing a multi-modal quantizing prior that penalizes weights of low variance unless they lie close to one of the target values for quantization.

As a result, weights are either drawn to one of the quantization target values or they are assigned large variance values-see Fig. 1 .

After training, our method yields a Bayesian neural network with a multi-modal posterior over weights (typically with one mode fixed at 0), which is the basis for subsequent pruning and quantization.

Additionally, posterior uncertainties can also be interesting for network introspection and analysis, as well as for obtaining uncertainty estimates over network predictions BID9 BID8 BID5 .

After pruning and hard quantization, and without the need for additional fine-tuning, our method yields a deterministic feed-forward neural network with heavily quantized weights.

Our method is applicable to pre-trained networks but can also be used for training from scratch.

Target values for quantization can either be manually fixed or they can be learned during training.

We demonstrate our method for the case of ternary quantization on LeNet-5 (MNIST) and DenseNet (CIFAR-10).

Figure 1: Distribution of weights (means θ and log-variance log σ 2 ) before and after VNQ training of LeNet-5 on MNIST (validation accuracy before: 99.2% vs. after 195 epochs: 99.3%).

Top row: scatter plot of weights (blue dots) per layer.

Means were initialized from pre-trained deterministic network, variances with log σ 2 = −8.

Bottom row: corresponding density 1 .

Red shaded areas show the funnel-shaped "basins of attraction" induced by the quantizing prior.

Positive and negative target values for ternary quantization have been learned per layer.

After training, weights with small expected absolute value or large variance (log α ij ≥ log T α = 2 corresponding to the funnel marked by the red dotted line) are pruned and remaining weights are quantized without loss in accuracy.

Our method extends recent work that uses a (variational) Bayesian objective for neural network pruning .

In this section, we first motivate such an approach by discussing that the objectives of compression (in the minimum-description-length sense) and Bayesian inference are well-aligned.

We then briefly review the core ingredients that are combined in Sparse Variational Dropout .

The final idea (and also the starting point of our method) is to learn dropout noise levels per weight and prune weights with large dropout noise.

Learning dropout noise per weight can be done by interpreting dropout training as variational inference of an approximate weight-posterior under a sparsity inducing prior -this is known as Variational Dropout which is described in more detail below, after a brief introduction to modern approximate posterior inference in Bayesian neural networks by optimizing the evidence lower bound via stochastic gradient ascent and reparameterization tricks.

Bayesian inference over model parameters automatically penalizes overly complex parametric models, leading to an automatic regularization effect BID14 BID13 ) (see , where the authors show that Sparse Variational Dropout (Sparse VD) successfully prevents a network from fitting unstructured data, that is a random labeling).

The automatic regularization is based on the objective of maximizing model evidence, also know as marginal likelihood.

A very complex model might have a particular parameter setting that achieves extremely good likelihood given the data, however, since the model evidence is obtained via marginalizing parameters, overly complex models are penalized for having many parameter settings with poor likelihood.

This effect is also known as "Bayesian Occams Razor" in Bayesian model selection BID36 BID11 .

The argument can be extended to variational Bayesian inference (with some caveats) via the equivalence of the variational Bayesian objective and the Minimum description length (MDL) principle BID42 BID14 BID13 BID34 .

The evidence lower bound (ELBO), which is maximized in variational inference, is composed of two terms: L E , the average message length required to transmit outputs (labels) to a receiver that knows the inputs and the posterior over model parameters and L C , the average message length to transmit the posterior parameters to a receiver that knows the prior over parameters: DISPLAYFORM0 Maximizing the ELBO minimizes the total message length: max L ELBO = min L E + L C , leading to an optimal trade-off between short description length of the data and the model (thus, minimizing the sum of error cost L E and model complexity cost L C ).

Interestingly, MDL dictates the use of stochastic models since they are in general "more compressible" compared to deterministic models: high posterior uncertainty over parameters is rewarded by the entropy term in L C -higher uncertainty allows the quantization noise to be higher, thus, requiring lower bit-precision for a parameter.

Variational Bayesian inference can also be formally related to the information-theoretic framework for lossy compression, rate-distortion theory, BID3 BID47 BID12 .

The only difference is that rate-distortion requires the use of the optimal prior, which is the marginal over posteriors BID20 BID48 BID21 -providing an interesting connection to empirical Bayes where the prior is learned from the data.

Let D be a dataset of N pairs (x n , y n ) N n=1 and p(y|x, w) be a parameterized model that predicts outputs y given inputs x and parameters w. A Bayesian neural network models a (posterior) distribution over parameters w instead of just a point-estimate.

The posterior is given by Bayes' rule:

DISPLAYFORM0 , where p(w) is the prior over parameters.

Computation of the true posterior is in general intractable.

Common approaches to approximate inference in neural networks are for instance: MCMC methods pioneered in BID39 and later refined, e.g., via stochastic gradient Langevin dynamics BID51 , or variational approximations to the true posterior BID13 , Bayes by Backprop BID1 , Expectation Backpropagation BID44 , Probabilistic Backpropagation BID19 .

In the latter methods the true posterior is approximated by a parameterized distribution q φ (w).

Variational parameters φ are optimized by minimizing the Kullback-Leibler (KL) divergence from the true to the approximate posterior D KL (q φ (w)||p(w|D)).

Since computation of the true posterior is intractable, minimizing this KL divergence is approximately performed by maximizing the so-called "evidence lower bound" (ELBO) or "negative variational free energy" BID28 : DISPLAYFORM1 DISPLAYFORM2 where we have used the Reparameterization Trick 2 BID28 in Eq. (2) to get an unbiased, differentiable, minibatch-based Monte Carlo estimator of the expected log likelihood DISPLAYFORM3 .

Additionally, and in line with similar work BID34 BID40 , we use the Local Reparameterization Trick BID29 to further reduce variance of the stochastic ELBO gradient estimator, which locally marginalizes weights at each layer and instead samples directly from the distribution over pre-activations (which can be computed analytically).

See Appendix A.2 for more details on the Local reparameterization.

Commonly, the prior p(w) and the parametric form of the posterior q φ (w) are chosen such that the KL divergence term can be computed analytically (e.g. a fully factorized Gaussian prior and posterior, known as the mean-field approximation).

Due to the particular choice of prior in our work, a closed-form expression for the KL divergence cannot be obtained but instead we use a differentiable approximation (see Sec. 3.3).

Dropout BID45 ) is a method originally introduced for regularization of neural networks, where activations are stochastically dropped (i.e., set to zero) with a certain probability p during training.

It was shown that dropout, i.e., multiplicative noise on inputs, is equivalent to having noisy weights and vice versa BID50 BID29 .

DISPLAYFORM0 with ij ∼ N (0, 1).

In standard (Gaussian) dropout training, the dropout rates α (or p to be precise) are fixed and the expected log likelihood L D (φ) (first term in Eq. FORMULA2 ) is maximized with respect to the means θ.

BID29 show that Gaussian dropout training is mathematically equivalent to maximizing the ELBO (both terms in Eq. FORMULA2 ), under a prior p(w) and fixed α where the KL term does not depend on θ: DISPLAYFORM1 where the dependencies on α and θ of the terms in Eq. (1) have been made explicit.

The only prior that meets this requirement is the scale invariant log-uniform prior: DISPLAYFORM2 Using this interpretation, it becomes straightforward to learn individual dropout-rates α ij per weight, by including α ij into the set of variational parameters φ = (θ, α).

This procedure was introduced in BID29 under the name "Variational Dropout".

With the choice of a log-uniform prior (Eq. FORMULA7 ) and a factorized Gaussian approximate posterior q φ (w ij ) = N (θ ij , α ij θ 2 ij ) (Eq. (3)) the KL term in Eq. FORMULA2 is not analytically tractable, but the authors of BID29 present an approximation DISPLAYFORM3 see the original publication for numerical values of c 1 , c 2 , c 3 .

Note that due to the mean-field approximation, where the posterior over all weights factorizes into a product over individual weights q φ (w) = q φ (w ij ), the KL divergence factorizes into a sum of individual KL divergences DISPLAYFORM4

Learning dropout rates is interesting for network compression since neurons or weights with very high dropout rates p → 1 can very likely be pruned without loss in accuracy.

However, as the authors of Sparse Variational Dropout (sparse VD) report, the approximation in Eq. FORMULA8 is only accurate for α ≤ 1 (corresponding to p ≤ 0.5).

For this reason, the original variational dropout paper restricted α to values smaller or equal to 1, which are unsuitable for pruning. propose an improved approximation, which is very accurate on the full range of log α: DISPLAYFORM0 with k 1 = 0.63576, k 2 = 1.87320 and k 3 = 1.48695 and S denoting the sigmoid function.

Additionally, the authors propose to use an additive, instead of a multiplicative noise reparameterization, which significantly reduces variance in the gradient ∂L SGVB ∂θij for large α ij .

To achieve this, the multiplicative noise term is replaced by an exactly equivalent additive noise term σ ij ij with σ 2 ij = α ij θ 2 ij and the set of variational parameters becomes φ = (θ, σ): DISPLAYFORM1 After Sparse VD training, pruning is performed by thresholding DISPLAYFORM2 In a threshold of log α = 3 is used, which roughly corresponds to p > 0.95.

Pruning weights that lie above a threshold of T α leads to σ DISPLAYFORM3 which means effectively that weights with large variance but also weights of lower variance and a mean θ ij close to zero are pruned.

A visualization of the pruning threshold can be seen in Fig. 1 (the "central funnel", i.e., the area marked by the red dotted lines for a threshold for T α = 2).

Sparse VD training can be performed from random initialization or with pre-trained networks by initializing the means θ ij accordingly.

In Bayesian Compression BID34 and Structured Bayesian Pruning BID40 , Sparse VD has been extended to include group-sparsity constraints, which allows for pruning of whole neurons or convolutional filters (via learning their corresponding dropout rates).

For pruning weights based on their (learned) dropout rate, it is desirable to have high dropout rates for most weights.

Perhaps surprisingly, Variational Dropout already implicitly introduces such a "high dropout rate constraint" via the implicit prior distribution over weights.

The prior p(w) can be used to induce sparsity into the posterior by having high density at zero and heavy tails.

There is a well known family of such distributions: scale-mixtures of normals BID0 BID34 BID26 : DISPLAYFORM0 where the scales of w are random variables.

A well-known example is the spike-and-slab prior BID37 , which has a delta-spike at zero and a slab over the real line.

BID9 ; BID29 show how Dropout training implies a spike-and-slab prior over weights.

The log uniform prior used in Sparse VD (Eq. (5)) can also be derived as a marginalized scale-mixture of normals DISPLAYFORM1 also known as the normal-Jeffreys prior BID7 .

BID34 discuss how the log-uniform prior can be seen as a continuous relaxation of the spike-and-slab prior and how the alternative formulation through the normal-Jeffreys distribution can be used to couple the scales of weights that belong together and thus, learn dropout rates for whole neurons or convolutional filters, which is the basis for Bayesian Compression BID34 and Structured Bayesian Pruning BID40 .

We formulate the preparation of a neural network for a post-training quantization step as a variational inference problem.

To this end, we introduce a multi-modal, quantizing prior and train by maximizing the ELBO (Eq. FORMULA3 ) under a mean-field approximation of the posterior (i.e., a fully factorized Gaussian).

The goal of our algorithm is to achieve soft quantization, that is learning a posterior distribution such that the accuracy-loss introduced by post-training quantization is small.

Our variational posterior approximation and training procedure is similar to BID29 and with the crucial difference of using a quantizing prior that drives weights towards the target values for quantization.

The log uniform prior (Eq. FORMULA7 ) can be viewed as a continuous relaxation of the spike-and-slab prior with a spike at location 0 BID34 .

We use this insight to formulate a quantizing prior, a continuous relaxation of a "multi-spike-and-slab" prior which has multiple spikes at locations c k , k ∈ {1, . . .

, K}. Each spike location corresponds to one target value for subsequent quantization.

The quantizing prior allows weights of low variance only at the locations of the quantization target values c k .

The effect of using such a quantizing prior during Variational Network Quantization is shown in Fig. 1 .

After training, most weights of low variance are distributed very closely around the quantization target values c k and can thus be replaced by the corresponding value without significant loss in accuracy.

We typically fix one of the quantization targets to zero, e.g., c 2 = 0, which allows pruning weights.

Additionally, weights with a large variance can also be pruned.

Both kinds of pruning can be achieved with an α ij threshold (see Eq. FORMULA13 ) as in sparse Variational Dropout .

Following the interpretation of the log uniform prior p(w ij ) as a marginal over the scale-hyperparameter z ij , we extend Eq. (10) with a hyper-prior over locations DISPLAYFORM0 with p(z ij ) ∝ |z ij | −1 .

The location prior p m (m ij ) is a mixture of weighted delta distributions located at the quantization values c k .

Marginalizing over m yields the quantizing prior DISPLAYFORM1 In our experiments, we use K = 3, a k = 1/K ∀k and c 2 = 0 unless indicated otherwise.

Eq. (9) implies that using a threshold on α ij as a pruning criterion is equivalent to pruning weights whose value does not differ significantly from zero: DISPLAYFORM0 To be precise, T α specifies the width of a scaled standard-deviation band ±σ ij / √ T α around the mean θ ij .

If the value zero lies within this band, the weight is assigned the value 0.

For instance, a pruning threshold which implies p ≥ 0.95 corresponds to a variance band of approximately σ ij /4.

An equivalent interpretation is that a weight is pruned if the likelihood for the value 0 under the approximate posterior exceeds the threshold given by the standard-deviation band (Eq. (13)): DISPLAYFORM1 Extending this argument for pruning weights to a quantization setting, we design a post-training quantization scheme that assigns each weight the quantized value c k with the highest likelihood under the approximate posterior.

Since variational posteriors over weights are Gaussian, this translates into minimizing the squared distance between the mean θ ij and the quantized values c k : DISPLAYFORM2 Additionally, the pruning rate can be increased by first assigning a hard 0 to all weights that exceed the pruning threshold T α (see Eq. FORMULA13 ) before performing the assignment to quantization levels as described above.

Under the quantizing prior (Eq. FORMULA2 ) the KL divergence from the prior D KL (q φ (w)||p(w)) to the mean-field posterior is analytically intractable.

Similar to Kingma et al. FORMULA2 ; , we use a differentiable approximation F KL (θ, σ, c) 3 , composed of a small number of differentiable functions to keep the computational effort low during training.

We now present the approximation for a reference codebook c = [−r, 0, r], r = 0.2, however later we show how the approximation can be used for arbitrary ternary, symmetric codebooks as well.

The basis of our approximation is the approximation F KL,LU introduced by for the KL divergence from a log uniform prior to a Gaussian posterior (see Eq. FORMULA10 ) which is centered around zero.

We observe that a weighted mixture of shifted versions of F KL,LU can be used to approximate the KL divergence for our multi-modal quantizing prior (Eq. FORMULA2 ) (which is composed of shifted versions of the log uniform prior).

In a nutshell, we shift one version of F KL to each codebook entry c k and then use θ-dependent Gaussian windowing functions Ω(θ) to mix the shifted approximations (see more details in the Appendix A.3).

The approximation for the KL divergence from our multi-modal quantizing prior to a Gaussian posterior is given as DISPLAYFORM0 global behavior FORMULA2 with DISPLAYFORM1 We use τ = 0.075 in our experiments.

Illustrations of the approximation, including a comparison against the ground-truth computed via Monte Carlo sampling are shown in FIG1 .

Over the range of θ-and σ-values relevant to our method, the maximum absolute deviation from the ground-truth is 1.07 nats.

See FIG4 in the Appendix for a more detailed quantitative evaluation of our approximation.

This KL approximation in Eq. FORMULA2 , developed for the reference codebook c r = [−r, 0, r], can be reused for any symmetric ternary codebook c a = [−a, 0, a], a ∈ R + , since c a can be represented with the reference codebook and a positive scaling factor s, c a = sc r , s = a/r.

As derived in the Appendix (A.4), this re-scaling translates into a multiplicative re-scaling of the variational parameters θ and σ.

The KL divergence from a prior based on the codebook c a to the posterior q φ (w) is thus given by D KL (q φ (w)||p ca (w)) ≈ F KL (θ/s, σ/s, c r ).

This result allows learning the quantization level a during training as well.

In our experiments, we train with VNQ and then first prune via thresholding log α ij ≥ log T α = 2.

Remaining weights are then quantized by minimizing the squared distance to the quantization values c k (see Sec. 3.2).

We use warm-up BID43 , that is, we multiply the KL divergence term (Eq. (2)) with a factor β, where β = 0 during the first few epochs and then linearly ramp up to β = 1.

To improve stability of VNQ training, we ensure through clipping that log σ 2 ij ∈ (−10, 1) and θ ij ∈ (−a − 0.3679σ, a + 0.3679σ) (which corresponds to a shifted log α threshold of 2, that is, we clip θ ij if it lies left of the −a funnel or right of the +a funnel, compare Fig. 1 ).

This leads to a clipping-boundary that depends on trainable parameters.

To avoid weights getting stuck at these boundaries, we use gradient-stopping, that is, we apply the gradient to a so-called "shadow weight" and use the clipped weight-value only for the forward pass.

Without this procedure our method still works, but accuracies are a bit worse, particularly on CIFAR-10.

When learning codebook values weighting functions Ω(θ) that mix the shifted known approximation to form the final approximation F KL shown in the bottom row (gold), compared against the ground-truth (MC sampled).

Each column corresponds to a different value of σ.

A comparison between ground-truth and our approximation over a large range of σ and θ values is shown in the Appendix in FIG4 .

Note that since the priors are improper, KL approximation and ground-truth can only be compared up to an additive constant C -the constant is irrelevant for network training but has been chosen in the plot such that ground-truth and approximation align for large values of θ.a during training, we use a lower learning rate for adjusting the codebook, otherwise we observe a tendency for codebook values to collapse in early stages of training (a similar observation was made by ).

Additionally, we ensure a ≥ 0.05 by clipping.

We demonstrate our method with LeNet-5 4 (LeCun et al., 1998) on the MNIST handwritten digits dataset.

Images are pre-processed by subtracting the mean and dividing by the standard-deviation over the training set.

For the pre-trained network we run 5 epochs on a randomly initialized network (Glorot initialization, Adam optimizer), which leads to a validation accuracy of 99.2%.

We initialize means θ with the pre-trained weights and variances with log σ 2 = −8.

The warm-up factor β is linearly increased from 0 to 1 during the first 15 epochs.

VNQ training runs for a total of 195 epochs with a batch-size of 128, the learning rate is linearly decreased from 0.001 to 0 and the learning rate for adjusting the codebook parameter a uses a learning rate that is 100 times lower.

We initialize with a = 0.2.

Results are shown in Table 1 , a visualization of the distribution over weights after VNQ training is shown in Fig. 1 .We find that VNQ training sufficiently prepares a network for pruning and quantization with negligible loss in accuracy and without requiring subsequent fine-tuning.

Training from scratch yields a similar performance compared to initializing with a pre-trained network, with a slightly higher pruning rate.

Compared to pruning methods that do not consider few-bit quantization in their objective, we achieve significantly lower pruning rates.

This is an interesting observation since our method is based on a similar objective (e.g., compared to Sparse VD) but with the addition of forcing nonpruned weights to tightly cluster around the quantization levels.

Few-bit quantization severely limits network capacity.

Perhaps this capacity limitation must be countered by pruning fewer weights.

Our pruning rates are roughly in line with other papers on ternary quantization, e.g., BID53 , who report sparsity levels between 30% and 50% with their ternary quantization method.

Note that Table 1 : Results on LeNet-5 (MNIST), showing validation error, percentage of non-pruned weights and bit-precision per parameter.

Original is our pre-trained LeNet-5.

We show results after VNQ training (without pruning and quantization, denoted by "no P&Q") where weights were deterministically replaced by the full-precision means θ and for VNQ training with subsequent pruning and quantization (denoted by "P&Q").

"random init." denotes training with random weight initialization (Glorot).

We also show results of non-ternary or pruning-only methods (P): Deep Compression , Soft weight-sharing , Sparse VD , Bayesian Compression BID34 and Stuctured Bayesian Pruning BID40 0.86 --a direct comparison between pruning, quantizing and ternarizing methods is difficult and depends on many factors such that a fair computation of the compression rate that does not implicitly favor certain methods is hardly possible within the scope of this paper.

For instance, compression rates for pruning methods are typically reported under the assumption of a CSC storage format which would not fully account for the compression potential of a sparse ternary matrix.

We thus choose not to report any measures for compression rates, however for the methods listed in Table 1 , they can easily be found in the literature.

Our second experiment uses a modern DenseNet BID23 ) (k = 12, depth L = 76, with bottlenecks) on CIFAR-10 ( BID30 ).

We follow the CIFAR-10 settings of BID23 5 .

The training procedure is identical to the procedure on MNIST with the following exceptions: we use a batch-size of 64 samples, the warm-up weight β of the KL term is 0 for the first 5 epochs and is then linearly ramped up from 0 to 1 over the next 15 epochs, the learning rate of 0.005 is kept constant for the first 50 epochs and then linearly decreased to a value of 0.003 when training stops after 150 epochs.

We pre-train a deterministic DenseNet (reaching validation accuracy of 93.19%) to initialize VNQ training.

The codebook parameter for non-zero values a is initialized with the maximum absolute value over pre-trained weights per layer.

Results are shown in Table 2 .

A visualization of the distribution over weights after VNQ training is shown in the Appendix FIG3 .We generally observe lower levels of sparsity for DenseNet, compared to LeNet.

This might be due to the fact that DenseNet already has an optimized architecture which removed a lot of redundant parameters from the start.

In line with previous publications, we generally observed that the first and last layer of the network are most sensitive to pruning and quantization.

However, in contrast to many other methods that do not quantize these layers (e.g., BID53 ), we find that after sufficient training, the complete network can be pruned and quantized with very little additional loss in accuracy (see Table 2 ).

Inspecting the weight scatter-plot for the first and last layer (Appendix FIG3 , top-left and bottom-right panel) it can be seen that some weights did not settle on one of the 5 Our DenseNet(L = 76, k = 12) consists of an initial convolutional layer (3 × 3 with 16 output channels), followed by three dense blocks (each with 12 pairs of 1 × 1 convolution bottleneck followed by a 3 × 3 convolution, number of channels depends on growth-rate k = 12) and a final classification layer (global average pooling that feeds into a dense layer with softmax activation).

In-between the dense blocks (but not after the last dense block) are (pooling) transition layers (1 × 1 convolution followed by 2 × 2 average pooling with a stride of 2).

Table 2 : Results on DenseNet (CIFAR-10), showing the error on the validation set, the percentage of non-pruned weights and the bit-precision per weight.

Original denotes the pre-trained network.

We show results after VNQ training without pruning and quantization (weights were deterministically replaced by the full-precision means θ) denoted by "no P&Q", and VNQ with subsequent pruning and quantization denoted by "P&Q" (in the condition "(w/o 1)" we use full-precision means for the weights in the first layer and do not prune and quantize this layer).

prior modes (the "funnels") after VNQ training, particularly the first layer has a few such weights with very low variance.

It is likely that quantizing these weights causes the additional loss in accuracy that we observe when quantizing the whole network.

Without gradient stopping (i.e., applying gradients to a shadow weight at the trainable clipping boundary) we have observed that pruning and quantizing the first layer leads to a more pronounced drop in accuracy (about 3% compared to a network where the first layer is kept with full precision, not shown in results).

Our method is an extension of Sparse VD , originally used for network pruning.

In contrast, we use a quantizing prior, leading to a multi-modal posterior suitable for fewbit quantization and pruning.

Bayesian Compression and Structured Bayesian Pruning BID34 BID40 extend Sparse VD to prune whole neurons or filters via groupsparsity constraints.

Additionally, in Bayesian Compression the required bit-precision per layer is determined via the posterior variance.

In contrast to our method, Bayesian Compression does not explicitly enforce clustering of weights during training and thus requires bit-widths in the range between 5 and 18 bits.

Extending our method to include group-constraints for pruning is an interesting direction for future work.

Another Bayesian method for simultaneous network quantization and pruning is soft weight-sharing (SWS) , which uses a Gaussian mixture model prior (and a KL term without trainable parameters such that the KL term reduces to the prior entropy).

SWS acts like a probabilistic version of k-means clustering with the advantage of automatic collapse of unnecessary mixture components.

Similar to learning the codebooks in our method, soft weight-sharing learns the prior from the data, a technique known as empirical Bayes.

We cannot directly compare against soft weight-sharing since the authors do not report results on ternary networks.

BID10 learn dropout rates by using a continuous relaxation of dropout's discrete masks (via the concrete distribution).

The authors learn layer-wise dropout rates, which does not allow for dropout-rate-based pruning.

We experimented with using the concrete distribution for learning codebooks for quantization with promising early results but so far we have observed lower pruning rates or lower accuracy compared to VNQ.

A non-probabilistic state-of-the-art method for network ternarization is Trained Ternary Quantization BID53 which uses fullprecision shadow weights during training, but quantized forward passes.

Additionally it learns a (non-symmetric) scaling per layer for the non-zero quantization values, similar to our learned quantization level a. While the method achieves impressive accuracy, the sparsity and thus pruning rates are rather low (between 30% and 50% sparsity) and the first and last layer need to be kept with full precision.

A potential shortcoming of our method is the KL divergence approximation (Sec. 3.3) .

While the approximation is reasonably good on the relevant range of θ-and σ-values, there is still room for improvement which could have the benefit that weights are drawn even more tightly onto the quantization levels, resulting in lower accuracy loss after quantization and pruning.

Since our functional approximation to the KL divergence only needs to be computed once and an arbitrary amount of ground-truth data can be produced, it should be possible to improve upon the approximation presented here at least by some brute-force function approximation, e.g., a neural network, polynomial or kernel regression.

The main difficulty is that the resulting approximation must be differentiable and must not introduce significant computational overhead since the approximation is evaluated once for each network parameter in each gradient step.

We have also experimented with a naive Monte-Carlo approximation of the KL divergence term.

This has the disadvantage that local reparameterization (where pre-activations are sampled directly) can no longer be used, since weight samples are required for the MC approximation.

To keep computational complexity comparable, we used a single sample for the MC approximation.

In our LeNet-5 on MNIST experiment the MC approximation achieves comparable accuracy with higher pruning rates compared to our functional KL approximation.

However, with DenseNet on CIFAR-10 and the MC approximation validation accuracy plunges catastrophically after pruning and quantization.

See Sec. A.3 in the Appendix for more details.

Compared to similar methods that only consider network pruning, our pruning rates are significantly lower.

This does not seem to be a particular problem of our method since other papers on network ternarization report similar or even lower sparsity levels BID53 roughly achieve between 30% and 50% sparsity).

The reason for this might be that heavily quantized networks have a much lower capacity compared to full-precision networks.

This limited capacity might require that the network compensates by effectively using more weights such that the pruning rates become significantly lower.

Similar trends have also been observed with binary networks, where drops in accuracy could be prevented by increasing the number of neurons (with binary weights) per layer.

Principled experiments to test the trade-off between low bit-precision and sparsity rates would be an interesting direction for future work.

One starting point could be to test our method with more quantization levels (e.g., 5, 7 or 9) and investigate how this affects the pruning rate.

We follow Sparse VD and use the Local Reparameterization Trick BID29 and Additive Noise Reparmetrization to optimize the stochastic gradient variational lower bound L SGVB (Eq. (2)).

We optimize posterior means and log-variances (θ, log σ 2 ) and the codebook level a. We apply Variational Network Quantization to fully connected and convolutional layers.

Denoting inputs to a layer with A M ×I , outputs of a layer with B M ×O and using local reparameterization we get: DISPLAYFORM0 for a fully connected layer.

Similarly activations for a convolutional layer are computed as follows DISPLAYFORM1 2 denotes an element-wise operation, * is the convolution operation and vec(·) denotes reshaping of a matrix/tensor into a vector.

Under the quantizing prior (Eq. (12)) the KL divergence from the log uniform prior to the meanfield posterior D KL (q φ (w ij )||p(w ij )) is analytically intractable. presented an approximation for the KL divergence under a (zero-centered) log uniform prior (Eq. FORMULA7 ).

Since our quantizing prior is essentially a composition of shifted log uniform priors, we construct a composition of the approximation given by , shown in Eq. (7).

The original approximation can be utilized to calculate a KL divergence approximation (up to an additive constantC) from a shifted log-uniform prior p(w ij ) ∝ 1 |wij −r| to a Gaussian posterior q φ (w ij ) by transferring the shift to the posterior parameter θ DISPLAYFORM0 For small posterior variances σ 2 ij (σ ij r) and means near the quantization levels (i.e., |θ ij | ≈ r), the KL divergence is dominated by the mixture prior component located at the respective quantization level r. For these values of θ and σ, the KL divergence can be approximated by shifting the approximation F LU,KL (θ, σ) to the quantization level r, i.e., F LU,KL (θ ± r, σ).

For small σ and values of θ near zero or far away from any quantization level, as well as for large values of σ and arbitrary θ, the KL divergence can be approximated by the original non-shifted approximation F LU,KL (θ, σ).

Based on these observations we construct our KL approximation by properly mixing shifted versions of F LU,KL (θ ± r, σ).

We use Gaussian window functions Ω(θ ± r) to perform this weighting (to ensure differentiability).

The remaining θ domain is covered by an approximation located at zero and weighted such that this approximation is dominant near zero and far away from the quantization levels, which is achieved by introducing the constraint that all window functions sum up to one on the full θ domain.

See FIG1 for a visual representation of shifted approximations and their respective window functions.

We evaluate the quality of our KL approximation (Eq. (16)) by comparing against a ground-truth Monte Carlo approximation on a dense grid over the full range of relevant θ and σ values.

Results of this comparison are shown in FIG4 .

Alternatively to the functional KL approximation, one could also use a naive Monte Carlo approximation directly.

This has the disadvantage that local reparameterization can no longer be used, since actual samples of the weights must be drawn.

To assess the quality of our functional KL approximation, we also compare against experiments where we use a naive MC approximation of the KL divergence term, where we only use a single sample for approximating the expectation to keep computational complexity comparable to our original method.

Note that the "ground-truth" MC approximation used before to evaluate KL approximation quality uses many more samples which would be prohibitively expensive during training.

To test for the effect of FORMULA2 ) and the bottom panel shows the difference between both.

The maximum absolute error between our approximation and the ground-truth is 1.07 nats.local reparameterization in isolation we also show results for our functional KL approximation without using local reparameterization.

The results in Table 3 show that the naive MC approximation of the KL term leads to slightly lower validation error on MNIST (LeNet-5) (with higher pruning rates) but on CIFAR-10 (DenseNet) the validation error of the network trained with the naive MC approximation catastrophically increases after pruning and quantizing the network.

Except for removing local reparameterization or plugging in the naive MC approximation, experiments were ran as described in Sec. 4.

Table 3 : Comparing the effects of local reparameterization and naive MC approximation of the KL divergence.

"func.

KL approx" denotes our functional approximation of the KL divergence given by Eq. (16).

"naive MC approx" denotes a naive Monte Carlo approximation that uses a single sample only.

The first column of results shows the validation error after training, but without pruning and quantization (no P&Q), the next column shows results after pruning and quantization (results in brackets correspond to the validation error without pruning and quantizing the first layer).

Inspecting the distribution over weights after training with the naive MC approximation for the KL divergence, shown in FIG6 for LeNet-5 and in Fig. 6 for DenseNet, reveals that weight-means tend to be more dispersed and weight-variances tend to be generally lower than when training with our functional KL approximation (compare Fig. 1 for LeNet-5 and FIG3 for DenseNet).

We speculate that the combined effects of missing local reparameterization and single-sample MC approximation lead to more noisy gradients.

Figure 6: Visualization of distribution over DenseNet weights after training on CIFAR-10 with naive MC approximation for the KL divergence (and without local reparameterization).

Each panel shows one layer, starting in the top-left corner with the input-and ending with the final layer in the bottomright panel (going row-wise, that is first moving to the right as layers increase).

Validation accuracy before pruning and quantization is 79.25% but plunges to 22.29% after pruning and quantization.

<|TLDR|>

@highlight

We quantize and prune neural network weights using variational Bayesian inference with a multi-modal, sparsity inducing prior.

@highlight

Proposes to use a mixture of continuous spike propto 1/abs as prior for a Bayesian neural network and demonstrates the good performance with relatively sparsified convnets for minist and cifar-10.

@highlight

This paper presents a variational Bayesian approach for quantising neural network weights to ternary values post-training in a principled way.
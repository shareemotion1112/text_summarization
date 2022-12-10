Stochastic neural net weights are used in a variety of contexts, including regularization, Bayesian neural nets, exploration in reinforcement learning, and evolution strategies.

Unfortunately, due to the large number of weights, all the examples in a mini-batch typically share the same weight perturbation, thereby limiting the variance reduction effect of large mini-batches.

We introduce flipout, an efficient method for decorrelating the gradients within a mini-batch by implicitly sampling pseudo-independent weight perturbations for each example.

Empirically, flipout achieves the ideal linear variance reduction for fully connected networks, convolutional networks, and RNNs.

We find significant speedups in training neural networks with multiplicative Gaussian perturbations.

We show that flipout is effective at regularizing LSTMs, and outperforms previous methods.

Flipout also enables us to vectorize evolution strategies: in our experiments, a single GPU with flipout can handle the same throughput as at least 40 CPU cores using existing methods, equivalent to a factor-of-4 cost reduction on Amazon Web Services.

Stochasticity is a key component of many modern neural net architectures and training algorithms.

The most widely used regularization methods are based on randomly perturbing a network's computations BID29 BID7 .

Bayesian neural nets can be trained with variational inference by perturbing the weights BID4 BID0 .

Weight noise was found to aid exploration in reinforcement learning BID20 BID2 .

Evolution strategies (ES) minimizes a black-box objective by evaluating many weight perturbations in parallel, with impressive performance on robotic control tasks BID25 .

Some methods perturb a network's activations BID29 BID7 , while others perturb its weights BID4 BID0 BID20 BID2 BID25 .

Stochastic weights are appealing in the context of regularization or exploration because they can be viewed as a form of posterior uncertainty about the parameters.

However, compared with stochastic activations, they have a serious drawback: because a network typically has many more weights than units, it is very expensive to compute and store separate weight perturbations for every example in a mini-batch.

Therefore, stochastic weight methods are typically done with a single sample per mini-batch.

In contrast, activations are easy to sample independently for different training examples within a mini-batch.

This allows the training algorithm to see orders of magnitude more perturbations in a given amount of time, and the variance of the stochastic gradients decays as 1/N , where N is the mini-batch size.

We believe this is the main reason stochastic activations are far more prevalent than stochastic weights for neural net regularization.

In other settings such as Bayesian neural nets and evolution strategies, one is forced to use weight perturbations and live with the resulting inefficiency.

In order to achieve the ideal 1/N variance reduction, the gradients within a mini-batch need not be independent, but merely uncorrelated.

In this paper, we present flipout, an efficient method for decorrelating the gradients between different examples without biasing the gradient estimates.

Flipout applies to any perturbation distribution that factorizes by weight and is symmetric around 0-including DropConnect, multiplicative Gaussian perturbations, evolution strategies, and variational Bayesian neural nets-and to many architectures, including fully connected nets, convolutional nets, and RNNs.

In Section 3, we show that flipout gives unbiased stochastic gradients, and discuss its efficient vectorized implementation which incurs only a factor-of-2 computational overhead compared with shared perturbations.

We then analyze the asymptotics of gradient variance with and without flipout, demonstrating strictly reduced variance.

In Section 4, we measure the variance reduction effects on a variety of architectures.

Empirically, flipout gives the ideal 1/N variance reduction in all architectures we have investigated, just as if the perturbations were done fully independently for each training example.

We demonstrate speedups in training time in a large batch regime.

We also use flipout to regularize the recurrent connections in LSTMs, and show that it outperforms methods based on dropout.

Finally, we use flipout to vectorize evolution strategies BID25 , allowing a single GPU to handle the same throughput as 40 CPU cores using existing approaches; this corresponds to a factor-of-4 cost reduction on Amazon Web Services.

We use the term "weight perturbation" to refer to a class of methods which sample the weights of a neural network stochastically at training time.

More precisely, let f (x, W ) denote the output of a network with weights W on input x. The weights are sampled from a distribution q θ parameterized by θ.

We aim to minimize the expected loss DISPLAYFORM0 , where L is a loss function, and D denotes the data distribution.

The distribution q θ can often be described in terms of perturbations: W = W + ∆W , where W are the mean weights (typically represented explicitly as part of θ) and ∆W is a stochastic perturbation.

We now give some specific examples of weight perturbations.

Gaussian perturbations.

If the entries ∆W ij are sampled independently from Gaussian distributions with variance σ 2 ij , this corresponds to the distribution W ij ∼ N (W ij , σ 2 ij ).

Using the reparameterization trick BID9 , this can be rewritten as W ij = W ij + σ ij ij , where ij ∼ N (0, 1); this representation allows the gradients to be computed using backprop.

A variant of this is multiplicative Gaussian perturbation, where the perturbations are scaled according to the weights: DISPLAYFORM1 , where again ij ∼ N (0, 1).

Multiplicative perturbations can be more effective than additive ones because the information content of the weights is the same regardless of their scale.

DropConnect.

DropConnect BID30 ) is a regularization method inspired by dropout BID29 which randomly zeros out a random subset of the weights.

In the case of a 50% drop rate, this can be thought of as a weight perturbation where W = W/2 and each entry ∆W ij is sampled uniformly from ±W ij .Variational Bayesian neural nets.

Rather than fitting a point estimate of a neural net's weights, one can adopt the Bayesian approach of putting a prior distribution p(W ) over the weights and approximating the posterior distribution p(W |D) ∝ p(W )p(D|W ), where D denotes the observed data.

BID4 observed that one could fit an approximation q θ (W ) ≈ p(W |D) using variational inference; in particular, one could maximize the evidence lower bound (ELBO) with respect to θ: DISPLAYFORM2 The negation of the second term can be viewed as the description length of the data, and the negation of the first term can be viewed as the description length of the weights BID6 .

BID4 observed that if q is chosen to be a factorial Gaussian, sampling from θ can be thought of as Gaussian weight perturbation where the variance is adapted to maximize F. BID0 later combined this insight with the reparameterization trick BID9 to derive unbiased stochastic estimates of the gradient of F.Evolution strategies.

ES BID22 ) is a family of black box optimization algorithms which use weight perturbations to search for model parameters.

ES was recently proposed as an alternative reinforcement learning algorithm BID26 BID25 .

In each iteration, ES generates a collection of weight perturbations as candidates and evaluates each according to a fitness function F .

The gradient of the parameters can be estimated from the fitness function evaluations.

ES is highly parallelizable, because perturbations can be generated and evaluated independently by different workers.

Suppose M is the number of workers, W is the model parameter, σ is the standard deviation of the perturbations, α is the learning rate, F is the objective function, and ∆W m is the Gaussian noise generated at worker m. The ES algorithm tries to maximize E∆W F W + σ∆W .

The gradient of the objective function and the update rule can be given as: DISPLAYFORM3

In some cases, it's possible to reformulate weight perturbations as activation perturbations, thereby allowing them to be efficiently computed fully independently for different examples in a mini-batch.

In particular, BID10 showed that for fully connected networks with no weight sharing, unbiased stochastic gradients could be computed without explicit weight perturbations using the local reparameterization trick (LRT).

For example, suppose X is the input mini-batch, W is the weight matrix and B = XW is the matrix of activations.

The LRT samples the activations B rather than the weights W .

In the case of a Gaussian posterior, the LRT is given by: DISPLAYFORM0 where b m,j denotes the perturbed activations.

While the exact LRT applies only to fully connected networks with no weight sharing, BID10 also introduced variational dropout, a regularization method inspired by the LRT which performs well empirically even for architectures the LRT does not apply to.

Control variates are another general class of strategies for variance reduction, both for black-box optimization BID31 BID21 BID19 and for gradient-based optimization BID24 BID18 BID15 .

Control variates are complementary to flipout, so one could potentially combine these techniques to achieve a larger variance reduction.

We also note that the fastfood transform BID13 ) is based on similar mathematical techniques.

However, whereas fastfood is used to approximately multiply by a large Gaussian matrix, flipout preserves the random matrix's distribution and instead decorrelates the gradients between different samples.

As described above, weight perturbation algorithms suffer from high variance of the gradient estimates because all training examples in a mini-batch share the same perturbation.

More precisely, sharing the perturbation induces correlations between the gradients, implying that the variance can't be eliminated by averaging.

In this section, we introduce flipout, an efficient way to perturb the weights quasi-independently within a mini-batch.

We make two assumptions about the weight distribution q θ : (1) the perturbations of different weights are independent; and (2) the perturbation distribution is symmetric around zero.

These are nontrivial constraints, but they encompass important use cases: independent Gaussian perturbations (e.g. as used in variational BNNs and ES) and DropConnect with drop probability 0.5.

We observe that, under these assumptions, the perturbation distribution is invariant to elementwise multiplication by a random sign matrix (i.e. a matrix whose entries are ±1).

In the following, we denote elementwise multiplication by •. Observation 1.

Let q θ be a perturbation distribution that satisfies the above assumptions, and let ∆W ∼ q θ .

Let E be a random sign matrix that is independent of ∆W .

Then ∆W = ∆W • E is identically distributed to ∆W .

Furthermore, the loss gradients computed using ∆W are identically distributed to those computed using ∆W .Flipout exploits this fact by using a base perturbation ∆W shared by all examples in the mini-batch, and multiplies it by a different rank-one sign matrix for each example: DISPLAYFORM0 where the subscript denotes the index within the mini-batch, and r n and s n are random vectors whose entries are sampled uniformly from ±1.

According to Observation 1, the marginal distribution over gradients computed for individual training examples will be identical to the distribution computed using shared weight perturbations.

Consequently, flipout yields an unbiased estimator for the loss gradients.

However, by decorrelating the gradients between different training examples, we can achieve much lower variance updates when averaging over a mini-batch.

Vectorization.

The advantage of flipout over explicit perturbations is that computations on a minibatch can be written in terms of matrix multiplications.

This enables efficient implementations on GPUs and modern accelerators such as the Tensor Processing Unit (TPU) (Jouppi et al., 2017) .

Let x denote the activations in one layer of a neural net.

The next layer's activations are given by: DISPLAYFORM1 where φ denotes the activation function.

To vectorize these computations, we define matrices R and S whose rows correspond to the random sign vectors r n and s n for all examples in the mini-batch.

The above equation is vectorized as: DISPLAYFORM2 This defines the forward pass.

Because R and S are sampled independently of W and ∆W , we can backpropagate through Eqn.

4 to obtain derivatives with respect to W , ∆W , and X.Computational cost.

In general, the most expensive operation in the forward pass is matrix multiplication.

Flipout's forward pass requires two matrix multiplications instead of one, and therefore should be roughly twice as expensive as a forward pass with a single shared perturbation when the multiplications are done in sequence.

1 However, note that the two matrix multiplications are independent and can be done in parallel; this incurs the same overhead as the local reparameterization trick BID10 .A general rule of thumb for neural nets is that the backward pass requires roughly twice as many FLOPs as the forward pass.

This suggests that each update using flipout ought to be about twice as expensive as an update with a single shared perturbation (if the matrix multiplications are done sequentially); this is consistent with our experience.

Evolution strategies.

ES is a highly parallelizable algorithm; however, most ES systems are engineered to run on multi-core CPU machines and are not able to take full advantage of GPU parallelism.

Flipout enables ES to run more efficiently on a GPU because it allows each worker to evaluate a batch of quasi-independent perturbations rather than only a single perturbation.

To apply flipout to ES, we can simply replicate the starting state by the number of flipout perturbations N , at each worker.

Instead of Eqn.

1, the update rule using M workers becomes: DISPLAYFORM3 where m indexes workers, n indexes the examples in a worker's batch, and F mn is the reward evaluated with the n th perturbation at worker m. Hence, each worker is able to evaluate multiple perturbations as a batch, allowing for parallelism on a GPU architecture.

In this section, we analyze the variance of stochastic gradients with and without flipout.

We show that flipout is guaranteed to reduce the variance of the gradient estimates compared to using naïve shared perturbations.

DISPLAYFORM0 ) under the perturbation ∆W for a single training example x. (Note that G x is a random variable which depends on both x and ∆W .

We analyze a single entry of the gradient so that we can work with scalar-valued variances.)

We denote the gradient averaged over a mini-batch as the random variable DISPLAYFORM1 denotes a mini-batch of size N , and ∆W n denotes the perturbation for the n th example.

(The randomness comes from both the choice of B and the random perturbations.)

For simplicity, we assume that the x n are sampled i.i.d.

from the data distribution.

Using the Law of Total Variance, we decompose Var(G B ) into a data term (the variance of the exact mini-batch gradients) and an estimation term (the estimation variance for a fixed mini-batch): DISPLAYFORM2 Notice that the data term decays with N while the estimation term may not, due to its dependence on the shared perturbation.

But we can break the estimation term into two parts for which we can analyze the dependence on N .

To do this, we reformulate the standard shared perturbation scheme as follows: ∆W is generated by first sampling ∆W and then multiplying it by a random sign matrix rs as in Eqn.

3 -exactly like flipout, except that the sign matrix is shared by the whole minibatch.

According to Observation 1, this yields an identical distribution for ∆W to the standard shared perturbation scheme.

Based on this, we obtain the following decomposition: Theorem 2 (Variance Decomposition Theorem).

Define α, β, and γ to be DISPLAYFORM3 DISPLAYFORM4 Under the assumptions of Observation 1, the variance of the gradients under shared perturbations and flipout perturbations can be written in terms of α, β, and γ as follows:Fully independent perturbations: DISPLAYFORM5 Flipout: DISPLAYFORM6 Proof.

Details of the proof are provided in Appendix A.We can interpret α, β, and γ as follows.

First, α combines the data term from Eqn.

6 with the expected estimation variance for individual data points.

This corresponds to the variance of the gradients on individual training examples, so fully independent perturbations yield a total variance of α/N .

The other terms, β and γ, reflect the covariance between the estimation errors on different training examples as a result of the shared perturbations.

The term β reflects the covariance that results from sampling r and s, so it is eliminated by flipout, which samples these vectors independently.

Finally, γ reflects the covariance that results from sampling ∆W , which flipout does not eliminate.

Empirically, for all the neural networks we investigated, we found that α β γ.

This implies the following behavior for Var(G B ) as a function of N : for small N , the data term α/N dominates, giving a 1/N variance reduction; with shared perturbations, once N is large enough that α/N < β, the variance Var(G B ) levels off to β.

However, flipout continues to enjoy a 1/N variance reduction in this regime.

In principle, flipout's variance should level off at the point where α/N < γ, but in all of our experiments, γ was small enough that this never occurred: flipout's variance was approximately α/N throughout the entire range of N values we explored, just as if the perturbations were sampled fully independently for every training example.

We first verified empirically the variance reduction effect of flipout predicted by Theorem 2; we measured the variance of the gradients under different perturbations for a wide variety of neural network architectures and batch sizes.

In Section 4.2, we show that flipout applied to Gaussian perturbations and DropConnect is effective at regularizing LSTM networks.

In Section 4.3, we demonstrate that flipout converges faster than shared perturbations when training with large minibatches.

Finally, in Section 4.4 we present experiments combining Evolution Strategies with flipout in both supervised learning and reinforcement learning tasks.

In our experiments, we consider the four architectures shown in TAB1 (details in Appendix B).

Since the main effect of flipout is intended to be variance reduction of the gradients, we first estimated the gradient variances of several architectures with mini-batch sizes ranging from 1 to 8196 FIG0 ).

We experimented with three perturbation methods: a single shared perturbation per minibatch, the local reparameterization trick (LRT) of BID10 , and flipout.

For each of the FC, ConVGG, and LSTM architectures, we froze a partially trained network to use for all variance estimates, and we used multiplicative Gaussian perturbations with σ 2 = 1.

We computed Monte Carlo estimates of the gradient variance, including both the data and estimation terms in Eqn.

6.

Confidence intervals are based on 50 independent runs of the estimator.

Details are given in Appendix C.The analysis in Section 3.2 makes strong predictions about the shapes of the curves in FIG0 .

By Theorem 2, the variance curves for flipout and shared perturbations each have the form a + b/N , where N is the mini-batch size.

On a log-log plot, this functional form appears as a linear regime with slope -1, a constant regime, and a smooth phase transition in between.

Also, because the distribution of individual gradients is identical with and without flipout, the curves must agree for N = 1.

Our plots are consistent with both of these predictions.

We observe that for shared perturbations, the phase transition consistently occurs for mini-batch sizes somewhere between 100 and 1000.

In contrast, flipout gives the ideal linear variance reduction throughout the range of mini-batch sizes we investigated, i.e., its behavior is indistinguishable from fully independent perturbations.

As analyzed by BID10 , the LRT gradients are fully independent within a mini-batch, and are therefore guaranteed to achieve the ideal 1/N variance reduction.

Furthermore, they reduce the variance below that of explicit weight perturbations, so we would expect them to achieve smaller variance than flipout, as shown in FIG0 .

However, flipout is applicable to a wider variety of architectures, including convolutional nets and RNNs.

We evaluated the regularization effect of flipout on the character-level and word-level language modeling tasks with the Penn Treebank corpus (PTB) BID16 .

We compared flipout to several other methods for regularizing RNNs: naïve dropout BID32 , variational dropout BID3 , recurrent dropout BID27 , zoneout BID12 , and DropConnect BID17 .

BID32 apply dropout only to the feed-forward connections of an RNN (to the input, output, and connections between layers).

The other methods regularize the recurrent connections as well: BID27 apply dropout to the cell update vector, with masks sampled either per step or per sequence; BID3 apply dropout to the forward and recurrent connections, with all dropout masks sampled per sequence.

BID17 use DropConnect to regularize the hidden-to-hidden weight matrices, with a single DropConnect mask shared between examples in a mini-batch.

We denote their model WD (for weight-dropped LSTM).Character-Level.

For our character-level experiments, we used a single-layer LSTM with 1000 hidden units.

We trained each model on non-overlapping sequences of 100 characters in batches of size 32, using the AMSGrad variant of Adam (Reddi et al., 2018) with learning rate 0.002.

We perform early stopping based on validation performance.

Here, we applied flipout to the hidden-tohidden weight matrix.

More hyperparameter details are given in Appendix D. The results, measured in bits-per-character (BPC) for the validation and test sequences of PTB, are shown in Table 2 .

In the table, shared perturbations and flipout (with Gaussian noise sampling) are denoted by Mult.

Gauss and Mult.

Gauss + Flipout, respectively.

We also compare to RBN (recurrent batchnorm) (Cooijmans et al., 2017) and H-LSTM+LN (HyperLSTM + LayerNorm) BID5 .

Mult.

Gauss + Flipout outperforms the other methods, and achieves the best reported results for this architecture.

Word-Level.

For our word-level experiments, we used a 2-layer LSTM with 650 hidden units per layer and 650-dimensional word embeddings.

We trained on sequences of length 35 in batches of size 40, for 100 epochs.

We used SGD with initial learning rate 30, and decayed the learning rate by a factor of 4 based on the nonmonotonic criterion introduced by BID17 .

We used flipout to implement DropConnect, as described in Section 2.1, and call this WD+Flipout.

We applied WD+Flipout to the hidden-to-hidden weight matrices for recurrent regularization, and used the same hyperparameters as BID17 .

We used embedding dropout (setting rows of the embedding matrix to 0) with probability 0.1 for all regularized models except Gal, where we used Table 3 : Perplexity on the PTB word-level validation and test sets.

All results are from our own experiments.

probability 0.2 as specified in their paper.

More hyperparameter details are given in Appendix D.

We show in Table 3 that WD+Flipout outperforms the other methods with respect to both validation and test perplexity.

In Appendix E.4, we show that WD+Flipout yields significant variance reduction for large mini-batches, and that when training with batches of size 8192, it converges faster than WD.

Theorem 2 and FIG0 suggest that the variance reduction effect of flipout is more pronounced in the large mini-batch regime.

In this section, we train a Bayesian neural network with mini-batches of size 8192 and show that flipout speeds up training in terms of the number of iterations.

We trained the FC and ConvLe networks from Section 4.1 using Bayes by Backprop BID0 .

Since our primary focus is optimization, we focus on the training loss, shown in FIG2 for FC, we compare flipout with shared perturbations and the LRT; for ConvLe, we compare only to shared perturbations since the LRT does not give an unbiased gradient estimator.

We found that flipout converged in about 3 times fewer iterations than shared perturbations for both models, while achieving comparable performance to the LRT for the FC model.

Because flipout is roughly twice as expensive as shared perturbations (see Section 3.1), this corresponds to a 1.5x speedup overall.

Curves for the training and test error are given in Appendix E.2.

ES typically runs on multiple CPU cores.

The challenge in making ES GPU-friendly is that each sample requires computing a separate weight perturbation, so traditionally each worker can only generate one sample at a time.

In Section 3.1, we showed that ES with flipout allows each worker to evaluate a batch of perturbations, which can be done efficiently on a GPU.

However, flipout induces correlations between the samples, so we investigated whether these correlations cause a slowdown in training relative to fully independent perturbations (which we term "IdealES").

In this section, we show empirically that flipout ES is just as sample-efficient as IdealES, and consequently one can obtain significantly higher throughput per unit cost using flipout ES on a GPU.The ES gradient defined in Eqn.

1 has high variance, so a large number of samples are generally needed before applying an update.

We found that 5,000 samples are needed to achieve stable performance in the supervised learning tasks.

Standard ES runs the forward pass 5,000 times with independent weight perturbations, which sees little benefit to using a GPU over a CPU.

FlipES allows the same number of samples to be evaluated using a much smaller number of explicit perturbations.

Throughout the experiments, we ran flipout with mini-batches of size 40 (i.e. N = 40 in Eqn.

5).We compared IdealES and FlipES with a fully connected network (FC) on the MNIST dataset.

FIG2 shows that we incur no loss in performance when using pseudo-independent noise.

Next, we compared FlipES and cpuES (using 40 CPU cores) in terms of the per-update time with respect to the model size.

The result (in Appendix E.3) shows that FlipES scales better because it runs on the GPU.

Finally, we compared FlipES and the backpropagation algorithm on both FC and ConvLe.

FIG2 and FIG2 show that FlipES achieves data efficiency comparable with the backpropagation algorithm.

IdealES has a much higher computational cost than backpropagation, due to the large number of forward passes.

FlipES narrows the computational gap between them.

Although ES is more expensive than backpropagation, it can be applied to models which are not fully differentiable, such as models with a discrete loss (e.g., accuracy or BLEU score) or with stochastic units.

We have introduced flipout, an efficient method for decorrelating the weight gradients between different examples in a mini-batch.

We showed that flipout is guaranteed to reduce the variance compared with shared perturbations.

Empirically, we demonstrated significant variance reduction in the large batch setting for a variety of network architectures, as well as significant speedups in training time.

We showed that flipout outperforms dropout-based methods for regularizing LSTMs.

Flipout also makes it practical to apply GPUs to evolution strategies, resulting in substantially increased throughput for a given computational cost.

We believe flipout will make weight perturbations practical in the large batch setting favored by modern accelerators such as Tensor Processing Units (Jouppi et al., 2017) .

DISPLAYFORM0 In this section, we provide the proof of Theorem 2 (Variance Decomposition Theorem).Proof.

We use the notations from Section 3.2.

Let x, x denote two training examples from the mini-batch B, and ∆W, ∆W denote the weight perturbations they received.

We begin with the decomposition into data and estimation terms (Eqn.

6), which we repeat here for convenience: DISPLAYFORM1 The data term from Eqn.

13 can be simplified: DISPLAYFORM2 We break the estimation term from Eqn.

13 into variance and covariance terms: DISPLAYFORM3 We now separately analyze the cases of fully independent perturbations, shared perturbations, and flipout.

Fully independent perturbations.

If the perturbations are fully independent, the second term in Eqn.

15 disappears.

Hence, combining Eqns.

13, 14, and 15, we are left with DISPLAYFORM4 which is just α/N .

Recall that we reformulate the shared perturbations in terms of first sampling ∆W , and then letting ∆W = ∆W • rs , where r and s are random sign vectors shared by the whole batch.

Using the Law of Total Variance, we break the second term in Eqn.

15 into a part that comes from sampling ∆W and a part that comes from sampling r and s. DISPLAYFORM0 Since the perturbations are shared, ∆W = ∆W , so this can be simplified slightly to: DISPLAYFORM1 Plugging these two terms into the second term of Eqn.

15 yields

Here, we provide details of the network configurations used for our experiments (Section 4).The FC network is a 3-layer fully-connected network with 512-512-10 hidden units.

ConvLe is a LeNet-like network BID14 where the first two layers are convolutional with 32 and 64 filters of size [5, 5] , and use ReLU non-linearities.

A 2 × 2 max pooling layer follows after each convolutional layer.

Dimensionality reduction only takes place in the pooling layer; the stride for pooling is two and padding is used in the convolutional layers to keep the dimension.

Two fully-connected layers with 1024 and 10 hidden units are used to produce the classification result.

ConVGG is based on the VGG16 network BID28 .

We modified the last fully connected layer to have 10 output dimensions for our experiments on CIFAR-10.

We didn't use batch normalization for the variance reduction experiment since it introduces extra stochasticity.

The architectures used for the LSTM experiments are described in Section 4.2.

The hyperparameters used for the language modelling experiments are provided in Appendix D.

Given a network architecture, we compute the empirical stochastic gradient update variance as follows.

We start with a moderately pre-trained model, such as a network with 85% training accuracy on MNIST.

Without updating the parameters, we obtain the gradients of all the weights by performing a feed-forward pass, that includes sampling ∆W , R, and S, followed by backpropagation.

The gradient variance of each weight is computed by repeating this procedure 200 times in the experiments.

Let Var lj denote the estimate of the gradient variance of weight j in layer l. We compute the gradient variance as follows: DISPLAYFORM0 where g i lj is the gradient received by weight j in layer l. We estimate the variance of the gradients in layer l by averaging the variances of the weights in that layer,Ṽ = 1 |J| j Var lj .

In order to compute a confidence interval on the gradient variance estimate, we repeat the above procedure 50 times, yielding a sequence of average variance estimates, V 1 , . . .

, V 50 .

FIG0 , we compute the 90% confidence intervals of the variance estimates with a t-test.

For ConVGG, multiple GPUs were needed to run the variance reduction experiment with large mini-batch sizes (such as 4096 and 8192).

In such cases, it is computationally efficient to generate independent weight perturbations on different GPUs.

However, since our aim was to understand the effects of variance reduction independent of implementation, we shared the base perturbation among all GPUs to produce the plot shown in FIG0 .

We show in Appendix E that flipout yields lower variance even when we sample independent perturbations on different GPUs.

For the LSTM variance reduction experiments, we used the two-layer LSTM described in Section 4.2, trained for 3 epochs on the word-level Penn Treebank dataset.

FIG0 , we split large mini-batches (size 128 and higher) into sub-batches of size 64; we sampled one base perturbation ∆W that was shared among all sub-batches, and we sampled independent R and S matrices for each sub-batch.

Long Short-Term Memory networks (LSTMs) are defined by the following equations: DISPLAYFORM0 where i t , f t , and o t are the input, forget, and output gates, respectively, g t is the candidate update, and • denotes elementwise multiplication.

Naïve application of dropout on the hidden state of an LSTM is not effective, because it leads to significant memory loss over long sequences.

Several approaches have been proposed to regularize the recurrent connections, based on applying dropout to specific terms in the LSTM equations.

BID27 propose to drop the cell update vector, with a dropout mask d t sampled either per-step or per-sequence: DISPLAYFORM1 Gal & Ghahramani FORMULA3 BID12 propose to zone out units rather than dropping them; the hidden state and cell values are either stochastically updated or maintain their previous value: DISPLAYFORM2 , with zoneout masks d

For the word-level models (Table 3) , we used gradient clipping threshold 0.25 and the following hyperparameters:• For Gal & Ghahramani (2016), we used variational dropout with the parameters given in their paper: 0.35 dropout probability on inputs and outputs, 0.2 hidden state dropout, and 0.2 embedding dropout.• For BID27 , we used 0.1 embedding dropout, 0.5 dropout on inputs and outputs, and 0.3 dropout on cell updates, with per-step mask sampling.• For BID12 , we used 0.1 embedding dropout, 0.5 dropout on inputs and outputs, and cell and hidden state zoneout probabilities of 0.25 and 0.025, respectively.

• For WD BID17 , we used the parameters given in their paper: 0.1 embedding dropout, 0.4 dropout probability on inputs and outputs, and 0.3 dropout probability on the output between layers (the same masks are used for each step of a sequence).

We use 0.5 probability for DropConnect applied to the hidden-to-hidden weight matrices.• For WD+Flipout, we used the same parameters as BID17 , given above, but we regularized the hidden-to-hidden weight matrices with the variant of flipout described in Section 2.1, which implements DropConnect with probability 0.5.For the character-level models (Table 2) , we used orthogonal initialization for the LSTM weight matrices, gradient clipping threshold 1, and did not use input or output dropout.

The input characters were represented as one-hot vectors.

We used the following hyperparameters for each model:• For recurrent dropout BID27 , we used 0.25 dropout probability on the cell state, and per-step mask sampling.• For Zoneout BID12 , we used 0.5 and 0.05 for the cell and hidden state zoneout probabilities, respectively.

• For the variational LSTM BID3 , we used 0.25 hidden state dropout.•

For the flipout and shared perturbation LSTMs, we sampled Gaussian noise with σ = 1 for the hidden-to-hidden weight matrix.

As discussed in Appendix B, training on multiple GPUs naturally induces independent noise for each sub-batch.

FIG5 shows that flipout still achieves lower variance than shared perturbations in such cases.

When estimating the variance with mini-batch size 8192, running on four GPUs naturally induces four independent noise samples, for each sub-batch of size 2048; this yields lower variance than using a single noise sample.

Similarly, for mini-batch size 4096, two independent noise samples are generated on separate GPUs.

E.2 LARGE BATCH TRAINING WITH FLIPOUT FIG6 shows the training and test error for the large mini-batch experiments described in Section 4.3.

For both FC and ConvLe networks, we used the Adam optimizer with learning rate 0.003.

We downscaled the KL term by a factor of 10 to achieve higher accuracy.

While FIG2 shows that flipout converges faster than shared perturbations, FIG6 shows that flipout has the same generalization ability as shared perturbations (the faster convergence doesn't result in overfitting).

The variance reduction offered by flipout allows us to use DropConnect BID30 efficiently in a large mini-batch setting.

Here, we use flipout to implement DropConnect as described in Section 2.1, and use it to regularize an LSTM word-level language model.

We used the LSTM architecture proposed by BID17 , which has 400-dimensional word embedddings and three layers with hidden dimension 1150.

Following BID17 , we tied the weights of the embedding layer and the decoder layer.

BID17 use DropConnect to regularize the hidden-to-hidden weight matrices, with a single mask shared for all examples in a batch.

We used flipout to achieve a different DropConnect mask per example.

We applied WD+Flipout to both the hidden-to-hidden (h2h) and input-to-hidden (i2h) weight matrices, and compared to the model from BID17 , which we call WD (for weight-dropped LSTM), with DropConnect applied to both h2h and i2h.

Both models use embedding dropout 0.1, output dropout 0.4, and have DropConnect probability 0.5 for the i2h and h2h weights.

Both models were trained using Adam with learning rate 0.001.

FIG8 compares the variance of the gradients of the first-layer hidden-to-hidden weights between WD and WD+Flipout, and shows that flipout achieves significant variance reduction for mini-batch sizes larger than 256.

FIG9 shows the training curves of both models with batch size 8192.

We see that WD+Flipout converges faster than WD, and achieves a lower training perplexity, showcasing the optimization benefits of flipout in large mini-batch settings.

Training curves for WD and WD+Flipout, with batch size 8192.

<|TLDR|>

@highlight

We introduce flipout, an efficient method for decorrelating the gradients computed by stochastic neural net weights within a mini-batch by implicitly sampling pseudo-independent weight perturbations for each example.
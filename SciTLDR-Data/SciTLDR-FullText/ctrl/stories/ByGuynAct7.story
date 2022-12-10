Bayesian inference is known to provide a general framework for incorporating prior knowledge or specific properties into machine learning models via carefully choosing a prior distribution.

In this work, we propose a new type of prior distributions for convolutional neural networks, deep weight prior (DWP), that exploit generative models to encourage a specific structure of trained convolutional filters e.g., spatial correlations of weights.

We define DWP in the form of an implicit distribution and propose a method for variational inference with such type of implicit priors.

In experiments, we show that DWP improves the performance of Bayesian neural networks when training data are limited, and initialization of weights with samples from DWP accelerates training of conventional convolutional neural networks.

Bayesian inference is a tool that, after observing training data, allows to transforms a prior distribution over parameters of a machine learning model to a posterior distribution.

Recently, stochastic variational inference (Hoffman et al., 2013 ) -a method for approximate Bayesian inference -has been successfully adopted to obtain a variational approximation of a posterior distribution over weights of a deep neural network .

Currently, there are two major directions for the development of Bayesian deep learning.

The first direction can be summarized as the improvement of approximate inference with richer variational approximations and tighter variational bounds BID4 .

The second direction is the design of probabilistic models, in particular, prior distributions, that widen the scope of applicability of the Bayesian approach.

Prior distributions play an important role for sparsification BID25 , quantization and compression BID6 of deep learning models.

Although these prior distributions proved to be helpful, they are limited to fully-factorized structure.

Thus, the often observed spatial structure of convolutional filters cannot be enforced with such priors.

Convolutional neural networks are an example of the model family, where a correlation of the weights plays an important role, thus it may benefit from more flexible prior distributions.

Convolutional neural networks are known to learn similar convolutional kernels on different datasets from similar domains BID31 BID39 .

Based on this fact, within a specific data domain, we consider a distribution of convolution kernels of trained convolutional networks.

In the rest of the paper, we refer to this distribution as the source kernel distribution.

Our main assumption is that within a specific domain the source kernel distribution can be efficiently approximated with convolutional kernels of models that were trained on a small subset of problems from this domain.

For example, given a specific architecture, we expect that kernels of a model trained on notMNIST dataset -a dataset of grayscale images -come from the same distribution as kernels of the model trained on MNIST dataset.

In this work, we propose a method that estimates the source kernel distribution in an implicit form and allows us to perform variational inference with the specific type of implicit priors.

Our contributions can be summarized as follows:1.

We propose deep weight prior, a framework that approximates the source kernel distribution and incorporates prior knowledge about the structure of convolutional filters into the prior distribution.

We also propose to use an implicit form of this prior (Section 3.1).2.

We develop a method for variational inference with the proposed type of implicit priors (Section 3.2).3.

In experiments (Section 4), we show that variational inference with deep weight prior significantly improves classification performance upon a number of popular prior distributions in the case of limited training data.

We also find that initialization of conventional convolution networks with samples from a deep weight prior leads to faster convergence and better feature extraction without training i.e., using random weights.

In Bayesian setting, after observing a dataset D = {x 1 , . . . , x N } of N points, the goal is to transform our prior knowledge p(ω) of the unobserved distribution parameters ω to the posterior distribution p(ω | D).

However, computing the posterior distribution through Bayes rule DISPLAYFORM0 may involve computationally intractable integrals.

This problem, nonetheless, can be solved approximately.

Variational Inference (Jordan et al., 1999) is one of such approximation methods.

It reduces the inference to an optimization problem, where we optimize parameters θ of a variational approximation q θ (ω), so that KL-divergence between q θ (ω) and p(ω | D) is minimized.

This divergence in practice is minimized by maximizing the variational lower bound L(θ) of the marginal log-likelihood of the data w.r.t parameters θ of the variational approximation q θ (W ).

DISPLAYFORM1 where DISPLAYFORM2 The variational lower bound L(θ) consists of two terms: 1) the (conditional) expected log likelihood L D , and 2) the regularizer DISPLAYFORM3 However, in case of intractable expectations in equation 1 neither the variational lower bound L(θ) nor its gradients can be computed in a closed form.

Recently, Kingma & Welling (2013) and BID28 proposed an efficient mini-batch based approach to stochastic variational inference, so-called stochastic gradient variational Bayes or doubly stochastic variational inference.

The idea behind this framework is reparamtetrization, that represents samples from a parametric distribution q θ (ω) as a deterministic differentiable function ω = f (θ, ) of parameters θ and an (auxiliary) noise variable ∼ p( ).

Using this trick we can efficiently compute an unbiased stochastic gradient ∇ θ L of the variational lower bound w.r.t the parameters of the variational approximation.

Bayesian Neural Networks.

The stochastic gradient variational Bayes framework has been applied to approximate posterior distributions over parameters of deep neural networks .

We consider a discriminative problem, where dataset D consists of N object-label pairs DISPLAYFORM4 .

For this problem we maximize the variational lower bound L(θ) with respect to parameters θ of a variational approximation q θ (W ): DISPLAYFORM5 where W denotes weights of a neural network, q θ (W ) is a variational distribution, that allows reparametrization (Kingma & Welling, 2013; BID7 and p(W ) is a prior distribution.

In the simplest case q θ (W ) can be a fully-factorized normal distribution.

However, more expressive variational approximations may lead to better quality of variational inference BID38 .

Typically, Bayesian neural networks use fully-factorized normal or log-uniform priors .Variational Auto-encoder.

Stochastic gradient variational Bayes has also been applied for building generative models.

The variational auto-encoder proposed by Kingma & Welling (2013) maximizes a variational lower bound L(θ, φ) on the marginal log-likelihood by amortized variational inference: DISPLAYFORM6 where an inference model q θ (z i | x i ) approximates the posterior distribution over local latent variables z i , reconstruction model p φ (x i | z i ) transforms the distribution over latent variables to a conditional distribution in object space and a prior distribution over latent variables p(z i ).

The vanilla VAE defines q θ (z | x), p φ (x | z), p(z) as fully-factorized distributions, however, a number of richer variational approximations and prior distributions have been proposed BID27 BID11 BID33 .

The approximation of the data distribution can then be defined as an intractable integral p(x) ≈ p φ (x | z)p(z) dz which we will refer to as an implicit distribution.

In this section, we introduce the deep weight prior -an expressive prior distribution that is based on generative models.

This prior distribution allows us to encode and favor the structure of learned convolutional filters.

We consider a neural network with L convolutional layers and denote parameters of l-th convolutional layer as w l ∈ R I l ×O l ×H l ×W l , where I l is the number of input channels, O l is the number of output channels, H l and W l are spatial dimensions of kernels.

Parameters of the neural network are denoted as W = (w 1 , . . .

w L ).

A variational approximation q θ (W ) and a prior distribution p(W ) have the following factorization over layers, filters and channels: DISPLAYFORM0 where w l ij ∈ R H l ×W l is a kernel of j-th channel in i-th filter of l-th convolutional layer.

We also assume that q θ (W ) allows reparametrization.

The prior distribution p(W ), in contrast to popular prior distributions, is not factorized over spatial dimensions of the filters H l , W l .For a specific data domain and architecture, we define the source kernel distribution -the distribution of trained convolutional kernels of the l-th convolutional layer.

The source kernel distribution favors learned kernels, and thus it is a very natural candidate to be the prior distribution p l (w l ij ) for convolutional kernels of the l-th layer.

Unfortunately, we do not have access to its probability density function (p.d.f.) , that is needed for most approximate inference methods e.g., variational inference.

Therefore, we assume that the p.d.f.

of the source kernel distribution can be approximated using kernels of models trained on external datasets from the same domain.

For example, given a specific architecture, we expect that kernels of a model trained on CIFAR-100 dataset come from the same distribution as kernels of the model trained on CIFAR-10 dataset.

In other words, the p.d.f.

of the source kernel distribution can be approximated using a small subset of problems from a specific data domain.

In the next subsection, we propose to approximate this intractable probability density function of the source kernel distribution using the framework of generative models.

Require: variational approximations q(w | θ l ij ) and reverse models r(z | w; ψ l ) Require: reconstruction models p(w | z; φ l ), priors for auxiliary variables p l (z)while not converged dô M ← mini-batch of objects form dataset D w l ij ← sample weights from q(w|θ l ij ) with reparametrization z l ij ← sample auxiliary variables from r(z |ŵ DISPLAYFORM0 Update parameters θ and ψ using gradientĝ and a stochastic optimization algorithm end while return Parameters θ, ψ

In this section, we discuss explicit and implicit approximationsp l (w) of the probability density function p l (w) of the source kernel distribution of l-th layer.

We assume to have a trained convolutional neural network, and treat kernels from the l-th layer of this network w l ij ∈ R H l ×W l as samples from the source kernel distribution of l-th layer p l (w).Explicit models.

A number of approximations allow us to evaluate probability density functions explicitly.

Such families include but are not limited to Kernel Density Estimation (Silverman, 1986) , Normalizing Flows BID27 BID5 and PixelCNN (van den Oord et al., 2016) .

For these families, we can estimate the KL-divergence D KL (q(w | θ l ij ) p l (w)) and its gradients without a systematic bias, and then use them for variational inference.

Despite the fact that these methods provide flexible approximations, they usually demand high memory or computational cost ).Implicit models.

Implicit models, in contrast, can be more computationally efficient, however, they do not provide access to an explicit form of probability density functionp l (w).

We consider an approximation of the prior distribution p l (w) in the following implicit form: DISPLAYFORM0 where a conditional distribution p(w | z; φ l ) is an explicit parametric distribution and p l (z) is an explicit prior distribution that does not depend on trainable parameters.

Parameters of the conditional distribution p(w | z; φ l ) can be modeled by a differentiable function g(z; φ l ) e.g. neural network.

Note, that while the conditional distribution p(w | z; φ l ) usually is a simple explicit distribution, e.g. fully-factorized Gaussian, the marginal distributionp l (w) is generally a more complex intractable distribution.

Parameters φ l of the conditional distribution p(w | z; φ l ) can be fitted using the variational autoencoder framework.

In contrast to the methods with explicit access to the probability density, variational auto-encoders combine low memory cost and fast sampling.

However, we cannot obtain an unbiased estimate the logarithm of probability density function logp l (w) and therefore cannot build an unbiased estimator of the variational lower bound (equation 3).

In order to overcome this limitation we propose a modification of variational inference for implicit prior distributions.

Stochastic variational inference approximates a true posterior distribution by maximizing the variational lower bound L(θ) (equation 1), which includes the KL-divergence D KL (q(W ) p(W )) between a variational approximation q θ (W ) and a prior distribution p(W ).

In the case of simple prior and variational distributions (e.g. Gaussian), the KL-divergence can be computed in a closed form or unbiasedly estimated.

Unfortunately, it does not hold anymore in case of an implicit prior dis- DISPLAYFORM0 In that case, the KL-divergence cannot be estimated without bias.

DISPLAYFORM1 (a) Learning of DWP with VAE (b) Learned filters (c) Samples from DWP Figure 1 : At subfig.

1(a) we show the process of learning the prior distribution over kernels of one convolutional layer.

First, we train encoder r(z | w; φ l ) and decoder p(w | z; ψ l ) with VAE framework.

Then, we use the decoder to construct the priorp l (w).

At subfig.

1(b) we show a batch of learned kernels of shape 7×7 form the first convolutional layer of a CNN trained on NotMNIST dataset, at subfig.

1(c) we show samples form the deep weight prior that is learned on these kernels.

To make the computation of the variational lower bound tractable, we introduce an auxiliary lower bound on the KL-divergence.

KL-divergence: DISPLAYFORM2 where r(z | w; ψ l ) is an auxiliary inference model for the prior of l-th layerp l (w), The final auxiliary variational lower bound has the following form: DISPLAYFORM3 The lower bound L aux is tight if and only if the KL-divergence between the auxiliary reverse model and the intractable posterior distribution over latent variables z given w is zero (Appendix A).In the case when q θ (w), p(w | z; φ l ) and r(z | w; ψ l ) are explicit parametric distributions which can be reparametrized, we can perform an unbiased estimation of a gradient of the auxiliary variational lower bound L aux (θ, ψ) (equation 8) w.r.t.

parameters θ of the variational approximation q θ (W ) and parameters ψ of the reverse models r(z | w; ψ l ).

Then we can maximize the auxiliary lower bound w.r.t.

parameters of the variational approximation and the reversed models L aux (θ, ψ) → max θ,ψ .

Note, that parameters φ of the prior distributionp(W ) are fixed during variational inference, in contrast to the Empirical Bayesian framework BID19 .

Algorithm 1 describes stochastic variational inference with an implicit prior distribution.

In the case when we can calculate an entropy H(q) or the divergence D KL (r(z | w; ψ l ) p l (z)) explicitly, the variance of the estimation of the gradient ∇L aux (θ, ψ) can be reduced.

This algorithm can also be applied to an implicit prior that is defined in the form of Markov chain: DISPLAYFORM4 where p(z t+1 | z t ) is a transition operator , see Appendix A. We provide more details related to the form of p(w | z; φ l ), r(z | w; ψ l ) and p l (z) distributions in Section 4.

In this subsection we explain how to train deep weight prior models for a particular problem.

We present samples from learned prior distribution at Figure 1

Source datasets of kernels.

For kernels of a particular convolutional layer l, we train an individual prior distributionp l (w) = p(w | z; φ l )p l (z) dz.

First, we collect a source dataset of the kernels of the l-th layer of convolutional networks (source networks) trained on a dataset from a similar domain.

Then, we train reconstruction models p(w | z; φ l ) on these collected source datasets for Figure 2: For different sizes of training set of MNIST and CIFAR-10 datasets, we demonstrate the performance of variational inference with a fully-factorized variational approximation with three different prior distributions: deep weight prior (dwp), log-uniform, and standard normal.

We found that variational inference with a deep weight prior distribution achieves better mean test accuracy comparing to learning with standard normal and log-uniform prior distributions.

each layer, using the framework of variational auto-encoder (Section 2).

Finally, we use the reconstruction models to construct priorsp l (w) as shown at Figure 1(a) .

In our experiments, we found that regularization is crucial for learning of source kernels.

It helps to learn more structured and less noisy kernels.

Thus, source models were learned with L 2 regularization.

We removed kernels of small norm as they have no influence upon predictions , but they make learning of the generative model more challenging.

Reconstruction and inference models for prior distribution.

In our experiments, inference models r(z | w; ψ l ) are fully-factorized normal-distributions N (z | µ ψ l (w), diag(σ 2 ψ l (w))), where parameters µ ψ l (w) and σ ψ l (w) are modeled by a convolutional neural network.

The convolutional part of the network is constructed from several convolutional layers that are alternated with ELU BID3 and max-pooling layers.

Convolution layers are followed by a fully-connected layer with 2 · z l dim output neurons, where z l dim is a dimension of the latent representation z, and is specific for a particular layer.

Reconstruction models p(w | z; φ l ) are also modeled by a fully-factorized normal-distribution N (w | µ φ l (z), diag(σ 2 φ l (z))) and network for µ φ l and σ 2 φ l has the similar architecture as the inference model, but uses transposed convolutions.

We use the same architectures for all prior models, but with slightly different hyperparameters, due to different sizes of kernels.

We also use fullyfactorized standard Gaussian prior p l (z i ) = N (z i | 0, 1) for latent variables z i .

We provide a more detailed description at Appendix F.

We apply deep weight prior to variational inference, random feature extraction and initialization of convolutional neural networks.

In our experiments we used MNIST (LeCun et al., 1998), NotM-NIST BID1 , CIFAR-10 and CIFAR-100 BID13 ) datasets.

Experiments were implemented 1 using PyTorch BID26 .

For optimization we used Adam (Kingma & Ba, 2014) with default hyperparameters.

We trained prior distributions on a number of source networks which were learned from different initial points on NotMNIST and CIFAR-100 datasets for MNIST and CIFAR-10 experiments respectively.

In this experiment, we performed variational inference over weights of a discriminative convolutional neural network (Section 3) with three different prior distributions for the weights of the convolutional layers: deep weight prior (dwp), standard normal and log-uniform .

We did not perform variational inference over the parameters of the fully connected layers.

We used We study the influence of initialization of convolutional filters on the performance of random feature extraction.

In the experiment, the weights of convolutional filters were initialized randomly and fixed.

The initializations were sampled from deep weight prior (dwp), learned filters (filters) and samples from Xavier distribution (xavier).

We performed the experiment for different size of the model, namely, to obtain models of different sizes we scaled a number of filters in all convolutional layers linearly by k. For every size of the model, we averaged results by 10 runs.

We found that initialization with samples from deep weight prior and learned filters significantly outperform Xavier initialization.

Although, initialization with filters performs marginally better, dwp does not require to store a potentially big set of all learned filters.

We present result for MNIST and CIFAR-10 datasets at sub figs. 3(a) and 3(b) respectively.

a fully-factorized variational approximation with additive parameterization proposed by and local reparametrization trick proposed by .

Note, that our method can be combined with more complex variational approximations, in order to improve variational inference.

On MNIST dataset we used a neural network with two convolutional layers with 32, 128 filters of shape 7 × 7, 5 × 5 respectively, followed by one linear layer with 10 neurons.

On the CIFAR dataset we used a neural network with four convolutional layers with 128, 256, 256 filters of shape 7 × 7, 5 × 5, 5 × 5 respectively, followed by two fully connected layers with 512 and 10 neurons.

We used a max-pooling layer BID22 After the first convolutional layer.

All layers were divided with leaky ReLU nonlinearities BID23 .At figure 2 we report accuracy for variational inference with different sizes of training datasets and prior distributions.

Variational inference with deep weight prior leads to better mean test accuracy, in comparison to log-uniform and standard normal prior distributions.

Note that the difference gets more significant as the training set gets smaller.

Convolutional neural networks produce useful features even if they are initialized randomly BID30 BID9 BID35 .

In this experiment, we study an influence of different random initializations of convolutional layers -that is fixed during training -on the performance of convolutional networks of different size, where we train only fully-connected layers.

We use three initializations for weights of convolutional layers: learned kernels, samples from deep weight prior, samples from Xavier distribution BID8 .

We use the same architectures as in Section 4.1.

We found that initializations with samples from deep weight prior and learned kernels significantly outperform the standard Xavier initialization when the size of the network is small.

Initializations with samples form deep weight prior and learned filters perform similarly, but with deep weight prior we can avoid storing all learned kernels.

At FIG1 , we show results on MNIST and CIFAR-10 for different network sizes, which are obtained by scaling the number of filters by k.

Deep learning models are sensitive to initialization of model weights.

In particular, it may influence the speed of convergence or even a local minimum a model converges to.

In this experiment, we study the influence of initialization on the convergence speed of two settings: a variational auto- Figure 4: We found that initialization of weights of the models with deep weight priors or learned filters significantly increases the training speed, comparing to Xavier initialization.

At subplot 4(a) we report a variational lower bound for variational auto-encoder, at subplots 4(b) and 4(c) we report accuracy for convolution networks on MINTS and CIFAR-10.encoder on MNIST, and convolutional networks on MNIST and CIFAR-10.

We compare three different initializations of weights of conventional convolutional layers: learned filters, samples from deep weight prior and samples form Xavier distribution.

Figure 4 provides the results for a convolutional variational auto-encoder trained on MNIST and for a convolutional classification network trained on CIFAR-10 and MNIST.

We found that deep weight prior and learned filters initializations perform similarly and lead to significantly faster convergence comparing to standard Xavier initialization.

Deep weight prior initialization however does not require us to store a possibly large set of filters.

Also, we plot samples from variational auto-encoders at a different training steps Appendix E.

The recent success of transfer learning BID39 shows that convolutional networks produce similar convolutional filters while being trained on different datasets from the same domain e.g. photo-realistic images.

In contrast to Bayesian techniques BID12 , these methods do not allow to obtain a posterior distribution over parameters of the model, and in most cases, they require to store convolutional weights of pre-trained models and careful tuning of hyperparameters.

The Bayesian approach provides a framework that incorporates prior knowledge about weights of a machine learning model by choosing or leaning a prior distribution p(w).

There is a huge amount of works on prior distributions for Bayesian inference BID19 BID37 , where empirical Bayes -an approach that tunes parameters of the prior distribution on the training data -plays an important role BID19 .

These methods are widely used for regularization and sparsification of linear models BID0 , however, applied to deep neural networks , they do not take into account the structure of the model weights, e.g. spatial correlations, which does matter in case of convolutional networks.

Our approach allows to perform variational inference with an implicit prior distribution, that is based on previously observed convolutional kernels.

In contrast to an empirical Bayes approach, parameters φ of a deep weight prior (equation 6) are adjusted before the variational inference and then remain fixed.

Prior to our work implicit models have been applied to variational inference.

That type of models includes a number of flexible variational distributions e.g., semi-implicit BID38 and Markov chain BID14 approximations.

Implicit priors have been used for introducing invariance properties BID24 , improving uncertainty estimation BID18 and learning meta-representations within an empirical Bayes approach (Karaletsos et al., 2018) .In this work, we propose to use an implicit prior distribution for stochastic variational inference and develop a method for variational inference with the specific type of implicit priors.

The approach also can be generalized to prior distributions in the form of a Markov chain.

We show how to use this framework to learn a flexible prior distribution over kernels of Bayesian convolutional neural networks.

In this work we propose deep weight prior -a framework for designing a prior distribution for convolutional neural networks, that exploits prior knowledge about the structure of learned convolutional filters.

This framework opens a new direction for applications of Bayesian deep learning, in particular to transfer learning.

Factorization.

The factorization of deep weight prior does not take into account inter-layer dependencies of the weights.

Although a more complex factorization might be a better fit for CNNs.

Accounting inter-layer dependencies may give us an opportunity to recover a distribution in the space of trained networks rather than in the space of trained kernels.

However, estimating prior distributions of more complex factorization may require significantly more data and computational budget, thus the topic needs an additional investigation.

Inference.

An alternative to variational inference with auxiliary variables is semi-implicit variational inference BID38 .

The method was developed only for semiimplicit variational approximations, and only the recent work on doubly semi-implicit variational inference generalized it for implicit prior distributions .

These algorithms might provide a better way for variational inference with a deep weight prior, however, the topic needs further investigation.

DISPLAYFORM0 is a transition operator, and z = (z 0 , . . . , z T ) .

Unfortunately, gradients of L cannot be efficiently estimated, but we construct a tractable lower bound L aux for L: DISPLAYFORM1 Inequality 13 has a very natural interpretation.

The lower bound L aux is tight if and only if the KLdivergence between the auxiliary reverse model and the posterior intractable distribution p(z | w) is zero.

The deep weight prior (Section 3) is a special of Markov chain prior for T = 0 and p(w) = p(w | z)p(z)dz.

The auxiliary variational bound has the following form: DISPLAYFORM2 where the gradients in equation 14 can be efficiently estimated in case q(w), for explicit distributions q(w), p φ (w | z), r(z | w) that can be reparametrized.

During variational inference with deep weight prior (Algorithm 1) we optimize a new auxiliary lower bound L aux (θ, ψ) on the evidence lower bound L(θ).

However, the quality of such inference depends on the gap G(θ, ψ) between the original variational lower bound L(θ) and the variational lower bound in auxiliary space L aux (θ, ψ):

The gap G(θ, ψ) cannot be calculated exactly, but it can be estimated by using tighter but less computationally efficient lower bound.

We follow BID2 and construct tighter lower bound DISPLAYFORM0 The estimate L

(θ, ψ) converges to L(θ) with K goes to infinity BID2 .

We estimate the gap with K = 10000 as follows: DISPLAYFORM0 DISPLAYFORM1 The estimate of the gap, however, may be not very accurate and we consider it as a sanity check.

We examined a multivariate normal distributionp l (w) = N (w|µ l , Σ l ).

We used a closed-form maximum-likelihood estimation for parameters µ l , Σ l over source dataset of learned kernels for each layer.

We conducted the same experiment as in Section 4.1 for MNIST dataset for this gaussian prior, the results presented at FIG3 .

We found that the gaussian prior performs marginally worse than deep weight prior, log-uniform and standard normal.

The gaussian prior could find a bad local optima and fail to approximate potentially multimodal source distribution of learned kernels.

Conv, 64, 3 × 3 Conv, 128, 1 × 1 Conv, 32, 3 × 3 ConvT, 64, 3 × 3 Conv, 64, 3 × 3 ConvT, 128, 3 × 3 Conv, 64, 3 × 3 ConvT, 64, 3 × 3 Conv, 128, 3 × 3 ConvT, 128, 3 × 3 Conv, 64, 3 × 3 ConvT, 32, 3 × 3 Conv, 128, 3 × 3 ConvT, 64, 1 × 1 2 × Linear, z dim 2 × ConvT, 1, 1 × 1 2 × Linear, z dim 2 × Conv, 1, 1 × 1 = 260040 params = 304194 params = 56004 params = 56674 params Table 3 : Network Architectures for MNIST and CIFAR-10/CIFAR-100 datasets (Section 4).

Encoder5x5 Decoder5x5 Encoder7x7 Decoder7x7

<|TLDR|>

@highlight

The generative model for kernels of convolutional neural networks, that acts as a prior distribution while training on new datasets.

@highlight

A method for modeling convolutional neural networks using a Bayes method.

@highlight

Proposes the 'deep weight prior': the idea is to elicit a prior on an auxilary dataset and then use that prior over the CNN filters to jump start inference for a data set of interest.

@highlight

This paper explores learning informative priors for convolutional neural network models with similar problem domains by using autoencoders to obtain an expressive prior on the filtered weights of the trained networks.
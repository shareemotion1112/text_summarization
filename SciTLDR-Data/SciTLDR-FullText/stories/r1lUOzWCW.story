We investigate the training and performance of generative adversarial networks using the Maximum Mean Discrepancy (MMD) as critic, termed MMD GANs.

As our main theoretical contribution, we clarify the situation with bias in GAN loss functions raised by recent work: we show that gradient estimators used in the optimization process for both MMD GANs and Wasserstein GANs are unbiased, but learning a discriminator based on samples leads to biased gradients for the generator parameters.

We also discuss the issue of kernel choice for the MMD critic, and characterize the kernel corresponding to the energy distance used for the Cramér GAN critic.

Being an integral probability metric, the MMD benefits from training strategies recently developed for Wasserstein GANs.

In experiments, the MMD GAN is able to employ a smaller critic network than the Wasserstein GAN, resulting in a simpler and faster-training algorithm with matching performance.

We also propose an improved measure of GAN convergence, the Kernel Inception Distance, and show how to use it to dynamically adapt learning rates during GAN training.

Generative Adversarial Networks (GANs; BID10 provide a powerful method for general-purpose generative modeling of datasets.

Given examples from some distribution, a GAN attempts to learn a generator function, which maps from some fixed noise distribution to samples that attempt to mimic a reference or target distribution.

The generator is trained to trick a discriminator, or critic, which tries to distinguish between generated and target samples.

This alternative to standard maximum likelihood approaches for training generative models has brought about a rush of interest over the past several years.

Likelihoods do not necessarily correspond well to sample quality BID13 , and GAN-type objectives focus much more on producing plausible samples, as illustrated particularly directly by Danihelka et al. (2017) .

This class of models has recently led to many impressive examples of image generation (e.g. Huang et al., 2017a; Jin et al., 2017; Zhu et al., 2017) .GANs are, however, notoriously tricky to train (Salimans et al., 2016) .

This might be understood in terms of the discriminator class.

BID10 showed that, when the discriminator is trained to optimality among a rich enough function class, the generator network attempts to minimize the Jensen-Shannon divergence between the generator and target distributions.

This result has been extended to general f -divergences by Nowozin et al. (2016) .

According to BID1 , however, it is likely that both the GAN and reference probability measures are supported on manifolds within a larger space, as occurs for the set of images in the space of possible pixel values.

These manifolds might not intersect at all, or at best might intersect on sets of measure zero.

In this case, the Jensen-Shannon divergence is constant, and the KL and reverse-KL divergences are infinite, meaning that they provide no useful gradient for the generator to follow.

This helps to explain some of the instability of GAN training.

The lack of sensitivity to distance, meaning that nearby but non-overlapping regions of high probability mass are not considered similar, is a long-recognized problem for KL divergence-based discrepancy measures (e.g. Gneiting & Raftery, 2007, Section 4.2) .

It is natural to address this problem using Integral Probability Metrics (IPMs; Müller, 1997) : these measure the distance between probability measures via the largest discrepancy in expectation over a class of "well behaved" witness functions.

Thus, IPMs are able to signal proximity in the probability mass of the generator and reference distributions.

(Section 2 describes this framework in more detail.)

BID1 proposed to use the Wasserstein distance between distributions as the discriminator, which is an integral probability metric constructed from the witness class of 1-Lipschitz functions.

To implement the Wasserstein critic, Arjovsky et al. originally proposed weight clipping of the discriminator network, to enforce k-Lipschitz smoothness.

Gulrajani et al. (2017) improved on this result by directly constraining the gradient of the discriminator network at points between the generator and reference samples.

This new Wasserstein GAN implementation, called WGAN-GP, is more stable and easier to train.

A second integral probability metric used in GAN variants is the maximum mean discrepancy (MMD), for which the witness function class is a unit ball in a reproducing kernel Hilbert space (RKHS).

Generative adversarial models based on minimizing the MMD were first considered by Li et al. (2015) and Dziugaite et al. (2015) .

These works optimized a generator to minimize the MMD with a fixed kernel, either using a generic kernel on image pixels or by modeling autoencoder representations instead of images directly.

BID9 instead minimized the statistical power of an MMD-based test with a fixed kernel.

Such approaches struggle with complex natural images, where pixel distances are of little value, and fixed representations can easily be tricked, as in the adversarial examples of BID10 .Adversarial training of the MMD loss is thus an obvious choice to advance these methods.

Here the kernel MMD is defined on the output of a convolutional network, which is trained adversarially.

Recent notable work has made use of the IPM representation of the MMD to employ the same witness function regularization strategies as BID1 and Gulrajani et al. (2017) , effectively corresponding to an additional constraint on the MMD function class.

Without such constraints, the convolutional features are unstable and difficult to train BID9 .

Li et al. (2017b) essentially used the weight clipping strategy of Arjovsky et al., with additional constraints to encourage the kernel distribution embeddings to be injective.

1 In light of the observations by Gulrajani et al., however, we use a gradient constraint on the MMD witness function in the present work (see Sections 2.1 and 2.2).2 Bellemare et al. (2017) 's method, the Cramér GAN, also used the gradient constraint strategy of Gulrajani et al. in their discriminator network.

As we discuss in Section 2.3, the Cramér GAN discriminator is related to the energy distance, which is an instance of the MMD (Sejdinovic et al., 2013) , and which can therefore use a gradient constraint on the witness function.

Note, however, that there are important differences between the Cramér GAN critic and the energy distance, which make it more akin to the optimization of a scoring rule: we provide further details in Appendix A. Weight clipping and gradient constraints are not the only approaches possible: variance features (Mroueh et al., 2017) and constraints (Mroueh & Sercu, 2017) can work, as can other optimization strategies (Berthelot et al., 2017; Li et al., 2017a) .Given that both the Wasserstein distance and the MMD are integral probability metrics, it is of interest to consider how they differ when used in GAN training.

Bellemare et al. (2017) showed that optimizing the empirical Wasserstein distance can lead to biased gradients for the generator, and gave an explicit example where optimizing with these biased gradients leads the optimizer to incorrect parameter values, even in expectation.

They then claim that the energy distance does not suffer from these problems.

As our main theoretical contribution, we substantially clarify the bias situation in Section 3.

First, we show (Theorem 1) that the natural maximum mean discrepancy estimator, including the estimator of energy distance, has unbiased gradients when used "on top" of a fixed deep network representation.

The generator gradients obtained from a trained representation, however, will be biased relative to the desired gradients of the optimal critic based on infinitely many samples.

This situation is exactly analogous to WGANs: the generator's gradients with a fixed critic are unbiased, but gradients from a learned critic are biased with respect to the supremum over critics.

MMD GANs, though, do have some advantages over Wasserstein GANs.

Certainly we would not expect the MMD on its own to perform well on raw image data, since these data lie on a low dimensional manifold embedded in a higher dimensional pixel space.

Once the images are mapped through appropriately trained convolutional layers, however, they can follow a much simpler distribution with broader support across the mapped domain: a phenomenon also observed in autoencoders (Bengio et al., 2013) .

In this setting, the MMD with characteristic kernels BID4 shows strong discriminative performance between distributions.

To achieve comparable performance, a WGAN without the advantage of a kernel on the transformed space requires many more convolutional filters in the critic.

In our experiments (Section 5), we find that MMD GANs achieve the same generator performance as WGAN-GPs with smaller discriminator networks, resulting in GANs with fewer parameters and computationally faster training.

Thus, the MMD GAN discriminator can be understood as a hybrid model that plays to the strengths of both the initial convolutional mappings and the kernel layer that sits on top.

We begin with a review of the MMD and relate it to the loss functions used by other GAN variants.

Through its interpretation as an integral probability metric, we show that the gradient penalty of Gulrajani et al. (2017) applies to the MMD GAN.

We consider a random variable X with probability measure P, which we associate with the generator, and a second random variable Y with probability measure Q, which we associate with the reference sample that we wish to learn.

Our goal is to measure the distance from P to Q using samples drawn independently from each distribution.

The maximum mean discrepancy is a metric on probability measures BID6 , which falls within the family of integral probability metrics (Müller, 1997) ; this family includes the Wasserstein and Kolmogorov metrics, but not for instance the KL or χ 2 divergences.

Integral probability metrics make use of a class of witness functions to distinguish between P and Q, choosing the function with the largest discrepancy in expectation over P, Q, DISPLAYFORM0 The particular witness function class F determines the probability metric.

3 For example, the Wasserstein-1 metric is defined using the 1-Lipschitz functions, the total variation by functions with absolute value bounded by 1, and the Kolmogorov metric using the functions of bounded variation 1.

For more on this family of distances, see e.g. BID3 .In this work, our witness function class F will be the unit ball in a reproducing kernel Hilbert space H, with positive definite kernel k(x, x ).

The key aspect of a reproducing kernel Hilbert space is the reproducing property: for all f ∈ H, f (x) = f, k(x, ·) H .

We define the mean embedding of the probability measure P as the element µ P ∈ H such that E P f (X) = f, µ P H ; it is given by µ P = E X∼P k(·, X).

The maximum mean discrepancy (MMD) is defined as the IPM (1) with F the unit ball in H, MMD(P, Q; H) = sup DISPLAYFORM0 The witness function f * that attains the supremum has a straightforward expression (Gretton et al., 2012, Section 2.3), DISPLAYFORM1 DISPLAYFORM2 and an unbiased estimator of the squared MMD is (Gretton et al., 2012, Lemma 6) DISPLAYFORM3 When the kernel is characteristic BID4 BID5 , the embedding µ P is injective (i.e., associated uniquely with P).

Perhaps the best-known characteristic kernel is the exponentiated quadratic kernel, also known as the Gaussian RBF kernel, DISPLAYFORM4 Both the kernel and its derivatives decay exponentially, however, causing significant problems in high dimensions, and especially when used in gradient-based representation learning.

The rational quadratic kernel DISPLAYFORM5 with α > 0 corresponds to a scaled mixture of exponentiated quadratic kernels, with a Gamma(α, 1) prior on the inverse lengthscale (Rasmussen & Williams, 2006, Section 4.2) .

This kernel will be the mainstay of our experiments, as its tail behaviour is much superior to that of the exponentiated quadratic kernel; it is also characteristic.

The MMD has been a popular choice for the role of a critic in a GAN.

This idea was proposed simultaneously by Dziugaite et al. (2015) and Li et al. (2015) , with numerous recent follow-up works BID9 Liu, 2017; Li et al., 2017b; Bellemare et al., 2017) .

As a key strategy in these recent works, the MMD of (4) is not computed directly on the samples; rather, the samples first pass through a mapping function h, generally a convolutional network.

Note that we can think of this either as the MMD with kernel k on features h(x), or simply as the MMD with kernel κ(x, y) = k(h(x), h(y)).

The challenge is to learn the features h so as to maximize the MMD, without causing the critic to collapse to a trivial answer early in training.

Bearing in mind that the MMD is an integral probability metric, strategies developed for training the Wasserstein GAN critic can be directly adopted for training the MMD critic.

Li et al. (2017b) employed the weight clipping approach of BID1 , though they motivated it using different considerations.

Gulrajani et al. (2017) found a number of issues with weight clipping, however: it oversimplifies the loss functions given standard architectures, the gradient decays exponentially as we move up the network, and it seems to require the use of slower optimizers such as RMSProp rather than standard approaches such as Adam (Kingma & Ba, 2015) .It thus seems preferable to adopt Gulrajani et al.'s proposal of regularising the critic witness (3) by constraining its gradient norm to be nearly 1 along randomly chosen convex combinations of generator and reference points, αx i + (1 − α)y j for α ∼ Uniform(0, 1).

This was motivated by the observation that the Wasserstein witness satisfies this property (their Lemma 1), but perhaps its main benefit is one of regularization: if the critic function becomes too flat anywhere between the samples, the generator cannot easily follow its gradient.

We will thus follow this approach, as did Bellemare et al. (2017) , whose model we describe next.

5 By doing so, we implicitly change the definition of the distance being approximated; we leave study of the differences to future work.

By analogy, Liu et al. (2017) give some basic properties for the distance used by Gulrajani et al. (2017) .

Liu (2017) and Bellemare et al. (2017, Section 4) proposed to use the energy distance as the critic in an adversarial network.

The energy distance BID12 Lyons, 2013 ) is a measure of divergence between two probability measures, defined as

Many other GAN variants fall into the framework of IPMs (e.g. Mroueh et al., 2017; Mroueh & Sercu, 2017; Berthelot et al., 2017) .

Notably, although BID10 motivated GANs as estimating the Jensen-Shannon divergence, they can also be viewed as minimizing the IPM defined by the classifier family (Arora et al., 2017; Liu et al., 2017) , thus motivating applying the gradient penalty to original GANs (Fedus et al., 2018) .

Liu et al. (2017) in particular study properties of these distances.

The issue of biased gradients in GANs was brought to prominence by Bellemare et al. (2017, Section 3) , who showed bias in the gradients of the empirical Wasserstein distance for finite sample sizes, and demonstrated cases where this bias could lead to serious problems in stochastic gradient descent, even in expectation.

They then claimed that the energy distance used in the Cramér GAN critic does not suffer from these problems.

We will now both formalize and clarify these results.

First, Bellemare et al.'s proof that the gradient of the energy distance is unbiased was incomplete: the essential step in the reasoning, the exchange in the order of the expectations and the derivatives, is simply assumed.

8 We show that one can exchange the order of expectations and derivatives, under very mild assumptions about the distributions in question, the form of the network, and the kernel: Theorem 1.

Let G ψ : Z → X and h θ : X → R d be deep networks, with parameters ψ ∈ R m ψ and θ ∈ R m θ , of the form defined in Appendix C.1 and satisfying Assumptions C and D (in Appendix C.2).

This includes almost all feedforward networks used in practice, in particular covering convolutions, max pooling, and ReLU activations.

Let P be a distribution on X such that E[ X 2 ] exists, and likewise Z a distribution on Z such that E[ Z 2 ] exists.

P and Z need not have densities.

DISPLAYFORM0 be a kernel function satisfying the growth assumption Assumption E for some α ∈ [1, 2].

All kernels considered in this paper satisfy this assumption; see the discussion after Corollary 3.For µ-almost all (ψ, θ) ∈ R m ψ +m θ , where µ is the Lebesgue measure, the function DISPLAYFORM1 is differentiable at (ψ, θ), and moreover DISPLAYFORM2 Thus for µ-almost all (ψ, θ), DISPLAYFORM3 This result is shown in Appendix C, specifically as Corollary 3 to Theorem 5, which is a quite general result about interchanging expectations and derivatives of functions of deep networks.

The proof is more complex than a typical proof that derivatives and integrals can be exchanged, due to the non-differentiability of ReLU-like functions used in deep networks.

But this unbiasedness result is not the whole story.

In WGANs, the generator attempts to minimize the loss function DISPLAYFORM4 based on an estimateŴ(X, Y): first critic parameters θ are estimated on a "training set" X tr , Y tr , i.e. all points seen in the optimization process thus far, and then the distance is estimated on the remaining "test set" X te , Y te , i.e. the current minibatch, as DISPLAYFORM5 (After the first pass through the training set, these two sets will not be quite independent, but for large datasets they should be approximately so.)

Theorem 2 (in Appendix B.2) shows that this estimator W is biased; Appendix B.4 further gives an explicit numerical example.

This almost certainly implies that ∇ ψŴ is biased as well, by Theorem 4 (Appendix B.3).

9 Yet, for fixed θ, Corollary 1 shows that the estimator (13) has unbiased gradients; it is only the procedure which first selects a θ based on training samples and then evaluates (13) which is a biased estimator of (12).The situation with MMD GANs, including energy distance-based GANs, is exactly analogous.

We have (11): for almost all particular critic representations h θ , the estimator of MMD 2 is unbiased.

But the population divergence the generator attempts to minimize is actually DISPLAYFORM6 a distance previously studied by BID2 as well as Li et al. (2017b) .

An MMD GAN's effective estimator ofη is also biased by Theorem 2 (see particularly Appendix B.5); by Theorem 4, its gradients are also almost certainly biased.

In both cases, the bias vanishes as the selection of θ becomes better; in particular, no bias is introduced by the use of a fixed (and potentially small) minibatch size, but rather by the optimization procedure for θ and the total number of samples seen in training the discriminator.

Yet there is at least some sense in which MMD GANs might be considered "less biased" than WGANs.

Optimizing the generator parameters of a WGAN while holding the critic parameters fixed is not sensible: consider, for example, P a point mass at 0 ∈ R and Q a point mass at q ∈ R. If q > 0, an optimal θ might correspond to the witness function f (t) = t; if we hold this witness function f fixed, the optimal q is at −∞, rather than at the correct value of 0.

But if we hold an MMD GAN's critic fixed and optimize the generator, we obtain the GMMN model (Li et al., 2015; Dziugaite et al., 2015) .

Here, because the witness function still adapts to the observed pair of distributions, the correct distribution P = Q will always be optimal.

Bad solutions might also seem to be optimal, but they can never seem arbitrarily better.

Thus unbiased gradients of MMD 2 u might somehow be more meaningful to the optimization process than unbiased gradients of (13); exploring and formalizing this intuition is an intriguing area for future work.

One challenge in comparing GAN models, as we will do in the next section, is that quantitative comparisons are difficult.

Some insight can be gained by visually examining samples, but we also consider the following approaches to evaluate GAN methods.

Inception score This metric, proposed by Salimans et al. (2016) , is based on the classification output p(y | x) of the Inception model BID11 .

Defined as exp (E x KL(p(y | x) p(y))), it is highest when each image's predictive distribution has low entropy, but the marginal predictive distribution p(y) = E x p(y | x) has high entropy.

This score correlates somewhat with human judgement of sample quality on natural images, but it has some issues, especially when applied to domains which do not represent a variety of the types of classes in ImageNet.

In particular, it knows nothing about the desired distribution for the model.

The Fréchet Inception Distance, proposed by Heusel et al. (2017) , avoids some of the problems of Inception by measuring the similarity of the samples' representations in the Inception architecture (at the pool3 layer, of dimension 2048) to those of samples from the target distribution.

The FID fits a Gaussian distribution to the hidden activations for each distribution and then computes the Fréchet distance, also known as the Wasserstein-2 distance, between those Gaussians.

Heusel et al.show that unlike the Inception score, the FID worsens monotonically as various types of artifacts are added to CelebA images -though in our Appendix E we found the Inception score to be more mono- tonic than did Heusel et al., so this property may not be very robust to small changes in evaluation methods.

Note also that the estimator of FID is biased; 10 we will discuss this issue shortly.

KID We propose a metric similar to the FID, the Kernel Inception Distance, to be the squared MMD between Inception representations.

We use a polynomial kernel, k(x, y) = DISPLAYFORM0 where d is the representation dimension, to avoid correlations with the objective of MMD GANs as well as to avoid tuning any kernel parameters.

11 This can also be viewed as an MMD directly on input images with the kernel K(x, y) = k(φ(x), φ(y)), with φ the function mapping images to Inception representations.

Compared to the FID, the KID has several advantages.

First, it does not assume a parametric form for the distribution of activations.

This is particularly sensible since the representations have ReLU activations, and therefore are not only never negative, but do not even have a density: about 2% of components in Inception representations are typically exactly zero.

With the cubic kernel we use here, the KID compares skewness as well as the mean and variance.

Also, unlike the FID, the KID has a simple unbiased estimator.12 It also shares the behavior of the FID as artifacts are added to images (Appendix E).

FIG2 demonstrates the empirical bias of the FID and the unbiasedness of the KID by comparing the CIFAR-10 train and test sets.

The KID ( FIG2 ) converges quickly to its presumed true value of 0; even for very small n, simple Monte Carlo estimates of the variance provide a reasonable measure of uncertainty.

By contrast, the FID estimate ( FIG2 ) does not behave so nicely: at n = 2 000, when the KID estimator is essentially always 0, the FID estimator is still quite large.

Even at n = 10 000, the full size of the CIFAR test set, the FID still seems to be decreasing from its estimate of about 8.1 towards zero, showing the strong persistence of bias.

This highlights that FID scores can only be compared to one another with the same value of n.

Yet even for the same value of n, there is no particular reason to think that the bias in the FID estimator will be the same when comparing different pairs of distributions.

In Appendix D, we demonstrate two situations where F ID(P 1 , Q) < F ID(P 2 , Q), but for insufficent numbers of samples the estimator usually gives the other ordering.

This can happen even where all distributions in question are one-dimensional Gaussians, as Appendix D.1 shows analytically.

Appendix D.2 also empirically demonstrates this on distributions more like the ones used for FID in practice, giving a 10 This is easily seen when the true FID is 0: here the estimator may be positive, but can never be negative.

Note also that in fact no unbiased estimator of the FID exists; see Appendix D.3.11 k is the default polynomial kernel in scikit-learn (Pedregosa et al., 2011) .

12 Because the computation of the MMD estimator scales like O(n 2 d), we recommend using a relatively small n and averaging over several estimates; this is closely related to the block estimator of BID16 .

The FID estimator, for comparison, takes time O(nd 2 + d 3 ), and is substantially slower for d = 2048.simple example with d = 2048 where even estimating with n = 50 000 samples reliably gives the wrong ordering between the models.

Moreover, Monte Carlo estimates of the variance are extremely small even when the estimate is very far from its asymptote, so it is difficult to judge the reliability of an estimate, and practitioners may be misled by the very low variance into thinking that they have obtained the true value.

Thus comparing FID estimates bears substantial risks.

KID estimates, by contrast, are unbiased and asymptotically normal.

For models on MNIST, we replace the Inception featurization with features from a LeNet-like convolutional classifier 13 (LeCun et al., 1998) , but otherwise compute the scores in the same way.

We also considered the diagnostic test of Arora & Zhang FORMULA0 , which estimates the approximate number of "distinct" images produced by a GAN.

The amount of subjectivity in what constitutes a duplicate image, however, makes it hard to reliably compare models based on this diagnostic.

Comparisons likely need to be performed both with a certain notion of duplication in mind and by a user who does not know which models are being compared, to avoid subconscious biases; we leave further exploration of this intriguing procedure to future work.

In supervised deep learning, it is common practice to dynamically reduce the learning rate of an optimizer when it has stopped improving the metric on a validation set.

So far, this does not seem to be common in GAN-type models, so that learning rate schedules must be tuned by hand.

We propose instead using an adaptive scheme, based on comparing the KID score for samples from a previous iteration to that from the current iteration.

To avoid setting an explicit threshold on the change in the numerical value of the score, we use a p-value obtained from the relative similarity test of Bounliphone et al. (2016) .

If the test does not indicate that our current model is closer to the validation set than the model from a certain number of iterations ago at a given significance level, we mark it as a failure; when a given number of failures occur in a row, we decrease the learning rate.

Bounliphone et al.'s test is for the hypothesis MMD(P 1 , Q) < MMD(P 2 , Q), and since the KID can be viewed as an MMD on image inputs, we can apply it directly.

We compare the quality of samples generated by MMD GAN using various kernels with samples obtained by WGAN-GP (Gulrajani et al., 2017) and Cramér GAN (Bellemare et al., 2017) on four standard benchmark datasets: the MNIST dataset of 28 × 28 handwritten digits 15 , the CIFAR-10 dataset of 32 × 32 photos (Krizhevsky, 2009), the LSUN dataset of bedroom pictures resized to 64 × 64 BID14 , and the CelebA dataset of celebrity face images resized and cropped to 160 × 160 (Liu et al., 2015) .For most experiments, except for those with the CelebA dataset, we used the DCGAN architecture (Radford et al., 2016) for both generator and critic.

For MMD losses, we used only 16 top-layer neurons in the critic; more did not seem to improve performance, except for the distance kernel for which 256 neurons in the top layer was advantageous.

As Bellemare et al. (2017) advised to use at least 256-dimensional critic output, this enabled exact comparison between Cramér GAN and energy distance MMD, which are directly related (Section 2.3).

For the generator we used the standard number of convolutional filters (64 in the second-to-last layer); for the critic, we compared networks with 16 and 64 filters in the first convolutional layer.

13 github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py 14 We use the slight corrections to the asymptotic distribution of the MMD estimator given by BID9 in this test.15 yann.lecun.com/exdb/mnist/ 16 In the DCGAN architecture the number of filers doubles in each consecutive layer, so an f -filter critic has f , 2f , 4f and 8f convolutional filters in layers 1-4, respectively.

For the higher-resolution model for the CelebA dataset, we used a 5-layer DCGAN critic and a 10-layer ResNet generator 17 , with 64 convolutional filters in the last/first layer.

This allows us to compare the performance of MMD GANs with a more complex architecture.

Models with smaller critics run considerably faster: on our systems, the 16-filter DCGAN networks typically ran at about twice the speed of the 64-filter ones.

Note that the critic size is far more important to training runtime than the generator size: we update the critic 5 times for each generator step, and moreover the critic network is run on two batches each time we use it, one from P and one from Q. Given the same architecture, all models considered here run at about the same speed.

We evaluate several MMD GAN kernel functions in our experiments.

18 The simplest is the linear kernel: k dot (x, y) = x, y , whose MMD corresponds to the distance between means (this is somewhat similar to the feature matching idea of Salimans et al., 2016) .

We also use the exponentiated quadratic (5) and rational quadratic (6) functions, with mixtures of lengthscales, DISPLAYFORM0 where Σ = {2, 5, 10, 20, 40, 80}, A = {.2, .5, 1, 2, 5}. For the latter, however, we found it advantageous to add a linear kernel to the mixture, resulting in the mixed RQ-dot kernel k rq * = k rq + k dot .

Lastly we use the distance-induced kernel k dist ρ1,0 of (8), using the Euclidean distance ρ 1 so that the MMD is the energy distance.

19 We also considered Cramér GANs, with the surrogate critic (10), and WGAN-GPs.

Each model was trained with a batch size of 64, and 5 discriminator updates per generator update.

For CIFAR-10, LSUN and CelebA we trained for 150 000 generator updates, while for MNIST we used 50 000.

The initial learning rate was set to 10 −4 and followed the adaptive scheme described in Section 4.1, with KID compared between the current model and the model 20 000 generator steps earlier (5 000 for MNIST), every 2 000 steps (500 for MNIST).

After 3 consecutive failures to improve, the learning rate was halved.

This approach allowed us to avoid manually picking a different learning rate for each of the considered models.

We scaled the gradient penalty by 1, instead of the 10 recommended by Gulrajani et al. (2017) and Bellemare et al. (2017) ; we found this to usually work slightly better with MMD models.

With the distance kernel, however, we scale the penalty by 10 to allow direct comparison with Cramér GAN.Quantitative scores are estimated based on 25 000 generator samples (100 000 for MNIST), and compared to 25 000 dataset elements (for LSUN and CelebA) or the standard test set (10 000 images held out from training for MNIST and CIFAR-10).

Inception and FID scores were computed using 10 bootstrap resamplings of the given images; the KID score was estimated based on 100 repetitions of sampling 1 000 elements without replacement.

Code for our models is available at github.com/mbinkowski/MMD-GAN.MNIST All of the models achieved good results, measured both visually and in quantitative scores; full results are in Appendix F. FIG3 , however, shows the evolution of our quantitative criteria throughout the training process for several models.

This shows that the linear kernel dot and rbf kernel rbf are clearly worse than the other models at the beginning of the training process, but both improve eventually.

rbf , however, never fully catches up with the other models.

There is also some evidence that dist, and perhaps WGAN-GP, converge more slowly than rq and Cramér GAN.

Given their otherwise similar properties, we thus recommend the use of rq kernels over rbf in MMD GANs and limit experiments for other datasets to rq and dist kernels.

Full results are shown in Appendix F. Small-critic MMD GAN models approximately match large-critic WGAN-GP models, at substantially reduced computational cost.

17 As in Gulrajani et al. (2017) , we use a linear layer, 4 residual blocks and one convolutional layer.

18 Because these higher-resolution experiments were slower to run, for CelebA we trained MMD GAN with only one type of kernel.

19 We also found it helpful to add an activation penalty to the critic representation network in certain MMD models.

Otherwise the representations h θ sometimes chose very large values, which for most kernels does not change the theoretical loss (defined only in terms of distances) but leads to floating-point precision issues.

We use a combined L 2 penalty on activations across all critic layers, with a factor of 1 for rq * and 0.0001 for dist.

TAB0 presents scores for models trained on the LSUN Bedrooms dataset; samples from most of these models are shown in Figure 3 .

Comparing the models' Inception scores with the one achieved by the test set makes clear that this measure is not meaningful for this datasetnot surprisingly, given the drastic difference in domain from ImageNet class labels.

In terms of KID and FID, MMD GANs outperform Cramér and WGAN-GP for each critic size.

Although results with the smaller critic are worse than with the large one for each considered model, small-critic MMD GANs still produce reasonably good samples, which certainly is not the case for WGAN-GP.

Although a small-critic Cramér GAN produces relatively good samples, the separate objects in these pictures often seem less sharp than the MMD rq* samples.

With a large critic, both Cramér GAN and MMD rq* give good quality samples, many of which are hardly distinguishable from the test set by eye.

CelebA Scores for the CelebA dataset are shown in TAB1 ; MMD GAN with rq* kernel outperforms both WGAN-GP and Cramér GAN in KID and FID.

Samples in Figure 4 show that for each of the models there are many visually pleasing pictures among the generated ones, yet unrealistic images are more common for WGAN-GP and Cramér.

These results illustrate the benefits of using the MMD on deep convolutional feaures as a GAN critic.

In this hybrid system, the initial convolutional layers map the generator and reference image distributions to a simpler representation, which is well suited to comparison via the MMD.

The MMD in turn employs an infinite dimensional feature space to compare the outputs of these convolutional layers.

By comparison, WGAN-GP requires a larger discriminator network to achieve similar performance.

It is interesting to consider the question of kernel choice: the distance kernel and RQ kernel are both characteristic BID4 , and neither suffers from the fast decay of the exponentiated quadratic kernel, yet the RQ kernel performs slightly better in our experiments.

The relative merits of different kernel families for GAN training will be an interesting topic for further study.

It is not immediately obvious how to interpret the surrogate loss (10).

An insight comes from considering the score function associated with the energy distance, which we now briefly review (Gneiting & Raftery, 2007) .

A scoring rule is a function S(P, y), which is the loss incurred when a forecaster makes prediction P, and the event y is observed.

The expected score is the expectation under Q of the score, DISPLAYFORM0 If a score is proper, then the expected score obtained when P = Q is greater or equal than the expected score for P = Q, S(Q, Q) ≥ S(P, Q).A strictly proper scoring rule shows an equality only when P and Q agree.

We can define a divergence measure based on this score, DISPLAYFORM1 Bearing in mind the definition of the divergence (15), it is easy to see (Gneiting & Raftery, 2007, eq. 22 ) that the energy distance (7) arises from the score function DISPLAYFORM2 The interpretation is straightforward: the score of a reference sample y is determined by comparing its average distance to a generator sample with the average distance among independent generator samples, E P ρ(X, X ).

If we take an expectation over Y ∼ Q, we recover the scoring rule optimized by the DISCO Nets algorithm (Bouchacourt et al., 2016, Section 3.3).As discussed earlier, the Cramér GAN critic does not use the energy distance (7) directly on the samples, but first maps the samples through a function h, for instance a convolutional network; this should be chosen to maximize the discriminative performance of the critic.

Writing this mapping as h, we break the energy distance down as D e (P, Q) = S(Q, Q) − S(P, Q), where DISPLAYFORM3 and DISPLAYFORM4 When training the discriminator, the goal is to maximize the divergence by learning h, and so both FORMULA0 and FORMULA0 change: in other words, divergence maximization is not possible without two independent samples Y, Y from the reference distribution Q.An alternative objective in light of the score interpretation, however, is to simply optimize the average score (17).

In other words, we would find features h that make the average distance from generator to reference samples much larger than the average distance between pairs of generator samples.

We no longer control the term encoding the "variability" due to Q, E Q ρ(h(Y ), h(Y )), which might therefore explode: for instance, h might cause h(Y ) to disperse broadly, and far from the support of P, assuming sufficient flexibility to keep E P ρ(h(X), h(X )) under control.

We can mitigate this by controlling the expected norm E Q ρ(h(Y ), 0), which has the advantage of only requiring a single sample to compute.

For example, we could maximize DISPLAYFORM5 This resembles the Cramér GAN critic (10), but the generator-to-generator distance is scaled differently, and there is an additional term: E P ρ(h(X), 0) is being maximized in (10), which is more difficult to interpret.

An argument has been made (in personal communication with Bellemare et al.) that this last term is required if the function f c in (9) is to be a witness of an integral probability metric (1), although the asymmetry of this witness in P vs Q needs to be analyzed further.

We will now show that all estimators of IPM-like distances and their gradients are biased.

Appendix B.1 defines a slight generalization of IPMs, used to analyze MMD GANs in the same framework as WGANs, and a class of estimators that are a natural model for the estimator used in GAN models.

Appendix B.2 both shows that not only are this form of estimators invariably biased in nontrivial cases, and moreover no unbiased estimator can possibly exist; Appendix B.3 then demonstrates that any estimator with non-constant bias yields a biased gradient estimator.

Appendices B.4 and B.5 demonstrate specific examples of this bias for the Wasserstein and maximized-MMD distances.

We will first define a slight generalization of IPMs: we will use this added generality to help analyze MMD GANs in Appendix B.5.Definition 1 (Generalized IPM).

Let X be some domain, with M a class of probability measures on X .

20 Let F be some parameter set, and J : DISPLAYFORM0 For example, if F is a class of functions f : X → R, and J is given by DISPLAYFORM1 then we obtain integral probability metrics (1).

Given samples X ∼ P m and Y ∼ Q n , letP denote the empirical distribution of X (an equal mixture of point masses at each X i ∈ X), and similarlyQ for Y. Then we have a simple estimator of (19) which is unbiased for fixed f : DISPLAYFORM2 Definition 2 (Data-splitting estimator of a generalized IPM).

Consider the distance (18), with objective J and parameter class F. Suppose we observe iid samples X ∼ P m , Y ∼ Q n , for any two distributions P, Q. A data-splitting estimator is a functionD(X, Y) which first randomly splits the sample X into X tr , X te and Y into Y tr , Y te , somehow selects a critic functionf X tr ,Y tr ∈ F independently of X te , Y te , and then returns a result of the form DISPLAYFORM3 These estimators are defined by three components: the choice of relative sizes of the train-test split, the selection procedure forf X tr ,Y tr , and the estimatorĴ. The most obvious selection procedure iŝ DISPLAYFORM4 though of course one could use regularization or other techniques to select a different f ∈ F, and in practice one will use an approximate optimizer.

Lopez-Paz & Oquab (2017) used an estimator of exactly this form in a two-sample testing setting.

As noted in Section 3, this training/test split is a reasonable match for the GAN training process.

As we optimize a WGAN-type model, we compute the loss (or its gradients) on a minibatch, while the current parameters of the critic are based only on data seen in previous iterations.

We can view the current minibatch as X te , Y te , all previously-seen data as X tr , Y tr , and the current critic function asf X tr ,Y tr .

Thus, at least in the first pass over the training set, WGAN-type approaches exactly fit the data-splitting form of Definition 2; in later passes, the difference from this setup should be relatively small unless the model is substantially overfitting.

We first show, in Theorem 2, that data-splitting estimators are biased downwards.

Although this provides substantial intuition about the situation in GANs, it leaves open the question of whether some other unbiased estimator might exist; Theorem 3 shows that this is not the case.

Theorem 2.

Consider a data-splitting estimator (Definition 2) of the generalized IPM D (Definition 1) based on an unbiased estimatorĴ of J: for any fixed f ∈ F, DISPLAYFORM0 Then either the selection procedure is almost surely perfect, DISPLAYFORM1 or else the estimator has a downward bias: DISPLAYFORM2 Proof.

Since X tr , Y tr are independent of X te , Y te , DISPLAYFORM3 Define the suboptimality off X tr ,Y tr as DISPLAYFORM4 .

Note that ε ≥ 0, since D(P, Q) = sup f ∈F J(f, P, Q) and so for any f ∈ F we have J(f, P, Q) ≤ D(P, Q).Thus, either Pr(ε = 0) = 1, in which case (21) holds, or else E[ε] > 0, giving (22).Theorem 2 makes clear that asf X tr ,Y tr converges to its optimum, the bias ofD should vanish (as in Bellemare et al., 2017, Theorem 3).

Moreover, in the GAN setting the minibatch size only directly determines X te , Y te , which do not contribute to this bias; bias is due rather to the training procedure and the number of samples seen through the training process.

As long asf X tr ,Y tr is not optimal, however, the estimator will remain biased.

Many estimators of IPMs do not actually perform this data splitting procedure, instead estimating D(P, Q) with the distance between empirical distributions D(P,Q).

The standard biased estimator of the MMD (Gretton et al., 2012, Equation 5 ), the IPM estimators of BID6 , and the empirical Wasserstein estimator studied by Bellemare et al. (2017) are all of this form.

These estimators, as well as any other conceivable estimator, are also biased: Theorem 3.

Let P be a class of distributions such that {(1 − α)P 0 + αP 1 : 0 ≤ α ≤ 1} ⊆ P, where P 0 = P 1 are two fixed distributions.

Let D be an IPM (1).

There does not exist any estimator of D which is unbiased on P.Proof.

We use a technique inspired by Bickel & Lehmann (1969) .

Suppose there is an unbiased estimatorD(X, Y) of D: for some finite m and n, if X = {X 1 , . . .

, DISPLAYFORM5 Fix P 0 , P 1 , and Q ∈ P, and consider the function DISPLAYFORM6 Thus R(α) is a polynomial in α of degree at most m. DISPLAYFORM7 where (23) used our general assumption about IPMs that if f ∈ F, we also have −f ∈ F. But R(α) is not a polynomial with any finite degree.

Thus no such unbiased estimatorD exists.

Note that the proof of Theorem 3 does not readily extend to generalized IPMs, and so does not tell us whether an unbiased estimator of the MMD GAN objective (14) can exist.

Also, attempting to apply the same argument to squared IPMs would give the square of (24), which is a quadratic function in α.

Thus tells us that although no unbiased estimator for a squared IPM can exist with only m = 1 sample point, one can exist for m ≥ 2, as indeed (4) does for the squared MMD.

We will now show that biased estimators, except for estimators with a constant bias, must also have biased gradients.

Assume that, as in the GAN setting, Q is given by a generator network G ψ with parameter ψ and inputs Z ∼ Z, so that Y = G ψ (Z) ∼ Q ψ .

The generalized IPM of FORMULA0 is now a function of ψ, which we will denote as DISPLAYFORM0 Consider an estimatorD(ψ) of D(ψ).

Theorem 4 shows that whenD(ψ) and D(ψ) are differentiable, the gradient ∇ ψD (ψ) is an unbiased estimator for ∇ ψ D(ψ) only if the bias ofD(ψ) doesn't depend on ψ.

This is exceedingly unlikely to happen for the biased estimatorD(W ) defined in Theorem 2, and indeed Theorem 3 shows cannot happen for any IPM estimator.

Theorem 4.

Let D : Ψ → R be a function on a parameter space Ψ ⊆ R d , with a random estimator D : Ψ → R which is almost surely differentiable.

Suppose thatD has unbiased gradients: DISPLAYFORM1 Then, for each connected component of Ψ, DISPLAYFORM2 where the constant can vary only across distinct connected components.

Proof.

Let ψ 1 and ψ 2 be an arbitrary pair of parameter values in Ψ, connected by some smooth path r : [0, 1] → Ψ with r(0) = ψ 1 , r(1) = ψ 2 .

For example, if Ψ is convex, then paths of the form r(t) = tψ 1 + (1 − t)ψ 2 are sufficient.

Using Fubini's theorem and standard results about path integrals, we have that DISPLAYFORM3

+ const for all ψ in the same connected component of Ψ.

Theorems 2 and 3 hold for the original WGANs, whose critic functions are exactly L-Lipschitz, considering F as the set of L-Lipschitz functions so that D F is L times the Wasserstein distance.

They also hold for either WGANs or WGAN-GPs with F the actual set of functions attainable by the critic architecture, so that D is the "neural network distance" of Arora et al. (2017) or the "adversarial divergence" of Liu et al. (2017) .It should be obvious that for nontrivial distributions P and Q and reasonable selection criteria for f X tr ,Y tr , (21) does not hold, and thus FORMULA2 does (so that the estimate is biased downwards).

Theorem 3 also shows this is the case on reasonable families of input distributions, and moreover that the bias is not constant, so that gradients are biased by Theorem 4.Example For an explicit demonstration, consider the Wasserstein case, F the set of 1-Lipschitz functions, with P = N (1, 1) and Q = N (0, 1).

Here D F (P, Q) = 1; the only critic functions f ∈ F which achieve this are f (t) = t + C for C ∈ R.If we observe only one training pair X tr ∼ P and Y tr ∼ Q, when X tr > Y tr , f 1 (t) = t is a maximizer of (20), leading to the expected estimate J(f 1 , P, Q) = 1.

But with probability Φ −1/ √ 2 ≈ 0.24 it happens that X tr < Y tr .

In such cases, FORMULA2 could give e.g. f −1 (t) = −t, giving the expected response J(f −1 , P, Q) = −1; the overall expected estimate of the estimator using this critic selection procedure is then DISPLAYFORM0 The only way to achieve ED F (X, Y) = 1 would be a "stubborn" selection procedure which chooses f 1 + C no matter the given inputs.

This would have the correct output ED F (X, Y) = 1 for this (P, Q) pair.

Applying this same procedure to P = N (−1, 1) and Q = N (0, 1), however, would then give ED F (X, Y) = −1, when it should also be 1.

Recall the distance η(P, Q) = sup θ MMD 2 (h θ (P), h θ (Q)) defined by (14).

MMD GANs can be viewed as estimating η according to the scheme of Theorem 2, with F the set of possible parameters θ, J(θ, DISPLAYFORM0 ).

Clearly our optimization scheme for θ does not almost surely yield perfect answers, and so again we have DISPLAYFORM1 As m tr , n tr → ∞, as for Wasserstein it should be the case thatη → η.

This is shown for certain kernels, along with the rate of convergence, by Sriperumbudur et al. (2009a, Section 4) .It should also be clear that in nontrivial situations, this bias is not constant, and hence gradients are biased by Theorem 4.Example For a particular demonstration, consider DISPLAYFORM2 with h θ : R 2 → R given by h θ (x) = θ T x, θ = 1, so that h θ chooses a one-dimensional projection of the two-dimensional data.

Then use the linear kernel k dot , so that the MMD is simply the difference in means between projected features: DISPLAYFORM3 , and DISPLAYFORM4 Clearly η(P, Q) = 1, which is obtained by θ ∈ {(−1, 0), (1, 0)}; any other valid θ will yield a strictly smaller value of MMD 2 (h θ (P), h θ (Q)).The MMD GAN estimator of η, if the optimum is achieved, useŝ FIG2 , (1, 0)}; thus by Theorem 2, Eη(X, Y) < η(P, Q) = 1.

(A numerical simulation for the former gives a value around 0.6 when m tr = n tr = 2.)

DISPLAYFORM5

We now proceed to prove Theorem 1 as a corollary to the Theorem 5, our main result about exchanging gradients and expectations of deep networks.

Exchanging the gradient and the expectation can often be guaranteed using a standard result in measure theory (see Proposition 1), as a corollary of the Dominated Convergence theorem (Proposition 2).

This result, however, requires the property Proposition 1.(ii): for almost all inputs X, the mapping is differentiable on the entirety of a neighborhood around θ.

This order of quantifiers is important: it allows the use of the mean value theorem to control the average rate of change of the function, and the result then follows from Proposition 2.For a neural network with the ReLU activation function, however, this assumption doesn't hold in general.

For instance, if θ = (θ 1 , θ 2 ) ∈ R 2 with θ 2 = 0 and X ∈ R, one can consider this very simple function: h θ (X) = max(0, θ 1 + θ 2 X).

For any fixed value of θ, the function h θ (X) is differentiable in θ for all X in R except for X θ = −θ 1 /θ 2 .

However, if we consider a ball of possible θ values B(θ, r), the function is not differentiable on the set {−θ 1 /θ 2 ∈ R | θ ∈ B(θ, r)}, which can have positive measure for many possible distributions for X.In Theorem 5, we provide a proof that derivatives and expectations can be exchanged for all parameter values outside of a "bad set" Θ P , without relying on Proposition 1.(ii).

This can be done using Lemma 1, which takes advantage of the particular structure of neural networks to control the average rate of change without using the mean value theorem.

Dominated convergence (Proposition 2) can then be applied directly.

We also show in Proposition 3 that the set Θ P , of parameter values where Theorem 5 might not hold, has zero Lebesgue measure.

This relies on the standard Fubini theorem (Klenke, 2008, Theorem 14.16) and Lemma 4, which ensures that the network θ → h θ (X) is differentiable for almost all parameter values θ when X is fixed.

Although Lemma 4 might at first sight seem obvious, it requires some technical considerations in topology and differential geometry.

Proposition 1 (Differentiation Lemma (e.g. Klenke, 2008, Theorem 6.28) ).

Let V be a nontrivial open set in R m and let P be a probability distribution on R d .

Define a map h : R d × V → R n with the following properties: DISPLAYFORM0 (iii) There exists a P-integrable function g : DISPLAYFORM1 .

Proposition 2 (Dominated Convergence Theorem (e.g. Klenke, 2008, Corollary 6.26) ).

Let P be a probability distribution on R d and f a measurable function.

Let (f n ) n∈N be a sequence of of integrable functions such that for P-almost all X ∈ R d , f n (X) → f (X) as n goes to ∞. Assume that there is a dominating function g: f n (X) ≤ g(X) for P-almost all X ∈ R d for all n ∈ N, and E P [g(X)]

< ∞. Then f is P-integrable, and E P [f n (X)] → E P [f (X)] as n goes to ∞.

We would like to consider general feed-forward networks with a directed acyclic computation graph G. Here, G consists of L + 1 nodes, with a root node i = 0 and a leaf node i = L. We denote by π(i) the set of parent nodes of i. The nodes are sorted according to a topological order: if j is a parent node of i, then j < i.

Each node i for i > 0 computes a function f i , which outputs a vector in R di based on its input in R d π(i) , the concatenation of the outputs of each layer in π(i).

DISPLAYFORM0 We define the feed-forward network that factorizes according the graph G and with functions f i recursively: DISPLAYFORM1 where h π(i) is the concatenation of the vectors h j for j ∈ π(i).

The functions f i can be of two types:• Affine transform (Linear Module): DISPLAYFORM2 is a known linear operator on the weights W i , which can account for convolutions and similar linear operations.

We will sometimes use Y to denote the augmented vector Y 1 , which accounts for bias terms.• Non-linear: These f i have no learnable weights.

f i can potentially be non-differentiable, such as max pooling, ReLU, and so on.

Some conditions on f i will be required (see Assumption D); the usual functions used in practice satisfy these conditions.

Denote by C the set of nodes i such that f i is non-linear.

θ is the concatenation of parameters of all linear modules: DISPLAYFORM3 .., L}. Call the total number of parameters m = i∈C c m i , so that θ ∈ R m .

The feature vector of the network corresponds to the output of the last node L and will be denoted DISPLAYFORM4 The subscript θ stands for the parameters of the network.

We will sometimes use h θ (X) to denote explicit dependence on X, or omit it when X is fixed.

Also define a "top-level function" to be applied to DISPLAYFORM5 This function might simply be K (U ) = U , as in Corollaries 1 and 2.

But it also allows us to represent the kernel function of an MMD GAN in Corollary 3: here we take X to be the two inputs to the kernel stacked together, apply the network to each of the two inputs with the same parameters in parallel, and then compute the kernel value between the two representations with K .

K will have different smoothness assumptions than the preceding layers (Assumption B).

We will need the following assumptions at various points, where α ≥ 1: DISPLAYFORM0 B The function K is continuously differentiable, and satisfies the following growth conditions where C 0 and C 1 are constants: DISPLAYFORM1 , each real analytic on R d π(i) , which agree with f i on the closure of a set D DISPLAYFORM2 ∀Y ∈D k i .

These sets D k i are disjoint, and cover the whole input space: DISPLAYFORM3 DISPLAYFORM4 Another example is when f i computes max-pooling on two inputs.

In that case we have K i = 2, and each domain D k i corresponds to a half plane (see FIG6 ).

Each domain is defined by one inequality DISPLAYFORM5 DISPLAYFORM6 When f i is analytic on the whole space, DISPLAYFORM7 , which can be defined by a single function (S i,k = 1) of G i,1,1 (Y ) = 1.

This case corresponds to most of the differentiable functions used in deep learning, such as the softmax, sigmoid, hyperbolic tangent, and batch normalization functions.

Other activation functions, such as the ELU (Clevert et al., 2016) , are piecewise-analytic and also satisfy Assumptions C and D.

We first state the main result, which implies Theorem 1 via Corollaries 1 to 3.

The proof depends on various intermediate results which will be established afterwards.

] is differentiable at θ 0 , and DISPLAYFORM0 where µ is the Lebesgue measure.

Proof.

Let θ 0 be such that the function θ → h θ (X) is differentiable at θ 0 for P-almost all X. By Proposition 3, this is the case for µ-almost all θ 0 in R m .Consider a sequence (θ n ) n∈N that converges to θ 0 ; there is then an R > 0 such that θ n − θ 0 < R for all n ∈ N. Letting X be in R d , Lemma 2 gives that DISPLAYFORM1 It also follows that: DISPLAYFORM2 converges point-wise to 0 and is bounded by the integrable function 2F (X).

Therefore by the dominated convergence theorem (Proposition 2) it follows that DISPLAYFORM3 Finally we define the sequence DISPLAYFORM4 which is upper-bounded by E P [M n (X)] and therefore converges to 0.

By the sequential characterization of limits in Lemma 3, it follows that E P [K (h θ (X))] is differentiable at θ 0 , and its differential is given by DISPLAYFORM5 These corollaries of Theorem 5 apply it to specific GAN architectures.

Here we use the distribution Z to represent the noise distribution.

Corollary 1 (WGANs).

Let P and Z be two distributions, on X and Z respectively, each satisfying Assumption A for α = 1.

Let G ψ : Z → X be a generator network and D θ : X → R a critic network, each satisfying Assumptions C and D. Then, for µ-almost all (θ, ψ), we have that DISPLAYFORM6 Proof.

By linearity, we only need the following two results: DISPLAYFORM7 The first follows immediately from Theorem 5, using the function K (U ) = U (which clearly satisfies Assumption B for α = 1).

The latter does as well by considering that the augmented network DISPLAYFORM8 still satisifes the conditions of Theorem 5.

Corollary 2 (Original GANs).

Let P and Z be two distributions, on X and Z respectively, each satisfying Assumption A for α = 1.

Let G ψ : Z → X be a generator network, and D θ : X → R a discriminator network, each satisfying Assumptions C and D. Further assume that the output of D is almost surely bounded: there is some γ > 0 such that for µ-almost all (θ, ψ), Pr DISPLAYFORM0 Then we have the following: Proof.

The log function is real analytic and (1/γ)-Lipschitz on (γ, 1 − γ).

The claim therefore follows from Theorem 5, using the networks log DISPLAYFORM1 DISPLAYFORM2 The following assumption about a kernel k implies Assumption B when used as a top-level function K :

E Suppose k is a kernel such that there are constants C 0 , C 1 where DISPLAYFORM3 Corollary 3 (MMD GANs).

Let P and Z be two distributions, on X and Z respectively, each satisfying Assumption A for some α ≥ 1.

Let k be a kernel satisfying Assumption E. Let G ψ : Z → X be a generator network and D θ : X → R a critic representation network each satisfying Assumptions C and D. Then DISPLAYFORM4 Proof.

Consider the following augmented networks: DISPLAYFORM5 (1) has inputs distributed as P × Z, which satisfies Assumption A with the same α as P and Z, and h (1) satisfies Assumptions C and D. The same is true of h (2) and h (3) .

Moreover, the function DISPLAYFORM6 satisfies Assumption B. Thus Theorem 5 applies to each of h (1) , h (2) , and h (3) .

Considering the form of MMD 2 u (4), the result follows by linearity and the fact that MMD Each of the kernels considered in this paper satisfies Assumption E with α at most 2:• k dot (x, y) = x, y works with α = 2, C 0 = 1, C 1 = 1.• k rbf σ of (5) works with α = 2, C 0 = 1, DISPLAYFORM7 • k rq α of (6) works with α = 2, C 0 = 1, C 1 = √ 2.• k dist ρ β ,0 of (8), using ρ β (x, y) = x − y β with 1 ≤ β ≤ 2, works with α = β, C 0 = 3, C 1 = 4β.

Since the existence of a moment implies the existence of all lower-order moments by Jensen's inequality, this finalizes the proof of Theorem 1.

DISPLAYFORM8 with: DISPLAYFORM9 When i is not a linear layer, then by Assumption C f i is M -Lipschitz.

Thus we can directly get the needed functions by recursion: DISPLAYFORM10 Lemma 2.

Let R be a positive constant and θ ∈ R m .

Under Assumptions A to C, the following hold for all θ ∈ B(θ, R) and all X in R d : DISPLAYFORM11 Proof.

We will first prove the following inequality: DISPLAYFORM12 Let t be in [0, 1] and define the function f by DISPLAYFORM13 Then f (0) = K (V ) and f (1) = K (U ).

Moreover, f is differentiable and its derivative is given by: DISPLAYFORM14 Using Assumption B one has that: DISPLAYFORM15 The conclusion follows using the mean value theorem.

Now choosing U = h θ (X) and V = h θ (X) one gets the following: DISPLAYFORM16 Under Assumption C, it follows by Lemma 1 that: DISPLAYFORM17 The functions a, b, α, β defined in Lemma 1 are continuous, and hence all bounded on the ball B(θ, R); choose D > 0 to be a bound on all of these functions.

It follows after some algebra that DISPLAYFORM18 α is concave on t ≥ 0, and so we have that DISPLAYFORM19 via Jensen's inequality and Assumption A. We also have E [1 + X ] < ∞ by the same assumption.

Thus F (X) is integrable.

Lemma 3.

Let f : R m → R be a real valued function and g a vector in R m such that: DISPLAYFORM20 for all sequences (θ n ) n∈N converging towards θ 0 with θ n = θ 0 .

Then f is differentiable at θ 0 , and its differential is g.

Recall the definition of a differential: g is the differential of f at θ 0 if DISPLAYFORM0 The result directly follows from the sequential characterization of limits.

The last result required for the proof of Theorem 5 is Proposition 3.

We will first need some additional notation.

For a given node i, we will use the following sets of indices to denote "paths" through the network's computational graph: DISPLAYFORM0 Note that ∂i ⊆ ¬i, and that ¬i = ∂i ∪ ¬π(i).If a(i) is the set of ancestors of node i, we define a backward trajectory starting from node i as an element q of the form: DISPLAYFORM1 where k j are integers in [K j ].

We call T (i) the set of such trajectories for node i.

For p ∈ P of the form p = (i, k, s), the set of parameters for which we lie on the boundary of p is DISPLAYFORM2 We also denote by ∂S p the boundary of the set S p .

If Q is a subset of P , we use the following notation for convenience: DISPLAYFORM3 For a given θ 0 ∈ R m , the set of input vectors X ∈ R d such that h θ0 is not differentiable is DISPLAYFORM4 Consider a random variable X in the input space R d , following the distribution P. For a given distribution P, we introduce the following set of "critical" parameters: DISPLAYFORM5 This is the set of parameters θ where the network is not differentiable for a non-negligible set of datasets X.Finally, for a given X ∈ R d , set of parameters for which the network is not differentiable is DISPLAYFORM6 We are now ready to state and prove the remaining result.

Proposition 3.

Under Assumption D, the set Θ P has 0 Lebesgue measure for any distribution P.Proof.

Consider the following two sets: DISPLAYFORM7 By virtue of Theorem I in BID15 Piranian (1966) , it follows that the set of nondifferentiability of continuous functions is measurable.

It is easy to see then, that D and Q are also measurable sets since the network is continuous.

Note that we have the inclusion D ⊆ Q.

We endow the two sets with the product measure ν := µ × P, where µ is the Lebesgue measure.

Therefore ν(D) ≤ ν(Q).

On one hand, Fubini's theorem tells us: DISPLAYFORM8 By Lemma 4, we have that µ(Θ X ) = 0; therefore ν(Q) = 0 and hence ν(D) = 0.

On the other hand, we use again Fubini's theorem for ν(D) to write: DISPLAYFORM9 For all θ ∈ Θ P , we have P(N (θ)) > 0 by definition.

Thus ν(D) = 0 implies that µ(Θ P ) = 0.

Lemma 4.

Under Assumption D, for any X in R d , the set Θ X has 0 Lebesgue measure: µ(Θ X ) = 0.Proof.

We first show that Θ X ⊆ ∂S P , which was defined by (25).Let θ 0 be in Θ X .

By Assumption D, it follows that θ 0 ∈ S P .

Assume for the sake of contradiction that θ 0 / ∈ ∂S P .

Then applying Lemma 5 to the output layer, i = L, implies that there is a real analytic function f (θ) which agrees with h θ on all θ ∈ B(θ 0 , η) for some η > 0.

Therefore the network is differentiable at θ 0 , contradicting the fact that θ 0 ∈ Θ X .

Thus Θ X ⊆ ∂S P .Lemma 6 then establishes that µ(∂S P ) = 0, and hence µ(Θ X ) = 0.Lemma 5.

Let i be a node in the graph.

Under Assumption D, if θ ∈ R m \ ∂S ¬i , then there exist η > 0 and a trajectory q ∈ T (i) such that h i θ = f q (θ ) for all θ in the ball B(θ, η).

Here f q is the real analytic function on R m defined with the same structure as h θ , but replacing each nonlinear f j with the analytic function f kj j for (j, k j ) ∈ q.

Proof.

We proceed by recursion on the nodes of the network.

If i = 0, we trivially have h 0 θ = X, which is real analytic on R m .

Assume the result for ¬π(i) and let θ ∈ R m \ ∂S ¬i .

In particular θ ∈ R m \ ∂S ¬π(p) .

By the recursion assumption, we get: DISPLAYFORM0 with f q real analytic in R m .If θ / ∈ S ∂i , then there is some sufficiently small η > 0 such that B(θ, η ) does not intersect S ∂i .Therefore, by Assumption D, there is some DISPLAYFORM1 ) for all θ ∈ B(θ, η ), where f k i is one of the real analytic functions defining f i .

By FORMULA2 we then have DISPLAYFORM2 Otherwise, θ ∈ S ∂i .

Then, noting that by assumption θ / ∈ ∂S ∂i , it follows that for small enough η > 0, we have B(θ, η ) ⊆ S ∂i .

Denote by A the set of index triples p ∈ ∂i such that θ ∈ S p ; A is nonempty since θ ∈ S ∂i .

Therefore θ ∈ p∈A S p , and θ / ∈ p∈A c S p .

We will show that for η small enough, B(θ, η ) ⊆ p∈A S p .

Assume for the sake of contradiction that there exists a sequence of (parameter, index-triple) pairs (θ n , p n ) such that p n ∈ A c , θ n ∈ S pn , and θ n → θ.

p n is drawn from a finite set and thus has a constant subsequence, so we can assume without loss of generality that p n = p 0 for some p 0 ∈ A c .

Since S p0 is a closed set by continuity of the network and G p0 , it follows that θ ∈ S p0 by taking the limit.

This contradicts the fact that θ / ∈ p∈A c S p .

Hence, for η small enough, DISPLAYFORM3 , where ⊕ denotes concatenation, it finally follows that h i θ = f q0 (θ ) for all θ in B(θ, min(η, η )), and f q0 is the real analytic function on R m as described.

DISPLAYFORM4 Proof.

We will proceed by recursion.

For i = 0 we trivially have ∂S ¬0 = ∅, thus µ(∂S ¬0 ) = 0.

Thus assume that µ(∂S ¬π(i) ) = 0.For s = (p, q), the pair of an index triple p ∈ ∂i and a trajectory q ∈ T (i), define the set DISPLAYFORM5 where f q is the real analytic function defined in Lemma 5 which locally agrees with h DISPLAYFORM6 We will now prove that for any θ in ∂S ∂i \ ∂S ¬π(i) , there exists s ∈ ∂i × T (i) such that θ ∈ M s and µ(M s ) = 0.

We proceed by contradiction.

DISPLAYFORM7 Moreover, since θ ∈ ∂S ∂i , there exists p ∈ ∂i such that G p (h π(i) θ ) = 0.

This means that for s = (p, q), we have θ ∈ M s .

If µ(M s ) > 0, then by Lemma 7 M s = R m , hence we would have B(θ, η) ⊆ M s .

By (28) it would then follow that B(θ, η) ⊆ S ∂i .

This contradicts the fact that θ is in ∂S ∂i , and hence µ(M s ) = 0.We have shown that ∂S ∂i \ ∂S ¬π(i) ⊆ s∈A M s , where the sets M s have zero Lebesgue measure and A ⊆ P × L j=0 T (j) is finite.

This implies: DISPLAYFORM8 Using the recursion assumption µ(∂S ¬π(i) ) = 0, one concludes that µ(∂S ¬i ) = 0.

Hence for the last node L, recalling that ¬L = P one gets µ(∂S P ) = 0.Lemma 7.

Let θ → F (θ) : R m → R be a real analytic function on R m and define the set: DISPLAYFORM9 Then either µ(M) = 0 or F is identically zero.

Proof.

This result is shown e.g. as Proposition 0 of Mityagin (2015).

We now further study the bias behavior of the FID estimator (Heusel et al., 2017) mentioned in Section 4.We will refer to the Fréchet Inception Distance between two distributions, letting µ P denote the mean of a distribution P and Σ P its covariance matrix, as DISPLAYFORM0 .

This is motivated because it coincides with the Fréchet (Wasserstein-2) distance between normal distributions.

Although the Inception coding layers to which the FID is applied are not normally distributed, the FID remains a well-defined pseudometric between arbitrary distributions whose first two moments exist.

The usual estimator of the FID based on samples {X i } m i=1 ∼ P m and {Y j } n j=1 ∼ P n is the plug-in estimator.

First, estimate the mean and covariance with the standard estimators: DISPLAYFORM1 LettingP X be a distribution matching these moments, e.g. N μ X ,Σ X , the estimator is given by DISPLAYFORM2 In Appendices D.1 and D.2, we exhibit two examples where FID(P 1 , Q) < FID(P 2 , Q), but the estimator FID(P 1 , Q) is usually greater than FID(P 2 , Q) with an equal number of samples m from P 1 and P 2 , for a reasonable number of samples.

(As m → ∞, of course, the estimator is consistent, and so the order will eventually be correct.)

We assume here an infinite number of samples n from Q for simplicity; this reversal of ordering is even easier to obtain when n = m. It is also trivial to achieve when the number of samples from P 1 and P 2 differ, as demonstrated by FIG2 .Note that Appendices D.1 and D.2 only apply to this plug-in estimator of the FID; it remains conceivable that there would be some other estimator for the FID which is unbiased.

Appendix D.3 shows that this is not the case: there is no unbiased estimator of the FID.

We will first show that the estimator can behave poorly even with very simple distributions.

When P = N (µ P , Σ P ) and Q = N (µ Q , Σ Q ), it is well-known that DISPLAYFORM0 where W is the Wishart distribution.

Then we have DISPLAYFORM1 The remaining term E Tr Σ XΣY 1 2 is more difficult to evaluate, because we must consider the correlations across dimensions of the two estimators.

But if the distributions in question are one-dimensional, denoting Σ P = σ .Thus the expected estimator for one-dimensional normals becomes DISPLAYFORM2 Now, consider the particular case DISPLAYFORM3 where the inequality follows because DISPLAYFORM4

The example of Appendix D.1, though indicative in that the estimator can behave poorly even with very simple distributions, is somewhat removed from the situations in which we actually apply the FID.

Thus we now empirically consider a more realistic setup.

First, as noted previously, the hidden codes of an Inception coding network are not well-modeled by a normal distribution.

They are, however, reasonably good fits to a censored normal distribution ReLU(X), where X ∼ N (µ, Σ) and ReLU(X) i = max(0, X i ).

Using results of Rosenbaum (1961) , it is straightforward to derive the mean and variance of ReLU(X) BID8 , and hence to find the population value of FID(ReLU(X), ReLU(Y )).Let d = 2048, matching the Inception coding layer, and consider DISPLAYFORM0 T , with C a d × d matrix whose entries are chosen iid standard normal.

For one particular random draw of C, we found that FID(P 1 , Q) ≈ 1123.0 > 1114.8 ≈ FID(P 2 , Q).

Yet with m = 50 000 samples, FID(P 1 , Q) ≈ 1133.7 (sd 0.2) < 1136.2 (sd 0.5) ≈ FID(P 2 , Q).

The variance in each estimate was small enough that of 100 evaluations, the largest FID(P 1 , Q) estimate was less than the smallest FID(P 2 , Q) estimate.

At m = 100 000 samples, however, the ordering of the estimates was correct in each of 100 trials, with FID(P 1 , Q) ≈ 1128.0 (sd 0.1) and FID(P 2 , Q) ≈ 1126.4 (sd 0.4).

This behavior was similar for other random draws of C.This example thus gives a case where, for the dimension and sample sizes at which we actually apply the FID and for somewhat-realistic distributions, comparing two models based on their FID estimates will not only not reliably give the right ordering -with relatively close true values and high dimensions, this is not too surprising -but, more distressingly, will reliably give the wrong answer, with misleadingly small variance.

This emphasizes that unbiased estimators, like the natural KID estimator, are important for model comparison.

We can also show, using the reasoning of Bickel & Lehmann (1969) that we also employed in Theorem 3, that there is no estimator of the FID which is unbiased for all distributions.

Fix a target distribution Q, and define the quantity F (P) = FID(P, Q).

Also fix two distributions P 0 = P 1 .

Suppose there exists some estimatorF (X) based on a sample of size n for which DISPLAYFORM0 This function R(α) is therefore a polynomial in α of degree at most n.

But let's consider the following one-dimensional case: DISPLAYFORM1 The mean and variance of (1 − α)P 0 + αP 1 can be written as DISPLAYFORM2 Note that (µ α − µ) 2 + σ 2 α + σ 2 is a quadratic function of α.

However, σ α is polynomial in α only in the trivial case when P 0 = P 1 .

Thus R(α) is not a polynomial when P 0 = P 1 , and so no estimator of the FID to an arbitrary fixed normal distribution Q can be unbiased on any class of distributions which includes two-component Gaussian mixtures.

There is also no unbiased estimator is available in the two-sample setting, where Q is also unknown, by the same trivial extension to this argument as in Theorem 3.Unfortunately, this type of analysis can tell us nothing about whether there exists an estimator which is unbiased on normal distributions.

Given that the distributions used for the FID in practice are clearly not normal, however, a practical unbiased estimator of the FID is impossible.

We replicate here the experiments of Heusel et al.'s Appendix 1, which examines the behavior of the Inception and FID scores as images are increasingly "disturbed," and additionally consider the KID.

As the "disturbance level" α is increased, images are altered more from the reference distribution.

FIG2 show the FID, KID, and negative (for comparability) Inception score for both CelebA (left) and CIFAR-10 (right); each score is scaled to [0, 1] to be plotted on one axis, with minimal and maximal values shown in the legend.

Note that Heusel et al. compared means and variances computed on 50 000 random disturbed CelebA images to those computed on the full 200 000 dataset; we instead use the standard traintest split, computing the disturbances on the 160 000-element training set and comparing to the 20 000-element test set.

In this (very slightly) different setting, we find the Inception score to be monotonic with increasing noise on more of the disturbance types than did Heusel et al. (2017) .

We also found similar behavior on the CIFAR-10 dataset, again comparing the noised training set (size 50 000) to the test set (size 10 000).

This perhaps means that the claimed non-monotonicity of the Inception score is quite sensitive to the exact experimental setting; further investigation into this phenomenon would be intriguing for future work.

MNIST After training for 50 000 generator iterations, all variants achieved reasonable results.

Among MMD models, only the distance kernel saw an improvement with more neurons in the top layer.

TAB4 Examining samples during training, we observed that rbf more frequently produces extremely "blurry" outputs, which can persist for a substantial amount of time before eventually resolving.

This makes sense, given the very fast gradient decay of the rbf kernel: when generator samples are extremely far away from the reference samples, slight improvements yield very little reward for the generator, and so bad samples can stay bad for a long time.

Scores for various models trained on CIFAR-10 are shown in TAB5 .

The scores for rq with a small critic network approximately match those of WGAN-GP with a large critic network, at substantially reduced computational cost.

With a small critic, WGAN-GP, Cramér GAN and the distance kernel all performed very poorly.

Samples from these models are presented in FIG2 .

FIG2 : Samples from the models listed in TAB4 .

Rational-quadratic and Gaussian kernels obtain retain sample quality despite reduced discriminator complexity.

Each of these models generates good quality samples with the standard DCGAN discriminator (critic size 64).

@highlight

Explain bias situation with MMD GANs; MMD GANs work with smaller critic networks than WGAN-GPs; new GAN evaluation metric.
This paper presents a Mutual Information Neural Estimator (MINE) that is linearly scalable in dimensionality as well as in sample size.

MINE is  back-propable and we prove that it is strongly consistent.

We illustrate a handful of applications in which MINE is succesfully applied  to enhance the property of generative models in both unsupervised and supervised settings.

We apply our framework to estimate the information bottleneck, and apply it in tasks related to supervised classification problems.

Our results  demonstrate substantial added flexibility and improvement in these settings.

Mutual information is an important quantity for expressing and understanding the relationship between random variables.

As a fundamental tool of data science, it has found application in a range of domains and tasks, including applications to biomedical sciences, blind source separation (BSS, e.g., independent component analysis, BID23 , information bottleneck (IB, BID45 , feature selection BID28 BID36 , and causality BID8 .In contrast to correlation, mutual information captures the absolute statistical dependency between two variables, and thus can act as a measure of true dependence.

Put simply, mutual information is the shared information of two random variables, X and Z, defined on the same probability space, (X ⇥ Z, F), where X ⇥ Z is the domain over both variables (such as R m ⇥ R n ), and F is the set of all possible outcomes over both variables.

The mutual information has the form 1 : DISPLAYFORM0 where P XZ : F ! [0, 1] is a probabilistic measure (commonly known as a joint probability distribution in this context), and P X = R Z dP XZ and P Z = R X dP XZ are the marginals.

The mutual information is notoriously difficult to compute.

Exact computation is only tractable with discrete variables (as the sum can be computed exactly) or with a limited family of problems where the probability distributions are known and for low dimensions.

For more general problems, common approaches include binning BID18 BID14 , kernel density estimation BID32 BID28 , Edgeworth expansion based estimators BID47 and likelihood-ratio estimators based on support vector machines (SVMs, e.g., BID43 .

While the mutual information can be estimated from empirical samples with these estimators, they still make critical assumptions about the underlying distribution of samples, and estimate errors can reflect this.

In addition, these estimators typically do not scale well with sample size or dimension.

More recently, there has been great progress in the estimation of f -divergences BID34 and integral probability metrics (IPMs, Sriperumbudur et al., 2009 ) using deep neural networks (e.g., in the context of f -divergences and the Wasserstein distance or Fisher IPMs, BID35 BID4 BID33 .

These methods are at the center of generative adversarial networks (GANs Goodfellow et al., 2014) , which train a generative model without any explicit assumptions about the underlying distribution of the data.

One perspective on these works is that, given the correct constraints on a neural network, the network can be used to compute a variational lower-bound on the distance or divergence of implicit probability measures.

In this paper we look to extend this estimation strategy to mutual information as given in equation 1, which we note corresponds to the Kullback-Leibler (KL-) divergence BID27 between the joint, P XZ and the product of the marginal distributions, P X ⌦ P Z , i.e., D KL (P XZ || P X ⌦ P Z ).

This observation can be used to help formulate variational Bayes in terms of implicit distributions BID30 or INFOMAX BID7 .We introduce an estimator for the mutual information based on the Donsker-Varadhan representation of the KL-divergence BID38 .

As with those introduced by BID35 , our estimator is scalable, flexible, and is completely trainable via back-propagation.

The contributions of this paper are as follows.• We introduce the mutual information neural estimator (MINE), providing its theoretical bases and generalizability to other information metrics.•

We illustrate that our estimator can be used to train a model with improved support coverage and richer learned representation for training adversarial models (such as adversariallylearned inferences, ALI, Dumoulin et al., 2016 ).•

We demonstrate how to use MINE to improve reconstructions and inference in Adversarially Learned Inference Dumoulin et al. FORMULA0 on large scale Datasets.• We show that our estimator provides a method of performing the Information Bottleneck method BID45 in a continuous setting, and that this approach outperforms variational bottleneck methods BID1 .

Mutual information is a Shannon entropy-based measure of dependence between random variables.

Following the definition in Equation 1, the mutual information can be understood as the decrease in the uncertainty of X given Z: DISPLAYFORM0 where H is the Shannon entropy and H(Z | X) is the conditional entropy of Z given X (the amount of information in Z not given from X).

Using simple manipulation, we write the mutual information as the Kullback-Leibler (KL-) divergence between the joint, P XZ , and the product of the marginals P X ⌦ P Z : DISPLAYFORM1 where H(X, Z) is the joint entropy of X and Z. It can be noted here that the mutual information is zero exactly when the KL-divergence is zero.

The intuitive meaning is immediately clear: the larger the divergence between the joint and the product of the marginals, the stronger the dependence between X and Z.There is also a strong connection between the mutual information and the structure between random variables.

We briefly touch upon this subject in Appendix 6.1.

MINE relies on the Donsker-Varadhan representation of the KL-divergence, which provides a tight lower-bound on the mutual information.

The KL-divergence between two probability distributions P and Q on a measure space ⌦, with P absolutely continuous with respect to Q, is defined as DISPLAYFORM0 where the argument of the log is the density ratio 2 and E P denotes the expectation with respect to P. It follows from Jensen's inequality that the KL-divergence is always non-negative and vanishes if and only if P = Q.The following theorem gives a variational representation of the KL-divergence: Theorem 1 (Donsker-Varadhan representation).

The KL divergence between any two distributions P and Q, with P ⌧ Q, admits the following dual representation BID16 : DISPLAYFORM1 where the supremum is taken over all functions T such that the two expectations are finite.

Given any subclass F of such functions, this yields the lower bound: DISPLAYFORM2 The bound in Equation 6 is known as the compression lemma in the PAC-Bayes literature BID5 .

A simple proof goes as follows.

Given T 2 F, consider the Gibbs distribution G defined by DISPLAYFORM3 .

By construction, DISPLAYFORM4 The gap between left and right hand sides of Equation 6 can then be written as: DISPLAYFORM5 and we conclude by the positivity of the KL-divergence.

The identity (8) also shows that the bound is tight whenever G = P, namely for optimal functions T ⇤ taking the form DISPLAYFORM6 for some constant C 2 R.It is interesting to compare the Donsker-Varadhan bound with other variational bounds proposed in the literature.

The variational divergence estimation proposed in BID34 and used in BID35 and BID30 , leads to the following bound: DISPLAYFORM7 Although both bounds are tight for sufficiently large families F, the Donsker-Varadhan bound is stronger in the sense that for any fixed T , the right hand side of Equation 6 is larger than the right hand side 3 of Equation 10.

We perform numerical comparisons in Section 4.1.We refer to the work by BID38 for a derivation of both representations (6) and (10) from unifying point of view of Fenchel duality, in the more general context of f -divergences.

We are interested in the case of a joint random variable (X, Z) on a joint probability space ⌦ = X ⇥ Z, and where P = P XZ is the joint distribution, Q = P X ⌦ P Z is the product distribution.

P is then absolutely continuous with respect to Q. Using the expression (3) for the mutual information in terms of a KL-divergence, we obtain the following representation: DISPLAYFORM0 2 Although the discussion is more general, we can think of P and Q as being distributions on some compact domain ⌦ ⇢ R d , with density p and q respect the Lebesgue measure , so that DKL = R p log p q d .

3 To see this, just apply the identity x e log x with DISPLAYFORM1 The inequality in Equation 11 is intuitive in terms of deep learning optimization.

The idea is to parametrize the functions T : X ⇥ Z !

R in F by a deep neural network with parameters ✓ 2 ⇥, turning the infinite dimensional problem into a much easier parametric optimization problem.

In the following we call T ✓ the statistic network.

The expectations in the above lower-bound can then be estimated by Monte-Carlo (MC) sampling using empirical samples (x, z) ⇠ P XZ .

Samples x ⇠ P X andz ⇠ P Z from the marginals are obtained by simply dropping x, z from samples (x, z) and (x,z) ⇠ P XZ .

The objective can be maximized by gradient ascent.

In what follows we use the notationP DISPLAYFORM2 X for the empirical distribution associated to a given set of n iid samples drawn for P X .

If we denotê DISPLAYFORM3 as the optimal set of parameters under the above conditions, we obtain the Mutual Information Neural Estimator (MINE):Definition 3.1 (Mutual information neural estimator (MINE)).

DISPLAYFORM4 Algorithm 1 presents details of the implementation of MINE.

DISPLAYFORM5 .

Evaluate the lower-bound DISPLAYFORM6 .

Update the statistic network parameters until convergenceWe will also use an adaptive gradient clipping method to ensure stability whenever MINE is used in conjunction with another adversarial objective.

The details of this are provided in Appendix 6.3.

In this section we discuss the consistency of MINE.

The estimator relies on (i) a neural network architecture and (ii) a choice of n samples from the data distribution P XZ .

We define consistency in the following way: Definition 3.2 (Consistency).

The estimator \ I(X; Z) n is (strongly) consistent if for all ✏ > 0, then there exists a positive integer N and a choice of neural network architecture such that: DISPLAYFORM0 In other words, the estimator converges to the true mutual information as n !

1, almost surely over the choice of samples.

The question of consistency breaks into two problems: an approximation problem related to the size of the family F, and inducing the gap in the inequality (11) ; and an estimation problem related to the use of empirical measures in (12).

The first problem is addressed by the universal approximation theorem for neural networks BID22 .

For the second problem, classical consistency theorems for extremum estimators apply BID46 , under mild conditions on the parameter space.

This leads to the two lemmas below.

The proofs are given in Appendix 6.2.

In what follows we use the notationÎ[T ] for the argument of the supremum in Equation FORMULA0 : DISPLAYFORM1 There exists a feedforward network function T✓ : ⌦ !

R such that DISPLAYFORM2 A fortiori if F is any family of functions having T✓ as one of its elements, DISPLAYFORM3 Lemma 2.

Let ⌘ > 0.

Let F be the family of functions T ✓ : ⌦ !

R defined by a given network architecture.

We assume the parameters ✓ are restricted to some compact domain ⇥ ⇢ R k .

Then there exists N 2 N such that DISPLAYFORM4 These results lead to the following consistency theorem.

Theorem 2.

MINE as defined by Equ.

12 and 13 is a (strongly) consistent.

Proof.

Let ✏ > 0.

We apply the two Lemma to find a a family of neural network function F and N 2 N such that FORMULA0 and FORMULA0 hold with ⌘ = ✏/2.

By the triangular inequality, for all n N and with probability one, we have that DISPLAYFORM5 which proves consistency.

We close this section by pointing out that the previous construction can be extended to more general information measures based on so-called f -divergences BID2 : DISPLAYFORM0 indexed by a convex function f : [0, 1) !

R such that f (1) = 0.

The KL-divergence is a special case of f -divergence with f (u) = u log(u).

Just as the mutual information can be understood as the KL-divergence between the joint and product of marginals distributions, we can define a family of f -information measures as f -divergences: DISPLAYFORM1 An analogue for f -divergences of the Donsker-Varadhan representation of Theorem 1 can be found in BID38 .

The key idea is to express f -divergences in terms of convex operators, and to leverage Fenchel-Legendre duality to obtain variational representation in terms of the convex conjugate BID37 .

This allows a straightforward extension of MINE to a mutual finformation estimator, following the construction of of the previous section.

The study of such information measures and their estimators is left for future work.

In this section, we present applications of mutual information through the mutual information neural estimator (MINE), as well as competing methods that are designed to achieve the same goals.

We also present experimental results touching on each of these applications.

Mutual information is an important quantity for analyzing and understanding the statistical dependencies between random variables.

The most straightforward application for MINE then is estimation of the mutual information.

Related works on estimating mutual information There are a number of methods that can also be used to estimate mutual information given only empirical samples of the joint distribution of variables of interest.

The fundamental difficulty in estimation is the intractability of joint and product of marginals, as exact computation requires integration over the joint of continuous variables.

BID26 proposes a k-NN estimator based on estimating the entropy terms of the mutual information; and this comes with the usual limitations of non-parametric methods.

Van Hulle FORMULA1 presents an estimator built around the Edgeworth series BID21 .

The entropy of the distribution is approximated by a Gaussian with additional correction brought by higher-order cumulants.

This method is only tractable in very low-dimensional data and breaks down when departure from Gaussianity is too severe.

BID43 exploits a likelihood-ratio estimator using kernel methods.

Other recent works include BID24 ; BID40 ; BID31 .MINE, on the other hand, inherits all the benefits of neural networks in scalability and can, in principle, calculate the mutual information using a large number of high-dimensional samples.

We posit then that, given empirical samples of two random variables, X and Z, and a high-enough capacity neural network, MINE will provide good estimates for the mutual information without the necessary constraints of the methods mentioned above.

Experiment: estimating mutual information between two Gaussians We begin by comparing MINE to the k-means-based non-parametric estimator found in BID26 .

In our experiment, we consider two bivariate Gaussian random variables X a and X b with correlation, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0., 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] .

As the mutual information is invariant to continuous bijective transformation of the considered variables, it is enough to consider standardized Gaussians marginals.

We also compare two versions of MINE: the version of the current paper based on the Donsker-Varadhan representation 5 of the KL divergence; and the one based on the f -divergence representation 10 proposed by BID34 and used in BID35 and BID30 .

DISPLAYFORM0 Our results are presented in Figure 1 and 2.

We observe that both MINE and Kraskov's estimation are virtually indistinguishable from the ground truth; and that MINE provides a much tighter estimate of the mutual information than the version using the bound of BID34 .

Mode-dropping BID10 ) is a common pathology of generative adversarial networks (GANs, BID19 where the generator does not generate all of the modes in the target dataset (such as not generating images that correspond to specific labels).

We identify at least two source of mode dropping in GANs:• Discriminator liability: In this case, the discriminator classifies only a fraction of the real data as real.

As a consequence of this, there is no gradient for the generator to learn to generate modes that have poor representation under the discriminator.

• Generator liability: The generator is greedy and concentrates its probability mass on the smallest subset most likely to fool the discriminator.

Here, the generator simply focuses on a subset of modes which maximize the discriminator's Bayes risk.

We focus here on the second type of mode-dropping.

In order to alleviate the greedy behavior of the generator, we encourage the generator to maximize the entropy of the generated data.

This can be achieved by modifying the GAN objective for the generator with a mutual information term.

Our treatment involves the typical GAN setting in BID19 .

We denote by p real the real data distribution on X , and by p gen the generated distribution, induced by a function 4 G : Z !

X from a (relatively simple, such as a spherical Gaussian) prior density p(z), so that DISPLAYFORM0 for all functions f on X .

In this setting, the discriminator D : X !

R, which is modeled by a deep neural network with sigmoid nonlinearity, is optimized so as to maximize the value function: DISPLAYFORM1 As observed in BID35 , maximizing the value function amounts to maximizing the variational lower-bound of 2 ⇤ D JS (P||Q) 2 log 2, where D JS is the Jensen-Shannon divergence.

The generator is then optimized to minimize V alternatively as the discriminator maximizes it.

In practice, however, we will use a proxy to be maximized by the generator, E pgen [log(D(x)], which can palliate vanishing gradients.

In order to palliate mode-dropping, our strategy is to maximize the entropy of the generated data.

Since G(Z) is a deterministic function of Z, the conditional entropy H(G(Z)|Z) is zero and thus DISPLAYFORM2 In other words, the entropy can be estimated using MINE.

The generator objective then becomes: DISPLAYFORM3 As the samples G(z) are differentiable w.r.t.

the parameters of G and MINE is a completely differentiable function, we can maximize the mutual information using back-propagation and gradient descent by only specifying this additional loss term.

Since the mutual information is unbounded, we use adaptive gradient clipping to ensure stability (see Appendix 6.3).Related works on mode-dropping In mode regularized GANs, BID10 proposes to learn a reconstruction distribution, then teach the generator to sample from it.

The intuition behind this is that the reconstruction distribution is a de-noised or smoothed version of the data distribution, and thus easier to learn.

However, the connection to reducing mode dropping is only indirect.

InfoGAN ) is a method which attempts to improve mode coverage by leveraging the Agokov and Baber conditional entropy variational lower-bound BID6 .

This bound involves approximating the intractable conditional distribution P Z|X by using a tractable recognition network, F : X !

Z.

In this setting, the variational approach bounds the conditional entropy, H(X | Z), which effectively maximizes a variational lower bound on the entropy H(G(Z)).

VEEGAN Srivastava et al. (2017) , like InfoGAN, makes use of a recognition network to maximize the Agokov and Baber variational lower-bound, but is trained like adversarially learned inference (ALI, Dumoulin et al., 2016, , see the following section for details).

Since, at convergence the joint distributions of the generative and recognition networks are matched, this has the effect of minimizing the conditional entropy, H(X|Z).Our approach is closest to that of BID13 , where they also formulated a GAN with entropy regularization of the generator.

Interestingly, they show that, in the context of Energy-based GANs, such a regularization strategy yields a discriminator score function that at equilibrium is proportional to the log-density of the empirical distribution.

The main difference between their work and our regularized GAN formulation is that we use MINE to estimate entropy while they used a nonparametric estimate that does not scale particularly well with dimensionality of the data domain.

Experiment: swiss-roll and 25-Gaussians datasets Here, we apply MINE to improve mode coverage when training a generative adversarial network (GAN, BID19 .

Following Equation 21, we estimate the mutual information using MINE and use this estimate to maximize the entropy of the generator.

We demonstrate this effect on a Swiss-roll dataset, comparing two models, one with = 0 (which corresponds to the orthodox GAN as in BID19 ) and one with = 1.0, which corresponds to entropy-maximization.

Our results on the swiss-roll FIG2 ) and the 25-Gaussians FIG3 ) datasets show improved mode coverage over the baseline with no mutual information objective.

This confirms our hypothesis that maximizing mutual information helps against mode-dropping in this simple setting.

Adversarial bi-directional models are an extension of GANs which incorporate a reverse model F : X !

Z. These were introduced in adversarially-learned inference (ALI, Dumoulin et al., 2016) , closely related BiGAN BID15 , and variants that minimize the condi- tional entropy (ALICE, BID29 .

These models train a discriminator to maximize the value function of Equation 19 over the two joint distributions p enc (x, z) = p enc (z|x)p(x) and p dec (x, z) = p dec (x|z)p(z) over X ⇥ Z, induced by the forward (encoder) and reverse (decoder) models, respectively.

In principle, ALI should be able to learn a feature representation as well as palliate mode dropping.

However, in practice ALI guarantees neither due to identifiability issues BID29 .

This is further evident as the generated samples from the forward model can be poor reconstructions of the data given the inferred latent representations from the reverse model.

In order to address these issues, ALICE introduces an additional term to minimize the conditional entropy by minimizing the reconstruction error.

To demonstrate the connection to mutual information, it can be shown (see the Appendix, Section 6.4, for a proof) that the reconstruction error is bounded as: DISPLAYFORM0 If H penc (Z) is fixed (which can be accomplished in how the reverse model is defined), then matching the joint distributions during training in addition to maximizing the mutual information between X and Z will lower the reconstruction error.

In order to ensure H penc (Z) is fixed, we model the conditional density p(z|x) with a deep neural network that outputs the means µ = F (x) of a spherical Gaussian with fixed variance = 1.

We assume that the generating distribution is the same as with GANs in the previous section.

The objectives for training a bi-directional adversarial model then becomes: DISPLAYFORM1 We will show that a bi-directional model trained in this way has the benefits of higher mutual information, including better mode coverage and reconstructions.

Experiment: bi-directional adversarial model with mutual information maximization In this section we compare MINE to existing bi-directional adversarial models in terms of euclidean reconstructions, reconstruction accuracy, and MS-SSIM metric BID48 .

One of the potential features of a good generative model is how close the reconstructions are to the original in pixel space.

Adding MINE to a bi-directional adversarial model gets us closer to this objective.

We train MINE on datasets of increasing order of complexity: a toy dataset composed of 25-Gaussians, MNIST, and the CelebA dataset.

FIG4 shows the reconstruction ability of MINE compared to ALI.

Although ALICE does perfect reconstruction (which is in its explicit formulation), we observe significant mode-dropping in the sample space.

MINE does a balanced job of reconstructing along with capturing all the modes of the underlying data distribution.

ALICE with the adversarial loss has the best reconstruction, though at the expense of sample quality.

Overall, MINE provides both very good reconstructions and the best mode representation in its samples.

Next, we use MS-SSIM BID48 scores to measure the likelihood of generated samples within the class.

Table 1 compares MINEto the existing baselines in terms of euclidean reconstruction errors, reconstruction accuracy, and MS-SSIM metric.

MINE does a better job than ALI in terms of reconstruction errors by a good margin and is competitive to ALICE with respect to reconstruction accuracy and MS-SSIM.

Table 1 : Comparison of MINE with other bi-directional adversarial models in terms of euclidean reconstruction error, reconstruction accuracy, and ms-ssim on MNIST dataset.

We used MLP both in the generator and discriminator identical to the setting described in BID39 and MLP Statistics network for this task.

MINE does a decent job compared to ALI in terms of reconstructions.

Though the explicit reconstruction based baselines do better than MINE in terms of tasks related to reconstructions, they lag behind in MS-SSIM scores.

The Information Bottleneck (IB, BID45 is an information theoretic method for extracting relevant information, or yielding a representation, that an input X 2 X contains about an output Y 2 Y. An optimal representation of X would capture the relevant factors and compress X by diminishing the irrelevant parts which do not contribute to the prediction of Y .

IB was recently TAB0 : Comparison of MINE with other bi-directional adversarial models in terms of euclidean reconstruction error, reconstruction accuracy, and MS-SSIM on CelebA faces dataset.

We can see that the trend remains same from MNIST results.

MINE achieves a substantial decrease in reconstruction errors without compromising on better MS-SSIM score.covered in the context of deep learning BID44 .

As such, IB can be seen as a process to construct an approximate of minimally sufficient statistics of the data.

IB seeks a feature map, or encoder, q(Z | X), that would induce the Markovian structure X !

Z !

Y .

This is done by minimizing the IB Lagrangian, DISPLAYFORM0 which appears as a the standard cross-entropy loss augmented with a regularizer promoting minimality of the representation BID0 .

Here we propose to estimate the regularizer with MINE.Related works and information bottleneck with MINE In the discrete setting, BID45 uses the Blahut-Arimoto Algorithm BID3 , which can be understood as cyclical coordinate ascent in function spaces.

While the information bottleneck is successful and popular in a discrete setting, its application to the continuous setting was stifled by the intractability of the continuous mutual information.

Nonetheless, the Information Bottleneck was applied in the case of jointly Gaussian random variables in BID11 .

MINE estimate the mutual information directly.

As such, it allows for general posterior as it does not require densities.

Thus MINE allows the use of general encoders/posteriors.

Experiment: Permutation-invariant MNIST classification Here, we demonstrate an implementation of the Information Bottleneck objective on a permutation invariant MNIST using MINE.

We use a similar setup as BID1 , except that we do not use their approach to averaging the weights.

The architecture of the encoder is an MLP with two hidden layers and an output of 256 dimensions.

The decoder is a simple softmax.

As Alemi et al. FORMULA0 is using a variational bound on the conditional entropy, their approach requires a tractable density.

They opt for a conditional Gaussian encoder z = µ(x) + ✏, where ✏ ⇠ N (0, I).

As MINE does not require a tractable density, we consider three type of encoders:• A Gaussian encoder as in BID1 • An additive noise encoder, z = enc(x + ✏) DISPLAYFORM1 Our results can be seen in Table 3 , and this shows MINE as being superior in all of these settings.

Table 3 : Permutation Invariant MNIST misclassification rate using information bottleneck methods.

We proposed a mutual information estimator, which we called the mutual information neural estimator (MINE), that is scalable in dimension and sample-size.

We demonstrated the efficiency of this estimator by applying it in a number of settings.

First, a term of mutual information can be introduced alleviate mode-dropping issue in generative adversarial networks (GANs, Goodfellow et al., 2014) .

Mutual information can also be used to improve inference and reconstructions in adversarially-learned inference (ALI, Dumoulin et al., 2016) .

Finally, we showed that our estimator allows for tractable application of Information bottleneck methods BID45 in a continuous setting.through co-occurrence.

We illustrate this perspective by considering distributions on natural image manifolds.

Consider a random image in [0, 1] d by randomly sampling the intensity of each pixel independently.

This image will show very little structure when compared to an image sampled form the manifold of natual images, M nature ⇢ [0, 1] d , as the latter is is bound to respect a number of physically possible priors (such as smoothness).

We expect the mutual information of the pixels of images arising from M nature to be high.

Differently put, the larger the number of simultaneously co-occurring subset of pixels in [0, 1] d , the higher the mutual information.

In the language of cumulants tensors, the larger ponderation of higher order cumulants tensor in the cumulant generating function of the joint distribution over the pixels, the higher the mutual information, and the more structure there is to be found amongst the pixels.

Note that the case of mutually independent pixels corresponds to joint distribution where the only cumulants contributing the joint distribution are of order one.

This is the corner case where the joint distribution equals the product of marginals.

Thus in order to assess the amount of structure it is enough to score how the joint distribution is different from the product of marginals.

As we show in the paper, this principle can be extended to different divergences as well.

This section presents the proofs of the Lemma leading to the consistency result in Theorem 2.

In what follows, we will assume that the input space ⌦ = X ⇥ Z is a compact domain of R d and all measures are absolutely continuous with respect to the Lebesgue measure.

To avoid unnecessary heavy notation, we denote P = P XZ and Q = P X ⌦ P Z for the joint distribution and product of marginals.

We will restrict to families of feedforward functions with continuous activations, with a single output neuron, so that a given architecture defines a continuous mapping (!, ✓) !

T ✓ (!) from ⌦ ⇥ ⇥ to R.

Consider the function T ⇤ = log

Here we clarify relationship between reconstruction error and mutual information, by proving the bound in Equ 22.

We begin with a definition: Definition 6.1 (Reconstruction Error).

We consider encoder and decoder models giving conditional distributions p enc (z|x) and p dec (x|z) over the data and latent variables.

If p(x) denotes the marginal data distribution, the reconstruction error is defined as DISPLAYFORM0 We can rewrite the reconstruction error in terms of the joints p enc (x, z) = p enc (z|x)p(x) and p dec (x, z) = p dec (x|z)p(z).

Elementary manipulations give:R = E (x,z)⇠penc log p enc (x, z) p dec (x, z) E (x,z)⇠penc log p enc (x, z) + E z⇠penc(z) log p(z)where p enc (z) is the marginal on Z induced by the encoder.

The first term is the KL-divergence D KL (p enc || p dec ) ; the second term is the joint entropy H penc (x, z).

The third term can be written as E z⇠penc(z) log p(z) = D KL (p enc (z) || p(z)) H penc (z)Finally, the identity DISPLAYFORM1 yields the following expression for the reconstruction error: DISPLAYFORM2 Since the KL-divergence is positive, we obtain the bound: DISPLAYFORM3 which is tight whenever the induced marginal p enc (z) matches the prior distribution p(z).

@highlight

A scalable in sample size and dimensions mutual information estimator.
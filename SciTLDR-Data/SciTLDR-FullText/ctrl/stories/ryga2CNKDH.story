Deep generative models have achieved remarkable progress in recent years.

Despite this progress, quantitative evaluation and comparison of generative models remains as one of the important challenges.

One of the most popular metrics for evaluating generative models is the log-likelihood.

While the direct computation of log-likelihood can be intractable, it has been recently shown that the log-likelihood of some of the most interesting generative models such as variational autoencoders (VAE) or generative adversarial networks (GAN) can be efficiently estimated using annealed importance sampling (AIS).

In this work, we argue that the log-likelihood metric by itself cannot represent all the different performance characteristics of generative models, and propose to use rate distortion curves to evaluate and compare deep generative models.

We show that we can approximate the entire rate distortion curve using one single run of AIS for roughly the same computational cost as a single log-likelihood estimate.

We evaluate lossy compression rates of different deep generative models such as VAEs, GANs (and its variants) and adversarial autoencoders (AAE) on MNIST and CIFAR10, and arrive at a number of insights not obtainable from log-likelihoods alone.

Generative models of images represent one of the most exciting areas of rapid progress of AI (Brock et al., 2019; Karras et al., 2018b; a) .

However, evaluating the performance of generative models remains a significant challenge.

Many of the most successful models, most notably Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) , are implicit generative models for which computation of log-likelihoods is intractable or even undefined.

Evaluation typically focuses on metrics such as the Inception score (Salimans et al., 2016) or the Fréchet Inception Distance (FID) score (Heusel et al., 2017) , which do not have nearly the same degree of theoretical underpinning as likelihood-based metrics.

Log-likelihoods are one of the most important measures of generative models.

Their utility is evidenced by the fact that likelihoods (or equivalent metrics such as perplexity or bits-per-dimension) are reported in nearly all cases where it's convenient to compute them.

Unfortunately, computation of log-likelihoods for implicit generative models remains a difficult problem.

Furthermore, log-likelihoods have important conceptual limitations.

For continuous inputs in the image domain, the metric is often dominated by the fine-grained distribution over pixels rather than the high-level structure.

For models with low-dimensional support, one needs to assign an observation model, such as (rather arbitrary) isotropic Gaussian noise (Wu et al., 2016) .

Lossless compression metrics for GANs often give absurdly large bits-per-dimension (e.g. 10 14 ) which fails to reflect the true performance of the model (Grover et al., 2018; Danihelka et al., 2017) .

See Theis et al. (2015) for more discussion of limitations of likelihood-based evaluation.

Typically, one is not interested in describing the pixels of an image directly, and it would be sufficient to generate images close to the true data distribution in some metric such as Euclidean distance.

For this reason, there has been much interest in Wasserstein distance as a criterion for generative models, since the measure exploits precisely this metric structure Gulrajani et al., 2017; Salimans et al., 2018) .

However, Wasserstein distance remains difficult to approximate, and hence it is not routinely used to evaluate generative models.

We aim to achieve the best of both worlds by measuring lossy compression rates of deep generative models.

In particular, we aim to estimate the rate distortion function, which measures the number of bits required to match a distribution to within a given distortion.

Like Wasserstein distance, it can exploit the metric structure of the observation space, but like log-likelihoods, it connects to the rich literature of probabilistic and information theoretic analysis of generative models.

By focusing on different parts of the rate distortion curve, one can achieve different tradeoffs between the description length and the fidelity of reconstruction -thereby fixing the problem whereby lossless compression focuses on the details at the expense of high-level structure.

It has the further advantage that the distortion metric need not have a probabilistic interpretation; hence, one is free to use more perceptually valid distortion metrics such as structural similarity (SSIM) (Wang et al., 2004) or distances between hidden representations of a convolutional network (Huang et al., 2018) .

Algorithmically, computing rate distortion functions raises similar challenges to estimating loglikelihoods.

We show that the rate distortion curve can be computed by finding the normalizing constants of a family of unnormalized probability distributions over the noise variables z. Interestingly, when the distortion metric is squared error, these distributions correspond to the posterior distributions of z for Gaussian observation models with different variances; hence, the rate distortion analysis generalizes the evaluation of log-likelihoods with Gaussian observation models.

Annealed Importance Sampling (AIS) (Neal, 2001 ) is currently the most effective general-purpose method for estimating log-likelihoods of implicit generative models, and was used by Wu et al. (2016) to compare log-likelihoods of a variety of implicit generative models.

The algorithm is based on gradually interpolating between a tractable initial distribution and an intractable target distribution.

We show that when AIS is used to estimate log-likelihoods under a Gaussian observation model, the sequence of intermediate distributions corresponds to precisely the distributions needed to compute the rate distortion curve.

Since AIS maintains a stochastic lower bound on the normalizing constants of these distributions, it automatically produces an upper bound on the entire rate distortion curve.

Furthermore, the tightness of the bound can be validated on simulated data using bidirectional Monte Carlo (BDMC) (Grosse et al., 2015; Wu et al., 2016) .

Hence, we can approximate the entire rate distortion curve for roughly the same computational cost as a single log-likelihood estimate.

We use our rate distortion approximations to study a variety of variational autoencoders (VAEs) (Kingma & Welling, 2013) , GANs and adversarial autoencoders (AAE) (Makhzani et al., 2015) , and arrive at a number of insights not obtainable from log-likelihoods alone.

For instance, we observe that VAEs and GANs have different rate distortion tradeoffs: While VAEs with larger code size can generally achieve better lossless compression rates, their performances drop at lossy compression in the low-rate regime.

Conversely, expanding the capacity of GANs appears to bring substantial reductions in distortion at the high-rate regime without any corresponding deterioration in quality in the low-rate regime.

We find that increasing the capacity of GANs by increasing the code size (width) has a qualitatively different effect on the rate distortion tradeoffs than increasing the depth.

We also find that that different GAN variants with the same code size achieve nearly identical RD curves, and that the code size dominates the performance differences between GANs.

Annealed importance sampling (AIS) (Neal, 2001 ) is a Monte Carlo algorithm based on constructing a sequence of n intermediate distributions

, where k ∈ {0, . . .

, n}, between a tractable initial distribution p 0 (z) and the intractable target distribution p n (z).

At the the k-th state (0 ≤ k ≤ n), the forward distribution q f and the un-normalized backward distributionq b are

where T k is an MCMC kernel that leaves p k (z) invariant; andT k is its reverse kernel.

We run M independent AIS chains, numbered i = 1, . . .

, M . Let z i k be the k-th state of the i-th chain.

The

Bidirectional Monte Carlo.

We know that the log partition function estimate logẐ is a stochastic lower bound on log Z (Jensen's inequality).

As the result, using the forward AIS distribution as the proposal distribution results in a lower bound on the data log-likelihood.

By running AIS in reverse, however, we obtain an upper bound on log Z. However, in order to run the AIS in reverse, we need exact samples from the true posterior, which is only possible on the simulated data.

The combination of the AIS lower bound and upper bound on the log partition function is called bidirectional Monte Carlo (BDMC), and the gap between these bounds is called the BDMC gap (Grosse et al., 2015) .

We note that AIS combined with BDMC has been used to estimate log-likelihoods for implicit generative models (Wu et al., 2016) .

In this work, we validate our AIS experiments by using the BDMC gap to measure the accuracy of our partition function estimators.

Let x be a random variable that comes from the data distribution p d (x).

Shannon's fundamental compression theorem states that we can compress this random variable losslessly at the rate of H(x).

But if we allow lossy compression, we can compress x at the rate of R, where R ≤ H(x), using the code z, and have a lossy reconstructionx = f (z) with the distortion of D, given a distortion measure d(x,x) = d(x, f (z)).

The rate distortion theory quantifies the trade-off between the lossy compression rate R and the distortion D. The rate distortion function R(D) is defined as the minimum number of bits per sample required to achieve lossy compression of the data such that the average distortion measured by the distortion function is less than D. Shannon's rate distortion theorem states that R(D) equals the minimum of the following optimization problem: min

where the optimization is over the channel conditional distribution q(z|x).

Suppose the datadistribution is p d (x).

The channel conditional q(z|x) induces the joint distribution q(z, x) = p d (x)q(z|x), which defines the mutual information I(z; x).

q(z) is the marginal distribution over z of the joint distribution q(z, x), and is called the output marginal distribution.

We can rewrite the optimization of Eq. 5 using the method of Lagrange multipliers as follows: min

The goal of generative modeling is to learn a model distribution p(x) to approximate the data distribution p d (x).

Implicit generative models define the model distribution p(x) using a latent variable z with a fixed prior distribution p(z) such as a Gaussian distribution, and a decoder or generator network which computesx = f (z).

In some cases (e.g. VAEs, AAEs), the generator explicitly parameterizes a conditional distribution p(x|z), such as a Gaussian observation model N (x; f (z), σ 2 I).

But in implicit models such as GANs, the generator directly outputsx = f (z).

In order to treat VAEs and GANs under a consistent framework, we ignore the Gaussian observation model of VAEs (thereby treating the VAE decoder as an implicit model), and use the squared error distortion of

2 .

However, we note that it is also possible to assume a Gaussian observation model with a fixed σ 2 for GANs, and use the Gaussian negative log-likelihood (NLL) as the distortion measure for both VAEs and GANs: d(x, f (z)) = − log N (x; f (z), σ 2 I).

This is equivalent to squared error distortion up to a linear transformation.

In this section, we describe the rate-prior distortion function, as a variational upper bound on the true rate distortion function.

We must modify the standard rate distortion formalism slightly in order to match the goals of generative model evaluation.

Specifically, we are interested in evaluating lossy compression with coding schemes corresponding to particular trained generative models, including the fixed prior p(z).

For models such as VAEs, KL(q(z|x) p(z)) is standardly interpreted as the description length of z. Hence, we adjust the rate distortion formalism to use E p d (x) KL(q(z|x) p(z)) in place of I(x, z), and refer to this as the rate-prior objective.

The rate-prior objective upper bounds the standard rate:

In the context of variational inference, q(z|x) is the posterior, q(z) = p d (x)q(z|x)dx is the aggregated posterior (Makhzani et al., 2015) , and p(z) is the prior.

In the context of rate distortion theory, q(z|x) is the channel conditional, q(z) is the output marginal, and p(z) is the variational output marginal distribution.

The inequality is tight when p(z) = q(z), i.e., the variational output marginal (prior) is equal to the output marginal (aggregated posterior).

We note that the upper bound of Eq. 7 has been used in other algorithms such as the Blahut-Arimoto algorithm (Arimoto, 1972) or the variational information bottleneck algorithm (Alemi et al., 2016) .

Analogously to the rate distortion function, we define the rate-prior distortion function R p (D) as the minimum value of the rate-prior objective for a given distortion D. More precisely,

We can rewrite the optimization of Eq. 8 using the method of Lagrange multipliers as follows:

Conveniently, the Lagrangian decomposes into independent optimization problems for each x, allowing us to treat this as an optimization problem over q(z|x) for fixed x. We can compute the rate distortion curve by sweeping over β rather than by sweeping over D.

Now we describe some of the properties of the rate-prior distortion function R p (D), which are straightforward analogues of well-known properties of the rate distortion function.

Proposition 1.

R p (D) has the following properties:

is non-increasing and convex function of D.

(c) The rate-prior distortion optimization of Eq. 9 has a unique global optimum which can be expressed as q *

Proof.

The proofs are provided in Appendix C.1.

Prop.

1b states that for any prior p(z), R p (D) is a variational upper-bound on R(D).

More specifically, we have R(D) = min p(z) R(D), which implies that for any given β, there exists a prior p * β (z), for which the variational gap between rate distortion and rate-prior distortion functions at β is zero.

Fig. 1a shows a geometrical illustration of Prop.

1b for three values of β ∈ {0.25, 1, 4}. We can see in this figure that all R p (D) curves are upper bounds on R(D), and for any given β, R p * β (D) is tangent to both R p (D) and to the line with the slope of β passing through the optimal solution.

If the decoder outputs a probability distribution (as in a VAE), we can define the distortion metric to coincide with the negative log-likelihood (NLL): d(x, f (z)) = − log p(x|z).

We now describe some of the properties of the rate-prior distortion functions with NLL distortions.

Proposition 2.

The rate-prior distortion function R p (D) with NLL distortion of − log p(x|z) has the following properties:

The global optimum of rate-prior distortion optimization (Eq. 9) can be expressed as q *

(c) At β = 1, the negative summation of rate-prior and distortion is the true log-likelihood:

Proof.

The proofs are provided in Appendix C.2. .

At β = 1, let L * and L p be the negative summation of rate and distortion on the rate distortion and rate-prior distortion curves respectively (shown in Fig. 1b) .

From Prop.

2c we know that L p is the true log-likelihood of the generative model.

From Prop.

1b, we can conclude that L * = max p(z) L p .

This reveals an important relationship between rate distortion theory and generative modeling that was also observed in Lastras (2019): for a given generative model with a fixed conditional p(x|z), the best log-likelihood L p that can be achieved by optimizing the prior p(z) is the L * , which can be found by solving the rate distortion problem.

Furthermore, the corresponding optimal prior p * (z) is the output marginal of the optimal channel conditional of the rate distortion problem at β = 1.

Fig. 1b shows the rate-prior distortion function R p * (D) corresponding to p * (z).

In a "good" generative model, where the model distribution is close to the data-distribution, the negative log-likelihood −L p is close to the entropy of data H d , and the variational gap between R p (D) and R(D) is tight.

In the previous section, we introduced the rate-prior distortion function R p (D) and showed that it upper bounds the true rate distortion function R(D).

However, evaluating R p (D) is also intractable.

In this section, we show how we can upper bound R p (D) using a single run of the AIS algorithm.

AIS Chain.

We fix a temperature schedule 0 = β 0 < β 1 < . . .

< β n = ∞. For the k-th intermediate distribution, we use the optimal channel conditional q k (z|x) and partition function Z k (x), corresponding to points along R p (D) and derived in Prop.

1c:

Conveniently, this choice coincides with geometric averages, the typical choice of intermediate distributions for AIS, i.e, the k th step happens to be the optimal solutions for β k .

This chain is shown in Fig. 2 .

For the transition operator, we use Hamiltonian Monte Carlo (Neal et al., 2011) .

At the k-th step, the rate-prior R k (x) and the distortion D k (x) are

AIS Rate-Prior Distortion Curve.

For each data point x, we run M independent AIS chains, numbered i = 1, . . .

, M , in the forward direction.

At the k-th state of the i-th chain, let z i k be the state, w i k be the AIS importance weights, andw i k be the normalized AIS importance weights.

We denote the AIS distribution at the k-th step as the distribution obtained by first sampling from all

, and then re-sampling the samples based on their normalized importance weightsw i k (see Section 2.1 and Appendix C.4 for more details).

More formally q

Using the AIS distribution q AIS k (z|x) defined in Eq. 12, we now define the AIS distortion D AIS k (x) and the AIS rate-prior R AIS k (x) as follows:

We now define the AIS rate-prior distortion curve R

Proposition 3.

The AIS rate-prior distortion curve upper bounds the rate-prior distortion function:

Proof.

The proof is provided in Appendix C.4.

Estimated AIS Rate-Prior Distortion Curve.

Although the AIS distribution can be easily sampled from, its density is intractable to evaluate.

As the result, evaluating R

Having found the estimatesD

, we propose to estimate the rate as follows:

We define the estimated AIS rate-prior distortion curveR Fig. 1b ) as an RD curve obtained by tracing pairs of rate distortion estimates

.

Proposition 4.

The estimated AIS rate-prior distortion curve upper bounds the AIS rate-prior distortion curve in expectation:

Proof.

The proof is provided in Appendix C.4.

In summary, from Prop.

1, Prop.

3 and Prop.

4, we can conclude that the estimated AIS rate-prior distortion curve upper bounds the true rate distortion curve in expectation (shown in Fig. 1b ):

In all the experiments, we plot the estimated AIS rate-prior distortion functionR Accuracy of AIS Estimates.

While the above discussion focuses on obtaining upper bounds, we note that AIS is one of the most accurate general-purpose methods for estimating partition functions, and therefore we believe our AIS upper bounds to be fairly tight in practice.

In theory, for large number of intermediate distributions, the AIS variance is proportional to 1/M K (Neal, 2001; 2005) , where M is the number of AIS chains and K is the number of intermediate distributions.

For the main experiments of our paper, we evaluate the tightness of the AIS estimate by computing the BDMC gap, and show that in practice our upper bounds are tight (Appendix D).

The Rate Distortion Tradeoff in the AIS Chain.

Different values of β corresponds to different tradeoffs between the compression rate and the distortion in a given generative model.

β = 0 corresponds to the case where q 0 (z|x) = p(z).

In this case, the compression rate is zero, and the distortion would be large, since in order to reconstruct x, we simply sample from the prior and generate a randomx that is completely independent of x. In this case, the distortion would

In the case of probabilistic decoders with NLL distortion, another interesting intermediate distribution is β = 1, where the optimal channel conditional is the true posterior of the generative model q (z|x) = p(z|x).

In this case, as shown in Prop.

2c, the summation of the rate-prior and the distortion term is the negative of true log-likelihood of the generative model.

As β n → ∞, the network cares more about the distortion and less about the compression rate.

In this case, the optimal channel conditional would be q n (z|x) = δ(z − z ML (x)), where

.

In other words, since the network only cares about the distortion, the optimal channel condition puts all its mass on z ML which minimizes the distortion.

However, the network would require infinitely many bits to precisely represent this delta function, and thus the rate goes to infinity.

Evaluation of Implicit Generative Models .

Quantitative evaluation of the performance of GANs has been a challenge for the field since the beginning.

Many heuristic measures have been proposed, such as the Inception score (Salimans et al., 2016) and the Fréchet Inception Distance (FID) (Heusel et al., 2017) .

One of the main drawbacks of the IS or FID is that a model that simply memorizes the training dataset would obtain a near-optimal score.

Another, drawback of these methods is that they use a pretrained ImageNet classifier weights which makes them sensitive to the weights (Barratt & Sharma, 2018) of the classifier, and less applicable to other domains and datasets.

Another evaluation method that sometimes is being used is the Parzen window estimate, which can be shown to be an instance of AIS with zero intermediate distributions, and thus has a very large variance.

Another evaluation method of GANs that was proposed in Metz et al. (2018) is measuiring the ability of the generator network to reconstruct the samples from the data distribution.

This metric is similar to the distortion obtained at the high-rate regime of our rate distortion framework when β → ∞. Another related work is GILBO , which similar to our framework does not require the generative model to have a tractable posterior and thus allows direct comparison of VAEs and GANs.

However, GILBO can only evaluate the performance of the generative model on the simulated data and not the true data distribution.

Rate Distortion Theory and Generative Models.

Perhaps the closest work to ours is "Fixing a Broken ELBO" , which plots rate-prior distortion curves for VAEs.

Our work is different than in two key aspects.

First, in the rate-prior distortion function is evaluated by fixing the architecture of the neural network, and learning the distortion measure d(x, f (z)) in addition to learning q(z|x).

Whereas, in our definition of rate distortion, we assumed the distortion measure is fixed and given by a trained generative model.

As the result, we plot the rate-prior distortion curve for a particular generative model, rather than a particular architecture.

The second key difference is that, consistent with the Shannon's rate distortion Deep and Shallow GANs

(a) MNIST Comparing GANs on CIFAR10

Figure 3: The rate distortion curves of GANs.

theorem, we find the optimal channel conditional q * (z|x) by using AIS; while in , q(z|x) is a variational distribution that is restricted to a variational family.

See Appendix A for a discussion of related works about practical compression schemes, distortionperception tradeoffs, and precision-recall tradeoffs.

In this section, we use our rate distortion approximations to answer the following questions: How do different generative models such as VAEs, GANs and AAEs perform at different lossy compression rates?

What insights can we obtain from the rate distortion curves about different characteristics of generative models?

What is the effect of the code size (width), depth of the network, or the learning algorithm on the rate distortion tradeoffs?

Rate Distortion Curves of GANs.

Fig. 3 shows rate distortion curves for GANs trained on MNIST and CIFAR-10.

We varied the dimension of the noise vector z, as well as the depth of the decoder.

For the GAN experiments on MNIST (Fig. 3a) , the label "deep" corresponds to three hidden layers of size 1024, and the label "shallow" corresponds to one hidden layer of size 1024.

We trained shallow and deep GANs with Gradient Penalty (GAN-GP) (Gulrajani et al., 2017) with the code size d ∈ {2, 5, 10, 100} on MNIST.

For the GAN experiments on CIFAR-10 ( Fig. 3b) , we trained the DCGAN (Radford et al., 2015) , GAN with Gradient Penalty (GP) (Gulrajani et al., 2017) , SN-GAN (Miyato et al., 2018) , and BRE-GAN (Cao et al., 2018) , with the code size of d ∈ {2, 10, 100}. In both the MNIST and CIFAR experiments, we observe that in general increasing the code size has the effect of extending the curve leftwards.

This is expected, since the high-rate regime is effectively measuring reconstruction ability, and additional dimensions in z improves the reconstruction.

We can also observe from Fig. 3a that increasing the depth pushes the curves down and to the left.

In other words, the distortion in both high-rate and mid-rate regimes improves.

In these regimes, increasing the depth increases the capacity of the network, which enables the network to make a better use of the information in the code space.

In the low-rate regime, however, increasing the depth, similar to increasing the latent size, does not improve the distortion.

Comparing GANs, VAEs and AAEs

Figure 4: RD curves of VAEs, GANs, AAEs.

Rate Distortion Curves of VAEs.

Fig. 4 compares VAEs, AAEs and GP-GANs (Gulrajani et al., 2017) with the code size of d ∈ {2, 10, 100}, and the same decoder architecture on the MNIST dataset.

In general, we can see that in the mid-rate to high-rate regimes, VAEs achieve better distortions than GANs with the same architecture.

This is expected as the VAE is trained with the ELBO objective, which encourages good reconstructions (in the case of factorized Gaussian decoder).

We can see from Fig. 4 increasing the capacity reduces the distortion at the high-rate regime, at the expense of increasing the distortion in the low-rate regime (or equivalently, increasing the rate required to adequately approximate the data).

We believe the performance drop of VAEs in the low-rate regime is symptomatic of the "holes problem" (Rezende & Viola, 2018; Makhzani et al., 2015) in the code space of VAEs with large code size: because these VAEs allocate a large fraction of their latent spaces to garbage images, it requires many bits to get close to the image manifold.

Interestingly, this trade-off could also help explain the well-known problem of blurry samples from VAEs: in order to avoid garbage samples (corresponding to large distortion in the low-rate regime), one needs to reduce the capacity, thereby increasing the distortion at the high-rate regime.

By contrast, GANs do not suffer from this tradeoff, and one can train high-capacity GANs without sacrificing performance in the low-rate regime.

Rate Distortion Curves of AAEs.

The AAE was introduced by Makhzani et al. (2015) to address the holes problem of VAEs, by directly matching the aggregated posterior to the prior in addition to optimizing the reconstruction cost.

Fig. 4 shows the RD curves of AAEs.

In comparison to GANs, AAEs can match the low-rate performane of GANs, but achieve a better high-rate performance.

This is expected as AAEs directly optimize the reconstruction cost as part of their objective.

In comparison to VAEs, AAEs perform slightly worse at the high-rate regime, which is expected as the adversarial regularization of AAEs is stronger than the KL regularization of VAEs.

But AAEs perform slightly better in the low-rate regime, as they can alleviate the holes problem to some extent.

Since log-likelihoods constitute only a scalar value, they are unable to distinguish different aspects of a generative model which could be good or bad, such as the prior or the observation model.

Here, we show that two manipulations which damage a trained VAE in different ways result in very different behavior of the RD curves.

Our first manipulation, originally proposed by Theis et al. (2015) , is to use a mixture of the VAE's density and another distribution concentrated away from the data distribution.

As pointed out by Theis et al. (2015) , this results in a model which achieves high log-likelihood while generating poor samples.

Specifically, after training the VAE10 on MNIST, we "damage" its prior p(z) = N (0, I) by altering it to a mixture prior (1 − α)p(z) + αq(z), where q(z) = N (0, 10I) is a "poor" prior, which is chosen to be far away from the original prior p(z); and α is close to 1.

This process would results in a "poor" generative model that generates garbage samples most of the time (more precisely with the probability of α).

Suppose p(x) and q(x) are the likelihood of the good and the poor generative models.

It is straightforward to see that log q(x) is at most 4.6 nats worse that log p(x), and thus log-likelihood fails to tell these models apart: Fig. 5a plots the rate distortion curves of this model for different values of α.

We can see that the high-rate and log-likelihood performance of the good and poor generative models are almost identical, whereas in the low-rate regime, the RD curves show significant drop in the performance and successfully detect this failure mode of log-likelihood.

(b) VAE, GANs and AAEs Figure 6 : The RD curves of GANs, VAEs and AAEs with MSE distortion on the deep feature space.

The behavior is qualitatively similar to the results for MSE in images (see Fig. 3 and Fig. 4) , suggesting that the RD analysis is not particularly sensitive to the particular choice of metric.

Our second manipulation is to damage the decoder by adding a Gaussian blur kernel after the output layer.

Fig. 5b shows the rate distortion curves for different standard deviations of the Gaussian kernel.

We can see that, in contrast to the mixture prior experiment, the high-rate performance of the VAE drops due to inability of the decoder to output sharp images.

However, we can also see an improvement in the low-rate performance of the VAE.

This is because (similarly to log-likelihoods with Gaussian observation models) the data distribution does not necessarily achieve the minimal distortion, and in fact, in the extremely low-rate regime, blurring appears to help by reducing the average Euclidean distance between low-rate reconstructions and the input images.

Hence, our two manipulations result in very different effects to the RD curves, suggesting that these curves provide a much richer picture of the performance of generative models, compared to scalar log-likelihoods.

The experiments discussed above all used pixelwise MSE as the distortion metric.

However, for natural images, one could use more perceptually valid distortion metrics such as SSIM ( ).

Fig. 6 shows the RD curves of GANs, VAEs, and AAEs on the MNIST dataset, using the MSE on the deep features of a CNN as distortion metric.

In all cases, the qualitative behavior of the RD curves with this distortion metric closely matches the qualitative behaviors for pixelwise MSE.

We can see from Fig. 6a that similar to the RD curves with MSE distortion, GANs with different depths and code sizes have the same low-rate performance, but as the model gets deeper and wider, the RD curves are pushed down and extended to the left.

Similarly, we can see from Fig. 6b that compared to GANs and AAEs, VAEs generally have a better high-rate performance, but worse lowrate performance.

The fact that the qualitative behaviors of RD curves with this metric closely match those of pixelwise MSE indicates that the results of our analysis are not overly sensitive to the particular choice of distortion metric.

In this work, we studied rate distortion approximations for evaluating different generative models such as VAEs, GANs and AAEs.

We showed that rate distortion curves provide more insights about the model than the log-likelihood alone while requiring roughly the same computational cost.

For instance, we observed that while VAEs with larger code size can generally achieve better lossless compression rates, their performances drop at lossy compression in the low-rate regime.

Conversely, expanding the capacity of GANs appears to bring substantial reductions in distortion at the high-rate regime without any corresponding deterioration in quality in the low-rate regime.

This may help explain the success of large GAN architectures (Brock et al., 2019; Karras et al., 2018a; b) .

We also discovered that increasing the capacity of GANs by increasing the code size (width) has a very different effect than increasing the depth.

The former extends the rate distortion curves leftwards, while the latter pushes the curves down.

We also found that different GAN variants with the same code size has almost similar rate distortion curves, and that the code size dominates the algorithmic differences of GANs.

Overall, lossy compression yields a richer and more complete picture of the distribution modeling performance of generative models.

The ability to quantitatively measure performance tradeoffs should lead to algorithmic insights which can improve these models.

Practical Compression Schemes.

We have justified our use of compression terminology in terms of Shannon's fundamental result implying that there exist a rate distortion code for any rate distortion pair that is achievable according to the rate distortion function.

Interestingly, for lossless compression with generative models, there is a practical compression scheme which nearly achieves the theoretical rate (i.e. the negative ELBO): bits-back encoding.

The basic scheme was proposed by Wallace (1990); Hinton & Van Camp (1993) , and later implemented by Frey & Hinton (1996) .

Practical versions for modern deep generative models were developed by Townsend et al. (2019); Kingma et al. (2019) .

We do not currently know of an analogous practical scheme for lossy compression with deep generative models.

Other researchers have developed practical coding schemes achieving variational rate distortion bounds for particular latent variable models which exploited the factorial structure of the variational posterior (Ballé et al., 2018; Theis et al., 2017) .

These methods are not directly applicable in our setting, since we don't assume an explicit encoder network, and our variational posteriors lack a convenient factorized form.

We don't know whether our variational approximation will lead to a practical lossy compression scheme, but the successes for other variational methods give us hope.

Relationship with the Rate-Distortion-Perception Tradeoff.

Our work is related to Blau & Michaeli (2019) which incorporates a perceptual quality loss function in the rate-distortion framework and characterizes the triple tradeoff between rate distortion and perception.

More specifically, Blau & Michaeli (2019) defines the perceptual loss using a divergence between the marginal reconstruction distribution and the data distribution.

This perceptual loss is then incorporated as an additional constraint in the rate-distortion framework to encourage the reconstruction distribution to perceptually look like the data distribution.

It is shown that as the perceptual constraint becomes tighter, the rate-distortion function elevates more.

In our rate-prior distortion framework, we are also enforcing a perceptual constraint on the reconstruction distribution by incorporating the regularization term of KL(q(z) p(z)) in the rate-distortion objective, which encourages matching the aggregated posterior to the prior (Makhzani et al., 2015) .

More precisely, let us define the reconstruction distribution r(x) as the the distribution obtained by passing the data distribution through the encoder and then the decoder:

It can be shown that the regularization term KL(q(z) p(z)) upper bounds KL(r(x) p(x)):

(20) In other words, in the rate-prior distortion optimization, for a given distortion constraint, we are not only interested in minimizing the rate I(x; z), but also at the same time, we are interested in preserving the perceptual quality of the reconstruction distribution by matching it to the model distribution.

In the low-rate regime, when the model is allowed to have large distortions, the model obtains small rates and at the same time preserves the perceptual distribution of the reconstruction samples.

As the distortion constraint becomes tighter, the model starts to trade off the rate I(x; z) and the perceptual quality KL(q(z) p(z)), which results in an elevated rate distortion curve.

Relationship with the Precision-Recall Tradeoff.

One of the main drawbacks of the IS or FID is that they can only provide a single scalar value that cannot distinguish the mode dropping behavior from the mode inventing behavior (generating outlier or garbage samples) in generative models.

In order to address this issue, Lucic et al. (2018); Sajjadi et al. (2018) propose to study the precision-recall tradoff for evaluating generative models.

In this context, high precision implies that the samples from the model distribution are close to the data distribution, and high recall implies the generative model can reconstruct (or generate a sample similar to) any sample from the data distribution.

The precision-recall curves enable us to identify both the mode dropping and the mode inventing behavior of the generative model.

More specifically, mode dropping drops the precision of the model at the high-recall regime, and mode inventing drops the precision of the model in the low-recall regime.

Our rate-prior distortion framework can be thought as the information theoretic analogue of the precision-recall curves, which extends the scalar notion of log-likelihood to rate distortion curves.

More specifically, in our framework, mode dropping drops the distortion performance of the model at the high-rate regime, and mode inventing drops the distortion performance of the model at the low-rate regime.

In Section 6, we will empirically study the effect of mode dropping and mode inventing on our rate-prior distortion curves.

AIS Validation.

We conducted several experiments to validate the correctness of our implementation and the accuracy of the AIS estimates.

Firstly, we compared our AIS results with the analytical solution of rate-prior distortion curve on a linear VAE (derived in Appendix D.3.1) trained on MNIST.

As shown in Fig. 7 , the RD curve estimated by AIS agrees closely with the analytical solution.

Secondly, for the main experiments of the paper, we evaluated the tightness of the AIS estimate by computing the BDMC gap.

The largest BDMC gap for VAEs and AAEs was 0.127 nats, and the largest BDMC gap for GANs was 1.649 nats, showing that our AIS upper bounds are tight.

More details are provided in Appendix D.3.2.

APPENDIX C PROOFS C.1 PROOF OF PROP.

1.

] is a linear function of the channel conditional distribution q(z|x).

The mutual information is a convex function of q(z|x).

The KL(q(z) p(z)) is also convex function of q(z), which itself is a linear function of q(z|x).

Thus the rate-prior objective is a convex function of q(z|x).

Suppose for the distortions D 1 and D 2 , q 1 (z|x) and q 2 (z|x) achieve the optimal rates in Eq. 5 respectively.

Suppose the conditional q λ (z|x) is defined as q λ (z|x) = λq 1 (z|x) + (1 − λ)q 2 (z|x).

The rate-prior objective that the conditional q λ (z|x) achieves is I λ (z; x) + KL(q λ (z) p(z)), and the distortion D λ that this conditional achieves is

which proves the convexity of R p (D).

Alternative Proof of Prop.

1a.

We know the rate-prior term E p d (x) KL(q(z|x) p(z)) is a convex function of q(z|x), and E q(x,z) [d(x, f (z))] is a linear and thus convex function of q(z|x).

As the result, the following optimization problem is a convex optimization problem.

min

The rate distortion function R p (D) is the perturbation function of the convex optimization problem of Eq. 22.

The convexity of R p (D) follows from the fact that the perturbation function of any convex optimization problem is a convex function (Boyd & Vandenberghe, 2004) .

Proof of Prop.

1b.

We have min

= min

= min

where in Eq. 24, we have used the fact that for any function f (x, y), we have min

and in Eq. 25, we have used the fact that KL(q(z) p(z)) is minimized when p(z) = q(z).

Proof of Prop.

1c.

In Prop.

1a, we showed that the rate-prior term is a convex function of q(z|x), and that the distortion is a linear function of q(z|x).

So the summation of them in Eq. 9 will be a convex function of q(z|x).

The unique global optimum of this convex optimization can be found by rewriting Eq. 9 as

where Z β (x) = p(z) exp(−βd(x, f (z)))dz.

The minimum of Eq. 28 is obtained when the KL divergence is zero.

Thus the optimal channel conditional is q *

Proof of Prop.

2a.

R(D) ≤ R p (D) was proved in Prop.

1b.

To prove the first inequality, note that the summation of rate and distortion is

where q * (x, z) is the optimal joint channel conditional, and q * (z) and q * (x|z) are its marginal and conditional.

The equality happens if there is a joint distribution q(x, z), whose conditional q(x|z) = p(x|z), and whose marginal over x is p d (x).

But note that such a joint distribution might not exist for an arbitrary p(x|z).

Proof of Prop.

2b.

The proof can be easily obtained by using d(x, f (z)) = − log p(x|z) in Prop.

1c.

Proof of Prop.

2c.

Based on Prop.

2b, at β = 1, we have AIS has the property that for any step k of the algorithm, the set of chains up to step k, and the partial computation of their weights, can be viewed as the result of a complete run of AIS with target distribution q * k (z|x).

Hence, we assume without loss of generality that we are looking at a complete run of AIS (but our analysis applies to the intermediate distributions as well).

2.

Compute the importance weights and normalized importance weights of each chain using w

3.

Select a chain index S with probability ofw

5. Keep the unselected chain values and re-label them as (z

where −S denotes the set of all indices except the selected index S. 6.

Return z = z 1 k .

More formally, the AIS distribution is

Using the AIS distribution q AIS k (z|x) defined as above, we define the AIS distortion D AIS k (x) and the AIS rate-prior R

In order to estimate R

We would like to prove that

The proof of Eq. 40 is straightforward:

Eq. 42 shows thatD

We also know logẐ AIS k (x) obtained by Eq. 38 is the estimate of the log partition function, and by the Jenson's inequality lower bounds in expectation the true log partition function: Domke & Sheldon (2018) )

In order to simplify notation, suppose z

Using the above notation, Eq. 44 can be re-written aŝ

Hence,

where the inequality follows from the monotonicity of KL divergence.

Rearranging terms, we bound the rate:

Eq. 49 shows thatR AIS k (x) upper bounds the AIS rate-prior R AIS k (x) in expectation.

We also showed D AIS k (x) is an unbiased estimate of the AIS distortion D AIS k (x).

Hence, the estimated AIS rate-prior curve upper bounds the AIS rate-prior distortion curve in expectation:

The code for reproducing all the experiments of this paper will be open sourced publicly.

We used MNIST (LeCun et al., 1998) and CIFAR-10 (Krizhevsky & Hinton, 2009 ) datasets in our experiments.

Real-Valued MNIST.

For the VAE experiments on the real-valued MNIST dataset (Fig. 4a) , we used the "VAE-50" architecture described in (Wu et al., 2016) , and only changed the code size in our experiments.

The decoder variance is a global parameter learned during the training.

The network was trained for 1000 epochs with the learning rate of 0.0001 using the Adam optimizer (Kingma & Ba, 2014) .

For the GAN experiments on MNIST (Fig. 3a) , we used the "GAN-50" architecture described in (Wu et al., 2016) .

In order to stabilize the training dynamic, we used the gradient penalty (GP) (Salimans et al., 2016) .

In our deep architectures, we used code sizes of d ∈ {2, 5, 10, 100} and three hidden Figure 8: The rate-prior distortion curves obtained by adaptively tuning the HMC parameters in the preliminary run, and pre-loading the HMC parameters in the second formal run.

"rs" in the legend indicates the random seed used in the second run.

layers each having 1024 hidden units to obtain the following GAN models: Deep-GAN2, Deep-GAN5, Deep-GAN10 and Deep-GAN100.

The shallow GANs architectures are similar to the deep architectures but with one layer of hidden units.

For the CIFAR-10 experiments (Fig. 3b) , we experimented with different GAN models such as DCGAN (Radford et al., 2015) , DCGAN with Gradient Penalty (GP-GAN) (Gulrajani et al., 2017) , Spectral Normalization (SN-GAN) (Miyato et al., 2018) , and DCGAN with Binarized Representation Entropy Regularization (BRE-GAN) (Cao et al., 2018) .

The numbers at the end of each GAN name in Fig. 3b indicate the code size.

For each RD curve, there are 1999 points computed with only one AIS chain, for 999 β s spaced linearly from β max to 1 and another 999 β s spaced linearly from 1 to β min , and plus β = 1, thus 1999 points.

β min = 1 12 for all models.

β max = 1 0.0003 ≈ 3333 for 100 dimensional models such as GAN100, VAE100 or AAE 100, and β max = For the 2, 5 and 10 dimensional models, N = 40000, and the above procedure will result in 60292 intemediate distributions in total.

For 100 dimensional models, to ensure accuracy of our AIS estimator with a small BDMC gap, we used N = 1600000 and the above procedure will result in 1611463 intermediate distributions in total.

We used 20 leap frog steps for HMC, 40 independent chains, on a single batch of 50 images.

On MNIST, we also tested with a larger batch of 500 MNIST images, but did not observe significant difference compared with a batch 50 images, thus we did all of our experiments with a single batch 50 images.

On a P100 GPU, for Adaptive Tuning of HMC Parameters.

While running the AIS chain, the parameters of the HMC kernel cannot be adaptively tuned, since it would violate the Markovian property of the chain.

So in order to be able to adaptively tune HMC parameters such as the number of leapfrog steps and the step size, in all our experiments, we first do a preliminary run where the HMC parameters are adaptively tuned to yield an average acceptance probability of 65% as suggested in Neal (2001) .

Then in the second "formal" run, we pre-load and fix the HMC parameters found in the preliminary run, and start the chain with a new random seed to obtain our final results.

Interestingly, we observed that the difference in the RD curves obtained from the preliminary run and the formal runs with various different random seeds is very small, as shown in Fig. 8 .

This figure shows that the AIS with the HMC kernel is robust against different choices of random seeds for approximating the RD curve of VAE10.

We conducted several experiments to validate the correctness of our implementation and the accuracy of the AIS estimates.

We compared our AIS results with the analytical solution of the rate-prior distortion optimization on a linear VAE trained on MNIST as shown in Fig. 7 .

In order to derive the analytical solution, we first find the optimal distribution q * β (z|x) from Prop.

2b.

For simplicity, we assume a fixed identity covariance matrix I at the output of the conditional likelihood of the linear VAE decoder.

In other words, the decoder of the VAE is simply: x = Wz + b + , where x is the observation, z is the latent code vector, W is the decoder weight matrix and b is the bias.

The observation noise of the decoder is ∼ N (0, I).

It's easy to show that the conditional likelihood raised to a power β is: p(x|z) β = N (x|Wz + b,

For numerical stability, we can further simplify thw above by taking the SVD of W : let W = UDV , and then apply the Woodbury Matrix Identity to the matrix inversion, we can get:

Where R β is a diagonal matrix with the i th diagonal entry being

Where k is the dimension of the latent code z. With negative log-likelihood as the distortion metric, the analytically form of distortion term is: We evaluated the tightness of the AIS estimate by computing the BDMC gaps using the same AIS settings.

Fig. 9 , shows the BDMC gaps at diffrent compression rates for the VAE, GAN and AAE experiments on the MNIST dataset.

The largest BDMC gap for VAEs and AAEs is 0.127 nats, and the largest BDMC gap for GANs is 1.649 nats, showing that our AIS upper bounds are tight.

In this section, we visualize the high-rate (β ≈ 3500) and low-rate (β = 0) reconstructions of the MNIST images for VAEs, GANs and AAEs with different hidden code sizes.

The qualitative results are shown in Fig. 10 and Fig. 11 , which is in consistent with the quantitative results presented in experiment section of the paper.

(h) High Rate VAE100 (i) High Rate AAE100 (j) High Rate GAN100 Figure 11 : High-rate reconstructions (β max ) of VAEs, GANs and AAEs on MNIST.

β max = 3333 for 100 dimensional models, and β max = 36098 for the 2 and 10 dimensional models.

<|TLDR|>

@highlight

We study rate distortion approximations for evaluating deep generative models, and show that rate distortion curves provide more insights about the model than the log-likelihood alone while requiring roughly the same computational cost.
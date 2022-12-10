Score matching provides an effective approach to learning flexible unnormalized models, but its scalability is limited by the need to evaluate a second-order derivative.

In this paper,we connect a general family of learning objectives including score matching to Wassersteingradient flows.

This connection enables us to design a scalable approximation to theseobjectives, with a form similar to single-step contrastive divergence.

We present applications in training implicit variational and Wasserstein auto-encoders with manifold-valued priors.

Unnormalized models define the model distribution as q(x; θ) ∝ exp(−E(x; θ)), where E(x; θ) is an energy function that can be parameterized by e.g. DNNs.

Unnormalized models can be used directly for density estimation, but another important application is in gradient estimation for implicit variational inference, where we can use score estimation in latent space to approximate an intractable learning objective.

This approach leads to improved performance in training implicit auto-encoders (Song et al., 2019) .

Maximum likelihood estimation for unnormalized models is intractable, and score matching (Hyvärinen, 2005 ) is a popular alternative.

Score matching optimizes the Fisher divergence

where we denote the data distribution as p. Hyvärinen (2005) shows D F is equivalent to E p(x) ∆ log q(x; θ) + 1 2 ∇ log q(x; θ) 2 , where ∆ = i ∂ 2 i is the Laplacian; the equivalent form can be estimated using samples from p.

So far, when E has a complex parameterization, calculating the equivalent objective is still difficult, as it involves the second-order derivatives; and in practice, people turn to scalable approximations of the score matching objective (Song et al., 2019; Hyvarinen, 2007; Vincent, 2011) or other objectives such as the kernelized Stein discrepancy (KSD; Liu et al., 2016b; Liu and Wang, 2017) .

However, these approximations are developed on a case-by-case basis, leaving important applications unaddressed; for example, there is a lack of scalable learning methods for models on manifolds (Mardia et al., 2016) .

In this work, we present a unifying perspective to this problem, and derive scalable approximations for a variety of objectives including score matching.

We start by interpreting these objectives as the initial velocity of certain distribution-space gradient flows, which are simulated by common samplers.

This novel interpretation leads to a scalable approximation algorithm for all such objectives, reminiscent to single-step contrastive divergence (CD-1).

We refer to any objective bearing the above interpretation as above as a "minimum velocity learning objective", a term coined in the unpublished work Movellan (2007) .

Our formulation is a distribution-space generalization of their work, and applies to different objectives as the choice of distribution space varies.

Another gap we fill in is the development of a practically applicable algorithm: while the idea of approximating score matching with CD-1 is also explored in (Hyvarinen, 2007; Movellan, 2007) , previously the approximation suffers from an infinite variance problem, and is thus believed to be impractical (Hyvarinen, 2007; Saremi et al., 2018) ; we present a simple fix to this issue.

Additionally, we present an approximation to the objective function instead of its gradient, thus enabling the use of regularization like early-stopping.

Other related work will be reviewed in Appendix C.

One important application of our framework is in learning unnormalized models on manifolds.

This is needed in areas such as image analysis (Srivastava et al., 2007) , geology (Davis and Sampson, 1986) and bioinformatics (Boomsma et al., 2008) .

Moreover, as we present an approximation to the Riemannian score matching objective, it enables flexible inference for VAEs and WAEs with manifold-valued latent variables, as it enables gradient estimation for implicit variational distributions on manifolds.

It is believed that auto-encoders with a manifold-valued latent space can capture the distribution of certain types of data better (Mathieu et al., 2019; Anonymous, 2020; Davidson et al., 2018) .

As we will see in Section 3, our method leads to improved performance of VAEs and WAEs.

We now present our framework, which concerns all learning objectives of the following form:

where q θ is the model distribution, {p t } is the gradient flow of KL q in a suitable distribution space (e.g. the 2-Wasserstein space), and KL q is the exclusive KL divergence functional, p → KL(p q θ ).

We refer to these objectives as "minimum velocity learning (MVL) objectives", since (3) will show that they correspond to the initial velocity of the gradient flow.

(2) subsumes the Fisher divergence for score matching as a special case, since from the properties of the 2-Wasserstein space (please refer to Appendix A for the necessary preliminary knowledge), we have D F (p|q) = 1 2 grad p KL q 2 , where the gradient and norm are defined in the 2-Wasserstein space, and the data manifold X is endowed with the Euclidean metric.

Rearranging terms, we get

i.e., score matching is a special case of the MVL objective, when the space of distributions is chosen as the 2-Wasserstein space P(X ).

In certain cases, the gradient flow of KL q corresponds to common samplers, and can be efficiently simulated: e.g., the gradient flow in P(X ) is the (Riemannian) Langevin dynamics.

Now we utilize this connection to design a scalable approximation to these objectives.

First, note (3) holds regardless of the chosen space of distributions.

Denote

As the first term in (4) is independent of θ, the MVL objective is always equivalent to the second term.

We approximate it by simulating a modified gradient flow: letp t be the distribution obtained by running the sampler targeting q 1/2 .

Then

(5) can be approximated by replacing the limit with a fixed , and running the corresponding sampler starting from a mini-batch of training data.

The approximation becomes unbiased when → 0.

A Control Variate The approximation to (5) will have small bias.

However, it suffers from high variance when the sampler consists of Itô diffusion (e.g. when it is Langevin dynamics).

Fortunately, we can solve this problem with a control variate.

To illustrate this, suppose the MVL objective is defined using Langevin dynamics (LD).

Without loss of generality, we assume we use a batch size of 1 in the approximation, so approximation isL

, where x + is sampled from the training data, and Z ∼ N (0, I).

By Taylor expansion 1 ,

and as → 0, VarL = Θ −1 → ∞. Thus a control variate is needed.

In this LD example, the control variate is

More generally, the control variate is always the inner product of ∇ x E(x + ) and the diffusion term in the sampler.

As a side product of our work, we note that similar control variate can be obtained for CD-1 and denoising score matching, and it solves their problem.

See Appendix B.2.

As an application, let us consider learning unnormalized models on Riemannian manifolds.

In this case, the gradient flow of KL q in the 2-Wasserstein space becomes the Riemannian Langevin dynamics (Xifara et al., 2014) , and the approximate MVL objective becomes

is a sample from the Riemannian LD, and z ∼ N (0, G −1 (y)).

This is the Riemannian score matching objective (Mardia et al., 2016) , which also has the form of (1), but the norm is defined by the Riemannian metric of X .

From this example, we can see the power of our framework, which enables us to approximate new objectives with ease.

We now apply our approximation to learning implicit variational auto-encoders (VAEs) and Wasserstein auto-encoders (WAEs) with manifold-valued prior.

First we review the use of score estimator in learning implicit auto-encoding models.

We use VAE as an example; for WAE with KL penalty, the derivation is similar, see Song et al. (2019) .

The VAE objective is E p(x) E q(z|x;φ) log p(z)p(x|z;θ) q(z|x;φ) , where q(z|x; φ) is the push-forward measure of N (0, I) by f (·; φ).

The objective is intractable, as the entropy term H[q(z|x; φ)] is intractable; however, we can show that (Li and Turner, 2018)

Thus to approximate the objective, it suffices to approximate the score function ∇ z log q(z).

This can be implemented by learning an unnormalized model using score matching.

As the score matching objective directly aligns the learnt score function ∇ z E to the data score, it can be viewed as score estimation using conservative fields, thus it leads to a better approximation to the gradient ∇ φ H[q(z)] compared to indirect approximations such as the adversarial density ratio estimators.

As we turn to the case where the latent space is an embedded manifold, the original score matching objective can no long be used, since q(z) no longer has a density w.r.t.

the Lebesgue measure in the embedded space.

However, we can still do score estimation on the manifold, i.e. estimate the log derivative of the density w.r.t.

the manifold Hausdorff measure.

This can be done by fitting an unnormalized model on manifold, using the approximate objective developed in Section 2.2.

We note that in this case, (8) still holds, and we can still estimate the gradient of the ELBO using the score estimate.

See Appendix E.4 for details.

Empirical Evaluations We apply our method to train implicit hyperspherical VAEs (Davidson et al., 2018) with implicit encoders and WAEs, on the MNIST dataset.

Our experiment setup follows Song et al. (2019) , with the exception that we parameterize an energy network E(z; ψ) and uses its gradient as the score estimate, instead of parameterizing a score network.

Detailed setup and additional synthetic experiments are in Appendix D.

For the VAE experiment, we compare with VAEs with explicit variational posteriors, as well as Euclidean VAEs, and report negative log likelihood estimated with annealed importance sampling (Neal, 2001) ; for the WAE experiment, we compare with WAE-GAN (Arjovsky et al., 2017) , and report the FID score (Heusel et al., 2017) .

The results are summarized in Figure 1 .

We can see that in all cases, hyperspherical prior outperforms the Euclidean prior, and our method leads to improved performance.

Interestingly, for VAEs with explicit encoders, hyperspherical VAE could not match the performance of Euclidean VAE in high dimensions; this is consistent with the result in Davidson et al. (2018) , who incorrectly conjectured that hyperspherical prior is inadequate in high dimensions; we can see that the problem is actually the lack of flexibility in inference, which our method addresses.

In this section, we review background knowledge needed in this work, most importantly Wasserstein gradient flow and its connection to sampling algorithms.

A (differential) manifold M is a topological space locally diffeomorphic to an Euclidean or Hilbert space.

A manifold is covered by a set of charts, which enables the use of coordinates locally, and specifies a set of basis {∂ i } in the local tangent space.

A Riemannian manifold further possesses a Riemannian structure, which assigns to each tangent space T p M an inner product structure.

The Riemannian structure can be described using coordinates w.r.t.

local charts.

The manifold structure enables us to differentiate a function along curves.

Specifically, consider a curve c : [0, T ] → M, and a smooth function f : M → R. At c(t) ∈ M, a tangent vector

A tangent vector field assigns to each p ∈ M a tangent vector V p ∈ T p M.

It determines a flow, a set of curves {φ p (t) : p ∈ M} which all have V φp(t) as their velocity.

On Riemannian manifolds, the gradient of a smooth function f is a tangent vector field p → grad p f such that

It determines the gradient flow, which generalizes the Euclidean-space notion dx = ∇ x f (x)dt.

We will work with two types of manifolds: the data space X when we apply our method to manifold-valued data, and the space of probability distributions over X .

On the space of distributions, we are mostly interested in the 2-Wasserstein space P(X ), a Riemannian manifold.

The following properties of P(X ) will be useful for our purposes (Villani, 2008): 1.

Its tangent space T p P(X ) can be identified as a subspace of the space of vector fields on X ; the Riemannian metric of P(X ) is defined as

for all p ∈ P(X ), X, Y ∈ T p P(X ); the inner product on the right hand side above is determined by the Riemannian structure of X .

We will also consider a few other spaces of distributions, including the Wasserstein-Fisher-Rao space (Lu et al., 2019) , and the H-Wasserstein space introduced in (Liu, 2017) .

On the data space, we need to introduce the notion of density, i.e. the Radon-Nikodym derivative w.r.t.

a suitable base measure.

The Hausdorff measure is one such choice; it reduces to the Lebesgue measure when X = R n .

In most cases, distributions on manifolds are specified using their density w.r.t.

the Hausdorff measure; e.g. "uniform" distributions has constant densities in this sense.

Finally, the data space X will be embedded in R n ; we refer to real-valued functions on the space of distributions as functionals; we denote the functional q → KL(q p) as KL p ; we adopt the Einstein summation convention, and omit the summation symbol when an index appears both as subscript and superscript on one side of an equation, e.g.

Now we review the sampling algorithms considered in this work.

They include diffusion-based MCMC, particle-based variational inference, and other stochastic interacting particle systems.

Riemannian Langevin Dynamics Suppose our target distribution has density p(x) w.r.t.

the Hausdorff measure of X .

In a local chart U ⊂ X , let G : U → R m×m be the coordinate matrix of its Riemannian metric.

Then the Riemannian Langevin dynamics corresponds to the following stochastic differential equation in the chart 2 :

where

and (g ij ) is the coordinate of the matrix G −1 .

It is known (Villani, 2008) that the Riemannian Langevin dynamics is the gradient flow of the KL functional KL p (q) := KL(q p) in the 2-Wasserstein space P(X ).

Particle-based Samplers A range of samplers approximate the gradient flow of KL p in various spaces, using deterministic or stochastic interacting particle systems.

3 For instance, Stein variational gradient descent (SVGD; Liu and Wang, 2016) simulates the gradient flow in the so-called H-Wasserstein space (Liu, 2017), which replaces the Riemannian structure in P(X ) with the RKHS inner product.

Birth-death accelerated Langevin dynamics (Lu et al., 2019 ) is a stochastic interacting particle system that simulates to the gradient flow of KL p in the Wasserstein-Fisher-Rao space.

Finally, the stochastic particle-optimization sampler (SPOS; Zhang et al., 2018; Chen et al., 2018) combines the dynamics of SVGD and Langevin dynamics; as we will show in Appendix E.2, SPOS also has a gradient flow structure.

In this section we present additional discussions about our framework.

In Section B.1 we discuss other objectives that can be derived from our framework; in Section B.2 we show 2. (11) differs from the form in some literature (e.g. Ma et al., 2015) , as in our case, the density of the target measure is defined w.r.t.

the Hausdorff measure of X , instead of the Lebesgue measure in Ma et al. (2015) .

See (Xifara et al., 2014, eq (12) ) or (Hsu, 2008, Section 1.5).

3.

There are other particle-based samplers (Liu et al., 2019b,a; Taghvaei and Mehta, 2019) corresponding to accelerated gradient flows.

However, as we will be interested in the initial velocity of the flow, they do not lead to new objectives in our framework.

our control variate could be applied to CD-1 and denoising score matching; finally, while readers familiar with Riemannian Brownian motion may be concerned about the use of local coordinates in our Riemannian score matching approximation (6), we show in Section B.3 it does not lead to issues.

As our derivation is independent of the distribution space of choice, we can derive approximations to other learning objectives using samplers other than Langevin dynamics, as reviewed in Section A.2.

An example is Riemannian Langevin dynamics which we have discussed in the main text; another example is when we choose the sampler as SVGD.

In this case, we will obtain an approximation to the kernelized Stein discrepancy, generalizing the derivation in (Liu and Wang, 2017) .

When the sampling algorithm is chosen as SPOS, the corresponding MVL objective will be an interpolation between KSD and the Fisher divergence.

See Appendix E.3 for derivations.

Finally, the use of birth-death accelerated Langevin dynamics leads to a novel learning objective.

In terms of applications, our work focuses on learning neural energy-based models, and we find these objectives do not improve over score matching in this aspect.

However, these derivations generalize previous discussions, and establish new connections between sampling algorithms and learning objectives.

It is also possible that these objectives could be useful in other scenarios, such as learning kernel exponential family models (Sriperumbudur et al., 2017) , direct estimation of the score function (Li and Turner, 2018; Shi et al., 2018) , and improving the training of GANs (Liu and Wang, 2017) or amortized variational inference methods (Ruiz and Titsias, 2019).

As a side product of our work, we show in this section that our variance analysis explains the pitfall of two well-known approximations to the score matching objective: CD-1 (Hyvarinen, 2007) and denoising score matching (Vincent, 2011) .

Both approximations become unbiased as a step-size hyper-parameter → 0, but could not match the performance of exact score matching in practice, as witnessed in Hyvarinen (2007); Saremi et al. (2018); Song et al. (2019) .

Our analysis leads to novel control variates for these approximators.

As we will show in Section D.1, the variance-reduced versions of the approximations have comparable performance to the exact score matching objective.

The first two terms inside the norm represent a noise corrupted sample, and ψ θ represents a "single-step denoising direction" (Raphan and Simoncelli, 2011).

It is proved that the optimal ψ satisfies ψ = σ 2 ∇ logp, wherep is the density of the corrupted distribution (Raphan and Simoncelli, 2011; Vincent, 2011) .

Consider the stochastic estimator of (13).

We assume a batch size of 1, and denote the data sample as x. To keep notations consistent, denote = σ 2 , ψ θ (x) = ∇ x E(x; θ).

Then the stochastic estimator iŝ

Denotex := x + √ z. By Taylor expansion we havê

As

which is known as the Hutchinson's trick (Hutchinson, 1990) ,

But V ar(B) = O( 2 ), so as → 0, the rescaled estimator −2L dsm becomes unbiased with infinite variance; and subtracting (B) from (A) results in a finite-variance estimator.

Proposed as an approximation to the maximum likelihood estimate, the K-step contrastive divergence (CD-K) learning rule updates the model parameter with

where ν is the learning rate, and p K is obtained from p by running K steps of MCMC. (18) does not define a valid objective, since p K also depends on θ; however, Hyvarinen (2007) proved that when K = 1 and the sampler is the Langevin dynamics, (18) recovers the gradient of the score matching objective.

Using the same derivation as in the main text, we can see that as the step-size of the sampler approaches 0 (and ν is re-scaled appropriately), the gradient produced by CD-1 also suffers from infinite variance, and this can be fixed using the same control variate.

However, practical utility of CD-1 is still hindered by the fact that it does not correspond to a valid learning objective; consequently, it is impossible to monitor the training process for CD-1, or introduce regularizations such as early stopping 4 .

Readers familiar with Riemannian Brownian motion will notice that we used local coordinates when deriving the MPF objective, and this is only valid until the particle exits the local chart.

In this section, we show that this does not affect the validity of our method; specifically, we prove in Proposition 3 that the local coordinate representation lead to valid approximation to the MVL objective in the compact case.

We also argue in Remark 4 that the use of local coordinate does not lead to numerical instability.

4.

In practice, the term EpE − Ep K E is often used to tract the training process of CD-K. It is not a proper loss; we can see from (4) that when K = 1 and → 0, EpE − Ep K E is significantly different from the proper score matching (MVL) loss, by a term of

Remark 1 While a result more general than Proposition 3 is likely attainable (e.g. by replacing compactness of X with quadratic growth of the energy), this is out of the scope of our work; for our purpose, it is sufficient to note that the proposition covers manifolds like S n , and the local coordinate issue will not exist in manifolds possessing a global chart, such as H n .

Lemma 2 (Theorem 3.6.1 in (Hsu, 2002) ) For any manifold M, x ∈ M, and a normal neighborhood B of x, there exists constant C > 0 such that the first exit time τ from B, of the Riemannian Brownian motion starting from x, satisfies

for any L ≥ 1.

Proposition 3 Assume the data manifold X is compact, and for all θ, E(·; θ) is in C 1 .

Let L mvl_rld be defined as in (6), X t following the true Riemannian Langevin dynamics targeting

i.e. (6) recovers true WMVL objective.

Proof By the tower property of conditional expectation, it suffices to prove the result when P (X 0 = x) = 1 for some x. Choose a normal neighborhood B centered at x such that B is contained by our current chart, and has distance from the boundary of the chart bounded by some δ > 0.

Let C,τ be defined as in Lemma 2.

Recall the Riemannian LD is the sum of a drift and the Riemannian BM.

Since X is compact and E is in C 1 , the drift term in the SDE will have norm bounded by some finite C. Thus the first exit time of the Riemannian LD is greater than min(τ , δ/C) =: τ .

Let X t follow the true Riemannian LD,X t = X t when t < τ , and be such that E(X t ) = 0 afterwards.

5 By Hsu (2008), until τ ,X t follows the local coordinate representation of Riemannian LD (11), thus on the event { ≤ τ },X would correspond to y − in (7).

As X is compact, the continuous energy function E is bounded by |E(·)| ≤ A for some finite A. Then for sufficiently small ,

In the above the first term converges to d dt E(E(X t )) t=0 as → 0, and

Hence the proof is complete.

5.

This is conceptually similar to the standard augmentation used in stochastic process texts; from a algorithmic perspective it can be implemented by modifying the algorithm so that in the very unlikely event when y − escapes the chart, we return 0 as the corresponding energy.

We note that this is unnecessary for manifolds like S n , since the charts can be extended to R d and hence τ = ∞.

Remark 4 It is argued that simulating diffusion-based MCMC in local coordinates leads to numeric instabilities (Byrne and Girolami, 2013; Liu et al., 2016a) .

We stress that in our setting of approximating MVL objectives, this is not the case.

The reason is that we only need to do a single step of MCMC, with arbitrarily small step-size.

Therefore, we could use different step-size for each sample, based on the magnitude of g and log q in their locations.

We can also choose different local charts for each sample, which is justified by the proposition above.

Our work concerns scalable learning algorithms for unnormalized models.

This is a longstanding problem in literature, and some of the previous work is discussed in Section 1.

Apart from those work, it is worth mentioning Liu and Wang (2017), which also designed a CD-like algorithm to approximate the kernelized Stein discrepancy; as we have discussed in Section B.1, in our framework there exists a similar algorithm, as well as a slight generalization when we replace SVGD with SPOS.

Other notable work includes noise contrastive estimation (Gutmann and Hyvärinen, 2010) and Parzen score matching (Raphan and Simoncelli, 2011).

However, to our knowledge, they have not been applied to complex unnormalized models such as those parameterized by DNNs, and a comparison would fall out of the scope of this work.

Apart from the MVL formulation used in this work, there also exists other work on the connection between learning objectives of unnormalized model and infinitesimal actions of sampling dynamics.

The minimum probability flow framework (Sohl-Dickstein et al., 2011) studies the slightly different objective lim →0 1 KL(p 0 p ), where {p t } is the trajectory of the sampler.

It also recovers score matching as a special instance; however, it does not lead to scalable learning algorithms for continuous-state unnormalized models as our method does; instead, its main application is in discrete-state models.

Many of the MVL objectives we have derived are also instances of the Stein discrepancy (Gorham et al., 2019; Barp et al., 2019) .

This interpretation is helpful in establishing theoretical properties, but it does not lead to scalable implementations of these objectives, that do not depend on higher-order derivatives.

To demonstrate the proposed estimators have small bias and variance, we first evaluate them on low-dimensional synthetic data.

We also verify the claim in Section B.2 that our control variate improves the performance of CD-1 and DSM.

In this section, we evaluate our MVL approximation to the Euclidean score matching objective, as well as the variance-reduced DSM objective 6 .

6.

An experiment confirming the effectiveness of our control variate on CD-1 is presented in Appendix D.1.3.

We evaluate the bias and variance of our estimators by comparing them to sliced score matching (SSM), an unbiased estimator for the score matching objective.

We choose the data distribution p as the 2-D banana dataset from Wenliang et al. (2018) , and the model distribution q θ as an EBM trained on that dataset.

We estimate the squared bias with a stochastic upper bound, using 5,000,000 samples.

More specifically, denote the two methods as

, respectively.

We estimate the squared bias as

, where

Observe that this expectation of the estimate upper bounds the true squared bias following Cauchy's inequality; and the bias → 0 as K, M → 0.

We choose K = 100, M = 50000 and plot the confidence interval.

We also use these samples to estimate the variance of our estimator.

For the model distribution q, we choose an EBM as stated in the main text.

The energy of the model is parameterized as follows: we parameterize a d-dimensional vector ψ(x; θ) using a feed-forward network, then return x ψ(x; θ) as the energy function.

This is inspired by the "score network" parameterization in (Song et al., 2019); we note that this choice has little influence on the synthetic experiments (and is merely chosen here for consistency), but leads to improved performance in the AE experiments.

Finally, ψ(x; θ) is parameterized with 2 hidden layers and Swish activation (Ramachandran et al., 2017) , and each layer has 100 units.

We apply spectral normalization (Miyato et al., 2018) to the intermediate layers.

We train the EBM for 400 iterations with our approximation to the score matching objective, using a batch size of 200 and a learning rate of 4 × 10 −3 .

The choice of training objective is arbitrary; changing it to sliced score matching does not lead to any notable difference, as is expected from this experiment.

The results are shown in Figure 2 , in which we plot the (squared) bias and variance for both estimators, with varying step-size.

The bias plot is shown in the left.

We can see that for both estimators, the bias is negligible at ≤ 10 −2 .

We further use a z-test to compare the mean of the two estimators (for = 6 × 10 −5 ) with the mean of SSM.

The p value is 0.48 for our estimator and 0.19 for DSM, indicating there is no significant difference in either case.

The variance of the estimators, with and without our control variate, are shown in Figure  2 right.

As expected, the variance grows unbounded in absence of the control variate, and is approximately constant when it is added.

From the scale of the variance, we can see that it is exactly this variance problem that causes the failure of the original DSM estimator.

We now evaluate our approximation to the Riemannian score matching objective, by learning an unnormalized model.

The data distribution is chosen as a mixture of two von Mises distributions on S 1 :

p(x) = 0.7p vM (x|(0, 1), 2) + 0.3p vM (x|(0.5, −0.5), 3), where p vM is the von Mises density

The energy function in the model is parameterized with a feed-forward network, using the same score-network-inspired parameterization as in the last experiment.

The network uses tanh activation and has 2 hidden layers, each layer with 100 units.

We generate 50,000 samples from p(x) for training.

We use full batch training and train for 6,000 iterations, using a learning rate of 5 × 10 −4 .

The step-size hyperparameter in the MVL approximation is set to 10 −5 .

Results We plot the log densities of the ground truth distribution as well as the learned model in Figure 3 .

We can see the two functions matches closely, suggesting our method is suitable for density estimation on manifolds.

To verify our control variate also solves the variance issue in CD-1, we train EBMs using CD-1 with varying step-size, with and without our control variate, and compare the score matching loss to EBMs trained with our method as well as sliced score matching.

We use a separate experiment for CD-1 since it only estimates the gradient of the score matching loss.

The score matching loss is calculated using SSM on training set, and averaged over 3 separate runs.

We use the cosine dataset in (Wenliang et al., 2018) ; the energy parameterization is the same as in Section D.1.1.

The results are shown in Figure 4 .

We can see that with the introduction of the control variate, CD-1 performs as well as other score matching methods.

In all auto-encoder experiments, setup follows from (Song et al., 2019) whenever they applies.

The only difference is that for score estimation, we parameterize the energy function, and use its gradient as the score estimate, as opposed to directly parameterizing the score function as done in (Song et al., 2019) .

This modification makes our method applicable; essentially, it corrects the score estimation in (Song et al., 2019) so that it constitute a conservative field, which is a desirable property since score functions should be conservative.

For this reason, we re-implement all experiments for Euclidean-prior auto-encoders to ensure a fair comparison.

The results are slightly worse than (Song et al., 2019) for the VAE experiment, but significantly better for WAE experiments.

It should be noted that for the VAE experiment, our implicit hyperspherical VAE result is still better than the implicit Euclidean VAE result reported in (Song et al., 2019) .

The (conditional) energy function in this experiment is parameterized using the score-net-inspired method described in Appendix D.1, with a feed-forward network.

The network has 2 hidden layers, each with 256 hidden units.

We use tanh activation for the network, and do not apply spectral normalization.

When training the energy network, we add a L2 regularization term for the energy scale, with coefficient 10 −4 .

The coefficient is determined by grid search on {10 −3 , 10 −4 , 10 −5 }, using AIS-estimated likelihood on a heldout set created from the training set.

The step-size of the MVL approximation is set to 10 −3 ; we note that the performance is relatively insensitive w.r.t.

the step-size inside the range of [10 −4 , 10 −2 ], as suggested by the synthetic experiment.

Outside this range, using a smaller step-size makes the result worse, presumably due to floating point errors.

For implicit models, the test likelihood is computed with annealed importance sampling, using 1,000 intermediate distributions, following (Song et al., 2019) .

The transition operator in AIS is HMC for Euclidean-space latents, and Riemannian LD for hyperspherical latents.

The training setup follows from (Song et al., 2019) : for all methods, we train for 100,000 iterations using RMSProp use a batch size of 128, and a learning rate of 10 −3 .

WAE Experiment on MNIST For our method, the energy network is parameterized in the same way as in the VAE experiments.

When training the energy network, we use a step-size of 10 −4 , and apply L2 regularization on the energy scale with coefficient 10 −5 .

For the WAE-GAN baseline, we use the Wasserstein GAN (Arjovsky et al., 2017) , and parameterize its critic as a feed-forward network with 2 hidden layers, each with 256 units.

We experimented with both the standard parameterization (i.e. put a linear layer after the last hidden layer that outputs scalar) as well as the score-network-like parameterization used in our energy network, and found the results to be similar.

We use tanh activation, apply spectral normalization and a L2 regularization with coefficient 10 −4 .

The rest of the training setup follows from (Song et al., 2019) : training for 100,000 iterations using RMSProp, a batch size of 128, and a learning rate of 10 −3 .

The Lagrange multiplier hyperparameter λ in the WAE objective is fixed at 10.

The energy network is parameterized in the same way as in (Song et al., 2019) .

For our method, we use a step-size of 10 −4 .

For the GAN baseline, we use the standard parameterization for the critic, i.e. the final linear layer outputs a scalar; the previous layers follow the same architecture of ours.

In both methods we use a L2 regularization with coefficient 10 −5 .

Following (Song et al., 2019), we train for 100,000 iterations, using RMSProp and a learning rate of 10 −4 .

FID scores are calculated using the implementation in (Heusel et al., 2017) .

Notations In this section, let the parameter space be d-dimensional, and define

While in the main text, we identified the tangent space of P(X ) as a subspace of L 2 (ρX → R d ) for clarity, here we use the equivalent definition T ρ (P(X )) := {s ∈ L 2 (ρX → R) : E ρ s = 0} following (Otto, 2001 ).

The two definition are connected by the transform

In this section, we give a formal derivation of SPOS as the gradient flow of the KL divergence functional, with respect to a new metric.

Recall the SPOS sampler targeting distribution (with density) φ corresponds to the following density evolution: ∂ t ρ t = −∇ · (ρ t (x) (φ * ρt,φ (x) + α∇ log(φ/ρ)) νt(x) ) where α > 0 is a hyperparameter, and φ * ρt,φ (x) := E ρt(x ) (S φ ⊗ k)(x , x) := E ρt(x ) [(∇ x log φ(x ))k(x , x) + ∇ x k(x , x)] is the SVGD update direction (Liu and Wang, 2016; Liu, 2017) .

Fix ρ, define the integral operator

and define the tensor product operator K ⊗d

which we will derive shortly at the end of this subsection.

Subsequently, we have

The rest of our derivation follows (Otto, 2001; Liu, 2017 ): consider the function space H ρ,α := {(αId + K ⊗d ρt )[∇h]}, where h : X → R is any square integrable and differentiable function.

It connects to the tangent space of P(X ) if we consider s = −∇ · (ρp) for anỹ p ∈ H ρ,α .

Define on H ρ,α the inner product f, g Hρ,α := f, (αId + K

It then determines a Riemannian metric on the function space.

Forp ∈ H ρ,α and s = −∇·(ρp), by (20) we have ν t ,p Hρ,α = E ρt(x) ∇ log(φ/ρ t )(x),p(x) = − log φ ρ t (∇ · (pρ))dx = −(dKL φ )(s),

i.e. with respect to the metric (21), SPOS is the gradient flow of the (negative) KL divergence functional.

Derivation of (19) let (λ i , ψ i ) ∞ i=1 be its eigendecomposition (i.e. the Mercer representation).

For j ∈ [d] let ψ i,j := ψ i e j where {e j } d j=1 is the coordinate basis in R d , so {λ −1/2 i ψ i,j } becomes an orthonormal basis in H ⊗d .

Now we calculate the coordinate of φ * ρ,φ in this basis.

S φ is known to satisfy the Stein's identity

for all g ∈ H. Thus, we can subtract E ρ S ρ (K ρ [ψ i,j ]) from the right hand side of (22) without changing its value, and it becomes

As the equality holds for all i, k, we completed the derivation of (19).

By (20) and (21), the MVL objective derived from SPOS is

Hρ,α = ∇ log(φ/ρ t ), (αId + K ⊗d )∇ log(φ/ρ t ) L 2 (ρX →R d ) .

In the right hand side above, the first term in the summation is the Fisher divergence, and the second is the kernelized Stein discrepancy (Liu et al., 2016b, Definition 3.2) .

We note that a similar result for SVGD has been derived in (Liu and Wang, 2017) , and our derivations connect to the observation that Langevin dynamics can be viewed as SVGD with a Dirac function kernel (thus SPOS also corresponds to SVGD with generalized-function-valued kernels).

In this section we derive (8), when the latent-space distribution q φ (z) is defined on a pdimensional manifold embedded in some Euclidean space, and H[q φ (z)] is the relative entropy w.r.t.

the Hausdorff measure.

The derivation is largely similar to the Euclidean case, and we only include it here for completeness.

(8) holds because

where (i) follows from Theorem 2.10.10 in Federer (2014) , and (ii) follows from the same theorem as well as the fact that E q φ (z) [∇ φ log q φ (z)] = ∇ φ q φ (z)dz = 0.

@highlight

We present a scalable approximation to a wide range of EBM objectives, and applications in implicit VAEs and WAEs
In some misspecified settings, the posterior distribution in Bayesian statistics may lead to inconsistent estimates.

To fix this issue, it has been suggested to replace the likelihood by a pseudo-likelihood, that is the exponential of a loss function enjoying suitable robustness properties.

In this paper, we build a pseudo-likelihood based on the Maximum Mean Discrepancy, defined via an embedding of probability distributions into a reproducing kernel Hilbert space.

We show that this MMD-Bayes posterior is consistent and robust to model misspecification.

As the posterior obtained in this way might be intractable, we also prove that reasonable variational approximations of this posterior enjoy the same properties.

We provide details on a stochastic gradient algorithm to compute these variational approximations.

Numerical simulations indeed suggest that our estimator is more robust to misspecification than the ones based on the likelihood.

Bayesian methods are very popular in statistics and machine learning as they provide a natural way to model uncertainty.

Some subjective prior distribution π is updated using the negative log-likelihood n via Bayes' rule to give the posterior π n (θ) ∝ π(θ) exp(− n (θ)).

Nevertheless, the classical Bayesian methodology is not robust to model misspecification.

There are many cases where the posterior is not consistent (Barron et al., 1999; Grünwald and Van Ommen, 2017) , and there is a need to develop methodologies yielding robust estimates.

A way to fix this problem is to replace the log-likelihood n by a relevant risk measure.

This idea is at the core of the PAC-Bayes theory (Catoni, 2007) and Gibbs posteriors (Syring and Martin, 2018) ; its connection with Bayesian principles are discussed in Bissiri et al. (2016) .

Knoblauch et al (2019) builds a general representation of Bayesian inference in the spirit of Bissiri et al. (2016) and extends the representation to the approximate inference case.

In particular, the use of a robust divergence has been shown to provide an estimator that is robust to misspecification (Knoblauch et al, 2019) .

For instance, Hooker and Vidyashankar (2014) investigated the case of Hellinger-based divergences, Ghosal and Basu (2016) , Futami et al (2017), and Nakagawa et al. (2019) used robust β-and γ-divergences, while Catoni (2012) , Baraud and Birgé (2017) and Holland (2019) replaced the logarithm of the log-likelihood by wisely chosen bounded functions.

Refer to Jewson et al (2018) for a complete survey on robust divergence-based Bayes inference.

In this paper, we consider the Maximum Mean Discrepancy (MMD) as the alternative loss used in Bayes' formula, leading to a pseudo-posterior that we shall call MMD-Bayes in the following.

MMD is built upon an embedding of distributions into a reproducing kernel Hilbert space (RKHS) that generalizes the original feature map to probability measures, and allows to apply tools from kernel methods in parametric estimation.

Our MMD-Bayes posterior is related to the kernel-based posteriors in Fukumizu et al. (2013) , Park et al. (2016) and Ridgway (2017) , even though it is different.

More recently, Briol et al. (2019) introduced a frequentist minimum distance estimator based on the MMD distance, that is shown to be consistent and robust to small deviations from the model.

We show that our MMD-Bayes retains the same properties, i.e is consistent at the minimax optimal rate of convergence as the minimum MMD estimator, and is also robust to misspecification, including data contamination and outliers.

Moreover, we show that these guarantees are still valid when considering a tractable approximation of the MMD-Bayes via variational inference, and we support our theoretical results with experiments showing that our approximation is robust to outliers for various estimation problems.

All the proofs are deferred to the appendix.

Let us introduce the background and theoretical tools required to understand the rest of the paper.

We consider in a measurable space X, X a collection of n independent and identically distributed (i.i.d) random variables X 1 , ..., X n ∼ P 0 where P 0 is the generating distribution.

We index a statistical model {P θ /θ ∈ Θ} by a parameter space Θ, without necessarily assuming that the true distribution P 0 belongs to the model.

Let us consider some integrally strictly positive definite kernel k 1 bounded by a positive constant, say 1.

We then denote the associated RKHS (H k , ·, · H k ) satisfying the reproducing property f (x) = f, k(x, ·) H k for any f ∈ H k and any x ∈ X. We define the notion of kernel mean embedding, a Hilbert space embedding that maps probability distributions into the RKHS H k .

Given a distribution P , the kernel mean embedding µ P ∈ H k is

Then we define the MMD between two probability distributions P and Q simply as the distance in H k between their kernel mean embeddings:

Under the assumptions we made on the kernel, the kernel mean embedding is injective and the maximum mean discrepancy is a metric, see Briol et al. (2019) .

We motivate the use of MMD as a robust metric in Appendix D.

In this paper, we adopt a Bayesian approach.

We introduce a prior distribution π over the parameter space Θ equipped with some sigma-algebra.

Then we define our pseudo-Bayesian distribution π β n given a prior π on Θ:

δ X i is the empirical measure and β > 0 is a temperature parameter.

In this section, we show that the MMD-Bayes is consistent when the true distribution belongs to the model, and is robust to misspecification.

To obtain the concentration of posterior distributions in models that contain the generating distribution, Ghosal et al. (2000) introduced the so-called prior mass condition that requires the prior to put enough mass to some neighborhood (in Kullback-Leibler divergence) of the true distribution.

This condition was widely studied since then for more general pseudo-posterior distributions (Bhattacharya et al., 2019; Alquier and Ridgway, 2017; Chérief-Abdellatif and Alquier, 2018) .

Unfortunately, this prior mass condition is (by definition) restricted to cases when the model is well-specified or at least when the true distribution is in a very close neighborhood of the model.

We formulate here a robust version of the prior mass condition which is based on a neighborhood of an approximation θ * of the true parameter instead of the true parameter itself.

The following condition is suited to the MMD metric, recovers the usual prior mass condition when the model is well-specified and still makes sense in misspecified cases with potentially large deviations to the model assumptions:

Prior mass condition: Let us denote θ * = arg min θ∈Θ D k (P θ , P 0 ) and its neighborhood B n = {θ ∈ Θ/D k (P θ , P θ * ) ≤ n −1/2 }.

Then (π, β) is said to satisfy the prior mass condition C(π, β) when π(B n ) ≥ e −β/n .

In the usual Bayesian setting, the computation of the prior mass is a major difficulty (Ghosal et al., 2000) , and it can be hard to know whether the prior mass condition is satisfied or not.

Nevertheless, here the condition does not only hold on the prior distribution π but also on the temperature parameter β.

Hence, it is always possible to choose β large enough so that the prior mass condition is satisfied.

We refer the reader to Appendix E for an example of computation of such a prior mass and valid values of β.

The following theorem expressed as a generalization bound shows that the MMD-Bayes posterior distribution is robust to misspecification under the robust prior mass condition.

Note that the rate n −1/2 is exactly the one obtained by the frequentist MMD estimator of Briol et al. (2019) and is minimax optimal (Tolstikhin et al., 2017) :

Theorem 1 Under the prior mass condition C(π, β):

The second theorem investigates concentration of the MMD-Bayes posterior in the wellspecified case.

It shows that the prior mass condition C(π, β) ensures that the MMD-Bayes concentrates to P 0 at the minimax rate n −1/2 : Theorem 2 Let us consider a well-specified model.

Then under the prior mass condition C(π, β), we have in probability for any M n → +∞:

Note that we obtain the concentration to the true distribution P 0 = P θ * at the minimax rate n −1/2 for well-specified models.

Unfortunately, the MMD-Bayes is not tractable in complex models.

In this section, we provide an efficient implementation of the MMD-Bayes based on VI retaining the same theoretical properties.

Given a variational set of tractable distributions F, we define the variational approximation of π β n as the closest approximation (in KL divergence) to the target MMD posterior:π

Under similar conditions to those in Theorems 1 and 2,π β n is guaranteed to be n −1/2 -consistent as the MMD-Bayes.

Most works ensuring the consistency or the concentration of variational approximations of posterior distributions use the extended prior mass condition, an extension of the prior mass condition that applies to variational approximations rather than on the distributions they approximate (Alquier et al., 2016; Alquier and Ridgway, 2017; Bhattacharya et al., 2018; Chérief-Abdellatif and Alquier, 2018; Chérief-Abdellatif, 2019a,b) .

Here, we extend our previous prior mass condition to variational approximations but also to misspecification.

In addition to the prior mass condition inspired from Ghosal et al. (2000) , the variational set F must contain probability distributions that are concentrated around the best approximation P θ * .

This robust extended prior mass condition can be formulated as follows:

Assumption : We assume that there exists a distribution ρ n ∈ F such that:

Remark 3 When the restriction of π to the MMD-ball B n centered at θ * of radius n −1/2 belongs to F, then Assumption (4.1) becomes the standard robust prior mass condition, i.e. π(B n ) ≥ e −β/n .

In particular, when F is the set of all probability measures -that is, in the case where there is no variational approximation -then we recover the standard condition.

Theorem 4 Under the extended prior mass condition (4.1),

Moreover, if the model is well-specified, then under the prior mass condition C(π, β), we have in probability for any M n → +∞:

In this section, we show that the variational approximation is robust in practice when estimating a Gaussian mean and a uniform distribution in the presence of outliers.

We consider here a d-dimensional parametric model and a Gaussian mean-field variational set

, using componentwise multiplication.

Inspired from the stochastic gradient descent of Dziugaite et al (2015) , Li and Zemel (2015) and Briol et al. (2019) based on a U-statistic approximation of the MMD criterion, we design a stochastic gradient descent that is suited to our variational objective.

The algorithm is described in details in Appendix G.

We perform short simulations to provide empirical support to our theoretical results.

Indeed, we consider the problem of Gaussian mean estimation in the presence of outliers.

The experiment consists in randomly sampling n = 200 i.i.d observations from a Gaussian distribution N (2, 1) but some corrupted observations are replaced by samples from a standard Cauchy distribution C(0, 1).

The fraction of outliers used was ranging from 0 to 0.20 with a step-size of 0.025.

We repeated each experiment 100 times and considered the square root of the mean square error (MSE).

The plots we obtained demonstrate that our method performs comparably to the componentwise median (MED) and even better as the number of outliers increases, and clearly outperforms the maximum likelihood estimator (MLE).

We also conducted the simulations for multidimensional Gaussians and for the robust estimation of the location parameter of a uniform distribution.

We refer the reader to Appendix H for more details on these simulations.

In this paper, we showed that the MMD-Bayes posterior concentrates at the minimax convergence rate and is robust to model misspecification.

We also proved that reasonable variational approximations of this posterior retain the same properties, and we proposed a stochastic gradient algorithm to compute such approximations that we supported with numerical simulations.

An interesting future line of research would be to investigate if the i.i.d assumption can be relaxed and if the MMD-based estimator is also robust to dependency in the data.

Appendix A. Proof of Theorem 1.

In order to prove Theorem 1, we first need two preliminary lemmas.

The first one ensures the convergence of the empirical measureP n to the true distribution P 0 (in MMD distance D k ) at the minimax rate n −1/2 , and which is an expectation variant of Lemma 1 in Briol et al. (2019) that holds with high probability:

The rate n −1/2 is known to be minimax in this case, see Theorem 1 in Tolstikhin et al. (2017) .

The second lemma is a simple triangle-like inequality that will be widely used throughout the proofs of the paper:

Lemma 6 We have for any distributions P , P and Q:

Proof The chain of inequalities follow directly from the triangle inequality and inequality 2ab ≤ a 2 + b 2 .

Let us come back to the proof of Theorem 1.

An important point is that the MMDBayes can also be defined using an argmin over the set M 1 + (Θ) of all probability distributions absolutely continuous with respect to π and the Kullback-Leibler divergence KL(· ·):

This is an immediate consequence of Donsker and Varadhan's variational inequality, see e.g Catoni (2007) .

Using the triangle inequality, Lemma 5, Lemma 6 for different settings of P , P and Q, and Jensen's inequality:

which gives, using Lemma 5 and the triangle inequality again:

We remind that θ * = arg min θ∈Θ D k (P θ , P 0 ).

This bound can be formulated in the following way when ρ is chosen to be equal to π restricted to B n :

Finally, as soon as the prior mass condition C(π, β) is satisfied, we get:

Appendix B. Proof of Theorem 2.

In case of well-specification, Formula (3.1) simply becomes according to Jensen's inequality:

Hence, it is sufficient to show that the inequality above implies the concentration of the MMD-Bayes to the true distribution.

This is a simple consequence of Markov's inequality.

Indeed, for any M n → +∞:

which guarantees the convergence in mean of π β n D k (P θ , P 0 ) > M n · n −1/2 to 0, which leads to the convergence in probability of π β n D k (P θ , P 0 ) > M n ·n −1/2 to 0, i.e. the concentration of MMD-Bayes to P 0 at rate n −1/2 .

Formula (4.2) can be proven easily as for the proof of Theorem 1.

Indeed, we use the expression of the variational approximation of the MMD-Bayes using an argmin over the set F:π

This is yet an application of Donsker and Varadhan's lemma.

Then, as previously:

Hence, under the extended prior mass condition (4.1), we have directly:

The proof of Formula (4.3) follows the lines of the proof of Theorem 2.

Appendix D. An example of robustness of the MMD distance.

In this appendix, we try to give some intuition on the choice of MMD-Bayes rather than the classical regular Bayesian distribution.

To do so, we show a simple misspecified example for which the MMD distance is more suited than the classical Kullback-Leibler (KL) divergence used in the Bayes rule in the definition of the classical Bayesian posterior.

We consider the Huber's contamination model described as follows.

We observe a collection of random variables X 1 , ..., X n .

There are unobserved i.i.d random variables Z 1 , ..., Z n ∼ Ber( ) and a distribution Q, such that the distribution of X i given Z i = 0 is a Gaussian N (θ 0 , σ 2 ) where the distribution of X i given Z i = 0 is Q. The observations X i 's are independent.

This is equivalent to considering a true distribution P 0 = (1− )N (θ 0 , σ 2 )+ Q. Here, ∈ (0, 1/2) is the contamination rate, σ 2 is a known variance and Q is the contamination distribution that is taken here as N (θ c , σ 2 ), where θ c is the mean of the corrupted observations.

The true parameter of interest is θ 0 and the model is composed Gaussian distributions {P θ = N (θ, σ 2 )/θ ∈ R d }.

The goal in this appendix is to show that we exactly recover the true parameter θ 0 with the minimizer of the MMD distance to the true distribution P 0 , whereas it is not the case with the KL divergence.

We use a Gaussian kernel k(x, y) = exp(− x − y 2 /γ 2 ).

We have remind that P θ = N (θ, σ 2 I d ) where θ ∈ Θ = R d .

For independent X and Y following respectively P θ and P θ , we get (

and the square of this random variable is a noncentral chi-square random variable:

, and then t = −(2σ 2 )/γ 2 gives:

Thus,

and

Hence, the minimizer of D k (P 0 , P θ ) w.r.t θ, i.e the maximizer of:

is θ 0 itself as ≤ 1/2.

Computation of the KL divergence to the true distribution:

In this case, easy computations lead for any θ to:

where

is the cross-entropy of P θ and P θ , and

where N (x|m, σ 2 ) is the probability density function of N (m, σ 2 ) evaluated at x. Hence, the minimizer of KL (P 0 P θ ) w.r.t θ, i.e the minimizer of:

is (1 − )θ 0 + θ c , which can be far away from θ 0 in situations when the corrupted mean θ c is very far from the true parameter θ 0 .

Appendix E. An example of computation of a robust prior mass.

In this appendix, we tackle the computation of a prior mass in the Gaussian mean estimation problem, and we show that it leads to a wide range of values of β satisfying the prior mass condition C(π, β) for a standard normal prior π.

We recall that the prior mass condition C(π, β) is satisfied as soon as there exists a function f such that:

β ≥ − log π(B n )n.

In practice, lower bounds of the form π(B n ) ≥ Le −f (θ * ) naturally appear when computing the prior mass π(B n ).

Only f (θ * ) depends on the parameter θ * corresponding to the best approximation in the model of the true distribution in the MMD sense, that is the true parameter itself when the model is well-specified.

Hence, it is sufficient to choose a value of the temperature parameter β ≥ f (θ * )−log L n in order to obtain the prior mass condition.

We conduct the computation in a misspecified case, where we assume that a proportion 1 − of the observations are sampled i.i.d from a σ 2 -variate Gaussian distribution of interest P θ 0 , but that the remaining observations are corrupted and can take any arbitrary value.

We consider the model of Gaussian distributions {P θ = N (θ, σ 2 )/θ ∈ R d }.

This adversarial contamination model is more general than Huber's contamination model presented in Appendix D. Note that when = 0, then the model is well-specified and the distribution of interest P θ 0 is also the true distribution P 0 .

We use the Gaussian kernel k(x, y) = exp(− x − y 2 /γ 2 ) and the standard normal prior π = N (0, I d ).

We write the inequality defining parameters θ belonging to B n :

Note that when the model is well-specified, the we get θ * = θ 0 .

According to derivations performed in Appendix D, we have for any θ:

Hence, Inequality (E.1) is equivalent to:

We denote s n = 4σ 2 +γ 2 2n

and B(θ, s n ) the ball of radius s n and centered at θ.

Let us compute the prior mass of B n :

Actually, the point that minimizes θ → e − θ 2 /2 on B(θ * , s n ) is θ * (1 + s n / θ * ).

Thus:

We recall the formula of the volume of the d-dimensional ball:

Hence:

As could be expected for a standard normal prior, the larger the value of θ * , the smaller can be the prior mass.

We denote

Hence, for the standard normal prior π, values of β leading to consistency of the MMDBayes are:

In particular, when γ 2 is of order d, then using Stirling's approximation, we get a lower bound on the valid values of β of order (up to a logarithmic factor): n max θ * 2 , d β.

Note that when the log-density log p θ (x) is not differentiable, it is often possible to compute the stochastic gradients involving θ 1 , ..., θ M directly, without using the Monte Carlo samples Y 1 , ..., Y M .

For instance, when the model is a uniform distribution P θ = U([θ − a, θ + a]) and when the kernel can be written as k(x, y) = K(x − y) for some function K (such as Gaussian kernels), we have: Results: The error of our estimators as a function of the contamination ratio is plotted in Figures 1, 2 and 3 .

These plots show that our method is applicable to various problems and leads to a good estimator for all of them.

Indeed, the plots in Figures 1  and 2 show that the MSE for the MMD estimator performs as well as the componentwise median and even better when the number of outliers in the dataset increases, much better than the MLE in the robust Gaussian mean estimation problem, and is not affected that much by the presence of outliers in the data.

For the uniform location parameter estimation problem addressed in Figure 3 , the MMD estimator is clearly the one that performs the best and is not affected by a reasonable proportion of outliers, contrary to the method of moments which square root of MSE is increasing linearly with and to the MLE that gives inconsistent estimates as soon as there is an outlier in the data.

<|TLDR|>

@highlight

Robust Bayesian Estimation via Maximum Mean Discrepancy
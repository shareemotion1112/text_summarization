Normalizing constant (also called partition function, Bayesian evidence, or marginal likelihood) is one of the central goals of Bayesian inference, yet most of the existing methods are both expensive and inaccurate.

Here we develop a new approach, starting from posterior samples obtained with a standard Markov Chain Monte Carlo (MCMC).

We apply a novel Normalizing Flow (NF) approach to obtain an analytic density estimator from these samples, followed by Optimal Bridge Sampling (OBS) to obtain the normalizing constant.

We compare our method which we call Gaussianized Bridge Sampling (GBS) to existing methods such as Nested Sampling (NS) and Annealed Importance Sampling (AIS) on several examples, showing our method is both significantly faster and substantially more accurate than these methods, and comes with a reliable error estimation.

Normalizing constant, also called partition function, Bayesian evidence, or marginal likelihood, is the central object of Bayesian methodology.

Despite its importance, existing methods are both inaccurate and slow, and may require specialized tuning.

One such method is Annealed Importance Sampling (AIS), and its alternative, Reverse AIS (RAIS), which can give stochastic lower and upper bounds to the normalizing constant, bracketing the true value (Neal, 2001; Grosse et al., 2015) .

However, as the tempered distribution may vary substantially with temperature, it can be expensive to obtain good samples at each temperature, which can lead to poor estimates (Murray et al., 2006) .

Nested sampling (NS) is another popular alternative (Skilling, 2004; Handley et al., 2015) , which can be significantly more expensive than standard sampling methods in higher dimensions but, as we show, can also lead to very inaccurate estimates.

Moreover, there is no simple way to know how accurate the estimate is.

Here we develop a new approach to the problem, combining Normalizing Flow (NF) density estimators with Optimal Bridge Sampling (OBS).

In a typical Bayesian inference application, we first obtain posterior samples using one of the standard Markov Chain Monte Carlo (MCMC) methods.

In our approach we use these samples to derive the normalizing constant with relatively few additional likelihood evaluations required, making the additional cost of normalizing constant estimation small compared to posterior sampling.

All of our calculations are run on standard CPU platforms, and will be available in the BayesFast Python package.

Let p(x) and q(x) be two possibly unnormalized distributions defined on Ω, with normalizing constants Z p and Z q .

For any function α(x) on Ω, we have

if the integral exists.

Suppose that we have samples from both p(x) and q(x), and we know Z q , then Equation (1) gives

which is the Bridge Sampling estimation of normalizing constant (Meng and Wong, 1996) .

It can be shown that many normalizing constant estimators, including Importance Sampling and Harmonic Mean, are special cases with different choices of bridge function α(x) (Gronau et al., 2017) .

For a given proposal function q(x), an asymptotically optimal bridge function can be constructed, such that the ratio r = Z p /Z q is given by the root of the following score function equation

where n p and n q are the numbers of samples from p(x) and q(x).

For r ≥ 0, S(r) is monotonic and has a unique root, so one can easily solve it with e.g. secant method.

This estimator is optimal, in the sense that its relative mean-square error is minimized (Chen et al., 2012) .

Choosing a suitable proposal q(x) for Bridge Sampling can be challenging, as it requires a large overlap between q(x) and p(x).

One approach is Warp Bridge Sampling (WBS) (Meng and Schilling, 2002) , which transforms p(x) to a Gaussian with linear shifting, rescaling and symmetrizing.

As we will show, this approach can be inaccurate or even fail completely for more complicated probability densities.

As stated above, an appropriate proposal q(x) which has large overlap with p(x) is required for OBS to give accurate results.

In a typical MCMC analysis we have samples from the posterior, so one can obtain an approximate density estimation q(x) from these samples using a bijective NF.

In this approach one maps p(x) to an unstructured distribution such as zero mean unit variance Gaussian N (0, I).

For density evaluation we must also keep track of the Jacobian of transformation |dΨ/dx|, so that our estimated distribution is q(x) = N (0, I)|dΨ/dx|, where Ψ(x) is the transformation.

The probability density q(x) is normalized, so we know Z q = 1.

There have been various methods of NF recently proposed in machine learning literature (Dinh et al., 2014 (Dinh et al., , 2016 Papamakarios et al., 2017) , which however failed on several examples we present below.

Moreover, we observed that training with these is very expensive and can easily dominate the overall computational cost.

For these reasons we instead develop Iterative Neural Transform (INT), a new NF approach, details of which will be presented elsewhere.

It is based on combining optimal transport and information theory, repeatedly finding and transforming one dimensional marginals that are the most deviant between the target and proposal (Gaussian) distributions.

After computing dual representation of Wasserstein-1 distance to find the maximally non-Gaussian directions, we apply a bijective transformation that maximizes the entropy along these directions.

For this we use a non-parametric spline based transformation that matches the 1-d cumulative distribution function (CDF) of the data to a Gaussian CDF, where kernel density estimation (KDE) is used to smooth the probability density marginals.

We found that using a fixed number of 5 to 10 iterations is sufficient for evidence estimation, and the computational cost of our NF density estimation is small when compared to the cost of sampling.

We propose the following Gaussianized Bridge Sampling (GBS) approach, which combines OBS with NF density estimation.

In our typical application, we first run No-U-Turn Sampler (NUTS) (Hoffman and Gelman, 2014 ) to obtain 2n p samples from p(x) if its gradient is available, while affine invariant sampling (Foreman-Mackey et al., 2013) can be used in the gradient-free case.

To avoid underestimation of Z p (Overstall and Forster, 2010) , these 2n p samples are divided into two batches, and we fit INT with the first batch of n p samples to obtain the proposal q(x).

Then we draw n q samples from q(x) and evaluate their corresponding p(x), where n q is determined by an adaptive rule (see Appendix B.4).

We solve for the normalizing constant ratio r with Equation (3), using these n q samples from q(x) and the second batch of n p samples from p(x) (also evaluating their corresponding q(x)), and report the result in form of ln Z p , with its error approximated by the relative mean-square error of Z p given in Equation (9) (Chen et al., 2012) .

We used four test problems to compare the performance of various estimators.

See Appendix A and B for more details of the examples and algorithms.

(1) The 16-d Funnel example is adapted from Neal et al. (2003) .

The funnel structure is common in Bayesian hierarchical models, and in practice it is recommended to reparameterize the model to overcome the pathology (Betancourt and Girolami, 2015) .

Here we stick to the original parameterization for test purpose.

(2) The 32-d Banana example comes from a popular variant of multidimensional Rosenbrock function (Rosenbrock, 1960) , which is composed of 16 uncorrelated 2-d bananas.

In addition, we apply a random 32-d rotation to the bananas, which makes all the parameters correlated with each other.

(3) The 48-d Cauchy example is adapted from the LogGamma example in Feroz et al. (2013); Buchner (2016) .

In contrast to the original example, where the mixture structure only exists in the first two dimensions, we place a mixture of two heavy-tailed Cauchy distributions along every dimension.

(4) The 64-d Ring example has strong non-linear correlation between the parameters, as the marginal distribution of every two successive parameters is ring-shaped.

See Figure 1 for a comparison of the estimators.

For all of the four test examples, the proposed GBS algorithm gives the most accurate result and a valid error estimation.

We use NS as implemented in dynesty (Speagle, 2019) with its default settings.

For all other cases, we use NUTS as the MCMC transition operator.

We chose to run (R)AIS with equal number of evaluations as our GBS, but as seen from Figure 1 this number is inadequate for (R)AIS, which needs about 10-100 times more evaluations to achieve sufficient accuracy (see Appendix B.3).

In contrast, if we run GBS with 4 times fewer evaluations (Gaussianized Bridge Sampling Lite, GBSL), we achieve an unbiased result with a larger error than GBS, but still smaller than other estimators.

For comparison we also show results replacing OBS with IS (GIS) or HM (GHM), while still using INT for q(x).

Although GIS and GHM are better than NS or (R)AIS, they are worse than GBS(L), highlighting the importance of OBS.

Finally, we also compare to WBS, which uses a very simple proposal distribution q(x), and fails on several examples, highlighting the importance of using a more expressive NF for q(x).

For our GBS(L), most of evaluation time is used to get the posterior samples with standard MCMC, which is a typical Bayesian inference goal, and the additional cost to evaluate evidence is small compared to the MCMC (see Appendix B.4).

In contrast, Thermodynamic Integration (TI) or (R)AIS is more expensive than posterior sampling, since the chains need to be accurate at every intermediate state (Neal, 1993) .

The same comment applies to NS, which is more expensive than the MCMC approaches we use here for posterior analysis, especially when non-informative prior is used.

We present a new method to estimate the normalizing constant (Bayesian evidence) in the context of Bayesian analysis.

Our starting point are the samples from the posterior using standard MCMC based methods, and we assume that these have converged to the correct probability distribution.

In our approach we combine OBS with INT, a novel NF based density estimator, showing on several high dimensional examples that our method outperforms other approaches in terms of accuracy and computational cost, and provides a reliable error estimate.

The model likelihood is

with flat prior x 1 ∼ U(−4, 4), x 2:n ∼ U(−30, 30).

We use ln Z p = −63.4988 as the fiducial value, and the corner plot of the first four dimensions is shown in Figure 2 .

The model likelihood is

with flat prior U(−15, 15) on all the parameters.

The rotation matrix A is generated from a random sample of SO(n), and the same A is used for all the simulations.

We use ln Z p = −127.364 as the fiducial value, and the corner plot of the first four dimensions, without or with the random rotation, is shown in Figure 3 .

The strong degeneracy can no longer be identified in the plot once we apply the rotation, however it still exists and hinders most estimators from getting reasonable results.

The model likelihood is

with flat prior U(−100, 100) on all the parameters.

We use ln Z p = −254.627 as the fiducial value, and the corner plot of the first four dimensions is shown in Figure 4 .

The model likelihood is

, a = 2, b = 1, n = 64, (7) with flat prior U(−5, 5) on all the parameters.

We use ln Z p = −114.492 as the fiducial value, and the corner plot of the first four dimensions is shown in Figure 5 .

We use dynamic NS implemented in dynesty, which is considered more efficient than static NS.

Traditionally, NS does not need the gradient of the likelihood, at the cost of lower sampling efficiency in high dimensions.

Since analytic gradient of the four examples is available, we follow dynesty's default setting, which requires the gradient to perform Hamitonian Slice Sampling for dimensions d > 20.

While for dimensions 10 ≤ d ≤ 20, random walks sampling is used instead.

dynesty also provides an error estimate for the evidence; see Speagle (2019) for details.

For (R)AIS, we divide the warm-up iterations of NUTS into two equal stages, and the (flat) prior is used as the base density.

In the first stage, we set β = 0.5 and adapt the mass matrix and step size of NUTS, which acts as a compromise between the possibly broad prior and narrow posterior.

In the second stage, we set β = 0 (β = 1) for AIS (RAIS) to get samples from the prior (posterior).

After warm-up, we use the following sigmoidal schedule to perform annealing,

where σ denotes the logistic sigmoid function and we set δ = 4 (Grosse et al., 2015) .

We use 1,000 warm-up iterations for all the four examples, and adjust the number of states T so that it needs roughly the same number of evaluations as GBS in total.

The exact numbers are listed in Table 1 .

We run 16 chains for each case, and average reported ln Z p of different chains, which gives a stochastic lower (upper) bound for AIS (RAIS) according to Jensen's inequality.

The uncertainty is estimated from the scatter of different chains, and should be understood as the error of the lower (upper) bound of ln Z p , instead of ln Z p itself.

Funnel Banana Cauchy Ring AIS 800 2000 3000 3500 RAIS 700 1500 2500 3000 Table 1 : The number of states T used by (R)AIS.

Using the mass matrix and step size of NUTS adapted at β = 0.5, and the prior as base density, may account for the phenomenon that RAIS failed to give an upper bound in the Banana example: the density is very broad at high temperatures and very narrow at low temperatures, which is difficult for samplers adapted at a single β.

One may remedy this issue by using a better base density that is closer to the posterior, but this will require delicate hand-tuning and is beyond the scope of this paper.

While the upper (lower) bounds of (R)AIS are valid in the limit of a very large number of samples, achieving this limit may be extremely costly in practice.

The remaining normalizing constant estimators require a sufficient number of samples from p(x), which we obtain with NUTS.

For WBS, GBS, GIS and GHM, we run 8 chains with 2,500 iterations for the Funnel and Banana examples, and 5,000 iterations for the Cauchy and Ring examples, including the first 20% warm-up iterations, which are removed from the samples.

Then we fit INT using 10 iterations for GBS, GIS and GHM, whose computation cost (a few seconds for the Funnel example) is small or negligible relative to NUTS sampling, and does not depend on the cost of ln p(x) evaluations.

For GBSL, the number of NUTS chains, NUTS iterations and INT iterations are all reduced by half, leading to a factor of four decrease in the total computation cost.

The relative mean-square error of OBS is minimized and given by

where

np+nq .

Here p (x) = p(x)/Z p and q(x) should be normalized densities.

We assume the samples from q(x) are independent, whereas the samples from p(x) may be autocorrelated, and τ f 2 is the integrated autocorrelation time of f 2 (x p ) (Frühwirth-Schnatter, 2004) , which is estimated by the autocorr module in emcee (Foreman-Mackey et al., 2013) .

Analogous expressions can be derived for IS and HM,

The claimed uncertainty in Figure 1 is obtained by assuming that the error is Gaussian distributed.

There can be different strategies to allocate samples for BS.

In the literature, it is recommended that one draws samples from p(x) and q(x) based on equal-sample-size or equal-time allocation (Bennett, 1976; Meng and Wong, 1996) .

Since NUTS based sampling usually requires at least hundreds of evaluations to obtain one effective sample from p(x) in high dimensions (Hoffman and Gelman, 2014) , which is orders of magnitude more expensive than our NF based sampling for q(x), it could be advantageous to set n q > n p .

Throughout this paper, the following adaptive strategy is adopted to determine n q for GBS(L).

After obtaining 2n p samples from p(x), we divide them into two equal batches, which will be used for fitting the proposal q(x) and evaluating the evidence, respectively.

As an starting point, we draw n q,0 = n p samples from q(x) and estimate the error of OBS using Equation (9).

Note that the right side of Equation (9) is composed of two terms, and only the first term will decrease as one increases n q but fixes n p .

Assuming that current samples provide an accurate estimate of the variance and expectation terms in Equation (9), one can solve for n q such that f err , the fraction of q(x) contributions in Equation (9), is equal to some specified value, which we set to 0.1.

Since the n q,0 samples from q(x) can be reused, if n q < n q,0 , no additional samples are required and we set n q = n q,0 .

On the other hand, we also require that f eva , the fraction of p(x) evaluations that are used for the q(x) samples, is no larger than 0.1, although this constraint is usually not activated in practice.

We use 0.1 as the default values of f err and f eva , so that the additional cost of evidence evaluation is small relative to the cost of sampling, while using a larger n q alone can no longer significantly improve the accuracy of normalizing constant.

However, if one wants to put more emphasis on posterior sampling (evidence estimation), a larger (smaller) f err and/or smaller (larger) f eva can be used.

In principle, it is also possible to use different number of p(x) samples to fit the proposal and evaluate the evidence, in contrast to equal split used in Overstall and Forster (2010), which we leave for feature research.

For GIS and WBS, we use the same n q as solved for GBS(L).

No samples from p(x) are required to estimate normalizing constant for GIS, so in this case all the 2n p samples will be used to fit INT.

While for GHM, no samples from q(x) are required.

Note that for WBS, the additional p(x) evaluations required for evidence estimation is n p + 2n q instead of n q , which comes from the symmetrization of ln p(x).

<|TLDR|>

@highlight

We develop a new method for normalization constant (Bayesian evidence) estimation using Optimal Bridge Sampling and a novel Normalizing Flow, which is shown to outperform existing methods in terms of accuracy and computational time.
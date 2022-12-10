Learning in Gaussian Process models occurs through the adaptation of hyperparameters of the mean and the covariance function.

The classical approach entails maximizing the marginal likelihood yielding fixed point estimates (an approach called Type II maximum likelihood or ML-II).

An alternative learning procedure is to infer the posterior over hyperparameters in a hierarchical specification of GPs we call Fully Bayesian Gaussian Process Regression (GPR).

This work considers two approximations to the intractable hyperparameter posterior, 1) Hamiltonian Monte Carlo (HMC) yielding a sampling based approximation and 2) Variational Inference (VI) where the posterior over hyperparameters is approximated by a factorized Gaussian (mean-field) or a full rank Gaussian accounting for correlations between hyperparameters.

We analyse the predictive performance for fully Bayesian GPR on a range of benchmark data sets.

The Gaussian process (GP) posterior is heavily influenced by the choice of the covariance function which needs to be set a priori.

Specification of a covariance function and setting the hyperparameters of the chosen covariance family are jointly referred to as the model selection problem (Rasmussen and Williams, 2004) .

A preponderance of literature on GPs address model selection through maximization of the marginal likelihood, ML-II (MacKay, 1999) .

This is an attractive approach as the marginal likelihood is tractable in the case of a Gaussian noise model.

Once the point estimate hyperparameters have been selected typically using conjugate gradient methods the posterior distribution over latent function values and hence predictions can be derived in closed form; a compelling property of GP models.

While straightforward to implement the non-convexity of the marginal likelihood surface can pose significant challenges for ML-II.

The presence of multiple modes can make the process prone to overfitting especially when there are many hyperparameters.

Further, weakly identified hyperparameters can manifest in flat ridges in the marginal likelihood surface (where different combinations of hyperparameters give similar marginal likelihood value) (Warnes and Ripley, 1987) making gradient based optimisation extremely sensitive to starting values.

Overall, the ML-II point estimates for the hyperparameters are subject to high variability and underestimate prediction uncertainty.

The central challenge in extending the Bayesian treatment to hyperparameters in a hierarchical framework is that their posterior is highly intractable; this also renders the predictive posterior intractable.

The latter is typically handled numerically by Monte Carlo integration yielding a non-Gaussian predictive posterior; it yields in fact a mixture of GPs.

The key question about quantifying uncertainty around covariance hyperparameters is examining how this effect propagates to the posterior predictive distribution under different approximation schemes.

Given observations (X, y)

where y i are noisy realizations of some latent function values f corrupted with Gaussian noise,

j ) denote a positive definite covariance function parameterized with hyperparameters θ and the corresponding covariance matrix K θ .

The hierarchical GP framework is given by, Prior over hyperparameters θ ∼ p(θ)

The generative model in (1) implies the joint posterior over unknowns given as,

where Z is the unknown normalization constant.

The predictive distribution for unknown test inputs X integrates over the joint posterior,

(where we have suppressed the conditioning over inputs X, X for brevity).

The inner integral p(f |f , y, θ)p(f |θ, y)df reduces to the standard GP predictive posterior with fixed hyperparameters,

where,

where K θ denotes the covariance matrix evaluated between the test inputs X and K * θ denotes the covariance matrix evaluated between the test inputs X and training inputs X.

Under a Gaussian noise setting the hierarchical predictive posterior is reduced to,

where f is integrated out analytically and θ j are draws from the hyperparameter posterior.

The only intractable integral we need to deal with is p(θ|y) ∝ p(y|θ)p(θ) and predictive posterior follows as per eq. (6).

Hence, the hierarchical predictive posterior is a multivariate mixture of Gaussians (Appendix section 6.2).

The distinct advantage of HMC over other MCMC methods is the suppression of the random walk behaviour typical of Metropolis and variants.

Refer to Neal et al. (2011) for a detailed tutorial.

In the experiments we use a self-tuning variant of HMC called the No-U-TurnSampler (NUTS) proposed in Hoffman and Gelman (2014) in which the path length is deterministically adjusted for every iteration.

Empirically, NUTS is shown to work as well as a hand-tuned HMC.

By using NUTS we avoid the overhead in determining good values for the step-size ( ) and path length (L).

We use an identity mass matrix with 500 warm-up iterations and run 4 chains to detect mode switching which can sometimes adversely affect predictions.

Further, the primary variables are declared as the log of the hyperparameters log(θ) as this eliminates the positivity constraints that we otherwise we need to account for.

The computational cost of the HMC scheme is dominated by the need to invert the covariance matrix K θ which is O(N 3 ).

We largely follow the approach in Kucukelbir et al. (2017) .

We transform the support of hyperparameters θ such that they live in the real space R J where J is the number of hyperparameters.

Let η = g(θ) = log(θ) and we proceed by setting the variational family to,

in the mean-field approximation where λ mf = (µ 1 , . . .

, µ J , ν 1 , . . . , ν J ) is the vector of unconstrained variational parameters (log(σ 2 j ) = ν j ) which live in R 2J .

In the full rank approximation the variational family takes the form,

where we use the Cholesky factorization of the covariance matrix Σ so that the variational parameters λ f r = (µ, L) are unconstrained in R J+J(J+1)/2 .

The variational objective, ELBO is maximised in the transformed η space using stochastic gradient ascent and any intractable expectations are approximated using monte carlo integration.

where the term |J g −1 (η)| denotes the Jacobian of the inverse transformation

hinges on automatic differentiation and the re-parametrization trick (Kingma and Welling (2013)).

The computational cost per iteration is O(N M J) where J is the number of hyperparameters and M is the number of MC samples used in computing stochastic gradients.

We evaluate 4 UCI benchmark regression data sets under fully Bayesian GPR (see Table  1 ).

For VI we evaluate the mean-field and full-rank approximations.

The top line shows the baseline ML-II method.

The two metrics shown are: 1) RMSE -square root mean squared error and 2) NLPD -negative log of the predictive density averaged across test data.

Except for 'wine' which is a near linear dataset, HMC and full-rank variational schemes exceed the performance of ML-II.

By looking at Fig.1 one can notice how the prediction intervals under the full Bayesian schemes capture the true data points.

HMC generates a wider span of functions relative to VI (indicated by the uncertainty interval 1 ).

The mean-field (MF) performance although inferior to HMC and full-rank (FR) VI still dominates the ML-II method.

Further, while HMC is the gold standard and gives a more exact approximation, the VI schemes provide a remarkably close approximation to HMC in terms of error.

The higher RMSE of the MF scheme compared to FR and HMC indicates that taking into account correlations between the hyperparameters improves prediction quality.

Data set CO 2 Wine Concrete Airline

We demonstrate the feasibility of fully Bayesian GPR in the Gaussian likelihood setting for moderate sized high-dimensional data sets with composite kernels.

We present a concise comparative analysis across different approximation schemes and find that VI schemes based on the Gaussian variational family are only marginally inferior in terms of predictive performance to the gold standard HMC.

While sampling with HMC can be tuned to generate samples from multi-modal posteriors using tempered transitions (Neal, 1996) , the predictions can remain invariant to samples from different hyperparameter modes.

Fully Bayesian bottom: Airline).

In the CO 2 data where we undertake long-range extrapolation, the uncertainty intervals under the full Bayesian schemes capture the true observations while ML-II underestimates predictive uncertainty.

For the Airline dataset, red in each twoway plot denotes ML-II, the uncertainty intervals under the full Bayesian schemes capture the upward trend better than ML-II.

The latter also misses on structure that the other schemes capture.

inference in GPs is highly intractable and one has to consider the trade-off between computational cost, accuracy and robustness of uncertainty intervals.

Most interesting real-world applications of GPs entail hand-crafted kernels involving many hyperparameters where there risk of overfitting is not only higher but also hard to detect.

A more robust solution is to integrate over the hyperparameters and compute predictive intervals that reflect these uncertainties.

An interesting question is whether conducting inference over hierarchies in GPs increases expressivity and representational power by accounting for a more diverse range of models consistent with the data.

More specifically, how does it compare to the expressivity of deep GPs (Damianou and Lawrence, 2013) with point estimate hyperparameters.

Further, these general approximation schemes can be considered in conjunction with different incarnations of GP models where transformations are used to warp the observation space yielding warped GPs (Snelson et al., 2004) or warp the input space either using parametric transformations like neural nets yielding deep kernel learning (Wilson et al., 2016) or non-parametric ones yielding deep GPs (Damianou and Lawrence, 2013 6.

Appendix

In early accounts, Neal (1998), Williams and Rasmussen (1996) and Barber and Williams (1997) explore the integration over covariance hyperparameters using HMC in the regression and classification setting.

More recently, Murray and Adams (2010) use a slice sampling scheme for covariance hyperparameters in a general likelihood setting specifically addressing the coupling between latent function values f and hyperparameters θ.

Filippone et al. (2013) conduct a comparative evaluation of MCMC schemes for the full Bayesian treatment of GP models.

Other works like Hensman et al. (2015) explore the MCMC approach to variationally sparse GPs by using a scheme that jointly samples inducing points and hyperparameters.

Flaxman et al. (2015) explore a full Bayesian inference framework for regression using HMC but only applies to separable covariance structures together with grid-structured inputs for scalability.

On the variational learning side, Snelson and Ghahramani (2006) ; Titsias (2009) jointly select inducing points and hyperparameters, hence the posterior over hyperparameters is obtained as a side-effect where the inducing points are the main goal.

In more recent work, Yu et al. (2019) propose a novel variational scheme for sparse GPR which extends the Bayesian treatment to hyperparameters.

Extract the 2.5 th percentile ⇒ f i(r l ) where r l = 2.5 100 × T Extract the 97.5 th percentile ⇒ f i(ru) where r u = 97.5

All the four data sets use composite kernels constructed from base kernels.

Table 2 summarizes the base kernels used and the set of hyperparameters for each kernel.

All hyperparameters are given vague N (0, 3) priors in log space.

Due to the sparsity of Airline data, several of the hyperparameters were weakly identified and in order to constrain inference to a reasonable range we resorted to a tighter normal prior around the ML-II estimates and Gamma(2, 0.1) priors for the noise hyperparameters.

All the experiments were done in python using pymc3 (Salvatier et al., 2016).

In the case of HMC, 4 chains were run to convergence and one chain was selected to compute predictions.

For mean-field and full rank VI, a convergence threshold of 1e-4 was set for the variational parameters, optimisation terminated when all the variational parameters (means and standard deviations) concurrently changed by less than 1e-4.

For 'wine' and 'concrete' data sets we use a random 50/50 training/test split.

For 'CO 2 ' we use the first 545 observations as training and for 'Airline' we use the first 100 observations as training.

Table 2 : Base kernels used in the UCI experiments.

k SE denotes the squared exponential kernel, k ARD denotes the automatic relevance determination kernel (squared exponential over dimensions), k P er denotes the periodic kernel, k RQ denotes the rational quadratic kernel and k N oise denotes the white kernel for stationary noise.

Data set Composite Kernel

In the figures and tables below, a prefix 's' denotes signal std.

deviation, a prefix 'ls' denotes lengthscale and a prefix 'n' denotes noise std.

deviation.

The figure below shows marginal posteriors of the hyperparamters used in the Airline kernel.

We can make the following remarks:

1.

It is evident that sampling and variational optimisation do not converge to the same region of the hyperparameter space as ML-II.

2.

Given that the predictions are better under the full Bayesian schemes, this indicates that ML-II is in an inferior local optimum.

3.

The mean-field marginal posteriors are narrower than the full rank and HMC posteriors as is expected.

Full rank marginal posteriors closely approximate the HMC marginals.

4.

The noise std.

deviation distribution learnt under the full Bayesian schemes is higher than ML-II point estimate indicating overfitting in this particular example.

@highlight

Analysis of Bayesian Hyperparameter Inference in Gaussian Process Regression 
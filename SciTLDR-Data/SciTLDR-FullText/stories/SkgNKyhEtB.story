Surrogate models can be used to accelerate approximate Bayesian computation (ABC).

In one such framework the discrepancy between simulated and observed data is modelled with a Gaussian process.

So far principled strategies have been proposed only for sequential selection of the simulation locations.

To address this limitation, we develop Bayesian optimal design strategies to parallellise the expensive simulations.

We also touch the problem of quantifying the uncertainty of the ABC posterior due to the limited budget of simulations.

Approximate Bayesian computation (Marin et al., 2012; Lintusaari et al., 2017 ) is used for Bayesian inference when the analytic form of the likelihood function of a statistical model of interest is either unavailable or too costly to evaluate, but simulating the model is feasible.

Unfortunately, many models e.g. in genomics and epidemiology (Numminen et al., 2013; Marttinen et al., 2015; McKinley et al., 2018) and climate science (Holden et al., 2018) are costly to simulate making sampling-based ABC inference algorithms infeasible.

To increase sample-efficiency of ABC, various methods using surrogate models such as neural networks (Papamakarios and Murray, 2016; Papamakarios et al., 2019; Lueckmann et al., 2019; Greenberg et al., 2019) and Gaussian processes (Meeds and Welling, 2014; Wilkinson, 2014; Gutmann and Corander, 2016; Järvenpää et al., 2018 Järvenpää et al., , 2019a have been proposed.

In one promising surrogate-based ABC framework the discrepancy between the observed and simulated data is modelled with a Gaussian process (GP) (Gutmann and Corander, 2016; Järvenpää et al., 2018 Järvenpää et al., , 2019a .

Sequential Bayesian experimental design (or active learning) methods to select the simulation locations so as to maximise the sample-efficiency in this framework were proposed by Järvenpää et al. (2019a) .

However, one often has access to multiple computers to run some of the simulations in parallel.

In this work, motivated by the related problem of batch Bayesian optimisation (Ginsbourger et al., 2010; Desautels et al., 2014; Shah and Ghahramani, 2015; Wu and Frazier, 2016) and the parallel GP-based method by Järvenpää et al. (2019b) for inference tasks where noisy and potentially expensive log-likelihood evaluations can be obtained, we resolve this limitation by developing principled batch simulation methods which considerably decrease the wall-time needed for ABC inference.

The posterior distribution is often summarised for further decision making using e.g. expectation and variance.

When the computational resources for ABC inference are limited, it would be important to assess the accuracy of such summaries, but this has not been explicitly acknowledged in earlier work.

We devise an approximate numerical method to propagate the uncertainty of the discrepancy, represented by the GP model, to the resulting ABC posterior summaries.

We call our resulting framework as Bayesian ABC in analogy with the related problems of Bayesian quadrature (O'Hagan, 1991; Osborne et al., 2012; Briol et al., 2019) and Bayesian optimisation (BO) (Brochu et al., 2010; Shahriari et al., 2015) .

Let π(θ) denote the prior density of the (continuous) parameters θ ∈ Θ ⊂ R p of a statistical model of interest and π(x obs | θ) corresponding intractable likelihood function.

Standard ABC algorithms such as the ABC rejection sampler target the approximate posterior

batch setting where b simulations are simultaneously selected to be computed in parallel at each iteration of our algorithm.

Consider a loss function l : D 2 → R + so that l(π ABC , d) quantifies the penalty of reporting d ∈ D as our ABC posterior when the true one is π ABC ∈ D. Given D t , the one-batch-ahead Bayes-optimal selection of the next batch of b

, where

In Eq. 3, an expectation over b future discrepancy evaluations ∆ * = (∆ * 1 , . . .

, ∆ * b ) at locations θ * needs to be computed assuming ∆ * follows the current GP model.

The expectation is taken of the Bayes risk L(Π f Dt∪D * ) resulting from the nested decision problem of choosing the estimator d, assuming ∆ * are known and merged with current data

.

Using a loss function based onπ

) 2 dθ between the unnormalised ABC posteriorπ f ABC and its estimatord, then the optimal estimator is the mean in Eq. 11.

The resulting expected integrated variance (EIV) acquisition function, denoted as

where

and T is Owen's T function.

We use greedy optimisation as is also common in batch BO (see, e.g., Snoek et al., 2012; and the integral over Θ is computed using importance sampling.

We can also show that the corresponding L 1 loss function produces the marginal median in Eq. 13 of the Appendix as the optimal estimator.

The resulting acquisition function, called expected integrated MAD (EIMAD), in addition to some heuristically-motivated batch methods used as baselines (called MAXV, MAXMAD), are developed in Appendix A.2.

Pointwise marginal uncertainty of the unnormalised ABC posteriorπ f ABC was used for selecting the simulation locations.

However, knowingπ f ABC and its marginal uncertainty in some individual θ-values is not very helpful for understanding the accuracy of the final estimate of π f ABC .

Computing the distribution of e.g. ABC posterior expectation or marginals using π f ABC in Eq. 2 is clearly more intuitive.

Unfortunately, such computations are difficult due to the nonlinear dependence on the infinite-dimensional quantity f .

We propose a simulation-based approach where we combine drawing of GP sample paths and normalised importance sampling.

For full details and an illustration, see Appendix A.3 and Fig. 3 .

We use two real-world simulation models to compare the performance of the sequential and synchronous batch versions of the acquisition methods.

As a simple baseline, we consider random points (RAND) drawn from the prior.

ABC-MCMC (Marjoram et al., 2003) with extensive simulations is used to compute the ground truth ABC posterior.

Median total variation distance (TV) over 50 repeated simulations is used to measure the quality of approximations.

See Appendix B for further details and C for additional results.

Lorenz model.

This model describes the dynamics of slow weather variables and their dependence on unobserved fast weather variables.

The model is represented by a coupled stochastic differential equation which can only be solved numerically resulting in an intractable likelihood.

The model has two parameters θ = (θ 1 , θ 2 ) which we estimate from timeseries data.

See Thomas et al. (2018) for full details and the experimental set-up that we also use, with the exception that we set θ ∼ U([0, 5]×[0, 0.5]).

The results are shown in Fig. 1(a) .

Furthermore, Fig. 1 (b-c) demonstrates the uncertainty quantification of the expectation of the model-based ABC posterior.

The effect of batch size is shown in Fig. 2(c) .

Bacterial infections model.

This model describes transmission dynamics of bacterial infections in day care centers and features intractable likelihood function (Numminen et al., 2013) .

We estimate the internal, external and co-infection parameters β ∈ [0, 11], Λ ∈ [0, 2] and θ ∈ [0, 1], respectively, using true data (Numminen et al., 2013) and uniform priors.

The discrepancy is formed as in Gutmann and Corander (2016) .

The results with all methods are shown in Fig. 2(a) and Fig. 2(b) shows the effect of batch size for the two best methods.

Discussion.

We obtain reasonable posterior approximations considering the very limited budget of simulations.

EIV and EIMAD tend to produce more stable and accurate ABC posterior estimates than MAXV and MAXMAD.

Difference in approximation quality between EIV and EIMAD, both based on the same Bayesian decision theoretic framework but different loss functions, was small.

In all cases, our batch strategies produced similar evaluation locations as the corresponding sequential methods.

This suggests that substantial improvements in wall-time can be obtained when the simulations are costly.

The convergence of the uncertainty in the ABC posterior expectation in Fig. 1(b-c) is approximately towards the true ABC posterior expectation due to a slight GP misspecification.

The ABC posterior marginals of the bacterial infection model in Appendix C contain some uncertainty after 600 iterations which our approach allows to rigorously quantify.

Developing more effective (analytical) methods for computing these uncertainty estimates is an interesting topic for future work.

corresponding vector of test point θ.

For further details of GP regression, see e.g. Rasmussen and Williams (2006) .

Formulas for the mean, median and variance ofπ f ABC were derived by Järvenpää et al. (2019a) in the case of a zero mean GP prior.

It is easy to see that these formulas hold also for our more general GP model.

For example,

where med denotes the marginal (i.e. elementwise) median.

The EIMAD acquisition function, denoted as L m t (θ * ) can be shown to be

where, similarly as for EIV in Eq. 4, T is Owen's T function (Owen, 1956 ) and a t is given by Eq. 12.

MAD stands for mean absolute deviation (around median).

We do not show a detailed derivation of EIV and EIMAD acquisition functions here but only note that these can be obtained using similar computations as in Järvenpää et al. (2019a,b) .

We consider also a heuristic acquisition function which evaluates where the pointwise uncertainty ofπ f ABC (θ) is highest.

Such intuitive strategy is sometimes called as uncertainty sampling and used, e.g., by Gunter et al. (2014) , Järvenpää et al. (2019a) and Chai and Garnett (2019) .

When variance is used as the measure of uncertainty ofπ f ABC (θ), we call the method as MAXV and when MAD is used, we obtain an alternative strategy called analogously MAXMAD.

The resulting acquisition functions can be computed analytically.

Specifically, the variance is computed using Eq. 14.

A similar formula can be derived for MAD.

Finally, we propose a heuristic approach from BO (Snoek et al., 2012) to parallellise MAXV and MAXMAD strategies: The first point in the batch is chosen as in the sequential case.

The further points are iteratively selected as the locations where the expected variance (or MAD), taken with respect to the discrepancy values of the pending points, that is points that have been already chosen to the current batch, is highest.

The resulting acquisition functions are immediately obtained as the integrands of Eq. 4 and 15.

A.3.

Uncertainty quantification of the ABC posterior ( 1 one MCMC sampling from the instrumental density and scales as O(nt 2 ) so that n can be large.

Total cost is O((n +ñ)t 2 +ñ 2 (t + s) +ñ 3 ).

We briefly describe some key details of our algorithm and the experiments.

Locations for fitting the initial GP model are sampled from the uniform prior in all cases.

We take 10 initial points for 2D and 20 for 3D cases.

We use b = 0, B ij = 10 2 1 i=j and include basis functions of the form 1, θ i , θ 2 i .

The discrepancy ∆ θ is assumed smooth and we use the squared exponential covariance function

are given weakly informative priors and their values are obtained using MAP estimation at each iteration.

Owen's T function values are computed using a C-implementation of the algorithm by Patefield and Tandy (2000) .

For simplicity and to ensure meaningful comparisons to ground-truth, we fix ε to certain small predefined values although, in practice, its value is set adaptively (Järvenpää et al., 2019a) or based on pilot runs.

We compute the estimate of the unnormalised ABC posterior using the Eq. 11 for MAXV, EIV, RAND and Eq. 13 for MAXMAD, EIMAD.

Adaptive MCMC (Haario et al., 2006 ) is used to sample from the resulting ABC posterior estimates and from instrumental densities needed for IS approximations.

TV denotes the median total variation distance between the estimated ABC posterior and the true one (2D) or the average TV between their marginal TV values (3D) computed numerically over 50 repeated runs.

Iteration (i.e. number of batches chosen) serves as a proxy for wall-time.

The number of simulations i.e. the maximum value of t is fixed in all experiments and the batch methods thus finish earlier.

Mahalanobis distance was used as the discrepancy for Lorenz model.

The simulation model was run 500 times to estimate the covariance matrix of the six summary statistics by Hakkarainen et al. (2012) at the true parameter and the the inverse of the covariance matrix was used in the Mahalanobis distance.

Of course, such discrepancy is unavailable in practice because the true parameter is unknown and the computational budget limited.

However, as the main goal of this paper is to approximate any given ABC posterior with a limited simulation budget, we chose our target ABC posterior this way.

Gutmann and Corander (2016) defined a discrepancy for the bacterial infections model by summing four L 1 -distances computed between certain individual summaries.

For details, see example 7 in Gutmann and Corander (2016) .

We used the same discrepancy except that we further took square root of their discrepancy function.

We obtained a similar ABC posterior as the original article (Numminen et al., 2013) where ABC-PMC algorithm and slightly different approach for comparing the data sets were used.

show the 10 initial points and the black dots 100 additional points selected using each acquisition function (the last two batches in the second row are however highlighted by red plus-signs and crosses).

The TV value in the title shows the total variation distance between the true and estimated ABC posteriors for each particular case.

Fig. 5 and 6 show typical estimated ABC posterior densities of the Lorenz and bacterial infections models, respectively.

These results are shown to demonstrate the accuracy obtainable with very limited simulations.

These particular results were obtained with the sequential EIV method using 600 iterations corresponding to 610 simulations (Lorenz model) or 620 simulations (bacterial infections model).

Fig. 7 illustrates the ABC posterior uncertainty quantification for the bacterial infections model.

Sequential EIV method was used and one typical case is shown.

The results suggest that while the ABC posterior is well estimated at the last iteration, there is some uncertainty left about its exact shape.

@highlight

We propose principled batch Bayesian experimental design strategies and a method for uncertainty quantification of the posterior summaries in a Gaussian process surrogate-based approximate Bayesian computation framework.
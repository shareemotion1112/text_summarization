Estimating covariances between financial assets plays an important role in risk management.

In practice, when the sample size is small compared to the number of variables, the empirical estimate is known to be very unstable.

Here, we propose a novel covariance estimator based on the Gaussian Process Latent Variable Model (GP-LVM).

Our estimator can be considered as a non-linear extension of standard factor models with readily interpretable parameters reminiscent of market betas.

Furthermore, our Bayesian treatment naturally shrinks the sample covariance matrix towards a more structured matrix given by the prior and thereby systematically reduces estimation errors.

Finally, we discuss some financial applications of the GP-LVM model.

Many financial problems require the estimation of covariance matrices between given assets.

This may be useful to optimize one's portfolio, i.e.: maximize the portfolio returns w T r and/or minimize the volatility √ w T Kw.

Indeed, Markowitz received a Noble Price in economics for his treatment of modern portfolio theory BID9 .

In practice, estimating historical returns and high-dimensional covariance matrices is challenging and often times equally weighted portfolio outperforms the portfolio constructed from sample estimates BID6 .

The estimation of covariance matrices is especially hard, when the number of assets is large compared to the number of observations.

Sample estimations in those cases are very unstable or can even become singular.

To cope with this problem, a wide range of estimators, e.g. factor models such as the single-index model BID16 or shrinkage estimators BID8 , have been developed and employed in portfolio optimization.

With todays machine learning techniques we can even further improve those estimates.

Machine learning has already arrived in finance.

BID10 trained an agent via reinforcement learning to optimally execute trades.

BID4 forecast asset prices with neural networks and BID1 with Gaussian Processes.

Recently, BID5 made an ansatz to optimally allocate portfolios using deep autoencoders.

BID21 used Gaussian Processes to build volatility models and BID20 to estimate time varying covariance matrices.

Bayesian machine learning methods are used more and more in this domain.

The fact, that in Bayesian framework parameters are not treated as true values, but as random variables, accounts for estimation uncertainties and can even alleviate the unwanted impacts of outliers.

Furthermore, one can easily incorporate additional information and/or personal views by selecting suitable priors.

In this paper, we propose a Bayesian covariance estimator based on the Gaussian Process Latent Variable Model (GP-LVM) BID7 , which can be considered as a non-linear extension of standard factor models with readily interpretable parameters reminiscent of market betas.

Our Bayesian treatment naturally shrinks the sample covariance matrix (which maximizes the likelihood function) towards a more structured matrix given by the prior and thereby systematically reduces estimation errors.

We evaluated our model on the stocks of S&P500 and found significant improvements in terms of model fit compared to classical linear models.

Furthermore we suggest some financial applications, where Gaussian Processes can be used as well.

That includes portfolio allocation, price prediction for less frequently traded stocks and non-linear clustering of stocks into their sub-sectors.

In section 2 we begin with an introduction to the Bayesian non-parametric Gaussian Processes and discuss the associated requirements for learning.

Section 3 introduces the financial background needed for portfolio optimization and how to relate it to Gaussian Processes.

In section 4 we conduct experiments on covariance matrix estimations and discuss the results.

We conclude in section 5.

In this paper, we utilize a Bayesian non-parametric machine learning approach based on Gaussian Processes (GPs).

Combining those with latent variable models leads to Gaussian Process Latent Variable Models (GP-LVMs), that we use to estimate the covariance between different assets.

These approaches have been described in detail in BID7 BID12 .

We provide a brief review here.

Subsequently, we show, how to relate those machine learning approaches to the known models in finance, e.g. the single-index model BID16 .

A Gaussian Process (GP) is a generalization of the Gaussian distribution.

Using a GP, we can define a distribution over functions f (x), where x ∈ R Q and f ∈ R. Like a Gaussian distribution, the GP is specified by a mean and a covariance.

In the GP case, however, the mean is a function of the input variable m(x) and the covariance is a function of two variables k(x, x ), which contains information about how the GP evaluated at x and x covary DISPLAYFORM0 We write f ∼ GP(m(·), k(·, ·)).

Any finite collection of function values, at DISPLAYFORM1 where µ = (m(x 1 ), ..., m(x N )) T is the mean vector and K ∈ R N ×N is the Gram matrix with entries K ij = k(x i , x j ).

We refer to the covariance function as kernel function.

The properties of the function f (i.e. smoothness, periodicity) are determined by the choice of this kernel function.

For example, sampled functions from a GP with an exponentiated quadratic covariance function k SE (x, x ) = σ 2 exp(−0.5||x − x || 2 2 /l 2 ) smoothly vary with lengthscale l and are infinitely often differentiable.

Given a dataset D of N input points X = (x 1 , ..., x N ) T and N corresponding targets y = (y 1 , ..., y N ) T , the predictive distribution for a zero mean GP at N * new locations X * reads BID12 y DISPLAYFORM2 where DISPLAYFORM3 DISPLAYFORM4 K X * X ∈ R N * ×N is the covariance matrix between the GP evaluated at X * and X, K XX ∈ R N ×N is the covariance matrix of the GP evaluated at X. As we can see in equations FORMULA3 and FORMULA4 , the kernel function plays a very important role in the GP framework and will be important for our financial model as well.

Often times we are just given a data matrix Y ∈ R N ×D and the goal is to find a lower dimensional representation X ∈ R N ×Q , without losing too much information.

Principal component analysis (PCA) is one of the most used techniques for reducing the dimensions of the data, which has also been motivated as the maximum likelihood solution to a particular form of Gaussian Latent Variable Model BID17 .

PCA embeds Y via a linear mapping into the latent space X. BID7 introduced the Gaussian Process Latent Variable Model (GP-LVM) as a non-linear extension of probabilistic PCA.

The generative procedure takes the form DISPLAYFORM0 where DISPLAYFORM1 T is a group of D independent samples from a GP, i.e. f d ∼ GP(0, k(·, ·)).

By this, we assume the rows of Y to be jointly Gaussian distributed and the columns to be independent, i.e. each sample Y :,d ∼ N (Y :,d |0, K) where K = k(X, X) + σ 2 I and σ 2 denotes the variance of the random noise .

The marginal likelihood of Y becomes BID7 DISPLAYFORM2 The dependency on the latent positions X and the kernel hyperparameters is given through the kernel matrix K. As suggested by Lawrence FORMULA0 , we can optimize the log marginal likelihood log p(Y |X) with respect to the latent positions and the hyperparameters.

Optimization can easily lead to overfitting.

Therefore, a fully Bayesian treatment of the model would be preferable but is intractable.

BID18 introduced a variational inference framework, which not only handles the problem of overfitting but also allows to automatically select the dimensionality of the latent space.

Instead of optimizing equation FORMULA7 , we want to calculate the posterior using Bayes rule DISPLAYFORM0 , which is intractable.

The idea behind variational Bayes is to approximate the true posterior p(X|Y ) by another distribution q(X), selected from a tractable family.

The goal is to select the one distribution q(X), that is closest to the true posterior p(X|Y ) in some sense.

A natural choice to quantify the closeness is given by the Kullback-Leibler divergence (Cover & Thomas, 1991) DISPLAYFORM1 By definingp(X|Y ) = p(Y |X)p(X) as the unnormalized posterior, equation FORMULA9 becomes DISPLAYFORM2 (10) with the first term on the right hand side being known as the evidence lower bound (ELBO).

Equation (10) is the objective function we want to minimize with respect to q(X) to get a good approximation to the true posterior.

Note that on the left hand side only the ELBO is q dependent.

So, in order to minimize (10), we can just as well maximize the ELBO.

Because the Kullback-Leibler divergence is non-negative, the ELBO is a lower bound on the evidence log p(Y )1 .

Therefore, this procedure not only gives the best approximation to the posterior, but it also bounds the evidence, which serves as a measure of the goodness of our fit.

The number of latent dimensions Q can be chosen to be the one, which maximizes the ELBO.So, GP-LVM is an algorithm, which reduces the dimensions of our data-matrix Y ∈ R N ×D from D to Q in a non-linear way and at the same time estimates the covariance matrix between the N points.

The estimated covariance matrix can then be used for further analysis.

Now we have a procedure to estimate the covariance matrix between different datapoints.

This section discusses how we can relate this to financial models.

The Capital Asset Pricing Model (CAPM) describes the relationship between the expected returns of an asset r n ∈ R D for D days and its risk β n DISPLAYFORM0 where r f ∈ R D is the risk free return on D different days and r m is the market return on D different days.

The main idea behind CAPM is, that an investor needs to be compensated for the risk of his holdings.

For a risk free asset with β n = 0, the expected return E[r n ] is just the risk free rate r f .

If an asset is risky with a risk β n = 0, the expected return E[r n ] is increased by β n E[r m ], wherer m is the excess return of the marketr m = r m − r f .We can write down equation FORMULA11 in terms of the excess returnr = r − r f and get DISPLAYFORM1 wherer n is the excess return of a given asset andr m is the excess return of the market (also called a risk factor).

Arbitrage pricing theory BID13 generalizes the above model by allowing multiple risk factors F beside the marketr m .

In particular, it assumes that asset returns follow a factor structure r n = α n + F β n + n , (13) with n denoting some independent zero mean noise with variance σ 2 n .

Here, F ∈ R D×Q is the matrix of Q factor returns on D days and β n ∈ R Q is the loading of stock n to the Q factors.

Arbitrage pricing theory BID14 then shows that the expected excess returns adhere to DISPLAYFORM2 i.e. the CAPM is derived as special case when assuming a single risk factor (single-index model).To match the form of the GP-LVM (see equation FORMULA5 ), we rewrite equation FORMULA1 as DISPLAYFORM3 where r ∈ R N ×D is the return matrix 2 .

Note that assuming latent factors distributed as F ∼ N (0, 1) and marginalizing over them, equation FORMULA1 is a special case of equation FORMULA3 with f drawn from a GP mapping β n to r n with a linear kernel.

Interestingly, this provides an exact correspondence with factor models by considering the matrix of couplings B = (β 1 , ..., β N )T as the latent space positions 3 .

In this perspective the factor model can be seen as a linear dimensionality reduction, where we reduce the N ×D matrix r to a low rank matrix B of size N ×Q. By chosing a non-linear kernel k(·, ·) the GP-LVM formulation readily allows for non-linear dimensionality reductions.

Since, it is generally known, that different assets have different volatilities, we further generalize the model.

In particular, we assume the noise to be a zero mean Gaussian, but allow for different variances σ 2 n for different stocks.

For this reason, we also have to parameterize the kernel (covariance) matrix in a different way than usual.

Section 4.1 explains how to deal with that.

The model is then approximated using variational inference as described in section 2.3.

After inferring B and the hyperparameters of the kernel, we can calculate the covariance matrix K and use it for further analysis.

BID9 provided the foundation for modern portfolio theory, for which he received a Nobel Price in economics.

The method analyses how good a given portfolio is, based on the mean and the variance of the returns of the assets contained in the portfolio.

This can also be formulated as an optimization problem for selecting an optimal portfolio, given the covariance between the assets and the risk tolerance q of the investor.

Given the covariance matrix K ∈ R N ×N , we can calculate the optimal portfolio weights w by DISPLAYFORM0 wherer is the mean return vector.

Risk friendly investors have a higher q than risk averse investors.

The model is constrained by w = 1.

Sincer is very hard to estimate in general and we are primarily interested in the estimation of the covariance matrix K, we set q to zero and get DISPLAYFORM1 This portfolio is called the minimal risk portfolio, i.e. the solution to equation FORMULA5 provides the weights for the portfolio, which minimizes the overall risk, assuming the estimated K is the true covariance matrix.

In this section, we discuss the performance of the GP-LVM on financial data.

After describing the data collection and modeling procedure, we evaluate the model on the daily return series of the S&P500 stocks.

Subsequently, we discuss further financial applications.

In particular, we show how to build a minimal risk portfolio (this can easily be extended to maximizing returns as well), how to fill-in prices for assets which are not traded frequently and how to visualize sector relations between different stocks (latent space embedding).

For a given time period, we take all the stocks from the S&P500, whose daily close prices were available for the whole period 4 .

The data were downloaded from Yahoo Finance.

After having the close prices in a matrix p ∈ R N ×(D+1) , we calculate the return matrix r ∈ R N ×D , where r nd = (p n,d − p n,d−1 )/p n,d−1 .

r builds the basis of our analysis.

We can feed r into the GP-LVM.

The GP-LVM procedure, as described in section 2, assumes the likelihood to be Gaussian with the covariance given by the kernel function for each day and assumes independency over different days.

We use the following kernel functions DISPLAYFORM0 and the stationary kernels DISPLAYFORM1 where d ij = ||β i − β j || 2 is the Euclidean distance between β i and β j .

σ 2 is the kernel variance and l kernel lengthscale.

Note that since the diagonal elements of stationary kernels are the same, they are not well suited for an estimation of a covariance matrix between different financial assets.

Therefore, in the case of stationary kernel we decompose our covariance matrix K cov into a vector of coefficient scales σ and a correlation matrix K corr , such that K cov = ΣK corr Σ, where Σ is a diagonal matrix with σ on the diagonal.

The full kernel function k(·, ·) at the end is the sum of the noise kernel k noise and one of the other kernels.

In matrix form we get DISPLAYFORM2 where B = (β 1 , ..., β N ) T .

We chose the following priors DISPLAYFORM3 The prior on B determines how much space is allocated to the points in the latent space.

The volume of this space expands with higher latent space dimension, which make the model prone to overfitting.

To cope with that, we assign an inverse gamma prior to the lengthscale l and σ (σ in the linear kernel has a similar functionality as l in the stationary kernels).

It allows larger values for higher latent space dimension, thereby shrinking the effective latent space volume and exponentially suppresses very small values, which deters overfitting as well.

The parameters of the priors are chosen such that they allow for enough volume in the latent space for roughly 100-150 datapoints, which we use in our analysis.

If the number of points is drastically different, one should adjust the parameters accordingly.

The kernel standard deviations σ noise and σ are assigned a half Gaussian prior with variance 0.5, which is essentially a flat prior, since the returns are rarely above 0.1 for a day.

Model inference under these specifications (GP-LVM likelihood for the data and prior for B and all kernel hyperparameters σ, σ noise , l and σ, which we denote by θ) was carried out by variational inference as described in section 2.3.

To this end, we implemented all models in the probabilistic programming language Stan BID0 , which supports variational inference out of the box.

The source code is available on Github (link anonymized for review).

We tested different initializations for the parameter (random, PCA solution and Isomap solution), but there were no significant differences in the ELBO.

So, we started the inference 50 times with random initializations and took the result with highest ELBO for each kernel function and Q.

The GP-LVM can be evaluated in many different ways.

Since, it projects the data from a Ddimensional space to a Q-dimensional latent space, we can look at the reconstruction error.

A suitable measure of the reconstruction error is the R-squared (R 2 ) score, which is equal to one if there is no reconstruction error and decreases if the reconstruction error increases.

It is defined by DISPLAYFORM0 where y = (y 1 , ..., y N ) T are the true values, f = (f 1 , ...f N ) T are the predicted values andȳ is the mean of y. In the following, we look at the R 2 as a function of the latent dimension Q for different kernels.

FIG0 shows the results for three non-linear kernels and the linear kernel.

Only a single dimension in the non-linear case can already capture almost 50% of the structure, whereas the linear kernel is at 15%.

As one would expect, the higher Q, the more structure can be learned.

But at some point the model will also start learning the noise and overfit.

As introduced in section 2.3, the ELBO is a good measure to evaluate different models and already incorporates models complexity (overfitting).

FIG0 shows the ELBO as a function of the latent dimension Q. Here we see, that the model selects just a few latent dimensions.

Depending on the used kernel, latent dimensions from three to five are already enough to capture the structure.

If we increase the dimensions further, the ELBO starts dropping, which is a sign of overfitting.

As can be seen from FIG0 , we do not need to go to higher dimensions.

Q between two and five is already good enough and captures the structure that can be captured by the model.

The GP-LVM provides us the covariance matrix K and the latent space representation B of the data.

We can build a lot of nice applications upon that, some of which are discussed in this section.

After inferring B and θ, we can reconstruct the covariance matrix K using equation (20) .

Thereafter, we only need to minimize equation FORMULA5 , which provides the weights w for our portfolio in the future.

Minimization of FORMULA5 is done under the constraints: n w n = 1 and 0 < w n < 0.1, ∀n.

These constraints are commonly employed in practice and ensure that we are fully invested, take on long positions only and prohibit too much weight for a single asset.

For our tests, we proceed as follows: First, we get data for 60 randomly selected stocks from the S&P500 from Jan 2008 to Jan 2018.

Then, we learn w from the past year and buy accordingly for the next six months.

Starting from Jan 2008, this procedure is repeated every six months.

We calculate the average return, standard deviation and the Sharpe ratio.

BID15 suggested the Sharpe ratio as a measure for a portfolio's performance, which is the average return earned per unit of volatility and can be calculated by dividing the mean return of a series by its standard deviation.

TAB0 shows the performance of the portfolio for different kernels for Q = 3.

For the GP-LVM we chose the linear, SE, EXP and M32 kernels and included the performance given by the sample covariance matrix, i. DISPLAYFORM0 r nd , the shrunk Ledoit-Wolf covariance matrix 5 BID8 and the equally weighted portfolio, where w = (1, 1, ..., 1)/N .

Non-linear kernels have the minimal variance and at the same time the best Sharpe ratio values.

Note that we are building a minimal variance portfolio and therefore not maximizing the mean returns as explained in section 3.2.

For finite q in equation FORMULA4 one can also build portfolios, which not only minimize risk but also maximize the returns.

Another requirement for that would be to have a good estimator for the expected return as well.

Regulation requires fair value assessment of all assets (Financial Accounting Standards Board, 2006) , including illiquid and infrequently traded ones.

Here, we demonstrate how the GP-LVM could be used to fill-in missing prices, e.g. if no trading took place.

For illustration purposes, we continue working with our daily close prices of stocks from the S&P500, but the procedure can be applied to any other asset class and time resolution.

First, we split our data r into training and test set.

The test set contains days where the returns of assets are missing and our goal is to accurately predict the return.

In our case, we use stock data from Jan 2017 to Oct 2017 for training and Nov 2017 to Dez 2017 to test.

The latent space B and the hyperparameters θ are learned from the training set.

Given B and θ, we can use the standard GP equations (eq. FORMULA3 and FORMULA4 ) to get the posterior return distribution.

A suggestion for the value of the return at a particular day can be made by the mean of this distribution.

Given N stocks for a particular day d, we fit a GP to N − 1 stocks and predict the value of the remaining stock.

We repeat the process N times, each time leaving out a different stock (Leave-one-out cross-validation).

FIG1 shows the R 2 -score and the average absolute deviation of the suggested return to the real return.

The average is build over all stocks and all days in the test set.

The predictions with just the historical mean have a negative R 2 -score.

The linear model is better than that.

But if we switch to non-linear kernels, we can even further increase the prediction score.

For Q between 2 and 4 we obtain the best results.

Note that to make a decent suggestion for the return of an asset, there must be some correlated assets in the data as well.

Otherwise, the model has no information at all about the asset, we want to predict the returns for.

The 2-dimensional latent space can be visualized as a scatter plot.

For a stationary kernel function like the SE, the distance between the stocks is directly related to their correlation.

In this case, the latent positions are even easier to interpret than market betas.

As an example, FIG2 shows the 2-D latent space from 60 randomly selected stocks from the S&P500 from Jan 2017 to Dec 2017.

Visually stocks from the same sector tend to cluster together and we consider our method as an alternative to other methods for detecting structure in financial data BID19 .

We applied the Gaussian Process Latent Variable Model (GP-LVM) to estimate the covariance matrix between different assets, given their time series.

We then showed how the GP-LVM can be seen as a non-linear extension to the CAPM with latent factors.

Based on the R 2 -score and the ELBO, we concluded, that for fixed latent space dimension Q, every non-linear kernel can capture more structure than the linear one.

The estimated covariance matrix helps us to build a minimal risk portfolio according to Markowitz Portfolio theory.

We evaluated the performance of different models on the S&P500 from year 2008 to 2018.

Again, non-linear kernels had lower risk in the suggested portfolio and higher Sharpe ratios than the linear kernel and the baseline measures.

Furthermore, we showed how to use the GP-LVM to fill in missing prices of less frequently traded assets and we discussed the role of the latent positions of the assets.

In the future, one could also put a Gaussian Process on the latent positions and allow them to vary in time, which would lead to a time-dependent covariance matrix.

The authors thank Dr. h.c.

Maucher for funding their positions.

<|TLDR|>

@highlight

Covariance matrix estimation of financial assets with Gaussian Process Latent Variable Models

@highlight

Illustrates how the Gaussian Process Latent Variable Model (GP-LVM) can replace classical linear factor models for the estimation of covariance matrices in portfolio optimization problems.

@highlight

This paper uses standard GPLVMs to model the covariance structure and a latent space representation of S&P500 financial time series, to optimize portfolios and predict missing values.

@highlight

This paper proposes to use a GPLVM to model financial returns
We study the following three fundamental problems about ridge regression: (1) what is the structure of the estimator?

(2) how to correctly use cross-validation to choose the regularization parameter?

and (3) how to accelerate computation without losing too much accuracy?

We consider the three problems in a unified large-data linear model.

We give a precise representation of ridge regression as a covariance matrix-dependent linear combination of the true parameter and the noise.

We study the bias of $K$-fold cross-validation for choosing the regularization parameter, and propose a simple bias-correction.

We analyze the accuracy of primal and dual sketching for ridge regression, showing they are surprisingly accurate.

Our results are illustrated by simulations and by analyzing empirical data.

Ridge or 2 -regularized regression is a widely used method for prediction and estimation when the data dimension p is large compared to the number of datapoints n. This is especially so in problems with many good features, where sparsity assumptions may not be justified.

A great deal is known about ridge regression.

It is Bayes optimal for any quadratic loss in a Bayesian linear model where the parameters and noise are Gaussian.

The asymptotic properties of ridge have been widely studied (e.g., Tulino & Verdú, 2004; Serdobolskii, 2007; Couillet & Debbah, 2011; Dicker, 2016; Dobriban & Wager, 2018, etc) .

For choosing the regularization parameter in practice, cross-validation (CV) is widely used.

In addition, there is an exact shortcut (e.g., Hastie et al., 2009, p. 243) , which has good consistency properties (Hastie et al., 2019) .

There is also a lot of work on fast approximate algorithms for ridge, e.g., using sketching methods (e.g., el Alaoui & Mahoney, 2015; Chen et al., 2015; Wang et al., 2018; Chowdhury et al., 2018, among

Here we seek to develop a deeper understanding of ridge regression, going beyond existing work in multiple aspects.

We work in linear models under a popular asymptotic regime where n, p → ∞ at the same rate (Marchenko & Pastur, 1967; Serdobolskii, 2007; Couillet & Debbah, 2011; Yao et al., 2015) .

In this framework, we develop a fundamental representation for ridge regression, which shows that it is well approximated by a linear scaling of the true parameters perturbed by noise.

The scaling matrices are functions of the population-level covariance of the features.

As a consequence, we derive formulas for the training error and bias-variance tradeoff of ridge.

Second, we study commonly used methods for choosing the regularization parameter.

Inspired by the observation that CV has a bias for estimating the error rate (e.g., Hastie et al., 2009, p. 243) , we study the bias of CV for selecting the regularization parameter.

We discover a surprisingly simple form for the bias, and propose a downward scaling bias correction procedure.

Third, we study the accuracy loss of a class of randomized sketching algorithms for ridge regression.

These algorithms approximate the sample covariance matrix by sketching or random projection.

We show they can be surprisingly accurate, e.g., they can sometimes cut computational cost in half, only incurring 5% extra error.

Even more, they can sometimes improve the MSE if a suboptimal regularization parameter is originally used.

Our work leverages recent results from asymptotic random matrix theory and free probability theory.

One challenge in our analysis is to find the limit of the trace tr (Σ 1 + Σ −1

−1 /p, where Σ 1 and Σ 2 are p × p independent sample covariance matrices of Gaussian random vectors.

The calculation requires nontrivial aspects of freely additive convolutions (e.g., Voiculescu et al., 1992; Nica & Speicher, 2006) .

Our work is connected to prior works on ridge regression in high-dimensional statistics (Serdobolskii, 2007) and wireless communications (Tulino & Verdú, 2004; Couillet & Debbah, 2011) .

Among other related works, El Karoui & Kösters (2011) discuss the implications of the geometric sensitivity of random matrix theory for ridge regression, without considering our problems.

El Karoui (2018) and Dicker (2016) study ridge regression estimators, but focus only on the risk for identity covariance.

Hastie et al. (2019) study "ridgeless" regression, where the regularization parameter tends to zero.

Sketching is an increasingly popular research topic, see Vempala (2005) ; Halko et al. (2011); Mahoney (2011); Woodruff (2014) ; Drineas & Mahoney (2017) and references therein.

For sketched ridge regression, Zhang et al. (2013a; b) study the dual problem in a complementary finite-sample setting, and their results are hard to compare.

Chen et al. (2015) propose an algorithm combining sparse embedding and the subsampled randomized Hadamard transform (SRHT), proving relative approximation bounds.

Wang et al. (2017) study iterative sketching algorithms from an optimization point of view, for both the primal and the dual problems.

Dobriban & Liu (2018) study sketching using asymptotic random matrix theory, but only for unregularized linear regression.

Chowdhury et al. (2018) propose a data-dependent algorithm in light of the ridge leverage scores.

Other related works include Sarlos (2006) The structure of the paper is as follows: We state our results on representation, risk, and biasvariance tradeoff in Section 2.

We study the bias of cross-validation for choosing the regularization parameter in Section 3.

We study the accuracy of randomized primal and dual sketching for both orthogonal and Gaussian sketches in Section 4.

We provide proofs and additional simulations in the Appendix.

Code reproducing the experiments in the paper are available at https://github.

com/liusf15/RidgeRegression.

We work in the usual linear regression model Y = Xβ + ε, where each row x i of X ∈ R n×p is a datapoint in p dimensions, and so there are p features.

The corresponding element y i of Y ∈ R n is its continous response (or outcome).

We assume mean zero uncorrelated noise, so Eε = 0, and Cov [ε] = σ 2 I n .

We estimate the coefficient β ∈ R p by ridge regression, solving the optimization problemβ = arg min

where λ > 0 is a regularization parameter.

The solution has the closed form

We work in a "big data" asymptotic limit, where both the dimension p and the sample size n tend to infinity, and their aspect ratio converges to a constant, p/n → γ ∈ (0, ∞).

Our results can be interpreted for any n and p, using γ = p/n as an approximation.

We recall that the empirical spectral distribution (ESD) of a p×p symmetric matrix Σ is the distribution

δ λi where λ i , i = 1, . . . , p are the eigenvalues of Σ, and δ x is the point mass at x. This is a standard notion in random matrix theory, see e.g., Marchenko & Pastur (1967); Tulino & Verdú (2004) ; Couillet & Debbah (2011); Yao et al. (2015) .

The ESD is a convenient tool to summarize all information obtainable from the eigenvalues of a matrix.

For instance, the trace of Σ is proportional to the mean of the distribution, while the condition number is related to the range of the support.

As is common, we will work in models where there is a sequence of covariance matrices Σ = Σ p , and their ESDs converges in distribution to a limiting probability distribution.

The results become simpler, because they depend only on the limit.

By extension, we say that the ESD of the n × p matrix X is the ESD of X X/n.

We will consider some very specific models for the data, assuming it is of the form X = U Σ 1/2 , where U has iid entries of zero mean and unit variance.

This means that the datapoints, i.e., the rows of X, have the form x i = Σ 1/2 u i , i = 1, . . .

, p, where u i have iid entries.

Then Σ is the "true" covariance matrix of the features, which is typically not observed.

These types of models for the data are very common in random matrix theory, see the references mentioned above.

Under these models, it is possible to characterize precisely the deviations between the empirical covariance matrix Σ = n −1 X X and the population covariance matrix Σ, dating back to the well known classical Marchenko-Pastur law for eigenvectors (Marchenko & Pastur, 1967) , extended to more general models and made more precise, including results for eigenvectors (see e.g., Tulino & Verdú, 2004; Couillet & Debbah, 2011; Yao et al., 2015, and references therein) .

This has been used to study methods for estimating the true covariance matrix, with several applications (e.g., Paul & Aue, 2014; Bun et al., 2017) .

More recently, such models have been used to study high dimensional statistical learning problems, including classification and regression (e.g., Zollanvari & Genton, 2013; Dobriban & Wager, 2018) .

Our work falls in this line.

We start by finding a precise representation of the ridge estimator.

For random vectors u n , v n of growing dimension, we say u n and v n are deterministic equivalents, if for any sequence of fixed (or random and independent of u n , v n ) vectors w n such that lim sup w n 2 < ∞ almost surely, we have |w n (u n − v n )| → 0 almost surely.

We denote this by u n v n .

Thus linear combinations of u n are well approximated by those of v n .

This is a somewhat non-standard definition, but it turns out that it is precisely the one we need to use prior results from random matrix theory such as from (Rubio & Mestre, 2011) .

We extend scalar functions f : R → R to matrices in the usual way by functional calculus, applying them to the eigenvalues and keeping the eigenvectors.

If M = V ΛV is a spectral decomposition of M , then we define f (M ) := V f (Λ)V , where f (Λ) is the diagonal matrix with entries f (Λ ii ).

For a fixed design matrix X, we can write the estimator aŝ

However, for a random design, we can find a representation that depends on the true covariance Σ, which may be simpler when Σ is simple, e.g., when Σ = I p is isotropic.

Theorem 2.1 (Representation of ridge estimator).

Suppose the data matrix has the form X = U Σ 1/2 , where U ∈ R n×p has iid entries of zero mean, unit variance and finite 8 + c-th moment for some c > 0, and Σ = Σ p ∈ R p×p is a deterministic positive definite matrix.

Suppose that n, p → ∞ with p/n → γ > 0.

Suppose the ESD of the sequence of Σs converges in distribution to a probability measure with compact support bounded away from the origin.

Suppose that the noise is Gaussian, and that β = β p is an arbitrary sequence of deterministic vectors, such that lim sup β 2 < ∞.

Then the ridge regression estimator is asymptotically equivalent to a random vector with the following representation:β

Here Z ∼ N (0, I p ) is a random vector that is stochastically dependent only on the noise ε, and A, B are deterministic matrices defined by applying the scalar functions below to Σ:

Here c p := c(n, p, Σ, λ) is the unique positive solution of the fixed point equation

This result gives a precise representation of the ridge regression estimator.

It is a sum of two terms: the true coefficient vector β scaled by the matrix A(Σ, λ), and the noise vector Z scaled by the matrix B(Σ, λ).

The first term captures to what extent ridge regression recovers the "signal".

Morever, the noise term Z is directly coupled with the noise in the original regression problem, and thus also the estimator.

The result would not hold for an independent noise vector Z.

However, the coefficients are not fully explicit, as they depend on the unknown population covariance matrix Σ, as well as on the fixed-point variable c p .

Some comments are in order:

1.

Structure of the proof.

The proof is quite non-elementary and relies on random matrix theory.

Specifically, it uses the language of the recently developed "calculus of deterministic equivalents" (Dobriban & Sheng, 2018) , and results by (Rubio & Mestre, 2011) .

A general takeaway is that for n not much larger than p, the empirical covariance matrix Σ is not a good estimator of the true covariance matrix Σ. However, the deviation of linear functionals of Σ, can be quantified.

In particular, we have

in the sense that linear combinations of the entries of the two matrices are close (see the proof for more details).

2.

Understanding the resolvent bias factor c p .

Thus, c p can be viewed as a resolvent bias factor, which tells us by what factor Σ is multiplied when evaluating the resolvent ( Σ + λI) −1 , and comparing it to its naive counterpart (Σ + λI) −1 .

It is known that c p is well defined, and this follows by a simple monotonicity argument, see Hachem et al. (2007) ; Rubio & Mestre (2011) .

Specifically, the left hand side of (2) is decreasing in c p , while the right hand size is increasing in Also c p is the derivative of c p , when viewing it as a function of z := −λ.

An explicit expression is provided in the proof in Section A.1, but is not crucial right now.

Here we discuss some implications of this representation.

For uncorrelated features, Σ = I p , A, B reduce to multiplication by scalars.

Hence, each coordinate of the ridge regression estimator is simply a scalar multiple of the corresponding coordinate of β.

One can use this to find the bias in each individual coordinate.

Training error and optimal regularization parameter.

This theorem has implications for understanding the training error, and optimal regularization parameter of ridge regression.

As it stands, the theorem itself only characterizes the behavior og linear combinations of the coordinates of the estimator.

Thus, it can be directly applied to study the bias Eβ(λ) − β of the estimator.

However, it cannot directly be used to study the variance; as that would require understanding quadratic functionals of the estimator.

This seems to require significant advances in random matrix theory, going beyond the results of Rubio & Mestre (2011) .

However, we show below that with additional assumptions on the structure of the parameter β, we can derive the MSE of the estimator in other ways.

We work in a random-effects model, where the p-dimensional regression parameter β is random, each coefficient has zero mean Eβ i = 0, and is normalized so that Varβ i = α 2 /p.

This ensures that the signal strength E β 2 = α 2 is fixed for any p.

The asymptotically optimal λ in this setting is always λ * = γσ 2 /α 2 see e.g., Tulino & Verdú (2004) ; Dicker (2016); Dobriban & Wager (2018) .

The ridge regression estimator with λ = pσ 2 /(nα 2 ) is the posterior mean of β, when β and ε are normal random variables.

For a distribution F , we define the quantities

(i = 1, 2, . . .).

These are the moments of the resolvent and its derivatives (up to constants).

We use the following loss functions: mean squared estimation error: M (β) = E β − β 2 2 , and residual or training error:

Theorem 2.2 (MSE and training error of ridge).

Suppose β has iid entries with

. .

, p and β is independent of X and ε.

Suppose X is an arbitrary n × p matrix depending on n and p, and the ESD of X converges weakly to a deterministic distribution F as n, p → ∞ and p/n → γ.

Then the asymptotic MSE and residual error of the ridge regression estimatorβ(λ) has the form

Bias-variance tradeoff.

Building on this, we can also study the bias-variance tradeoff of ridge regression.

Qualitatively, large λ leads to more regularization, and decreases the variance.

However, it also increases the bias.

Our theory allows us to find the explicit formulas for the bias and variance as a function of λ.

See Figure 1 for a plot and Sec. A.3 for the details.

As far as we know, this is one of the few examples of high-dimensional asymptotic problems where the precise form of the bias and variance can be evaluated.

Bias-variance tradeoff at optimal λ * = γσ 2 /α 2 . (see Figure 6 ) This can be viewed as the "pure" effect of dimensionality on the problem, keeping all other parameters fixed, and has intriguing properties.

The variance first increases, then decreases with γ.

In the "classical" low-dimensional case, most of the risk is due to variance, while in the "modern" high-dimensional case, most of it is due to bias.

This is consistent with other phenomena in proportional-limit asymptotics, e.g., that the map between population and sample eigenvalue distributions is asymptotically deterministic (Marchenko & Pastur, 1967) .

Future applications.

This fundamental representation may have applications to important statistical inference questions.

For instance, inference on the regression coefficient β and the noise variance σ 2 are important and challenging problems.

Can we use our representation to develop debiasing techniques for this task?

This will be interesting to explore in future work.

How can we choose the regularization parameter?

In practice, cross-validation (CV) is the most popular approach.

However, it is well known that CV has a bias for estimating the error rate, because it uses a smaller number of samples than the full data size (e.g., Hastie et al., 2009, p. 243) .

In this section, we study related questions, proposing a bias-correction method for the optimal regularization parameter.

This is closely connected to the previous section, because it relies on the same random-effects theoretical framework.

In fact, our conclusions here are a direct consequence of the properties of that framework.

Setup.

Suppose we split the n datapoints (samples) into K equal-sized subsets, each containing n 0 = n/K samples.

We use the k-th subset (X k , Y k ) as the validation set and the other K − 1 subsets (X −k , Y −k ), with total sample size n 1 = (K − 1)n/K as the training set.

We find the ridge For the error bar, we take n = 1000, p = 90, K = 5, and average over 90 different sub-datasets.

For the test error, we train on 1000 training datapoints and fit on 9000 test datapoints.

The debiased λ reduces the test error by 0.00024, and the minimal test error is 0.8480.

Right: Cross-validation on the flights dataset Wickham (2018) .

For the error bar, we take n = 300, p = 21, K = 5, and average over 180 different sub-datasets.

For the test error, we train on 300 datapoints and fit on 27000 test datapoints.

The debiased λ reduces the test error by 0.0022, and the minimal test error is 0.1353.

The expected cross-validation error is, for isotropic covariance, i.e., Σ = I,

Bias in CV.

When n, p tend to infinity so that p/n → γ > 0, and in the random effects model with Eβ i = 0, Varβ i = α 2 /p described above, the minimizer of CV (λ) tends to λ * k =γσ 2 /α 2 , wherẽ γ is the limiting aspect ratio of X −k , i.e.γ = γK/(K − 1).

Since the aspect ratios of X −k and X differ, the limiting minimizer of the cross-validation estimator of the test error is biased for the limiting minimizer of the actual test error, which is λ * = γσ 2 /α 2 .

Bias-correction.

Suppose we have foundλ * k , the minimizer of CV (λ).

Afterwards, we usually refit ridge regression on the entire dataset, i.e., find

Based on our bias calculation, we propose to use a bias-corrected parameter

So if we use 5 folds, we should multiply the CV-optimal λ by 0.8.

We find it surprising that this theoretically justified bias-correction does not depend on any unknown parameters, such as β, α 2 , σ 2 .While the bias of CV is widely known, we are not aware that this bias-correction for the regularization parameter has been proposed before.

Numerical examples.

Figure 2 shows on two empirical data examples that the debiased estimator gets closer to the optimal λ than the original minimizer of the CV.

However, in this case it does not significantly improve the test error.

Simulation results in Section A.4 also show that the bias-correction correctly shrinks the regularization parameter and decreases the test error.

We also consider examples where p n (i.e., γ 1), because this is a setting where it is known that the bias of CV can be large (Tibshirani & Tibshirani, 2009 ).

However, in this case, we do not see a significant improvement.

Extensions.

The same bias-correction idea also applies to train-test validation.

In addition, there is a special fast "short-cut" for leave-one-out cross-validation in ridge regression (e.g., Hastie et al., 2009, p. 243) , which has the same cost as one ridge regression.

The minimizer converges to λ * (Hastie et al., 2019) .

However, we think that the bias-correction idea is still valuable, as the idea applies beyond ridge regression: CV selects regularization parameters that are too large.

See Section A.5 for more details and experiments comparing different ways of choosing the regularization parameter.

A final important question about ridge regression is how to compute it in practice.

In this section, we study that problem in the same high-dimensional model used throughout our paper.

The computation complexity of ridge regression, O(np min(n, p)), can be intractable in modern large-scale data analysis.

Sketching is a popular approach to reducing the time complexity by reducing the sample size and/or dimension, usually by random projection or sampling (e.g. Mahoney, 2011; Woodruff, 2014; Drineas & Mahoney, 2016) .

Specifically, primal sketching approximates the sample covariance matrix X X/n by X L LX/n, where L is an m × n sketching matrix, and m < n. If L is chosen as a suitable random matrix, then this can still approximate the original sample covariance matrix.

Then the primal sketched ridge regression estimator iŝ

Dual sketching reduces p instead.

An equivalent expression for ridge regression isβ = n −1 X XX /n + λI n −1 Y .

Dual sketched ridge regression reduces the computation cost of the Gram matrix XX , approximating it by XRR X for another sketching matrix

The sketching matrices R and L are usually chosen as random matrices with iid entries (e.g., Gaussian ones) or as orthogonal matrices.

In this section, we study the asymptotic MSE for both orthogonal (Section 4.1) and Gaussian sketching (Section 4.2).

We also mention full sketching, which performs ridge after projecting down both X and Y .

In section A.11, we find its MSE.

However, the other two methods have better tradeoffs, and we can empirically get better results for the same computational cost.

First we consider primal sketching with orthogonal projections.

These can be implemented by subsampling, Haar distributed matrices, or subsampled randomized Hadamard transforms (Sarlos, 2006) .

We recall that the standard Marchenko-Pastur (MP) law is the probability distribution which is the limit of the ESD of X X/n, when the n × p matrix X has iid standard Gaussian entries, and n, p → ∞ so that p/n → γ > 0, which has an explicit density (Marchenko & Pastur, 1967; Bai & Silverstein, 2010) .

Theorem 4.1 (Primal orthogonal sketching).

Suppose β has iid entries with

. .

, p and β is independent of X and ε.

Suppose X has iid standard normal entries.

We compute primal sketched ridge regression (5) with an m × n orthogonal matrix L (m < n, LL = I m ).

Let n, p and m tend to infinity with p/n → γ ∈ (0, ∞) and m/n → ξ ∈ (0, 1).

Then the MSE ofβ p (λ) has the limit

where θ i (γ, λ) = (x + λ) −i dF γ (x) and F γ is the standard Marchenko-Pastur law with aspect ratio γ.

Structure of the proof.

The proof is in Section A.6, with explicit formulas in Section A.6.1.

The θ i are related to the resolvent of the MP law and its derivatives.

In the proof, we decompose the MSE as the sum of variance and squared bias, both of which further reduce to the traces of certain random matrices, whose limits are determined by the MP law F γ and λ.

The two terms on the RHS of Equation (7) are the limits of squared bias and variance, respectively.

There is an additional key step in the proof, which introduces the orthogonal complement L 1 of the matrix L such that L L + L 1 L 1 = I n , which leads to some Gaussian random variables appearing in the proof, and simplifies calculations.

Figure 3 (left) shows a good match with our theory.

It also shows that sketching does not increase the MSE too much.

In this case, by reducing the sample size to half the original one, we only increase the MSE by a factor of 1.05.

This shows sketching can be very effective.

We also see in Figure 3 (right) that variance is compromised much more than bias.

Robustness to tuning parameter.

The reader may wonder how strongly this depends on the choice of the regularization parameter λ.

Perhaps ridge regression works poorly with this λ, so sketching cannot worsen it too much?

What happens if we take the optimal λ instead of a fixed one?

In experiments in Section A.12 we show that the behavior is quite robust to the choice of regularization parameter.

The next theorem states a result for dual orthogonal sketching.

Let n, p and d go to infinity with p/n →

γ ∈ (0, ∞) and d/n → ζ ∈ (0, γ).

Then the MSE ofβ d (λ) has the limit

, and F ζ is the standard Marchenko-Pastur law.

Proof structure and simulations.

The proof in Section A.7 follows similar path to the previous one.

Hereθ i comes in because of the companion Stieltjes transform of MP law.

The simulation results shown in Figure 11 agrees well with our theory.

They are similar to the ones before: sketching has favorable properties, and the bias increases less than the variance.

Optimal tuning parameters.

For both primal and dual sketching, the optimal regularization parameter minimizing the MSE seems analytically intractable.

Instead, we use a numerical approach in our experiments, based on a binary search.

Since this is one-dimensional problem, there are no numerical issues.

See Figure 13 in Section A.12.3.

It is of special interest to investigate extreme projections, where the sketching dimension is much reduced compared to the sample size, so m n. This corresponds to ξ = 0.

This can also be viewed as a scaled marginal regression estimator, i.e.,β ∝ X Y .

For dual sketching, the same case can be recovered with ζ = 0.

Another interest of studying this special case is that the formula for MSE simplifies a lot.

Moreover, the optimal λ * that minimizes this equals γσ 2 /α 2 + 1 + γ and the optimal MSE is M (λ

The proof is in Section A.8.

When is the optimal MSE of marginal regression small?

Compared to the MSE of the zero estimator α 2 , it is small when γ(σ 2 /α 2 + 1) + 1 is large.

In Figure 4 (left), we compare marginal and ridge regression for different aspect ratios and SNR.

When the signal to noise ratio (SNR) α 2 /σ 2 is small or the aspect ratio γ is large, marginal regression does not increase the MSE much.

As a concrete example, if we take α 2 = σ 2 = 1 and γ = 0.7, the marginal MSE is 1 − 1/2.4 ≈ 0.58.

The optimal ridge MSE is about 0.52, so their ratio is only ca.

0.58/0.52 ≈ 1.1.

It seems quite surprising that a simple-minded method like marginal regression can work so well.

However, the reason is that when the SNR is small, we cannot expect ridge regression to have good performance.

Large γ can also be interpreted as small SNR, where ridge regression works poorly and sketching does not harm performance too much.

In this section, we study Gaussian sketching.

The following theorem states the bias of dual Gaussian sketching.

The bias is enough to characterize the performance in the high SNR regime where α/σ → ∞, and we discuss the extension to low SNR after the proof.

About the proof.

The proof is in Section A.9.We mention that the same result holds when the matrices involved have iid non-Gaussian entries, but the proof is more technical.

The current proof is based on free probability theory (e.g., Voiculescu et al., 1992; Hiai & Petz, 2006; Couillet & Debbah, 2011) .

The function m is the Stieltjes transform of the free additive convolution of a standard MP law F 1/ξ and a scaled inverse MP law λ/γ · F −1 1/γ (see the proof).

Numerics.

To evaluate the formula, we note that m −1 (m(0)) = 0, so m(0) is a root of m −1 .

Also, dm(0)/dz equals 1/(dm −1 (y)/dy| y=m(0) ), the reciprocal of the derivative of m −1 evaluated at m(0).

We use binary search to find the numerical solution.

The theoretical result agrees with the simulation quite well, see Figure 4 .

Somewhat unexpectedly, the MSE of dual sketching can be below the MSE of ridge regression, see Figure 4 .

This can happen when the original regularization parameter is suboptimal.

As d grows, the MSE of Gaussian dual sketching converges to that of ridge regression.

We have also found the bias of primal Gaussian sketching.

However, stating the result requires free probability theory, and so we present it in the Appendix, see Theorem A.1.

To further validate our results, we present additional simulations in Sec. A.12, for both fixed and optimal regularization parameters after sketching.

A detailed study of the computational cost for sketching in Sec. A.13 concludes, as expected, that primal sketching can reduce cost when p < n, while dual sketching can reduce it when p > n; and also provides a more detailed analysis.

where c p := c(n, p, Σ, λ) is the unique positive solution of the fixed point equation

Here, using the terminology of the calculus of deterministic equivalents (Dobriban & Sheng, 2018) , two sequences of (not necessarily symmetric) n × n matrices A n , B n of growing dimensions are equivalent, and we write A n B n if lim n→∞ tr [C n (A n − B n )] = 0 almost surely, for any sequence C n of (not necessarily symmetric) n × n deterministic matrices with bounded trace norm, i.e., such that lim sup C n tr < ∞ (Dobriban & Sheng, 2018) .

Informally, linear combinations of the entries of A n can be approximated by the entries of B n .

We start withβ

Then, by the general MP law written in the language of the calculus of deterministic equivalents

By the definition of equivalence for vectors,

We note a subtle point here.

The rank of the matrix M := ( Σ + λI p ) −1 Σ is at most n, and so it is not a full rank matrix when n < p.

In contrast, c p Σ(c p Σ + λI) −1 can be a full rank matrix.

Therefore, for the vectors β in the null space of Σ, which is also the null space of X, we certainly have that the two sides are not equal.

However, here we assumed that the matrix X is random, and so its null space is a random max(p − n, 0) dimensional linear space.

Therefore, for any fixed vector β, the random matrix M will not contain it in its null space with high probability, and so there is no contradiction.

We should also derive an asymptotic equivalent for

Figure 5: Simulation for ridge regression.

We take n = 1000, λ = 0.3.

Also, X has iid N (0, 1) entries,

, with α = 3, σ = 1.

The standard deviations are over 50 repetitions.

The theoretical lines are plotted according to Theorem 2.2.

The MSE is normalized by the norm of β.

Suppose we have Gaussian noise, and let Z ∼ N (0, I p ).

Then we can write

So the question reduces to finding a deterministic equivalent for h( Σ), where h(

By the calculus of determinstic equivalents:

(c p Σ + λI) −1 .

Moreover, fortunately the limit of the second part was recently calculated in (Dobriban & Sheng, 2019) .

This used the so-called "differentiation rule" of the calculus of deterministic equivalents to find

The derivative c p = dc p /dz has been found in Dobriban & Sheng (2019) , in the proof of Theorem 3.1, part 2b.

The result is (with γ p = p/n, H p the spectral distribution of Σ, and T a random variable distributed according to H p )

So, we find the final answer

A.2 RISK ANALYSIS Figure 5 shows a simulation result.

We see a good match between theory and simulation.

A.2.1 PROOF OF THEOREM 2.2

Proof.

The MSE ofβ has the form

where

We assume that X has iid entries of zero mean and unit variance, and that Eβ = 0, Var [β] = α 2 /pI p .

As p/n → γ as n goes to infinity, the ESD of 1 n X X converges to the MP law F γ .

So we have

For the standard Marchenko-Pastur law (i.e., when Σ = I p ), we have the explicit forms of θ 1 and θ 2 .

Specifically,

where

It is known that the limiting Stieltjes transform m Fγ := m γ of Σ has the explicit form (Marchenko & Pastur, 1967) :

As usual in the area, we use the principal branch of the square root of complex numbers.

Hence

.

Also

For the residual,

Next,

The limiting MSE decomposes into a limiting squared bias and variance.

The specific forms of these are

See Figure 1 for a plot.

We can make several observations.

1.

The bias increases with λ, starting out at zero for λ = 0 (linear regression), and increasing to α 2 as λ → ∞ (zero estimator).

2.

The variance decreases with λ, from γσ 2 x −1 dF γ (x) to zero.

3.

In the setting plotted in the figure, when α 2 and σ 2 are roughly comparable, there are additional qualitative properties we can investigate.

When γ is small, the regularization parameter λ influences the bias more strongly than the variance (i.e., the derivative of the normalized quantities in the range plotted is generally larger for the normalized squared bias).

In contrast when γ is large, the variance is influenced more.

Next we consider how bias and variance change with γ at the optimal λ * = γσ 2 /α 2 .

This can be viewed as the "pure" effects of dimensionality on the problem, keeping all other parameters fixed.

Ineed, α 2 /σ 2 can be viewed as the signal-to-noise ratio (SNR), and is fixed.

This analysis allows us to study for the best possible estimator (ridge regression, a Bayes estimator), behaves with the dimension.

We refer to Figure 6 , where we make some specific choices of α and σ.

1.

Clearly the overall risk increases, as the problem becomes harder with increasing dimension.

This is in line with our intuition.

2.

The classical bias-variance tradeoff can be summarized by the equation

where we made explicit the dependence of the bias and variance on λ, and where M * (α, γ) is the minimum MSE achievable, also known as the Bayes error, for which there are explicit formulas available (Tulino & Verdú, 2004; Dobriban & Wager, 2018) .

3.

The variance first increases, then decreases with γ.

This shows that in the "classical" low-dimensional case, most of the risk is due to variance, while in the "modern" highdimensional case, most of it is due to bias.

This observation is consistent with other phenomena in proportional-limit asymptotics, for instance that the map between population and sample eigenvalue distributions is asymptotically deterministic (Marchenko & Pastur, 1967; Bai & Silverstein, 2010) .

See Figure 7 .

We consider both small and large γ.

Our bias-correction procedure shrinks the λ to the correct direction and decreases the test error.

It is also shown that the one-standard-error rule (e.g., Hastie et al., 2009) does not perform well here.

: Left: we generate a training set (n = 1000, p = 700, γ = 0.7, α = σ = 1) and a test set (n test = 500) from the same distribution.

We split the training set into K = 5 equally sized folds and do cross-validation.

The blue error bars plot the mean and standard error of the K test errors.

The red dotted line indicates the "one-standard-error" location.

The green dashed line indicates the optimal λ

Another possible prediction method is to use the average of the ridge estimators computed during cross-validation.

Here it is also natural to use the CV-optimal regularization parameters, averaginĝ

This has the advantage that it does not require refitting the ridge regression estimator, and also that we use the optimal regularization parameter.

The same bias in the regularization parameter also applies to train-test validation.

Since the number of samples is changed when restricting to the training set, the optimal λ chosen by train-test validation is also biased for the true regularization parameter minimizing the test error.

We will later see in simulations ( Figure 8 ) that retraining the ridge regression estimator on the whole data will still significantly improve the performance (this is expected based on our results on CV).

For prediction, here we can also use ridge regression on the training set.

This effectively reduces sample size n → n train , where n train is the sample size of the training set.

However, if the training set grows such that n/n train → 1 while n train → ∞, the train-test split has asymptotically optimal performance.

There is a special "short-cut" for leave-one-out in ridge regression, which saves us from burdensome computation.

Write loo(λ) for the leave-one-out estimator of prediction error with parameter λ.

Instead of doing ridge regression n times, we can calculate the error explicitly as

where S(λ) = X(X X + nλI) −1 X .

The minimizer of loo(λ) is asymptotically optimal, i.e., it converges to λ * (Hastie et al., 2019) .

However, the computational cost of this shortcut is the same as that of a train-test split.

Therefore, the method described above has the same asymptotic performance.

Figure 8 : Comparing different ways of doing cross-validation.

We take n = 500, p = 550, α = 20, σ = 1, K = 5.

As for train-test validation, we take 80% of samples to be training set and the rest 20% be test set.

The error bars are the mean and standard deviation over 20 repetitions.

Simulations: Figure 8 shows simulation results comparing different cross-validation methods:

1.

kf -k-fold cross-validation by taking the average of the ridge estimators at the CV-optimal regularization parameter.

2.

kf refit -k-fold cross-validation by refitting ridge regression on the whole dataset using the CV-optimal regularization parameter.

3.

kf bic -k-fold cross-validation by refitting ridge regression on the whole dataset using the CV-optimal regularization parameter, with bias correction.

4. tt -train-test validation, by using the ridge estimator computed on the train data, at the validation-optimal regularization parameter.

Note: we expect this to be similar, but worse than the "kf" estimator.

5. tt refit -train-test validation by refitting ridge regression on the whole dataset, using the validation-optimal regularization parameter.

Note: we expect this to be similar, but slightly worse than the "kf refit" estimator.

6.

tt bic -train-test validation by refitting ridge regression on the whole dataset using the CV-optimal regularization parameter, with bias correction.

7. loo -leave-one-out Figure 8 shows that the naive estimators (kf and tt) can be quite inaccurate without refitting or bias correction.

However, if we either refit or bias-correct, the accuracy improves.

In this case, there seems to be no significant difference between the various methods.

A.6 PROOF OF THEOREM 4.1 Proof.

Suppose m/n → ξ as n goes to infinity.

Forβ p , we have

Denote M = X L LX/n + λI p −1 , the resolvent of the sketched matrix.

We further assume that X has iid N (0, 1) entries and

Therefore, using that Cov [β] = α 2 /p · I p , we find the bias as

By the properties of Wishart matrices (e.g., Anderson, 2003; Muirhead, 2009) , we have

Recalling that m, n → ∞ such that m/n → ξ, and that

Moreover,

Here we used the additional definitions

Note that these can be connected to the previous definitions by

Therefore the AMSE ofβ p is

A.6.1 ISOTROPIC CASE Consider the special case where Γ = I, that is, X has iid N (0, 1) entries.

Then F γ is the standard MP law, and we have the explicit forms for

The results are obtained by the contour integral formula

See Proposition 2.10 of Yao et al. (2015) .

A.7 PROOF OF THEOREM 4.2

Proof.

Suppose d/p → ζ as n goes to infinity.

Forβ d , we have

Denote M = XRR X /n + λI n −1 .

Note that, using that

The limit of the trace term is not entirely trivial, but it can be calculated by (1) observing that the m × p sketched data matrix P = LX has iid normal entries (2) thus the operator norm of P P/n vanishes, (3) and so by a simple matrix perturbation argument the trace concentrates around p/λ 2 .

This gives the rough steps of finding the above limit.

Moreover,

From this it is elementary to find the optimal λ and its objective value.

A.9 PROOF OF THEOREM 4.4

Proof.

Note that the bias can be written as

So we need to find the law of p) .

Then W and G −1 are asymptotically freely independent.

The l.s.d.

of W/d is the MP law F 1/ξ while the l.s.d.

of G/p is the MP law F 1/γ .

We need to find the additive free convolution W Ḡ , wherē

Recall that the R-transform of a distribution F is defined by

is the inverse function of the Stieltjes transform of F (e.g., Voiculescu et al., 1992; Hiai & Petz, 2006; Couillet & Debbah, 2011) .

We can find the R-transform by solving

The Stieltjes transform of G −1 is

The statement requires some notions from free probability, see e.g., Voiculescu et al. (1992) , and X L LX/(nd) + λI p −1 X = X (L LXX /(nd) + λI n ) −1 .

Thus

First we find the l.s.d.

of (

which is similar to ( Also note that

We write A = We will find an expression for this using free probability.

For this we will need to use some series expansions.

There are two cases, depending on whether the operator norm of BA −1 is less than or greater than unity, leading to different series expansions.

We will work out below the first case, but the second case is similar and leads to the same answer.

Since A and B are asymptotically freely independent in the free probability space arising in the limit (e.g., Voiculescu et al., 1992; Hiai & Petz, 2006; Couillet & Debbah, 2011) , and the polynomial (a −1 b) i+j+1 a −1 b −1 involves an alternating sequence of a, b, we have

where a and the b are free random variables and τ is their law.

Specifically, a is a free random variable with the MP law F 1/ξ and b is λ γb −1 , whereb is a free r.v.

with MP law F 1/γ .

Moreover, they are freely independent.

Hence, we have Published as a conference paper at ICLR 2020 First we fix the regularization parameter at the optimal value for original ridge regression.

The results are visualized in Figure 12 .

On the x axis, we plot the reduction in sample size m/n for primal sketch, and the reduction in dimension d/p for dual sketch.

In this case, primal and dual sketch will increase both bias and variance, and empirically in the current case, dual sketch increases them more.

So in this particular case, primal sketch is preferred.

We find the optimal regularization parameter λ for primal and dual orthogonal sketching.

Then we use the optimal regularization parameter for all settings, see Figure 13 .

Both primal and dual sketch increase the bias, but decrease the variance.

It is interesting to note that, for equal parameters ξ and Figure 12 : Fixed regularization parameter λ = 0.7, optimal for original ridge, in a setting where γ = 0.7, and α 2 = σ 2 .

Figure 13: Primal and dual sketch at optimal λ.

We take γ = 0.7 and let ξ range between 0.001 and 1, where for primal sketch ξ = r/n while for dual sketch ξ = d/p.

@highlight

We study the structure of ridge regression in a high-dimensional asymptotic framework, and get insights about cross-validation and sketching.

@highlight

A theoretical study of ridge regression by exploiting a new asymptotic characterisation of the ridge regression estimator.
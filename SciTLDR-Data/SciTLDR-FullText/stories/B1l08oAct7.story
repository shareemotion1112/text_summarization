Bayesian neural networks (BNNs) hold great promise as a flexible and principled solution to deal with uncertainty when learning from finite data.

Among approaches to realize probabilistic inference in deep neural networks, variational Bayes (VB) is theoretically grounded, generally applicable, and computationally efficient.

With wide recognition of potential advantages, why is it that variational Bayes has seen very limited practical use for BNNs in real applications?

We argue that variational inference in neural networks is fragile: successful implementations require careful initialization and tuning of prior variances, as well as controlling the variance of Monte Carlo gradient estimates.

We provide two innovations that aim to turn VB into a robust inference tool for Bayesian neural networks: first, we introduce a novel deterministic method to approximate moments in neural networks, eliminating gradient variance; second, we introduce a hierarchical prior for parameters and a novel Empirical Bayes procedure for automatically selecting prior variances.

Combining these two innovations, the resulting method is highly efficient and robust.

On the application of heteroscedastic regression we demonstrate good predictive performance over alternative approaches.

Bayesian approaches to neural network training marry the representational flexibility of deep neural networks with principled parameter estimation in probabilistic models.

Compared to "standard" parameter estimation by maximum likelihood, the Bayesian framework promises to bring key advantages such as better uncertainty estimates on predictions and automatic model regularization (MacKay, 1992; Graves, 2011) .

These features are often crucial for informing downstream decision tasks and reducing overfitting, particularly on small datasets.

However, despite potential advantages, such Bayesian neural networks (BNNs) are often overlooked due to two limitations: First, posterior inference in deep neural networks is analytically intractable and approximate inference with Monte Carlo (MC) techniques can suffer from crippling variance given only a reasonable computation budget (Kingma et al., 2015; Molchanov et al., 2017; Miller et al., 2017; BID8 .

Second, performance of the Bayesian approach is sensitive to the choice of prior BID1 , and although we may have a priori knowledge concerning the function represented by a neural network, it is generally difficult to translate this into a meaningful prior on neural network weights.

Sensitivity to priors and initialization makes BNNs non-robust and thus often irrelevant in practice.

In this paper, we describe a novel approach for inference in feed-forward BNNs that is simple to implement and aims to solve these two limitations.

We adopt the paradigm of variational Bayes (VB) for BNNs (Hinton & van Camp, 1993; MacKay, 1995c) which is normally deployed using Monte Carlo variational inference (MCVI) (Graves, 2011; Blundell et al., 2015) .

Within this paradigm we address the two shortcomings of current practice outlined above: First, we address the issue of high variance in MCVI, by reducing this variance to zero through novel deterministic approximations to variational inference in neural networks.

Second, we derive a general and robust Empirical Bayes (EB) approach to prior choice using hierarchical priors.

By exploiting conjugacy we derive data-adaptive closed-form variance priors for neural network weights, which we experimentally demonstrate to be remarkably effective.

Combining these two novel ingredients gives us a performant and robust BNN inference scheme that we refer to as "deterministic variational inference" (DVI).

We demonstrate robustness and improved predictive performance in the context of non-linear regression models, deriving novel closed-form results for expected log-likelihoods in homoscedastic and heteroscedastic regression (similar derivations for classification can be found in the appendix).Experiments on standard regression datasets from the UCI repository, (Dheeru & Karra Taniskidou, 2017) , show that for identical models DVI converges to local optima with better predictive loglikelihoods than existing methods based on MCVI.

In direct comparisons, we show that our Empirical Bayes formulation automatically provides better or comparable test performance than manual tuning of the prior and that heteroscedastic models consistently outperform the homoscedastic models.

Concretely, our contributions are:??? Development of a deterministic procedure for propagating uncertain activations through neural networks with uncertain weights and ReLU or Heaviside activation functions.??? Development of an EB method for principled tuning of weight priors during BNN training.??? Experimental results showing the accuracy and efficiency of our method and applicability to heteroscedastic and homoscedastic regression on real datasets.

We start by describing the inference task that our method must solve to successfully train a BNN.

Given a model M parameterized by weights w and a dataset D = (x, y), the inference task is to discover the posterior distribution p(w|x, y).

A variational approach acknowledges that this posterior generally does not have an analytic form, and introduces a variational distribution q(w; ??) parameterized by ?? to approximate p(w|x, y).

The approximation is considered optimal within the variational family for ?? * that minimizes the Kullback-Leibler (KL) divergence between q and the true posterior.

Introducing a prior p(w) and applying Bayes rule allows us to rewrite this as optimization of the quantity known as the evidence lower bound (ELBO): DISPLAYFORM0 Analytic results exist for the KL term in the ELBO for careful choice of prior and variational distributions (e.g. Gaussian families).

However, when M is a non-linear neural network, the first term in equation 1 (referred to as the reconstruction term) cannot be computed exactly: this is where MC approximations with finite sample size S are typically employed: DISPLAYFORM1 log p(y|w (s) , x), w (s) ??? q(w; ??).Our goal in the next section is to develop an explicit and accurate approximation for this expectation, which provides a deterministic, closed-form expectation calculation, stabilizing BNN training by removing all stochasticity due to Monte Carlo sampling.

Figure 1 shows the architecture of the computation of E w???q [log p(D|w)] for a feed-forward neural network.

The computation can be divided into two parts: first, propagation of activations though parameterized layers and second, evaluation of an unparameterized log-likelihood function (L).

In this section, we describe how each of these stages is handled in our deterministic framework.

We begin by considering activation propagation (figure 1(a)), with the aim of deriving the form of an approximationq(a L ) to the final layer activation distribution q(a L ) that will be passed to the likelihood computation.

We compute a L by sequentially computing the distributions for the activations in the preceding layers.

Concretely, we define the action of the l th layer that maps a DISPLAYFORM0 to a l as follows: DISPLAYFORM1 where f is a non-linearity and {W l , b l } ??? w are random variables representing the weights and biases of the l th layer that are assumed independent from weights in other layers.

For notational clarity, in the following we will suppress the explicit layer index l, and use primed symbols to denote variables from the (l ??? 1) th layer, e.g. a = a (l???1) .

Note that we have made the non-conventional choice to draw the boundaries of the layers such that the linear transform is applied after the nonlinearity.

This is to emphasize that a l is constructed by linear combination of many distinct elements of h , and in the limit of vanishing correlation between terms in this combination, we can appeal to the central limit theorem (CLT).

Under the CLT, for a large enough hidden dimension and for variational distributions with finite first and second moments, elements a i will be normally distributed regardless of the potentially complicated distribution for h j induced by f 1 .

We empirically observe that this claim is approximately valid even when (weak) correlations appear between the elements of h during training (see section 3.1.1).Having argued that a adopts a Gaussian form, it remains to compute the first and second moments.

In general, these cannot be computed exactly, so we develop an approximate expression.

An overview of this derivation is presented here with more details in appendix A. First, we model W , b and h as independent random variables, allowing us to write: DISPLAYFORM2 where we have employed the Einstein summation convention and used angle brackets to indicate expectation over q. If we choose a variational family with analytic forms for weight means and covariances (e.g. Gaussian with variational parameters W ji and Cov(W ji , W lk )), then the only difficult terms are the moments of h: Table 1 : Forms for the components of the approximation in equation 6 for Heaviside and ReLU non-linearities.

?? is the CDF of a standard Gaussian, SR is a "soft ReLU" that we define as SR(x) = ??(x) + x??(x) where ?? is a standard Gaussian,?? = 1 ??? ?? 2 , g h = arcsin ?? and g r = g h + ?? 1+?? DISPLAYFORM3 DISPLAYFORM4 where we have used the Gaussian form of a parameterized by mean a and covariance ?? , and for brevity we have omitted the normalizing constants.

Closed form solutions for the integral in equation 4 exist for Heaviside or ReLU choices of non-linearity f (see appendix A).

Furthermore, for these non-linearities, the a j ??? ????? and a l ??? ????? asymptotes of the integral in equation 5 have closed form.

FIG2 shows schematically how these asymptotes can be used as a first approximation for equation 5.

This approximation is improved by considering that (by definition) the residual decays to zero far from the origin in the ( a j , a l ) plane, and so is well modelled by a decaying function exp[???Q( a j , a l , ?? )], where Q is a polynomial in a with a dominant positive even term.

In practice we truncate Q at the quadratic term, and calculate the polynomial coefficients by matching the moments of the resulting Gaussian with the analytic moments of the residual.

Specifically, using dimensionless variables ?? i = a i / ?? ii and ?? ij = ?? ij / ?? ii ?? jj , this improved approximation takes the form where the expressions for the dimensionless asymptote A and quadratic Q are given in table table 1 propagate moments all the way through the network to compute the mean and covariances ofq(a L ), our explicit multivariate Gaussian approximation to q(a L ).

Any deep learning framework supporting special functions arcsin and ?? will immediately support backpropagation through the deterministic expressions we have presented.

Below we briefly empirically verify the presented approximation, and in section 3.2 we will show how it is used to compute an approximate log-likelihood and posterior predictive distribution for regression and classification tasks.

DISPLAYFORM5

Approximation accuracy The approximation derived above relies on three assumptions.

First, that some form of CLT holds for the hidden units during training where the iid assumption of the classic CLT is not strictly enforced; second, that a quadratic truncation of Q is sufficient 2 ; and third that there are only weak correlation between layers so that they can be represented using independent variables in the variational distribution.

To provide evidence that these assumptions hold in practice, we train a small ReLU network with two hidden layers each of 128 units to perform 1D heteroscedastic regression on a toy dataset of 500 points drawn from the distribution shown in FIG3 (b).

Deeper networks and skip connections are considered in appendix C. The training objective is taken from section 4, and the only detail required here is that a L is a 2-element vector where the elements are labelled as (m, ).

We use a diagonal Gaussian variational family to represent the weights, but we preserve the full covariance of a during propagation.

Using an input x = 0.25 (see arrow, FIG3 (b)) we compute the distributions for m and both at the start of training (where we expect the iid assumption to hold) and at convergence (where iid does not necessarily hold).

FIG3 shows the comparison between a L distributions reported by our deterministic approximation and MC evaluation using 20k samples from q(w; ??).

This comparison is qualitatively excellent for all cases considered. .

Whereas MCVI can always trade compute and memory for accuracy by choosing a small value for S, the inherent scaling of DVI with d could potentially limit its practical use for networks with large hidden size.

To avoid this limitation, we also consider the case where only the diagonal entries Cov(h j , h j ) are computed and stored at each layer.

We refer to this method as "diagonal-DVI" (dDVI), and in section 6 we show the surprising result that the strong test performance of DVI is largely retained by dDVI across a range of datasets.

FIG4 shows the time required to propagate activations through a single layer using the MCVI, DVI and dDVI methods on a Tesla V100 GPU.

As a rough rule of thumb (on this hardware), for layer sizes of practical relevance, we see that absolute DVI runtimes roughly equate to MCVI with S = 300 and dDVI runtime equates to S = 1.

To use the moment propagation procedure derived above for training BNNs, we need to build a function L that maps final layer activations a L to the expected log-likelihood term in equation 1 (see figure 1(b) ).

In appendix B.1 we show the intuitive result that this expected log-likelihood over q(w) can be rewritten as an expectation overq(a L ).

DISPLAYFORM0 With this form we can derive closed forms for specific tasks; for brevity we focus on the regression case and refer the reader to appendices B.4 and B.5 for the classification case.

Regression Case For simplicity we consider scalar y and a Gaussian noise model parameterized by mean m(x; w) and heteroscedastic log-variance log ?? 2 y (x) = (x; w).

The parameters of this Gaussian are read off as the elements of a 2-dimensional output layer a L = (m, ) so that p(y|a L ) = N y|m, e .

Recall that these parameters themselves are uncertain and the statistics a L and ?? L can be computed following section 3.1.

Inserting the Gaussian forms for p(y|a L ) and q(a L ) into equation 7 and performing the integral (see appendix B.2) gives a closed form expression for the ELBO reconstruction term: DISPLAYFORM1 This heteroscedastic model can be made homoscedastic by setting = ?? = ?? m = 0.

The expression in equation 8 completes the derivations required to implement the closed form approximation to the ELBO reconstruction term for training a network.

In addition, we can also compute a closed form approximation to the predictive distribution that is used at test-time to produce predictions that incorporate all parameter uncertainties.

By approximating the moments of the posterior predictive and assuming normality (see appendix B.3), we find: DISPLAYFORM2

So far, we have described methods for deterministic approximation of the reconstruction term in the ELBO.

We now turn to the KL term.

For a d-dimensional Gaussian prior p(w) = N (?? p , ?? p ), the KL divergence with the Gaussian variational distribution q = N (?? q , ?? q )

has closed form: DISPLAYFORM0 However, this requires selection of (?? p , ?? p ) for which there is usually little intuition beyond arguing ?? p = 0 by symmetry and choosing ?? p to preserve the expected magnitude of the propagated activations (Glorot & Bengio, 2010; He et al., 2015) .

In practice, variational Bayes for neural network parameters is sensitive to the choice of prior variance parameters, and we will demonstrate this problem empirically in section 6 (figure 5).To make variational Bayes robust we parameterize the prior hierarchically, retaining a conditional diagonal Gaussian prior and variational distribution on the weights.

The hierarchical prior takes the form s ??? p(s); w ??? p(w|s), using an inverse gamma distribution on s as the conjugate prior to the elements of the diagonal Gaussian variance.

We partition the weights into sets {??} that typically coincide with the layer partitioning 3 , and assign a single element in s to each set: DISPLAYFORM1 for shape ?? and scale ??, and where w ?? i is the i th weight in set ??.

Rather than taking the fully Bayesian approach, we adopt an empirical Bayes approach (Type-2 MAP), optimizing s ?? , assuming that the integral is dominated by a contribution from this optimal value s ?? = s ?? * .

We use the data to inform the optimal setting of s ?? * to produce the tightest ELBO: DISPLAYFORM2 Writing out the integral for the KL in equation 12, substituting in the forms of the distributions in equation 11 and differentiating to find the optimum gives DISPLAYFORM3 where ??? ?? is the number of weights in the set ??.

The influence of the data on the choice of s ?? * is made explicit here through dependence on the learned variational parameters ?? q and ?? q .

Using s ?? * to populate the elements of the diagonal prior variance ?? p , we can evaluate the KL in equation 10 under the empirical Bayes prior.

Optimization of the resulting ELBO then simultaneously tunes the variational distribution and prior.

In the experiments we will demonstrate that the proposed empirical Bayes approach works well; however, it only approximates the full Bayesian solution, and it could fail if we were to allow too many degrees of freedom.

To see this, assume we were to use one prior per weight element, and we would also define a hyperprior for each prior mean.

Then, adjusting both the prior variance and prior mean using empirical Bayes would always lead to a KL-divergence of zero and the ELBO objective would degenerate into maximum likelihood.

Bayesian neural networks have a rich history.

In a 1992 landmark paper David MacKay demonstrated the many potential benefits of a Bayesian approach to neural network learning (MacKay, 1992) ; in particular, this work contained a convincing demonstration of naturally accounting for model flexibility in the form of the Bayesian Occam's razor, facilitating comparison between different models, accurate calibration of predictive uncertainty, and to perform learning robust to overfitting.

However, at the time Bayesian inference was achieved only for small and shallow neural networks using a comparatively crude Laplace approximation.

Another early review article summarizing advantages and challenges in Bayesian neural network learning is (MacKay, 1995c) .This initial excitement around Bayesian neural networks led to two main methods being developed; First, Hinton & van Camp (1993) and MacKay (1995b) developed the variational Bayes (VB) approach for posterior inference.

Whereas Hinton & van Camp (1993) were motivated from a minimum description length (MDL) compression perspective, MacKay (1995b) motivated his equivalent ensemble learning method from a statistical physics perspective of variational free energy minimization.

Barber & Bishop (1998) extended the methodology for two-layer neural networks to use general multivariate Normal variational distributions.

Second, Neal (1993) developed efficient gradient-based Monte Carlo methods in the form of "hybrid Monte Carlo", now known as Hamiltonian Monte Carlo, and also raised the question of prior design and limiting behaviour of Bayesian neural networks.

Rebirth of Bayesian neural networks.

After more than a decade of no further work on Bayesian neural networks Graves (2011) revived the field by using Monte Carlo variational inference (MCVI) to make VB practical and scalable, demonstrating gains in predictive performance on real world tasks.

Since 2015 the VB approach to Bayesian neural networks is mainstream (Blundell et al., 2015) ; key research drivers since then are the problems of high variance in MCVI and the search for useful variational families.

One approach to reduce variance in feedforward networks is the local reparameterization trick (Kingma et al., 2015) (see appendix E).

To enhance the variational families more complicated distributions such as Matrix Gaussian posteriors (Louizos & Welling, 2016) , multiplicative posteriors (Kingma et al., 2015) , and hierarchical posteriors (Louizos & Welling, 2017) are used.

Both our methods, the deterministic moment approximation and the empirical Bayes estimation, can potentially be extended to these richer families.

Prior choice.

Choosing priors in Bayesian neural networks remains an open issue.

The hierarchical priors for feedforward neural networks that we use have been investigated before by BID1 and MacKay (1995a) , the latter proposing a "cheap and cheerful" heuristic, alternating optimization of weights and inverse variance parameters.

Barber & Bishop (1998) also used a hierarchical prior and an efficient closed-form factored VB approximation; our approach can be seen as a point estimate to their approach in order to enable use of our closed-form moment approximation.

Note that Barber & Bishop (1998) manipulate an expression for h j h l into a one-dimensional integral, whereas our approach gives closed form approximations for this integral without need for numerical integration.

Graves (2011) also used hierarchical Gaussian priors with flat hyperpriors, deriving a closed-form update for the prior mean and variance.

Compared to these prior works our approach is rigorous and with sufficient data accurately approximates the Bayesian approach of integrating over the prior parameters.

Alternative inference procedures.

As an alternative to variational Bayes, probabilistic backpropagation (PBP) (Hern??ndez-Lobato & Adams, 2015) applies approximate inference in the form of assumed density filtering (ADF) to refine a Gaussian posterior approximation.

Like in our work, each update to the approximate posterior requires propagating means and variances of activations through the network. (Hern??ndez-Lobato & Adams, 2015) only consider the diagonal propagation case and homoscedastic regression.

Since the original work, PBP has been generalized to classification (Ghosh et al., 2016) and richer posterior families such as the matrix variate Normal posteriors BID5 .

Our moment approximation could be used to improve the inference accuracy of PBP, and since we handle minibatches of data rather than processing one data point at a time, our method is more computationally efficient.

Gaussianity in neural networks.

These methods allow for approximate posterior parameter inference using unbiased log-likelihood estimates.

Stochastic gradient Langevin dynamics (SGLD) was the first method in this class BID7 .

SGLD is particularly simple and efficient to implement, but recent methods increase efficiency in the case of correlated posteriors by estimating the Fisher information matrix (Ahn et al., 2012) and extend Hamiltonian Monte Carlo to the stochastic gradient case (Chen et al., 2014) .

A complete characterization of SG-MCMC methods is given by (Ma et al., 2015; Gong et al., 2018) .

However, despite this progress, important theoretical questions regarding approximation guarantees for practical computational budgets remain BID0 .

Moreover, while SG-MCMC methods work robustly in practice, they remain computationally inefficient, especially because evaluation of the posterior predictive requires evaluating an ensemble of models.

Wild approximations.

The above methods are principled but often require sophisticated implementations; recently, a few methods aim to provide "cheap" approximations to the Bayes posterior.

Dropout has been interpreted by Gal & Ghahramani (2016) to approximately correspond to variational inference.

Likewise, Bootstrap posteriors (Lakshminarayanan et al., 2017; Fushiki et al., 2005; Harris, 1989) have been proposed as a general, robust, and accurate method for posterior inference.

However, obtaining a bootstrap posterior ensemble of size k is computationally intense at k times the computation of training a single model.

We implement 4 deterministic variational inference (DVI) as described above to train small ReLU networks on UCI regression datasets (Dheeru & Karra Taniskidou, 2017) .

The experiments address the claims that our methods for eliminating gradient variance and automatic tuning of the prior improve the performance of the final trained model.

In Appendix D we present extended results to demonstrate that our method is competitive against a variety of models and inference schemes.

DISPLAYFORM0 where N is the number of elements in W (see appendix A.1).

Additionally, both methods use the same EB prior from equation 13 with a broad inverse Gamma hyperprior (?? = 1, ?? = 10) and an independent s ?? for each linear transformation.

Each dataset is split into random training and test sets with 90% and 10% of the data respectively.

This splitting process is repeated 20 times and the average test performance of each method at convergence is reported in table 2 (see also learning curves in appendix F).

We see that DVI consistently outperforms MCVI, by up to 0.35 nats per data point on some datasets.

The computationally efficient diagonal-DVI (dDVI) surprisingly retains much of this performance.

By default we use the heteroscedastic model, and we observe that this uniformly delivers better results than a homoscedastic model (hoDVI; rightmost column in table 2) on these datasets with no overfitting issues 6 .

Empirical Bayes In FIG5 we compare the performance of networks trained with manual tuning of a fixed Gaussian prior to networks trained with the automatic EB tuning.

We find that the EB method consistently finds priors that produce models with competitive or significantly improved test log-likelihood relative to the best manual setting.

Since this observation holds across all datasets considered, we say that our method is "robust".

Note that the EB method can outperform manual tuning because it automatically finds different prior variances for each weight matrix, whereas in the manual tuning case we search over a single hyperparameter controlling all prior variances.

An additional ablation study showing the relative contribution of our deterministic approach and the EB prior are shown in appendix D.1.

We introduced two innovations to make variational inference for neural networks more robust: 1.

an effective deterministic approximation to the moments of activations of a neural networks; and 2. a simple empirical Bayes hyperparameter update.

We demonstrate that together these innovations make variational Bayes a competitive method for Bayesian inference in neural heteroscedastic regression models.

Bayesian neural networks have been shown to substantially improve upon standard networks in these settings where calibrated predictive uncertainty estimates, sequential decision making, or continual learning without catastrophic forgetting are required (see e.g. BID3 Gal et al. (2017) ; Nguyen et al. FORMULA0 ).

In future work, the new innovations proposed in this paper can be applied to these areas.

In the sequential decision making and continual learning applications, approximate Bayesian inference must be run as an inner loop of a larger algorithm.

This requires a robust and automated version of BNN training: this is precisely where we believe the innovations in this paper will have large impact since they pave the way to automated and robust deployment of BBNs that do not involve an expert in-the-loop.

Under assumption of independence of h, W and b, we can write: DISPLAYFORM0 which is seen in the main text as equation 3.

For Heaviside and ReLU activation functions, closed forms exist for h j in equation 14: DISPLAYFORM1 where SR(x) ?? ?? = ??(x) + x??(x) is a "soft ReLU", ?? and ?? represent the standard Gaussian PDF and CDF, and we have introduced the dimensionless variables ?? j = a j / ?? jj .

These results are is sufficient to evaluate equation 14, so in the following sections we turn to each term from equation 15.

In the general case, we can use the results from section A.2 to evaluate off-diagonal h j h l .

However, in our experiments we always consider the the special case where Cov(W ji , W lk ) is diagonal.

In this case we can write the first term in equation 15 as (reintroducing the explicit summation): DISPLAYFORM0 i.e. this term is a diagonal matrix with the diagonal given by the left product of the vector v j = h j h j with the matrix Var(W ki ).

Note that h j h j can be evaluated analytically for Heaviside and ReLU activation functions: DISPLAYFORM1 Evaluation of Cov(h j , h l ) requires an expression for h j h l .

From equation 5, we write: DISPLAYFORM2 where P is the quadratic form: DISPLAYFORM3 Here we have introduced further dimensionless variables ?? j = ?? j / ?? jj , ?? l = ?? l / ?? ll and ?? jl = ?? jl / ?? jj ?? ll .

We can then rewrite equation 16 in terms of a dimensionless integral I using a scale factor S jl that is 1 for the Heaviside non-linearity or ?? jl /?? jl for ReLU: DISPLAYFORM4 The normalization constant, Z, is evaluated by integrating over e ???P/2 and is explicitly written as Z = 2???? jl , where?? jl = 1 ??? ?? 2 jl .

Now, following equation 6, we have the task to write I as an asymptote A plus a decaying correction e ???Q .

To evaluate A and Q, we have to insert the explicit form of the non-linearity f , which we do for Heaviside and ReLU functions in the next sections.

For the Heaviside activation, we can represent the integral I as the shaded area under the Gaussian in the upper-left quadrant shown below.

In general, this integral does not have a closed form.

However, for ?? j ??? ???, vanishing weight appears under the Gaussian in the upper-right quadrant, so we can write down the asymptote of the integral in this limit: DISPLAYFORM0 Here we performed the integral by noticing that the outer integral over ?? j marginalizes out ?? j from the bivariate Gaussian, leaving the inner integral as the definition of the Gaussian CDF.

By symmetry, we also have lim ?? l ?????? I = ??(?? j ) and lim ?? j,l ????????? I = 0.

We can then write down the following symmetrized form that satisfies all the limits required to qualify as an asymptote: DISPLAYFORM1 To compute the correction factor we evaluate the derivatives of (I ??? A) at the origin up to second order to match the moments of e ???Q for quadratic Q. Description of this process is found below Zeroth derivative At the origin ?? j = ?? l = 0, we can diagonalize the quadratic form P : DISPLAYFORM2 .

Performing this change of variables in the integral gives: DISPLAYFORM3 where we integrated in polar coordinates over the region H in which the Heaviside function is non-zero.

The angle ?? can be found from the coordinate transform between ?? and ?? as 7 : DISPLAYFORM4 Since A| ?? =0 = ??(0)??(0) = 1/4, we can evaluate: DISPLAYFORM5 Here we use the identity cos(2 arctan x) = cos DISPLAYFORM6 First derivative Performing a change of variables x i = ?? i ??? ?? i , we can write I as: DISPLAYFORM7 where H is the Heaviside function.

Now, using ??? x H(x) = ??(x), we have: DISPLAYFORM8 In addition, using ??? x ??(x) = ??(x), we have: DISPLAYFORM9 By symmetry (I ??? A) also has zero gradient with respect to ?? l at the origin.

Therefore Q has no linear term in ?? .Second derivative Taking another derivative in equation 17 gives: DISPLAYFORM10 where we used the identity f (x)??? x ??(x)dx = ??? ??(x)??? x f (x)dx, which holds for arbitrary f .

In addition, we have: DISPLAYFORM11 and the same result holds for the second derivative w.r.t.

?? l .

To complete the Hessian, it is a simple extension of previous results to show that: DISPLAYFORM12 Now that we have obtained derivatives of the residual (I ??? A) up to second order we propose a correction factor of the form e ???Q where Q is truncated at quadratic terms: DISPLAYFORM13 We then find the coefficients {??, ??, ??} by matching ( ! =) derivatives at ?? = 0: DISPLAYFORM14 This yields the expression seen in table 1 of the main text.

As in the Heaviside case, we begin by computing the asymptote of I by inspecting the limit as ?? j ??? ???: DISPLAYFORM0 Now, we construct a full 2-dimensional asymptote by symmetrizing equation 18 (using properties SR(x) ??? x and ??(x) ??? 1 as x ??? ??? to check that the correct limits are preserved after symmetrizing): DISPLAYFORM1 Next we compute the correction factor e ???Q .

The details of this procedure closely follow those for the Heaviside non-linearity of the previous section, so we omit them here (and in practice we use Mathematica to perform the intermediate calculations).

The final result is presented in table 1 of the main text.

Here we give derivations of expressions quoted in section 3.2.

In section B.1 we justify the intuitive result that expectation of the ELBO reconstruction term over q(w; ??) can be re-written as an expectation overq(a L ).

We then derive expected log-likelihoods and posterior predictive distributions for the cases of univariate Gaussian regression and classification.

The latter sections are arranged as follows:

Log-likelihood section B.2 section B.4 Posterior predictive section B.3 section B.5 DISPLAYFORM0 We begin by rewriting the reconstruction term for data point (x, y) in terms of a L : DISPLAYFORM1 where we have suppressed explicit conditioning on x for brevity.

Our goal now is to perform the integral over w, leaving the expectation in terms of a L only, thus allowing it to be evaluated using the approximationq(a L ) from section 3.1.To eliminate w, consider the case where the output of the model is a distribution p(y|a L ) that is a parameter-free transformation of a L (e.g. a L are logits of a softmax distribution for classification or the moments of a Gaussian for regression).

Since the model output is conditioned only on a L , we must have p(y|w) = p(y|a L ) for all configurations w that satisfy the deterministic transformation a L = M(x; w), where M is the neural network (i.e p(y|w) = p(y|a L ) for all w where q(w|a L ) is non-zero).

This allows us to write: DISPLAYFORM2 so the reconstruction term becomes: DISPLAYFORM3 This establishes the equivalence given in equation 7 in the main text.

Since we are using an approximation to q, we will actually compute E a L ???q(a L ) log p(y|a L ) .

Here we give a derivation of equation 8 from the main text.

Throughout this section we label the 2 elements of the final activation vector as a L = (m, ).

We first insert the Gaussian form for p(y|a L ) ??? N m, e into the log-likelihood expression: DISPLAYFORM0 Now we use the Gaussian form ofq(a L ): DISPLAYFORM1 and note that DISPLAYFORM2 where e = (0, 1) is the unit vector in the coordinate, and we completed the square to obtain the final line.

Inserting equation 20 into equation 19 and marginalizing out the coordinate gives: DISPLAYFORM3 dm .Finally, performing the integral over m gives the result seen in equation 8.

Here we give a derivation of equation 9 from the main text.

We first calculate the first and second moments of the predictive distribution under the approximation q(a L ) ???q(a L ): DISPLAYFORM0 where the final integral in the variance computation is performed by inserting the Gaussian form for q(a L ) and completing the square.

Then we assume normality of the predictive distribution to obtain the result in equation 9.

There is no exact form for the expected log-likelihood for multivariate classification with logits a L .

However, using the second-order Delta method BID4 , we find the expansion DISPLAYFORM0 To derive this expansion, we first state the second order expansion for the expectation of a function g of random variable x using the Delta method as follows 8 : DISPLAYFORM1 where C ij = Cov(x i , x j ).

Now we note that the logsumexp function has a simple Hessian ??? 2 ???xi???xj logsumexp(x) = ?? ij p i ??? p i p j , where p = softmax(x).

Putting these results together allows us to write: DISPLAYFORM2 This result is sufficient to complete the derivation of equation 21 and enable training of a classifier using our method.

Using the same second-order Delta method, we find the following expansion for the posterior predictive distribution: DISPLAYFORM0 where p = softmax( a L ).For this expansion, we begin by computing the Hessian: DISPLAYFORM1 where p = softmax(x), and we used the intermediate result DISPLAYFORM2 Then we can form the product: DISPLAYFORM3 and insert this into equation 22 to obtain equation 23.Preliminary experiments show that good results are obtained either using these approximations or a lightweight MC approximation just to perform the mapping of a L to (log)p after the deterministic heavy-lifting of computing a L .

In this work we are primarily concerned with demonstrating the benefits of the moment propagation method from section 3.1, so we limit our experiments to regression examples without additional complication from approximation of the likelihood function.

Here we consider the applicability of our method to the regime of deep, narrow networks.

This regime is challenging because for small hidden dimension the Gaussian approximation for a (reliant on the CLT) breaks down, and these errors accumulate as the net becomes deep.

We empirically explore this potential problem by investigating deep networks containing 5 layers of only 5, 25 or 125 units each.

FIG6 shows results analogous to figure 3 that qualitatively illustrate how well our approximation matches the true variational distribution of output activations both at the start and end of training.

We see that our CLT-based approximation is good in the 125-and 25-unit cases, but is poor in the 5-unit case.

Since it is generally considered that optimization of neural networks only works well in the high dimensional setting with at least a few tens of hidden units, these empirical observations suggest that our approximation is applicable in practically relevant architectures.

Training deep networks is considered difficult even in the traditional maximum-likelihood setting due to the problems of exploding and vanishing gradients.

A popular approach to combat these issues is to add skip connections to the architecture.

Here we derive the necessary results to add skip connections to our deterministic BNN.We consider a simple layer with skip connections of the following form: DISPLAYFORM0 The moment propagation expressions for this layer are (using the bilinearity of Cov): DISPLAYFORM1 where ?? i and Cov(?? i , ?? k ) can be computed using analogy to equations 14 and 15.

This just leaves computation of Cov(a i , ?? k ) and its transpose, which can be performed analytically using integral results and methods borrowed from appendix A. DISPLAYFORM2 Using this result, we implement a 5-layer, 25-unit network with skip connections.

In FIG6 we qualitatively verify the validity of our approximation on this architecture by observing a good match with Monte Carlo simulations using 20k samples.

Here we include comparison with a number of different models and inference schemes on the 9 UCI datasets considered in the main text.

We report test log-likelihoods at convergence and find that our method is competitive or superior to a range of state-of-the-art techniques (reproduced from Bui et al. (2016)

Here we provide an ablation study that indicates the individual contributions of (1) the deterministic approximation and (2) the the empirical Bayes prior.

We consider all combinations of DVI or MCVI with and without empirical Bayes.

In the DVI-fixed and MCVI-fixed cases without empirical Bayes we use a fixed zero-mean Gaussian prior during training and we perform separate runs to tune the prior variance, reporting the best performance achieved (cf.

figure 5) 9 .

Since the EB approach requires no hyperparameter tuning between the datasets shown, these results hide the considerable computational advantaged that the EB approach brings.

By eliminating MC sampling and its associated variance entirely, our method directly tackles the problem of high variance gradient estimates that hinder MC approaches to training of BNNs.

Alternative methods that only reduce variance have been considered, and among these, the local 1.13 ?? 0.00 1.12 ?? 0.00 1.14 ?? 0.00 1.13 ?? 0.00 nava 6.29 ?? 0.04 6.32 ?? 0.04 5.94 ?? 0.05 6.00 ?? 0.02 powe ???2.80 ?? 0.00 ???2.80 ?? 0.01 ???2.80 ?? 0.00 ???2.80 ?? 0.00 prot ???2.85 ?? 0.01 ???2.84 ?? 0.01 ???2.87 ?? 0.01 ???2.89 ?? 0.01 wine ???0.90 ?? 0.01 ???0.94 ?? 0.01 ???0.92 ?? 0.01 ???0.94 ?? 0.01 yach ???0.47 ?? 0.03 ???0.49 ?? 0.03 ???0.68 ?? 0.03 ???0.56 ?? 0.03 Table 5 : Ablation study of all combinations of DVI and MCVI with EB or a fixed prior.

One standard deviation error in the last significant digit is shown in paraentheses.

reparameterization trick (Kingma et al., 2015) is particularly popular.

Similar to our approach, the local reparameterization trick maps the uncertainty in the weights to an uncertainty in activations, however, unlike the fully deterministic DVI, MC methods are then used to propagate this uncertainty through non-linearities.

The benefits of MCVI with the reparameterization trick (rMCVI) over vanilla MCVI are two-fold:??? The variance of the gradient estimates during back propagation are reduced (see details in Kingma et al. (2015) ).???

Since the sampling dimension in rMCVI only appears on the activations and not on the weights, an H ?? H linear transform can be implemented using SB ?? H by H ?? H matrix multiplies (where S is the number of samples and B is the batch size).

This contrasts with the S ?? B ?? H by S ?? H ?? H batched matrix multiply required for MCVI.

Although both of these algorithms have the same asymptotic complexity O(SBH H), a single large matrix multiplication is generally more efficient on GPUs than smaller batched matrix multiplies.

FIG8 shows empirical studies of the gradient variance and runtime for rMCVI vs. MCVI applied to the model described in section 3.1.1 and figure 3.

To evaluate the gradient variance, we initialize the model with partially trained weights and measure the variance of the gradient of the ELBO reconstruction term L with respect to variational parameters.

Specifically, we inspect the gradient with respect to the parameters ?? The plots in figure 7 serve to show that rMCVI is not fundamentally different from MCVI, and the performance of one (on either the speed or variance metric) can be transformed into the other by varying the number of samples.

A comparison of DVI with rMCVI is included in table 3 using the implementation labelled as "VI(KW)-1".

F LEARNING CURVES FIG10 shows the test log-likelihood during the training of the models from table 2 using DVI and MCVI inference algorithms.

Since the underlying model is identical, both methods should achieve the same test log-likelihood given infinite time and infinite MC samples (or a suitable learning rate schedule) to mitigate the increased variance of the MCVI method.

However, since we use only 10 samples and do not employ a leaning rate schedule, we find that MCVI converges to a log-likelihood that is consistently worse than that achieved by DVI.

@highlight

A method for eliminating gradient variance and automatically tuning priors for effective training of bayesian neural networks

@highlight

Proposes a new approach to perform deterministic variational inference for feed-forward BNN with specific nonlinear activation functions by approximating layerwise moments.

@highlight

The paper considers a purely deterministic approach to learning variational posterior approximations for Bayesian neural networks.
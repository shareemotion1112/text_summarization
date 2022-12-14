Modelling statistical relationships beyond the conditional mean is crucial in many settings.

Conditional density estimation (CDE) aims to learn the full conditional probability density from data.

Though highly expressive, neural network based CDE models can suffer from severe over-fitting when trained with the maximum likelihood objective.

Due to the inherent structure of such models, classical regularization approaches in the parameter space are rendered ineffective.

To address this issue, we develop a model-agnostic noise regularization method for CDE that adds random perturbations to the data during training.

We demonstrate that the proposed approach corresponds to a smoothness regularization and prove its asymptotic consistency.

In our experiments, noise regularization significantly and consistently outperforms other regularization methods across seven data sets and three CDE models.

The effectiveness of noise regularization makes neural network based CDE the preferable method over previous non- and semi-parametric approaches, even when training data is scarce.

While regression analysis aims to describe the conditional mean E[y|x] of a response y given inputs x, many problems such as risk management and planning under uncertainty require gaining insight about deviations from the mean and their associated likelihood.

The stochastic dependency of y on x can be captured by modeling the conditional probability density p(y|x).

Inferring such a density function from a set of empirical observations {(x n , y n )} N n=1 is typically referred to as conditional density estimation (CDE) and is the focus of this paper.

In the recent machine learning literature, there has been a resurgence of interest in high-capacity density models based on neural networks (Dinh et al., 2017; Ambrogioni et al., 2017; Kingma & Dhariwal, 2018) .

Since this line of work mainly focuses on the modelling of images based on large scale data sets, over-fitting and noisy observations are of minor concern in this context.

In contrast, we are interested in CDE in settings where data may be scarce and noisy.

When combined with maximum likelihood estimation, the flexibility of such high-capacity models results in over-fitting and poor generalization.

While regression typically assumes Gaussian conditional noise, CDE uses expressive distribution families to model deviations from the conditional mean.

Hence, the overfitting problem tends to be even more severe in CDE than in regression.

Classical regularization of the neural network weights such as weight decay (Pratt & Hanson, 1989) has been shown to be effective for regression and classification.

However, in the context of CDE, the output of the neural network merely controls the parameters of a density model such as a Gaussian Mixture or Normalizing Flow.

This makes the standard regularization methods in the parameter space less effective and harder to analyze.

Aiming to address this issue, we propose and analyze noise regularization, a method well-studied in the context of regression and classification, for the purpose of conditional density estimation.

In that, the paper attempts to close a gap in previous research.

By adding small random perturbations to the data during training, the conditional density estimate is smoothed and tends to generalize better.

In fact, we show that adding noise during maximum likelihood estimation is equivalent to penalizing the second derivatives of the conditional log-probability.

Visually, the respective regularization term punishes very curved or even spiky density estimators in favor of smoother variants, which proves to be a favorable inductive bias in many applications.

Moreover, under some regularity conditions, we show that the proposed regularization scheme is asymptotically consistent, converging to the unbiased maximum likelihood estimator.

This does not only support the soundness of the proposed method but also endows us with useful insight in how to set the regularization intensity relative to the data dimensionality and training set size.

Overall, the proposed noise regularization scheme is easy to implement and agnostic to the parameterization of the CDE model.

We empirically demonstrate its effectiveness on three different neural network based models.

The experimental results show that noise regularization outperforms other regularization methods significantly and consistently across various data sets.

Finally, we demonstrate that, when properly regularized, neural network based CDE is able to improve upon state-of-the art non-parametric estimators, even when only 400 training observations are available.

Density Estimation.

Let X be a random variable with probability density function (PDF) p(x) defined over the domain X ??? R dx .

Given a collection D = {x 1 , ..., x n } of observations sampled from p(x), the goal is to find a good estimatef (x) of the true density function p.

In parametric estimation, the PDFf is assumed to belong to a parametric family F = {f ?? (??)|?? ??? ??} where the density function is described by a finite dimensional parameter ?? ??? ??. The standard method for estimating ?? is maximum likelihood estimation, wherein ?? is chosen so that the likelihood of the data D is maximized.

This is equivalent to minimizing the Kullback-Leibler divergence between the empirical data distribution p D (x) = 1 n n i=1 ??(||x ??? x i ||) (i.e., mixture of point masses in the observations x i ) and the parametric distributionf ?? :

From a geometric perspective, (1) can be viewed as an orthogonal projection of p D (x) onto F w.r.t.

the reverse KL-divergence.

Hence, (1) is also commonly referred to as an M-projection (Murphy, 2012; Nielsen, 2018) .

In contrast, non-parametric density estimators make implicit smoothness assumptions through a kernel function.

The most popular non-parametric method, kernel density estimation (KDE), places a symmetric density function K(z), the so-called kernel, on each training data point x n (Rosenblatt, 1956; Parzen, 1962) .

The resulting density estimate reads asq(

.

Beyond the appropriate choice of K(??), a central challenge is the selection of the bandwidth parameter h which controls the smoothness of the estimated PDF (Li & Racine, 2007) .

Conditional Density Estimation (CDE).

Let (X, Y ) be a pair of random variables with respective domains X ??? R dx and Y ??? R dy and realizations x and y. Let p(y|x) = p(x, y)/p(x) denote the conditional probability density of y given x. Typically, Y is referred to as a dependent variable (explained variable) and X as conditional (explanatory) variable.

Given a dataset of observations D = {(x n , y n )} N n=1 drawn from the joint distribution (x n , y n ) ??? p(x, y), the aim of conditional density estimation (CDE) is to find an estimatef (y|x) of the true conditional density p(y|x).

In the context of CDE, the KL-divergence objective is expressed as expectation over p(x):

Corresponding to (1), we refer to the minimization of (2) w.r.t.

?? as conditional M-projection.

Given a dataset D drawn i.i.d.

from p(x, y), the conditional MLE following from (2) can be stated as

3 RELATED WORK The first part of this section discusses relevant work in the field of CDE, focusing on high-capacity models that make little prior assumptions.

The second part relates our approach to previous regularization and data augmentation methods.

Non-parametric CDE.

A vast body of literature in statistics and econometrics studies nonparametric kernel density estimators (KDE) (Rosenblatt, 1956; Parzen, 1962) and the associated bandwidth selection problem, which concerns choosing the appropriate amount of smoothing (Silverman, 1982; Hall et al., 1992; Cao et al., 1994) .

To estimate conditional probabilities, previous work proposes to estimate both the joint and marginal probability separately with KDE and then computing the conditional probability as their ratio (Hyndman et al., 1996; Li & Racine, 2007) .

Other approaches combine non-parametric elements with parametric elements (Tresp, 2001; Sugiyama & Takeuchi, 2010; Dutordoir et al., 2018) .

Despite their theoretical appeal, non-parametric density estimators suffer from poor generalization in regions where data is sparse (e.g., tail regions), causing rapid performance deterioration as the data dimensionality increases (Scott & Wand, 1991) .

CDE based on neural networks.

Most work in machine learning focuses on flexible parametric function approximators for CDE.

In our experiments, we use the work of Bishop (1994) and Ambrogioni et al. (2017) , who propose to use a neural network to control the parameters of a mixture density model.

A recent trend in machine learning are latent density models such as cGANs (Mirza & Osindero, 2014) and cVAEs (Sohn et al., 2015) .

Although such methods have been shown successful for estimating distributions of images, the probability density function (PDF) of such models is intractable.

More promising in this sense are normalizing flows (Rezende & Mohamed, 2015; Dinh et al., 2017; Trippe & Turner, 2018) , since they provide the PDF in tractable form.

We employ a neural network controlling the parameters of a normalizing flow as our third CDE model to showcase the empirical efficacy of our regularization approach.

Regularization.

Since neural network based CDE models suffer from severe over-fitting when trained with the MLE objective, they require proper regularization.

Classical regularization of the parameters such as weight decay (Pratt & Hanson, 1989; Krogh & Hertz, 1992; Nowlan & Hinton, 1992) , l 1 /l 2 -penalties (Mackay, 1992; Ng, 2004) and Bayesian priors (Murray & Edwards, 1993; Hinton & Van Camp, 1993) have been shown to work well in the regression and classification setting.

However, in the context of CDE, it is less clear what kind of inductive bias such a regularization imposes on the density estimate.

In contrast, our regularization approach is agnostic w.r.t.

parametrization and is shown to penalize strong variations of the log-density function.

Regularization methods such as dropout are closely related to ensemble methods (Srivastava et al., 2014) .

Thus, they are orthogonal to our work and can be freely combined with noise regularization.

Adding noise during training.

Adding noise during training is a common scheme that has been proposed in various forms.

This includes noise on the neural network weights or activations (Wan et al., 2013; Srivastava et al., 2014; Gal & Uk, 2016) and additive noise on the gradients for scalable MCMC posterior inference (Welling & Teh, 2011; Chen et al., 2014) .

While this line of work corresponds to noise in the parameter space, other research suggests to augment the training data through random and/or adversarial transformations of the data (Sietsma & Dow, 1991; Burges & Sch??lkopf, 1996; Goodfellow et al., 2015; Yuan et al., 2017) .

Our approach transforms the training observations by adding small random perturbations.

While this form of regularization has been studied in the context of regression and classification problems (Holmstrom & Koistinen, 1992a; Webb, 1994; Bishop, 1995; Natarajan et al., 2013; Maaten et al., 2013) , this paper focuses on the regularization of CDE.

In particular, we build on top of the results of Webb (1994) showing that training with noise corresponds to a penalty on strong variations of the log-density and extend previous consistency results for regression of Holmstrom & Koistinen (1992a) to the more general setting of CDE.

To our best knowledge, this is also the first paper to evaluate the empirical efficacy of noise regularization for density estimation.

When considering expressive families of conditional densities, standard maximum likelihood estimation of the model parameters ?? is ill suited.

As can be observed in Figure 1 , simply minimizing the negative log-likelihood of the data leads to severe over-fitting and poor generalization beyond the training data.

Hence, it is necessary to impose additional inductive bias, for instance, in the form of regularization.

Unlike in regression or classification, the form of inductive bias imposed by popular regularization techniques such as weight decay (Krogh & Hertz, 1991; Kuka??ka et al., 2017) is less clear in the CDE setting, where the neural network weights often only indirectly control the probability density through a unconditional density model, e.g., a Gaussian Mixture.

We propose to add noise perturbations to the data points during the optimization of the log-likelihood objective.

This can be understood as replacing the original data points (x i , y i ) by random variables tions K x (?? x ) and K y (?? y ) respectively.

Further, we choose the noise to be zero centered as well as identically and independently distributed among the data dimensions, with standard deviation h:

This can be seen as data augmentation, where "synthetic" data is generated by randomly perturbing the original data.

Since the supply of noise vectors is technically unlimited, an arbitrary large augmented data set can be generated by repetitively sampling data points from D, and adding a random perturbation vector to the respective data point.

This procedure is formalized in Algorithm 1.

For notational brevity, we set Z := X ?? Y, z := (x , y ) and denotef ?? (z) :=f ?? (y|x).

The presented noise regularization approach is agnostic to whether we are concerned with unconditional or conditional MLE.

Thus, the generic notation also allows us to generalize the results to both settings (derived in the remainder of the paper).

Require: D = {z 1 , ..., z n }, noise intensity h Require: number of perturbed samples r, 1: for j = 1 to r do 2:

Select i ??? {1, ..., n} with equal prob.

Draw perturbation ?? ??? K 6:

When considering highly flexible parametric families such as Mixture Density Networks (MDNs) (Bishop, 1994) , the maximum likelihood solution in line 5 of Algorithm 1 is no longer tractable.

In such case, one typically resorts to numerical optimization techniques such as mini-batch gradient descent and variations thereof.

In this context, the generic procedure in Algorithm 1 can be transformed into a simple extensions of mini-batch gradient descent on the MLE objective (see Algorithm 2).

Specifically, each mini-batch is perturbed with i.i.d.

noise before computing the MLE objective function (forward pass) and the respective gradients (backward pass).

Intuitively, the previously presented variable noise can be interpreted as "smearing" the data points during the maximum likelihood estimation.

This alleviates the jaggedness of the density estimate arising from an un-regularized maximum likelihood objective in flexible density classes.

We will now give this intuition a formal foundation, by mathematically analyzing the effect of the noise perturbations.

Before discussing the particular effects of randomly perturbing the data during conditional maximum likelihood estimation, we first analyze noise regularization in a more general case.

Let l ( D) be a loss function over a set of data points D = {z 1 , ..., z n }, which can be partitioned into a sum of losses l(D) = n i=1 l(z i ), corresponding to each data point z i : The expected loss l(z i +??), resulting from adding random perturbations, can be approximated by a second order Taylor expansion around z i .

Using the assumption about ?? in (4), the expected loss an be written as

where l(z i ) is the loss without noise and

???z 2 (z) zi the Hessian of l w.r.t z, evaluated at z i .

Assuming that the noise ?? is small in its magnitude, O(?? 3 ) is negligible.

This effect has been observed earlier by Webb (1994) and Bishop (1994) .

See Appendix A for derivations.

When concerned with maximum likelihood estimation of a conditional densityf ?? (y|x), the loss function coincides with the negative conditional log-likelihood l(y i , x i ) = ??? logf ?? (y i |x i ).

Let the standard deviation of the additive data noise ?? x , ?? y be h x and h y respectively.

Maximum likelihood estimation (MLE) with data noise is equivalent to minimizing the loss

(6) In that, the first term corresponds to the standard MLE objective, while the other two terms constitute a form of smoothness regularization.

The second term in (6) penalizes large negative second derivatives of the conditional log density estimate logf ?? (y|x) w.r.t.

y.

As the MLE objective pushes the density estimate towards high densities and strong concavity in the data points y i , the regularization term counteracts this tendency to over-fit and overall smoothes the fitted distribution.

The third term penalizes large negative second derivatives w.r.t.

the conditional variable x, thereby regularizing the sensitivity of the density estimate to changes in the conditional variable.

The intensity of the noise regularization can be controlled through the variance (h 2 x and h 2 y ) of the random perturbations.

Figure 1 illustrates the effect of the introduced noise regularization scheme on MDN estimates.

Plain maximum likelihood estimation (left) leads to strong over-fitting, resulting in a spiky distribution that generalizes poorly beyond the training data.

In contrast, training with noise regularization (center and right) results in smoother density estimates that are closer to the true conditional density.

We now establish asymptotic consistency results for the proposed noise regularization.

In particular, we show that, under some regularity conditions, concerning integrability and decay of the noise regularization, the solution of Algorithm 1 converges to the asymptotic MLE solution.

a continuous function of z and ??.

Moreover, we assume that the parameter space ?? is compact.

In the classical MLE setting, the idealized loss, corresponding to a (conditional) M-projection of the true data distribution onto the parametric family, reads as

As we typically just have a finite number of samples from p(z), the respective empirical estimat??

is used as training objective.

Note that we now define the loss as function of ??, and, for fixed ??, treat l n (??) as a random variable.

Under some regularity conditions, one can invoke the uniform law of large numbers to show consistency of the empirical ML objective in the sense that sup ??????? |l n (??) ??? l(??)| a.s.

??? ??? ??? 0 (see Appendix B for details).

In case of the presented noise regularization scheme, the maximum likelihood estimation is performed using on the augmented data {z j } rather than the original data {z i }.

For our analysis, we view Algorithm 1 from a slightly different angle.

In fact, the data augmentation procedure of uniformly selecting a data point from {z 1 , ..., z n } and perturbing it with a noise vector drawn from K can be viewed as drawing i.i.d.

samples from a kernel density estimateq

.

Hence, maximum likelihood estimation with variable noise can be understood as 1.

forming a kernel density estimateq (h) n of the training data 2. followed by a (conditional) M-projection ofq (h) n onto the parametric family.

In that, step 2 aims to find the ?? * that minimizes the following objective:

Since (8) is generally intractable, r samples are drawn from the kernel density estimate, forming the following Monte Carlo approximation of (8) which corresponds to the loss in line 5 Algorithm 1:

We are concerned with the consistency of the training procedure in Algorithm 1, similar to the classical MLE consistency result discussed above.

Hence, we need to show that

??? ??? ??? 0 as n, r ??? ???. We begin our argument by decomposing the problem into easier sub-problems.

In particular, the triangle inequality is used to obtain the following upper bound:

Note thatl

n,r (??) is based on samples from the kernel density estimate, which are obtained by adding random noise vectors ?? ??? K(??) to our original training data.

Since we can sample an unlimited amount of such random noise vectors, r can be chosen arbitrarily high.

This allows us to make sup ??????? |l

n (??)| arbitrary small by the uniform law of large numbers.

In order to make sup ??????? |l (h) n (??) ??? l(??)| small in the limit n ??? ???, the sequence of bandwidth parameters h n needs to be chosen appropriately.

Such results can then be combined using a union bound argument.

In the following we outline the steps leading us to the desired results.

In that, the proof methodology is similar to Holmstrom & Koistinen (1992b) .

While they show consistency results for regression with a quadratic loss function, our proof deals with generic and inherently unbounded log-likelihood objectives and thus holds for a much more general class of learning problems.

The full proofs can be found in the Appendix.

Initially, we have to make asymptotic integrability assumptions that ensure that the expectations in l (h) n (??) and l(??) are well-behaved in the limit (see Appendix C for details).

Given respective integrability, we are able to obtain the following proposition.

Proposition 1 Suppose the regularity conditions (28) and (29) are satisfied, and that

almost surely.

In (11) we find conditions on the asymptotic behavior of the smoothing sequence (h n ).

These conditions also give us valuable guidance on how to properly choose the noise intensity in line 4 of Algorithm 1 (see Section 4.3 for discussion).

The result in (12) demonstrates that, under the discussed conditions, replacing the empirical data distribution with a kernel density estimate still results in an asymptotically consistent maximum likelihood objective.

However, as previously discussed, l

n (??) is intractable and, thus, replaced by its sample estimatel

n,r .

Since we can draw an arbitrary amount of samples fromq (h) n , we can approximate l (h) n (??) with arbitrary precision.

Given a fixed data set D of size n > n 0 , this means that lim r?????? sup ???????

n (??) = 0 almost surely, by (29) and the uniform law of large numbers.

Since our original goal was to also show consistency for n ??? ???, this result is combined with Proposition 1, obtaining the following consistency theorem.

Theorem 1 Suppose the regularity conditions (28) and (29) are satisfied, h n fulfills (11) and ?? is compact.

Then, lim

almost surely.

In that, lim used to denote the limit superior ("lim sup") of a sequence.

Training a (conditional) density model with noise regularization means minimizingl

n,r (??) w.r.t.

??.

As result of this optimization, one obtains a parameter vector?? (h) n,r , which we hope is close to the minimizing parameter?? of the ideal objective function l(??).

In the following, we establish asymptotic consistency results, similar to Theorem 1, in the parameter space.

Therefore we first have to formalize the concept of closeness and optimality in the parameter space.

Since a minimizing parameter?? of l(??) may not be unique, we define ?? * = {?? * | l(?? * ) ??? l(??) ????? ??? ??} as the set of global minimizers of l(??), and d(??, ?? * ) = min ?? * ????? * {||?? ??? ?? * || 2 } as the distance of an arbitrary parameter ?? to ?? * .

Based on these definitions, it can be shown that Algorithm 1 is asymptotically consistent in a sense that the minimizer of??

n,r converges almost surely to the set of optimal parameters ?? * .

Theorem 2 Suppose the regularity conditions (28) and (29) are satisfied, h n fulfills (11) and ?? is compact.

For r > 0 and n > n 0 , let??

n,r ??? ?? be a global minimizer of the empirical objectivel

almost surely.

Note that Theorem 2 considers global optimizers, but equivalently holds for compact neighborhoods of a local minimum ?? * (see discussion in Appendix C).

After discussing the properties of noise regularization, we are interested in how to properly choose the noise intensity h, for different training data sets.

Ideally, we would like to choose h so that |l

| is minimized, which is practically not feasible since l(??) is intractable.

Inequality (30) gives as an upper bound on this quantity, suggesting to minimize l 1 distance between the kernel density estimate q (h) n and the data distribution p(z).

This is in turn a well-studied problem in the kernel density estimation literature (see e.g., Devroye & Luc (1987) ).

Unfortunately, general solutions of this problem require knowing p(z) which is not the case in practice.

Under the assumption that p(z) and the kernel function K are Gaussian, the optimal bandwidth can be derived as h = 1.06??n (Silverman, 1986) .

In that,?? denotes the estimated standard deviation of the data, n the number of data points and d the dimensionality of Z. This formula is widely known as the rule of thumb and often used as a heuristic for choosing h.

In addition, the conditions in (11) give us further intuition.

The first condition tells us that h n needs to decay towards zero as n becomes large.

This reflects the general theme in machine learning that the more data is available, the less inductive bias / regularization should be imposed.

The second condition suggests that the bandwidth decay must happen at a rate slower than n ??? 1 d .

For instance, the rule of thumb fulfills these two criteria and thus constitutes a useful guideline for selecting h. However, for highly non-Gaussian data distributions, the respective h n may decay too slowly and a faster decay rate such as n ??? 1 1+d may be appropriate.

This section provides a detailed experimental analysis of the proposed method, aiming to empirically validate the theoretical arguments outlined previously and investigating the practical efficacy of our regularization approach.

In all experiments we use Gaussian pertubations of the data, i.e., K(??) = N (0, I).

Since one of the key features of our noise regularization scheme is that it is agnostic to the choice of model, we evaluate its performance on three different neural network based CDE models: Mixture Density Networks (MDN) (Bishop, 1994) , Kernel Mixture Networks (KMN) (Ambrogioni et al., 2017) and Normalizing Flows Networks (NFN) (Rezende & Mohamed, 2015; Trippe & Turner, 2018) .

In our experiments, we consider both simulated as well as real-world data sets.

In particular, we simulate data from a 4-dimensional Gaussian Mixture (d x = 2, d y = 2) and a Skew-Normal distribution whose parameters are functionally dependent on x (d x = 1, d y = 1).

In terms of real-world data, we use the following three data sources:

Euro Stoxx: Daily stock-market returns of the Euro Stoxx 50 index conditioned on various stock return factors relevant in finance (d x = 14, d y = 1).

UCI datasets: Standard data sets from the UCI machine learning repository (Dua & Graff, 2017) .

In particular, Boston Housing (

The reported scores are test log-likelihoods, averaged over at least 5 random seeds alongside the respective standard deviation.

For further details regarding the data sets and simulated data, we refer to Appendix E. The experiment data and code is available at TODO

We complement the discussion in 4.3 with an empirical investigation of different schedules of h n .

In particular, we compare a) the rule of thumb h n ??? n ??? 1 4+d b) a square root decay schedule h n ??? n ??? 1 1+d c) a constant bandwidth h n = const.

??? (0, ???) and d) no noise regularization, i.e. h n = 0.

Figure 2 plots the respective test log-likelihoods against an increasing training set size n for the two simulated densities Gaussian Mixture and Skew Normal.

First, we observe that bandwidth rates which conform with the decay conditions seem to converge in performance to the non-regularized maximum likelihood estimator (red) as n becomes large.

This reflects the theoretical result of Theorem 1.

Second, a fixed bandwidth across n (green), violating (11), imposes asymptotic bias and thus saturates in performance vastly before its counterparts.

Third, as hypothesized, the relatively slow decay of h n through the rule of thumb works better for data distributions that have larger similarities to a Gaussian, i.e., in our case the Skew Normal distribution.

In contrast, the highly non-Gaussian data from the Gaussian Mixture requires faster decay rates like the square root decay schedule.

Most importantly, noise regularization substantially improves the estimator's performance when only little training data is available.

We now investigate how the proposed noise regularization scheme compares to classical regularization techniques.

In particular, we consider an l 1 and l 2 -penalty on the neural network weights as regularization term, the weight decay technique of Loshchilov & Hutter (2019) 1 , as well a Bayesian neural network (Neal, 2012) trained with variational inference using a Gaussian prior and posterior (Blei et al., 2017) .

First, we study the performance of the regularization techniques on our two simulation benchmarks.

Figure 3 depicts the respective test log-likelihood across different training set sizes.

For each regularization method, the regularization hyper-parameter has been optimized via grid search.

As one would expect, the importance of regularization, i.e., performance difference to un-regularized model, decreases as the amount of training data becomes larger.

The noise regularization scheme Table 1 : Comparison of various regularization methods for three neural network based CDE models across 5 data sets.

We report the test log-likelihood and its respective standard deviation (higher log-likelihood values are better).

yields similar performance across the different CDE models while the other regularizers vary greatly in their performance depending on the different models.

This reflects the fact that noise regularization is agnostic to the parameterization of the CDE model while regularizers in the parameter space are dependent on the internal structure of the model.

Most importantly, noise regularization performs well across all models and sample sizes.

In the great majority of configurations it outperforms the other methods.

Especially when little training data is available, noise regularization ensures a moderate test error while the other approaches mostly fail to do so.

Next, we consider real world data sets.

Since now the amount of data we can use for hyper-parameter selection, training and evaluation is limited, we use 5-fold cross-validation to select the parameters for each regularization method.

The test log-likelihoods, reported in Table 1 , are averages over 3 different train/test splits and 5 seeds each for initializing the neural networks.

The held out test set amounts to 20% of the overall data sets.

Consistent with the results of the simulation study, noise regularization outperforms the other methods across the great majority of data sets and CDE models.

We benchmark neural network based density estimators against state-of-the art CDE approaches.

While neural networks are the obvious choice when a large amount of training data is available, we pose the questions how such estimators compete against well-established non-parametric methods in small data regimes.

In particular, we compare to the three following CDE methods:

Conditional Kernel Density Estimation (CKDE).

Non-parametric method that forms a KDE of both p(x, y) and p(x) to compute its estimate asp(y|x) :=p(x, y)/p(x) (Li & Racine, 2007 Table 2 : Comparison of conditional density estimators across 5 data sets.

Reported is the test loglikelihood and its respective standard deviation (higher log-likelihood values are better).

-Neighborhood kernel density estimation (NKDE).

Non-parametric method that considers only a local subset of training points to form a density estimate.

Semi-parametric estimator that computes the conditional density as linear combination of fixed kernels (Sugiyama & Takeuchi, 2010) .

For the kernel density estimation based methods CKDE and NKDE, we perform bandwidth selection via the rule of thumb (R.O.T) (Silverman, 1982; Sheather & Jones, 1991) and via maximum likelihood leave-one-out cross-validation (CV-ML) (Rudemo, 1982; Hall et al., 1992) .

In case of LSCDE, MDN, KMN and NFN, the respective hyper-parameters are selected via 5-fold cross-validation grid search on the training set.

Note that, in contrast to Section 5.2 which focuses on regularization parameters, the grid search here extends to more hyper-parameters.

The respective test log-likelihood scores are listed in Table 2 .

For the majority of data sets, all three neural network based methods outperform all of the non-and semi-parametric methods.

Perhaps surprisingly, it can be seen that, when properly regularized, neural network based CDE works well even when training data is scarce, such as in case of the Boston Housing data set.

This paper addresses conditional density estimation with high-capacity models.

In particular, we propose to add small random perturbations to the data during training.

We demonstrate that the resulting noise regularization method corresponds to a smoothness regularization and prove its asymptotic consistency.

The experimental results underline the effectiveness of the proposed method, demonstrating that it consistently outperforms other regularization methods across various conditional density models and data sets.

This makes neural network based CDE the preferable method, even when only little training data is available.

While we assess the estimator performance in terms of the test log-likelihood, an interesting question for future research is whether the noise regularization also improves the respective uncertainty estimates for downstream tasks such as safe control and decision making.

Let l(D) be a loss function over a set of data points D = {z 1 , ..., z N }, which can be partitioned into a sum of losses corresponding to each data point x n :

Also, let each z i be perturbed by a random noise vector ?? ??? K(??) with zero mean and i.i.d.

elements, i.e. E ?????K(??) [??] = 0 and E ?????K(??) ?? n ?? j = h 2 I (16) The resulting loss l(z i + ??) can be approximated by a second order Taylor expansion around z i

Assuming that the noise ?? is small in its magnitude, O(?? 3 ) may be neglected.

The expected loss under K(??) follows directly from (17):

Using the assumption about ?? in (16) we can simplify (18) as follows:

In that, l(z i ) is the loss without noise and

we denote the elements of the column vector z.

The objective function corresponding to a conditional M-projection.

The sample equivalent:

Corollary 1 Let ?? be a compact set and andf ?? :

Proof.

The corollary follows directly from the uniform law of large numbers.

Lemma 1 Suppose for some > 0 there exists a constant B ( )

and there exists an n 0 such that for all n > n 0 there exists a constant B ( )

almost surely.

Then, the inequality

where C is a constant holds with probability 1 for all n > n 0 .

Proof of Lemma 1 Using Hoelder's inequality and the nonnegativity of p andq

Employing the regularity conditions (28) and (29) and writing

with probability 1.

Lemma 1 states regularity conditions ensuring that the expectations in l (h) n (??) and l(??) are wellbehaved in the limit.

In particular, (28) and (29) imply uniform and absolute integrability of the loglikelihoods under the respective probability measures induced by p andq (h) n .

Since we are interested in the asymptotic behavior, it is sufficient for (29) to hold for n large enough with probability 1.

Inequality (30) shows that we can make |l (h) n (??) ??? l(??)| small by reducing the l 1 -distance between the true density p and the kernel density estimateq (h) n .

There exists already a vast body of literature, discussing how to properly choose the kernel K and the bandwidth sequence (h n ) so that |q

We employ the results in Devroye (1983) for our purposes, leading us to Proposition 1.

Proof of Proposition 1.

Let A denote the event that ???n 0 ???n > n 0 inequality (30) holds for some constant C ( ) .

From our regularity assumptions it follows that P(A c ) = 0.

Given that A holds, we just have to show that |q

??? ??? ??? 0.

Then, the upper bound in (30) tends to zero and we can conclude our proposition.

For any ?? > 0 let B n denote the event

whereinq (h) n (z) is a kernel density estimate obtained based on n samples from p(z).

Under the conditions in (11) we can apply Theorem 1 of Devroye (1983) , obtaining an upper bound on the probability that (31) does not hold, i.e. ???u, m 0 such that P(B c n ) ??? e ???un for all n > m 0 .

Since we need both A and B n for n ??? ??? to hold, we consider the intersection of the events (A ??? B n ).

Using a union bound argument it follows that ???k 0 such that ???n > k 0 :

Note that we can simply choose k 0 = max{n 0 , m 0 } for this to hold.

Hence,

e u ???1 < ??? and by the Borel-Cantelli lemma we can conclude that lim

In that 1(A) denotes an indicator function which returns 1 if A is true and 0 else.

Next we consider the probability that the convergence in (35) holds for random Z (n) :

Note that we can dP (Z (n) ) move outside of the inner integrals, since Z (n) is independent from I

and ?? (r) .

Hence, we can conclude that (35) also holds, which we denote as event A, with probability 1 for random training data.

From Proposition 1 we know, that

with probability 1.

We denote the event that (38) holds as B. Since P (A c ) = P (B c ) = 0, we can use a union bound argument to show that P (A ??? B) = 1.

From (35) and (33) it follows that for any n > n 0 , lim

with probability 1.

Finally, we combine this result with (38), obtaining that

almost surely, which concludes the proof.

Proof of Theorem 2.

The proof follows the argument used in Theorem 1 of White (1989) .

In the following, we assume that (13) holds.

From Theorem 1 we know that this is the case with probability 1.

Respectively, we only consider realizations of our training data Z (n) and noise samples I (r) , ??

for which the convergence in (13) holds (see proof of Theorem 1 for details on this notation).

For such realization, let (??

From the triangle inequality, it follows that for any > 0 there exists k 0 so that ???k > k 0

given the convergence established in Theorem 1 and the continuity of l in ??.

Next, the result above is extended to

which again holds for k large enough.

This due to (41),

the minimizer of ?? i k ,j k , and ?? i k ,j k (??) ??? l(??) < by Theorem 1.

Because can be made arbitrarily small, l(?? 0 ) ??? l(??) as k ??? ???. Because ?? ??? ?? is arbitrary, ?? 0 must be in ?? * .

In turn, since (n i ) i , (r i,j ) j and (i k ) k , (j k ) k were chosen arbitrarily, every limit point of a sequence (v i k ,j k ) k must be in ?? * .

In the final step, we proof the theorem by contradiction.

Suppose that (14) does not hold.

In this case, there must exist an > 0 and sequences

However, by the previous argument the limit point of the any sequence

where chosen from a set with probability mass of 1, we can conclude our proposition that lim

Discussion of Theorem 2.

Note that, similar to ?? * ,??

n,r does not have to be unique.

In case there are multiple minimizers ofl (h) n,r , we can chose one of them arbitrarily and the proof of the theorem still holds.

Theorem 2 considers global optimizers over a set of parameters ??, which may not be attainable in practical settings.

However, the application of the theorem to the context of local optimization is straightforward when ?? is chosen as a compact neighborhood of a local minimum ?? * of l (Holmstrom & Koistinen, 1992b (Bishop, 1994) .

In particular, the parameters of the unconditional mixture distribution p(y) are outputted by the neural network, which takes the conditional variable x as input.

For our purpose, we employ a Gaussian Mixture Model (GMM) with diagonal covariance matrices as density model.

The conditional density estimatep(y|x) follows as weighted sum of K Gaussian??

wherein w k (x; ??) denote the weight, ?? k (x; ??) the mean and ?? 2 k (x; ??) the variance of the k-th Gaussian component.

All the GMM parameters are governed by the neural network with parameters ?? and input x.

The mixing weights w k (x; ??) must resemble a categorical distribution, i.e. it must hold that K k=1 w k (x; ??) = 1 and w k (x; ??) ??? 0 ???k.

To satisfy the conditions, the softmax linearity is used for the output neurons corresponding to w k (x; ??).

Similarly, the standard deviations ?? k (x) must be positive, which is ensured by a sofplus non-linearity.

Since the component means ?? k (x; ??) are not subject to such restrictions, we use a linear output layer without non-linearity for the respective output neurons.

For the experiments in 5.2 and 5.1, we set K = 10 and use a neural network with two hidden layers of size 32.

While MDNs resemble a purely parametric conditional density model, a closely related approach, the Kernel Mixture Network (KMN), combines both non-parametric and parametric elements (Ambrogioni et al., 2017) .

Similar to MDNs, a mixture density model ofp(y) is combined with a neural network which takes the conditional variable x as an input.

However, the neural network only controls the weights of the mixture components while the component centers and scales are fixed w.r.t.

to x.

For each of the kernel centers, M different scale/bandwidth parameters ?? m are chosen.

As for MDNs, we employ Gaussians as mixture components, wherein the scale parameter directly coincides with the standard deviation.

Let K be the number of kernel centers ?? k and M the number of different kernel scales ?? m .

The KMN conditional density estimate reads as follows:

As previously, the weights w k,m correspond to a softmax function.

The M scale parameters ?? m are learned jointly with the neural network parameters ??.

The centers ?? k are initially chosen by k-means clustering on the {y i } n i=1 in the training data set.

Overall, the KMN model is more restrictive than MDN as the locations and scales of the mixture components are fixed during inference and cannot be controlled by the neural network.

However, due to the reduced flexibility of KMNs, they are less prone to over-fit than MDNs.

For the experiments in 5.2 and 5.1, we set K = 50 and M = 2.

The respective neural network has two hidden layers of size 32.

The Normalizing Flow Network (NFN) is similar to the MDN and KMN in that a neural network takes the conditional variable x as its input and outputs parameters for the distribution over y. For the NFN, the distribution is given by a Normalizing Flow (Rezende & Mohamed, 2015) .

It works by transforming a simple base distribution and an accordingly distributed random variable Z 0 through a series of invertible, parametrized mappings f = f N ??? ?? ?? ?? ??? f 1 into a successively more complex distribution p(f (Z 0 )).

The PDF of samples z N ??? p(f (Z 0 )) can be evaluted using the change-ofvariable formula:

log det ???f n ???z n???1

The Normalizing Flows from Rezende & Mohamed (2015) were introduced in the context of posterior estimation in variational inference.

They are optimized for fast sampling while the likelihood evaluation for externally provided data is comparatively slow.

To make them useful for CDE, we invert the direction of the flows, defining a mapping from the transformed distribution p(Z N ) to the base distribution p(Z 0 ) by settingf ???1 i (z i ) = f i (z i ).

We experimented with three types of flows: planar flows, radial flows as parametrized by Trippe & Turner (2018) and affine flows f ???1 (z) = exp(a)z +b. We have found that one affine flow combined with multiple radial flows performs favourably in most settings.

For the experiments in 5.2 and 5.1, we used a standard Gaussian as the base distribution that is transformed through one affine flow and ten radial flows.

The respective neural network has two hidden layers of size 32.

The data generating process (x, y) ??? p(x, y) resembles a bivariate joint-distribution, wherein x ??? R follows a normal distribution and y ??? R a conditional skew-normal distribution (And??l et al., 1984) .

The parameters (??, ??, ??) of the skew normal distribution are functionally dependent on x. Specifically, the functional dependencies are the following:

??(x) = a * x + b a, b ??? R (47)

??(x) = ?? low + 1 1 + e ???x * (?? high ??? ?? low )

y ??? SkewN ormal ??(x), ??(x), ??(x)

Accordingly, the conditional probability density p(y|x) corresponds to the skew normal density function:

In that, N (??) denotes the density, and ??(??) the cumulative distribution function of the standard normal distribution.

The shape parameter ??(x) controls the skewness and kurtosis of the distribution.

We set ?? low = ???4 and ?? high = 0, giving p(y|x) a negative skewness that decreases as x increases.

This distribution will allow us to evaluate the performance of the density estimators in presence of skewness, a phenomenon that we often observe in financial market variables.

Figure 4a illustrates the conditional skew normal distribution.

The joint distribution p(x, y) follows a Gaussian Mixture Model in R 4 with 5 Gaussian components, i.e. K = 5.

We assume that x ??? R 2 and y ??? R 2 can be factorized, i.e.

p(x, y) =

When x and y can be factorized as in (52), the conditional density p(y|x) can be derived in closed form:

wherein the mixture weights are a function of x:

For details and derivations we refer the interested reader to Guang Sung (2004) and Gilardi et al. (2002) .

The weights w k are sampled from a uniform distribution U (0, 1) and then normalized to sum to one.

The component means are sampled from a spherical Gaussian with zero mean and standard deviation of ?? = 1.5.

The covariance matrices ?? y,k ) and ?? y,k ) are sampled from a Gaussian with mean 1 and standard deviation 0.5, and then projected onto the cone of positive definite matrices.

Since we can hardly visualize a 4-dimensional GMM, Figure 4b depicts a 2-dimensional equivalent, generated with the procedure explained above.

The goal is to predict the conditional probability density of 1-day log-returns, conditioned on 14 explanatory variables.

These conditional variables comprise classical return factors from finance as well as option implied moments.

For details, we refer to Rothfuss et al. (2019) .

Overall, the target variable is one-dimensional, i.e. y ??? Y ??? R, whereas the conditional variable x constitutes a 14-dimensional vector, i.e. x ??? X ??? R 14 .

@highlight

A model-agnostic regularization scheme for neural network-based conditional density estimation.
Deep latent variable models have become a popular model choice due to the scalable learning algorithms introduced by (Kingma & Welling 2013, Rezende et al. 2014).

These approaches maximize a variational lower bound on the intractable log likelihood of the observed data.

Burda et al. (2015) introduced a multi-sample variational bound, IWAE, that is at least as tight as the standard variational lower bound and becomes increasingly tight as the number of samples increases.

Counterintuitively, the typical inference network gradient estimator for the IWAE bound performs poorly as the number of samples increases (Rainforth et al. 2018, Le et al. 2018).

Roeder et a. (2017) propose an improved gradient estimator, however, are unable to show it is unbiased.

We show that it is in fact biased and that the bias can be estimated efficiently with a second application of the reparameterization trick.

The doubly reparameterized gradient (DReG) estimator does not suffer as the number of samples increases, resolving the previously raised issues.

The same idea can be used to improve many recently introduced training techniques for latent variable models.

In particular, we show that this estimator reduces the variance of the IWAE gradient, the reweighted wake-sleep update (RWS) (Bornschein & Bengio 2014), and the jackknife variational inference (JVI) gradient (Nowozin 2018).

Finally, we show that this computationally efficient, drop-in estimator translates to improved performance for all three objectives on several modeling tasks.

Following the influential work by BID20 BID30 , deep generative models with latent variables have been widely used to model data such as natural images BID29 BID14 , speech and music time-series BID8 BID11 BID22 , and video BID1 BID15 BID9 .

The power of these models lies in combining learned nonlinear function approximators with a principled probabilistic approach, resulting in expressive models that can capture complex distributions.

Unfortunately, the nonlinearities that empower these model also make marginalizing the latent variables intractable, rendering direct maximum likelihood training inapplicable.

Instead of directly maximizing the marginal likelihood, a common approach is to maximize a tractable lower bound on the likelihood such as the variational evidence lower bound (ELBO) BID19 BID3 .

The tightness of the bound is determined by the expressiveness of the variational family.

For tractability, a factorized variational family is commonly used, which can cause the learned model to be overly simplistic.

BID5 introduced a multi-sample bound, IWAE, that is at least as tight as the ELBO and becomes increasingly tight as the number of samples increases.

Counterintuitively, although the bound is tighter, BID28 theoretically and empirically showed that the standard inference network gradient estimator for the IWAE bound performs poorly as the number of samples increases due to a diminishing signal-to-noise ratio (SNR).

This motivates the search for novel gradient estimators.

BID31 proposed a lower-variance estimator of the gradient of the IWAE bound.

They speculated that their estimator was unbiased, however, were unable to prove the claim.

We show that it is in fact biased, but that it is possible to construct an unbiased estimator with a second application of the reparameterization trick which we call the IWAE doubly reparameterized gradient (DReG) estimator.

Our estimator is an unbiased, computationally efficient drop-in replacement, and does not suffer as the number of samples increases, resolving the counterintuitive behavior from previous work BID28 .

Furthermore, our insight is applicable to alternative multisample training techniques for latent variable models: reweighted wake-sleep (RWS) BID4 and jackknife variational inference (JVI) BID27 .In this work, we derive DReG estimators for IWAE, RWS, and JVI and demonstrate improved scaling with the number of samples on a simple example.

Then, we evaluate DReG estimators on MNIST generative modeling, Omniglot generative modeling, and MNIST structured prediction tasks.

In all cases, we demonstrate substantial unbiased variance reduction, which translates to improved performance over the original estimators.

Our goal is to learn a latent variable generative model p ?? (x, z) = p ?? (z)p ?? (x|z) where x are observed data and z are continuous latent variables.

The marginal likelihood of the observed data, p ?? (x) = p ?? (x, z) dz, is generally intractable.

Instead, we maximize a variational lower bound on log p ?? (x) such as the ELBO DISPLAYFORM0 where q(z|x) is a variational distribution.

Following the influential work by BID20 BID30 , we consider the amortized inference setting where q ?? (z|x), referred to as the inference network, is a learnable function parameterized by ?? that maps from x to a distribution over z. The tightness of the bound is coupled to the expressiveness of the variational family (i.e., {q ?? } ?? ).

As a result, limited expressivity of {q ?? } ?? , can negatively affect the learned model.

BID5 introduced the importance weighted autoencoder (IWAE) bound which alleviates this coupling DISPLAYFORM1 with z 1:K ??? i q ?? (z i |x).

The IWAE bound reduces to the ELBO when K = 1, is non-decreasing as K increases, and converges to log p ?? (x) as K ??? ??? under mild conditions BID5 .

When q ?? is reparameterizable 1 , the standard gradient estimator of the IWAE bound is DISPLAYFORM2 A single sample estimator of this expectation is typically used as the gradient estimator.

As K increases, the bound becomes increasingly tight, however, BID28 show that the signal-to-noise ratio (SNR) of the inference network gradient estimator goes to 0.

This does not 1 Meaning that we can express zi as z( i, ??), where z is a deterministic, differentiable function and p( i) does not depend on ?? or ??.

This allows gradients to be estimated using the reparameterization trick BID20 BID30 .

Notably, the recently introduced implicit reparameterization trick has expanded the class of distributions for which we can compute reparameterization gradients to include Gaussian mixtures among many others BID12 BID18 BID17 BID10 .

happen for the model parameters (??).

Following up on this work, demonstrate that this deteriorates the performance of learned models on practical problems.

Because the IWAE bound converges to log p ?? (x) (as K ??? ???) regardless of q ?? , ??'s affect on the bound must diminish as K increases.

It may be tempting to conclude that the SNR of the inference network gradient estimator must also decrease as K ??? ???. However, low SNR is a limitation of the gradient estimator, not necessarily of the bound.

Although the magnitude of the gradient converges to 0, if the variance of the gradient estimator decreases more quickly, then the SNR of the gradient estimator need not degrade.

This motivates the search for lower variance inference network gradient estimators.

To derive improved gradient estimators for ??, it is informative to expand the total derivative 2 of the IWAE bound with respect to ?? DISPLAYFORM3 Previously, BID31 found that the first term within the parentheses of Eq. 3 can contribute significant variance to the gradient estimator.

When K = 1, this term analytically vanishes in expectation, so when K > 1 they suggested dropping it.

Below, we abbreviate this estimator as STL.

As we show in Section 6.1, the STL estimator introduces bias when K > 1.

Our insight is that we can estimate the first term within the parentheses of Eq. 3 efficiently with a second application of the reparameterization trick.

To see this, first note that DISPLAYFORM0 so it suffices to focus on one of the K terms.

Because the derivative is a partial derivative ??? ????? , it treats z i = z( i , ??) as a constant, so we can freely change the random variable that the expectation is over to z 1:K .

Now, DISPLAYFORM1 where z ???i = z 1:i???1,i+1:K is the set of z 1:K without z i .

The inner expectation resembles a REIN-FORCE gradient term BID32 , where we interpret wi j wj as the "reward".

Now, we can use the following well-known equivalence between the REINFORCE gradient and the reparameterization trick gradient (See Appendix 8.1 for a derivation) DISPLAYFORM2 This holds even when f depends on ??.

Typically, the reparameterization gradient estimator has lower variance than the REINFORCE gradient estimator because it directly takes advantage of the derivative of f .

Applying the identity from Eq. 5 to the right hand side of Eq. 4 gives DISPLAYFORM3 This last expression can be efficiently estimated with a single Monte Carlo sample.

When z i is not reparameterizable (e.g., the models in BID25 ), we can use a control variate (e.g., 1 K ??? ????? log q ?? (z i |x)).

In both cases, when K = 1, this term vanishes exactly and we recover the estimator proposed in BID31 for the ELBO.

However, when K > 1, there is no reason to believe this term will analytically vanish.

Substituting Eq. 6 into Eq. 3, we obtain a simplification due to cancellation of terms DISPLAYFORM4 We call the algorithm that uses the single sample Monte Carlo estimator of this expression for the inference network gradient the IWAE doubly reparameterized gradient estimator (IWAE-DReG).

This estimator has the property that when q(z|x) is optimal (i.e., q(z|x) = p(z|x)), the estimator vanishes exactly and has zero variance, whereas this does not hold for the standard IWAE gradient estimator.

We provide an asymptotic analysis of the IWAE-DReG estimator in Appendix 8.2.

The conclusion of that analysis is that, in contrast to the standard IWAE gradient estimator, the SNR of the IWAE-DReG estimator exhibits the same scaling behaviour of O( ??? K) for both the generation and inference network gradients (i.e., improving in K).

Now, we review alternative training algorithms for deep generative models and derive their doubly reparameterized versions.

Bornschein & Bengio (2014) introduced RWS, an alternative multi-sample update for latent variable models that uses importance sampling.

Computing the gradient of the log marginal likelihood DISPLAYFORM0 requires samples from p ?? (z|x), which is generally intractable.

We can approximate the gradient with a self-normalized importance sampling estimator DISPLAYFORM1 where z 1:K ??? i q ?? (z i |x).

Interestingly, this is precisely the same as the IWAE gradient of ??, so the RWS update for ?? can be interpreted as maximizing the IWAE lower bound in terms of ??.

Instead of optimizing a joint objective for p and q, RWS optimizes a separate objective for the inference network.

BID4 propose a "wake" update and a "sleep" update for the inference network.

provide empirical support for solely using the wake update for the inference network, so we focus on that update.

The wake update approximately minimizes the KL divergence from p ?? (z|x) to q ?? (z|x).

The gradient of the KL term is DISPLAYFORM2 The wake update of the inference network approximates the intractable expectation by selfnormalized importance sampling DISPLAYFORM3 with z i ??? q ?? (z i |x).

note that this update does not suffer from diminishing SNR as K increases.

However, a downside is that the updates for p and q are not gradients of a unified objective, so could potentially lead to instability or divergence.

The wake update gradient for the inference network (Eq. 8) can be reparameterized DISPLAYFORM0 (9) We call the algorithm that uses the single sample Monte Carlo estimator of this expression as the wake update for the inference network RWS-DReG.Interestingly, the inference network gradient estimator from BID31 can be seen as the sum of the IWAE gradient estimator and the wake update of the inference network (as the wake update minimizes, we add the negative of Eq. 9).

Their positive results motivate further exploration of convex combinations of IWAE-DReG and RWS-DReG DISPLAYFORM1 We refer to the algorithm that uses the single sample Monte Carlo estimator of this expression as DReG(??).

When ?? = 1, this reduces to RWS-DReG, when ?? = 0, this reduces to IWAE-DReG and when ?? = 0.5, this reduces STL.

Alternatively, BID27 reinterprets the IWAE lower bound as a biased estimator for the log marginal likelihood.

He analyzes the bias and introduces a novel family of estimators, Jackknife Variational Inference (JVI), which trade off reduction in bias for increased variance.

This additional flexibility comes at the cost of no longer being a stochastic lower bound on the log marginal likelihood.

The first-order JVI has significantly reduced bias compared to IWAE, which empirically results in a better estimate of the log marginal likelihood with fewer samples BID27 .

For simplicity, we focus on the first-order JVI estimator DISPLAYFORM0 It is straightforward to apply our approach to higher order JVI estimators.

The JVI estimator is a linear combination of K and K ??? 1 sample IWAE estimators, so we can use the doubly reparameterized gradient estimator (Eq. 7) for each term.5 RELATED WORK BID25 introduced a generalized framework of Monte Carlo objectives (MCO).

The log of an unbiased marginal likelihood estimator is a lower bound on the log marginal likelihood by Jensen's inequality.

In this view, the ELBO can be seen as the MCO corresponding to a single importance sample estimator of the marginal likelihood with q ?? as the proposal distribution.

Similarly, IWAE corresponds to the K-sample estimator.

BID24 show that the tightness of an MCO is directly related to the variance of the underlying estimator of the marginal likelihood.

However, BID28 point out issues with gradient estimators of multi-sample lower bounds.

In particular, they show that although the IWAE bound is tighter, the standard IWAE gradient estimator's SNR scales poorly with large numbers of samples, leading to degraded performance.

experimentally investigate this phenomenon and provide empirical evidence of this degradation across multiple tasks.

They find that RWS BID4 does not suffer from this issue and find that it can outperform models trained with the IWAE bound.

We conclude that it is not sufficient to just tighten the bound; it is important to understand the gradient estimators of the tighter bound as well.

Wake-sleep is an alternative approach to fitting deep generative models, first introduced in BID16 as a method for training Hemholtz machines.

It was extended to the multi-sample setting by BID4 and the sequential setting in BID13 .

It has been applied to generative modeling of images BID0 .

To evaluate DReG estimators, we first measure variance and signal-to-noise ratio (SNR) 3 of gradient estimators on a toy example which we can carefully control.

Then, we evaluate gradient variance and model learning on MNIST generative modeling, Omniglot generative modeling, and MNIST structured prediction tasks.

We reimplemented the Gaussian example from BID28 .

Consider the generative model with z ??? N (??, I) and x|z ??? N (z, I) and inference network q ?? (z|x) ??? N (Ax + b, 2 3 I), where ?? = {A, b}. As in BID28 , we sample a set of parameters for the model and inference network close to the optimal parameters (perturbed by zero-mean Gaussian noise with standard deviation 0.01), then estimate the gradient of the inference network parameters for increasing number of samples (K).In addition to signal-to-noise ratio (SNR), we plot the squared bias and variance of the gradient estimators 4 in Fig. 1 .

The bias is computed relative to the expected value of the IWAE gradient estimator.

As a result, although the average of K ELBO gradient estimators is an unbiased estimator of the ELBO gradient, it is a biased gradient estimator of the IWAE objective.

Importantly, SNR does not penalize estimators that are biased, so trivial constant estimators can have infinite SNR.

Thus, it is important to consider additional evaluation measures as well.

As K increases, the SNR of the IWAE-DReG estimator increases, whereas the SNR of the standard gradient estimator of IWAE goes to 0, as previously reported.

Furthermore, we can see the bias present in the STL estimator.

As a check of our implementation, we verified that the observed "bias" for IWAE-DReG was statistically indistinguishable from 0 with a paired t-test.

For the biased estimators (e.g., STL), we could easily reject the null hypothesis with few samples.

Figure 1: Signal-to-noise ratios (SNR), bias squared, and variance of gradient estimators with increasing K over 10 random trials with 1000 measurement samples per trial (mean in bold).

The observed "bias" for IWAE-DReG is not statistically significant under a paired t-test (as expected because IWAE-DReG is unbiased).

IWAE-DReG is unbiased, its SNR increases with K, and it has the lowest variance of the estimators considered here.

Training generative models of the binarized MNIST digits dataset is a standard benchmark task for latent variable models.

For this evaluation, we used the single latent layer architecture from BID5 .

The generative model used 50 Gaussian latent variables with an isotropic prior and passed z through two deterministic layers of 200 tanh units to parameterize factorized Bernoulli outputs.

The inference network passed x through two deterministic layers of 200 tanh units to parameterize a factorized Gaussian distribution over z. Because our interest was in improved gradient estimators and optimization performance, we used the dynamically binarized MNIST dataset, which minimally suffers from overfitting.

We used the standard split of MNIST into train, validation, and test sets.

We trained models with the IWAE gradient, the RWS wake update, and with the JVI estimator.

In all three cases, the doubly reparameterized gradient estimator reduced variance 5 and as a result substantially improved performance FIG1 .

We found similar behavior with different numbers of samples ( FIG2 and Appendix Fig. 8 ).

Interestingly, the biased gradient estimators STL and RWS-DReG perform best on this task with RWSDReG slightly outperforming STL.

As observed in , RWS increasingly outperforms IWAE as K increases.

Finally, we experimented with convex combinations of IWAE-DReG and RWS-DReG (right FIG2 ).

On this dataset, convex combinations that heavily weighted RWS-DReG had the best performance.

However, as we show below, this is task dependent.

Next, we performed the analogous experiment with the dynamically binarized Omniglot dataset using the same model architecture.

Again, we found that the doubly reparameterized gradient estimator reduced variance and as a result improved test performance (Figs. 5 and 6 in the Appendix).

Structured prediction is another common benchmark task for latent variable models.

In this task, our goal is to model a complex observation x given a context c (i.e., model the conditional distribution p(x|c)).

We can use a conditional latent variable model p ?? (x, z|c) = p ?? (x|z, c)p ?? (z|c), however, as before, computing the marginal likelihood is generally intractable.

It is straightforward to adapt the bounds and techniques from the previous section to this problem.

The right plot compares performance as the convex combination between IWAE-DReG and RWS-DReG is varied (Eq. 10).

To highlight differences, we plot the difference between the test IWAE bound and the test IWAE bound IWAE-DReG achieved at that step.

To evaluate our method in this context, we use the standard task of modeling the bottom half of a binarized MNIST digit from the top half.

We use a similar architecture, but now learn a conditional prior distribution p ?? (z|c) where c is the top half of the MNIST digit.

The conditional prior feeds c to two deterministic layers of 200 tanh units to parameterize a factorized Gaussian distribution over z. To model the conditional distribution p ?? (x|c, z), we concatenate z with c and feed it to two deterministic layers of 200 tanh units to parameterize factorized Bernoulli outputs.

As in the previous tasks, the doubly reparameterized gradient estimator improves across all three updates (IWAE, RWS, and JVI; Appendix Fig. 7 ).

However, on this task, the biased estimators (STL and RWS) underperform unbiased IWAE gradient estimators FIG3 .

In particular, RWS becomes unstable later in training.

We suspect that this is because RWS does not directly optimize a consistent objective.

The right plot compares performance as the convex combination between IWAEDReG and RWS-DReG is varied (Eq. 10).

To highlight differences, we plot the difference between the test IWAE bound and the test IWAE bound IWAE-DReG achieved at that step.

In this work, we introduce doubly reparameterized estimators for the updates in IWAE, RWS, and JVI.

We demonstrate that across tasks they provide unbiased variance reduction, which leads to improved performance.

Furthermore, DReG estimators have the same computational cost as the original estimators.

As a result, we recommend that DReG estimators be used instead of the typical gradient estimators.

Variational Sequential Monte Carlo BID24 BID26 and Neural Adapative Sequential Monte Carlo BID13 extend IWAE and RWS to sequential latent variable models, respectively.

It would be interesting to develop DReG estimators for these approaches as well.

We found that a convex combination of IWAE-DReG and RWS-DReG performed best, however, the weighting was task dependent.

In future work, we intend to apply ideas from BID2 to automatically adapt the weighting based on the data.

Finally, the form of the IWAE-DReG estimator (Eq. 7) is surprisingly simple and suggests that there may be a more direct derivation that is applicable to general MCOs.

0 1000 2000 3000 4000 5000Steps ( Steps ( The right plot compares performance as the convex combination between IWAEDReG and RWS-DReG is varied.

To highlight differences, we plot the difference between the test IWAE bound and the test IWAE bound IWAE-DReG achieved at that step.

Given a function f (z, ??), we have DISPLAYFORM0 0 1000 2000 3000 4000 5000Steps ( for a reparameterizable distribution q ?? (z).

To see this, note that DISPLAYFORM1 via the REINFORCE gradient.

On the other hand, DISPLAYFORM2 via the reparameterization trick.

Thus, we conclude that DISPLAYFORM3 from which the identity follows.

At a high level, BID28 show that the expected value of the IWAE gradient of the inference network collapses to zero with rate 1/K, while its standard deviation is only shrinking at a rate of 1/ ??? K. This is the essence of the problem that results in the SNR (expectation divided by standard deviation) of the inference network gradients going to zero at a rate O( DISPLAYFORM0 , worsening with K. In contrast, BID28 show that the generation network gradients scales like O( ??? K), improving with K.Because the IWAE-DReG estimator is unbiased, we cannot hope to change the scaling of the expected value in K, but we can hope to change the scaling of the variance.

In particular, in this subsection, we provide an informal argument, via the delta method, that the standard deviation of IWAE-DReG scales like K ???3/2 , which results in an overall scaling of O( ??? K) for the inference network gradient's SNR (i.e., increasing with K).

Thus, the SNR of the IWAE-DReG estimator improves similarly in K for both inference and generation networks.

We will appeal to the delta method on a two-variable function g : R 2 ??? R. Define the following notation for the partials of g evaluated at the mean of random variables X, Y , DISPLAYFORM1 The delta method approximation of Var(g(X, Y )) is given by (Section 5.

Var(Y ) K 4 Because w i are all mutually independent, we get Var(Y ) = KVar(w i ).

Similarly for Var(X) and u i .

Because the w i and u i are identically distributed and independent for i = j, we have Cov(X, Y ) = KCov(w i , u i ).

All together we can see that Var(g(X, Y )) scales like K ???3 .

Thus, the standard deviation scales like K ???3/2 .

In the main text, we assumed that ?? and ?? were disjoint, however, it can be helpful to share parameters between p and q (e.g., BID11 ).

With the IWAE bound, we differentiate a single objective with respect to both the p and q parameters.

Thus it is straightforward to adapt IWAE and IWAE-DReG to the shared parameter setting.

In this section, we discuss how to deal with shared parameters in RWS.Suppose that both p and q are parameterized by ??.

If we denote the unshared parameters of q by ??, then we can restrict the RWS wake update to only ??.

Alternatively, with a modified RWS wake update, we can derive a single surrogate objective for each scenario such that taking the gradient with respect to ?? results in the proper update.

For clarity, we introduce the following modifier notation for p ?? (x, z i ), q ?? (z i |x), and w i which are functions of ?? and z i = z(??, i ).

We useX to mean X with stopped gradients with respect to z i ,X to mean X with stopped gradients with respect to ?? (but not ?? is not stopped in z(??, i )), andX to mean X with stopped gradients for all variables.

Then, we can use the following surrogate objectives: IWAE: DISPLAYFORM0

@highlight

Doubly reparameterized gradient estimators provide unbiased variance reduction which leads to improved performance.

@highlight

Author experimentally found that the estimator of the existing work(STL) is biased and proposes to reduce the bias to improve the gradient estimator of the ELBO.
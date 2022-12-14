The standard variational lower bounds used to train latent variable models produce biased estimates of most quantities of interest.

We introduce an unbiased estimator of the log marginal likelihood and its gradients for latent variable models based on randomized truncation of infinite series.

If parameterized by an encoder-decoder architecture, the parameters of the encoder can be optimized to minimize its variance of this estimator.

We show that models trained using our estimator give better test-set likelihoods than a standard importance-sampling based approach for the same average computational cost.

This estimator also allows use of latent variable models for tasks where unbiased estimators, rather than marginal likelihood lower bounds, are preferred, such as minimizing reverse KL divergences and estimating score functions.

Latent variable models are powerful tools for constructing highly expressive data distributions and for understanding how high-dimensional observations might possess a simpler representation.

Latent variable models are often framed as probabilistic graphical models, allowing these relationships to be expressed in terms of conditional independence.

Mixture models, probabilistic principal component analysis (Tipping & Bishop, 1999) , hidden Markov models, and latent Dirichlet allocation (Blei et al., 2003) are all examples of powerful latent variable models.

More recently there has been a surge of interest in probabilistic latent variable models that incorporate flexible nonlinear likelihoods via deep neural networks (Kingma & Welling, 2014) .

These models can blend the advantages of highly structured probabilistic priors with the empirical successes of deep learning (Johnson et al., 2016; Luo et al., 2018) .

Moreover, these explicit latent variable models can often yield relatively interpretable representations, in which simple interpolation in the latent space can lead to semantically-meaningful changes in high-dimensional observations (e.g., Higgins et al. (2017) ).

It can be challenging, however, to fit the parameters of a flexible latent variable model, since computing the marginal probability of the data requires integrating out the latent variables in order to maximize the likelihood with respect to the model parameters.

Typical approaches to this problem include the celebrated expectation maximization algorithm (Dempster et al., 1977) , Markov chain Monte Carlo, and the Laplace approximation.

Variational inference generalizes expectation maximization by forming a lower bound on the aforementioned (log) marginal likelihood, using a tractable approximation to the unmanageable posterior over latent variables.

The maximization of this lower bound-rather than the true log marginal likelihood-is often relatively straightforward when using automatic differentiation and Monte Carlo sampling.

However, a lower bound may be ill-suited for tasks such as posterior inference and other situations where there exists an entropy maximization objective; for example in entropy-regularized reinforcement learning (Williams & Peng, 1991; Mnih et al., 2016; Norouzi et al., 2016) which requires minimizing the log probability of the samples under the model.

While there is a long history in Bayesian statistics of estimating the marginal likelihood (e.g., Newton & Raftery (1994) ; Neal (2001)), we often want high-quality estimates of the logarithm of the marginal likelihood, which is better behaved when the data is high dimensional; it is not as susceptible to underflow and it has gradients that are numerically sensible.

However, the log transformation introduces some challenges: Monte Carlo estimation techniques such as importance sampling do not straightforwardly give unbiased estimates of this quantity.

Nevertheless, there has been significant work to construct estimators of the log marginal likelihood in which it is possible to explicitly trade off between bias against computational cost (Burda et al., 2016; Bamler et al., 2017; Nowozin, 2018) .

Unfortunately, while there are asymptotic regimes where the bias of these estimators approaches zero, it is always possible to optimize the parameters to increase this bias to infinity.

In this work, we construct an unbiased estimator of the log marginal likelihood.

Although there is no theoretical guarantee that this estimator has finite variance, we find that it can work well in practice.

We show that this unbiased estimator can train latent variable models to achieve higher test log-likelihood than lower bound estimators at the same expected compute cost.

More importantly, this unbiased estimator allows us to apply latent variable models in situations where these models were previously problematic to optimize with lower bound estimators.

Such applications include latent variable modeling for posterior inference and for reinforcement learning in high-dimensional action spaces, where an ideal model is one that is highly expressive yet efficient to sample from.

Latent variable models (LVMs) describe a distribution over data in terms of a mixture over unobserved quantities.

Let p ?? (x) be a family of probability density (mass) functions on a data space X , indexed by parameters ??.

We will generally refer to this as a "density" for consistency, even when the data should be understood to be discrete; similarly we will use integrals even when the marginalization is over a discrete set.

In a latent variable model, p ?? (x) is defined via a space of latent variables Z, a family of mixing measures on this latent space with density denoted p ?? (z), and a conditional distribution p ?? (x | z).

This conditional distribution is sometimes called an "observation model" or a conditional likelihood.

We will take ?? to parameterize both p ?? (x | z) and p ?? (z) in the service of determining the marginal p ?? (x) via the mixture integral:

This simple formalism allows for a large range of modeling approaches, in which complexity can be baked into the latent variables (as in traditional graphical models), into the conditional likelihood (as in variational autoencoders), or into both (as in structured VAEs).

The downside of this mixing approach is that the integral may be intractable to compute, making it difficult to evaluate p ?? (x)-a quantity often referred to in Bayesian statistics and machine learning as the marginal likelihood or evidence.

Various Monte Carlo techniques have been developed to provide consistent and often unbiased estimators of p ?? (x), but it is usually preferable to work with log p ?? (x) and unbiased estimation of this quantity has, to our knowledge, not been previously studied.

Fitting a parametric distribution to observed data is often framed as the minimization of a difference between the model distribution and the empirical distribution.

The most common difference measure is the forward Kullback-Leibler (KL) divergence; if p data (x) is the empirical distribution and p ?? (x) is a parametric family, then minimizing the KL divergence (D KL ) with respect to ?? is equivalent to Since expectations can be estimated in an unbiased manner using Monte Carlo procedures, simple subsampling of the data enables powerful stochastic optimization techniques, with stochastic gradient descent in particular forming the basis for learning the parameters of many nonlinear models.

However, this requires unbiased estimates of ??? ?? log p ?? (x), which are not available for latent variable models.

Instead, a stochastic lower bound of log p ?? (x) is often used and then differentiated for optimization.

Though many lower bound estimators (Burda et al., 2016; Bamler et al., 2017; Nowozin, 2018) are applicable, we focus on an importance-weighted evidence lower bound (Burda et al., 2016) .

This lower bound is constructed by introducing a proposal distribution q(z; x) and using it to form an importance sampling estimate of the marginal likelihood:

If K samples are drawn from q(z; x) then this provides an unbiased estimate of p ?? (x) and the biased "importance-weighted autoencoder" estimator IWAE K (x) of log p ?? (x) is given by

The special case of K = 1 generates an unbiased estimate of the evidence lower bound (ELBO), which is often used for performing variational inference by stochastic gradient descent.

While the IWAE lower bound acts as a useful replacement of log p ?? (x) in maximum likelihood training, it may not be suitable for other objectives such as those that involve entropy maximization.

We discuss tasks for which a lower bound estimator would be ill-suited in Section 3.4.

There are two properties of IWAE that will allow us to modify it to produce an unbiased estimator: First, it is consistent in the sense that as the number of samples K increases, the expectation of IWAE K (x) converges to log p ?? (x).

Second, it is also monotonically non-decreasing in expectation:

These properties are sufficient to create an unbiased estimator using the Russian roulette estimator.

In order to create an unbiased estimator of the log probability function, we employ the Russian roulette estimator (Kahn, 1955) .

This estimator is used to estimate the sum of infinite series, where evaluating any term in the series almost surely requires only a finite amount of computation.

Intuitively, the Russian roulette estimator relies on a randomized truncation and upweighting of each term to account for the possibility of not computing these terms.

To illustrate the idea, let??? k denote the k-th term of an infinite series.

Assume the partial sum of the series ??? k=1??? k converges to some quantity we wish to obtain.

We can construct a simple estimator by always computing the first term then flipping a coin b ??? Bernoulli(q) to determine whether we stop or continue evaluating the remaining terms.

With probability 1 ??? q, we compute the rest of the series.

By reweighting the remaining future terms by 1 /(1???q), we obtain an unbiased estimator:

To obtain the "Russian roulette" (RR) estimator (Forsythe & Leibler, 1950) , we repeatedly apply this trick to the remaining terms.

In effect, we make the number of terms a random variable K, taking values in 1, 2, . . .

to use in the summation (i.e., the number of successful coin flips) from some distribution with probability mass function p(K) = P(K = K) with support over the positive integers.

With K drawn from p(K), the estimator takes the form:

The equality on the right hand of equation 7 holds so long as (i) P(K ??? k) > 0, ???k > 0, and (ii) the series converges absolutely, i.e., Chen et al. (2019) ; Lemma 3).

This condition ensures that the average of multiple samples will converge to the value of the infinite series by the law of large numbers.

However, the variance of this estimator depends on the choice of p(K) and can potentially be very large or even infinite (McLeish, 2011; Rhee & Glynn, 2015; Beatson & Adams, 2019 We can turn any absolutely convergent series into a telescoping series and apply the Russian roulette randomization to form an unbiased stochastic estimator.

We focus here on the IWAE bound described in Section 2.2.

converges absolutely, we apply equation 7 to construct our estimator, which we call SUMO (Stochastically Unbiased Marginalization Objective).

The detailed derivation of SUMO is in Appendix A.1.

The randomized truncation of the series using the Russian roulette estimator means that this is an unbiased estimator of the log marginal likelihood, regardless of the distribution p(K):

where the expectation is taken over p(K) and q(z; x) (see Algorithm 1 for our exact sampling procedure).

Furthermore, under some conditions, we have

To efficiently optimize a limit, one should choose an estimator to minimize the product of the second moment of the gradient estimates and the expected compute cost per evaluation.

The choice of p(K) effects both the variance and computation cost of our estimator.

Denoting?? := ??? ???? and ???

, the Russian roulette estimator is optimal across a broad family of unbiased randomized truncation estimators if the ??? g k are statistically independent, in which case it has second moment E||??||

2/P(K???k) (Beatson & Adams, 2019) .

While the Algorithm 1 Computing SUMO, an unbiased estimator of log p(x).

Input:

are not in fact strictly independent with our sampling procedure (Algorithm 1), and other estimators within the family may perform better, we justify our choice by showing that E??? i ??? j for i = j converges to zero much faster than E??? 2 k (Appendices A.2 & A.3).

In the following, we assume independence of ??? g k and choose p(K) to minimize the product of compute and variance.

We first show that E||??? .5 ).

This implies the optimal compute-variance product (Rhee & Glynn, 2015; Beatson & Adams, 2019 ) is given by

2 ).

In our case, this gives P(K ??? k) = 1 /k, which results in an estimator with infinite expected computation and no finite bound on variance.

In fact, any p(K) which gives rise to provably finite variance requires a heavier tail than P(K ??? k) = 1 /k and so will have infinite expected computation.

Though we could not theoretically show that our estimator and gradients have finite variance, we empirically find that gradient descent converges -even in the setting of minimizing log probability.

We plot ||??? k || 2 2 for the toy variational inference task used to assess signal to noise ratio in Tucker et al. (2018) and Rainforth et al. (2018b) , and find that they converge faster than 1 k 2 in practice (Appendix A.6).

While this indicates the variance is better than the theoretical bound, an estimator having infinite expected computation cost will always be an issue as it indicates significant probability of sampling arbitrarily large K.

We therefore modify the tail of the sampling distribution such that the estimator has finite expected computation:

We typically choose ?? = 80, which gives an expected computation cost of approximately 5 terms.

One way to improve the RR estimator is to construct it so that some minimum number of terms (denoted here as m) are always computed.

This puts a lower bound on the computational cost, but can potentially lower variance, providing a design space for trading off estimator quality against computational cost.

This corresponds to a choice of RR estimator in which

This computes the sum out to m terms (effectively computing IWAE m ) and then estimates the remaining difference with Russian roulette:

In practice, instead of tuning parameters of p(K), we set m to achieve a given expected computation cost per estimator evaluation for fair comparison with IWAE and related estimators.

The SUMO estimator does not require amortized variational inference, but the use of an "encoder" to produce an approximate posterior q(z; x) has been shown to be a highly effective way to perform rapid feedforward inference in neural latent variable models.

We use ?? to denote the parameters of the encoder q ?? (z; x).

However, the gradients of SUMO with respect to ?? are in expectation zero precisely because SUMO is an unbiased estimator of log p ?? (x), regardless of our choice of q ?? (z; x).

Nevertheless, we would expect the choice of q ?? (z; x) significantly impacts the variance of our estimator.

As such, we optimize q ?? (z; x) to reduce the variance of the SUMO estimator.

We can obtain unbiased gradients in the following way (Ruiz et al., 2016; Tucker et al., 2017) :

Notably, the expectation of this estimator depends on the variance of SUMO, which we have not been able to bound.

In practice, we observe gradients which are sometimes very large.

We apply gradient clipping to the encoder to clip gradients which are excessively large in magnitude.

This helps stabilize the training progress but introduces bias into the encoder gradients.

Fortunately, the encoder itself is merely a tool for variance reduction, and biased gradients with respect to the encoder can still significantly help optimization.

Here we list some applications where an unbiased log probability is useful.

Using SUMO to replace existing lower bound estimates allows latent variable models to be used for new applications where a lower bound is inappropriate.

As latent variable models can be both expressive and efficient to sample from, they are frequently useful in applications where the data is high-dimensional and samples from the model are needed.

Minimizing log p ?? (x).

Some machine learning objectives include terms that seek to increase the entropy of the learned model.

The "reverse KL" objective-often used for training models to perform approximate posterior inferences-minimizes E x???p ?? (x) [log p ?? (x) ??? log ??(x)] where ??(x) is a target density that may only be known up a normalization constant.

Local updates of this form are the basis of the expectation propagation procedure (Minka, 2001) .

This objective has also been used for distilling autoregressive models that are inefficient at sampling (Oord et al., 2018) .

Moreover, reverse KL is connected to the use of entropy-regularized objectives (Williams & Peng, 1991; Ziebart, 2010; Mnih et al., 2016; Norouzi et al., 2016) in decision-making problems, where the goal is to encourage the decision maker toward exploration and prevent it from settling into a local minimum.

Unbiased score function ??? ?? log p ?? (x).

The score function is the gradient of the log-likelihood with respect to the parameters and has uses in estimating the Fisher information matrix and performing stochastic gradient Langevin dynamics (Welling & Teh, 2011) , among other applications.

Of particular note, the REINFORCE gradient estimator (Williams, 1992)-generally applicable for optimizing objectives of the form max ?? E x???p ?? (x) [R(x)]-is estimated using the score function.

This can be replaced with the gradient of SUMO which itself is an estimator of the score func-

where the inner expectation is over the stochasticity of the SUMO estimator.

Such estimators are often used for reward maximization in reinforcement learning where p ?? (x) is a stochastic policy.

There is a long history in Bayesian statistics of marginal likelihood estimation in the service of model selection.

The harmonic mean estimator (Newton & Raftery, 1994) , for example, has a long (and notorious) history as a consistent estimator of the marginal likelihood that may have infinite variance (Murray & Salakhutdinov, 2009 ) and exhibits simulation psuedo-bias (Lenk, 2009 ).

The Chib estimator (Chib, 1995) , the Laplace approximation, and nested sampling (Skilling, 2006) are alternative proposals that can often have better properties (Murray & Salakhutdinov, 2009 ).

Annealed importance sampling (Neal, 2001 ) probably represents the gold standard for marginal likelihood estimation.

These, however, turn into consistent estimators at best when estimating the log marginal probability (Rainforth et al., 2018a) .

Bias removal schemes such as jackknife variational inference (Nowozin, 2018) have been proposed to debias log-evidence estimation, IWAE in particular.

Hierarchical IWAE (Huang et al., 2019 ) uses a joint proposal to induce negative correlation among samples and connects the convergence of variance of the estimator and the convergence of the lower bound.

Russian roulette also has a long history.

It dates back to unpublished work from von Neumann and Ulam, who used it to debias Monte Carlo methods for matrix inversion (Forsythe & Leibler, 1950) and particle transport problems (Kahn, 1955) .

It has gained popularity in statistical physics (Spanier & Gelbard, 1969; Kuti, 1982; Wagner, 1987) , for unbiased ray tracing in graphics and rendering (Arvo & Kirk, 1990) , and for a number of estimation problems in the statistics community (Wei & Murray, 2017; Lyne et al., 2015; Rychlik, 1990; Jacob & Thiery, 2015; Jacob et al., 2017) .

It has also been independently rediscovered many times (Fearnhead et al., 2008; McLeish, 2011; Rhee & Glynn, 2012; Tallec & Ollivier, 2017) .

The use of Russian roulette estimation in deep learning and generative modeling applications has been gaining traction in recent years.

It has been used to solve short-term bias in optimization problems (Tallec & Ollivier, 2017; Beatson & Adams, 2019) .

Wei & Murray (2017) Though we extend latent variable models to applications that require unbiased estimates of log probability and benefit from efficient sampling, an interesting family of models already fulfill these requirements.

Normalizing flows (Rezende & Mohamed, 2015; Dinh et al., 2017) offer exact log probability and certain models have been proven to be universal density estimators (Huang et al., 2018) .

However, these models often require restrictive architectural choices with no dimensionalityreduction capabilities, and make use of many more parameters to scale up (Kingma & Dhariwal, 2018) than alternative generative models.

Discrete variable versions of these models are still in their infancy and make use of biased gradients (Tran et al., 2019; Hoogeboom et al., 2019) , whereas latent variable models naturally extend to discrete observations.

We first compare the performance of SUMO when used as a replacement to IWAE with the same expected cost on density modeling tasks.

We make use of two benchmark datasets: dynamically binarized MNIST (LeCun et al., 1998) and binarized OMNIGLOT (Lake et al., 2015) .

We use the same neural network architecture as IWAE (Burda et al., 2016) .

The prior p(z) is a 50-dimensional standard Gaussian distribution.

The conditional distributions p(x i |z) are independent Bernoulli, with the decoder parameterized by two hidden layers, each with 200 tanh units.

The approximate posterior q(z; x) is also a 50-dimensional Gaussian distribution with diagonal covariance, whose mean and variance are both parameterized by two hidden layers with 200 tanh units.

We reimplemented and tuned IWAE, obtaining strong baseline results which are better than those previously reported.

We then used the same hyperparameters to train with the SUMO estimator.

We find clipping very large gradients can help performance, as large gradients may be infrequently sampled.

This introduces a small amount of bias into the gradients while reducing variance, but can nevertheless help achieve faster convergence and should still result in a less-biased estimator.

A posthoc study of the effect on final test performance as a function of this bias-variance tradeoff mechanism is discussed in Appendix A.7.

We note that gradient clipping is only done for the density modeling experiments.

The averaged test log-likelihoods and standard deviations over 3 runs are summarized in Table 1 .

To be consistent with existing literature, we evaluate our model using IWAE with 5000 samples.

In all the cases, SUMO achieves slightly better performance than IWAE with the same expected cost.

We also bold the results that are statistically insignificant from the best performing model according to an unpaired t-test with significance level 0.05.

However, we do see diminishing returns as we increase k, suggesting that as we increase compute, the variance of our estimator may impact performance more than the bias of IWAE.

We move on to our first task for which a lower bound estimate of log probability would not suffice.

The reverse KL objective is useful when we have access to a (possibly unnormalized) target distribution but no efficient sampling algorithm.

A major problem with fitting latent variables models to this objective is the presence of an entropy maximization term, effectively a minimization of log p ?? (x).

Estimating this log marginal probability with a lower bound estimator could result in optimizing ?? to maximize the bias of the estimator instead of the true objective.

Our experiments demonstrate that this causes IWAE to often fail to optimize the objective unless we use a large amount of computation. (Burda et al., 2016) 85 Figure 1: We trained latent variable models for posterior inference, which requires minimizing log probability under the model.

Training with IWAE leads to optimizing for the bias while leaving the true model in an unstable state, whereas training with SUMO-though noisy-leads to convergence.

Modifying IWAE.

The bias of the IWAE estimator can be interpreted as the KL between an importance-weighted approximate posterior q IW (z; x) implicitly defined by the encoder and the true posterior p(z|x) (Domke & Sheldon, 2018) .

Both the encoder and decoder parameters can therefore affect this bias.

In practice, we find that the encoder optimization proceeds at a faster timescale than the decoder optimization: i.e., the encoder can match q IW (z; x) to the decoder's p(z|x) more quickly than the latter can match an objective.

For this reason, we train the encoder to reduce bias and use a minimax training objective

Though this is still a lower bound with unbounded bias, it makes for a stronger baseline than optimizing q(z; x) in the same direction as p(x, z).

We find that this approach can work well in practice when k is set sufficiently high.

We choose a "funnel" target distribution (Figure 1 ) similar to the distribution used as a benchmark for inference in Neal et al. (2003), where p * has support in R 2 and is defined p * (x 1 , x 2 ) = N (x 1 ; 0, 1.35 2 )N (x 2 ; 0, e 2x1 ) We use neural networks with one hidden layer of 200 hidden units and tanh activations for both the encoder and decoder networks.

We use 20 latent variables, with p(z), p ?? (x|z), and q ?? (z; x) all being Gaussian distributed.

Figure 2 shows the learning curves when using IWAE and SUMO.

Unless k is set very large, IWAE will at some point start optimizing the bias instead of the actual objective.

The reverse KL is a non-negative quantity, so any estimate significantly below zero can be attributed to the unbounded bias.

On the other hand, SUMO Figure 3: Latent variable policies allow faster exploration than autoregressive policy models, while being more expressive than an independent policy.

SUMO works well with entropy regularization, whereas IWAE is unstable and converges to similar performance as the non-latent variable model.

correctly optimizes for the objective even with a small expected cost.

Increasing the expected cost k for SUMO reduces variance.

For the same expected cost, SUMO can optimize the true objective but IWAE cannot.

We also found that if k is set sufficiently large, then IWAE can work when we train using the minimax objective in equation 15, suggesting that a sufficiently debiased estimator can also work in practice.

However, this requires much more compute and likely does not scale compared to SUMO.

We also visualize the contours of the resulting models in Figure 1 .

For IWAE, we visualize the model a few iterations before it reaches numerical instability.

Let us now consider the problem of finding the maximum of a non-differentiable function, a special case of reinforcement learning without an interacting environment.

Variational optimization (Staines & Barber, 2012) can be used to reformulate this as the optimization of a parametric distribution,

which is now a differentiable function with respect to the parameters ??, whose gradients can be estimated using a combination of the REINFORCE gradient estimator and the SUMO estimator (equation 13).

Furthermore, entropy regularized reinforcement learning-where we maximize R(x) + ??H(p ?? ) with H(p ?? ) being the entropy of p ?? (x)-encourages exploration and is inherently related to minimizing a reverse KL objective with the target being an exponentiated reward (Norouzi et al., 2016) .

For concreteness, we focus on the problem of quadratic pseudo-Boolean optimization (QPBO) where the objective is to maximize

where

??? {0, 1} are binary variables.

Without further assumptions, QPBO is NPhard (Boros & Hammer, 2002) .

As there exist complex dependencies between the binary variables and optimization of equation 16 requires sampling from the policy distribution p ?? (x), a model that is both expressive and allows efficient sampling would be ideal.

For this reason, we motivate the use of latent variable models with independent conditional distributions, which we trained using the SUMO objective.

Our baselines are an autoregressive policy, which captures dependencies but for which sampling must be performed sequentially, and an independent policy, which is easy to sample from but captures no dependencies.

We note that Haarnoja et al. (2018) also argued for latent variable policies in favor of learning diverse strategies but ultimately had to make use of normalizing flows which did not require marginalization.

We constructed one problem instance for each d ??? {100, 500}, which we note are already intractable for exact optimization.

For each instance, we randomly sampled the weights w i and w ij uniformly from the interval [???1, 1].

Figure 3 shows the performance of each policy model.

In general, the independent policy is quick to converge to a local minima and is unable to explore different regions, whereas more complex models have a better grasp of the "frontier" of reward distributions during optimization.

The autoregressive works well overall but is much slower to train due to its sequential sampling procedure; with d = 500, it is 19.2?? slower than training with SUMO.

Surprisingly, we find that estimating the REINFORCE gradient with IWAE results in decent performance when no entropy regularization is present.

With entropy regularization, all policies improve significantly; however, training with IWAE in this setting results in performance similar to the independent model.

On the other hand, SUMO works with both REINFORCE gradient estimation and entropy regularization, albeit at the cost of slower convergence due to variance.

We introduced SUMO, a new unbiased estimator of the log probability for latent variable models, and demonstrated tasks for which this estimator performs better than standard lower bounds.

Specifically, we investigated applications involving entropy maximization where a lower bound performs poorly, but our unbiased estimator can train properly with relatively smaller amount of compute.

In the future, we plan to investigate new families of gradient-based optimizers which can handle heavy-tailed stochastic gradients.

It may also be fruitful to investigate the use of convex combination of consistent estimators within the SUMO approach, as any convex combination is unbiased, or to apply variance reduction methods to increase stability of training with SUMO.

Brian D Ziebart.

Modeling purposeful adaptive behavior with the principle of maximum causal entropy.

PhD thesis, figshare, 2010.

A APPENDIX

where z 1 , .., z k are sampled independently from q(z; x).

And we define the k-th term of the infinite

.

Using the properties of IWAE in equation 6, we have??? k (x) ??? 0, and

which means the series converges absolutely.

This is a sufficient condition for finite expectation of the Russian roulette estimator (Chen et al. (2019) ; Lemma 3).

Applying equation 7 to the series:

Let

, Hence our estimator is constructed:

And it can be easily seen from equation 22 and equation 23 that SUMO is an unbiased estimator of the log marginal likelihood:

A.2 CONVERGENCE OF ??? k

We follow the analysis of JVI (Nowozin, 2018) , which applied the delta method for moments to show the asymptotic results on the bias and variance of IWAE k both at a rate of O( and we define Y k := 1 k k i=1 w i as the sample mean and we have E[

We note that we rely on ||Y k ??? ??|| < 1 for this power series to converge.

This condition was implicitly assumed, but not explicitly noted, in (Nowozin, 2018) .

This condition will hold for sufficiently large k so long as the moments of w i exist: one could bound the probability ||Y k ?????|| ??? 1 by Chebyshev's inequality or by the Central Limit Theorem.

We use the central moments

Expanding Eq. 28 to order two gives

Since we use cumulative sum to compute Y k and Y k+1 , we obtain

We note that

Without loss of generality, suppose j ??? k + 1,

For clarity, let C k = Y k ??? ?? be the zero-mean random variable.

Nowozin (2018) gives the relations

Expanding both the sums inside the brackets to order two:

We will proceed by bounding each of the terms (1), (2), (3), (4).

First, we decompose C j .

Let

We know that B k,j is independent of C k and

Now we show that (1) is zero:

We now investigate (2):

We now show that (3) is zero:

Finally, we investigate (4):

Using the relation in equation 36, we have

Assume that ??? ?? SUMO is bounded: it is sufficient that ??? ?? IWAE 1 is bounded and that the sampling probabilities are chosen such that the partial sums of

by the dominated convergence theorem, as long as SUMO is everywhere differentiable, which is satisfied by all of our experiments.

If ReLU neural networks are to be used, one may be able to show the same property using Theorem 5 of Bikowski et al. (2018) , assuming finite higher moments and Lipschitz constant.

The IWAE log likelihood estimate is:

The gradient of this with respect to ??, where ?? is either ?? or ??, is

We abbreviate w i := p ?? (x,zi) q ?? (zi|x) , and ?? i = dwi d?? .

In both ?? = ?? and ?? = ?? cases, it suffices to treat the w i and ?? i as i.i.d.

random variables with finite variance and expectation.

Being a likelihood ratio, w i could be ill behaved when the importance sampling distribution q ?? (z i |x) is is particularly mismatched from the true posterior p(z i |x) = p ?? (x,zi E z???p(z) p ?? (x,z) .

However, the analysis from IWAE (Burda et al., 2016) requires assuming that the likelihood ratios w i = p ?? (x,zi) q ?? (zi|x) are bounded, and we adopt this assumption.

Reasoning about when this assumption holds, and the behavior of IWAE-like estimators when it does not, is an interesting area for future work.

Consider the differences between two gradients: we label ??? g as follows:

We have:

We again let Y k denote the kth sample mean 1 k i w i .

Then:

The sample means Y k and?? k have finite expectation and variance.

The variance vanishes as k ??? ??? (but the expectation does not change).

The second term vanishes at a rate strictly faster than 1 k 2 : the variance of ?? k goes to zero as k ??? ???. But the first term does not: ?? k is a biased estimator of ?? ??? so E?? k does change with k, but it does not necessarily go to zero: A.7 BIAS-VARIANCE TRADEOFF VIA GRADIENT CLIPPING While SUMO is unbiased, its variance is extremely high or potentially infinite.

This property leads to poor performance compared to lower bound estimates such as IWAE when maximizing loglikelihood.

In order to obtain models with competitive log-likelihood values, we can make use of gradient clipping.

This allows us to ignore rare gradient samples with extremely large values due to the heavy-tailed nature of its distribution.

Gradient clipping introduces bias in favor of reduced variance.

Figure 6 shows how the performance changes as a function of the clipping value, and more importantly, the percentage of clipped gradients.

As shown, neither full clipping and no clipping are desirable.

We performed this experiment after reporting the results in Table 1 , so this grid search was not used to tune hyperparameter for our experiments.

As bias is introduced, we do not use gradient clipping for entropy maximization or policy gradient (REINFORCE).

Figure 6 : Test negative log-likelihood against the gradient clipping norm and clipping percentage, when training with SUMO (k=15).

In density modeling experiments, all the models are trained using a batch size of 100 and an Amsgrad optimizer (Reddi et al., 2018) with parameters lr = 0.001, ?? 1 = 0.9, ?? 2 = 0.999 and = 10 ???4 .

The learning rate is reduced by factor 0.8 with a patience of 50 epochs.

We use gradient norm scaling in both the inference and generative networks.

We train SUMO using the same architecture and hyperparameters as IWAE except the gradient clipping norm.

We set the gradient norm to 5000 for encoder and {20, 40, 60} for decoder in SUMO.

For IWAE, the gradient norm is fixed to 10 in all the experiments.

We report the performance of models with early stopping if no improvements have been observed for 300 epochs on the validation set.

We add additional plots of the test NLL against the norm and percentage of gradients clipped for the decoder in Figure 6 .

The plot is based on MNIST with expected number of compute k = 15.

Gradient clipping was not used in the other experiments except the density modeling ones, where it can be used as a tool to obtain a better bias-variance trade-off.

A.8.1 REVERSE KL AND COMBINATORIAL OPTIMIZATION These two tasks use the same encoder and decoder architecture: one hidden layer with tanh nonlinearities and 200 hidden units.

We set the latent state to be of size 20.

The prior is a standard Gaussian with diagonal covariance, while the encoder distribution is a Gaussian with parameterized diagonal covariance.

For reverse KL, we used independent Gaussian conditional likelihoods for p(x|z), while for combinatorial optimization we used independent Bernoulli conditional distribu-tions.

We found it helps stablize training for both IWAE and SUMO to remove momentum and used RMSprop with learning rate 0.00005 and epsilon 1e-3 for fitting reverse KL.

We used Adam with learning rate 0.001 and epsilon 1e-3, plus standard hyperparameters for the combinatorial optimization problems.

SUMO used an expected compute of 15 terms, with m = 5 and the tail-modified telescoping Zeta distribution.

<|TLDR|>

@highlight

We create an unbiased estimator for the log probability of latent variable models, extending such models to a larger scope of applications.
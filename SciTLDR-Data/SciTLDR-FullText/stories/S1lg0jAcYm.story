To backpropagate the gradients through stochastic binary layers, we propose the augment-REINFORCE-merge (ARM) estimator that is unbiased, exhibits low variance, and has low computational complexity.

Exploiting variable augmentation, REINFORCE, and reparameterization, the ARM estimator achieves adaptive variance reduction for Monte Carlo integration by merging two expectations via common random numbers.

The variance-reduction mechanism of the ARM estimator can also be attributed to either antithetic sampling in an augmented space, or the use of an optimal anti-symmetric "self-control" baseline function together with the REINFORCE estimator in that augmented space.

Experimental results show the ARM estimator provides state-of-the-art performance in auto-encoding variational inference and maximum likelihood estimation, for discrete latent variable models with one or multiple stochastic binary layers.

Python code for reproducible research is publicly available.

Given a function f (z) of a random variable z = (z 1 , . . .

, z V )T , which follows a distribution q φ (z) parameterized by φ, there has been significant recent interest in estimating φ to maximize (or minimize) the expectation of f (z) with respect to z ∼ q φ (z), expressed as DISPLAYFORM0 In particular, this expectation objective appears in both maximizing the evidence lower bound (ELBO) for variational inference BID12 and approximately maximizing the log marginal likelihood of a hierarchal Bayesian model BID1 , two fundamental problems in statistical inference.

To maximize (1), if ∇ z f (z) is tractable to compute and z ∼ q φ (z) can be generated via reparameterization as z = T φ ( ), ∼ p( ), where are random noises and T φ (·) denotes a deterministic transform parameterized by φ, then one may apply the reparameterization trick BID14 BID27 to compute the gradient as DISPLAYFORM1 This trick, however, is often inapplicable to discrete random variables, as widely used to construct discrete latent variable models such as sigmoid belief networks BID22 BID31 .To maximize (1) for discrete z, using the score function ∇ φ log q φ (z) = ∇ φ q φ (z)/q φ (z), one may compute ∇ φ E(φ) via REINFORCE BID38 as its high Monte-Carlo-integration variance often limits its use in practice.

Note that if f (z) depends on φ, then we assume it is true that E z∼q φ (z) [∇ φ f (z)] = 0.

For example, in variational inference, we need to maximize the ELBO as E z∼q φ (z) [f (z)], where f (z) = log[p(x | z)p(z)/q φ (z)].

In this case, although f (z) depends on φ, as E z∼q φ (z) [∇ φ log q φ (z)] = ∇ φ q φ (z)dz = ∇ φ q φ (z)dz = 0, we have E z∼q φ (z) [∇ φ f (z)] = 0.To address the high-variance issue, one may introduce an appropriate baseline (a.k.a.

control variate) to reduce the variance of REINFORCE BID24 BID26 BID19 BID9 BID20 BID29 BID21 .

Alternatively, one may first relax the discrete random variables with continuous ones and then apply the reparameterization trick to estimate the gradients, which reduces the variance of Monte Carlo integration at the expense of introducing bias BID11 .

Combining both REINFORCE and the continuous relaxation of discrete random variables, REBAR of BID35 and RELAX of BID7 both aim to produce a low-variance and unbiased gradient estimator by introducing a continuous relaxation based baseline function, whose parameters, however, need to be estimated at each mini-batch by minimizing the sample variance of the estimator with stochastic gradient descent (SGD).

Estimating the baseline parameters often clearly increases the computation.

Moreover, the potential conflict, between minimizing the sample variance of the gradient estimate and maximizing the expectation objective, could slow down or even prevent convergence and increase the risk of overfitting.

Another interesting variance-control idea applicable to discrete latent variables is using local expectation gradients, which estimates the gradients based on REINFORCE, by performing Monte Carlo integration using a single global sample together with exact integration of the local variable for each latent dimension BID34 .Distinct from the usual idea of introducing baseline functions and optimizing their parameters to reduce the estimation variance of REINFORCE, we propose the augment-REINFORCE-merge (ARM) estimator, a novel unbiased and low-variance gradient estimator for binary latent variables that is also simple to implement and has low computational complexity.

We show by rewriting the expectation with respect to Bernoulli random variables as one with respect to augmented exponential random variables, and then expressing the gradient as an expectation via REINFORCE, one can derive the ARM estimator in the augmented space with the assistance of appropriate reparameterization.

In particular, in the augmented space, one can derive the ARM estimator by using either the strategy of sharing common random numbers between two expectations, or the strategy of applying antithetic sampling.

Both strategies, as detailedly discussed in BID23 , can be used to explain why the ARM estimator is unbiased and could lead to significant variance reduction.

Moreover, we show that the ARM estimator can be considered as improving the REINFORCE estimator in an augmented space by introducing an optimal baseline function subject to an anti-symmetric constraint; this baseline function can be considered as a "self-control" one, as it exploits the function f itself and correlated random noises for variance reduction, and adds no extra parameters to learn.

This "self-control" feature makes the ARM estimator distinct from both REBAR and RELAX, which rely on minimizing the sample variance of the gradient estimate to optimize the baseline function.

We perform experiments on a representative toy optimization problem and both auto-encoding variational inference and maximum likelihood estimation for discrete latent variable models, with one or multiple binary stochastic layers.

Our extensive experiments show that the ARM estimator is unbiased, exhibits low variance, converges fast, has low computation, and provides state-of-the-art out-of-sample prediction performance for discrete latent variable models, suggesting the effectiveness of using the ARM estimator for gradient backpropagation through stochastic binary layers.

Python code for reproducible research is available at https://github.com/mingzhang-yin/ARM-gradient.

In this section, we first present the key theorem of the paper, and then provide its derivation.

With this theorem, we summarize ARM gradient ascent for multivariate binary latent variables in Algorithm 1, as shown in Appendix A. Let us denote σ(φ) = e φ /(1 + e φ ) as the sigmoid function and 1 [·] as an indicator function that equals to one if the argument is true and zero otherwise.

Theorem 1 (ARM).

For a vector of V binary random variables z = (z 1 , . . .

, z V ) T , the gradient of DISPLAYFORM0 expectations in (8) and applying reparameterization, we conclude the proof for (4) of Theorem 1 with DISPLAYFORM1 Alternatively, instead of generalizing the univariate ARM gradient as in (8) and FORMULA3 , we can first do a multivariate generalization of the univariate AR gradient in (6) as DISPLAYFORM2 The same as the derivation of the univariate ARM estimator, here we can arrive at (4) from (10) by either adding an antithetic sampling step, or subtracting the AR estimator by a baseline function as DISPLAYFORM3 which has zero mean, satisfies b(u) = −b(1 − u), and is distinct from previously proposed baselines in taking advantage of "self-control" for variance reduction and adding no extra parameters to learn.

For the univariate case, we show below that the ARM estimator has smaller worst-case variance than REINFORCE does.

The proof is deferred to Appendix C. Proposition 2 (Univariate gradient variance).

For the objective function DISPLAYFORM0 In the general setting, with DISPLAYFORM1 Uniform(0, 1), we define the ARM estimate of ∇ φv E(φ) with K Monte Carlo samples, denoted as g ARM K ,v , and the AR estimate with 2K Monte Carlo samples, denoted as g AR 2K ,v , using DISPLAYFORM2 where DISPLAYFORM3 .

Similar to the analysis in BID23 , the amount of variance reduction brought by the ARM estimator can be reflected by the ratio as We show below that under the anti-symmetric constraint b(u) = −b(1 − u), which implies that DISPLAYFORM4 DISPLAYFORM5 ] is a vector of zeros, Equation FORMULA0 is the optimal baseline function to be subtracted from the AR estimator for variance reduction.

The proof is deferred to Appendix C.Proposition 4 (Optimal anti-symmetric baseline).

For the gradient of E z∼q φ (z) [f (z)], the optimal anti-symmetric baseline function to be subtracted from the AR estimator g AR (u) = f (1 [u<σ(φ)] )(1 − 2u), which minimizes the variance of Monte Carlo integration, can be expressed as arg min DISPLAYFORM6 where DISPLAYFORM7 for all v} is the set of all zero-mean anti-symmetric baseline functions.

Note the optimal baseline function shown in (13) is exactly the same as (11), which is subtracted from the AR estimator in (10) to arrive at the ARM estimator in (4).

Corollary 5 (Lower variance than constant baseline).

The optimal anti-symmetric baseline function for the AR estimator, as shown in (13) and also in (11), leads to lower estimation variance than any constant based baseline function as DISPLAYFORM8 , where c v is a dimension-specific constant whose value can be optimized for variance reduction.

A latent variable model with multiple stochastic hidden layers can be constructed as DISPLAYFORM0 whose joint likelihood given the distribution parameters θ 0:T = {θ 0 , . . .

, θ T } is expressed as DISPLAYFORM1 In comparison to deterministic feedforward neural networks, stochastic ones can represent complex distributions and show natural resistance to overfitting BID22 BID31 BID32 BID25 BID9 BID32 .

However, the training of the network, especially if there are stochastic discrete layers, is often much more challenging.

Below we show for both auto-encoding variational inference and maximum likelihood estimation, how to apply the ARM estimator for gradient backpropagation in stochastic binary networks.

For auto-encoding variational inference BID14 BID27 , we construct a variational distribution as DISPLAYFORM0 with which the ELBO can be expressed as DISPLAYFORM1 DISPLAYFORM2 Proposition 6 (ARM backpropagation).

For a stochastic binary network with T binary stochastic hidden layers, constructing a variational auto-encoder (VAE) defined with b 0 = x and DISPLAYFORM3 for t = 1, . . .

, T , the gradient of the ELBO with respect to w t can be expressed as DISPLAYFORM4 The gradient presented in (19) can be estimated with a single Monte Carlo sample aŝ DISPLAYFORM5 where b DISPLAYFORM6 The proof of Proposition 6 is provided in Appendix C. Suppose the computation complexity of vanilla REINFORCE for a stochastic hidden layer is O(1), which involves a single evaluation of the function f and gradient backpropagation as ∇ wt T wt (b t−1 ), then for a T -stochastic-hidden-layer network, the computation complexity of vanilla REINFORCE is O(T ).

By contrast, if evaluating f is much less expensive in computation than gradient backpropagation, then the ARM estimator also has O(T ) complexity, whereas if evaluating f dominates gradient backpropagation in computation, then its worst-case complexity is O(2T ).

For maximum likelihood estimation, the log marginal likelihood can be expressed as DISPLAYFORM0 Generalizing Proposition 6 leads to the following proposition.

Proposition 7.

For a stochastic binary network defined as DISPLAYFORM1 the gradient of the lower bound in (21) with respect to θ t can be expressed as DISPLAYFORM2

To illustrate the working mechanism of the ARM estimator, related to BID35 and BID7 , we consider learning φ to maximize DISPLAYFORM0 , where p 0 ∈ {0.49, 0.499, 0.501, 0.51}.The optimal solution is σ(φ) = 1 [p0<0.5] .

The closer p 0 is to 0.5, the more challenging the optimization becomes.

We compare both the AR and ARM estimators to the true gradient as DISPLAYFORM1 and three previously proposed unbiased estimators, including REINFORCE, REBAR BID35 , and RELAX BID7 .

Since RELAX is closely related to REBAR in introducing stochastically estimated control variates to improve REINFORCE, and clearly outperforms RE-BAR in our experiments for this toy problem (as also shown in BID7 for p 0 = 0.49), we omit the results of REBAR for brevity.

With a single random sample u ∼ Uniform(0, 1) for Monte Carlo integration, the REINFORCE and AR gradients can be expressed as DISPLAYFORM2 while the ARM gradient can be expressed as DISPLAYFORM3 See BID7 for the details on RELAX.As shown in FIG0 , the REINFORCE gradients have large variances.

Consequently, a REINFORCE based gradient ascent algorithm may diverge if the gradient ascent stepsize is not sufficiently small.

For example, when p 0 = 0.501, the optimal value for the Bernoulli probability σ(φ) is 0, but the algorithm with 0.1 as the stepsize infers it to be close to 1 at the end of 2000 iterations of a random trial.

The AR estimator behaves similarly as REINFORCE does.

By contrast, both RELAX and ARM exhibit clearly lower estimation variance.

It is interesting to note that the trace plots of the estimated probability σ(φ) with the univariate ARM estimator almost exactly match these with the DISPLAYFORM4 2 ] via gradient ascent, where p0 ∈ {0.49, 0.499, 0.501, 0.51}; the optimal solution is σ(φ) = 1(p0 < 0.5).

Shown in Rows 1 and 2 are the trace plots of the true/estimated gradients ∇ φ E(φ) and estimated Bernoulli probability parameters σ(φ), with φ updated via gradient ascent.

Shown in Row 3 are the gradient variances for p0 = 0.49, estimated using K = 5000 Monte Carlo samples at each iteration; the theoretical gradient variances are also shown if they can be analytically calculated (see Appendices C and D and for related analytic expressions).true gradients, despite that the trace plots of the ARM gradients are distinct from these of the true gradients.

More specifically, while the true gradients smoothly evolve over iterations, the univariate ARM gradients are characterized by zeros and random spikes; this distinct behavior is expected by examining (38) in Appendix C, which suggests that at any given iteration, the univariate ARM gradient based on a single Monte Carlo sample is either exactly zero, which happens with probability σ(|φ|) − σ(−|φ|), or taking |[f (1) − f (0)](1/2 − u)| as its absolute value.

These observations suggest that by adjusting the frequencies and amplitudes of spike gradients, the univariate ARM estimator very well approximates the behavior of the true gradient for learning with gradient ascent.

In Figure 4 of Appendix D, we plot the gradient estimated with multiple Monte Carlo samples against the true gradient at each iteration, further showing the ARM estimator has the lowest estimation variance given the same number of Monte Carlo samples.

Moreover, in Figure 5 of Appendix D, for each estimator specific column, we plot against the value of φ the sample meanḡ, sample standard deviation s g , and the gradient signal-to-noise ratio defined as SNR g = |ḡ|/s g ; for each φ value, we use K = 1000 single-Monte-Carlo-sample gradient estimates to calculateḡ, s g , and SNR g .

Both figures further show that the ARM estimator outperforms not only REINFORCE, which has large variance, but also RELAX, which improves REINFORCE with an adaptively estimated baseline.

In Figure 5 of Appendix D, it is also interesting to notice that the gradient signal-to-noise ratio for the ARM estimator appears to be only a function of φ but not a function of p 0 ; this can be verified to be true using (23) and (39) in Appendix C, as the ratio of the absolute value of the true gradient |g φ | to var[g φ,ARM ], the standard deviation of the ARM estimate in (24), can be expressed as DISPLAYFORM5 We find that the values of the ratio shown above are almost exactly matched by the values of SNR g = |ḡ|/s g under the ARM estimator, shown in the bottom right subplot of Figure 5 .

Therefore, for this example optimization problem, the ARM estimator exhibits a desirable property in providing high gradient signal-to-noise ratios regardless of the value of p 0 .

To optimize a variational auto-encoder (VAE) for a discrete latent variable model, existing solutions often rely on biased but low-variance stochastic gradient estimators BID0 BID11 , unbiased but high-variance ones BID19 , or unbiased REINFORCE combined with computationally expensive baselines, whose parameters are estimated by minimizing the sample variance of the estimator with SGD BID35 BID7 .

By contrast, the ARM estimator exhibits low variance and is unbiased, efficient to compute, and simple to implement.

DISPLAYFORM0 For discrete VAEs, we compare ARM with a variety of representative stochastic gradient estimators for discrete latent variables, including Wake-Sleep BID10 , NVIL BID19 , LeGrad (Titsias & Lázaro-Gredilla, 2015) , MuProp BID9 , Concrete (GumbelSoftmax) BID11 , REBAR BID7 , and RELAX BID35 .

Following the settings in BID35 and BID7 , for the encoder defined in (15) and decoder defined in (16), we consider three different network architectures, as summarized in TAB1 , including "Nonlinear" that has one stochastic but two Leaky-ReLU (Maas et al., 2013) deterministic hidden layers, "Linear" that has one stochastic hidden layer, and "Linear two layers" that has two stochastic hidden layers.

We consider a widely used binarization BID30 BID16 BID39 , referred to as MNIST-static and available at http://www.dmi.usherb.ca/∼larocheh/mlpython/ modules/datasets/binarized mnist.html, making our numerical results directly comparable to those reported in the literature.

In addition to MNIST-static, we also consider MNIST-threshold (van den Oord et al., 2017), which binarizes MNIST by thresholding each pixel value at 0.5, and the binarized OMNIGLOT dataset.

We train discrete VAEs with 200 conditionally iid Bernoulli random variables as the hidden units of each stochastic binary layer.

We maximize a single-Monte-Carlo-sample ELBO using Adam BID13 , with the learning rate selected from {5, 1, 0.5} × 10 −4 by the validation set.

We set the batch size as 50 for MNIST and 25 for OMNIGLOT.

For each dataset, using its default training/validation/testing partition, we train all methods on the training set, calculate the validation log-likelihood for every epoch, and report the test negative log-likelihood when the validation negative log-likelihood reaches its minimum within a predefined maximum number of iterations.

We summarize the test negative log-likelihoods in TAB2 for MNIST-static.

We also summarize the test negative ELBOs in TAB4 of the Appendix, and provide related trace plots of the training and validation negative ELBOs on MNIST-static in Figure 2 , and these on MNIST-threshold and OMNIGLOT in Figures 6 and 7 of the Appendix, respectively.

For these trace plots, for a fair comparison of convergence speed between different algorithms, we use publicly available code and setting the learning rate of ARM the same as that selected by RELAX in BID7 .

Note as shown in Figures 2(a,d ) and 7(a,d), both REBAR and RELAX exhibit clear signs of overfitting on both MNIST-static and Omniglot using the "Nonlinear" architecture; as ARM runs much faster per iteration than both of them and do not exhibit overfitting given the same number of iterations, we allow ARM to run more stochastic gradient ascent steps under these two scenarios to check whether it will eventually overfit the training set.

These results show that ARM provides state-of-the-art performance in delivering not only fast convergence, but also low negative log-likelihoods and negative ELBOs on both the validation and test sets, with low computational cost, for all three different network architectures.

In comparison to the vanilla REINFORCE on MNIST-static, as shown in TAB2 (a), ARM achieves significantly lower test negative log-likelihoods, which can be explained by having much lower variance in its gradient estimation, while only costing 20% to 30% more computation time to finish the same number of iterations.

The trace plots in Figures 2, 6 , and 7 show that ARM achieves its objective better or on a par with the competing methods in all three different network architectures.

In particular, the performance of ARM on MNIST-threshold is significantly better, suggesting ARM is more robust, better resists overfitting, and has better generalization ability.

On both MNIST-static and OMNIGLOT, with the "Nonlinear" network architecture, both REBAR and RELAX exhibit severe overfitting, which could be caused by their training procedure, which updates the parameters of the baseline function by minimizing the sample variance of the gradient estimator using SGD.

For less overfitting linear and two-stochastic-layer networks, ARM overall performs better than both REBAR and RELAX and runs significantly faster (about 6-8 times faster) in terms of the computation time per iteration.

To understand why ARM has the best overall performance, we examine the trace plots of the logarithm of the estimated variance of gradient estimates in Figure 3 .

On the MNIST-static dataset with the "Nonlinear" network, the left subplot of Figure 3 shows that both REBAR and RELAX exhibit lower variance than ARM does for their single-Monte-Carlo-sample based gradient estimates; however, the corresponding trace plots of the validation negative ELBOs, shown in Figure 2 (a), suggest they both severely overfit the training data as the learning progresses; our hypothesis for this phenomenon is that REBAR and RELAX may favor suboptimal solutions that are associated with lower gradient variance; in other words, they may have difficulty in converging to local optimal solutions that are associated with high gradient variance.

For the "Linear" network architecture, the right subplot of Figure 3 shows that ARM exhibits lower variance for its gradient estimate than both REBAR and RELAX do, and Figure 2 (b) shows that none of them exhibit clear signs of overfitting; this observation could be used to explain why ARM results in the best convergence for both the training and validation negative ELBOs, as shown in Figure 2 (b).

Denoting x l , x u ∈ R 394 as the lower and upper halves of an MNIST digit, respectively, we consider a standard benchmark task of estimating the conditional distribution p θ0:2 (x l | x u ) BID25 BID0 BID9 BID11 BID35 , using a stochastic binary network with two stochastic binary hidden layers, expressed as DISPLAYFORM0 We set the network structure as 392-200-200-392, which means both b 1 and b 2 are 200 dimensional binary vectors and the transformation T θ are linear so the results are directly comparable with those in BID11 .

We approximate log p θ0:2 (x l | x u ) with log DISPLAYFORM1 .

We perform training with K = 1, which can also be considered as optimizing on a single-Monte-Carlo-sample estimate of the lower bound of the log marginal likelihood shown in (21).

We use Adam BID13 , with the learning rate set as 10 −4 , mini-batch size as 100, and number of epochs for training as 2000.

Given the inferred point estimate of θ 0:2 after training, we evaluate the accuracy of conditional density estimation by estimating the negative log-likelihood as − log p θ0:2 (x l | x u ), averaging over the test set using K = 1000.

We show example results of predicting the activation probabilities of the pixels of x l given x u in Figure 8 of the Appendix.

As shown in TAB3 , optimizing a stochastic binary network with the ARM estimator, which is unbiased and computationally efficient, achieves the lowest test negative log-likelihood, outperforming previously proposed biased stochastic gradient estimators on similarly structured stochastic networks, including DARN BID8 , straight through (ST) BID0 , slope-annealed ST BID4 , and ST Gumbel-softmax BID11 , and unbiased ones, including score-function (SF) and MuProp BID9 .

To train a discrete latent variable model with one or multiple stochastic binary layers, we propose the augment-REINFORCE-merge (ARM) estimator to provide unbiased and low-variance gradient estimates of the parameters of Bernoulli distributions.

With a single Monte Carlo sample, the estimated gradient is the product of uniform random noises and the difference of a function of two vectors of correlated binary latent variables.

Without relying on estimating a baseline function with extra learnable parameters for variance reduction, it maintains efficient computation and avoids increasing the risk of overfitting.

Applying the ARM gradient leads to not only fast convergence, but also low test negative log-likelihoods (and low test negative evidence lower bounds for variational inference), on both auto-encoding variational inference and maximum likelihood estimation for stochastic binary feedforward neural networks.

Some natural extensions of the proposed ARM estimator include generalizing it to multivariate categorical latent variables, combining it with a baseline or local-expectation based variance reduction method, and applying it to reinforcement learning whose action space is discrete.

Initialize w1:T , ψ randomly; while not converged do Sample a mini-batch of x from data; DISPLAYFORM0 ) T ∇w t Tw t (bt−1) ; end wt = wt + ρtgw t with step-size ρt end ψ = ψ + ηt∇ ψ f (b1:T ; ψ) with step-size ηt end

Let us denote t ∼ Exp(λ) as an exponential distribution, whose probability density function is defined as p(t | λ) = λe −λt , where λ > 0 and t > 0.

The mean and variance are E[t] = λ −1 and var[t] = λ −2 , respectively.

The exponential random variable t ∼ Exp(λ) can be reparameterized as t = /λ, ∼ Exp(1).

It is well known, e.g., in BID28 , that if t 1 ∼ Exp(λ 1 ) and t 2 ∼ Exp(λ 2 ) are two independent exponential random variables, then the probability that t 1 is smaller than t 2 can be expressed as P (t 1 < t 2 ) = λ 1 /(λ 1 + λ 2 ); moreover, since t 1 ∼ Exp(λ 1 ) is equal in distribution to 1 /λ 1 , 1 ∼ Exp(1) and t 2 ∼ Exp(λ 2 ) is equal in distribution to 2 /λ 2 , 2 ∼ Exp(1), we have DISPLAYFORM0 B.1 AUGMENTATION OF A BERNOULLI RANDOM VARIABLE AND REPARAMETERIZATION From (27) it becomes clear that the Bernoulli random variable z ∼ Bernoulli(σ(φ)) can be reparameterized by comparing two augmented exponential random variables as DISPLAYFORM1 Consequently, the expectation with respect to the Bernoulli random variable can be reparameterized as one with respect to two augmented exponential random variables as DISPLAYFORM2

Since the indicator function DISPLAYFORM0 is not differentiable, the reparameterization trick in FORMULA1 is not directly applicable to computing the gradient of (29).

Fortunately, as t 1 = 1 e −φ , 1 ∼ Exp(1) is equal in distribution to t 1 ∼ Exp(e φ ), the expectation in (29) can be further reparameterized as DISPLAYFORM1 and hence, via REINFORCE and then another reparameterization, we can express the gradient as DISPLAYFORM2 Similarly, we have DISPLAYFORM3 , and hence can also express the gradient as DISPLAYFORM4 Note that letting 1 , 2 iid ∼ Exp(1) is the same in distribution as letting DISPLAYFORM5 which can be proved using Exp(1) DISPLAYFORM6 , where u ∼ Uniform(0, 1)}, together with Lemma IV.3 of BID40 ; we use " d =" to denote "equal in distribution."

Thus, (B.2) can be reparameterized as DISPLAYFORM7 Applying Rao-Blackwellization BID3 , we can further express the gradient as DISPLAYFORM8 Therefore, the gradient estimator shown above, the same as (6), is referred to as the Augment-REINFORCE (AR) estimator.

A key observation of the paper is that by swapping the indices of the two iid standard exponential random variables in (32) , the gradient ∇ φ E(φ) can be equivalently expressed as DISPLAYFORM0 As the term inside the expectation in (31) and that in (35) could be highly positively correlated, we are motivated to merge FORMULA0 and FORMULA2 by sharing the same set of standard exponential random variables for Monte Carlo integration, which provides a new opportunity to well control the estimation variance BID23 .

More specifically, simply taking the average of FORMULA0 and FORMULA2 leads to DISPLAYFORM1 Note one may also take a weighted average of FORMULA0 and FORMULA2 , and optimize the combination weight to potentially further reduce the variance of the estimator.

We leave that for future study.

Note that (36) can be reparameterized as DISPLAYFORM2 Applying Rao-Blackwellization BID3 , we can further express the gradient as DISPLAYFORM3 Therefore, the gradient estimator shown above, the same as (7) , is referred to as the Augment-REINFORCE-merge (ARM) estimator.

Proof of Proposition 2.

Since the gradients g ARM (u, φ), g AR (u, φ), and g R (z, φ) are all unbiased, their expectations are the same as the true gradient DISPLAYFORM0 The second moment of g ARM (u, φ) can be expressed as DISPLAYFORM1 Thus, the variance of g ARM (u, φ) can be expressed as DISPLAYFORM2 which reaches its maximum at 0.039788 DISPLAYFORM3 2 .

For the REINFORCE gradient, we have DISPLAYFORM4 Therefore the variance can be expressed as DISPLAYFORM5 The largest variance satisfies DISPLAYFORM6 and hence when f is always positive or negative, we have DISPLAYFORM7 In summary, the ARM gradient has a variance that is bounded by 1 25 (f (1)−f (0)) 2 , and its worst-case variance is smaller than that of REINFORCE.Proof of Proposition 3.

We only need to prove for K = 1 and the proof for K > 1 automatically follows.

Since DISPLAYFORM8 which shows that the estimation variance of g ARM K ,v is guaranteed to be lower than that of the DISPLAYFORM9 when f is always positive or negative, the variance of g ARM K ,v is lower than that of g AR 2K ,v .

To maximize the variance reduction, it is equivalant to consider the constrained optimization problem DISPLAYFORM0 which is the same as a Lagrangian problem as DISPLAYFORM1

For the univariate AR gradient, we have DISPLAYFORM0 DISPLAYFORM1 , where p0 ∈ {0.49, 0.499, 0.501, 0.51} and the values of φ range from −2.5 to 2.5.

For each φ value, we compute for each estimator K = 1000 single-Monte-Carlo-sample gradient estimates, and use them to calculate their sample meanḡ, sample standard deviation sg, and gradient signal-tonoise ratio SNRg = |ḡ|/sg.

In each estimator specific column, we plotḡ, sg, and SNRg in Rows 1, 2, and 3, respectively.

The theoretical gradient standard deviations and gradient signal-to-noise ratios are also shown if they can be analytically calculated (see Eq. 25 and Appendices C and D and for related analytic expressions).

Figure 8: Randomly selected example results of predicting the lower half of a MNIST digit given its upper half, using a binary stochastic network, which has two binary linear stochastic hidden layers and is trained by the ARM estimator based maximum likelihood estimation.

Red squares highlight notable variations between two random draws.

@highlight

An unbiased and low-variance gradient estimator for discrete latent variable models

@highlight

Proposes a new variance-reduction technique to use when computing an expected loss gradient where the expectation is with respect to independent binary random variables.

@highlight

An algorithm combining Rao-Blackwellization and common random numbers for lowering the variance of the score-function gradient estimator in the special case of stochastic binary networks

@highlight

An unbiased and low variance augment-REINFORCE-merge (ARM) estimator for calculating and backpropagating gradients in binary neural networks
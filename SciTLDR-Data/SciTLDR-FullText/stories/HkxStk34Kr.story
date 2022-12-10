In this work we construct flexible joint distributions from low-dimensional conditional semi-implicit distributions.

Explicitly defining the structure of the approximation allows to make the variational lower bound tighter, resulting in more accurate inference.

Many recent advances in variational inference have been focused on different ways to estimate or bound the KL divergence between two complicated distributions.

They made it possible to perform variational inference with hierarchical distributions (Ranganath et al., 2016; Titsias and Ruiz, 2018; Sobolev and Vetrov, 2019) , semi-implicit distributions (Yin and Zhou, 2018; Molchanov et al., 2019) and even fully implicit distributions (Mescheder et al., 2017; Shi et al., 2017; Huszár, 2017) .

While these methods work well for low-dimensional cases, they can misbehave when the dimensionality of the problem grows.

In this work, we focus on semi-implicit variational inference, and consider structured multi-dimensional distributions.

We show that taking this structure into account, we can obtain a much tighter entropy bound and, consequentially, a much tighter evidence lower bound.

We also demonstrate that structured semi-implicit variational inference can successfully capture the multi-modal nature of the posterior distribution in deep Gaussian processes, and show a way to construct and learn an autoregressive semi-implicit model.

Variational inference provides a way to approximate the generally intractable posterior distribution p(z | D) in a probabilistic model with a parametric approximation q φ (z).

It typically requires the variational distribution q φ (z) to be reparameterizable and have a tractable log-density (Kingma and Welling, 2013) .

Semi-implicit variational inference (Yin and Zhou, 2018; Molchanov et al., 2019) extends this framework to so-called semi-implicit distributions.

By mixing a simple explicit distribution q φ (z | ) with an implicit distribution q( ) one obtains a so-called semi-implicit distribution with a generally intractable marginal density q φ (z).

A typical example of a semi-implicit model uses a Gaussian conditional distribution

, parameterized by neural networks µ φ ( ) and σ 2 φ ( ), whereas q( ) can be any fixed distribution that allows for efficient sampling 1 .

Unlike methods such as HVI, UIVI or IWHVI, SIVI does not need to access the density q( ).

The main idea behind semi-implicit variational inference, or SIVI, is to use K +1-sample estimates in order to obtain a lower bound on the entropy of such distribution.

Since all variables follow distribution q(·), we omit it for brevity.

This entropy bound can be used to construct a proper variational objective by bounding the KL-term in the evidence lower bound.

As with most implicit variational inference algorithms, the performance of semi-implicit variational inference can quickly degrade as the number of dimensions grows.

As SIVI essentially approximates a multi-dimensional distribution q φ (z) with a mixture of K + 1 Gaussian distributions in order to bound its entropy, it may require an exponentially large mixture size K to obtain an adequate approximation.

In order to solve this problem, we propose to factorize a high-dimensional joint semi-implicit distribution into a product of lowdimensional conditional semi-implicit distributions.

Here and after we abuse the notation and assume z 1..0 to denote an empty set.

In this case the entropy bounds can be written as follows:

This way we would only need to model low-dimensional semi-implicit distributions while still recovering a non-trivial joint distribution.

We provide two examples of models that follow such structure in Section 4.

It can be shown that given the same joint distribution q φ (z), taking the structure into account results in a tighter bound:

Theorem 1 For a structured semi-implicit distribution (3), the following inequalities hold:

We provide the proof of the theorem in appendix A. The main idea is to show that the structured SIVI bound with K + 1 samples of the mixing variable essentially approximates the marginal distribution q φ (z) with an exponentially large mixture of (K+1) d distributions, placed on a d-dimensional grid.

We apply SSIVI to deep Gaussian processes (Damianou and Lawrence, 2013; Salimbeni and Deisenroth, 2017) .

For detailed formulation of the deep GP model follow Appendix B.

over the inducing values u 1..L that factorizes across the layers.

Havasi et al. (2018) show that in practice the true posterior over the inducing values does not in fact factorize across layers, is non-Gaussian and is multimodal.

We get rid of all these limiting assumptions by using the following structured semi-implicit posterior approximation:

We use a fully-factorized Gaussian conditional distribution q φ (u l | l , u l−1 ) with means µ l φ ( l , u l−1 ) and scales σ l φ ( l , u l−1 ) parameterized by neural networks.

Using the structured SIVI bound (4), we obtain the final variational objective for SSIVI-DGP (see Appendix C for more details).

To demonstrate that SSIVI allows to recover multimodal posteriors with cross-layer dependencies, we consider the toy problem, proposed by Havasi et al. (2018) .

It is a noisefree (the likelihood variance is set to zero) regression problem consisting of seven training datapoints.

There are two natural modes in the posterior space, denoted Mode A and Mode B in the plots in Figure 1 .

We use SSIVI bound (35) with K = 100 and perform 3000 Adam (Kingma and Ba, 2014) updates with default hyperparameters and the learning rate set to 5 × 10 −3 .

We use seven inducing inputs on the first layer, fixed at the training point locations, and two inducing inputs on the second layers, fixed at 1 and −1.

Means and variances of the Gaussian conditional distributions q φ (u l | l , u l−1 ) are modeled by fully-connected neural networks with three hidden layers of 100 neurons each, and mixing variables l are sampled from 100-dimensional standard Gaussian distributions.

As shown in Figure 1 , DSVI (Salimbeni and Deisenroth, 2017) converges into one of them depending on the randomness in the initialization and the stochastic optimization process.

On the contrary, both SGHMC and SSIVI allow to capture all modes and the inter-layer dependencies.

We implement the structured semi-implicit distribution (3) in a general case using a recurrent neural network.

The generative process looks as follows:

In our experiments h(·, ·) is defined by two stacked GRU cells, and µ(·, ·) and σ 2 (·, ·) are defined as a fully-connected neural network with three hidden layers that outputs the mean and the log-scale of the one-dimensional Gaussian distribution.

All mixing variables i are scalar and follow the standard Gaussian distribution.

The width of all layers (both recurrent and fully-connected) is 100.

We train this model to generate samples from a synthetic multi-dimensional structured distribution p(z) = Laplace(z 1 | 0, 1)

To do this, we minimize the structured SIVI bound on the KL divergence KL (q φ (z) p(z)) with K = 100.

We use the SSIVI entropy bound (4) and estimate the cross-entropy using the reparameterization trick.

We perform 10000 steps with Adam with standard hyperparameters.

As one can see from Figure 2 , SSIVI provides a much tighter bound, and the gap between SIVI and SSIVI increases as the number of dimensions grows.

Proof The first inequality, namely the fact that H SSIVI K

[q φ (z)] is indeed a lower bound on the entropy H[q φ (z)], can be proven in exactly the same fashion as the corresponding proof for the original SIVI objective by Molchanov et al. (2019) .

The following proves the second inequality.

Firstly, let's rewrite the SIVI bound

Transition to line (11) holds since k are independent and identically distributed, making the expectations

We can expand the product of d sums in eq. (14) into a sum of (K + 1) d products of form

We thus obtain a mixture of (K + 1) d distributions that we denote asq φ (z | 0..K ).

Similarly to eq. (11), we can rewrite the expectation in eq. (14) as an expectation overq φ (z | 0..K ) sinceq φ (z | 0..K ) is also invariant to permutation of k .

This way we can rewrite the SSIVI bound as follows:

We can finally write down the gap between the SIVI and the SSIVI bounds:

This concludes proof of the theorem.

This section provides a brief overview of the definition of the DGP model, closely following DSVI.

A conventional single-output Gaussian Process model is defined as follows:

Here p(f | x, θ) is the Gaussian process prior, which is typically a zero-mean Gaussian distribution with the covariance matrix defined by a covariance function k θ (·, ·) (we denote it as K ·,· for brevity); y ∈ R N , f ∈ R N , x ∈ R N ×D .

The training of the parameters θ of the prior and the likelihood is performed using maximum marginal likelihood:

In order to reduce the required complexity, the sparse GP model (Titsias, 2009) introduces auxiliary variables, inducing inputs z ∈ R M ×D and values u ∈ R M :

In sparse GPs, direct maximization of the marginal likelihood is replaced with maximization of its lower bound (ELBO):

The approximate posterior q φ (f, u | x, z, θ) is specifically designed to reduce the computational complexity by cancelling out the most computation-heavy term p(f | u, x, z, θ):

This reduces the lower bound (22) to the following sparse GP ELBO:

which allows for doubly stochastic optimization.

A deep Gaussian process Damianou and Lawrence (2013); Salimbeni and Deisenroth (2017) is constructed as a chain of multi-output sparse Gaussian processes, or GP layers.

The output of each GP is considered as an input to the next GP.

The deep GP probabilistic model is defined similarly to conventional sparse GPs.

For each GP layer l, we have an output variable f l and a set of values u l corresponding to the inducing inputs z l The joint distribution over these variables is defined as follows (for brevity, we denote f 0 := x):

Similarly to sparse GPs, assuming a specific posterior approximation

, training the DGP involves bounding the data log marginal likelihood log p(y | x, θ) with the following variational lower bound, and then maximizing it w.r.t.

both the variational parameters φ and model parameters θ and z:

Appendix C. SSIVI for Sparse DGP

We substitute the factorized Gaussian approximation used in DSVI with a structured semiimplicit distribution:

Conventional variational inference for DGPs allows to integrate out the inducing values u analytically and obtain the marginal variational posteriors for q φ (f l | f l−1 , z l , θ).

This is not possible in SSIVI-DPGs, as now we have to explicitly condition the variational model on the inducing values from the previous layer.

Therefore we have to resort to plain MC estimation of the expected log-likelihood by sampling from the joint distribution q φ (f, u, | z, θ, x).

Now we modify lower bound (31) taking into account dependencies of inducing values between the layers and obtain the final objective for training SSIVI-DGPs:

@highlight

Utilizing the structure of distributions improves semi-implicit variational inference
We present Newtonian Monte Carlo (NMC), a method to improve Markov Chain Monte Carlo (MCMC) convergence by analyzing the first and second order gradients of the target density to determine a suitable proposal density at each point.

Existing first order gradient-based methods suffer from the problem of determining an appropriate step size.

Too small a step size and it will take a large number of steps to converge, while a very large step size will cause it to overshoot the high density region.

NMC is similar to the Newton-Raphson update in optimization where the second order gradient is used to automatically scale the step size in each dimension.

However, our objective is not to find a maxima but instead to find a parameterized density that can best match the local curvature of the target density.

This parameterized density is then used as a single-site Metropolis-Hastings proposal.



As a further improvement on first order methods, we show that random variables with constrained supports don't need to be transformed before taking a gradient step.

NMC directly matches constrained random variables to a proposal density with the same support thus keeping the curvature of the target density intact.



We demonstrate the efficiency of NMC on a number of different domains.

For statistical models where the prior is conjugate to the likelihood, our method recovers the posterior quite trivially in one step.

However, we also show results on fairly large non-conjugate models, where NMC performs better than adaptive first order methods such as NUTS or other inexact scalable inference methods such as Stochastic Variational Inference or bootstrapping.

Markov Chain Monte Carlo (MCMC) methods are often used to generate samples from an unnormalized probability density π(θ) that is easy to evaluate but hard to directly sample.

Such densities arise quite often in Bayesian inference as the posterior of a generative model p(θ, Y ) conditioned on some observations Y = y, where π(θ) = p(θ, y).

The typical setup is to select a proposal distribution q(.|θ) that proposes a move of the Markov chain to a new state θ * ∼ q(.|θ).

This Metropolis-Hastings acceptance rule is then used to accept or reject this move with probability: min 1, π(θ * )q(θ|θ * ) π(θ)q(θ * |θ) .

When θ ∈ R k , a common proposal density is the Gaussian distribution N (θ, 2 I k ) centered at θ with covariance 2 I k , where is the step size and I k is the identity matrix defined over R k,k .

This proposal forms the basis of the so-called Random Walk MCMC (RWM) first proposed in Metropolis et al. (1953) .

In cases where the target density π(θ) is differentiable, an improvement over the basic RWM method is to propose a new value in the direction of the gradient, as follows:

This method is known as Metropolis Adjusted Langevin Algorithm (MALA), and arises from an Euler approximation of a Langevin diffusion process (Robert and Tweedie, 1996) .

MALA has been shown to reduce the number of steps required for convergence to O(n 1/3 ) from O(n) for RWM (Roberts and Rosenthal, 1998 ).

An alternate approach, which also uses the gradient, is to do an L-step Euler approximation of Hamiltonian dynamics known as Hamiltonian Monte Carlo (Neal, 1993) , although it was originally published under the name Hybrid Monte Carlo (Duane et al., 1987) .

In HMC the number of steps, L, can be learned dynamically by the No-U-Turn Sampler (NUTS) algorithm (Hoffman and Gelman, 2014) .

However, in all three of the above algorithms -RWM, MALA, and HMC -there is an open problem of selecting the optimal step size.

Normally, the step size is adaptively learned by targeting a desired acceptance rate.

This has the unfortunate effect of picking the same step size for all the dimensions of θ, which forces the step size to accomodate the dimension with the smallest variance as pointed out in Girolami and Calderhead (2011) .

The same paper introduces alternate approaches, using Reimann manifold versions of MALA (MMALA) and HMC (RMHMC).

They propose a Reimann manifold using the expected Fisher information matrix plus the negative Hessian of the log-prior as a metric tensor, −E y|θ ∂ 2 ∂θ 2 log{p(y, θ)} , and proceed to derive the Langevin diffusion equation and Hamiltonian dynamics in this manifold.

The use of the above metric tensor does address the issue of differential scaling in each dimension.

However, the method as presented requires analytic knowledge of the Fisher information matrix.

This makes it difficult to design inference techniques in a generic way, and requires derivation on a per-model basis.

A more practical approach involves using the negative Hessian of the log-probability as the metric tensor, ∂ 2 ∂θ 2 log{p(y, θ)}. However, this encounters the problem that this is not necessarily positive definite throughout the state space.

An alternate approach for scaling the moves in each dimension is to use a preconditioning matrix M (Roberts and Stramer, 2002) in MALA, q(.|θ) = N θ + 2 M ∇ log{π(θ)}, 2 M , also known as the mass matrix in HMC and NUTS, but it's unclear how to compute this.

An alternate approach is to approximately compute the Hessian (Zhang and Sutton, 2011) using ideas from quasi-Newton optimization methods such as L-BFGS (Nocedal and Wright, 2006) .

This approach and its stochastic variant (Simsekli et al., 2016 ) use a fixed window of previous samples of size M to approximate the Hessian.

However, this makes the chain an order M Markov chain, which introduces considerable complexity in designing the transition kernel in addition to introducing a new parameter M .

The key observation in our work is that for single-site methods we only need to compute the Hessian of one coordinate at a time, and this is usually tractable.

The other key observation is that we don't need to always make a Gaussian proposer using the Hessian.

In some cases, other densities which are less concentrated such as Cauchy are more appropriate.

In general, the Hessian can be used for the purpose of matching the curvature of any parameterized density that best approximates the conditional posterior.

This approach of curvature-matching to an approximating density allows us to deal with constrained random variables without introducing a transformation such as in Stan (Carpenter et al., 2017) .

In the rest of the paper, we will describe our approach to exploit the curvature of the target density, and show some results on multiple data sets.

This paper introduces the Newtonian Monte Carlo (NMC) technique for sampling from a target distribution via a proposal distribution that incorporates curvature around the current sample location.

We wish to choose a proposal distribution that uses second order gradient information in order to closely match the target density.

Whereas related MCMC techniques discussed in Section 1 may utilize second order gradient information, those techniques typically use it only to adjust step size when simulating steps along the general direction of the target density's gradient.

Our proposed method involves matching the target density to a parameteric density that best explains the current state.

We have a library of 2-parameter target densities F i , and simple inference rules such that, given the first and second order gradients, we can solve the following two equations:

to determine α i and β i .

For example, in the case of θ ∈ R k , we use either the multivariate Gaussian or the multivariate Cauchy.

For the former, the update equation leads to the natural proposal,

The update term in the mean of this multivariate Gaussian is precisely the update term of the Newton-Raphson Method (Whittaker and Robinson, 1967) , which is where NMC gets its name from.

In case the estimated Σ has a negative eigenvalue we set those negative eigenvalues to a very small positive number, and reconstruct Σ. The full list of estimation methods are enumerated in Appendix A. For example, for positive real values we use a Gamma proposer,

and we don't need a log-transform to an unconstrained space.

In case multiple distributions can be fit, we pick the one which assigns the highest log-probability to the current state.

Even though we may pick a different proposer at each point in the state space, the choice of this proposer is a deterministic function of θ, and so we can precisely compute the MH acceptance probability.

We rely on generic Tensor libraries such as PyTorch (Paszke et al., 2017) that make it easy to write statistical models and also automatically compute the gradients.

This makes our approach easy to apply to models generically.

An important observation related to our method is that we don't need to compute the Hessian of all the parameters in the latent space.

Most statistical models can be decomposed into multiple latent variables.

This decomposition allows for single site MCMC methods that change the value of one variable at a time.

In this case, we only need to compute the gradient and Hessian of the target density w.r.t.

the variable being modified.

Consider a model with N variables each drawn from R K .

The full Hessian is of size (N K) 2 and has a cost of (N K) 3 to invert.

On the other hand, a single site approach computes N Hessians each of size K 2 with a total cost of N K 3 to invert.

In the case of conjugate models, our estimation methods automatically recover the appropriate conditional posterior distribution, such as the ones used in BUGS (Spiegelhalter et al., 1996) .

However, even in cases of non-conjugacy, our proposal distributions pick out reasonable approximations to the conditional posterior of each variable.

We present results of our experiments on three models -Bayesian Logistic Regression (Appendix B.1), Robust Regression (Appendix B.2), and a Crowd-Sourced Annotation Model (Passonneau and Carpenter (2014) , Dawid and Skene (1979) , Appendix B.3).

In each experiment, we drew a sample of the latent variables from the model and N observed variables.

Half of the observed variables were given to each inference engine, and the other half were used to compute the predictive likelihood over the posterior samples.

Figure 1 shows the relative convergence speed of various PPLs including Pyro (Bingham et al., 2019) and Stan on Bayesian Logistic Regression, which has a relatively easy log- concave posterior.

All of the inference engines except for Stan converge to the true posterior fairly quickly in terms of samples.

However, most of them are very slow (Table 1) and only Stan and NMC could be run on a larger data set.

NMC is nearly 5 times faster than Stan, which uses NUTS, in addition to converging faster.

Robust Regression, on the other hand, doesn't have a log-concave posterior and both JAGS (Plummer et al., 2003) and Stan struggle to converge (Figure 2 ).

In fact, Stan takes twice as much time for the same number of samples and it doesn't appear to have converged.

The final model, Crowd-Sourced Annotation Model, is a classic hierarchical statistical model.

Each random variable has a conjugate conditional posterior, and since JAGS is designed to exploit conjugacy it really shines in this example.

Unfortunately, the version of JAGS that we used kept crashing on larger data sets.

Figure 3 shows that NMC is easily able to keep up with JAGS in terms of number of samples and is only a factor of 2.5 slower on the small data set.

On the larger data set, NMC is nearly 7 times faster than Stan.

We have presented a novel MCMC method that uses the curvature of the target density to converge faster than existing state of the art methods, and without requiring any adaptive tuning.

As next steps, we will fully integrate NMC into a production PPL and evaluate its performance across a wider spectrum of illustrative and real-world use cases.

The multivariate Cauchy distribution has the log-density:

, and

Noting that the second term above is the outer product of the first gradient leads to the following estimation rules:

Half spaces refer to R + .

For example, the Gamma distribution, which has the log-density:

Gamma(x; α, β) = const(α, β) + (α − 1) log x − βx.

Thus,

Which leads to the estimation rules:

The K-simplexes refers to the set {x ∈ R + K | K i=1 x i = 1}.

We use the Dirichlet distribution to propose random variables with this support.

The log-density of the Dirichlet is given by,

We consider the modified density, which includes the simplex constraint,

Thus,

, and ∂ 2 ∂x i ∂x l Dir(x; α) = −δ il (α i − 1)

Which leads to the following robust estimation rule,

ii log π(x) − max j =i ∇ 2 ij log π(x) .

There are N items, K labelers, and each item could be one of C categories.

Each item i is labeled by a set J i of labelers.

z i is the true label for item i and y ij is the label provided to item i by labeler j. Each labeler l has a confusion matrix θ l such that θ lmn is the probability that an item with true class m is labeled n by l. Here α m ∈ R + C .

We set α mn = γ · ρ if m = n and α mn = γ · (1 − ρ) · 1 C−1 if m = n. Where γ is the concentration and ρ is the a-priori correctness of the labelers.

In this model, y il and J i are observed.

In our experiments we set γ = 10 and ρ = 0.5.

<|TLDR|>

@highlight

Exploit curvature to make MCMC methods converge faster than state of the art.
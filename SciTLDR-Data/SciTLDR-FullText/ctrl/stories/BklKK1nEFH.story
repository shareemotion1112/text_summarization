Transforming one probability distribution to another is a powerful tool in Bayesian inference and machine learning.

Some prominent examples are constrained-to-unconstrained transformations of distributions for use in Hamiltonian Monte-Carlo and constructing flexible and learnable densities such as normalizing flows.

We present Bijectors.jl, a software package for transforming distributions implemented in Julia, available at github.com/TuringLang/Bijectors.jl.

The package provides a flexible and composable way of implementing transformations of distributions without being tied to a computational framework.



We demonstrate the use of Bijectors.jl on improving variational inference by encoding known statistical dependencies into the variational posterior using normalizing flows, providing a general approach to relaxing the mean-field assumption usually made in variational inference.

When working with probability distributions in Bayesian inference and probabilistic machine learning, transforming one probability distribution to another comes up quite often.

For example, when applying Hamiltonian Monte Carlo on constrained distributions, the constrained density is usually transformed to an unconstrained density for which the sampling is performed (Neal, 2012) .

Another example is to construct highly flexible and learnable densities often referred to as normalizing flows (Dinh et al., 2014; Huang et al., 2018; Durkan et al., 2019) ; for a review see Kobyzev et al. (2019) .

When a distribution P is transformed into some other distribution Q using some measurable function b, we write Q = b * P and say Q is the push-forward of P .

When b is a differentiable bijection with a differentiable inverse, i.e. a diffeomorphism or a bijector (Dillon et al., 2017) , the induced or pushed-forward distribution Qit is obtained by a simple application of change of variables.

Specifically, given a distribution P on some Ω ⊆ R d with density p : Ω → [0, ∞), and a bijector b : Ω →Ω for someΩ ⊆ R d , the induced or pushed forward distribution Q = b * P has density q(y) = p b −1 (y) |det J b −1 (y)| or q b(x) = p(x) |det J b (x)|

As mentioned, one application of this idea is learnable bijectors such as normalizing flows.

One particular family of normalizing flow which has received a lot of attention is coupling flows (Dinh et al., 2014; Rezende and Mohamed, 2015; Huang et al., 2018) .

The idea is to use certain parts of the input vector x, say, x I 1 to construct parameters for a bijector f (the coupling law ), which is then applied to a different part of the input vector, say, x I 2 .

In full generality, a coupling flow c I 1 ,I 2 , the transformation in a coupling flow, is defined c I 1 ,I 2 (· ; f, θ) :

x I 2 → f x I 2 ; θ(x I 1 )

y I 2 → f −1 y I 2 ; θ(y I 1 )

where I 1 , I 2 ⊂ I := {1, . . .

, d} are disjoint.

As long as f · ; θ(x I 1 )

: R I 2 → R I 2 is a bijector, c I 1 ,I 2 is invertible since y I 1 = x I 1 .

Note the parameter-map θ can be arbitrarily complex.

Bijectors.jl is a framework for creating and using bijectors in the Julia programming language.

The main idea is to treat standard constrained-to-unconstrained bijectors, e.g. log : R → (0, ∞), and more complex and possibly parameterized bijectors, e.g. coupling flows, as the same just as they are mathematically the same.

This turns out to be quite a useful abstraction allowing seamless interaction between standard and learnable bijectors, making something like automatic differentiation variational inference (ADVI; Kucukelbir et al., 2016) easy to implement (see Source Code 1).

Table 1 shows supported mathematical operations.

Only b(x) and b −1 (y) need to be manually implemented for a new bijector b.

Another example is the introduction of neural autoregressive flows (NAFs; Huang et al., 2018) where the inverse autoregressive flow (IAF; Kingma et al., 2016) is extended by replacing the affine coupling law used in IAF with a monotonic deep neural network.

Despite the novel introduction of neural network, in Bijectors.jl the only difference between IAF and NAF is the choice of Bijector as the coupling law.

A summarization of the related work and how it compares to Bijectors.jl can be seen in Table 2 .

More detailed comparisons can be found in Appendix A.1.

1.

This refers to the torch.distributions submodule.

After our submission, another transformer module based on PyTorch was released in Pyro (Bingham et al., 2018) : pyro.distributions.transforms.

At the time of writing, we have not yet done a thorough comparison with this.

At a first glance its features seem similar in natu re to tensorflow.probability which can be found in Table 2 .

2.

Bijectors.jl is agnostic to array-types used, therefore GPU functionality is provided basically for free by using the independent package CuArrays.jl to construct the arrays.

forward(Q) (Ge et al., 2018) , using neural networks from Flux.jl to define the coupling law in a coupling flow, and so on.

For more examples, see the project website.

Inversed{Bijectors.

Logit{Float64},0}(Bijectors.

Logit{Float64}(0.0, 1.0)) 12 13 julia> # Works like a standard`Distribution1

We demonstrate how to use Bijectors.jl by a possible approach to relaxing the meanfield assumption commonly made in variational inference through the use of normalizing flows.

We consider a simple two-dimensional Gaussian with known covariance matrix with non-zero off-diagonal entries, i.e. different components are dependent, defined as follows

In this case we can obtain an analytical expression for the posterior p(m | {x i } n i=1 ), and can indeed observe that the covariance matrix has non-zero off-diagonals.

In this case, the mean-field assumption often made in variational inference is incorrect.

Recall that in variational inference the objective is to maximize the evidence lower bound (ELBO) of the variational posterior q(m) and the true posterior

For the ELBO of a transformed distribution, see Appendix B.2.

Here we propose using a mean-field multivariate normal as a starting point, and then combine this with coupling flows to encode the structure of the model we are approximating into the variational posterior at a low computational cost.

The idea is to encode an undirected edge between the random variables m 1 and m 2 by adding directed mappings in both directions; we do this by composing coupling flows c {2},{1} and c {1},{2} .

For the coupling flows, we experimented with two different coupling laws f , affine (Dinh et al., 2014) and the recently introduced rational-quadratic splines (Durkan et al., 2019) .

The parameter maps θ 1 and θ 2 , respectively, were defined by a simple neural network in both cases.

The resulting density, letting Q µ,σ be the distribution of an isotropic multivariate Gaussian with mean µ and variance σ, is given by

We then optimized the ELBO w.r.t.

the parameters of the neural networks θ 1 , θ 2 , µ and σ to obtain our variational posteriors.

The result of the standard mean-field VI (MFVI) and this particular normalizing flow VI (NFVI) applied to the model in Equation (2) can be seen in Figure 1 .

Here we observe that NFVI captures the correlation structure of the true posterior in Figure 1 (a) while MFVI, as expected, fails to do so.

This is also reflected in the value of the ELBO for the two approaches (see Appendix B.1).

This can potentially provide a flexible approach to taking advantage of structure in the joint distribution when performing variational inference without introducing a large number of parameters in addition to the mean-field parameters.

See Appendix B.1 for specifics of the experiment.

We presented Bijectors.jl, a framework for working with bijectors and thus transformations of distributions.

We then demonstrated the flexibility of Bijectors.jl in an application of introducing correlation structure to the mean-field ADVI approach.

We believe Bijectors.jl will be a useful tool for future research, especially in exploring normalizing flows and their place in variational inference.

An interesting note about the NF variational posterior we constructed is that it only requires a constant number of extra parameters on top of what is required by mean-field normal VI.

This approach can be applied in more general settings where one has access to the directed acyclic graph (DAG) of the generative model we want to perform inference.

Then this approach will scale linearly with the number of unique edges between random variables.

It is also possible in cases where we have an undirected graph representing a model by simply adding a coupling in both directions.

This would be very useful for tackling issues faced when using mean-field VI and would be of interest to explore further.

For related work we have mainly compared against Tensorflow's tensorflow probability, which is used by other known packages such pymc4, and PyTorch's torch.distributions, which is used by packages such as pyro.

Other frameworks which make heavy use of such transformations using their own implementations are stan, pymc3, and so on.

But in these frameworks the transformations are mainly used to transform distributions from constrained to unconstrained and vice versa with little or no integration between those transformation and the more complex ones, e.g. normalizing flows.

pymc3 for example support normalizing flows, but treat them differently from the constrained-to-unconstrained transformations.

This means that composition between standard and parameterized transformations is not supported.

Of particular note is the bijectors framework in tensorflow probability introduced in (Dillon et al., 2017) .

One could argue that this was indeed the first work to take such a drastic approach to the separation of the determinism and stochasticity, allowing them to implement a lot of standard distributions as a TransformedDistribution.

This framework was also one of the main motivations that got the authors of Bijectors.jl interested in making a similar framework in Julia.

With that being said, other than the name, we have not set out to replicate tensorflow probability and most of the direct parallels were observed after-the-fact, e.g. a transformed distribution is defined by the TransformedDistribution type in both frameworks.

Instead we believe that Julia is a language well-suited for such a framework and therefore one can innovate on the side of implementation.

For example in Julia we can make use of code-generation or meta-programming to do program transformations in different parts of the framework, e.g. the composition b • b −1 is transformed into the identity function at compile time.

Similar to tensorflow probability we can use higher-order bijectors to construct new bijectors.

Examples of such are Inverse, Compose, and Stacked.

A significant difference is that in Bijectors.jl, the constructors are rarely called explicitly by the user but instead through a completely intuitive interface, e.g. inv (b) gives you the Inverse, b1 • b2 gives you the composition of b1 and b2, stack(b1, b2) gives you the two bijectors "stacked" together.

Moreover, if b actually has a "named" inverse, e.g. b = Exp(), then inv(b) will result in Log() rather than some thin wrapper Inversed(Exp()).

Irregardless of whether the bijector has a named inverse or not, the dual-nature is exploited in compositions so that b • inv(b) results in Identity().

For type-stable code, this is all done at compile-time.

A particularly nice one is the Stacked(bijectors, ranges) which allows the user to specify which parts (or ranges) of the input vector should be passed to which of the "stacked".

For all methods acting on a Stacked the loop for iterating through the different ranges and applying the corresponding Bijector will be unrolled, meaning that this abstraction has a zero-cost overhead and the only cost is the evaluation of corresponding methods on for the bijectors it wraps.

In a limited sense Bijectors.jl can do what is known as program transformations.

A good example is b • b −1 resulting in identity at compile-time for simple transformations which we have mentioned before.

In tensorflow probability indeed b • b −1 is reduced to the identity mapping, not by collapsing the computational graph but instead by the use of caching.

This means that when (b • b −1 )(x) is evaluated, work will only be done for the b −1 (x) evaluation.

When b −1 (x) is evaluated by b, the cached value x used to evaluate b −1 just before will be returned immediately.

torch.distributions take a similar approach but because caching can come with its own issues, especially when used in conjunction with automatic differentiation, there are cases where it will fail, e.g. dependency reversal.

In Bijectors.jl there are two parts of this story.

First off, b • b −1 will, as noted earlier, be compiled to the identity map upon compilation, i.e. there is zero-overhead at run-time to this evaluation.

But one nice property of the Tensorflow and PyTorch approach which uses caching is that one can write code that looks like In Bijectors.jl this has to be done manually by the user through the forward method for a TransformedDistribution.

Recall from Table 1 that forward returns a 4-tuple (x, b(x), logabsdetjac (b, x) , logpdf(q, b(x))) using the most efficient computation path.

Therefore to replicate the above example in Bijectors.jl, we can do # Samples x from base and returns y = b(x) x, y, _, _ = forward(transformed_distribution) # do some more computation potentially involving y # ...

Therefore "caching" in Bijectors.jl cannot be done across function barriers at the time of writing (unless the function explicitly returns all values used).

On the bright side one can explicitly do caching, making it more difficult to do something wrong in addition to the fact that the computation is transparent from the users perspective.

Appendix B. Adding structure to mean-field VI using coupling flows

In the experimental setup we generate data by fixing m = 0 and generating n = 100 samples from Equation (2).

This resulted in a posterior multivariate normal with covariance matrix

This was done by design, as we specifically chose L = 10 0 10 10 to get a posterior covariance matrix with non-zero off-diagonals.

Let q µ,σ denote the density of a multivariate Gaussian with mean µ and diagonal covariance σ, and b denote the coupling flow in Equation (1) with f as a rational-quadratic spline (RQS) with K = 3 knot points and bin [−50, 50] , θ as a neural network consisting of one layer with a (3K − 1) × 1 weight matrix and bias with identity activation, i.e. a simple affine transformation.

See (Durkan et al., 2019) for more information on RQS.

We use Distributions.jl for implementation of the Gaussian multivariate distribution (Lin et al., 2019) .

We then performed variational inference on the model in Equation (2)

resulting in Figure 5 (c).

The resulting densities can be observed in Figure 3 .

Note that here we have used a slight abuse of notation writing max θ to mean "maximize wrt.

parameters of θ".

The expressions for the KL-divergence and the ELBO, which under the transformation by a bijector picks up an additional term, see Equation (5) and Equation (8), respectively.

In all cases we set the number of samples used in the Monte-Carlo estimate of the objective to be m = 50.

In all cases we used a DecayedADAGrad from Turing.jl to perform gradient updates.

This is a classical ADAGrad (Duchi et al., 2011) but with a decay for the accumulated gradient norms.

This is to circumvent the possibility of large initial gradient norms bringing all subsequent optimization steps to practically zero step-size.

For DecayedADAGrad we used a base step-size η = 0.001, post-factor decay β post = 1.0 and pre-factor decay β pre = 0.9, and we performed 5 000 optimization steps before terminating.

In general we of course do not have access to the true posterior and so we cannot minimize the KL-divergence between the variational posterior and the true posterior directly, but instead have to do so implicitly by minimizing the ELBO.

In theory there is no difference, but in practice one usually observe a significantly lower variance in the gradient estimates of the KL-divergence compared to the ELBO.

We therefore also performed VI using the KLdivergence to verify that the NF did not lack the expressibility to capture the true posterior, but that the slight inaccuracy in the variational posterior obtained by maximizing the ELBO was indeed due to the variance in the gradient estimate.

And, as expected, minimizing the KL-divergence directly in the MF-case did not provide much of a gain compared to maximizing the ELBO.

Numerical results for multiple runs where the ELBO was used as an objective can be seen in Table 3 and Figure 2 ; the NFVI approach consistently obtains lower KL divergence and a greater ELBO.

The main quantity of interest is the KL-divergence which quantifies the difference between the variational posterior and the true posterior.

The ELBO is a lower bound on the evidence and thus the actual values can vary widely across experiments.

Additionally, the difference between the ELBO of two distributions with respect to the same set of observations is equal to the difference between KL-divergence on that set of observations, and so we gain no additional information of the difference between the variational posterior and the true posterior by looking at the ELBO.

Therefore we visualize the KL-divergence instead of the ELBO in Figure 2 , but still provide the numerical values for both in Table 3 Table 3 : (Rational-Quadratic Spline coupling law) Exponentially smoothed estimates of the last 4000 (out of 5000) optimization steps.

As can be seen in Figure 2 , after the 1000th step is basically when the optimas are reached.

Here the ELBO has been used as the objective.

We also performed the same experiment using an affine transformation f as a coupling law.

The setup is identical, but now θ is a neural network consisting of two layers; the first layer is a dense layer with 2 × 1 weight matrix and bias and ReLU activation, and the second layer is a dense layer with 2 × 2 weight matrix and bias and identity activation.

In Flux.jl, which we have used for the neural network part of the bijector, this is given by Chain(Dense(1, 2, relu), Dense(2, 2))) (Innes, 2018) .

As one can see in Table 4 and Figure 4 , even with an affine coupling law we obtain very good approximations.

Table 4 : (Affine coupling law) Exponentially smoothed estimates of the last 4000 (out of 5000) optimization steps.

As can be seen in Figure 2 , after the 1000th step is basically when the optimas are reached.

Here the ELBO has been used as the objective.

Recall the definition of the Kullback-Leibler (KL) divergence, here relating the variational density q(z) and posterior p(z | {x} n i=1 ),

As per usual, we can rewrite this

where in the second-to-last equality we used the assumption that the observations are i.i.d.

and in the last equality we used the fact that log p(x i ) is independent of z for all i = 1, . . .

, n. We can then arrange this into

Observe that given a set of observations, the left-hand side is constant.

Therefore we can minimize the KL-divergence by maximizing the remaining terms on the right-hand side of the equation, which we call the evidence lower bound (ELBO)

Now suppose that the variational posterior q(z) is in fact a transformed distribution, say, with base density q 0 and using transformation b, i.e. Substituting these terms into the ELBO from Equation (6), we get

This expression ise very useful when q 0 is a density which it is computationally cheap to sample from and we have an analytical expression for the entropy of q 0 , e.g. if q 0 is the density of a mulitvariate Gaussian both of these conditions are satisfied.

In practice we use a Monte-Carlo estimate of the ELBO

where η k ∼ q 0 (η) for k = 1, . . .

, m. From this we can then obtain a Monte-Carlo estimate of the gradient wrt.

parameters.

<|TLDR|>

@highlight

We present a software framework for transforming distributions and demonstrate its flexibility on relaxing mean-field assumptions in variational inference with the use of coupling flows to replicate structure from the target generative model.
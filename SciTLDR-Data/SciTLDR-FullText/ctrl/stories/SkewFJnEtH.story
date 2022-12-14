In classic papers, Zellner (1988, 2002) demonstrated that Bayesian inference could be derived as the solution to an information theoretic functional.

Below we derive a generalized form of this functional as a variational lower bound of a predictive information bottleneck objective.

This generalized functional encompasses most modern inference procedures and suggests novel ones.

Consider a data generating process φ ∼ p(φ) from which we have some N draws that constitute our training set, x P = {x 1 , x 2 , . . .

, x N } ∼ p(x|φ).

We can also imagine (potentially infinitely many) future draws from this same process x F = {x N +1 , . . . } ∼ p(x|φ).

The predictive information I(x P ; x F ) 1 gives a unique measure of the complexity of a data generating process (Bialek et al., 2001 ).

The goal of learning is to capture this complexity.

To perform learning, we form a global representation of the dataset p(θ|x P ).

This can be thought of as a learning algorithm, that, given a set of observations, produces a summary statistic of the dataset that we hope is useful for predicting future draws from the same process.

This algorithm could be deterministic or more generally, stochastic.

For example, imagine training a neural network on some data with stochastic gradient descent.

Here the training data would be x P , the test data x F and the neural network parameters would be θ.

Our training procedure implicitly samples from the distribution p(θ|x P ).

How do we judge the utility of this learned global representation?

The mutual information I(θ; x F ) quantifies the amount of information our representation captures about future draws.

2 To maximize learning we therefore aim to maximize this quantity.

1.

We use I(x; y) for the mutual information between two random variables:

2.

It is interesting to note that in the limit of an infinite number of future draws, I(θ; x F ) approaches I(θ; φ).

Therefore, the amount of information we have about an infinite number of future draws from This is, of course, only interesting if we constrain how expressive our global representation is, for otherwise we could simply retain the full dataset.

The amount of information retained about the observed data: I(θ; x P ) is a direct measure of our representation's complexity.

The bits a learner extracts from data provides upper bounds on generalization (Bassily et al., 2017) .

Combined, these motivate the predictive information bottleneck objective, a generalized information bottleneck (Bialek et al., 2001; Tishby et al., 2000) :

We can turn this into an unconstrained optimization problem with the use of a Lagrange multiplier β: max

While this objective seems wholly out of reach, we can make progress by noting that our random variables satisfy the Markov chain: x F ← φ → x P → θ, in which θ and x F are conditionally independent given x P :

This implies:

and the equivalent unconstrained optimization problem: 3

The first term here: I(θ; x P |x F ) is the residual information between our global representation and the dataset after we condition on full knowledge of the data generating procedure.

This is a direct measure of the inefficiency of our proposed representation.

Simple variational bounds (Poole et al., 2019) can be derived for this objective, just as was done for the (local) information bottleneck objective in Alemi et al. (2016) .

First, we demonstrate a variational upper bound on I(θ;

the process is the same as the amount of information we have about the nature and identity of the data generating process itself.

3.

A similar transformation for the (local) variational information bottleneck appeared in Fischer (2019) .

4. · is used to denote expectations, and unless denoted otherwise with respect to the full joint density

Here we upper bound the residual information by using a variational approximation to p(θ|x F ), the marginal of our global representation over all datasets drawn from the same data generating procedure.

Any distribution q(θ) independent of x F suffices.

Next we variationally lower bound I(θ; x P )

with:

The entropy of the training data H(x P ) is a constant outside of our control that can be ignored.

Here we variationally approximate the "posterior" of our global representation with a factorized "likelihood": i q(x i |θ) = q(x P |θ) ∼ p(x P |θ).

Notice that while p(x P |θ) will not factorize in general, we can certainly consider a family of variational approximations that do.

Combining these variational bounds, we generate the objective:

We have thus derived, as a variational lower bound on the predictive information bottleneck, the objective Zellner (1988) postulates (with β = 1) is satisfied for inference procedures that optimally process information.

As Knoblauch et al. (2019) demonstrates, this encompasses a wide array of modern inference procedures, including Generalized Bayesian Inference (Bissiri et al., 2016 ) and a generalized Variational Inference, dubbed Gibbs VI (Alquier et al., 2016; Futami et al., 2017) .

5 Below we highlight some of these and other connections.

If, in Equation (8), we identity q(θ) with a fixed prior and q(x|θ) with a fixed likelihood of a generative model, optimizing this objective for p(θ|x P ) in the space of all probability densities gives the generalized Boltzmann distribution (Jaynes, 1957):

where Z is the partition function.

6 This is a generalized form of Bayesian Inference called the power likelihood (Holmes and Walker, 2017; Royall and Tsou, 2003) .

Here the inverse temperature β acts as a Lagrange multiplier controlling the trade-off between the amount of information we retain about our observed data (I(θ; x P )) and how much predictive information we capture (I(θ; x F )).

As β → ∞ (temperature goes to zero), we recover the maximum likelihood solution.

At β = 1 (temperature = 1) we recover ordinary Bayesian inference.

As β → 0 (temperature goes to infinity), we recover just prior predictive inference that ignores the data entirely.

These limits are summarized in Table 1 .

5.

To incorporate the Generalized VI (Knoblauch et al., 2019) with divergence measures other than KL, we need only replace our mutual informations (which are KL based) with their corresponding generalizations.

Table 1 : Power Bayes can be recovered as a variational lower bound on the predictive information bottleneck objective (Equation (5)).

More generally, notice that in Equation (8) the densities q(x|θ) and q(θ) are not literally the likelihood and prior of a generative model, they are variational approximations that we have complete freedom to specify.

This allows us to describe other more generalized forms of Bayesian inference such as Divergence Bayes or the full Generalized Bayes (Knoblauch et al., 2019; Bissiri et al., 2016) provided we can interpret the chosen loss function as a conditional distribution.

If we limit the domain of p(θ|x P ) to a restricted family of parametric distributions, we immediately recover not only standard variational inference, but a broad generalization known as Gibbs Variational Inference (Knoblauch et al., 2019; Alquier et al., 2016; Futami et al., 2017) .

Furthermore, nothing prevents us from making q(x|θ) or q(θ) themselves parametric and simultaneously optimizing those.

Optimizing the prior with a fixed likelihood, unconstrained p(θ|x P ), and β = 1 the objective mirrors Empirical Bayesian (Maritz and Lwin, 2018) approaches, including the notion of reference priors (Mattingly et al., 2018; Berger et al., 2009) .

Alternatively, optimizing a parametric likelihood with a parametric representation p(θ|x P ), fixed prior, and β = 1 equates to a Neural Process (Garnelo et al., 2018) .

Consider next data augmentation, where we have some stochastic process that modifies our data with implicit conditional density t(x |x).

If the augmentation procedure is centered about zero so that x t(x |x) = x and our chosen likelihood function is concave, then we have:

which maintains our bound.

For example, for an exponential family likelihood and any centered augmentation procedure (like additive mean zero noise), doing generalized Bayesian inference on an augmented dataset is also a lower bound on the predictive information bottleneck objective.

We have shown that a wide range of existing inference techniques are variational lower bounds on a single predictive information bottleneck objective.

This connection highlights the drawbacks of these traditional forms of inference.

In all cases considered in the previous section, we made two choices that loosened our variational bounds.

First, we approximated p(x P |θ), with a factorized approximation q(x P |θ) = i q(x i |θ).

Second, we approximated the future conditional marginal p(θ|x F ) = dx P p(θ|x P )p(x P |x F ) as an unconditional "prior".

Neither of these approximations is necessary.

For example, consider the following tighter "prior":

q(θ|x F ) ∼ dx P p(θ|x P )q(x P |x F ).

Here we reuse a tractable global representation p(θ|x P ) and instead create a variational approximation to the density of alternative datasets drawn from the same process: q(x P |x F ).

We believe this information-theoretic, representation-first perspective on learning has the potential to motivate new and better forms of inference.

7

<|TLDR|>

@highlight

Rederive a wide class of inference procedures from an global information bottleneck objective.
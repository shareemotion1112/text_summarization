Importance sampling (IS) is a standard Monte Carlo (MC) tool to compute information about random variables such as moments or quantiles with unknown distributions.

IS is  asymptotically consistent as the number of MC samples, and hence deltas (particles) that parameterize the density estimate, go to infinity.

However, retaining infinitely many particles is intractable.

We propose a scheme for only keeping a \emph{finite representative subset} of particles and their augmented importance weights that is \emph{nearly consistent}. To do so in {an online manner}, we approximate importance sampling in two ways.

First, we replace the deltas by kernels, yielding kernel density estimates (KDEs).

Second, we sequentially project KDEs onto nearby lower-dimensional subspaces.

We characterize the asymptotic bias of this scheme as determined by a compression parameter and kernel bandwidth, which yields a tunable tradeoff between consistency and memory.

In experiments,  we observe a favorable tradeoff between memory and accuracy, providing for the first time near-consistent compressions of arbitrary posterior distributions.

Importance sampling is a MC method that addresses Bayesian inference in cases where the distribution that relates observations to the hidden state is time-invariant (Tokdar and Kass, 2010).

More specifically, based upon independent samples from a proposal distribution, MC methods approximately compute expectations of arbitrary functions of the unknown parameter via weighted samples generated from the proposal.

Recently, use of importance distributions to weight updates, e.g., coordinate descent (Allen-Zhu et al., 2016; Csiba et al., 2015) or stochastic gradient descent (Borsos et al., 2018) , have been developed.

Doing so yields faster deep network training (Johnson and Guestrin, 2018; Katharopoulos and Fleuret, 2018) by weighting mini-batches (Hanzely and Richtárik, 2018) .

Furthermore, in reinforcement learning (RL), an agent chooses actions according to a policy and then updates the policy via rewards observed (Watkins and Dayan, 1992) ; however, this theoretically requires an inordinate amount of random actions to be chosen before reasonable performance is learned (Tsitsiklis, 1994; , an issue known as the exploreexploit tradeoff.

To lessen its deleterious effect, exploratory actions may be chosen via an importance distribution (Schaul et al., 2015) or policy updates may be chosen from previous experience known to be safe (Precup et al., 2000) .

Contributions.

We propose a compression scheme that operates within importance sampling, sequentially deciding which particles are statistically significant for the integral estimation.

To do so, we draw connections between proximal methods in optimization (Rockafellar, 1976) and importance distribution updates: we view the empirical measure defined by importance sampling as carrying out a sequence of projections of un-normalized empirical distributions onto subspaces of growing dimension.

Then, we augment the subspace selection by replacing it by one that is nearby (according to some metric) but with lower memory.

These lower-memory subspaces are selected based on greedy compression with a fixed budget parameter via matching pursuit (Pati et al., 1993) .

We combine this idea with kernel smoothing of the empirical measure in order to exploit the fact that compact spaces have finite covering numbers.

Consequently, we have characterized the asymptotic bias of this method as a tunable constant depending on the kernel bandwidth parameter and a compression parameter.

Experiments demonstrate that this approach yields an effective tradeoff of consistency and memory for MC methods.

In Bayesian inference (Särkkä, 2013) [Ch.

7] , we are interested in computing expectations

on the basis of a set of available observations {y k } k≤K , where φ : R p → R is an arbitrary function, x is a random variable taking values in X ⊂ R p which is typically interpreted as the hidden parameter, and y is some observation process whose realizations y k are assumed to be informative about parameter x. For example, φ(x) = x yields the computation of the posterior mean, and φ(x) = x p denotes the p-th moment.

In particular, define the posterior distribution

We seek to infer the posterior (2) with K data points {y k } k≤K available at the outset.

Even for this setting, estimating (2) has unbounded complexity (Li et al., 2005; Tokdar and Kass, 2010) when the posterior is unknown.

Thus, we prioritize efficient estimates of (2) from an online stream of samples from an importance distribution to be subsequently defined.

Begin by defining posterior q(x) and un-normalized posteriorq(x):

is a non-negative function proportional to posterior q(x|y):=q(x) = p x y 1 that integrates to normalizing constant Z:=p ({y k } k≤K ).

In Monte Carlo, we approximate (1) by sampling.

Hypothetically, we could draw N (not necessarily equal to K) samples x (n) ∼ q(x) and estimate the expectation in (1) by the sample average

, but typically it is difficult to obtain samples x (n) from posterior q(x) of the hidden state.

To circumvent this issue, define the importance distribution π(x) 2 with the same (or larger) support as true density q(x), and multiply and divide by π(x) inside the integral (1):

1. Note that q(x) andq(x) depend on the data {y k } k≤K , although we drop the dependence to ease notation.

2.

In general, the importance distribution could be defined over any observation process π(x {y k }), not necessarily associated with time indices k = 1, . . .

, K. We define it this way for simplicity.

where the ratio q(x)/π(x) is the Radon-Nikodym derivative, or unnormalized density, of the target q with respect to the proposal π.

Then, rather than requiring samples from true posterior x (n) ∼ q(x), one may sample from importance distribution x (n) ∼ π(x), n = 1, ..., N , and approximate (1) as

are the importance weights.

We note that in practice, we cannot calculate q(x (n) ) since the target distribution q(x) is unknown and hence we calculate it using Bayes rule as follows:

Substituting (5) into (3), we obtain g(x (n) ) :

.

Note that (4) is unbiased, i.e.,

and consistent with N .

Moreover, its variance depends on how well the importance density π(x) approximates the posterior (Elvira et al., 2019) .

Example priors and measurement models include Gaussian, Student's t, and Uniform.

Which one is appropriate depends on the context (Särkkä, 2013) .

The normalizing constant Z can be also estimated with IS asẐ := 1 N N n=1 g(x (n) ).

Hence, we can replace Z in Eq. (4) byẐ. Then, the new estimator is given by

where the "self-normalized"w (n) weights are definedw (n) :=

.

The estimator I N (φ) is the self-normalized importance sampling (SNIS) estimator.

It is important to note that the estimator I N (φ) can be viewed as integrating a function φ with respect to

, which is called the particle approximation of q where δ x (n) denotes the discrete Dirac delta (indicator) which is 1 if x = x (n) and null otherwise.

This delta expansion is one reason importance sampling is also referred to as histogram filters, as they quantify weighted counts of samples across the space.

As stated in (Agapiou et al., 2017) , for consistent estimates of (1), we require that N , the number of samples x n generated from the importance distribution, and hence the parameterization of the importance distribution, grows unbounded as it accumulates every particle previously generated, as N → ∞. We are interested in allowing N , the number of particles, to become large (possibly infinite), while the importance distribution's complexity is moderate, thus overcoming an instance of the curse of dimensionality in Monte Carlo methods.

Next, we proposed a compressed kernelized importance sampling algorithm summarized in Algorithm 1.

Require: Observation model p(y x) and prior p(x) or target distribution q(x) (if known), importance distribution π(x), Observation collection {y k } K k=1

for n = 0, 1, 2, . . .

, N do Simulate one sample from importance dist.

Compute the importance weight g(x (n) ) :=q

Normalize weights w (n) as follows:

Update kernel density via last sample & weight

Normalized weights to ensure valid probability measurew n Estimate the expectation asÎ n = |Dn| u=1w

In this section, we establish conditions under which the asymptotic bias is proportional to the kernel bandwidth and the compression parameter using posterior distributions given by Algorithm 1.

The results permits characterizing the bias of Algorithm 1 given next.

Under Assumptions 1-3 in (Koppel et al., 2019) , the estimator of Algorithm 1 exhibits posterior contraction:

and hence, as N → ∞, is consistent when compression budget and bandwidth go to null , h → 0.

Theorem 1 (proof in (Koppel et al., 2019) ) establishes that the compressed kernelized importance sampling scheme proposed in Algorithm 1 is nearly asymptotically consistent, and can be made arbitrarily close to exact consistency by sending the bandwidth h and compression budget to null.

However, when these parameters are fixed positive constants, they provide a tunable tradeoff between bias and memory.

Theorem 2: Denote asμ n the empirical distribution defined by Algorithm 1 whose model order is M n after n particles generated from importance density π(x).

Under some Assumptions (detailed in (Koppel et al., 2019) ), for compact feature space X and bounded importance weights g(x (n) ), M n < ∞ for all n. In this section, we conduct a simple numerical experiment to demonstrate the efficacy of the proposed algorithm in terms of balancing model parsimony and statistical consistency.

We consider the problem of estimating the expected value of function φ(x) with the target q(x) and the proposal π(x) given by

to demonstrate that generic Monte Carlo integration allows one to track generic quantities of random variables that are difficult to compute under usual probabilistic hypotheses.

Fig. 1a shows the un-normalized integral approximation error for Algorithms with and without compression, which are close, and the magnitude of the difference depends on the choice of compression budget.

This trend is corroborated in the evolution of (normalized) integral estimates in Fig. 1b : very little error is incurred by kernel smoothing and memoryreduction.

The actual magnitude of the error relative to the number of particles generated is displayed in Fig. 1c : observe that the error settles on the order of 10 −3 .

In Fig. 1d , we display the number of particles retained by Algorithm 1, which stabilizes to around 56, whereas the complexity of the empirical measure without compression grows linearly with sample index n, which noticeably grows unbounded.

@highlight

We proposed a novel compressed kernelized importance sampling algorithm.
Variational inference based on chi-square divergence minimization (CHIVI) provides a way to approximate a model's posterior while obtaining an upper bound on the marginal likelihood.

However, in practice CHIVI relies on Monte Carlo (MC) estimates of an upper bound objective that at modest sample sizes are not guaranteed to be true bounds on the marginal likelihood.

This paper provides an empirical study of CHIVI performance on a series of synthetic inference tasks.

We show that CHIVI is far more sensitive to initialization than classic VI based on KL minimization, often needs a very large number of samples (over a million), and may not be a reliable upper bound.

We also suggest possible ways to detect and alleviate some of these pathologies, including diagnostic bounds and initialization strategies.

Estimating the marginal likelihood in probabilistic models is the holy grail of Bayesian inference.

Marginal likelihoods allow us to compute the posterior probability of model parameters or perform Bayesian model selection (Bishop et al., 1995) .

While exact computation of the marginal is not tractable for most models, variational inference (VI) (Jordan et al., 1999 ) offers a promising and scalable approximation.

VI suggests choosing a simple family of approximate distributions q and then optimizing the parameters of q to minimize its divergence from the true (intractable) posterior.

The canonical choice is the KL divergence, where minimizing corresponds to tightening a lower bound on the marginal likelihood.

Recently, (Dieng et al., 2017a) showed that minimizing a χ 2 divergence leads to a chi-divergence upper bound ("CUBO").

Practitioners often wish to combine upper and lower bound estimates to "sandwich" the model evidence in a narrow range for later decision making, so the CUBO's flexible applicability to all latent variable models is appealing.

However, both the estimation of the upper bound and computing its gradient for minimization require Monte Carlo estimators to approximate tough integrals.

These estimators may have large variance even at modest number of samples.

A natural question is then how reliable CUBO minimization is in practice.

In this paper, we provide empirical evidence that CUBO optimization is often tricky, and the bound itself ends up being too loose even Figure 1: Minimizing χ 2 divergence using MC gradient estimates via the reparametrization trick can be challenging even with simple univariate Gaussian distributions.

Each column shows results under a different number of MC samples.

The last column compares ELBO and CUBO traces for S = 10 4 ; diamonds correspond to sanity-check estimator from Eq. (2).

Top row : variational parameter traces with fixed true variance but changing starting mean locations.

Bottom row: same, but with fixed true mean and changing start variance values.

using hundreds of samples.

Our contributions include: i) evaluation of the CUBO in two simple scenarios, and comparison to other bounds to gauge its utility; ii) empirical analysis of CUBO optimization in both scenarios, in terms of convergence rate and sensitivity to the number of samples; iii) review of alternative upper bounds and best practices for diagnosing and testing new bounds.

Let p(x, z) be the joint distribution of observed variables x and latent variables z. Variational inference (VI) approximates the posterior distribution p(z|x) through optimization.

The idea is to posit a family of variational distributions and find the member distribution q(z; λ) which is as close as possible to the true posterior.

Standard VI minimizes the KL divergence D KL q(z; λ)||p(z|x) .

Minimizing the KL divergence is equivalent to maximizing the evidence lower bound (ELBO) on the model evidence log p(x).

Alternatively, χ 2 variational inference (Dieng et al., 2017b) minimizes the χ 2 divergence D KL p(z|x)||q(z; λ) .

This is equivalent to minimizing the following upper bound (CUBO):

The expectation in the CUBO is usually intractable, so we use Monte Carlo samples to construct a biased estimate

where z (1) , . . . , z (S) ∼ q(z; λ).

In this paper, we consider two optimization strategies, both relying on the reparametrization trick (Kingma and Welling, 2013; Rezende et al., 2014; Titsias and Lázaro-Gredilla, 2014) : i) optimizing the CUBO directly in Eq. (1) using biased gradient estimators; ii) optimizing the exponentiated CUBO defined as L EXPCUBO (λ) = exp(2 L CUBO (λ)), whose gradients are unbiased but might suffer from higher variance.

We consider a simple inference scenario: minimizing the divergence between two univariate Gaussian distributions.

We assume no data x, such that the true posterior is just the prior fixed at p(z) .

= N (0, 1).

We consider two cases: a variational distribution q(z;μ,σ 2 ) with fixedσ = 1.0 and varying meanμ = {1, 2, 4, 10}, or the other way around, fixedμ = 0.0 and varyingσ = {0.1, 0.5, 2.0, 10.0}. All experiments were performed using stochastic gradient descent (Bottou, 2010) and grid-searching the learning rate for each different bound independently in a fine grid between 10 −4 and 1.0.

Fig. 1 shows the evolution of the variational parameters over time when minimizing the χ 2 divergence (ChiSq) or maximizing the KL divergence (KL) from different initialization points.

While the KL trajectories always converge to the true values, the ChiSq variational parameters fail to converge for 5 out of the 8 cases when the number of MC samples S = 100.

If we increase the number of samples S to 1M, 3 out of 8 cases still fail to find the true values.

Most alarming, in several cases, e.g., fixed mean and varyingσ initialized at 0.1, the CUBO MC estimator present values below 0 (the true marginal likelihood value), so it is not an upper bound anymore, even with 1M samples.

Appendix A show similar pathological behaviors for the exponentiated CUBO case.

To assess CUBO correctness, consider an alternative MC estimator that samples from the prior p, rather than from q:

where z (1) , . . . , z (S) ∼ p(z).

In general, since CUBO optimization is sensitive to initialization, it is a good practice to do warm initializations, either with MAP estimation or by performing KL optimization first during a few iterations.

We consider applying the CUBO training objective to the Latent Dirichlet Allocation (LDA) topic model (Blei et al., 2003) .

We focus on single-document inference, where the length of the document should directly impact posterior uncertainty about which topics are used.

We assume that there are K = 3 topics and V = 3 vocabulary words.

We are given a set of topic-word probabilities φ where φ kv is the probability of word v under topic k. Each document d is represented by counts of V discrete words or features, x d ∈ Z V + .

These counts are generated via a document-specific mixture of K topics, variational inference.

In particular, we explore two tasks: (i) estimating upper bounds on the marginal likelihood given a fixed q, and (ii) optimizing q to try to improve such bounds.

To assess the reliability of upper bound estimation using approximate distributions, we fit four possible q: one Dirichlet via closed-form updates optimizing the ELBO, and 3 separate Logistic Normal (LN) distributions fit via Monte-Carlo gradient descent steps (see details for each q in the appendix).

The 3 LNs are respectively a cold-started optimization of the ELBO, a warm-started optimization of the CUBO, and a cold-started optimization of the CUBO.

Warm-starting here means that the mean of q is set to the maximum likelihood estimator of the document-topic vector π d , while cold-starting has random parameters not informed by the data.

We hope that these detailed experiments tease apart the impact of initialization and optimization.

In Tab.

1 and Tab.

2, for each q described above, we compare CUBO to an alternative upper bound KLpq, detailed in Appendix B. For each stochastic upper bound estimator, we compute 20 replicates using each 100 samples and 100,000 samples, then report the median of these samples as well as 5-th and 95-th percentile value intervals.

Our conclusions are:

CHIVI parameter estimation often diverges for cold initializations.

We replicated this issue across many settings, as reported in Tab.

1.

CUBO estimators are overconfident.

Increasing sample size widens confidence intervals.

KLpq estimators are better behaved.

Consider Tab.

2's warm-init CUBO row (in Appendix A): At 100 samples the CUBO seems to be within (-1.03, 0.77), but at many more samples, the CUBO interval drops to (-0.86, -0.64), with a new median that is just barely contained in the previous interval.

In contrast, the 100 sample KLpq bound has an interval that shrinks.

ELBO optimization followed by CUBO computation may be enough.

The Dirichlet q optimized for the ELBO but then fitted into a CUBO estimator produces competitive bounds.

This suggests that it may not always be necessary to optimize the CUBO directly.

Table 2 : Topic model case study.

Bounds on marginal likelihood for a "short" toy document under an LDA topic model.

We infer an approximate posterior over doc-topic probabilities for a single document with just 1 word, using either closed-form coordinate ascent updates to fit a Dirichlet q (top row) or MC gradient updates to fit a LogisticNormal q (bottom rows) with 100 samples per gradient step.

Using the final fitted q, we then compute 20 replicates of our stochastic upper bounds on marginal likelihood using either the CUBO or the KLpq estimator (see Appendix B, using S = 10 2 or 10 5 samples for each.

We show the median value and the (5%, 95%) interval.

Appendix B. The "KLpq" bound : reliable but expensive.

Given any approximate posterior q(π d ) parameterized byv d ∈ V, the following is an upper bound on the marginal likelihood: Ji et al. (2010) show that minimizing this bound is equivalent to minimizing KL(p||q), which computes the asymmetric KL divergence in the opposite direction of typical variational methods, which minimize KL(q||p).

We suggest that this bound is a useful comparison point for the CUBO bound.

The "KLpq" upper bound can be approximated using S samples from the posterior

.

For our LDA model, we compute S samples from a Hamiltonian Monte Carlo posterior using Stan (Gelman et al., 2015) .

Because LN random variables are not very common, we write the log probability density function of the approximate posterior here, using results from Aitchison and Shen (1980, Eq. 1.3

The entropy of the distribution is then:

where we have used standard results to simplify that last term:

This expectation E π d ∼q [log π dk ] unfortunately has no closed form.

Reparameterization trick.

We can write the random variable π d as a deterministic transform of a standard normal random variable u d .

First, recall we can map any K − 1-length real vector u ∈ R K−1 to the K-dimensional simplex ∆ K via the softmax transformation:

This transformation is one-to-one invertible, and also differentiable w.r.t.

its input vector.

Now, to generate π d ∈ ∆ K , we can draw π d in three steps: (1) draw u d from a standard normal, (2) scale it with the appropriate mean and standard deviation parameters, and (3) apply the softmax transformation,

C.3.

LDA Optimization #3: Overcomplete-Logistic-Normal + MonteCarloGD

Transformation between overcomplete simplex and the reals We now consider an overcomplete representation of the K-dimensional simplex.

Rather than the minimal K − 1 parameters in the LN-Marg approximation above, let's look at transformations that use K free parameters.

In this overcomplete space, we must augment our probability vector π d ∈ ∆ K (which has only K − 1 degrees of freedom) with an additional scalar real random variable w d ∈ R, so the combined vector [π d1 . . .

π d,K−1 w d ] has the required K linearlyindependent dimensions.

Now, we can create an invertible transformation between two K-length vectors: a vector u of real values, and the augmented pair π, w:

Because this is an invertible transformation, we can compute the Jacobian:

. . .

. . .

Next, we wish to compute the determinant of this Jacobian, as a function of π and w. First, we perform row and column swaps until only the first column and first row have non-diagonal entries, like this:

Here, we have defined the remaining mass beyond the K − 1 independent entries of the vector π as rem(π) = 1 − K−1 k=1 π k for simplicity.

The number of swaps needed to create J from J is always an even number (there will be the some a swaps needed to fix the rows, and then the same number a swaps for the columns, so 2a swaps total).

Each single row or column swap changes the sign of the determinant but not the value.

An even number of swaps thus leaves the determinant unchanged: |J | = |J|.

We can then apply the Schur determinant formula, which says, for any square matrix, we can compute its determinant by manipulating its subcomponent blocks:

The simplification arises via algebra after plugging in the definition of rem(π).

Armed with the Jacobian and its determinant, we have all the tools needed to perform variational inference in this representation.

Approximate posterior: Overcomplete LN.

Returning to our topic modeling task, we consider again the LDA generative model for a document as a given, and wish to compute an approximate posterior for the document-topic vector π d .

We suggest an approximate posterior family based on the overcomplete logistic normal above.

We can draw samples from this in two steps.

This leads to the following log probability density function over the joint space of π, w ∈ ∆ K × R: log q(π, w) = log |det J(π, w)| +

Our generative model does not include the log-scale variable w d , but we can easily just give it a N (0, 1) prior and keep it decoupled from the data.

@highlight

An empirical study of variational inference based on chi-square divergence minimization, showing that minimizing the CUBO is trickier than maximizing the ELBO
Large matrix inversions have often been cited as a major impediment to scaling Gaussian process (GP) models.

With the use of GPs as building blocks for ever more sophisticated Bayesian deep learning models, removing these impediments is a necessary step for achieving large scale results.

We present a variational approximation for a wide range of GP models that does not require a matrix inverse to be performed at each optimisation step.

Our bound instead directly parameterises a free matrix, which is an additional variational parameter.

At the local maxima of the bound, this matrix is equal to the matrix inverse.

We prove that our bound gives the same guarantees as earlier variational approximations.

We demonstrate some beneficial properties of the bound experimentally, although significant wall clock time speed improvements will require future improvements in optimisation and implementation.

One major obstacle to the wider adoption of Gaussian Process (GP) (Rasmussen and Williams, 2006) based models is their computational cost, which is mainly caused by matrix inverses and determinants.

Advances in variational approximate inference methods have reduced the size of the matrices on which expensive operations need to be performed, leading to O N M 2 time costs instead of O N 3 (Titsias, 2009), with approximations arbitrarily good with M N (Burt et al., 2019) .

Minibatches of size B N can be used for training at a cost of O BM 2 + M 3 per iteration (Hensman et al., 2013) .

The usefulness of training with small minibatches is hampered by the iteration cost being dominated by O M 3 , which again comes from an inverse and determinant.

The computation is usually done using the Cholesky decomposition, which requires serial operations and high-precision arithmetic.

So in addition to being an asymptotically expensive operation, it is also poorly suited to modern deep learning hardware.

Removing these per-iteration matrix operations therefore seems necessary to speed up training.

In this work, we provide a variational lower bound that can be computed without expensive matrix operations like inversion.

Our bound can be used as a drop-in replacement to the existing variational method of Hensman et al. (2013 Hensman et al. ( , 2015 , and can therefore directly be applied in a wide variety of models, such as deep GPs (Damianou and Lawrence, 2013) .

We focus on the theoretical properties of this new bound, and show some initial experimental results for optimising this bound.

We hope to realise the full promise in scalability that this new bound has in future work.

We will consider a straightforward model where we want to learn some relation f : X → R with a GP prior through an arbitrary factorised likelihood.

We write the model as

using some abuse of notation for denoting the GP prior.

We need approximate inference to deal with a) the non-conjugate likelihood which prevents a closed-form solution, and b) the O(N 3 ) matrix operations that come from the Gaussian prior.

Our starting point is Hensman et al. (2013 Hensman et al. ( , 2015 , who propose a solution based on variational inference.

An inducing variable posterior is used, which is constructed by conditioning on M random variables u ∈ R M , and then specifying a free-form Gaussian distribution N (µ, S) for them:

where k ·· , k u· , and K uu are covariances between some the function value at some point · or inducing variables u. The inducing variables are commonly taken to be u = {f (z m )} M m=1 , making the covariances simple evaluations of the kernel k(·, · ).

We can minimise the KL divergence between q(f (·)) and p(f (·) | y) (Matthews et al., 2016) by maximising the bound from Hensman et al. (2015) (stochastic variational, "sv"):

where q(f (x n )) = N f n ; µ n , σ 2 n , with

The expensive O(M 3 ) operations are the inverse K uu in the approximate posterior (eq. 2) and KL term, and the log-determinant in the KL term.

In sections 3 and 4 we remove them from the approximate posterior and KL term respectively.

We begin by eliminating inverses from the expected log-likelihood term in eq. 3, which stem from the inverses in the predictive mean µ n and variance σ 2 n (eq. 2).

By reparameterising K −1 uu µ =μ and K −1 uu SK −1 uu =Ŝ, we can trivially get rid of all inverses, except for the term

uu k ufn , which we denote as σ 2 n|u and call the residual variance.

While we cannot similarly remove the inverse in σ 2 n|u , we note in lemma 7 (see appendix A for lemmas, proofs and details) that we can lower-bound the Gaussian expectation by using an upper bound for σ 2 n|u .

Upper bounds to σ 2 n|u were investigated by Gibbs and MacKay (1997) and in follow-on work by Davies (2015) in the context of conjugate gradient (CG) approximations to matrix inversion.

They note that for all values of a n we have

with equality when a n = K −1 uu k ufn (lemma 8).

We parameterise this upper bound as σ

and use it to construct a lower bound to L sv without inverses in the expectation terms:

Proposition 1 For log-concave ("lc") likelihoods, L lc is a valid lower bound to the log marginal likelihood, as

Remark 2 Since we are predicting with a different distribution than is in the KL, log p(y)− L lc is not the KL gap between the approximate and true posteriors.

With L lc we have a bound on the marginal likelihood that we can optimise with respect to the parameters of L sv , with the addition of T. Section 3.3 discusses more properties.

To create a proper variational method which also works with any likelihood, we need to use the same distribution in the predictions as in the KL term.

To do this, we find the q(u) in L sv that would give the µ n andσ 2 n from eq. 7.

We solve for S:

This shows that we can obtain inverse-free predictions using a simple reparameterisation of S in L sv .

This gives a new fully relaxed ("fr") bound L fr , as we are relaxing the optimisation by adding T as an additional variational parameter:

Remark 3 The fully relaxed bound L fr is simply a reparameterisation of L sv from Hensman et al. (2013 Hensman et al. ( , 2015 .

Each setting of T has a setting for L sv that is exactly equivalent.

This means that any model that relies on a variant of L sv for inference can be trivially adapted to use the inverse-free fully relaxed bound, by reparameterising the predictions and KL.

We may worry that additionally optimising over T prevents us from recovering the same result as from L sv , due to additional local optima or gradient variance.

The following two propositions show that this is not the case.

Proposition 5 The variance of ∇ T L lc is zero when T is at its optimum T = K −1 uu .

Here, we remove costly matrix operations from the KL term through unbiased estimation.

We start by highlighting the O(M 3 ) terms needed in L lc (L fr is similar):

The trace term requires full matrix multiplications as we parameteriseŜ = LL T , and computing the determinant typically requires a costly decomposition.

The trace term is dealt with using the The log|K uu | term is more challenging.

Since we use gradient-based optimisation, we will focus on obtaining an unbiased estimate of its gradient.

We follow Filippone and Engler (2015) by starting with the unbiased estimator K −1 = sr T , where s = K −1 uu r and E r [rr T ] = I. We then use the Unbiased LInear Systems SolvEr (ULISSE), a randomly truncated CG run, to compute an unbiased estimate of s. The key insight of ULISSE is that the conjugate gradient method expresses the solution s = K −1 uu r as a sum of separate terms.

The sum is randomly truncated, and each term is re-weighted by the probability of its inclusion to keep an unbiased estimate:

The cost of ULISSE is O(HĪM 2 ), whereĪ = E[I], so we needĪ to be small for the method to be practical, while keeping small gradient variance.

If we parameterise T = L T L T T (with L T lower triangular), we can use it as a preconditioner with the following property:

Proposition 6 When using L T as a preconditioner, ULISSE will converge in a single step when

uu .

This allowsĪ = 1 without adding additional variance.

The hope is that during optimisation T is updated quickly enough to remain close to the current K −1 uu , which would allow us to choose a smallĪ without adding significant variance.

We show the inverse free GP using the log-concave bound in fig. 1 , and a deep GP based on Salimbeni and Deisenroth (2017) and the fully relaxed bound in fig. 2 , both optimising T. We see that in fig. 1 the correct solution is completely recovered, while in fig. 2 a similar fit is achieved with a somewhat lower ELBO.

See appendix B for additional details.

We presented new variational bounds for GP models that function as drop-in replacements to those developed by Hensman et al. (2013 Hensman et al. ( , 2015 , but without needing to compute expensive matrix operations each iteration.

We prove their properties and show that they behave as expected in simple experiments using a single layer and deep GP.

We believe this method to be promising, as it removes the most frequently cited impediment against the scaling of GP models.

However, more improvements are needed to obtain the full practical benefits.

Lemma 7 The Gaussian expectation of a concave function φ(·) is lower-bounded by a Gaussian expectation with the same mean and a larger variance,

Proof We writex = x + , where ∼ N 0,σ 2 and x ∼ N µ, σ 2 .

We can then write the right-hand side of the inequality as a nested expectation:

.

We can move the inner expectation over into the argument of φ(·) by applying Jensen's inequality,

Lemma 8 For any a n , we have

with equality when a n = K −1 uu k ufn .

Proof This follows directly from a T n K uu a n − 2k

T ufn a n + k

Proposition 1 Proof We consider a single term in the sum in eq. 6, L

.

By lemma 7, for a concave log p(y n | f n ), we have

n ≥ σ 2 n , which is ensured byσ 2 n − σ 2 n =σ 2 n|u − σ 2 n|u and eq. 4.

At the optimum T = K −1 uu , the variance upper bound (5) is tight:σ 2 n|u = k fnfn + k

n|u , and we have equality L lc = L sv .

We consider the interesting case where N > M and where our kernels are nondegenerate to avoid underdetermined linear systems.

We first consider the case for L lc .

From proposition 1 we know that L lc = L sv when T = K −1 uu .

Here we additionally show that whenever T = K −1 uu , there is a non-zero gradient.

We begin by showing that the

uu .

Here, we denote n ∼ N (0, 1), and write the expected likelihood term of L lc as

where

We set the gradient w.r.t.

T to zero:

We write ∇ T α n = 1 2 α −1 n ∇ T (α 2 n ), as α 2 n is a quadratic function of T. Taking the sum over all N data points, and given that N > M , this quadratic has a unique optimum at T = K −1 uu , and so a unique point at which the gradients will be zero, at which L lc = L sv .

For L fr , we additionally need to consider the KL term, which is a convex function of S.

So if ∇ T S = 0, then the KL is at a stationary point as well.

In the L fr bound, we have

uu .

Proof We evaluate unbiased estimates of the gradients using the reparameterisation trick, with samples n ∼ N (0, 1), and f n = µ n + (σ 2 n ) 1 2 n (where µ n = k T ufnμ does not depend on T).

For the gradient of a single term L (n) lc = E fn [φ(f n )]

this gives the estimator:

The term k T ufn TK uu Tk ufn − 2k T ufn Tk ufn is a quadratic in T, with an optimum at

T ufn Tk ufn = 0, which makesĝ n = 0, irrespective of n. This implies that the reparameterisation gradient w.r.t.

T is zero whenever T is at its optimum, regardless of which minibatch or n is sampled.

K −1 uu = T = L T L T T , so I = K uu K −1 uu = K uu L T L T T .

Left-

For the examples in section 5 our methods can recover solutions similar to L sv , which computes inverses exactly.

The current set-up optimises all hyperparameters and variational parameters (including T) jointly using Adam (Kingma and Ba, 2014) .

We see in figs. 3 and 5 that the inverse-free methods do require more iterations to achieve a similar ELBO, although the difference in ELBO tends to exaggerate the visual difference in fit quality.

In fig. 4 we visualise the quality of the inverse that is obtained from optimising L lc for the single layer experiment.

We see that all initialisations recover T = K −1 uu almost perfectly.

Future improvements will focus on improved optimisation behaviour, and software improvements.

The optimisation surface becomes more challenging for larger datasets causing Adam to become unstable, so perhaps a different optimisation routine can be used to take advantage of structure in the objective functions.

Software improvements to allow taking advantage of using only matrix-vector products (Gardner et al., 2018) will also be needed to gain the full per-iteration speed-up that this method allows.

<|TLDR|>

@highlight

We present a variational lower bound for GP models that can be optimised without computing expensive matrix operations like inverses, while providing the same guarantees as existing variational approximations.
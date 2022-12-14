We propose that approximate Bayesian algorithms should optimize a new criterion, directly derived from the loss, to calculate their approximate posterior which we refer to as pseudo-posterior.

Unlike standard variational inference which optimizes a lower bound on the log marginal likelihood, the new algorithms can be analyzed to provide loss guarantees on the predictions with the pseudo-posterior.

Our criterion can be used to derive new sparse Gaussian process algorithms that have error guarantees applicable to various likelihoods.

Results in learning theory show that, under some general conditions, minimizing training set loss, also known as empirical risk minimization (ERM), provides good solutions in the sense that the true loss of such procedures is bounded relative to the best loss possible in hindsight.

Alternative algorithms such as structural risk minimization or regularized loss minimization (RLM) have similar guarantees under more general conditions.

On the other hand, Bayesian approaches are, in a sense, prescriptive.

Given prior and data, we calculate a posterior distribution that compactly captures all our knowledge about the problem.

Then, given a prediction task with an associated loss for wrong predictions, we pick the best prediction given our posterior.

This is optimal when the model is correct and the exact posterior is tractable.

However, the algorithmic choices are less clear with misspecified models or, even if the model is correct, when exact inference is not possible and the learning algorithm can only return an approximation to the posterior.

Since the choices are often heuristically motivated we call such approximations pseudo-posteriors.

The question is how the pseudo-posterior should be calculated.

In this paper we propose to use learning theory to guide this process.

To motivate our approach consider the variational approximation which is one of the most effective methods for approximate inference in Bayesian models.

In lieu of finding the exact posterior, variational inference maximizes the ELBO, a lower bound on the marginal likelihood.

It is well known that this can be seen alternatively as performing regularized loss minimization.

For example, in a model with parameters w, prior p(w), and data y where p(y|w, x) = i p(y i |w, x i ), we have log p(y) ??? ELBO E where q(w) is the variational posterior and we have suppressed the dependence on x for visual clarity.

Minimizing the negative ELBO, we have a loss term i E q(w) [??? log p(y i |w, x i )] and a regularization term d KL (q(w), p(w)).

The RLM viewpoint is attractive from the perspective of statistical learning theory because such algorithms are known to have good generalization guarantees (under some conditions).

However, the ELBO objective is not matched to the intended use of Bayesian predictors: given a posterior q(w) and test example x * , the Bayesian predictor first calculates the predictive distribution p(y * |x * ) = E q(w) [p(y * |x * , w)] and then, assuming we are interested in the log loss, suffers the loss ??? log p(y * |x * ).

In other words, seen from the perspective of learning theory, variational inference optimizes for

, which is the loss of the Bayesian predictor.

These observations immediately raise several questions: Should we design empirical risk minimization (ERM) algorithms minimizing L B that produce pseudo-posteriors?

Should a regularization term, e.g., d KL , be added?

Can we use standard analysis, that typically handles frequentist models, to provide guarantees for such algorithms?

We emphasize that this differs from standard non-Bayesian algorithms that perform ERM or RLM to find the best parameter w. Here, we propose to perform ERM or RLM to find the best pseudoposterior q(w) as given by the parameters that define it.

In this paper, we show that such an analysis can indeed be performed, and provide results which are generally applicable to Bayesian predictors optimized using ERM.

Then, we focus on sparse Gaussian processes (sGP) for which we develop risk bounds for a smoothed variant of log loss 1 and any observation likelihood (the non-conjugate case).

The significance of this is conceptual, in that it points to a different principle for designing approximate inference algorithms where we no longer aim to optimize the marginal likelihood (or ELBO), but instead a criterion that is directly related to the loss -this diverges from current practice in the literature.

The paper highlights sparse GP because it is an important model with significant recent interest and work.

But the approach and results are more generally applicable.

To illustrate this point the appendix shows how the results can be applied to the Correlated Topic Model (CTM) of Blei and Lafferty (2006) .

It is important to distinguish this work from two previous lines of work.

Our earlier work (Sheth and Khardon, 2017) made similar observations w.r.t.

the mismatch between the optimization criterion and the intended objective.

However, the goal there was to analyze existing algorithms where possible.

More concretely we showed that optimizing a criterion related to L G does have some risk guarantees, though these are weaker than the ones in this paper.

Here, we propose to explore new algorithms based on direct loss minimization with stronger associated guarantees.

In Alaoui and Mahoney (2015) and Burt et al. (2019) , the goal is to show that the sparse GP approximation can be chosen to be very close to the full GP solution.

Conditions on the kernel functions and on the algorithm to select inducing input locations and variational distribution are given for this to be true.

This is a very strong result showing that nothing is lost by using the sparse approximation.

However, in many cases, the number of inducing inputs required is too large (e.g., for Matern kernels).

In contrast, our analysis aims at identifying the best sGP posterior in terms of the resulting prediction performance, whether it is close to the full GP posterior or not.

In other words, we seek an "agnostic PAC guarantee" for the sparse GP posterior.

Due to space constraints, the main paper sketches the technical results with full details given in Appendices A to E. In short, three different approaches to proving agnostic PAC guarantees for learning with a Lipschitz loss under a bounded hypothesis space are provided.

The three results use slightly different variants of ERM as the optimization algorithm.

All three provide bounds if, in addition, the loss itself is bounded.

Approach 1 (Appendix A) uses this directly and proves bounds using a standard discretization argument.

Approach 2 (Appendix B) requires a bounded loss but adapts results based on Rademacher complexity (Meir and Zhang, 2003) to provide risk bounds that do not depend on the dimension of the hypothesis space and, in this way, potentially improves on approach 1.

Approach 3 (Appendix C), which we present below, is new and has the potential to provide bounds with unbounded losses, although, for the application in this paper, we will be using bounded loss functions.

We stress, though, that any of these approaches can be utilized to obtain guarantees under a Lipschitz loss and bounded hypothesis space.

Appendices D and E develop the details for sGP and CTM.

In the following, we consider a loss : ?? ?? (X, Y ) ??? R over a hypothesis space ?? ??? R M and example/label spaces X and Y .

We assume that the hypothesis space is closed and bounded w.r.t.

infinity norm with sup ??????? ||??|| ??? ??? B. We further assume that is L-Lipschitz in its first argument w.r.t.

the same norm, i.e., ?????,

), U denotes the uniform distribution, and ?? > 0 is a scalar.

The algorithm averages 2 the ERM objective of random neighbors of the solution??.

We have:

where

2.

Given the other approaches described in the appendix, it is reasonable to consider this an artifact of the proof.

In this case, ERM may be used directly.

The proof (Appendix C) uses the compression lemma, ??) , but applied to the variational parameters ?? in contrast with Germain et al. (2016) and Sheth and Khardon (2017) that applied it on w.

This new approach is the source of jitter in the randomized ERM objective.

Specifically we apply the compression lemma with q(??) = q jit (??|?? ERM ) and

.

This bounds the potential overfitting, expressed by 1 ?? f (??), by a KL term that we can compute explicitly and log E p(??) e f (??) which results in ??.

If the loss is bounded, i.e., | | ??? c, then, ??(??, n) ??? 2?? 2 c 2 n (see Germain et al. (2016) ; Sheth and Khardon (2017)) implying the following corollary showing that the expected risk of Randomized ERM is bounded by the risk of any posterior in ?? plus a term that decays at a rate of 1/ ??? n. ??(??, n) can be bounded under some conditions even if the loss is not bounded 3 but we leave further exploration of this for future work.

Corollary 2 If the loss is bounded, i.e., | | ??? c, then using ?? = ??? n we have

In the (zero-mean) sparse GP model of Titsias (2009), w represents the latent function at the M inducing inputs

Here, the pseudo-posterior is given by q(w|??) = N (w|m, C C) and the parameter space ?? includes both the mean and the Cholesky factor of the covariance of the pseudo-posterior, i.e., ?? m vec(C)

.

Given q(w|??), the induced distribution q(f |??) w p(f |w)q(w)dw can be calculated exactly from Gaussian identities.

Then, the log loss of the Bayesian prediction is ((m, C),

U U K U * , and const signifies terms that do not depend on m or C. To apply Corollary 2, we require a bounded loss function which is also Lipschitz w.r.t.

??.

To enable this, we define a "smoothed" log loss.

Assume for now that p ??? ?? < ???. We use a smoothing parameter ?? ??? (0, 1) and define nlog

is Lipschitz w.r.t.

?? and infinity norm yielding that nlog

and ?? min (K U U ) denotes the minimum eigenvalue of K U U .

Therefore, to apply Corollary 2 to any non-conjugate sparse GP model with smoothed log loss, all we need is to (i) verify that ????? s.t.

E q(f * |??) [p(y * |f * )] < ?? and (ii) calculate bounds on d df * p(y * |f * ) and

is easily achieved when Y is discrete, e.g., for binary classification and count regression.

For standard regression, we can guarantee this by lower bounding the noise variance ?? 2 Y and upper bounding the range of X, Y .

Bounds on the first and second derivatives (condition (ii)) are easily derived for the same likelihoods.

Corollary 3 Randomized ERM using smoothed log loss with the sparse GP predictive distribution enjoys the bounds of Corollary 2 for regression, binary classification, Poisson regression.

We have shown that ERM-type algorithms performing direct minimization of log loss have strong performance guarantees for the Bayesian predictor, and we applied these results to the non-conjugate sparse GP model under a smoothed log loss.

However, in some scenarios, we may want to minimize a different loss function requiring an explicit prediction.

In this case, given a posterior q(w) and example x with label y true , the Bayesian predictor first identifies the optimal prediction?? =?? q(w) (x) = arg min y???Y E q(w)p(y |x,w) [ (y, y )] and then suffers the loss (q(w), (x, y true )) = (?? q(w) (x), y true ).

Therefore, the natural loss term for optimization is L B = i (?? q(w) (x i ), y i ).

We note that L G from the introduction, which implicitly uses the Gibbs log loss, is even less directly related to the learning goal in this case.

On the other hand, the results of this paper do potentially apply to this more general setting as long as the conditions for the theorem hold.

Our theory does not directly apply to the square loss (?? ??? y) 2 because of the need for smoothing.

However, it is interesting to consider the use of DLM for square loss and the resulting algorithms.

In this case, sGP uses the standard regression model with Gaussian noise for prediction, that is, for calculating??.

It is well known that for the square loss the optimal predictor is the mean of the predictive distribution.

As discussed above, for sGP the mean of the predictive distribution on example i is equal to a i m where

Therefore the ERM algorithm will minimize i (a i m???y i ) 2 and similarly for the randomized ERM.

We therefore see that, if we do not use regularization, the optimization criterion does not depend on the covariance of w and the optimization simplifies into a sparse variant of kernel least squares.

The role of the posterior covariance, might become apparent with regularization, which might also be helpful to reduce some of the conditions in our theorem.

We leave the derivation of DLM algorithms and performance guarantees for other loss functions to future work.

The paper points out the potential of DLM to yield a new type of approximate pseudoBayesian algorithm.

In this paper we focused on the analysis of ERM and application to sparse GP.

There are many important questions for future work including analysis for RLM, analysis for hyperparameter selection, removing the need for bounded or smoothed loss in our theorem, and investigating empirical properties of these algorithmic variants.

This straightforward proof shows that having a Lipschitz condition and bounded loss are sufficiently strong to make the problem simple by essentially learning on a grid.

We include it here in order to put the other proofs and their potential improvements in context.

Let ?? ??? R M and || ?? || denote the infinity norm.

Recall that we assume a bounded loss for the application of the discretization approach, i.e., | | ??? c. Since ?? is assumed bounded, there exists a finite ??-cover of ??,??, i.e., ????? ??? ??, ????? ????? s.t.

||?? ?????|| ??? ??.

Let

For an arbitrary ?? ??? ??, let?? denote the closest point in?? to ??.

Since the loss is assumed to be L-Lipschitz in the hypothesis parameter, we have that

In addition, by combining the union bound and Hoeffding's bound for bounded loss | | ??? c we have that, with probability ??? 1 ??? ?? over the choice of sample S, for all ?? ?????:

Let ?? be any competitor for the posterior parameters.

With probability ??? 1 ??? ?? we have

??? E (x,y)???D (??, (x, y)) + 2c 2 log(2|??|/??) n + 2L??

where (e) follows because ERM minimizes training set loss.

With |??| ??? 2B ?? M , the terms on the RHS of (8) depending on ?? are given by 2c 2 log(2/??) + 2M log(

The last expression is optimized when ?? = 4cM 3 ??? nL

.

Hence, we have that, with probability ??? 1 ??? ?? over the choice of S, ????? ??? ??,

Appendix B. Rademacher complexity

In this section, we show how the result of Meir and Zhang (2003) can be adapted to handle Bayesian predictors.

Meir and Zhang (2003) assume a set of parameterized predictors h(x; w) : X ??? Y and, in addition, assume that predictions can be averaged so that E q(w|??) [h(x; w)] is a meaningful prediction.

One can then apply the loss (y, E q(w|??) [h(x; w)]).

For Bayesian predictors, we average the probabilities inp = E q(w|??) [p(y|x, w)] but not the predictions themselves.

Nonetheless, the same proof technique can be adapted to yield a result for some loss functions, specifically the smoothed log loss discussed in the main paper.

We next develop the details.

Note that, although the results of Meir and Zhang (2003) are for unbounded losses, their conditions are complex and it is not clear how to apply these results directly for Bayesian predictors such as the sparse GP discussed in this paper.

Assuming a family of distributions Q over w and an upper bound p(y|x, w) ??? p y|w (where p y|w is a constant), uniform convergence for the averaged predictor E q(w) [p(y|x, w) ] under the smoothed log loss nlog (??) () will be shown.

From Theorem 26.5.1 of Shalev-Shwartz and Ben-David (2014) 4 , for all q ??? Q, the following holds with probability 1 ??? ?? over the choice of S:

where H {E q(w) [p(??; w, ??)] : q ??? Q}, ??? stands for function composition,

4.

See also Corollary 4 of Meir and Zhang (2003) .

These results give one-sided bounds but can be easily adapted to give the two sided bound shown here.

for Rademacher variables ??, and c = max{| log(??)|,

Next, we slightly adapt the argument outlined in Sections 5 and 6.1 of Meir and Zhang (2003) .

Fix some constant ?? ??? (0, ???) and sample-independent distribution p(w) over w. By applying the compression lemma (Banerjee, 2006)

where (11) follows from the inequality E ?? i exp(?? i a i ) ??? exp(a 2 i /2) (Lemma A.6 of ShalevShwartz and Ben-David (2014)), and we have defined A sup q???Q KL(q(w), p(w)).

Opti-

.

Substituting this value in (12) results in

Utilizing (13) in (10), we have that with probability 1 ??? ?? over the choice of S, for all q ??? Q,

Defining Q A {q ??? Q s.t.

KL(q, p) ??? A}, and the ERM hypothesis as

we can use the above with the standard argument for ERM to get that, with probability 1 ??? ?? over the choice of S, for all q ??? Q A ,

Applications of this results to sparse GP are possible as outlined in the main paper.

Comparing this result to the discretization proof and randomization proof (below), we see that the requirements for Lipschitz constants are weaker.

Here, we only need L (??) whereas other proofs require a Lipschitz condition w.r.t.

the parameter ??.

This proof can potentially yield bounds that do not depend on the dimension M .

Note that, applied to Gaussian distributions, A implicitly depends on M , so a direct application does include such a dependence.

But Meir and Zhang (2003) show how to use structural risk minimization to get around this dimension dependence through data-dependent bounds.

Let?? denote some known subset of ??, i.e.,?? ??? ??, and let {q jit (??|??)} denote a family of distributions over ?? parameterized by members of the subset??. The members of the family are as yet unspecified, but represent "jitter" distributions which will be defined shortly.

Let

Note, where we exchange order of expectations in the following development, we assume the conditions of Fubini's theorem are met.

The following lemma is standard (see ShalevShwartz and Ben-David (2014) ):

Proof It is sufficient to prove that

holds for all?? ?????: Since?? ERM is the ERM hypothesis, it follows that ????? ?????,

Taking expectations of both sides w.r.t.

D n yields the result.

The following lemma uses a technique from Germain et al. (2016) and Sheth and Khardon (2017) .

The novelty, however, is to apply the compression lemma at a level higher than previous work.

Here, we use it at the level of parameters ?? defining the posterior distribution, which requires us to introduce the jitter, whereas previous work applied it at the level of base parameter w.

This gives a qualitatively different result.

Lemma 2.

Let p(??) be any sample-independent distribution over ?? and define

Then, ????? ?????,

Proof First, apply Fubini's theorem to change the order of expectations of

with q(??) = q jit (??|?? ERM ) and f (??) = ?? E (x,y)???D (??, (x, y)) ??? 1 n n i=1 (??, (x i , y i )) to the resulting expression within the expectation w.r.t.

S ??? D n .

Finally, take the expectation w.r.t.

S ??? D n and note that E S???D n log(??) ??? log E S???D n (??) by Jensen's inequality to yield the statement of the lemma.

Lemma 3.

Let some norm || ?? || over ?? be given.

For an L-Lipschitz function (??) w.r.t.

|| ?? ||, we have that ????? ?????, ????? ??? ??,

Since this holds for all values of ?? (given some ?? ), it also holds in expectation over any distribution in ??, specifically q jit (??|??).

Lemma 4.

Let p(??) be any sample-independent distribution over ?? and (??, (x, y)) be L-Lipschitz in its first argument.

Then, ????? ??? ??,

where D max ?? ?????\?? min?? ????? ||?? ??? ?? || and ??(??, n) is defined in Lemma 2.

Proof Following from the left inequality of (18) with?? = ?? =?? ERM , we have

Utilizing (20) in (17) yields that, ????? ?????,

Following from the right inequality of (18), we have that, ????? ?????, ????? ??? ??,

Utilizing (22) in (21) yields that, ????? ?????, ?? ??? ??,

Next, we develop a uniform bound over ?? for the term E q jit (??|??) ||?? ??? ?? || .

Since (23) holds for all?? ?????, we consider how?? can be selected per ?? ??? ??. First, note that

Now, when ?? ?????, a uniform bound over?? for E q jit (??|??) ||?? ?????|| translates to a uniform bound over ?? for E q jit (??|??) ||?? ??? ?? || since it is possible to select?? = ?? in (24).

When ?? ??? ??\??, the distance to the "closest" point in?? to ?? is min?? ????? ||?? ??? ?? ||.

Then, the second term of (24) is uniformly upper-bounded over ?? by D max ?? ?????\?? min?? ????? ||?? ??? ?? ||.

Combining the two cases yields the lemma.

Jitter distributions.

First, we define ?? parametrically as a function of some ?? > 0 and relative to??. Assume?? is a subset of some space T (equipped with norm || ?? ||), and define ?? as the set {?? ??? T s.t.

min?? ????? ||?? ?????|| ??? ??}. Then,?? ??? ??, and D ??? ??.

The ??-ball centered at?? ????? is denoted B ?? (??) {?? ??? ?? s.t.

||?? ?????|| ??? ??}. Let the jitter distribution q jit (??|??) be defined as the following uniform density with support in ??:

q jit (??|??) = .

For this choice of jitter distribution and prior, (19) becomes

If vol(B ?? (??)) = ???? M for some constants ??, M , then the RHS of (26) be the 1-norm.

Following standard Gaussian (see e.g., Rezende et al. (2014) ) and matrix

@highlight

This paper utilizes the analysis of Lipschitz loss on a bounded hypothesis space to derive new ERM-type algorithms with strong performance guarantees that can be applied to the non-conjugate sparse GP model.
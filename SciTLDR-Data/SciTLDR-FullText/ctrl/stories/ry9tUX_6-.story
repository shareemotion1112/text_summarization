We show that Entropy-SGD (Chaudhari et al., 2017), when viewed as a learning algorithm, optimizes a PAC-Bayes bound on the risk of a Gibbs (posterior) classifier, i.e., a randomized classifier obtained by a risk-sensitive perturbation of the weights of a learned classifier.

Entropy-SGD works by optimizing the bound’s prior, violating the hypothesis of the PAC-Bayes theorem that the prior is chosen independently of the data.

Indeed, available implementations of Entropy-SGD rapidly obtain zero training error on random labels and the same holds of the Gibbs posterior.

In order to obtain a valid generalization bound, we show that an ε-differentially private prior yields a valid PAC-Bayes bound, a straightforward consequence of results connecting generalization with differential privacy.

Using stochastic gradient Langevin dynamics (SGLD) to approximate the well-known exponential release mechanism, we observe that generalization error on MNIST (measured on held out data) falls within the (empirically nonvacuous) bounds computed under the assumption that SGLD produces perfect samples.

In particular, Entropy-SGLD can be configured to yield relatively tight generalization bounds and still fit real labels, although these same settings do not obtain state-of-the-art performance.

Optimization is central to much of machine learning, but generalization is the ultimate goal.

Despite this, the generalization properties of many optimization-based learning algorithms are poorly understood.

The standard example is stochastic gradient descent (SGD), one of the workhorses of deep learning, which has good generalization performance in many settings, even under overparametrization BID36 , but rapidly overfits in others BID44 .

Can we develop high performance learning algorithms with provably strong generalization guarantees?

Or is their a limit?In this work, we study an optimization algorithm called Entropy-SGD BID10 , which was designed to outperform SGD in terms of generalization error when optimizing an empirical risk.

Entropy-SGD minimizes an objective f : R p → R indirectly by performing (approximate) stochastic gradient ascent on the so-called local entropy DISPLAYFORM0 where C is a constant and N denotes a zero-mean isotropic multivariate normal distribution on R p .Our first contribution is connecting Entropy-SGD to results in statistical learning theory, showing that maximizing the local entropy corresponds to minimizing a PAC-Bayes bound BID31 on the risk of the so-called Gibbs posterior.

The distribution of w + ξ is the PAC-Bayesian "prior", and so optimizing the local entropy optimizes the bound's prior.

This connection between local entropy and PAC-Bayes follows from a result due to Catoni (2007, Lem.

1.1.3) in the case of bounded risk.

(See Theorem 4.1.)

In the special case where f is the empirical cross entropy, the local entropy is literally a Bayesian log marginal density.

The connection between minimizing PACBayes bounds under log loss and maximizing log marginal densities is the subject of recent work by BID19 .

Similar connections have been made by BID45 ; Zhang (2006b) ; BID20 ; BID21 .Despite the connection to PAC-Bayes, as well as theoretical results by Chaudhari et al. suggesting that Entropy-SGD may be more stable than SGD, we demonstrate that Entropy-SGD (and its corresponding Gibbs posterior) can rapidly overfit, just like SGD.

We identify two changes, motivated by theoretical analysis, that suffice to control generalization error, and thus prevent overfitting.

The first change relates to the stability of optimizing the prior mean.

The PAC-Bayes theorem requires that the prior be independent of the data, and so by optimizing the prior mean, Entropy-SGD invalidates the bound.

Indeed, the bound does not hold empirically.

While a PAC-Bayes prior may not be chosen based on the data, it can depend on the data distribution.

This suggests that if the prior depends only weakly on the data, it may be possible to derive a valid bound.

We formalize this intuition using differential privacy BID13 BID17 .

By modifying the cross entropy loss to be bounded and replacing SGD with stochastic gradient Langevin dynamics (SGLD; BID43 , the data-dependent prior mean can be shown to be (ε, δ )-differentially private BID42 BID34 .

We refer to the SGLD variant as Entropy-SGLD.

Using results connecting statistical validity and differential privacy (Dwork et al., 2015b, Thm.

11) , we show that an ε-differentially private prior mean yields a valid, though looser, generalization bound using the PAC-Bayes theorem. (See Theorem 5.4.)A gap remains between pure and approximate differential privacy.

Under some technical conditions, in the limit as the number of iterations diverges, the distribution of SGLD's output is known to converge weakly to the corresponding stationary distribution, which is the well-known exponential mechanism in differential privacy (Teh, Thiery, and Vollmer, 2016, Thm. 7) .

Weak convergence, however, falls short of implying that SGLD achieves pure ε-differential privacy.

We proceed under the approximation that SGLD enjoys the same privacy as the exponential release mechanism, and apply our ε-differentially private PAC-Bayes bound.

We find that the corresponding 95% confidence intervals are reasonably tight but still conservative in our experiments.

While the validity of our bounds are subject to our approximation, the bounds give us a view as to the limitations of combining differential privacy with PAC-Bayes bounds: when the privacy of Entropy-SGLD is tuned to contribute no more than 2ε 2 × 100 ≈ 0.2% to the generalization error, the test error of the learned network is 3-8%, which is approximately 5-10 times higher than the state of the art, which for MNIST is between 0.2-1%, although the community has almost certainly overfit its networks/learning rates/loss functions/optimizers to MNIST.

We return to these points in the discussion.

The second change pertains to the stability of the stochastic gradient estimate made on each iteration of Entropy-SGD.

This estimate is made using SGLD.

(Hence Entropy-SGD is SGLD within SGD.)

Chaudhari et al. make a subtle but critical modification to the noise term in SGLD update: the noise is divided by a factor that ranges from 10 3 to 10 4 .

(This factor was ostensibly tuned to produce good empirical results.)

Our analysis shows that, as a result of this modification, the Lipschitz constant of the objective function is approximately 10 6 -10 8 times larger, and the conclusion that the Entropy-SGD objective is smoother than the original risk surface no longer stands.

This change to the noise also negatively impacts the differential privacy of the prior mean.

Working backwards from the desire to obtain tight generalization bounds, we are led to divide the SGLD noise by a factor of only 4 √ m, where m is the number of data points.

(For MNIST, 4 √ m ≈ 16.)

The resulting bounds are nonvacuous and tighter than those recently published by BID18 , although it must be emphasized that the bound presented here hold subject to the approximation concerning privacy of the prior mean, which is certainly violated but to an unknown degree.

We begin with a review of some related work, before introducing sufficient background so that we can make a formal connection between local entropy and PAC-Bayes bounds.

We then introduce a differentially private PAC-Bayes bound.

In Section 6, we present experiments on MNIST which provide evidence for our theoretical analysis.

(Empirical validation is required in order to address the aforementioned gap between pure and approximate differential privacy.)

We close with a short discussion.

This work was inspired in part by BID44 , who highlight empirical properties of SGD that were not widely appreciated within the theory community, and propose a simple linear model to explain the phenomenon.

They observe that, without regularization, SGD can achieve zero training error on MNIST and CIFAR, even if the labels are chosen uniformly at random.

At the same time, SGD obtains weights with very small generalization error with the original labels.

The first observation is strong evidence that the set of classifier accessible to SGD within a reasonable number of iterations is extremely rich.

Indeed, with probability almost indistinguishable from one, fitting random labels on a large data set implies that the Rademacher complexity of this effective hypothesis class is essentially the maximum possible (Bartlett and Mendelson, 2003, Thm. 11) .The second observation suggests that SGD is performing some sort of capacity control.

Zhang et al. show that SGD obtains the minimum norm solution for a linear model, and thus performs implicit regularization.

They suggest a similar phenomenon may occur when using SGD to training neural networks.

Indeed, earlier work by Neyshabur, Tomioka, and Srebro (2014) observed similar phenomena and argued for the same point: implicit regularization underlies the ability of SGD to generalize, even under massive overparametrization.

Subsequent work by Neyshabur, Tomioka, and Srebro (2015) introduced "path" norms as a better measure of the complexity of ReLU networks.

Despite progress, these new norms have not yet lead to nonvacuous generalization bounds (Dziugaite and Roy, 2017, App.

D).There has been recent progress: Dziugaite and Roy (2017) describe PAC-Bayes bounds, built by perturbing the weights learned by SGD.

(The authors were motivated in part by Entropy-SGD and empirical findings relating to "flat minima".)

Their bounds are controlled by 1) the "flatness" of empirical risk surface near the SGD solution and 2) the L2 distance between the learned weights and the random initialization.

The bounds are also found to be numerically nonvacuous.

(We return to this aspect below.)

Similar bounds are studied in further depth by Neyshabur et al. (2017b) .

Recent advances have also identified new spectral norm bounds that correlate closely with generalization error and distinguish between true and random labels BID4 BID38 .Our work and Entropy-SGD both connect to early work by BID22 and BID23 , which introduced regularization schemes based on information-theoretic principles.

These ideas, now referred to as "flat minima", were related to minimizing PAC-Bayes bounds by BID18 , although these bounds are minimized with respect to the posterior, not the prior, as is done by Entropy-SGD.

BID1 provide an informationtheoretic argument for a generalization of the objective of Hinton and Camp.

Their objective takes the form of regularized empirical cross entropŷ DISPLAYFORM0 where Q and P are the prior and posterior on the weights, respectively.

For an appropriate range of β , linear PAC-Bayes bounds are exactly of this form.

In Achille and Soatto (2017) they empirically observe that varying β correlates with a degree of overfitting on a random label dataset.

BID1 also highlight the connections with variational inference BID25 .Our work also relates to renewed interest in nonvacuous generalization bounds BID26 BID27 , i.e., bounds on the numerical difference between the unknown classification error and the training error that are (much) tighter than the tautological upper bound of one.

Recently, Dziugaite and Roy (2017) demonstrated nonvacuous generalization bounds for random perturbations of SGD solutions using PAC-Bayes bounds for networks with millions of weights.(The algorithm can be viewed as variational dropout BID25 , with a proper data-dependent prior but without local reparametrization.)

Their work builds on the core insight demonstrated nearly 15 years ago by BID27 , who computed nonvacuous bounds for neural networks five orders of magnitude smaller.

A key aspect of our analysis relies on the stability of a data-dependent prior.

Stability has long been understood to relate to generalization BID8 .

Our analysis of Entropy-SGLD rests on results in differential privacy (see (Dwork, 2008) for a survey) and its connection to generalization BID17 BID16 BID7 BID40 , which can be viewed as a particularly stringent notion of stability.

Entropy-SGLD is an instance of differentially private empirical risk minimization, which is well studied, both in the abstract BID11 BID24 BID6 and in the particular setting of private training via SGD BID6 BID0 .

Our analysis also relates to the differential privacy of Bayesian and Gibbs posteriors, and approximate sampling algorithms BID35 BID6 BID12 BID42 BID34 .In effect, our differentially private PAC-Bayes bound uses a data-distribution-dependent prior, which are permitted in the PAC-Bayesian framework. (Priors must be independent of the data sample, however.

Differential privacy allows us to extract information about the distribution from a sample while maintaining statistical validity BID17 .)There is a growing body of work in the PAC-Bayes literature on data-distribution-dependent priors.

Write S for a data sample and Q(S) for a data-dependent PAC-Bayesian posterior (i.e., Q : Z m → M 1 (R p ) is a fixed learning algorithm for a randomized classifier).

BID9 makes an extensive study of data-distribution-dependent priors of the form P * = P * (Q) DISPLAYFORM1 .

While such priors were known to minimize the KL term in expectation, Catoni was the first to derive PAC-Bayes excess risk bounds using these priors: focusing on Gibbs posteriors Q(S) = Q P (S) def = P exp(−τR S ) for some fixed measure P, Catoni derives bounds on the complexity term KL(Q P (S)||P * (Q P )) that hold uniformly over all possible data distributions D. Catoni calls such priors and bounds "local".

BID30 extend this approach to generalization bounds and consider both data-independent and data-dependent choices for P. In the later case, P = P(S) and the generalization bound uses the local prior P * (Q P ) = E S∼D m [Q P(S) (S)].

In our work, we make a data-dependent but private choice of the prior P = P(S), and then use our differentially private PAC-Bayes generalization bound to control the generalization error of the associated Gibbs posterior Q P (S) in terms of KL(Q P (S)||P).

We also evaluated differentially private versions of local bounds, where the complexity term is a uniform bound on KL(Q P (S)||P * (Q P )).

The bounds were virtually indistinguishable, and so we do not report them here.3 PRELIMINARIES: SUPERVISED LEARNING, ENTROPY-SGD, AND PAC-BAYES Let Z be a measurable space, let D be an unknown distribution on Z, and consider the batch supervised learning setting under a loss function bounded below: having observed S ∼ D m , i.e., m independent and identically distributed samples from D, we aim to choose a predictor, parameterized by weight vector w ∈ R p , with minimal risk DISPLAYFORM2 where : R p × Z → R is measurable and bounded below.

(We ignore the possibility of constraints on the weight vector for simplicity.)

We will also consider randomized predictors, represented by probability measures Q ∈ M 1 (R p ) on R p , whose risks are defined via averaging, DISPLAYFORM3 where the second equality follows from Fubini's theorem and the fact that is bounded below.

Let S = (z 1 , . . .

, z m ) and letD DISPLAYFORM4 δ z i be the empirical distribution.

Given a weight distribution Q, such as that chosen by a learning algorithm on the basis of data S, its empirical risk DISPLAYFORM5 will be studied as a stand-in for its risk, which we cannot compute.

WhileR S (Q) is easily seen to be an unbiased estimate of R D (Q) when Q is independent of S, our goal is to characterize the (one-sided) generalization error R D (Q) −R S (Q) when Q is random and dependent on S.One of our focuses will be on classification, where Z = X × K, with K a finite set of classes/labels.

A product measurable (in practice, continuous) function f : DISPLAYFORM6 In this setting, 0-1 loss corresponds to g(y , y) = 1 if and only if y = y. In binary classification, we take K = {0, 1}.We will also consider parametric families of probability-distribution-valued classifiers f : DISPLAYFORM7 For every input x ∈ X, the output f (w, x) specifies a probability distribution on K. In this setting, (w, (x, y)) = g( f (w, x), y) for some g : DISPLAYFORM8 The standard loss is then the cross entropy, given by g((p 1 , . . .

, p K ), y) = − log p y .

(Under cross entropy loss, the empirical risk is, up to a multiplicative constant, a negative log likelihood.)

In the special case of binary classification, the output can be represented simply by an element of [0, 1], i.e., the probability the label is one.

The binary cross entropy, BCE , is given by g(p, y) = −y log(p) − (1 − y) log(1 − p).

Note that cross entropy loss is merely bounded below.

We will consider bounded modifications in Appendix B.2.We will sometimes refer to elements of R p and M 1 (R p ) as classifiers and randomized classifiers, respectively.

Likewise, we will often refer to the (empirical) risk as the (empirical) error.

Entropy-SGD is a gradient-based learning algorithm proposed by BID10 as an alternative to stochastic gradient descent on the empirical risk surfaceR S .

The authors argue that Entropy-SGD has better generalization performance and provide some empirical evidence.

Part of that argument is a theoretical analysis of the smoothness of the local entropy surface that Entropy-SGD optimizes in place of the empirical risk surface, as well as a uniform stability argument that they admit rests on assumptions that are violated, but to a small degree empirically.

As we have mentioned in the introduction, Entropy-SGD's modifications to the noise term in SGLD result in much worse smoothness.

We will modify Entropy-SGD in order to stabilize its learning and, up to some approximations, provably control overfitting.

Entropy-SGD is stochastic gradient ascent applied to the optimization problem:arg max DISPLAYFORM0 The objective F γ,τ (·; S) is known as the local entropy, and can be viewed as the log partition function of the unnormalized probability density function DISPLAYFORM1 (We will denote the corresponding distribution by G w,S γ,τ .)

Assuming that one can exchange differentiation and integration, it is straightforward to verify that DISPLAYFORM2 and then the local entropy F γ,τ (·; S) is even differentiable, even if the empirical riskR S is not.

Indeed, Chaudhari et al. show that the local entropy and its derivative are Lipschitz.

Chaudhari et al. argue informally that maximizing the local entropy leads to "flat minima" in the empirical risk surface, which several authors BID22 BID23 BID2 BID3 have argued is tied to good generalization performance (though none of these papers gives generalization bounds, vacuous or otherwise).

1 Chaudhari et al. propose a Monte Carlo estimate of the gradient, DISPLAYFORM3 1 The local entropy should not be confused with the smoothed risk surface obtained by convolution with a Gaussian kernel: in that case, every point on this surface represents the average risk of a network obtained by perturbing the network parameters according to a Gaussian distribution.

The local entropy also relates to a perturbation, but the perturbation is either accepted or rejected based upon its relative performance (as measured by the exponentiated loss) compared with typical perturbations.

Thus the local entropy perturbation concentrates on regions of weight space with low empirical risk, provided they have sufficient probability mass under the distribution of the random perturbation.

Section 4 yields further insight into the local entropy function.

Input: DISPLAYFORM0 w , µ ← w 3:for i ∈ {1, ..., L} do Run SGLD for L iterations.4: DISPLAYFORM1 Entropy-SGLD onlyStep along stochastic local entropy ∇ BID43 , which generates an exact sample in the limit of infinite computation and requires that the empirical risk be differentiable.

2 The final output of Entropy-SGD is the deterministic predictor corresponding to the final weights w * achieved by several epochs of optimization.

Algorithm 1 gives a complete description of the stochastic gradient step performed by Entropy-SGD.

If we rescale the learning rate, η ← 1 2 η τ, lines 6 and 7 are equivalent to DISPLAYFORM2 Notice that the noise term is multiplied by a factor of 2/τ.

This follows from the definition of the local entropy.

A multiplicative factor ε-called the "thermal noise", but playing exactly the same role as 2/τ here-appears in the original description of the Entropy-SGD algorithm given by Chaudhari et al. However, ε does not appear in the definition of local entropy used in their stability analysis.

Our derivations highlights that the scaling the noise term in SGLD update has a profound effect: the thermal noise exponentiates the density that defines the local entropy.

The smoothness analysis of Entropy-SGD does not take into consideration the role of ε, which is critical because Chaudhari et al. take ε to be as small as 10 −3 and 10 −4 .

Indeed, the conclusion that the local entropy surface is smoother no longer holds.

We will see that τ controls the differential privacy and thus the generalization error of Entropy-SGD.

Let Q, P be probability measures defined on R p , assume Q is absolutely continuous with respect to P, and write dQ dP : R p → R + ∪ {∞} for some Radon-Nikodym derivative of Q with respect to P. Then the Kullback-Liebler divergence (or relative entropy) of P from Q is defined to be DISPLAYFORM0 For p, q ∈ [0, 1], we will abuse notation and define DISPLAYFORM1 where B(p) denotes the Bernoulli distribution on {0, 1} with mean p.

We now present a PAC-Bayes theorem, first established by BID31 .

We focus on the setting of bounding the generalization error of a (randomized) classifier on a finite discrete set of labels K. The following variation is due to BID28 for 0-1 loss (see also BID26 and BID9 .)

Theorem 3.1 (PAC-Bayes BID31 BID28 ).

Under 0-1 loss, for every δ > 0, m ∈ N, distribution D on R k × K, and distribution P on R p , DISPLAYFORM2 We will also use the following variation of a PAC-Bayes bound, where we consider any bounded loss function.

Theorem 3.2 (Linear PAC-Bayes Bound (McAllester, 2013; BID9 ).

Fix λ > 1/2 and assume the loss takes values in an interval of length L max .

For every δ > 0, m ∈ N, distribution D on R k × K, and distribution P on R p , DISPLAYFORM3 We introduce several additional generalization bounds when we introduce differential entropy.

We now present our first contribution, a connection between the local entropy and PAC-Bayes bounds.

We begin with some notation for Gibbs distributions.

For a measure P on R p and function g : R p → R, let P[g] denote the expectation g(h)P(dh) and, provided P[g] < ∞, let P g denote the probability measure on R p , absolutely continuous with respect to P, with Radon-Nikodym derivative DISPLAYFORM0 .

A distribution of the form P exp(−τg) is generally referred to as a Gibbs distribution.

In the special case where P is a probability measure, we call P exp(−τR S ) a "Gibbs posterior".

for some λ > 1/2, and let P be a multivariate normal distribution with mean w and covariance matrix (τγ) −1 I p .

Then maximizing the local entropy F γ,τ (w; S) with respect to w is equivalent to minimizing a linear PAC-Bayes bound (Theorem 3.2) on the risk R D (G w,S γ,τ ) of the Gibbs posterior G w,S γ,τ = P exp(−τR S ) , where the bound is optimized with respect to the mean w of P.Proof.

Let m, δ , D, and P be as in Theorem 3.1 and let S ∼ D m .

The linear PAC-Bayes bound (Theorem 3.2) ensures that for any fixed λ > 1/2 and bounded loss function, with probability at least 1 − δ over the choice of S, the bound DISPLAYFORM1 holds for all Q ∈ M 1 (R p ).

Minimizing the upper bound on the risk R D (Q) of the randomized classifier Q is equivalent to the program DISPLAYFORM2 with r(h) = m λ L maxR S (h).

By (Catoni, 2007, Lem.

1.1.3) , for all Q ∈ M 1 (R p ) with KL(Q||P) < ∞, DISPLAYFORM3 Using Eq. (16), we may reexpress Eq. (15) as DISPLAYFORM4 By the nonnegativity of the Kullback-Liebler divergence, the infimum is achieved when the KL term is zero, i.e., when Q = P exp(−r) .

Then DISPLAYFORM5 Finally, it is plain to see that F γ,τ (w; DISPLAYFORM6 , and P = N (w, (τγ) −1 I p ) is a multivariate normal with mean w and covariance matrix (τγ) −1 I.The analysis falls short when the loss function is unbounded, because the PAC-Bayes bound we have used applies only to bounded loss functions.

BID19 described PAC-Bayes generalization bounds for unbounded loss functions. (See BID21 for related work on excess risk bounds and further references).

For their bounds to be evaluated on the negative log likelihood loss, one needs some knowledge of the data distribution in order to approximate certain statistics of the deviation of the empirical riskR S (w) from true risk R D (w).5 DATA-DEPENDENT PAC-BAYES PRIORS VIA DIFFERENTIAL PRIVACY Theorem 4.1 reveals that Entropy-SGD is optimizing a PAC-Bayes bound with respect to the prior.

As a result, the prior P depends on the sample S, and the hypotheses of the PAC-Bayes theorem (Theorem 3.1) are not met.

Naively, it would seem that this interpretation of Entropy-SGD cannot explain its ability to generalize.

Using tools from differential privacy BID13 , we show that if the prior term is optimized in a differentially private way, then a PAC-Bayes theorem still holds, at the cost of a slightly looser bound.

We will assume basic familiarity with differential privacy, but give basic definitions and results in Appendix A. We use the notation A : Z T for a (randomized) algorithm that takes as input an element in Z and produces an output in T .The key result we will employ is due to Dwork et al. (2015b, Thm. 11 ).

Theorem 5.1.

Let m ∈ N, let A : Z m T , let D be a distribution over Z, let β ∈ (0, 1), and, for each t ∈ T , fix a set R(t) ⊆ Z m such that P S∼D m (S ∈ R(t)) ≤ β .

If A is ε-differentially private for ε ≤ ln(1/β )

/(2m), then P S∼D m (S ∈ R(A (S))) ≤ 3 β .Using Theorem 5.1, one can compute tail bounds on the generalization error of fixed classifiers, and then, provided that a classifier is learned from data in a differentially private way, the tail bound holds on the classifier, with less confidence.

The following two tail bounds are examples of this idea.

The first is a simple variant of (Dwork et al., 2015b, Thm.

9) due to Oneto, Ridella, and Anguita (2017, Lem.

2).

Theorem 5.2.

Let m ∈ N and let A : Z m R p be ε-differentially private.

Then Oneto, Ridella, and Anguita, 2017, Lem.

3)).

Let m ∈ N and let A : Z m R p be ε-differentially private.

Then DISPLAYFORM7 DISPLAYFORM8

The PAC-Bayes theorem allows one to choose the prior based on the data-generating distribution D, but not on the data S ∼ D m .

Using differential privacy, we can consider a data-dependent prior P(S).Theorem 5.4.

Under 0-1 loss, for every δ > 0, m ∈ N, distribution D on R k ×K, and ε-differentially private data-dependent prior P : DISPLAYFORM0 It follows from the PAC-Bayes theorem (Theorem 3.1) that P S∼D m (S ∈ R(P)) ≤ β .

Theorem 5.1 implies that the bound holds with P replaced by P(S), provided that we inflate the probability of failure.

In particular, let δ = 3 β .

Then ln(1/β ) = 2 ln(3/δ ).

By Theorem 5.1, provided 2mε 2 ≤ ln(1/β ), then P S∼D m (S ∈ R(P(S)))

≤ δ .

It follows that, with probability no more than δ over S ∼ D m , there exists a distribution Q on R p such that DISPLAYFORM1 The bound stated in Eq. (19) follows immediately.

Note that the bound holds for any posterior Q, including one obtained by optimizing a different PAC-Bayes bound.

We have chosen to present a differentially private version of Theorem 3.1 rather than Theorem 3.2, because the former tends to be tighter numerically.

Giving a differentially private version of Theorem 3.2, or any other PAC-Bayes bound, should be straightforward: one merely needs to decide how to incorporate the constraint between ε, β , and m in Theorem 5.1.

We have chosen to deal with the constraint via a max operation affecting the width of the confidence interval.

Note that, in realistic scenarios, δ is large enough relative to ε that an ε-differentially private prior P(S) contributes 2ε 2 to the generalization error.

Therefore, ε must be much less than one to not contribute a nontrivial amount to the generalization error.

In order to match the m −1 rate by which the KL term decays, one must have ε ∈ O(m −1/2 ).

Our empirical studies use this rate.

We have already explained that the weights learned by Entropy-SGD can be viewed as the mean of a data-dependent prior P(S).

By Theorem 5.4 and the fact that post-processing does not decrease privacy, it would suffice to establish that the mean is ε-differentially private in order to obtain a risk bound on the corresponding Gibbs posterior classifier.

Entropy-SGD can be viewed as stochastic gradient ascent on the negative local entropy, but with biased gradient estimates.

The bias comes from the use of SGLD to compute the expectation in Eq. (8).

Putting aside this issue, existing privacy analyses of SGD worsen after every iteration.

For the number of iterations necessary to obtain reasonable weights, known upper bounds on the differential privacy of SGD yield vacuous generalization bounds.

The standard (if idealized) approach for optimizing a data-dependent objective in a private way is to use the exponential mechanism BID33 .

In the context of maximizing the local entropy, the exponential mechanism correspond to sampling exactly from the "local entropy (Gibbs) distribution" DISPLAYFORM0 where β > 0 and P is some measure on R p . (It is natural to take P to be Lebesgue measure, or a multivariate normal distribution, which would correspond to L2 regularization of the local entropy.)The following result establishes the privacy of a sample from the local entropy distribution: Theorem 5.5.

Let γ, τ > 0, and assume the range of the loss is contained in an interval of length L max .

One sample from the local entropy distribution P exp(β F γ,τ (·;S)) , is 2β L max τ m -differentially private.

Proof.

The result follows immediately from the following two lemmas.

Lemma 5.6 ((McSherry and Talwar, 2007, Thm. 6)).

Let q : Z m × R p → R be measurable, let P be a measure on R p , let β > 0, and assume P[exp(−β q(S, ·))]

< ∞ for all S ∈ Z m .

Let ∆q def = sup S,S sup w∈R p |q(S, w) − q(S , w)|, where the first supremum ranges over pairs S, S ∈ Z m that disagree on no more than one coordinate.

Let A : Z m R p , on input S ∈ Z m , output a sample from the Gibbs distribution P exp(−β q(S,·)) .

Then A is 2β ∆q-differentially private.

Lemma 5.7.

Let F γ,τ (w; S) be defined as Eq. (6), assume the range of the loss is contained in an interval of length L max , and define q(S, w) = −F γ,τ (w; S).

Then ∆q DISPLAYFORM1 Proof.

The proof essentially mirrors that of (McSherry and Talwar, 2007, Thm. 6 ).There are two obvious obstructions to using the exponential mechanism to pick a prior mean: first, cross-entropy loss can change in an unbounded way when swapping a single data point; second, sampling from the local entropy distribution exactly is hard in general.

To sidestep the first obstruction, we modify the underlying cross-entropy loss to be bounded by rescaling the probabilities output by the classifier to be bounded away from zero and one, allowing us to invoke Lemma 5.7. (Details of our modification of the cross entropy are described in Appendix B.2.1.)There is no simple way to sidestep the second obstruction.

Instead, we once again use SGLD to generate an approximate sample from the local entropy distribution.

In summary, to optimize the local entropy F γ,τ (·; S) in a private way to obtain the prior mean w, we repeatedly perform the SGLD update DISPLAYFORM2 where at each roundĝ(w) is an estimate of the gradient ∇ w F γ,τ (w; S). (Recall the identity Eq. (8).)

As in Entropy-SGD, we construct biased gradient estimates via an inner loop of SGLD.

In summary, the only change to Entropy-SGD is the addition of noise in the outer loop.

We call the resulting algorithm Entropy-SGLD. (See Algorithm 1.

Note that we take β = 1 in our experiments.)There have been a number of privacy analyses of SGLD BID35 BID6 BID12 BID42 BID34 .

Most of these analyses deliver (ε, δ )-differential privacy, but none of them take advantage of the fact that SGLD mixes in the limit as it converges weakly to the Gibbs distributions, under certain technical conditions (Teh, Thiery, and Vollmer, 2016, Thm. 7) .

In our analysis and bound calculations, we therefore make the approximation that SGLD has the same privacy as its limiting invariant measure, the exponential mechanism.

Building a less conservative model of the privacy of SGLD is an open problem.

However, by making this approximation, we may see the potential/limits of combining differentially private optimization and PAC-Bayesian bounds.

We return to the issues again in light of our empirical findings (Section 6) and in our discussion (Section 7).

The generalization bounds that we have devised are data-dependent and so the question of their utility is an empirical one that requires data.

In this section, we perform an empirical study of SGD, SGLD, Entropy-SGD, and Entropy-SGLD on the MNIST data set, on both convolutional and fully connected architectures, and compare our generalization bounds to estimates based on held-out data.

Under our privacy approximation, SGLD and Entropy-SGLD are ε-differentially private and we take advantage of this fact to apply differentially private versions of two tail bounds and our PACBayes bound.

The degree ε of privacy is determined by the τ parameter of the local entropy (C.f. thermal noise 2/τ), and then, in turn, ε contributes to our bounds on the generalization error.

As theory predicts, τ affects the degree of overfitting empirically, and no bound we compute is violated too frequently.

Of course, the validity of our generalization bounds rests on the degree to which our privacy approximation is violated.

3 We reflect on our approximation in light of our empirical results, and then return to this point in the discussion.

The weights learned by SGD, SGLD, and Entropy-SGD are treated differently from those learned by Entropy-SGLD.

In the former case, the weights parametrize a neural network as usual, and the training and test error are computed using these weights.

In the latter case, the weights are taken to be the mean of a multivariate normal prior, and we evaluate the training and test error of the The gap is an estimate of the generalization error.

On true labels, SGLD finds classifiers with relatively small generalization error.

At low thermal noise settings, SGLD (and its zero limit, SGD), achieve small empirical risk.

As we increase the thermal noise, the empirical 0-1 error increases, but the generalization error decreases.

At 0.1 thermal noise, risk is close to 50%. (top-right) On random labels, SGLD has high generalization error for thermal noise values 0.01 and below. (True error is 50%). (middle-left) On true labels, Entropy-SGD, like SGD and SGLD, has small generalization error.

For the same settings of thermal noise, empirical risk is lower. (middle-right) On random labels, Entropy-SGD overfits for thermal noise values 0.005 and below.

Thermal noise 0.01 produces good performance on both true and random labels. (bottom row) Entropy-SGLD is configured to be ε-differentially private with ε ≈ 0.0327 by setting τ = √ m, where m is the number of training samples. (bottom-left) On true labels, the generalization error for networks learned by Entropy-SGLD is close to zero.

Generalization bounds are relatively tight. (bottom-right) On random label, Entropy-SGLD does not overfit.

See Fig. 3 for SGLD bounds at same privacy setting.

associated Gibbs posterior (i.e., a randomized classifier).

We also report the performance of the (deterministic) network parametrized by these weights (called the "mean" classifier) in order to give a coarse statistic summarizing the local empirical risk surface.

Following BID44 , we study these algorithms on MNIST with its original ("true") labels, as well as on random labels.

Parameter τ that performs very well in one setting often does not perform well in the other.

Random labels mimic data where the Bayes error rate is high, and where overfitting can have severe consequences.

We use a two-class variant of MNIST (LeCun, Cortes, and Burges, 2010).

4 (See FIG7 and Appendix C for our experiments on the standard multiclass MNIST dataset.

They yield similar insight.)

Some experiments involve random labels, i.e., labels drawn independently and uniformly at random at the start of training.

We study three network architectures, abbreviated FC600, FC1200, and CONV.

Both FC600 and FC1200 are 3-layer fully connected networks, with 600 and 1200 units per hidden layer, respectively.

CONV is a convolutional architecture.

All three network architectures are taken from the MNIST experiments by BID10 , but adapted to our two-class version of MNIST.

5 Let S and S tst denote the training and test sets, respectively.

For all learning algorithms we track (i)R S (w) andR S tst (w), i.e., the training and test error for w.

We also track DISPLAYFORM0 .e.

, the mean training and test error of the local Gibbs distribution, viewed as a randomized classifier ("Gibbs") and, using the differential privacy bounds in Theorem 5.5, compute (iii) a PAC-Bayes bound on R D (G w,S γ,τ ) using Theorem 5.4 ("PAC-bound"); (iv) the mean of a Hoeffding-style bound on R D (w ), where w ∼ P exp(F γ,τ (·;S)) ,, using Theorem 5.2 ("H-bound");(v) an upper bound on the mean of a Chernoff-style bound on R D (w ), where w ∼ P exp(F γ,τ (·;S)) ,, using Theorem 5.3 ("C-bound").We also compute H-and C-bounds for SGLD, viewed as a sampler for w ∼ P exp(−τR S ) , where P here is Lebesgue measure.

In order for SGLD and Entropy-SGLD to be private, we modify the cross entropy loss function to be bounded.

We achieve this by an affine transformation of the neural networks output that prevents extreme probability (se Appendix B.2.1).

With the choice of τ = √ m, and the loss function taking values in an interval of length L max = 4, Entropy-SGLD is ε-differentially private, with ε ≈ 0.0327.

See Appendix B.2 for additional details.

Note that, in the calculation of (iii), we do not account for Monte Carlo error in our estimate ofR S (w).

The effect is small, given the large number of iterations of SGLD performed for each point in the plot.

Recall that DISPLAYFORM1 and so we may interpret the bounds in terms of the performance of a randomized classifier or the mean performance of a randomly chosen classifier.

Key results for the convolutional architecture (CONV) appear in FIG2 .

Results for FC600 and FC1200 appear in Fig. 2 of Appendix B. (Training the CONV network produces the lowest training/test errors and tightest generalization bounds.

Results and bounds for FC600 are nearly identical to those for FC1200, despite FC1200 having three times as many parameters.)The top row of FIG2 presents the performance of SGLD for various levels of thermal noise 2/τ under both true and random labels.

(Under our privacy approximation, we may also use SGLD to directly perform a private optimization of the empirical risk surface.

The level of thermal noise determines the differential privacy of SGLD and so we expect to see a tradeoff between empirical risk and generalization error.

Note that SGD is the same as SGLD with zero thermal noise.)

SGD achieves the smallest training and test error on true labels, but overfits the worst on random labels.

In comparison, SGLD's generalization performance improves with higher thermal noise, while its risk performance worsens.

At 0.05 thermal noise, SGLD achieves reasonable but relatively large risk but almost zero generalization error on both true and random labels.

Other thermal noise settings have either much worse risk or generalization performance.

The middle row of FIG2 presents the performance of Entropy-SGD for various levels of thermal noise 2/τ under both true and random labels.

As with SGD, Entropy-SGD's generalization performance improves with higher thermal noise, while its risk performance worsens.

At the same levels of thermal noise, Entropy-SGD outperforms the risk and generalization error of SGD.

At 0.01 thermal noise, Entropy-SGD achieves good risk and low generalization error on both true and random labels.

However, the test-set performance of Entropy-SGD at 0.01 thermal noise is still worse than that of SGD.

Whether this difference is due to SGD overfitting to the MNIST test set is unclear and deserves further study.

The bottom row of FIG2 presents the performance of Entropy-SGLD with τ = √ m on true and random labels.

(This corresponds to approximately 0.09 thermal noise.)

On true lables, both the mean and Gibbs classifier learned by Entropy-SGLD have approximately 2% test error and essentially zero generalization error, which is less than predicted by our bounds.

Our PAC-Bayes risk bounds are roughly 3%.

As expected by the theory, Entropy-SGLD does not overfit on random labels, even after thousands of epochs.

We find that our PAC-Bayes bounds are generally tighter than the H-and C-bounds.

All bounds are nonvacuous, though still loose.

The error bounds reported here are tighter than those reported by BID18 .

However, the validity of all three privacy-based bounds that we report rests on the privacy approximation regarding SGLD, and so interpreting these bounds requires some subtlety.

We achieve much tighter generalization bounds than previously reported, and better test error, but we are still far from the performance of SGD.

This is despite making a strong approximation, and so we might view these results as telling us the limits of combining differential privacy and PAC-Bayes bounds.

Weaker notions of stability/privacy may be necessary to achieve further improvement in generalization error and test error.

Despite the coarse privacy approximation, no bound is ever violated: possible explanations include the bounds simply being loose and/or the data being far from worst case.

Note that, given the number of experiments, we might even expect a violation for tight bounds.

Indeed, our performance on random labels supports the hypothesis that the privacy of (Entropy-)SGLD does not degrade over time, at least not in a way that can be detected by our experiments.

Our work reveals that Entropy-SGD can be understood as optimizing a PAC-Bayes generalization bound in terms of the bound's prior.

Because the prior must be independent of the data, the bound is invalid, and, indeed, we observe overfitting in our experiments with Entropy-SGD when the thermal noise 2/τ is set to 0.0001 as suggested by Chaudhari et al. for MNIST.

PAC-Bayes priors can, however, depend on the data distribution.

This flexibility seems wasted, since the data sample is typically viewed as one's only view onto the data distribution.

However, using differential privacy, we can span this gap.

By performing a private computation on the data, we can extract information about the underlying distribution, without undermining the statistical validity of a subsequent PAC-Bayes bound.

Our PAC-Bayes bound based on a differentially private prior is made looser by the use of a private data-dependent prior, but the gains in choosing a datadistribution-dependent prior more than make up for the expansion of the bound due to the privacy.

(The gains come from the KL term being much smaller on the account of the prior being better matched to the posterior.)

Understanding how our approach compares to local PAC-Bayes priors BID9 is an important open problem.

The most elegant way to make Entropy-SGD private is to replace SGD with a sample from the Gibbs distribution (known as the exponential mechanism in the differential privacy literature).

However, generating an exact sample is intractable, and so practicioners use SGLD to generate an approximate sample, relying on the fact that SGLD converges weakly to the exponential mechanism under certain technical conditions.

Our privacy approximation allows us to proceed with a theoretical analysis by assuming that SGLD achieves the same privacy as the exponential mechanism.

On the one hand, we do not find overt evidence that our approximation is grossly violated.

On the other, we likely do not require such strong privacy in order to control generalization error.

We might view our privacy-based bounds as being optimistic and representing the bounds we might be able to achieve rigorously should there be a major advance in private optimization.

(No analysis of the privacy of SGLD takes advantage of the fact that it mixes weakly.)

On the account of using private data-dependent priors, our bounds are significantly tighter than those reported by BID18 .

However, despite our bounds potentially being optimistic, the test set error we are able to achieve is still 5-10 times that of SGD.

Differential privacy may be too conservative for our purposes, leading us to underfit.

Indeed, we think it is unlikely that Entropy-SGD has strong differential privacy, yet we are able to achieve good generalization on both true and random labels under 0.01 thermal noise.

Identifying the appropriate notion of privacy/stability to combine with PAC-Bayes bounds is an important problem.

Despite our progress on building learning algorithms with strong generalization performance, and identifying a path to much tighter PAC-Bayes bounds, Entropy-SGLD learns much more slowly than Entropy-SGD, the risk of Entropy-SGLD is far from state of the art, and our PAC-Bayes bounds are loose.

It seems likely that there is a fundamental tradeoff between the speed of learning, the excess risk, and the ability to produce a certificate of one's generalization error via a rigorous bound.

Characterizing the relationship between these quantities is an important open problem.

A BACKGROUND: DIFFERENTIAL PRIVACY Here we formally define some of the differential privacy related terms used in the main text.

(See BID13 BID15 for more details.) Let U,U 1 ,U 2 , . . .

be independent uniform (0, 1) random variables, independent also of any random variables introduced by P and E, and let π : DISPLAYFORM0 Definition A.1.

A randomized algorithm A from R to T , denoted A : R T , is a measurable map A : [0, 1] × R → T .

Associated to A is a (measurable) collection of random variables {A r : r ∈ R} that satisfy A r = A (U, r).

When there is no risk of confusion, we will write A (r) for A r .

Definition A.2.

A randomized algorithm A : Z m T is (ε, δ )-differentially private if, for all pairs S, S ∈ Z m that differ at only one coordinate, and all measurable subsets B ⊆ T , we have P(A (S) ∈ B) ≤ e ε P(A (S ) ∈ B) + δ .We will write ε-differentially private to mean (ε, 0)-differentially private algorithm.

Definition A.3.

Let A : R T and A : DISPLAYFORM1 Lemma A.4 (post-processing).

Let A : Z m T be (ε, δ )-differentially private and let F : T T be arbitrary.

Then F • A is (ε, δ )-differentially private.

We studied three architectures: CONV, FC600, and FC1200.CONV was a convolutional neural network, whose architecture was the same as that used by BID10 for multiclass MNIST classification, except modified to produce a single probability output for our two-class variant of MNIST.

In particular, CONV has two convolutional layers, a fully connected ReLU layer, and a sigmoidal output layer, yielding 126, 711 parameters in total.

FC600 and FC1200 are fully connected 3-layer neural networks, with 600 and 1200 hidden units, respectively, yielding 834, 601 and 2, 385, 185 parameters in total, respectively.

We used ReLU activations for all but the last layer, which was sigmoidal to produce an output in [0, 1].In their MNIST experiments, BID10 use dropout and batch normalization.

We did not use dropout.

The bounds we achieved with and without batch norm were very similar.

Without batch norm, however, it was necessary to tune the learning rates.

Understanding the combination of SGLD and batch norm and the limiting invariant distribution, if any, is an important open problem.

B.2.1 OBJECTIVE All networks are trained to minimize a bounded variant of empirical cross entropy loss.

The change involves replacing g(p, y) = − log p with g(p, y) = − log ψ(p), where DISPLAYFORM0 is an affine transformation that maps to DISPLAYFORM1 , removing extreme probability values.

As a result, the binary cross entropy loss BCE is contained in an interval of length L max .

In particular, DISPLAYFORM2 We take L max = 4 in our experiments.

Ordinarily, an epoch implies one pass through the entire data set.

For SGD, each stochastic gradient step processes a minibatch of size K = 128.

Therefore, an epoch is m/K = 468 steps of SGD.

An epoch for Entropy-SGD and Entropy-SGLD is defined as follows: each iteration of the inner SGLD Test ( On true labels, SGLD learns a network with approximately 3% higher training and test error than the mean and Gibbs networks learned by Entropy-SGLD.

SGLD does not overfit on random labels, as predicted by theory.

The C-bound on the true error of this network is around 8%, which is worse than the roughly 4% C-bound on the mean classifier.loop processes a minibatch of size K = 128, and the inner loop runs for L = 20 steps.

Therefore, an epoch is m/(LK) steps of the outer loop.

In concrete terms, there are 20 steps of SGD per every one step of Entropy-SG(L)D. Concretely, the x-axis of our plots measure epochs divided by L. This choice, used also by BID10 , ensures that the wall-clock time of Entropy-SG(L)D and SGD align.

The step sizes for SGLD must be square summable but not summable.

The step sizes for the outer SGLD loop are of the form η t = ηt −0.6 , with η = 0.006 γτ .

The step sizes for the inner SGLD loop are of the form η t = ηt −1 , with η = 1 γτ .

The estimate produced by the inner SGLD loop is computed using a weighted average (line 8) with α = 0.75.

We use SGLD again when computing the PAC-Bayes generalization bound (Appendix B.3.2).

In this case, SGLD is used to sample from the local Gibbs distribution when estimating the Gibbs risk and the KL term.

We run SGLD for 1000 epochs to obtain our estimate.

Again, we use weighted averages, but with α = 0.005, in order to average over a larger number of samples and better control the variance.

We set γ = 1 and τ = √ m and keep the values fixed during optimization.

By Theorem 5.5, the value of τ, L max , and β determine the differential privacy of Entropy-SGLD.

In turn, the differential privacy parameter ε and confidence parameter δ contribute When the empirical error is close to zero, the KL version of the PAC-Bayes bound Theorem 3.1 is considerably tighter than the Hoeffding-style bound first described by BID31 .

However, using this relative entropy bound requires one to be able to compute the largest value p such that KL(q||p) ≤ c. There does not appear to be a simple formula for this value.

In practice, however, the value can be efficiently numerically approximated using, e.g., Newton's method.

See (Dziugaite and Roy, 2017, §2.2 and App.

B).

Let (w) = τR S (w).

By (Catoni, 2007, Lem Both terms have obvious Monte Carlo estimates: DISPLAYFORM0 where w 1 , . . . , w k are taken from a Markov chain targeting P exp(− ) , such as SGLD run for k 1 steps (which is how we computed our bounds), and log P[exp(− )] = log exp{− (w)} P(dw) DISPLAYFORM1 where h 1 , . . .

, h k are i.i.d.

P (which is a multivariate Gaussian in this case).

In the latter case, due to the concavity of log, the estimate is a lower bound with high probability, yielding a high probability upper bound on the KL term.

We evaluate the same generalization bounds on the standard MNIST classification task as in the MNIST binary labelling case.

All the details of the network architectures and parameters are as stated in Appendix B.2, with two exception: following BID10 , we use a fully connected network with 1024 hidden units per layer, denoted FC1024.

The neural network produces a probability vector (p 1 , . . .

, p K ) via a soft-max operation.

Ordinarily, we then apply the cross entropy loss corresponding to g FIG2 . .

, p K ), y) = − log p y .

When training privately, we use a bounded variant of the cross entropy loss, where the function g above is replaced by g((p 1 , . . .

, p K ), y) = − log ψ(p y ), and ψ is defined as in Eq. (25).

FC1024 network trained on true labels.

The train and test error suggest that the generalization gap is close to zero, while all three bounds exceed the test error by slightly more than 3%. (bottomleft) CONV network trained on true labels.

Both the train and the test errors are lower than those achieved by the FC1024 network.

We still do not observe overfitting.

The C-bound and PAC-Bayes bounds exceed the test error by ≈ 3%. (top-right) FC1024 network trained on random labels.

After approximately 1000 epochs, we notice overfitting by ≈ 2%.

Running Entropy-SGLD further does not cause an additional overfitting.

Theory suggests that our choice of τ prevents overfitting via differential privacy. (bottom-right) CONV network trained on random labels.

We observe almost no overfitting (less than 1%).

Both training and test error coincide and remain close to the guessing rate (90%).

<|TLDR|>

@highlight

We show that Entropy-SGD optimizes the prior of a PAC-Bayes bound, violating the requirement that the prior be independent of data; we use differential privacy to resolve this and improve generalization.
Motivated by applications to unsupervised learning, we consider the problem of measuring mutual information.

Recent analysis has shown that naive kNN estimators of mutual information have serious statistical limitations motivating more refined methods.

In this paper we prove that serious statistical limitations are inherent to any measurement method.

More specifically, we show that any distribution-free high-confidence lower bound on mutual information cannot be larger than $O(\ln N)$ where $N$ is the size of the data sample.

We also analyze the Donsker-Varadhan lower bound on KL divergence in particular and show that, when simple statistical considerations are taken into account, this bound can never produce a high-confidence value larger than $\ln N$. While large high-confidence lower bounds are impossible, in practice one can use estimators without formal guarantees.

We suggest expressing mutual information as a difference of entropies and using cross entropy as an entropy estimator.

We observe that, although cross entropy is only an upper bound on entropy, cross-entropy estimates converge to the true cross entropy at the rate of $1/\sqrt{N}$.

Motivated by maximal mutual information (MMI) predictive coding BID11 BID16 BID13 , we consider the problem of measuring mutual information.

A classical approach to this problem is based on estimating entropies by computing the average log of the distance to the kth nearest neighbor in a sample BID7 .

It has recently been shown that the classical kNN methods have serious statistical limitations and more refined kNN methods have been proposed BID5 .

Here we establish serious statistical limitations on any method of estimating mutual information.

More specifically, we show that any distribution-free high-confidence lower bound on mutual information cannot be larger than O(ln N ) where N is the size of the data sample.

Prior to proving the general case, we consider the particular case of the Donsker-Varadhan lower bound on KL divergence BID3 BID1 .

We observe that when simple statistical considerations are taken into account, this bound can never produce a highconfidence value larger than ln N .

Similar comments apply to lower bounds based on contrastive estimation.

The contrastive estimation lower bound given in BID13 does not establish mutual information of more than ln k where k is number of negative samples used in the contrastive choice.

The difficulties arise in cases where the mutual information I(x, y) is large.

Since I(x, y) = H(y) − H(y|x) we are interested in cases where H(y) is large and H(y|x) is small.

For example consider the mutual information between an English sentence and its French translation.

Sampling English and French independently will (almost) never yield two sentences where one is a plausible translation of the other.

In this case the DV bound is meaningless and contrastive estimation is trivial.

In this example we need a language model for estimating H(y) and a translation model for estimating H(y|x).

Language models and translation models are both typically trained with crossentropy loss.

Cross-entropy loss can be used as an (upper bound) estimate of entropy and we get an estimate of mutual information as a difference of cross-entropy estimates.

Note that the upper-bound guarantee for the cross-entropy estimator yields neither an upper bound nor a lower bound guarantee for a difference of entropies.

Similar observations apply to measuring the mutual information for pairs of nearby frames of video or pairs of sound waves for utterances of the same sentence.

We are motivated by the problem of maximum mutual information predictive coding BID11 BID16 BID13 .

One can formally define a version of MMI predictive coding by considering a population distribution on pairs (x, y) where we think of x as past raw sensory signals (images or sound waves) and y as a future sensory signal.

We consider the problem of learning stochastic coding functions C x and C y so as to maximize the mutual information I(C x (x), C y (y)) while limiting the entropies H(C x (x)) and H(C y (y)).

The intuition is that we want to learn representations C x (x) and C y (y) that preserve "signal" while removing "noise".

Here signal is simply defined to be a low entropy representation that preserves mutual information with the future.

Forms of MMI predictive coding have been independently introduced in BID11 under the name "information-theoretic cotraining" and in BID13 under the name "contrastive predictive coding".

It is also possible to interpret the local version of DIM (DIM(L)) as a variant of MMI predictive coding.

A closely related framework is the information bottleneck BID17 .

Here one again assumes a population distribution on pairs (x, y).

The objective is to learn a stochastic coding function C x so as to maximize I(C x (x), y) while minimizing I(C x (x), x).

Here one does not ask for a coding function on y and one does not limit H(C x (x)).Another related framework is INFOMAX BID8 BID2 .

Here we consider a population distribution on a single random variable x.

The objective is to learn a stochastic coding function C x so as to maximize the mutual information I(x, C x (x)) subject to some constraint or additional objective.

As mentioned above, in cases where I(C x (x), C y (y)) is large it seems best to train a model of the marginal distribution of P (C y ) and a model of the conditional distribution P (C y |C x ) where both models are trained with cross-entropy loss.

Section 5 gives various high confidence upper bounds on cross-entropy loss for learned models.

The main point is that, unlike lower bounds on entropy, high-confidence upper bounds on cross-entropy loss can be guaranteed to be close to the true cross entropy.

Out theoretical analyses will assume discrete distributions.

However, there is no loss of generality in this assumption.

Rigorous treatments of probability (measure theory) treat integrals (either Riemann or Lebesgue) as limits of increasingly fine binnings.

A continuous density can always be viewed as a limit of discrete distributions.

Although our proofs are given for discrete case, all our formal limitations on the measurement of mutual information apply to continuous case as well.

See BID9 for a discussion of continuous information theory.

Additional comments on this point are given in section 4.

Mutual information can be written as a KL divergence.

DISPLAYFORM0 Here P X,Y is a joint distribution on the random variables X and Y and P X and P Y are the marginal distributions on X and Y respectively.

The DV lower bound applies to KL-divergence generally.

To derive the DV bound we start with the following observation for any distributions P , Q, and G on the same support.

Our theoretical analyses will assume discrete distributions.

DISPLAYFORM1 Note that (1) achieves equality for G(z) = P (z) and hence we have DISPLAYFORM2 Here we can let G be a parameterized model such that G(z) can be computed directly.

However, we are interested in KL(P X,Y , P X P Y ) where our only access to the distribution P is through sampling.

If we draw a pair (x, y) and ignore y we get a sample from P X .

We can similarly sample from P Y .

So we are interested in a KL-divergence KL(P, Q) where our only access to the distributions P and Q is through sampling.

Note that we cannot evaluate (1) by sampling from P because we have no way of computing Q(z).

But through a change of variables we can convert this to an expression restricted to sampling from Q. More specifically we define G(z) in terms of an unconstrained function F (z) as DISPLAYFORM3 Substituting FORMULA3 into FORMULA2 gives DISPLAYFORM4 Equation FORMULA4 is the Donsker-Varadhan lower bound.

Applying this to mutual information we get DISPLAYFORM5 This is the equation underlying the MINE approach to maximizing mutual information BID1 .

It would seem that we can estimate both terms in (5) through sampling and be able to maximize I(X, Y ) by stochastic gradient ascent on this lower bound.

In this section we show that the DV bound (4) cannot be used to measure KL-divergences of more than tens of bits.

In fact we will show that no high-confidence distribution-free lower bound on KL divergence can be used for this purpose.

As a first observation note that (4) involves E z∼Q e F (z) .

This expression has the same form as the moment generating function used in analyzing large deviation probabilities.

The utility of expectations of exponentials in large deviation theory is that such expressions can be dominated by extremely rare events (large deviations).

The rare events dominating the expectation will never be observed by sampling from Q. It should be noted that the optimal value for F (z) in (4) is ln(P (z)/Q(z)) in which case the right hand side of (4) simplifies to KL(P, Q).

But for large KL divergence we will have that F (z) = ln(P (z)/Q(z)) is typically hundreds of bits and this is exactly the case where E z∼Q e F (z) cannot be measured by sampling from Q. If E z∼Q e F (z) is dominated by events that will never occur in sampling from Q then the optimization of F through the use of (4) and sampling from Q cannot possibly lead to a function F (z) that accurately models the desired function ln(P (z)/Q(z)).To quantitatively analyze the risk of unseen outlier events we will make use of the following simple lemma where we write P z∼Q (Φ[z]) for the probability over drawing z from Q that the statement DISPLAYFORM0 Outlier Risk Lemma: For a sample S ∼ Q N with N ≥ 2, and a property Φ[z] such that P z∼Q (Φ[z]) ≤ 1/N , the probability over the draw of S that no z ∈ S satisfies Φ[z] is at least 1/4.

Proof: The probability that Φ[z] is unseen in the sample is at least (1 − 1/N ) N which is at least 1/4 for N ≥ 2 and where we have DISPLAYFORM1 We can use the outlier risk lemma to perform a quantitative risk analysis of the DV bound (4).

We can rewrite (4) as DISPLAYFORM2 We can try to estimate B(P, Q, G) from samples S P and S Q , each of size N , from the population distributions P and Q respectively.

DISPLAYFORM3 While B(P, Q, F ) is a lower bound on KL(P, Q), the sample estimateB(S P , S Q , F ) is not.

To get a high confidence lower bound on KL(P, Q) we have to handle unseen outlier risk.

For a fair comparison with our analysis of cross-entropy estimators in section 5, we will limit the outlier risk by bounding F (z) to the interval [0, F max ].

The largest possible value ofB(S P , S q , F ) occurs when F (z) = F max for all z ∈ S P and F (z) = 0 for all z ∈ S Q .

In this case we getB(S P , S Q , F ) = F max .

But by the outlier risk lemma there is still at least a 1/4 probability that DISPLAYFORM4 Any high confidence lower boundB(S P , S Q , F ) must account for the unseen outlier risk.

In particular we must haveB DISPLAYFORM5 Our negative results can be strengthened by considering the preliminary bound (1) where G(z) is viewed as a model of P (z).

We can consider the extreme case of perfect modeling of the population P with a model G(z) where G(z) is computable.

In this case we have essentially complete access to the distribution P .

But even in this setting we have the following negative result.

Theorem 1 Let B be any distribution-free high-confidence lower bound on KL(P,Q) computed with complete knowledge of P but only a sample from Q.More specifically, let B(P, S, δ) be any real-valued function of a distribution P , a multiset S, and a confidence parameter δ such that, for any P , Q and δ, with probability at least (1 − δ) over a draw of S from Q N we have KL(P, Q) ≥ B(P, S, δ).

For any such bound, and for N ≥ 2, with probability at least 1 − 4δ over the draw of S from Q N we have B(P, S, δ) ≤ ln N.Proof.

Consider distributions P and Q and N ≥ 2.

DefineQ bỹ DISPLAYFORM6 We now have KL(P,Q) ≤ ln N .

We will prove that from a sample S ∼ Q N we cannot reliably distinguish between Q andQ.We first note that by applying the high-confidence guarantee of the bound toQ have DISPLAYFORM7 The distributionQ equals the marginal on z of a distribution on pairs (s, z) where s is the value of Bernoulli variable with bias 1/N such that if s = 1 then z is drawn from P and otherwise z is drawn from Q. By the outlier risk lemma the probability that all coins are zero is at least 1/4.

Conditioned on all coins being zero the distributionsQ N and Q N are the same.

Let Pure(S) represent the event that all coins are 0 and let Small(S) represent the event that B(P, S, δ) ≤ ln N .

We now have DISPLAYFORM8

Mutual information is a special case of KL-divergence.

It is possible that tighter lower bounds can be given in this special case.

In this section we show similar limitations on lower bounding mutual information.

We first note that a lower bound on mutual information implies a lower bound on entropy.

The mutual information between X and Y cannot be larger than information content of X alone.

So a lower bound on I(X, Y ) gives a lower bound on H(X).

We show that any distribution-free high-confidence lower bound on entropy requires a sample size exponential in the size of the bound.

The above argument seems problematic for the case of continuous densities as differential entropy can be negative.

However, for the continuous case we have DISPLAYFORM0 where C x and C y range over all maps from the underlying continuous space to discrete sets (all binnings of the continuous space).

Hence an O(ln N ) upper bound on the measurement of mutual information for the discrete case applies to the continuous case as well.

The type of a sample S, denoted T (S), is defined to be a function on positive integers (counts) where T (S)(i) is the number of elements of S that occur i times in S. For a sample of N draws we have N = i iT (S)(i).

The type T (S) contains all information relevant to estimating the actual probability of the items of a given count and of estimating the entropy of the underlying distribution.

The problem of estimating distributions and entropies from sample types has been investigated by various authors BID12 BID15 BID14 BID0 .

Here we give the following negative result on lower bounding the entropy of a distribution by sampling.

Theorem 2 Let B be any distribution-free high-confidence lower bound on H(P ) computed from a sample type T (S) with S ∼ P N .More specifically, let B(T , δ) be any real-valued function of a type T and a confidence parameter δ such that for any P , with probability at least (1 − δ) over a draw of S from P N , we have

For any such bound, and for N ≥ 50 and k ≥ 2, with probability at least 1 − δ − 1.01/k over the draw of S from P N we have B(T (S), δ) ≤ ln 2kN 2 .Proof: Consider a distribution P and N ≥ 100.

If the support of P has fewer than 2kN 2 elements then H(P ) < ln 2kN 2 and by the premise of the theorem we have that, with probability at least 1 − δ over the draw of S, B(T (S), δ) ≤ H(P ) and the theorem follows.

If the support of P has at least 2kN 2 elements then we sort the support of P into a (possibly infinite) sequence x 1 , x 2 , x 3 , . . .

so that P (x i ) ≥ P (x i+1 ).

We then define a distributionP on the elements x 1 , . . .

, x 2kN 2 bỹ DISPLAYFORM0 We will let Small(S) denote the event that B(T (S), δ) ≤ ln 2kN 2 and let Pure(S) abbreviate the event that no element x i for i > kN 2 occurs twice in the sample.

SinceP has a support of size 2kN 2 we have H(P ) ≤ ln 2kN 2 .

Applying the premise of the lemma toP gives DISPLAYFORM1 For a type T let P S∼P N (T ) denote the probability over drawing S ∼ P N that T (S) = T .

We now have DISPLAYFORM2 This gives the following.

DISPLAYFORM3 For i > kN 2 we haveP (x i ) ≤ 1/(kN 2 ) which gives DISPLAYFORM4 Using (1 − P ) ≥ e −1.01 P for P ≤ 1/100 we have the following birthday paradox calculation.

DISPLAYFORM5 Applying the union bound to FORMULA17 and FORMULA21 gives.

DISPLAYFORM6 By a derivation similar to that of (9) we get DISPLAYFORM7 Combining FORMULA19 , FORMULA1 and FORMULA1 gives DISPLAYFORM8

Since mutual information can be expressed as a difference of entropies, the problem of measuring mutual information can be reduced to the problem of measuring entropies.

In this section we show that, unlike high-confidence distribution-free lower bounds, high-confidence distribution-free upper bounds on entropy can approach the true cross entropy at modest sample sizes even when the true cross entropy is large.

More specifically we consider the cross-entropy upper bound.

DISPLAYFORM0 For G = P we get H(P, G) = H(P ) and hence we have DISPLAYFORM1 In practice P is a population distribution and G is model of P .

For example P might be a population distribution on paragraphs and G might be an autoregressive RNN language model.

In practice G will be given by a network with parameters Φ. In this setting we have the following upper bound entropy estimator.

Ĥ DISPLAYFORM2 The gap betweenĤ(P ) and H(P ) depends on the expressive power of the model class.

The statistical limitations on distribution-free high-confidence lower bounds on entropy do not arise for cross-entropy upper bounds.

For upper bounds we can show that naive sample estimates of the cross-entropy loss produce meaningful (large entropy) results.

We first define the cross-entropy estimator from a sample S.Ĥ DISPLAYFORM3 We can bound the loss of a model G by ensuring a minimum probability e −Fmax where F max is then the maximum possible log loss in the cross-entropy objective.

In language modeling a loss bound exists for any model that ultimately backs off to a uniform distribution on characters.

Given a loss bound of F max we have thatĤ(S, G) is just the standard sample mean estimator of an expectation of a bounded variable.

In this case we have the following standard confidence interval.

Theorem 3 For any population distribution P , and model distribution G with −ln G(x) bounded to the interval [0, F max ], with probability at least 1 − δ over the draw of S ∼ P N we have DISPLAYFORM4

It is also possible to give PAC-Bayesian bounds on H(P, G Φ ) that take into account the fact that G Φ is typically trained so as to minimize the empirical loss on the training data.

The PAC-Bayesian bounds apply to"broad basin" losses and loss estimates such as the following.

DISPLAYFORM0 Under mild smoothness conditions on G Φ (x) as a function of Φ we have DISPLAYFORM1 An L2 PAC-Bayesian generalization bound BID10 ) gives that for any parameterized class of models and any bounded notion of loss, and any λ > 1/2 and σ > 0, with probability at least 1 − δ over the draw of S from P N we have the following simultaneously for all parameter vectors Φ. DISPLAYFORM2 It is instructive to set λ = 5 in which case the bound becomes.

DISPLAYFORM3 While this bound is linear in 1/N , and tighter in practice than square root bounds, note that there is a small residual gap when holding λ fixed at 5 while taking N → ∞. In practice the regularization parameter λ can be tuned on holdout data.

One point worth noting is the form of the dependence of the regularization coefficient on F max , N and the basin parameter σ.

It is also worth noting that the bound can be given in terms of "distance traveled" in parameter space from an initial (random) parameter setting Φ 0 .

DISPLAYFORM4 Evidence is presented in BID4 that the distance traveled bounds are tighter in practice than traditional L2 generalization bounds.

Recall that in MMI predictive coding we assume a population distribution on pairs (x, y) where we think of x as past raw sensory signals (images or sound waves) and y as a future sensory signal.

We then consider the problem of learning stochastic coding functions C x and C y that maximizes the mutual information I(C x (x), C y (y)) while limiting the entropies H(C x (x)) and H(C y (y)).

Here we propose representing the mutual information as a difference of entropies.

I(C x (x), C y (y)) = H(C y (y)) − H(C y (y)|C x (x))When the coding functions are parameterized by a function Ψ, the above quantities become a function of Ψ. We can then formulate the following nested optimization problem.

The above quantities are expectations over the population distribution on pairs (x, y).

In practice we have only a finite sample form the population.

But the preceding section presents theoretical evidence that, unlike lower bound estimators, upper bound cross-entropy estimators can meaningfully estimate large entropies from feasible samples.

DISPLAYFORM0

Maximum mutual information (MMI) predictive coding seems well motivated as a method of unsupervised pretraining of representations that maintain semantic signal while dropping uninformative noise.

However, the maximization of mutual information is a difficult training objective.

We have given theoretical arguments that representing mutual information as a difference of entropies, and estimating those entropies by minimizing cross-entropy loss, is a more statistically justified approach than maximizing a lower bound on mutual information.

Unfortunately cross-entropy upper bounds on entropy fail to provide either upper or lower bounds on mutual information -mutual information is a difference of entropies.

We cannot rule out the possible existence of superintelligent models, models beyond current expressive power, that dramatically reduce cross-entropy loss.

Lower bounds on entropy can be viewed as proofs of the non-existence of superintelligence.

We should not surprised that such proofs are infeasible.

@highlight

We give a theoretical analysis of the measurement and optimization of mutual information.
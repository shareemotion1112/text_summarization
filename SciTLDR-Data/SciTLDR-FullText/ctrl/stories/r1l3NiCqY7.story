We show that if the usual training loss is augmented by a Lipschitz regularization term, then the networks generalize.

We prove generalization by first establishing a stronger convergence result, along with a rate of convergence.

A second result resolves a question posed in Zhang et al. (2016): how can a model distinguish between the case of clean labels, and randomized labels?

Our answer is that Lipschitz regularization using the Lipschitz constant of the clean data makes this distinction.

In this case, the model learns a different function which we hypothesize correctly fails to learn the dirty labels.

While deep neural networks networks (DNNs) give more accurate predictions than other machine learning methods BID30 , they lack some of the performance guarantees of these other methods.

One step towards performance guarantees for DNNs is a proof of generalization with a rate.

In this paper, we present such a result, for Lipschitz regularized DNNs.

In fact, we prove a stronger convergence result from which generalization follows.

We also consider the following problem, inspired by (Zhang et al., 2016) .

Problem 1.1. [Learning from dirty data]

Suppose we are given a labelled data set, which has Lipschitz constant Lip(D) = O(1) (see (3) below).

Consider making copies of 10 percent of the data, adding a vector of norm to the perturbed data points, and changing the label of the perturbed points.

Call the new, dirty, data setD. The dirty data has Lip(D) = O(1/ ).

However, if we compute the histogram of the pairwise Lipschitz constants, the distribution of the values on the right hand side of (3), are mostly below Lip(D) with a small fraction of the values being O(1/ ), since the duplicated images are apart but with different labels.

Thus we can solve (1) with L 0 estimate using the prevalent smaller values, which is an accurate estimate of the clean data Lipschitz constant.

The solution of (1) using such a value is illustrated on the right of Figure 1 .

Compare to the Tychonoff regularized solution on the right of Figure 2 .

We hypothesis that on dirty data the solution of (1) replaces the thin tall spikes with short fat spikes leading to better approximation of the original clean data.

In Figure 1 we illustrate the solution of (1) (with L 0 = 0), using synthetic one dimensional data.

In this case, the labels {−1, 0, 1} are embedded naturally into Y = R, and λ = 0.1.

Notice that the solution matches the labels exactly on a subset of the data.

In the second part of the figure, we show a solution with dirty labels which introduce a large Lipschitz constant, in this case, the solution reduces the Lipschitz constant, thereby correcting the errors.

Learning from dirty labels is studied in §2.4.

We show that the model learns a different function than the dirty label function.

We conjecture, based on synthetic examples, that it learns a better approximation to the clean labels.

We begin by establishing notation.

Consider the classification problem to fix ideas, although our restuls apply to other problems as well.

Definition 1.2.

Let D n = x 1 , . . .

, x n be a sequence of i.i.d.

random variables sampled from the probability distribution ρ.

The data x i are in X = [0, 1] d .

Consider the classification problem with D labels, and represent the labels by vertices of the probability simplex, Y ⊂ R D .

Write y i = u 0 (x i ) for the map from data to labels.

Write u(x; w) for the map from the input to data to the last layer of the network.1 Augment the training loss with Lipschitz regularization DISPLAYFORM0 The first term in (1) is the usual average training loss.

The second term in (1) the Lipschitz regularization term: the excess Lipschitz constant of the map u, compared to the constant L 0 .In order to apply the generalization theorem, we need to take L 0 ≥ Lip(u 0 ), the Lipschitz constant of the data on the whole data manifold.

In practice, Lip(u 0 ) can be estimated by the Lipschitz constant of the empirical data.

The definition of the Lipschitz constants for functions and data, as well as the implementation details are presented in §1.3 below.

Figure 1: Synthetic labelled data and Lipschitz regularized solution u. Left: The solution value matches the labels exactly on a large portion of the data set.

Right: dirtly labels: 10% of the data is incorrect; the regularized solution corrects the errors.

Our analysis will apply to the problem (1) which is convex in u, and does not depend explicitly on the weights, w. Of course, once u is restricted to a fixed neural network architecture, the corresponding minimization problem becomes non-convex in the weights.

Our analysis can avoid the dependence on the weights because we make the assumption that there are enough parameters so that u can exactly fit the training data.

The assumption is justified by Zhang et al. (2016) .

As we send n → ∞ for convergence, we require that the network also grow, in order to continue to satisfy this assumption.

Our results apply to other non-parametric methods in this regime.

Generalization bounds have been obtained previously via VC dimension analysis of neural networks (Bartlett, 1997).

The generalization rates have factors of the form A k for a k-layer neural network with bounds w i ≤ A for all weight vectors w i in the network.

Such bounds are only applicable for low-complexity networks.

Other works have considered connections between generalization and stability BID8 Xu & Mannor, 2012) .

More recently, BID5 proposed the Lipschitz constant of the network as a candidate measure for the Rademacher complexity, which is a measure of generalization (Shalev-Shwartz & Ben-David, 2014, Chapter 26) .

Also, BID12 showed that Lipschitz regularization can be viewed as a special case of distributional robustness.

Unlike other recent contributions such as BID26 , our analysis does not depend on the training method.

In fact, our analysis has more in common with inverse problems in image processing, such as Total Variation denoising and inpainting BID6 BID36 .

For further discussion, see Appendix C.The estimate of Lip(u; X) provided by (4) can be quite different from the the Tychonoff gradient regularization BID14 , DISPLAYFORM0 since (4) corresponds to a maximum of the values of the norms, and the previous equation corresponds to the mean-squared values.

In fact, recent work on semi-supervised learning suggests that higher p-norms of the gradient are needed for generalization when the data manifold is not well approximated by the data BID15 BID10 BID29 BID39 .

In Figure 2 we compare to the problems in Figure 1 using Tychonoff regularization.

The Tychonoff regularization is less effective at correcting errors.

The effect is more pronounced in higher dimensions.

Figure 2: Synthetic labelled data and Tychonoff regularized solution u. Left: The solution value matches the labels exactly on a large portion of the data set.

Right dirty labels: 10% of the data is incorrect; the regularized solution is not as effective at correcting errors.

The effect is more pronounced in higher dimensions.

An upper bound for the Lipschitz constant of the model is given by the norm of the product of the weight matrices (Szegedy et al., 2013, Section 4.3) .

Let w = (w 1 , . . .

, w J ) be the weight matrices for each layer.

Then DISPLAYFORM0 Regularization of the network using methods based on (2) has been implemented recently in BID23 and (Yoshida & Miyato, 2017) .

Because the upper bound in (2) does not take into account the coefficients in weight matrices which are zero due to the activation functions, the gap in the inequality can be off by factors of many orders of magnitude for deep networks BID19 .Implementing (4) can be accomplished using backpropagation in the x variable on each label, which can become costly for D large.

Special architectures could also be used to implement Lipschitz regularization, for example, on a restricted architecture, BID31 renormalized the weight matrices of each layer to be norm 1.Lipschitz regularization may help with adversarial examples BID40 BID22 which poses a problem for model reliability BID21 .

Since the Lipschitz constant L ℓ of the loss, ℓ, controls the norm of a perturbation DISPLAYFORM1 maps with smaller Lipschitz constants may be more robust to adversarial examples.

BID19 implemented Lipschitz regularization of the loss, and achieved better robustness against adversarial examples, compared to adversarial training BID22 alone.

Lipschitz regularization may also improve stability of GANs.

1-Lipschitz networks are also important for Wasserstein-GANs ) BID0 .

In (Wei et al., 2018) the gradient penalty away from norm 1 is implemented, augmented by a penalty around perturbed points, with the goal of improved stability.

Spectral regularization for GANs was implemented in BID33 ).

Definition 1.3 (Lipschitz constants of functions and data).

Choose norms · Y , and · X on X and Y , respectively.

The Lipschitz constant (in these norms) of a function u : DISPLAYFORM0 When X 0 is all of X, we write Lip(u; X) = Lip(u).

The Lipschitz constant of the data is given by DISPLAYFORM1 Finlay & Oberman FORMULA0 implement Lipschitz regularization as follows.

The basis for the implementation of the Lipschitz constant is Rademacher's Theorem BID18 , §3.1), which states that if a function g(x) is Lipschitz continuous then it is differentiable almost everywhere and DISPLAYFORM2 Restricting to a mini-batch, we obtain the following method for estimating the Lipschitz constant.

Let u(x; w) be a Lipschitz continuous function.

Then max DISPLAYFORM3 For vector valued functions, the appropriate matrix norm must be used, see §B.

The variational problem (1) admits Lipschitz continuous minimizers, but in general the minimizers are not unique.

When L 0 = Lip(u 0 ), it is clear that u 0 , is a solution of FORMULA0 : both the loss term and the regularization term are zero when applied to u 0 .

In addition, any L 0 -Lipschitz extension of u 0 | Dn is also a minimizer of (1), so solutions are not unique.

Let u n be any solution of the Lipschitz regularized variational problem (1).

We study the limit of u n as n → ∞. Since the empirical probability measures ρ n converge to the data distribution ρ, the continuum variational problem corresponding to (1) is min DISPLAYFORM0 where in FORMULA8 we have introduced the following notation.

Definition 2.1.

Given the loss function, ℓ, a map u : X → Y , and a probability measure, µ, supported on X, define DISPLAYFORM1 to be the expectation of the loss with respect to the measure.

In particular, the generalization loss of the map u : DISPLAYFORM2 for the average loss on the data set D n , where ρ n := 1 n δ xi is the empirical measure corresponding to D n .

Remark 2.2.

Generalization is defined in (Goodfellow et al., 2016, Section 5.2) as the expected value of the loss function on a new input sampled from the data distribution.

As defined, the full generalization error includes the training data, but it is of measure zero, so removing it does not change the value.

We introduce the following assumption on the loss function.

Assumption 2.3 (Loss function).

The function ℓ : Y × Y → R is a loss function if it satisfies (i) ℓ ≥ 0, (ii) ℓ(y 1 , y 2 ) = 0 if and only if y 1 = y 2 , and (iii) ℓ is strictly convex in y 1 .Example 2.4 (R D with L 2 loss).

Set Y = R D , and let each label be a basis vector.

Set ℓ(y 1 , y 2 ) = y 1 − y 2 2 2 to be the L 2 loss.

Example 2.5 (Classification).

In classification, the output of the network is a probability vector on the labels.

Thus Y = ∆ D , the D-dimensional probability simplex, and each label is mapped to a basis vector.

The cross-entropy loss ℓ DISPLAYFORM0 Example 2.6 (Regularized cross-entropy).

In the classification setting, it is often the case that the softmax function DISPLAYFORM1 is combined with the cross-entropy loss.

In this paper, we regard softmax as the last layer of the DNN, so we assume the output u(x) of the network lies in the probability simplex.

If the output, z, of the second to last layer of the DNN, which is the input to softmax in (6), lies in a compact set, i.e., |z j | ≤ C for all i and some C > 0, then softmax(z) j ≥ e −2C , and so the range of softmax lies in the set DISPLAYFORM2 which is strictly interior to the probability simplex.

Restricted to A, the cross-entropy loss ℓ KL is strongly convex and Lipschitz continuous, which is required in Theorems 2.12 and 2.11 below.

In our analysis, it is slightly more convenient to define the regularized cross entropy loss with parameter > 0 DISPLAYFORM3 For classification problems, where z = e k , we have ℓ KL (y, e k ) = −(1 + ) log((y k + )/(1 + )), which is Lipschitz and strongly convex for any 0 ≤ y i ≤ 1 within the probability simplex.

Thus, the regularized cross entropy ℓ KL satisfies the strong convexity and Lipschitz regularity required by Theorems 2.12 and 2.11 on the whole probability simplex.

Here, we show that solutions of the random variational problem (1) converge to solutions of (5).

We make the standard manifold assumption BID11 , and assume the data distribution ρ is a probability density supported on a compact, smooth, DISPLAYFORM0 We denote the probability density again by ρ : M → [0, ∞).

Hence, the data D n is a sequence x 1 , . . . , x n of i.i.d.

random variables on M with probability density ρ.

Associated with the random sample we have the closet point projection map σ n : X → {x 1 , . . .

, x n } ⊂ X that satisfies DISPLAYFORM1 for all x ∈ X. We recall that W 1,∞ (X; Y ) is the space of Lipschitz mappings from X to Y .

Throughout this section, C, c > 0 denote positive constants depending only on M, and we assume C ≥ 1 and 0 < c < 1.

We follow the analysis tradition of allowing the particular values of C and c to change from line to line.

We establish that that minimizers of (5) are unique on M in Theorem A.1, which follows from the strict convexity of the loss restricted to the data manifold M. See also FIG0 which shows how the solutions need not be unique off the data manifold.

Our first result is in the case where Lip[u 0 ] ≤ L 0 , and so the Lipschitz regularizer is not fully active.

This corresponds to the case of clean labels.

We state our result in generality, for approximate minimizers of (1), and specialize to the case DISPLAYFORM2 Theorem 2.7 (Convergence result).

Assume inf x∈M ρ(x) > 0.

For any t > 0, with probability at least 1 − Ct DISPLAYFORM3 is any sequence of minimizers of FORMULA0 and DISPLAYFORM4 and Theorem 2.7 applies to the sequence u n , yielding DISPLAYFORM5 It is important to note that Theorem 2.7 does not requires u n to be minimizers of (1)-we just require zero empirical loss, which is often achieved in practice (Zhang et al., 2016) .

This allows for approximation errors in solving (1) on the whole domain X, due to the restriction that u must be expressed via a Deep Neural Network.

As an immediate corollary, we can prove that the generalization loss converges to zero, and so we obtain generalization.

Corollary 2.9.

Assume that for some q ≥ 1 the loss ℓ satisfies DISPLAYFORM6 Then under the assumptions of Theorem 2.7 DISPLAYFORM7 holds with probability at least 1 − Ct −1 n −(ct−1) .Proof.

By FORMULA21 , we can bound the generalization loss as follows DISPLAYFORM8 The proof is completed by invoking Theorem 2.7.We now turn to the proof of Theorem 2.7, which requires a bound on the distance between the closest point projection σ n and the identity.

The result is standard in probability, and we include it for completeness in Lemma 2.10 proved in §A.1.

We refer the interested reader to BID34 for more details.

Lemma 2.10.

Suppose that inf M ρ > 0.

Then for any t > 0 DISPLAYFORM9 with probability at least 1 − Ct −1 n −(ct−1) .We now give the proof of Theorem 2.7.Proof of Theorem 2.7.

Since L[u n , ρ n ] = 0 we have u 0 (x i ) = u n (x i ) for all 1 ≤ i ≤ n. Thus for any x ∈ X we have DISPLAYFORM10 Therefore, we deduce DISPLAYFORM11 The proof is completed by invoking Lemma 2.10.

We now consider the setting of Problem 1.1, illustrated in Figure 1 right.

We assume that we only have access to a "dirty" label function, which corresponds to an additive error of the form DISPLAYFORM0 where u clean is the label function, and u e : X → Y is some error function, which is assumed to be zero with high probability.

Assume that the error vector e has a much larger Lipschitz constant than the labels, so that Lip(u 0 ) ≫ Lip(u clean ).We wish to fit the clean labels, while not fitting the errors, having access only to u 0 .

The labels correspond to the subset of the data which generate the low Lipschitz constant L clean , while the errors correspond to pairs of labels that generate a high Lipschitz constant.

Thus L clean can easily be estimated from the distribution of the pairwise Lipschitz constants of the data.

With the goal in mind, we set L 0 = L clean in (1).

The Lipschitz regularizer is active in (1), which can lead to the solution succeeding in avoiding the dirty labels, as in Figure 1 right.

Our main results (Theorems 2.12 and 2.11) show that minimizers of J n converge to minimizers of J almost surely as the number of training points n tends to ∞. It is beyond the scope of this work to estimate to what extent the errors are corrected, however we do know that the solution cannot fit u 0 due to the value of the Lipschitz constant, which is already an improvement over the case λ = 0.The proofs for this section can be found in Section A.2.

Theorem 2.11.

Suppose that ℓ : Y × Y → R is Lipschitz and strongly convex and let L = Lip(u 0 ).

Then for any t > 0, with probability at least 1 − 2t − m m+2 n −(ct−1) all minimizing sequences u n of (1) and all minimizers u * of (5) satisfy DISPLAYFORM1 The next result drops the assumption of strong convexity of the loss.

Theorem 2.12.

Suppose that inf M ρ > 0, ℓ : Y × Y → R is Lipschitz, and let u * ∈ W 1,∞ (X; Y ) be any minimizer of (5).

Then with probability one DISPLAYFORM2 where u n is any sequence of minimizers of (1).

Furthermore, every uniformly convergent subsequence of u n converges on X to a minimizer of (5).

Remark 2.13.

In Theorem 2.12 and Theorem 2.11, the sequence u n does not, in general, converge on the whole domain X. The important point is that the sequence converges on the data manifold M, and solves the variational problem (5) off of the manifold, which ensures that the output of the DNN is stable with respect to the input.

See FIG0 .

In this section we provide the proof of results stated in §2.3.

Theorem A.1.

Suppose the loss function satisfies Assumption 2.3.

If u, v ∈ W 1,∞ (X; Y ) are two minimizers of (5) and DISPLAYFORM0 Therefore, w is a minimizer of J and so we have equality above, which yields DISPLAYFORM1 Since ℓ is strictly convex in its first argument, it follows that u = v on M.Proof of Lemma 2.10 of §2.3.

There exists M such that for any 0 < ≤ M , we can cover M with N geodesic balls B 1 , B 2 , . . .

, B N of radius , where N ≤ C −m and C depends only on M BID25 .

Let Z i denote the number of random variables x 1 , . . .

, x n falling in B i .

Then DISPLAYFORM2 m .

Let A n denote the event that at least one B i is empty (i.e., Z i = 0 for some i).

Then by the union bound we deduce DISPLAYFORM3 In the event that A n does not occur, then each B i has at least one point, and so |x DISPLAYFORM4 The proof of Theorem 2.12 requires a preliminary Lemma.

Let DISPLAYFORM5 holds with probability at least 1 − 2t DISPLAYFORM6 The estimate FORMULA0 is called a discrepancy result BID41 BID25 , and is a uniform version of concentration inequalities.

A key tool in the proof of Lemma A.5 is Bernstein's inequality BID7 ), which we recall now for the reader's convenience.

For X 1 , . . .

, X n i.i.d.

with variance DISPLAYFORM7 surely for all i then Bernstein's inequality states that for any > 0 DISPLAYFORM8 Proof of Lemma A.5.

We note that it is sufficient to prove the result for w ∈ H L (X; Y ) with M wρ dV ol(x) = 0.

In this case, we have w(x) = 0 for some x ∈ M, and so w L ∞ (X;Y ) ≤ CL.We first give the proof for M = X = [0, 1] m .

We partition X into hypercubes B 1 , . . .

, B N of side length h > 0, where N = h −m .

Let Z j denote the number of x 1 , . . .

, x n falling in B j .

Then Z j is a Binomial random variable with parameters n and p j = Bj ρ dx ≥ ch m .

By the Bernstein inequality we have for each j that DISPLAYFORM9 provided 0 < ≤ h m .

Therefore, we deduce DISPLAYFORM10 holds with probability at least 1−2h DISPLAYFORM11 holds with probability at least 1 − 2t DISPLAYFORM12 trivially holds, and hence we can allow t > n/ log(n) as well.

We sketch here how to prove the result on the manifold M. We cover M with k geodesic balls of radius > 0, denoted B M (x 1 , ) , . . .

, B M (x k , ), and let ϕ 1 , . . . , ϕ k be a partition of unity subordinate to this open covering of M. For > 0 sufficiently small, the Riemannian exponential map exp x : B(0, ) ⊂ T x M → M is a diffeomorphism between the ball B(0, r) ⊂ T x M and the geodesic ball B M (x, ) ⊂ M, where T x M ∼ = R m .

Furthermore, the Jacobian of exp x at v ∈ B(0, r) ⊂ T x M, denoted by J x (v), satisfies (by the Rauch Comparison Theorem) DISPLAYFORM13 Therefore, we can run the argument above on the ball B(0, r) ⊂ R m in the tangent space, lift the result to the geodesic ball B M (x i , ) via the Riemannian exponential map exp x , and apply the bound DISPLAYFORM14 to complete the proof.

Remark A.6.

The exponent 1/(m + 2) is not optimal, but affords a very simple proof.

It is possible to prove a similar result with the optimal exponent 1/m in dimension m ≥ 3, but the proof is significantly more involved.

We refer the reader to BID41 for details.

Remark A.7.

The proof of Theorem 2.12 shows that (1) Γ-converges to (5) almost surely as n → ∞ in the L ∞ (X; Y ) topology.

Γ-convergence is a notion of convergence for functionals that ensures minimizers along a sequence of functionals converge to a minimizer of the Γ-limit.

While we do not use the language of Γ-convergence here, the ideas are present in the proof of Theorem 2.12.

We refer to BID9 for details on Γ-convergence.

Proof of Theorem 2.12.

By Lemma A.5 the event that lim DISPLAYFORM15 for all Lipschitz constants L > 0 has probability one.

For the rest of the proof we restrict ourselves to this event.

Let u n ∈ W 1,∞ (X; Y ) be a sequence of minimizers of (1), and let u * ∈ W 1,∞ (X; Y ) be any minimizer of (5).

Then since DISPLAYFORM16 we have Lip(u n ) ≤ Lip(u 0 ) =: L for all n. By the Arzelà-Ascoli Theorem BID37 there exists a subsequence u nj and a function u ∈ W 1,∞ (X; Y ) such that u nj → u uniformly as n j → ∞. Note we also have Lip(u) ≤ lim inf j→∞ Lip(u nj ).

Since DISPLAYFORM17 Therefore, u is a minimizer of J. By Theorem A.1, u = u * on M, and so u nj →

u * uniformly on M as j → ∞. Now, suppose that (8) does not hold.

Then there exists a subsequence u nj and δ > 0 such that DISPLAYFORM18 for all j ≥ 1.

However, we can apply the argument above to extract a further subsequence of u nj that converges uniformly on M to u * , which is a contradiction.

This completes the proof.

Proof of Theorem 2.11.

Let L = Lip(u 0 ).

By Lemma A.5 DISPLAYFORM19 ( FORMULA0 holds with probability at least 1 − 2t − m m+2 n −(ct−1) for any t > 0.

Let us assume for the rest of the proof that (13) holds. (LE) is not practical for large scale problems.

There has be extensive work on the Lipschitz Extension problem, see, BID28 , for example.

More recently, optimal Lipschitz extensions have been studied, with connections to Partial Differential Equations, see BID2 .

We can interpret (1) as a relaxed version of (LE), where λ −1 is a parameter which replaces the unknown Lagrange multiplier for the constraint.

Variational problems are fundamental tools in mathematical approaches to image processing BID3 Lipschitz regularization in not nearly as common.

It appears in image processing in (Pock et al., 2010, §4.4) BID16 and BID24 ).

Variational problems of the form (14) can be studied by the direct method in the calculus of variations BID13 .

The problem (14) can be discretized to obtain a finite dimensional convex convex optimization problem.

The variational problem can also be studied by finding the first variation, which is a Partial Differential Equation BID17 , which can then be solved numerically.

Both approaches are discussed in BID3 .

In FIG1 we compare different regularization terms, in one dimension.

The difference between the regularizers is more extreme in higher dimensions.

<|TLDR|>

@highlight

We prove generalization of DNNs by adding a Lipschitz regularization term to the training loss. We resolve a question posed in Zhang et al. (2016).
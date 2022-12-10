Randomly initialized first-order optimization algorithms are the method of choice for solving many high-dimensional nonconvex problems in machine learning, yet general theoretical guarantees cannot rule out convergence to critical points of poor objective value.

For some highly structured nonconvex problems however, the success of gradient descent can be understood by studying the geometry of the objective.

We study one such problem -- complete orthogonal dictionary learning, and provide converge guarantees for randomly initialized gradient descent to the neighborhood of a global optimum.

The resulting rates scale as low order polynomials in the dimension even though the objective possesses an exponential number of saddle points.

This efficient convergence can be viewed as a consequence of negative curvature normal to the stable manifolds associated with saddle points, and we provide evidence that this feature is shared by other nonconvex problems of importance as well.

Many central problems in machine learning and signal processing are most naturally formulated as optimization problems.

These problems are often both nonconvex and highdimensional.

High dimensionality makes the evaluation of second-order information prohibitively expensive, and thus randomly initialized first-order methods are usually employed instead.

This has prompted great interest in recent years in understanding the behavior of gradient descent on nonconvex objectives (18; 14; 17; 11) .

General analysis of first-and second-order methods on such problems can provide guarantees for convergence to critical points but these may be highly suboptimal, since nonconvex optimization is in general an NP-hard probem BID3 .

Outside of a convex setting (28) one must assume additional structure in order to make statements about convergence to optimal or high quality solutions.

It is a curious fact that for certain classes of problems such as ones that involve sparsification (25; 6) or matrix/tensor recovery (21; 19; 1) first-order methods can be used effectively.

Even for some highly nonconvex problems where there is no ground truth available such as the training of neural networks first-order methods converge to high-quality solutions (40).Dictionary learning is a problem of inferring a sparse representation of data that was originally developed in the neuroscience literature (30), and has since seen a number of important applications including image denoising, compressive signal acquisition and signal classification (13; 26) .

In this work we study a formulation of the dictionary learning problem that can be solved efficiently using randomly initialized gradient descent despite possessing a number of saddle points exponential in the dimension.

A feature that appears to enable efficient optimization is the existence of sufficient negative curvature in the directions normal to the stable manifolds of all critical points that are not global minima BID0 .

This property ensures that the regions of the space that feed into small gradient regions under gradient flow do not dominate the parameter space.

FIG0 illustrates the value of this property: negative curvature prevents measure from concentrating about the stable manifold.

As a consequence randomly initialized gradient methods avoid the "slow region" of around the saddle point.

Negative curvature helps gradient descent.

Red: "slow region" of small gradient around a saddle point.

Green: stable manifold associated with the saddle point.

Black: points that flow to the slow region.

Left: global negative curvature normal to the stable manifold.

Right: positive curvature normal to the stable manifold -randomly initialized gradient descent is more likely to encounter the slow region.

The main results of this work is a convergence rate for randomly initialized gradient descent for complete orthogonal dictionary learning to the neighborhood of a global minimum of the objective.

Our results are probabilistic since they rely on initialization in certain regions of the parameter space, yet they allow one to flexibly trade off between the maximal number of iterations in the bound and the probability of the bound holding.

While our focus is on dictionary learning, it has been recently shown that for other important nonconvex problems such as phase retrieval BID7 performance guarantees for randomly initialized gradient descent can be obtained as well.

In fact, in Appendix C we show that negative curvature normal to the stable manifolds of saddle points (illustrated in FIG0 ) is also a feature of the population objective of generalized phase retrieval, and can be used to obtain an efficient convergence rate.

Easy nonconvex problems.

There are two basic impediments to solving nonconvex problems globally: (i) spurious local minimizers, and (ii) flat saddle points, which can cause methods to stagnate in the vicinity of critical points that are not minimizers.

The latter difficulty has motivated the study of strict saddle functions (36; BID13 , which have the property that at every point in the domain of optimization, there is a large gradient, a direction of strict negative curvature, or the function is strongly convex.

By leveraging this curvature information, it is possible to escape saddle points and obtain a local minimizer in polynomial time.2 Perhaps more surprisingly, many known strict saddle functions also have the property that every local minimizer is global; for these problems, this implies that efficient methods find global solutions.

Examples of problems with this property include variants of sparse dictionary learning (38), phase retrieval (37), tensor decomposition BID13 , community detection (3) and phase synchronization BID4 .Minimizing strict saddle functions.

Strict saddle functions have the property that at every saddle point there is a direction of strict negative curvature.

A natural approach to escape such saddle points is to use second order methods (e.g., trust region BID8 or curvilinear search BID14 ) that explicitly leverage curvature information.

Alternatively, one can attempt to escape saddle points using first order information only.

However, some care is needed: canonical first order methods such as gradient descent will not obtain minimizers if initialized at a saddle point (or at a point that flows to one) -at any critical point, gradient descent simply stops.

A natural remedy is to randomly perturb the iterate whenever needed.

A line of recent works shows that noisy gradient methods of this form efficiently optimize strict saddle functions (24; 12; 20) .

For example, (20) obtains rates on strict saddle functions that match the optimal rates for smooth convex programs up to a polylogarithmic dependence on dimension.

Randomly initialized gradient descent?

The aforementioned results are broad, and nearly optimal.

Nevertheless, important questions about the behavior of first order methods for nonconvex optimization remain unanswered.

For example: in every one of the aforemented benign nonconvex optimization problems, randomly initialized gradient descent rapidly obtains a minimizer.

This may seem unsurprising: general considerations indicate that the stable manifolds associated with non-minimizing critical points have measure zero (29), this implies that a variety of small-stepping first order methods converge to minimizers in the large-time limit (23).

However, it is not difficult to construct strict saddle problems that are not amenable to efficient optimization by randomly initialized gradient descent -see BID11 for an example.

This contrast between the excellent empirical performance of randomly initialized first order methods and worst case examples suggests that there are important geometric and/or topological properties of "easy nonconvex problems" that are not captured by the strict saddle hypothesis.

Hence, the motivation of this paper is twofold: (i) to provide theoretical corroboration (in certain specific situations) for what is arguably the simplest, most natural, and most widely used first order method, and (ii) to contribute to the ongoing effort to identify conditions which make nonconvex problems amenable to efficient optimization.

Suppose we are given data matrix Y = y 1 , . . .

y p ∈ R n×p .

The dictionary learning problem asks us to find a concise representation of the data BID12 , of the form Y ≈ AX, where X is a sparse matrix.

In the complete, orthogonal dictionary learning problem, we restrict the matrix A to have orthonormal columns (A ∈ O(n)).

This variation of dictionary learning is useful for finding concise representations of small datasets (e.g., patches from a single image, in MRI FORMULA2 ).To analyze the behavior of dictionary learning algorithms theoretically, it useful to posit that Y = A 0 X 0 for some true dictionary A 0 ∈ O(n) and sparse coefficient matrix X 0 ∈ R n×p , and ask whether a given algorithm recovers the pair (A 0 , X 0 ).

BID3 In this work, we further assume that the sparse matrix X 0 is random, with entries i.i.d.

Bernoulli-Gaussian 5 .

For simplicity, we will let A 0 = I; our arguments extend directly to general A 0 via the simple change of variables q →

A * 0 q. (34) showed that under mild conditions, the complete dictionary recovery problem can be reduced to the geometric problem of finding a sparse vector in a linear subspace (31).

Notice that because A 0 is orthogonal, row(Y ) = row(X 0 ).

Because X 0 is a sparse random matrix, the rows of X 0 are sparse vectors.

Under mild conditions (34), they are the sparsest vectors in the row space of Y , and hence can be recovered by solving the conceptual optimization problem min q DISPLAYFORM0 This is not a well-structured optimization problem: the objective is discontinuous, and the constraint set is open.

A natural remedy is to replace the 0 norm with a smooth sparsity surrogate, and to break the scale ambiguity by constraining q to the sphere, giving DISPLAYFORM1 Here, we choose h µ (t) = µ log(cosh(t/µ)) as a smooth sparsity surrogate.

This objective was analyzed in (35), which showed that (i) although this optimization problem is nonconvex, when the data are sufficiently large, with high probability every local optimizer is near a signed column of the true dictionary A 0 , (ii) every other critical point has a direction of strict negative curvature, and (iii) as a consequence, a second-order Riemannian trust region method efficiently recovers a column of A 0 .

BID5 The Riemannian trust region method is of mostly theoretical interest: it solves complicated (albeit polynomial time) subproblems that involve the Hessian of f DL .

Note the similarity to the dictionary learning objective.

Right: The objective for complete orthogonal dictionary learning (discussed in section 6) for n = 3.In practice, simple iterative methods, including randomly initialized gradient descent are also observed to rapidly obtain high-quality solutions.

In the sequel, we will give a geometric explanation for this phenomenon, and bound the rate of convergence of randomly initialized gradient descent to the neighborhood of a column of A 0 .

Our analysis of f DL is probabilistic in nature: it argues that with high probability in the sparse matrix X 0 , randomly initialized gradient descent rapidly produces a minimizer.

To isolate more clearly the key intuitions behind this analysis, we first analyze the simpler separable objective DISPLAYFORM2 Figure 2 plots both f Sep and f DL as functions over the sphere.

Notice that many of the key geometric features in f DL are present in f Sep ; indeed, f Sep can be seen as an "ultrasparse" version of f DL in which the columns of the true sparse matrix X 0 are taken to have only one nonzero entry.

A virtue of this model function is that its critical points and their stable manifolds have simple closed form expressions (see Lemma 1).

Our problems of interest have the form DISPLAYFORM0 where f : R n → R is a smooth function.

We let ∇f (q) and ∇ 2 f (q) denote the Euclidean gradient and hessian (over R n ), and let grad [f ] (q) and Hess [f ] (q) denote their Riemannian counterparts (over S n−1 ).

We will obtain results for Riemannian gradient descent defined by the update DISPLAYFORM1 for some step size η > 0, where exp q : T q S n−1 → S n−1 is the exponential map.

The Riemannian gradient on the sphere is given by grad[f ]

(q) = (I − qq * )∇f (q).We let A denote the set of critical points of f over S n−1 -these are the pointsq s.t.

grad [f ] (q) = 0.

We letȂ denote the set of local minimizers, and "A its complement.

Both f Sep and f DL are Morse functions on S n−1 , 7 we can assign an index α to everyq ∈ A, which is the number of negative eigenvalues of Hess [f ]

(q).Our goal is to understand when gradient descent efficiently converges to a local minimizer.

In the small-step limit, gradient descent follows gradient flow lines γ : R → M, which are solution curves of the ordinary differential equatioṅ DISPLAYFORM2 To each critical point α ∈ A of index λ, there is an associated stable manifold of dimension dim(M) − λ, which is roughly speaking, the set of points that flow to α under gradient flow: DISPLAYFORM3 7 Strictly speaking, fDL is Morse with high probability, due to results of (38).

Negative curvature and efficient gradient descent.

The union of the light blue, orange and yellow sets is the set C. In the light blue region, there is negative curvature normal to ∂C, while in the orange region the gradient norm is large, as illustrated by the arrows.

There is a single global minimizer in the yellow region.

For the separable objective, the stable manifolds of the saddles and maximizers all lie on ∂C (the black circles denote the critical points, which are either maximizers " ", saddles " ", or minimizers " ").

The red dots denote ∂C ζ with ζ = 0.2.Our analysis uses the following convenient coordinate chart DISPLAYFORM0 where w ∈ B 1 (0).

We also define two useful sets: DISPLAYFORM1 Since the problems considered here are symmetric with respect to a signed permutation of the coordinates we can consider a certain C and the results will hold for the other symmetric sections as well.

We will show that at every point in C aside from a neighborhood of a global minimizer for the separable objective (or a solution to the dictionary problem that may only be a local minimizer), there is either a large gradient component in the direction of the minimizer or negative curvature in a direction normal to ∂C. For the case of the separable objective, one can show that the stable manifolds of the saddles lie on this boundary, and hence this curvature is normal to the stable manifolds of the saddles and allows rapid progress away from small gradient regions and towards a global minimizer BID7 .

These regions are depicted in Figure 3 .In the sequel, we will make the above ideas precise for the two specific nonconvex optimization problems discussed in Section 3 and use this to obtain a convergence rate to a neighborhood of a global minimizer.

Our analysis are specific to these problems.

However, as we will describe in more detail later, they hinge on important geometric characteristics of these problems which make them amenable to efficient optimization, which may obtain in much broader classes of problems.

In this section, we study the behavior of randomly initialized gradient descent on the separable function f Sep .

We begin by characterizing the critical points:Lemma 1 (Critical points of f Sep ).

The critical points of the separable problem (2) are DISPLAYFORM0 For every α ∈ A and corresponding a(α), for µ < c √ n log n the stable manifold of α takes the form DISPLAYFORM1 where c > 0 is a numerical constant.

Proof.

Please see Appendix ABy inspecting the dimension of the stable manifolds, it is easy to verify that that there are 2n global minimizers at the 1-sparse vectors on the sphere ± e i , 2 n maximizers at the least sparse vectors and an exponential number of saddle points of intermediate sparsity.

This is because the dimension of W s (α) is simply the dimension of b in 6, and it follows directly from the stable manifold theorem that only minimizers will have a stable manifold of dimension n − 1.

The objective thus possesses no spurious local minimizers.

When referring to critical points and stable manifolds from now on we refer only to those that are contained in C or on its boundary.

It is evident from Lemma 1 that the critical points in "A all lie on ∂C and that α∈ " A W s (α) = ∂C , and there is a minimizer at its center given by q(0) = e n .

We now turn to making precise the notion that negative curvature normal to stable manifolds of saddle points enables gradient descent to rapidly exit small gradient regions.

We do this by defining vector fields u (i) (q), i ∈ [n − 1] such that each field is normal to a continuous piece of ∂C ζ and points outwards relative to C ζ defined in 4.

By showing that the Riemannian gradient projected in this direction is positive and proportional to ζ, we are then able to show that gradient descent acts to increase ζ(q(w)) = qn w ∞ − 1 geometrically.

This corresponds to the behavior illustrated in the light blue region in Figure 3 .

DISPLAYFORM0 DISPLAYFORM1 where c > 0 is a numerical constant.

Proof.

Please see Appendix A.Since we will use this property of the gradient in C ζ to derive a convergence rate, we will be interested in bounding the probability that gradient descent initialized randomly with respect to a uniform measure on the sphere is initialized in C ζ .

This will require bounding the volume of this set, which is done in the following lemma: DISPLAYFORM2 Proof.

Please see Appendix D.3.

Using the results above, one can obtain the following convergence rate:Theorem 1 (Gradient descent convergence rate for separable function).

For any 0 < ζ 0 < 1, r > µ log on the separable objective (2) with µ < c2 √ n log n , enters an L ∞ ball of radius r around a global minimizer in T < C η √ n r 2 + log 1 ζ 0 iterations with probability DISPLAYFORM0 Proof.

Please see Appendix A.We have thus obtained a convergence rate for gradient descent that relies on the negative curvature around the stable manifolds of the saddles to rapidly move from these regions of the space towards the vicinity of a global minimizer.

This is evinced by the logarithmic dependence of the rate on ζ.

As was shown for orthogonal dictionary learning in (38), we also expect a linear convergence rate due to strong convexity in the neighborhood of a minimizer, but do not take this into account in the current analysis.

The proofs in this section will be along the same lines as those of Section 5.

While we will not describe the positions of the critical points explicitly, the similarity between this objective and the separable function motivates a similar argument.

It will be shown that initialization in some C ζ will guarantee that Riemannian gradient descent makes uniform progress in function value until reaching the neighborhood of a global minimizer.

We will first consider the population objective which corresponds to the infinite data limit DISPLAYFORM0 and then bounding the finite sample size fluctuations of the relevant quantities.

We begin with a lemma analogous to Lemma 2:Lemma 4 (Dictionary learning population gradient).

For w ∈ C ζ , r < |w i |, µ < c 1 r DISPLAYFORM1 √ ζ the dictionary learning population objective 8 obeys DISPLAYFORM2 where c θ depends only on θ, c 1 is a positive numerical constant and u (i) is defined in 7.Proof.

Please see Appendix BUsing this result, we obtain the desired convergence rate for the population objective, presented in Lemma 11 in Appendix B. After accounting for finite sample size fluctuations in the gradient, one obtains a rate of convergence to the neighborhood of a solution (which is some signed basis vector due to our choice A 0 = I) Theorem 2 (Gradient descent convergence rate for dictionary learning).

DISPLAYFORM3 , Riemannian gradient descent with step size η < c5θs n log np on the dictionary learning objective 1 with µ < c6 √ ζ0 DISPLAYFORM4 , enters a ball of radius c 3 s from a target solution in DISPLAYFORM5 iterations with probability DISPLAYFORM6 where y = DISPLAYFORM7 , P y is given in Lemma 10 and c i , C i are positive constants.

Proof.

Please see Appendix BThe two terms in the rate correspond to an initial geometric increase in the distance from the set containing the small gradient regions around saddle points, followed by convergence to the vicinity of a minimizer in a region where the gradient norm is large.

The latter is based on results on the geometry of this objective provided in (38).

The above analysis suggests that second-order properties -namely negative curvature normal to the stable manifolds of saddle points -play an important role in the success of randomly initialized gradient descent in the solution of complete orthogonal dictionary learning.

This was done by furnishing a convergence rate guarantee that holds when the random initialization is not in regions that feed into small gradient regions around saddle points, and bounding the probability of such an initialization.

In Appendix C we provide an additional example of a nonconvex problem that for which an efficient rate can be obtained based on an analysis that relies on negative curvature normal to stable manifolds of saddles -generalized phase retrieval.

An interesting direction of further work is to more precisely characterize the class of functions that share this feature.

The effect of curvature can be seen in the dependence of the maximal number of iterations T on the parameter ζ 0 .

This parameter controlled the volume of regions where initialization would lead to slow progress and the failure probability of the bound 1 − P was linear in ζ 0 , while T depended logarithmically on ζ 0 .

This logarithmic dependence is due to a geometric increase in the distance from the stable manifolds of the saddles during gradient descent, which is a consequence of negative curvature.

Note that the choice of ζ 0 allows one to flexibly trade off between T and 1 − P. By decreasing ζ 0 , the bound holds with higher probability, at the price of an increase in T .

This is because the volume of acceptable initializations now contains regions of smaller minimal gradient norm.

In a sense, the result is an extrapolation of works such as (23) that analyze the ζ 0 = 0 case to finite ζ 0 .Our analysis uses precise knowledge of the location of the stable manifolds of saddle points.

For less symmetric problems, including variants of sparse blind deconvolution (41) and overcomplete tensor decomposition, there is no closed form expression for the stable manifolds.

However, it is still possible to coarsely localize them in regions containing negative curvature.

Understanding the implications of this geometric structure for randomly initialized first-order methods is an important direction for future work.

One may hope that studying simple model problems and identifying structures (here, negative curvature orthogonal to the stable manifold) that enable efficient optimization will inspire approaches to broader classes of problems.

One problem of obvious interest is the training of deep neural networks for classification, which shares certain high-level features with the problems discussed in this paper.

The objective is also highly nonconvex and is conjectured to contain a proliferation of saddle points BID10 , yet these appear to be avoided by first-order methods BID15 for reasons that are still quite poorly understood beyond the two-layer case (39).[19]

Prateek Jain, Praneeth Netrapalli, and Sujay Sanghavi.

Low-rank matrix completion using alternating minimization.

DISPLAYFORM0 .

Thus critical points are ones where either tanh( q µ ) = 0 (which cannot happen on S n−1 ) or tanh( q µ ) is in the nullspace of (I − qq * ), which implies tanh( q µ ) = cq for some constant b. The equation tanh( x µ ) = bx has either a single solution at the origin or 3 solutions at {0, ±r(b)} for some r(b).

Since this equation must be solves simultaneously for every element of q, we obtain ∀i ∈ [n] : q i ∈ {0, ±r(b)}. To obtain solutions on the sphere, one then uses the freedom we have in choosing b (and thus r(b)) such that q = 1.

The resulting set of critical points is thus DISPLAYFORM1 To prove the form of the stable manifolds, we first show that for q i such that |q i | = q ∞ and any q j such that |q j | + ∆ = |q i | and sufficiently small ∆ > 0, we have DISPLAYFORM2 For ease of notation we now assume q i , q j > 0 and hence ∆ = q i − q j , otherwise the argument can be repeated exactly with absolute values instead.

The above inequality can then be written as DISPLAYFORM3 If we now define DISPLAYFORM4 where the O(∆ 2 ) term is bounded.

Defining a vector r ∈ R n by DISPLAYFORM5 we have r 2 = 1.

Since tanh(x) is concave for x > 0, and |r i | ≤ 1, we find DISPLAYFORM6 From DISPLAYFORM7 and thus q j ≥ 1 √ n − ∆. Using this inequality and properties of the hyperbolic secant we obtain DISPLAYFORM8 and plugging in µ = c √ n log n for some c < 1 DISPLAYFORM9 log n + log log n + log 4).We can bound this quantity by a constant, say h 2 ≤ 1 2 , by requiring DISPLAYFORM10 ) log n +

log log n ≤ − log 8and for and c < 1, using −

log n + log log n < 0 we have DISPLAYFORM11 Since ∆ can be taken arbitrarily small, it is clear that c can be chosen in an n-independent manner such that A ≤ − log 8.

We then find DISPLAYFORM12 since this inequality is strict, ∆ can be chosen small enough such that O(∆ 2 ) < ∆(h 1 − h 2 ) and hence h > 0, proving 9.It follows that under negative gradient flow, a point with |q j | < ||q|| ∞ cannot flow to a point q such that |q j | = ||q || ∞ .

From the form of the critical points, for every such j, q must thus flow to a point such that q j = 0 (the value of the j coordinate cannot pass through 0 to a point where |q j | = ||q || ∞ since from smoothness of the objective this would require passing some q with q j = 0, at which point grad [f Sep ] (q ) j = 0).As for the maximal magnitude coordinates, if there is more than one coordinate satisfying |q i1 | = |q i2 | = q ∞ , it is clear from symmetry that at any subsequent point q along the gradient flow line q i1 = q i2 .

These coordinates cannot change sign since from the smoothness of the objective this would require that they pass through a point where they have magnitude smaller than 1/ √ n, at which point some other coordinate must have a larger magnitude (in order not to violate the spherical constraint), contradicting the above result for non-maximal elements.

It follows that the sign pattern of these elements is preserved during the flow.

Thus there is a single critical point to which any q can flow, and this is given by setting all the coordinates with |q j | < q ∞ to 0 and multiplying the remaining coordinates by a positive constant to ensure the resulting vector is on S n .

Denoting this critical point by α, there is a vector b such that q = P S n−1 [a(α) + b] and supp(a(α))

∩ supp(b) = ∅, b ∞ < 1 with the form of a(α) given by 5 .

The collection of all such points defines the stable manifold of α.

Proof of Lemma 2: (Separable objective gradient projection).

i) We consider the sign(w i ) = 1 case; the sign(w i ) = −1 case follows directly.

Recalling that DISPLAYFORM13 qn , we first prove DISPLAYFORM14 for some c > 0 whose form will be determined later.

The inequality clearly holds for w i = q n .To DISPLAYFORM15 verify that it holds for smaller values of w i as well, we now show that ∂ ∂w i tanh w i µ − tanh q n µ w i q n − c(q n − w i ) < 0 which will ensure that it holds for all w i .

We define s 2 = 1 − ||w|| 2 + w 2 i and denote q n = s 2 − w 2 i to extract the w i dependence, givingWhere in the last inequality we used properties of the sech function and q n ≥ w i .

We thus want to show DISPLAYFORM16 and it follows that 10 holds.

For µ < 1 BID15 we are guaranteed that c > 0.From examining the RHS of 10 (and plugging in q n = s 2 − w 2 i ) we see that any lower bound on the gradient of an element w j applies also to any element |w i | ≤ |w j |.

Since for |w j | = ||w|| ∞ we have q n − w j = w j ζ, for every log( 1 µ )µ ≤ w i we obtain the bound DISPLAYFORM17 Proof of Theorem 1: (Gradient descent convergence rate for separable function).We obtain a convergence rate by first bounding the number of iterations of Riemannian gradient descent in C ζ0 \C 1 , and then considering DISPLAYFORM18 .

Choosing c 2 so that µ < 1 2 , we can apply Lemma 2, and for u defined in 7, we thus have DISPLAYFORM19 Since from Lemma 7 the Riemannian gradient norm is bounded by √ n, we can choose c 1 , c 2 such that µ log( DISPLAYFORM20 .

This choice of η then satisfies the conditions of Lemma 17 with r = µ log( DISPLAYFORM21 , M = √ n, which gives that after a gradient step DISPLAYFORM22 for some suitably chosenc > 0.

If we now define by w (t) the t-th iterate of Riemannian gradient descent and DISPLAYFORM23 and the number of iterations required to exit C ζ0 \C 1 is DISPLAYFORM24 To bound the remaining iterations, we use Lemma 2 to obtain that for every w ∈ C ζ0 \B ∞ r , DISPLAYFORM25 where we have used ||u DISPLAYFORM26 We thus have DISPLAYFORM27 Choosing DISPLAYFORM28 where L is the gradient Lipschitz constant of f s , from Lemma 5 we obtain DISPLAYFORM29 According to Lemma B, L = 1/µ and thus the above holds if we demand η < µ 2 .

Combining 12 and 13 gives DISPLAYFORM30 .To obtain the final rate, we use in g(w 0 ) − g * ≤ √ n andcη < 1 ⇒ 1 log(1+cη) <C cη for somẽ C > 0.

Thus one can choose C > 0 such that DISPLAYFORM31 From Lemma 1 the ball B ∞ r contains a global minimizer of the objective, located at the origin.

The probability of initializing in Ȃ C ζ0 is simply given from Lemma 3 and by summing over the 2n possible choices of C ζ0 , one for each global minimizer (corresponding to a single signed basis vector).

, where L is a lipschitz constant for ∇f (q), one has DISPLAYFORM0 Proof.

Just as in the euclidean setting, we can obtain a lower bound on progress in function values of iterates of the Riemannian gradient descent algorithm from a lower bound on the Riemannian gradient.

Consider f : S n−1 → R, which has L-lipschitz gradient.

Let q k denote the current iterate of Riemannian gradient descent, and let t k > 0 denote the step size.

Then we can form the Taylor approximation to f • Exp q k (v) at 0 q k : DISPLAYFORM1

where the matrix norm is the operator norm on R n×n .

Using the gradient-lipschitz property of f , we readily compute DISPLAYFORM0 since ∇f (0) = 0 and q k ∈ S n−1 .

We thus have DISPLAYFORM1 If we put v = −t k grad[f ](q k ) and write q k+1 = Exp q k (−t k grad [f ] (q k )), the previous expression becomes DISPLAYFORM2 .

Thus progress in objective value is guaranteed by lower-bounding the Riemannian gradient.

As in the euclidean setting, summing the previous expression over iterations k now yields DISPLAYFORM3 Plugging in a constant step size gives the desired result.

Lemma 6 (Lipschitz constant of ∇f ).

For any x 1 , x 2 ∈ R n , it holds DISPLAYFORM4 Proof.

It will be enough to study a single coordinate function of ∇f .

Using a derivative given in section D.1, we have for DISPLAYFORM5 A bound on the magnitude of the derivative of this smooth function implies a lipschitz constant for x → tanh(x/µ).

To find the bound, we differentiate again and find the critical points of the function.

We have, using the chain rule, d dx DISPLAYFORM6 (e x/µ + e −x/µ ) 3 .

The denominator of this final expression vanishes nowhere.

Hence, the only critical point satisfies x/µ = −x/µ, which implies x = 0.

Therefore it holds DISPLAYFORM7 which shows that tanh(x/µ) is (1/µ)-lipschitz.

Now let x 1 and x 2 be any two points of R n .

Then one has DISPLAYFORM8 completing the proof.

Proof of Lemma 4:(Dictionary learning population gradient).

For simplicity we consider the case sign(w i ) = 1.

The converse follows by a similar argument.

We have DISPLAYFORM0 Following the notation of (38), we write x j = b j v j where b j ∼ Bern(θ), v j ∼ N (0, 1) and denote the vectors of these variables by J , v respectively.

Defining DISPLAYFORM1

and similarly the second term in 15 is, with DISPLAYFORM0

We already have a lower bound in Lemma 20 of (38) that we can use for the second term, so we need an upper bound for the first term.

Following from p. 865, we define DISPLAYFORM0 , and defining DISPLAYFORM1 Where b k = (−β) k (k + 1).

Using B.3 from Lemma 40 in (38) we have DISPLAYFORM2 Where Φ c (x) is the complementary Gaussian CDF (The exchange of summation and expectation is justified since Y > 0 implies Z ∈ [0, 1], see proof of Lemma 18 in (38) for details).

Using the following bounds DISPLAYFORM3 2 /2 by applying the upper (lower) bound to the even (odd) terms in the sum, and then adding a non-negative quantity, we obtain DISPLAYFORM4 and using Lemma 17 in (38) ) and taking T → ∞ so that β → 1 we have DISPLAYFORM5 DISPLAYFORM6 giving the upper bound DISPLAYFORM7 while the lower bound (Lemma 20 in (38)) is DISPLAYFORM8

After conditioning on J \{n, i} the variables X + q n v n , X + q i v i are Gaussian.

We can thus plug the bounds into 16 to obtain DISPLAYFORM0 the term in the expectation is positive since q n > ||w|| ∞ (1 + ζ) > w i giving DISPLAYFORM1 To extract the ζ dependence we plug in q n > w i (1 + ζ) and develop to first order in ζ (since the resulting function of ζ is convex) giving DISPLAYFORM2 Given some ζ and r such that w i > r, if we now choose µ such that µ < Lemma 8 (Point-wise concentration of projected gradient).

For u (i) defined in 7, the gradient of the objective 1 obeys DISPLAYFORM3 Proof of Lemma 8: (Point-wise concentration of projected gradient).

If we denote by x i a column of the data matrix with entries x i j ∼ BG(θ), we have DISPLAYFORM4 .

Since tanh(x) is bounded by 1, DISPLAYFORM5 Invoking Lemma 21 from (38) and u 2 = 1 + DISPLAYFORM6 and using Lemma 36 in (38) with R = √ 2, σ = √ 2 we have DISPLAYFORM7 Lemma 9 (Projection Lipschitz Constant).

The Lipschitz constant for DISPLAYFORM8 Proof of Lemma 9: (Projection Lipschitz Constant).

We have DISPLAYFORM9 where we have defined DISPLAYFORM10 We also use the fact that tanh is bounded by 1 and s(w) is bounded by X ∞ .

We can then use Lemma 23 in (38) to obtain DISPLAYFORM11 Lemma 10 (Uniformized gradient fluctuations).

For all w ∈ C ζ , i ∈ [n], with probability P > P y we have DISPLAYFORM12 where DISPLAYFORM13 Proof of Lemma 10:(Uniformized gradient fluctuations).

For X ∈ R n×p with i.i.d.

BG(θ) entries, we define the event E ∞ ≡ {1 ≤ X ∞ ≤ 4 log(np)}. We have DISPLAYFORM14 For any ε ∈ (0, 1) we can construct an ε-net N for C ζ \B 2 1/20 DISPLAYFORM15 .

If we choose ε = y(θ,ζ) DISPLAYFORM16 We then denote by E g the event DISPLAYFORM17 2 in the result of Lemma 8 gives that for all w ∈ C ζ , i ∈ [n], DISPLAYFORM18 py(θ, ζ) Proof of Lemma 11: (Gradient descent convergence rate for dictionary learning -population).

The rate will be obtained by splitting C ζ0 into three regions.

We consider convergence to B 2 s (0) since this set contains a global minimizer.

Note that the balls in the proof are defined with respect to w. DISPLAYFORM19 The analysis in this region is completely analogous to that in the first part of the proof of Lemma 1.

For every point in this set we have DISPLAYFORM20 .

From Lemma 16 we know that , since for every point in this region r 3 ζ < 1, we have DISPLAYFORM21 DISPLAYFORM22 r = z(r, ζ) and we thus demand µ < √ ζ0 DISPLAYFORM23 and obtain from Lemma 4 that for |w i | > r DISPLAYFORM24 .

We now require η < , M = √ θn (since the maximal norm of the Riemannian gradient is √ θn from Lemma 12), obtaining that at every iteration in this region ζ ≥ ζ 1 + √ nc DL 2(8000(n − 1))

3/2 η and the maximal number of iterations required to obtain ζ > 8 and exit this region is given by DISPLAYFORM25 According to Proposition 7 in (38), which we can apply since s ≥ DISPLAYFORM26 .

Defining h(q) = w 2 2 , and denoting by q an update of Riemannian gradient descent with step size η, we have (using a Lagrange remainder term) DISPLAYFORM27 where in the last line we used q = cos(gη)q − sin(gη) DISPLAYFORM28 we obtain (using 18) DISPLAYFORM29 and thus choosing η < DISPLAYFORM30 we find DISPLAYFORM31 Under review as a conference paper at ICLR 2019 and in our region of interest w 2 < w 2 −csθη for somec > 0 and thus summing over iterations, we obtain for someC 2 > 0 DISPLAYFORM32 From Lemma 12, M = √ θn and thus with a suitably chosen c 2 > 0, η < c2s n satisfies the above requirement on η as well as the previous requirements, since θ < 1.

Combining these results gives, we find that when initializing in C ζ0 , the maximal number of iterations required for Riemannian gradient descent to enter B 2 s (0) is DISPLAYFORM0 for some suitably chosen C 1 , where t 1 , t 2 are given in 17,19.

The probability of such an initialization is given by the probability of initializing in one of the 2n possible choices of C ζ , which is bounded in Lemma 3.Once w ∈ B 2 s (0), the distance in R n−1 between w and a solution to the problem (which is a signed basis vector, given by the point w = 0 or an analog on a different symmetric section of the sphere) is no larger than s, which in turn implies that the Riemannian distance between ϕ(w) and a solution is no larger than c 3 s for some c 3 > 0.

We note that the conditions on µ can be satisfied by requiring µ < DISPLAYFORM1 where X is the data matrix with i.i.d.

BG(θ) entries.

Proof.

Denoting x ≡ (x, x n ) we have DISPLAYFORM2 and using Jensen's inequality, convexity of the L 2 norm and the triangle inequality to obtain DISPLAYFORM3 Similarly, in the finite sample size case one obtains DISPLAYFORM4 Proof of Theorem 2: (Gradient descent convergence rate for dictionary learning).

The proof will follow exactly that of Lemma 11, with the finite sample size fluctuations decreasing the guaranteed change in ζ or ||w|| at every iteration (for the initial and final stages respectively) which will adversely affect the bounds.

DISPLAYFORM5 To control the fluctuations in the gradient projection, we choose DISPLAYFORM6 which can be satisfied by choosing y(θ, ζ 0 ) = DISPLAYFORM7 for an appropriate c 7 > 0 .

According to Lemma 10, with probability greater than P y we then have DISPLAYFORM8 With the same condition on µ as in Lemma 11, combined with the uniformized bound on finite sample fluctuations, we have that at every point in this set DISPLAYFORM9 .

According to Lemma 12 the Riemannian gradient norm is bounded by M = √ n X ∞ .

Choosing r, b as in Lemma 11, we require η < for some chosenc > 0.

We then obtain DISPLAYFORM10 for a suitably chosen C 2 > 0.

The final bound on the rate is obtained by summing over the terms for the three regions as in the population case, and convergence is again to a distance of less than c 3 s from a local minimizer.

The probability of achieving this rate is obtained by taking a union bound over the probability of initialization in C ζ0 (given in Lemma 3) and the probabilities of the bounds on the gradient fluctuations holding (from Lemma 10 and FORMULA7 ).

Note that the fluctuation bound events imply by construction the event E ∞ = {1 ≤ X ∞ ≤ 4 log(np)} hence we can replace X ∞ in the conditions on η above by 4 log(np).

The conditions on η, µ can be satisfied by requiring η < c5θs n log np , µ < c6 √ ζ0 n 5/4 for suitably chosen c 5 , c 6 > 0.

The bound on the number of iterations can be simplified to the form in the theorem statement as in the population case.

We show below that negative curvature normal to stable manifolds of saddle points in strict saddle functions is a feature that is found not only in dictionary learning, and can be used to obtain efficient convergence rates for other nonconvex problems as well, by presenting an analysis of generalized phase retrieval that is along similar lines to the dictionary learning analysis.

We stress that this contribution is not novel since a more thorough analysis was carried out by BID7 .

The resulting rates are also suboptimal, and pertain only to the population objective.

Generalized phase retrieval is the problem of recovering a vector x ∈ C n given a set of magnitudes of projections y k = |x * a k | onto a known set of vectors a k ∈ C n .

It arises in numerous domains including microscopy (27), acoustics BID1 , and quantum mechanics (10) (see (33) for a review).

Clearly x can only be recovered up to a global phase.

We consider the setting where the elements of every a k are i. DISPLAYFORM0 We analyze the least squares formulation of the problem (7) given by DISPLAYFORM1 Taking the expectation (large p limit) of the above objective and organizing its derivatives using Wirtinger calculus FORMULA2 , we obtain DISPLAYFORM2 For the remainder of this section, we analyze this objective, leaving the consideration of finite sample size effects to future work.

In (37) it was shown that aside from the manifold of minimȃ DISPLAYFORM0 the only critical points of E[f ] are a maximum at z = 0 and a manifold of saddle points given by DISPLAYFORM1 where W ≡ {z|z * x = 0}. We decompose z as DISPLAYFORM2 where ζ > 0, w ∈ W .

This gives z 2 = w 2 + ζ 2 .

The choice of w, ζ, φ is unique up to factors of 2π in φ, as can be seen by taking an inner product with x. Since the gradient decomposes as follows: DISPLAYFORM3 the directions e iφ x x , w w are unaffected by gradient descent and thus the problem reduces to a two-dimensional one in the space (ζ, w ).

Note also that the objective for this twodimensional problem is a Morse function, despite the fact that in the original space there was a manifold of saddle points.

It is also clear from this decomposition of the gradient that the stable manifolds of the saddles are precisely the set W .It is evident from 24 that the dispersive property does not hold globally in this case.

For z / ∈ B ||x|| we see that gradient descent will cause ζ to decrease, implying positive curvature normal to the stable manifolds of the saddles.

This is a consequence of the global geometry of the objective.

Despite this, in the region of the space that is more "interesting", namely B ||x|| , we do observe the dispersive property, and can use it to obtain a convergence rate for gradient descent.

We define a set that contains the regions that feeds into small gradient regions around saddle points within B ||x|| by DISPLAYFORM4 We will show that, as in the case of orthogonal dictionary learning, we can both bound the probability of initializing in (a subset of) the complement of Q ζ0 and obtain a rate for convergence of gradient descent in the case of such an initialization.

plane.

The full red curves are the boundaries between the sets S 1 , S 2 , S 3 , S 4 used in the analysis.

The dashed red line is the boundary of the set Q ζ0 that contains small gradient regions around critical points that are not minima.

The maximizer and saddle point are shown in dark green, while the minimizer is in pink.

These are used to find the change in ζ, w at every iteration in each region: DISPLAYFORM5 We now show that gradient descent initialized in S 1 \Q ζ0 cannot exit ∪ 2 we are guaranteed from Lemma 13 that at every iteration ζ ≥ ζ 0 .

Thus the region with ζ < ζ 0 can only be entered if gradient descent is initialized in it.

It follows that initialization in S 1 \Q ζ0 rules out entering Q ζ0 at any future iteration of gradient descent.

Since this guarantees that regions that feed into small gradient regions are avoided, an efficient convergence rate can again be obtained.

Theorem 3 (Gradient descent convergence rate for generalized phase retrieval).

Gradient descent on 22 with step size η < DISPLAYFORM0 iterations with probability DISPLAYFORM1 ii) Since only a step from S 4 can decrease ζ, we have that for the initial point z 2 > x 2 .Combined with DISPLAYFORM2 this gives DISPLAYFORM3 and using the lower bound (1 − 2η x 2 c)ζ ≤ ζ we obtain DISPLAYFORM4 where in the last inequality we used c < DISPLAYFORM5 Proof of Lemma 14.

We use the fact that for the next iterate we have DISPLAYFORM6 We will also repeatedly use η < DISPLAYFORM7 which is a shown in Lemma 13.

DISPLAYFORM8 We want to show DISPLAYFORM9 (1 + c) x 2 .1) We have z ∈ S 3 ⇒ z 2 = (1 − ε) x 2 for some ε ≤ c and using 28 we must show DISPLAYFORM10 Proof of Theorem 3: (Gradient descent convergence rate for generalized phase retrieval).

We now bound the number of iterations that gradient descent, after random initialization in S 1 , requires to reach a point where one of the convergence criteria detailed in Lemma 15 is fulfilled.

From Lemma 14, we know that after initialization in S 1 we need to consider only the set DISPLAYFORM11 S i .

The number of iterations in each set will be determined by the bounds on the change in ζ, ||w|| detailed in 27.

Assuming we initialize with some ζ = ζ 0 .

Then the maximal number of iterations in this region is DISPLAYFORM0 since after this many iterations DISPLAYFORM1 The only concern is that after an iteration in S 3 ∪ S 4 the next iteration might be in S 2 .To account for this situation, we find the maximal number of iterations required to reach S 3 ∪ S 4 again.

This is obtained from the bound on ζ in Lemma 13.Using this result, and the fact that for every iteration in S 2 we are guaranteed ζ ≥ (1 + 2η x 2 c)ζ the number of iterations required to reach S 3 ∪ S 4 again is given by DISPLAYFORM2

The final rate to convergence is DISPLAYFORM0 C.9 Probability of the bound holdingThe bound applies to an initialization with ζ ≥ ζ 0 , hence in S 1 \Q ζ0 .

Assuming uniform initialization in S 1 , the set Q ζ0 is simply a band of width 2ζ 0 around the equator of the ball B x / √ 2 (in R 2n , using the natural identification of C n with R 2n ).

This volume can be calculated by integrating over 2n − 1 dimensional balls of varying radius.

DISPLAYFORM1 and by V (n) = π n/2 n 2 Γ( n 2 ) the hypersphere volume, the probability of initializing in S 1 ∩ Q ζ0 (and thus in a region that feeds into small gradient regions around saddle points) is DISPLAYFORM2 .

For small ζ we again find that P(fail) scales linearly with ζ, as was the case for the previous problems considered.

Proof of Lemma 3: (Volume of C ζ ).

We are interested in the relative volume DISPLAYFORM0 Vol(S n−1 ) ≡ V ζ .

Using the standard solid angle formula, it is given by DISPLAYFORM1 This integral admits no closed form solution but one can construct a linear approximation around small ζ and show that it is convex.

Thus the approximation provides a lower bound for V ζ and an upper bound on the failure probability.

From symmetry considerations the zero-order term is V 0 = 1 2n .

The first-order term is given by DISPLAYFORM2 We now require an upper bound for the second integral since we are interested in a lower bound for V ζ .

We can express it in terms of the second moment of the L ∞ norm of a Gaussian vector as follows: where µ(X) is the Gaussian measure on the vector X ∈ R n .

We can bound the first term using Combining these bounds, the leading order behavior of the gradient is DISPLAYFORM3 This linear approximation is indeed a lower bound, since using integration by parts twice we have (0) and this is the smallest L ∞ ball containing C ζ .Proof.

Given the surface of some L ∞ ball for w , we can ask what is the minimal ζ such that ∂C ζm intersects this surface.

This amounts to finding the minimal q n given some w ∞ .

Yet this is clearly obtained by setting all the coordinates of w to be equal to w ∞ (this is possible since we are guaranteed q n ≥ w ∞ ⇒ w ∞ ≤ where one instead maximizes q n with some fixed w ∞ .Given some surface of an L 2 ball, we can ask what is the minimal C ζ such that C ζ ⊆ B 2 r (0).

This is equivalent to finding the maximal ζ M such that ∂C ζ M intersects the surface of the L 2 ball.

Since q n is fixed, maximizing ζ is equivalent to minimizing w ∞ .

This is done by setting w ∞ = w √ n−1, which gives DISPLAYFORM4 The statement in the lemma follows from combining these results. .

If we now combine this with the fact that after a Riemannian gradient step cos(gη)q i − sin(gη) ≤ q i ≤ cos(gη)q i + sin(gη), the above condition on η implies the inequality ( * ), which in turn ensures that |w i | < r ⇒ |w i | < w ∞ : |w i | < |w i | + sin(gη) < r + gη < ( * )(1 − g 2 η 2 )b − gη < cos(gη) w ∞ − sin(gη) ≤ w ∞ Due to the above analysis, it is evident that any w i such that |w i | = w ∞ obeys |w i | > r, from which it follows that we can use 31 to obtain q n w ∞ − 1 = ζ ≥ ζ 1 + √ n 2 ηc(w)

@highlight

We provide an efficient convergence rate for gradient descent on the complete orthogonal dictionary learning objective based on a geometric analysis.
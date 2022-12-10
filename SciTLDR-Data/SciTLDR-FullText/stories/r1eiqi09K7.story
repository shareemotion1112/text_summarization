Several first order stochastic optimization methods commonly used in the Euclidean domain such as stochastic gradient descent (SGD), accelerated gradient descent or variance reduced methods have already been adapted to certain Riemannian settings.

However, some of the most popular of these optimization tools - namely Adam, Adagrad and the more recent Amsgrad - remain to be generalized to Riemannian manifolds.

We discuss the difficulty of generalizing such adaptive schemes to the most agnostic Riemannian setting, and then provide algorithms and convergence proofs for geodesically convex objectives in the particular case of a product of Riemannian manifolds, in which adaptivity is implemented across manifolds in the cartesian product.

Our generalization is tight in the sense that choosing the Euclidean space as Riemannian manifold yields the same algorithms and regret bounds as those that were already known for the standard algorithms.

Experimentally, we show faster convergence and to a lower train loss value for Riemannian adaptive methods over their corresponding baselines on the realistic task of embedding the WordNet taxonomy in the Poincare ball.

Developing powerful stochastic gradient-based optimization algorithms is of major importance for a variety of application domains.

In particular, for computational efficiency, it is common to opt for a first order method, when the number of parameters to be optimized is great enough.

Such cases have recently become ubiquitous in engineering and computational sciences, from the optimization of deep neural networks to learning embeddings over large vocabularies.

This new need resulted in the development of empirically very successful first order methods such as ADAGRAD BID5 , ADADELTA BID29 , ADAM BID9 or its recent update AMSGRAD BID18 .Note that these algorithms are designed to optimize parameters living in a Euclidean space R n , which has often been considered as the default geometry to be used for continuous variables.

However, a recent line of work has been concerned with the optimization of parameters lying on a Riemannian manifold, a more general setting allowing non-Euclidean geometries.

This family of algorithms has already found numerous applications, including for instance solving Lyapunov equations BID27 , matrix factorization BID23 , geometric programming BID22 , dictionary learning BID2 or hyperbolic taxonomy embedding BID15 BID6 BID4 BID14 .A few first order stochastic methods have already been generalized to this setting (see section 6), the seminal one being Riemannian stochastic gradient descent (RSGD) BID1 , along with new methods for their convergence analysis in the geodesically convex case .

However, the above mentioned empirically successful adaptive methods, together with their convergence analysis, remain to find their respective Riemannian counterparts.

Indeed, the adaptivity of these algorithms can be thought of as assigning one learning rate per coordinate of the parameter vector.

However, on a Riemannian manifold, one is generally not given an intrinsic coordinate system, rendering meaningless the notions sparsity or coordinate-wise update.

Our contributions.

In this work we (i) explain why generalizing these adaptive schemes to the most agnostic Riemannian setting in an intrinsic manner is compromised, and (ii) propose generalizations of the algorithms together with their convergence analysis in the particular case of a product of manifolds where each manifold represents one "coordinate" of the adaptive scheme.

Finally, we (iii) empirically support our claims on the realistic task of hyperbolic taxonomy embedding.

Our initial motivation.

The particular application that motivated us in developing Riemannian versions of ADAGRAD and ADAM was the learning of symbolic embeddings in non-Euclidean spaces.

As an example, the GloVe algorithm BID17 ) − an unsupervised method for learning Euclidean word embeddings capturing semantic/syntactic relationships − benefits significantly from optimizing with ADAGRAD compared to using SGD, presumably because different words are sampled at different frequencies.

Hence the absence of Riemannian adaptive algorithms could constitute a significant obstacle to the development of competitive optimization-based Riemannian embedding methods.

In particular, we believe that the recent rise of embedding methods in hyperbolic spaces could benefit from such developments BID15 BID6 b; BID4 BID28 .

We recall here some elementary notions of differential geometry.

For more in-depth expositions, we refer the interested reader to BID21 and BID19 .Manifold, tangent space, Riemannian metric.

A manifold M of dimension n is a space that can locally be approximated by a Euclidean space R n , and which can be understood as a generalization to higher dimensions of the notion of surface.

For instance, the sphere S := {x ∈ R n | x 2 = 1} embedded in R n is an (n − 1)-dimensional manifold.

In particular, R n is a very simple n-dimensional manifold, with zero curvature.

At each point x ∈ M, one can define the tangent space T x M, which is an n-dimensional vector space and can be seen as a first order local approximation of M around x. A Riemannian metric ρ is a collection ρ := (ρ x ) x∈M of inner-products ρ x (·, ·) : T x M × T x M → R on T x M, varying smoothly with x. It defines the geometry locally on M. For x ∈ M and u ∈ T x M, we also write u x := ρ x (u, u).

A Riemannian manifold is a pair (M, ρ).Induced distance function, geodesics.

Notice how a choice of a Riemannian metric ρ induces a natural global distance function on M. Indeed, for x, y ∈ M, we can set d(x, y) to be equal to the infimum of the lengths of smooth paths between x and y in M, where the length (c) of a path c is given by integrating the size of its speed vectorċ(t) ∈ T c(t) M, in the corresponding tangent space:(c) :=

Consider performing an SGD update of the form DISPLAYFORM0 where g t denotes the gradient of objective f t 1 and α > 0 is the step-size.

In a Riemannian manifold (M, ρ), for smooth f : M → R, BID1 defines Riemannian SGD by the following update: DISPLAYFORM1 where g t ∈ T xt M denotes the Riemannian gradient of f t at x t .

Note that when (M, ρ) is the Euclidean space (R n , I n ), these two match, since we then have exp x (v) = x + v.

Intuitively, applying the exponential map enables to perform an update along the shortest path in the relevant direction in unit time, while remaining in the manifold.

In practice, when exp x (v) is not known in closed-form, it is common to replace it by a retraction map R x (v), most often chosen as R x (v) = x + v, which is a first-order approximation of exp x (v).

Let's recall here the main algorithms that we are taking interest in.

ADAGRAD.

Introduced by BID5 , the standard form of its update step is defined as DISPLAYFORM0 Such updates rescaled coordinate-wise depending on the size of past gradients can yield huge improvements when gradients are sparse, or in deep networks where the size of a good update may depend on the layer.

However, the accumulation of all past gradients can also slow down learning.

ADAM.

Proposed by BID9 , the ADAM update rule is given by DISPLAYFORM1 where m t = β 1 m t−1 + (1−β 1 )g t can be seen as a momentum term and DISPLAYFORM2 2 is an adaptivity term.

When β 1 = 0, one essentially recovers the unpublished method RMSPROP BID24 , the only difference to ADAGRAD being that the sum is replaced by an exponential moving average, hence past gradients are forgotten over time in the adaptivity term v t .

This circumvents the issue of ADAGRAD that learning could stop too early when the sum of accumulated squared gradients is too significant.

Let us also mention that the momentum term introduced by ADAM for β 1 = 0 has been observed to often yield huge empirical improvements.

AMSGRAD.

More recently, BID18 identified a mistake in the convergence proof of ADAM.

To fix it, they proposed to either modify the ADAM algorithm with DISPLAYFORM3 which they coin AMSGRAD, or to choose an increasing schedule for β 2 , making it time dependent, which they call ADAMNC (for non-constant).

Intrinsic updates.

It is easily understandable that writing any coordinate-wise update requires the choice of a coordinate system.

However, on a Riemannian manifold (M, ρ), one is generally not 1 to be interpreted as the objective with the same parameters, evaluated at the minibatch taken at time t. 2 a small ε = 10 −8 is often added in the square-root for numerical stability, omitted here for simplicity.

3 with mt and vt defined by the same equations as in ADAM (see above paragraph).provided with a canonical coordinate system.

The formalism only allows to work with certain local coordinate systems, also called charts, and several different charts can be defined around each point x ∈ M. One usually says that a quantity defined using a chart is intrinsic to M if its definition does not depend on which chart was used.

For instance, it is known that the Riemannian gradient gradf of a smooth function f : M → R can be defined intrinsically to (M, ρ), but its Hessian is only intrinsically defined at critical points 4 .

It is easily seen that the RSGD update of Eq. FORMULA1 is intrinsic, since it only involves exp and grad, which are objects intrinsic to (M, ρ).

However, it is unclear whether it is possible at all to express either of Eqs. (3,4,5) in a coordinate-free or intrinsic manner.

A tempting solution.

Note that since an update is defined in a tangent space, one could be tempted to fix a canonical coordinate system e := (e (1) , ..., e (n) ) in the tangent space T x0 M R d at the initialization x 0 ∈ M, and parallel-transport e along the optimization trajectory, adapting Eq. FORMULA2 to: DISPLAYFORM0 where and (·) 2 denote coordinate-wise division and square respectively, these operations being taken relatively to coordinate system e t .

In the Euclidean space, parallel transport between two points x and y does not depend on the path it is taken along because the space has no curvature.

However, in a general Riemannian manifold, not only does it depend on the chosen path but curvature will also give to parallel transport a rotational component 5 , which will almost surely break the sparsity of the gradients and hence the benefit of adaptivity.

Besides, the interpretation of adaptivity as optimizing different features (i.e. gradient coordinates) at different speeds is also completely lost here, since the coordinate system used to represent gradients depends on the optimization path.

Finally, note that the techniques we used to prove our theorems would not apply to updates defined in the vein of Eq. (6).

From now on, we assume additional structure on (M, ρ), namely that it is the cartesian product of n Riemannian manifolds (M i , ρ i ), where ρ is the induced product metric: DISPLAYFORM0 Product notations.

The induced distance function d on M is known to be given by DISPLAYFORM1 Similarly, the exponential, log map and the parallel transport in M are the concatenations of those in each M i .Riemannian ADAGRAD.

We just saw in the above discussion that designing meaningful adaptive schemes − intuitively corresponding to one learning rate per coordinate − in a general Riemannian manifold was difficult, because of the absence of intrinsic coordinates.

Here, we propose to see each component x i ∈ M i of x as a "coordinate", yielding a simple adaptation of Eq. (3) as DISPLAYFORM2 On the adaptivity term.

Note that we take (squared) Riemannian norms g DISPLAYFORM3 in the adaptivity term rescaling the gradient.

In the Euclidean setting, this quantity is simply a scalar (g DISPLAYFORM4

In section 2, we briefly presented ADAGRAD, ADAM and AMSGRAD.

Intuitively, ADAM can be described as a combination of ADAGRAD with a momentum (of parameter β 1 ), with the slight modification that the sum of the past squared-gradients is replaced with an exponential moving average, for an exponent β 2 .

Let's also recall that AMSGRAD implements a slight modification of ADAM, allowing to correct its convergence proof.

Finally, ADAMNC is simply ADAM, but with a particular non-constant schedule for β 1 and β 2 .

On the other hand, what is interesting to note is that the schedule initially proposed by BID18 for β 2 in ADAMNC, namely β 2t := 1 − 1/t, lets v t recover the sum of squared-gradients of ADAGRAD.

Hence, ADAMNC without momentum (i.e. β 1t = 0) yields ADAGRAD.Assumptions and notations.

For 1 ≤ i ≤ n, we assume (M i , ρ i ) is a geodesically complete Riemannian manifold with sectional curvature lower bounded by κ i ≤ 0.

As written in Eq. FORMULA7 , let (M, ρ) be the product manifold of the (M i , ρ i )'s.

For each i, let X i ⊂ M i be a compact, geodesically convex set and define X := X 1 × · · · × X n , the set of feasible parameters.

Define Π Xi : M i → X i to be the projection operator, i.e. Π Xi (x) is the unique y ∈ X i minimizing d i (y, x).

Denote by P i , exp i and log i the parallel transport, exponential and log maps in DISPLAYFORM0 and by g i ∈ T x i M i the corresponding components of x and g. In the sequel, let (f t ) be a family of differentiable, geodesically convex functions from M to R. Assume that each X i ⊂ M i has a diameter bounded by D ∞ and that for all 1 ≤ i ≤ n, t ∈ [T ] and x ∈ X , (gradf t (x)) i xi ≤ G ∞ .

Finally, our convergence guarantees will bound the regret, defined at the end of T rounds as DISPLAYFORM1 Following the discussion in section 3.2 and especially Eq. (8), we present Riemannian AMSGRAD in FIG1 .

For comparison, we show next to it the standard AMSGRAD algorithm in FIG1 .Require: DISPLAYFORM2 Require: DISPLAYFORM3 DISPLAYFORM4 From these algorithms, RADAM and ADAM are obtained simply by removing the max operations, i.e. replacingv i t = max{v DISPLAYFORM5 The convergence guarantee that we obtain for RAMSGRAD is presented in Theorem 1, where the quantity ζ is defined by as DISPLAYFORM6 For comparison, we also show the convergence guarantee of the original AMSGRAD in appendix C. Note that when (M i , ρ i ) = R for all i, convergence guarantees between RAMSGRAD and AMSGRAD coincide as well.

Indeed, the curvature dependent quantity (ζ(κ i , D ∞ ) + 1)/2 in the Riemannian case then becomes equal to 1, recovering the convergence theorem of AMSGRAD.

It is also interesting to understand at which speed does the regret bound worsen when the curvature is small but non-zero: by a multiplicative factor of approximately 1 + D ∞ |κ|/6 (see Eq. FORMULA18 ).

Similar remarks hold for RADAMNC, whose convergence guarantee is shown in Theorem 2.

Finally, notice that β 1 := 0 in Theorem 2 yields a convergence proof for RADAGRAD, whose update rule we defined in Eq. (8).Theorem 1 (Convergence of RAMSGRAD).

Let (x t ) and (v t ) be the sequences obtained from Algorithm 1a, α t = α/ √ t, β 1 = β 11 , β 1t ≤ β 1 for all t ∈ [T ] and γ = β 1 / √ β 2 < 1.

We then have: DISPLAYFORM7 Proof.

See appendix A.Theorem 2 (Convergence of RADAMNC).

Let (x t ) and (v t ) be the sequences obtained from RADAMNC, α t = α/ √ t, β 1 = β 11 , β 1t = β 1 λ t−1 , λ < 1, β 2t = 1 − 1/t.

We then have: DISPLAYFORM8 Proof.

See appendix B.The role of convexity.

Note how the notion of convexity in Theorem 5 got replaced by the notion of geodesic convexity in Theorem 1.

Let us compare the two definitions: the differentiable functions f : R n → R and g : M → R are respectively convex and geodesically convex if for all x, y ∈ R n , u, v ∈ M: DISPLAYFORM9 But how does this come at play in the proofs?

Regret bounds for convex objectives are usually obtained by bounding T t=1 f t (x t ) − f t (x * ) using Eq. FORMULA0 for any x * ∈ X , which boils down to bounding each g t , x t − x * .

In the Riemannian case, this term becomes ρ xt (g t , − log xt (x * )).The role of the cosine law.

How does one obtain a bound on g t , x t − x * ?

For simplicity, let us look at the particular case of an SGD update, from Eq. (1).

Using a cosine law, this yields DISPLAYFORM10 One now has two terms to bound: (i) when summing over t, the first one simplifies as a telescopic summation; (ii) the second term T t=1 α t g t 2 will require a well chosen decreasing schedule for α.

In Riemannian manifolds, this step is generalized using the analogue lemma 6 introduced by , valid in all Alexandrov spaces, which includes our setting of geodesically convex subsets of Riemannian manifolds with lower bounded sectional curvature.

The curvature dependent quantity ζ of Eq. (10) appears from this lemma, letting us bound ρ DISPLAYFORM11 The benefit of adaptivity.

Let us also mention that the above bounds significantly improve for sparse (per-manifold) gradients.

In practice, this could happen for instance for algorithms embedding each word i (or node of a graph) in a manifold M i and when just a few words are updated at a time.

On the choice of ϕ i .

The fact that our convergence theorems (see lemma 3) do not require specifying ϕ i suggests that the regret bounds could be improved by exploiting momentum/acceleration in the proofs for a particular ϕ i .

Note that this remark also applies to AMSGRAD BID18 .

We empirically assess the quality of the proposed algorithms: RADAM, RAMSGRAD and RADAGRAD compared to the non-adaptive RSGD method (Eq. 2).

For this, we follow BID15 and embed the transitive closure of the WordNet noun hierarchy BID12 in the n-dimensional Poincaré model D n of hyperbolic geometry which is well-known to be better suited to embed tree-like graphs than the Euclidean space BID8 BID4 .

In this case, each word is embedded in the same space of constant curvature −1, thus M i = D n , ∀i.

Note that it would also be interesting to explore the benefit of our optimization tools for algorithms proposed in BID14 BID4 BID6 .

The choice of the Poincaré model is justified by the access to closed form expressions for all the quantities used in Alg.

1a: DISPLAYFORM0 n , where λ x = 2 1− x 2 is the conformal factor.• Riemannian gradients are rescaled Euclidean gradients: DISPLAYFORM1 • Distance function and geodesics, BID15 BID26 BID7 ).•

Exponential and logarithmic maps: DISPLAYFORM2 , where ⊕ is the generalized Mobius addition BID26 BID7 ).•

Parallel transport along the unique geodesic from x to y: P x→y (v) = λx λy · gyr[y, −x]v.

This formula was derived from BID26 BID7 , gyr being given in closed form in (Ungar, 2008, Eq. (1.27) ).Dataset & Model.

The transitive closure of the WordNet taxonomy graph consists of 82,115 nouns and 743,241 hypernymy Is-A relations (directed edges E).

These words are embedded in D n such that the distance between words connected by an edge is minimized, while being maximized otherwise.

We minimize the same loss function as BID15 which is similar with log-likelihood, but approximating the partition function using sampling of negative word pairs (non-edges), fixed to 10 in our case.

Note that this loss does not use the direction of the edges in the graph DISPLAYFORM3 Metrics.

We report both the loss value and the mean average precision (MAP) BID15 : for each directed edge (u, v), we rank its distance d(u, v) among the full set of ground truth negative examples {d(u , v)|(u , v) / ∈ E}. We use the same two settings as BID15 , namely: reconstruction (measuring representation capacity) and link prediction (measuring generalization).

For link prediction we sample a validation set of 2% edges from the set of transitive closure edges that contain no leaf node or root.

We only focused on 5-dimensional hyperbolic spaces.

Training details.

For all methods we use the same "burn-in phase" described in BID15 for 20 epochs, with a fixed learning rate of 0.03 and using RSGD with retraction as explained in Sec. 2.2.

Solely during this phase, we sampled negative words based on their graph degree raised at power 0.75.

This strategy improves all metrics.

After that, when different optimization methods start, we sample negatives uniformly.

We use n = 5, following BID15 .Optimization methods.

Experimentally we obtained slightly better results for RADAM over RAMS-GRAD, so we will mostly report the former.

Moreover, we unexpectedly observed convergence to lower loss values when replacing the true exponential map with its first order approximation − i.e. the retraction R x (v) = x + v − in both RSGD and in our adaptive methods from Alg.

1a.

One possible explanation is that retraction methods need fewer steps and smaller gradients to "escape" points sub-optimally collapsed on the ball border of D n compared to fully Riemannian methods.

As a consequence, we report "retraction"-based methods in a separate setting as they are not directly comparable to their fully Riemannian analogues.

Results.

We show in FIG2 results for "exponential" based and "retraction" based methods.

We ran all our methods with different learning rates from the set {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0}. For the RSGD baseline we show in orange the best learning rate setting, but we also show the previous lower (slower convergence, in blue) and the next higher (faster overfitting, in green) learning rates.

For RADAM and RAMSGRAD we only show the best settings.

We always use β 1 = 0.9 and β 2 = 0.999 for these methods as these achieved the lowest training loss.

RADAGRAD was consistently worse, so we do not report it.

As can be seen, RADAM always achieves the lowest training loss.

On the MAP metric for both reconstruction and link prediction settings, the same method also outperforms all the other methods for the full Riemannian setting (i.e. Tab.

2).

Interestingly, in the "retraction" setting, RADAM reaches the lowest training loss value and is on par with RSGD on the MAP evaluation for both reconstruction and link prediction settings.

However, RAMSGRAD is faster to converge in terms of MAP for the link prediction task, suggesting that this method has a better generalization capability.

After Riemannian SGD was introduced by BID1 , a pletora of other first order Riemannian methods arose, such as Riemannian SVRG , Riemannian Stein variational gradient descent BID10 , Riemannian accelerated gradient descent BID31 or averaged RSGD BID25 , along with new methods for their convergence analysis in the geodesically convex case .

Stochastic gradient Langevin dynamics was generalized as well, to improve optimization on the probability simplex BID16 .Let us also mention that BID20 proposed Riemannian counterparts of SGD with momentum and RMSprop, suggesting to transport the momentum term using parallel translation, which is an idea that we preserved.

However (i) no convergence guarantee is provided and (ii) their algorithm performs the coordinate-wise adaptive operations (squaring and division) w.r.t.

a coordinate system in the tangent space, which, as we discussed in section 3.1, compromises the possibility of obtaining convergence guarantees.

Finally, another version of Riemannian ADAM for the Grassmann manifold G(1, n) was previously introduced by BID3 , also transporting the momentum term using parallel translation.

However, their algorithm completely removes the adaptive component, since the adaptivity term v t becomes a scalar.

No adaptivity across manifolds is discussed, which is the main point of our discussion.

Moreover, no convergence analysis is provided either.

Driven by recent work in learning non-Euclidean embeddings for symbolic data, we propose to generalize popular adaptive optimization tools (e.g. ADAM, AMSGRAD, ADAGRAD) to Cartesian products of Riemannian manifolds in a principled and intrinsic manner.

We derive convergence rates that are similar to the Euclidean corresponding models.

Experimentally we show that our methods outperform popular non-adaptive methods such as RSGD on the realistic task of hyperbolic word taxonomy embedding.

DISPLAYFORM0 i * .

Combining the following formula 8 : DISPLAYFORM1 with the following inequality (given by lemma 6): DISPLAYFORM2 yields DISPLAYFORM3 where the use the notation ·, · x i for ρ DISPLAYFORM4 Now applying Cauchy-Schwarz' and Young's inequalities to the last term yields DISPLAYFORM5 From the geodesic convexity of f t for 1 ≤ t ≤ T , we have DISPLAYFORM6 Let's look at the first term.

Using β 1t ≤ β 1 and with a change of indices, we have DISPLAYFORM7 where the last equality comes from a standard telescopic summation.

We now need the following lemma.

Lemma 3.

DISPLAYFORM8 Proof.

Let's start by separating the last term, and removing the hat on v. Using that β 1k ≤ β 1 for all k ∈ [T ], (1 − β 1j )β DISPLAYFORM9 Finally, (1 − β 1j ) ≤ 1 and

The following lemma is a user-friendly inequality developed by in order to prove convergence of gradient-based optimization algorithms, for geodesically convex functions, in Alexandrov spaces.

Lemma 6 (Cosine inequality in Alexandrov spaces).

If a, b, c, are the sides (i.e., side lengths) of a geodesic triangle in an Alexandrov space with curvature lower bounded by κ, and A is the angle between sides b and c, then DISPLAYFORM0 Proof.

See section 3.1, lemma 6 of .Lemma 7 (An analogue of Cauchy-Schwarz).

For all p, k ∈ N * , u 1 , ..., u k ∈ R p , a 1 , ..., a k ∈ R + , we have Proof.

The proof consists in applying Cauchy-Schwarz' inequality two times: DISPLAYFORM1 DISPLAYFORM2 Finally, this last lemma is used by BID18 in their convergence proof for ADAMNC.

We need it too, in an analogue lemma.

Lemma 8 ( BID0 ).

For any non-negative real numbers y 1 , ..., y t , the following holds: DISPLAYFORM3

@highlight

Adapting Adam, Amsgrad, Adagrad to Riemannian manifolds. 
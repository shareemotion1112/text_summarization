This paper concerns dictionary learning, i.e., sparse coding, a fundamental representation learning problem.

We show that a subgradient descent algorithm, with random initialization, can recover orthogonal dictionaries on a natural nonsmooth, nonconvex L1 minimization formulation of the problem, under mild statistical assumption on the data.

This is in contrast to previous provable methods that require either expensive computation or delicate initialization schemes.

Our analysis develops several tools for characterizing landscapes of nonsmooth functions, which might be of independent interest for provable training of deep networks with nonsmooth activations (e.g., ReLU), among other applications.

Preliminary synthetic and real experiments corroborate our analysis and show that our algorithm works well empirically in recovering orthogonal dictionaries.

Dictionary learning (DL), i.e. , sparse coding, concerns the problem of learning compact representations, i.e., given data Y , one tries to find a representation basis A and coefficients X, so that Y ≈ AX where X is most sparse.

DL has numerous applications especially in image processing and computer vision (Mairal et al., 2014) .

When posed in analytical form, DL seeks a transformation Q such that QY is sparse; in this sense DL can be considered as an (extremely!) primitive "deep" network (Ravishankar & Bresler, 2013) .Many heuristic algorithms have been proposed to solve DL since the seminal work of Olshausen & Field (1996) , most of them surprisingly effective in practice (Mairal et al., 2014; Sun et al., 2015) .

However, understandings on when and how DL is solvable have only recently started to emerge.

Under appropriate generating models on A and X, Spielman et al. (2012) showed that complete (i.e., square, invertible) A can be recovered from Y , provided that X is ultra-sparse.

Subsequent works BID0 BID1 Chatterji & Bartlett, 2017; BID4 provided similar guarantees for overcomplete (i.e. fat) A, again in the ultra-sparse regime.

The latter methods are invariably based on nonconvex optimization with model-dependent initialization, rendering their practicality on real data questionable.

The ensuing developments have focused on breaking the sparsity barrier and addressing the practicality issue.

Convex relaxations based on the sum-of-squares (SOS) SDP hierarchy can recover overcomplete A when X has linear sparsity BID6 Ma et al., 2016; Schramm & Steurer, 2017) , while incurring expensive computation (solving large-scale SDP's or large-scale tensor decomposition).

By contrast, Sun et al. (2015) showed that complete A can be recovered in the linear sparsity regime by solving a certain nonconvex problem with arbitrary initialization.

However, the second-order optimization method proposed there is still expensive.

This problem is partially addressed by (Gilboa et al., 2018) which proved that the first-order gradient descent with random initialization enjoys a similar performance guarantee.

A standing barrier toward practicality is dealing with nonsmooth functions.

To promote sparsity in the coefficients, the 1 norm is the function of choice in practical DL, as is common in modern signal processing and machine learning BID10 : despite its nonsmoothness, this choice often admits highly scalable numerical methods, such as proximal gradient method and alternating directionThe reader is welcome to refer to our arXiv version for future updates.method (Mairal et al., 2014) .

The analyses in Sun et al. (2015) ; Gilboa et al. (2018) , however, focused on characterizing the algorithm-independent function landscape of a certain nonconvex formulation of DL, which takes a smooth surrogate to 1 to get around the nonsmoothness.

The tactic smoothing there introduced substantial analysis difficulty, and broke the practical advantage of computing with the simple 1 function.

In this paper, we show that working directly with a natural 1 norm formulation results in neat analysis and a practical algorithm.

We focus on the problem of learning orthogonal dictionaries: given data {y i } i∈ [m] generated as y i = Ax i , where A ∈ R n×n is a fixed unknown orthogonal matrix and each x i ∈ R n is an iid Bernoulli-Gaussian random vector with parameter θ ∈ (0, 1), recover A. This statistical model is the same as in previous works (Spielman et al., 2012; Sun et al., 2015) .Write Y .

= [y 1 , . . . , y m ] and similarly X .

= [x 1 , . . . , x m ].

We propose to recover A by solving the following nonconvex (due to the constraint), nonsmooth (due to the objective) optimization problem: DISPLAYFORM0 |q y i | subject to q 2 = 1.(1.1)Based on the statistical model, q Y = q AX has the highest sparsity when q is a column of A (up to sign) so that q A is 1-sparse.

Spielman et al. (2012) formalized this intuition and optimized the same objective as Eq. (1.1) with a q ∞ = 1 constraint, which only works when θ ∼ O(1/ √ n).

Sun et al. (2015) worked with the sphere constraint but replaced the 1 objective with a smooth surrogate, introducing substantial analytical and computational deficiencies as alluded above.

In constrast, we show that with sufficiently many samples, the optimization landscape of formulation (1.1) is benign with high probability (over the randomness of X), and a simple Riemannian subgradient descent algorithm can provably recover A in polynomial time.

Theorem 1.1 (Main result, informal version of Theorem 3.1).

Assume θ ∈ [1/n, 1/2].

For m ≥ Ω(θ −2 n 4 log 4 n), the following holds with high probability: there exists a poly(m, −1 )-time algorithm, which runs Riemannian subgradient descent on formulation (1.1) from at most O(n log n) independent, uniformly random initial points, and outputs a set of vectors { a 1 , . . .

, a n } such that up to permutation and sign change, a i − a i 2 ≤ for all i ∈ [n].In words, our algorithm works also in the linear sparsity regime, the same as established in Sun et al. (2015) ; Gilboa et al. (2018) , at a lower sample complexity O(n 4 ) in contrast to the existing O(n 5.5 ) in Sun et al. (2015) .

1 As for the landscape, we show that (Theorems 3.4 and 3.6) each of the desired solutions {±a i } i∈ [n] is a local minimizer of formulation (1.1) with a sufficiently large basin of attraction so that a random initialization will land into one of the basins with at least constant probability.

To obtain the result, we integrate and develop elements from nonsmooth analysis (on Riemannian manifolds), set-valued analysis, and random set theory, which might be valuable to studying other nonconvex, nonsmooth optimization problems.

Dictionary learning Besides the many results sampled above, we highlight similarities of our result to Gilboa et al. (2018) .

Both propose first-order optimization methods with random initialization, and several quantities we work with in the proofs are the same.

A defining difference is we work with the nonsmooth 1 objective directly, while Gilboa et al. (2018) built on the smoothed objective from Sun et al. (2015) .

We put considerable emphasis on practicality: the subgradient of the nonsmooth objective is considerably cheaper to evaluate than that of the smooth objective in Sun et al. (2015) , and in the algorithm we use Euclidean projection rather than exponential mapping to remain feasible-again, the former is much lighter for computation.

General nonsmooth analysis While nonsmooth analytic tools such as subdifferential for convex functions are now well received in machine learning and relevant communities, that for general functions are much less so.

The Clarke subdifferential and relevant calculus developed for the family of locally Lipschitz functions seem to be particularly relevant, and cover several families of functions of interest, such as convex functions, differentiable functions, and many forms of composition (Clarke, 1990; BID3 BID5 .

Remarkably, majority of the tools and results can be generalized to locally Lipschitz functions on Riemannnian manifolds (Ledyaev & Zhu, 2007; Hosseini & Pouryayevali, 2011) .

Our formulation (1.1) is exactly optimization of a locally Lipschitz function (as it is convex) on a Riemannian manifold (the sphere).

For simplicity, we try to avoid the full manifold language, nonetheless.

Nonsmooth optimization on Riemannian manifolds or with constraints Equally remarkable is many of the smooth optimization techniques and convergence results can be naturally adapted to optimization of locally Lipschitz functions on Riemannian manifolds (Grohs & Hosseini, 2015; Hosseini, 2015; Hosseini & Uschmajew, 2017; Grohs & Hosseini, 2016) .

New optimization methods such as gradient sampling and variants have been invented to solve general nonsmooth problems BID8 BID5 Curtis & Que, 2015; Curtis et al., 2017) .

Almost all available convergence results pertain to only global convergence, which is too weak for our purpose.

Our specific convergence analysis gives us a local convergence result (Theorem 3.8).Nonsmooth landscape characterization Nonsmoothness is not a big optimization barrier if the problem is convex; here we review some recent work on analyzing nonconvex nonsmooth problems.

Loh & Wainwright (2015) study the regularized empirical risk minimization problem with nonsmooth regularizers and show results of the type "all stationary points are within statistical error of ground truth" under certain restricted strong convexity of the smooth risk.

Duchi & Ruan (2017) ; Davis et al. (2017) study the phase retrieval problem with 1 loss, characterizing its nonconvex nonsmooth landscape and providing efficient algorithms.

There is a recent surge of work on analyzing one-hidden-layer ReLU networks, which are nonconvex and nonsmooth.

Algorithm-independent characterizations of the landscape are mostly local and require strong initialization procedures (Zhong et al., 2017) , whereas stronger global results can be established via designing new loss functions (Ge et al., 2017) , relating to PDEs (Mei et al., 2018) , or problem-dependent analysis of the SGD (Li & Yuan, 2017; Li & Liang, 2018) .

Our result provides an algorithm-independent chacaterization of the landscape of non-smooth dictionary learning, and is "almost global" in the sense that the initialization condition is satisifed by random initialization with high probability.

Other nonsmooth problems in application Prevalence of nonsmooth problems in optimal control and economics is evident from all monographs on nonsmooth analysis (Clarke, 1990; BID3 BID5 .

In modern machine learning and data analysis, nonsmooth functions are often taken to encode structural information (e.g., sparsity, low-rankness, quantization), or whenever robust estimation is desired.

In deep learning, the optimization problem is nonsmooth when nonsmooth activations are in use, e.g., the popular ReLU.

The technical ideas around nonsmooth analysis, set-valued analysis, and random set theory that we gather and develop here are particularly relevant to these applications.

Problem setup Given an unknown orthogonal dictionary A = [a 1 , . . .

, a n ] ∈ R n×n , we wish to recover A through m observations of the form The coefficient vectors x i are sampled from the Bernoulli-Gaussian distribution with parameter θ ∈ (0, 1), denoted as BG(θ): each entry x ij is independently drawn from a standard Gaussian with probability θ and zero otherwise.

The Bernoulli-Gaussian is a good prototype distribution for sparse vectors, as x i will be on average θ-sparse.

For any z ∼ iid Ber(θ), we let Ω denote the set of non-zero indices, which is a random set itself.

DISPLAYFORM0 We assume that n ≥ 3 and θ ∈ [1/n, 1/2].

In particular, θ ≥ 1/n is to require that each x i has at least one non-zero entry on average.

First-order geometry We will focus on the first-order geometry of the non-smooth objective Eq. (1.1): DISPLAYFORM1 In the whole Euclidean space R n , f is convex with sub-differential set DISPLAYFORM2 where sign(·) is the set-valued sign function (i.e. sign(0) = [−1, 1]).

As we minimize f subject to the constraint q 2 = 1, our problem is no longer convex.

The Riemannian sub-differential of f on S n−1 is defined as (Hosseini & Uschmajew, 2017) : DISPLAYFORM3 A point q is stationary for problem Eq. (1.1) if 0 ∈ ∂ R f (q).

We will not distinguish between local maxima and saddle points-we call a stationary point q a saddle point if there is a descent direction (i.e. direction along which the function is locally maximized at q).Set-valued analysis As the subdifferential is a set-valued mapping, analyzing it requires some setvalued analysis, which we briefly present here.

The addition of two sets is defined as the Minkowski summation: X + Y = {x + y : x ∈ X, y ∈ Y }.

The expectation of random sets is a straightforward extension of the Minkowski sum allowing any measurable "selection" procedure; for the concrete definition see (Molchanov, 2013) .

The Hausdorff distance between two sets is defined as DISPLAYFORM4 Basic properties about the Hausdorff distance are provided in Appendix A.1.Notations Bold small letters (e.g., x) are vectors and bold capitals are matrices (e.g., X).

The dotted equality .

= is for definition.

For any positive integer k, [k] .

= {1, . . . , k}. By default, · is the 2 norm if applied to a vector, and the operator norm if applied to a matrix.

C and c or any indexed versions are reserved for universal constants that may change from place to place.

We now state our main result, the recovery guarantee for learning orthogonal dictionary by solving formulation (1.1).

Theorem 3.1 (Recovering orthogonal dictionary via subgradient descent).

Suppose we observe DISPLAYFORM0 samples in the dictionary learning problem and we desire an accuracy ∈ (0, 1) for recovering the dictionary.

With probability at least 1 − exp −cmθ 3 n −3 log −3 m − exp (−c R/n), an algorithm which runs Riemannian subgradient descent R = C n log n times with independent random initializations on S n−1 outputs a set of vectors { a 1 , . . .

, a n } such that up to permutation and sign change, a i − a i 2 ≤ for all i ∈ [n].

The total number of subgradient descent iterations is bounded by DISPLAYFORM1 Here C, C , C , c, c > 0 are universal constants.

At a high level, the proof of Theorem 3.1 consists of the following steps, which we elaborate throughout the rest of this section.1.

Partition the sphere into 2n symmetric "good sets" and show certain directional gradient is strong on population objective E[f ] inside the good sets (Section 3.1).2.

Show that the same geometric properties carry over to the empirical objective f with high probability.

This involves proving the uniform convergence of the subdifferential set ∂f to E [∂f ] (Section 3.2).

3.

Under the benign geometry, establish the convergence of Riemannian subgradient descent to one of {±a i : i ∈ [n]} when initialized in the corresponding "good set" (Section 3.3).4.

Calling the randomly initialized optimization procedure O(n log n) times will recover all of {a 1 , . . .

, a n } with high probability, by a coupon collector's argument (Section 3.4).Scaling and rotating to identity Throughout the rest of this paper, we are going to assume WLOG that the dictionary is the identity matrix, i.e. A = I n , so that Y = X, f (q) = q X 1 , and the goal is to find the standard basis vectors {±e 1 , . . .

, ±e n }.

The case of a general orthogonal A can be reduced to this special case via rotating by A : q Y = q AX = (q ) X where q = A q and applying the result on q .

We also scale the objective by π/2 for convenience of later analysis.

We begin by characterizing the geometry of the expected objective E [f ].

Recall that we have rotated A to be identity, so that we have DISPLAYFORM0 Minimizers and saddles of the population objective We begin by computing the function value and subdifferential set of the population objective and giving a complete characterization of its stationary points, i.e. local minimizers and saddles.

Proposition 3.2 (Population objective value and gradient).

We have DISPLAYFORM1 (3.5) DISPLAYFORM2 The case k = 1 corresponds to the 2n global minimizers q = ±e i , and all other values of k correspond to saddle points.

A consequence of Proposition 3.3 is that the population objective has no "spurious local minima": each stationary point is either a global minimizer or a saddle point, though the problem itself is non-convex due to the constraint.

Identifying 2n "good" subsets We now define 2n subsets on the sphere, each containing one of the global minimizers {±e i } and possessing benign geometry for both the population and empirical objective, following (Gilboa et al., 2018) .

For any ζ ∈ [0, ∞) and i ∈ [n] define DISPLAYFORM3 For points in S DISPLAYFORM4 , the i-th index is larger than all other indices (in absolute value) by a multiplicative factor of ζ.

In particular, for any point in these subsets, the largest index is unique, so by Proposition 3.3 all population saddle points are excluded from these 2n subsets.

Intuitively, this partition can serve as a "tiebreaker": points in S (i+) ζ0 is closer to e i than all the other 2n − 1 signed basis vectors.

Therefore, we hope that optimization algorithms initialized in this region could favor e i over the other standard basis vectors, which we are going to show is indeed the case.

For simplicity, we are going to state our geometry results in S (n+) ζ ; by symmetry the results will automatically carry over to all the other 2n − 1 subsets.

Theorem 3.4 (Lower bound on directional subgradients).

Fix any ζ 0 ∈ (0, 1).

We have and all indices j = n such that q j = 0, DISPLAYFORM5 , we have that DISPLAYFORM6 These lower bounds verify our intuition: points inside S (n+) ζ0have subgradients pointing towards e n , both in a coordinate-wise sense and a combined sense: the direction e n − q n q is exactly the tangent direction of the sphere at q that points towards e n .

We now show that the benign geometry in Theorem 3.4 is carried onto the empirical objective f given sufficiently many samples, using a concentration argument.

The key result behind is the concentration of the empirical subdifferential set to the population subdifferential, where concentration is measured in the Hausdorff distance between sets.

Proposition 3.5 (Uniform convergence of subdifferential).

For any t ∈ (0, 1], when 10) with probability at least 1 − exp −cmθt 2 /log m , we have DISPLAYFORM0 DISPLAYFORM1 Here C, c ≥ 0 are universal constants.

The concentration result guarantees that the sub-differential set is close to its expectation given sufficiently many samples with high probability.

Choosing an appropriate concentration level t, the lower bounds on the directional subgradients carry over to the empirical objective f , which we state in the following theorem.

Theorem 3.6 (Directional subgradient lower bound, empirical objective).

There exist universal constants C, c ≥ 0 so that the following holds: for all ζ 0 ∈ (0, 1), when m ≥ Cn 4 θ −2 ζ −2 0 log 2 (n/ζ 0 ), with probability at least 1 − exp −cmθ 3 ζ 2 0 n −3 log −1 m , the following properties hold simultaneously for all the 2n subsets S DISPLAYFORM2 and all j ∈ [n] with q j = 0 and q DISPLAYFORM3 The consequence of Theorem 3.6 is two-fold.

First, it guarantees that the only possible stationary point of f in S (n+) ζ0 is e n : for every other point q = e n , property (b) guarantees that 0 / ∈ ∂ R f (q), therefore q is non-stationary.

Second, the directional subgradient lower bounds allow us to establish convergence of the Riemannian subgradient descent algorithm, in a way similar to showing convergence of unconstrained gradient descent on star strongly convex functions.

We now present an upper bound on the norm of the subdifferential sets, which is needed for the convergence analysis.

Proposition 3.7.

There exist universal constants C, c ≥ 0 such that sup ∂f (q) ≤ 2 ∀ q ∈ S n−1 (3.14) with probability at least 1 − exp −cmθ log −1 m , provided that m ≥ Cn log n. This particularly implies that DISPLAYFORM4 DISPLAYFORM5 Each iteration moves in an arbitrary Riemannian subgradient direction followed by a projection back onto the sphere.

We show that the algorithm is guaranteed to find one basis as long as the initialization is in the "right" region.

To give a concrete result, we set ζ 0 = 1/(5 log n).

Theorem 3.8 (One run of subgradient descent recovers one basis).

Let m ≥ Cθ −2 n 4 log 4 n and ∈ (0, 2θ/25].

With probability at least 1 − exp −cmθ 3 n −3 log −3 m the following happens.

If DISPLAYFORM0 1/(5 log n) , and we run the projected Riemannian subgradient descent with step size η (k) = k −α /(100 √ n) with α ∈ (0, 1/2), and keep track of the best function value so far until after iterate K is performed, producing q best .

Then, q best obeys 17) provided that DISPLAYFORM1 DISPLAYFORM2 , 64 DISPLAYFORM3 In particular, choosing α = 3/8 < 1/2, it suffices to let DISPLAYFORM4 Here C, C , c ≥ 0 are universal constants.

The above optimization result (Theorem 3.8) shows that Riemannian subgradient descent is able to find the basis vector e n when initialized in the associated region S (n+) 1/(5 log n) .

We now show that a simple uniformly random initialization on the sphere is guaranteed to be in one of these 2n regions with at least probability 1/2.

Lemma 3.9 (Random initialization falls in "good set").

Let q (0) ∼ Uniform(S n−1 ), then with probability at least 1/2, q (0) belongs to one of the 2n sets S DISPLAYFORM5

As long as the initialization belongs to S DISPLAYFORM0 1/(5 log n) , our finding-one-basis result in Theorem 3.8 guarantees that Riemannian subgradient descent will converge to e i or −e i respectively.

Therefore if we run the algorithm with independent, uniformly random initializations on the sphere multiple times, by a coupon collector's argument, we will recover all the basis vectors.

This is formalized in the following theorem.

Theorem 3.10 (Recovering the identity dictionary from multiple random initializations).

Let m ≥ Cn 4 θ −2 log 4 n and ∈ (0, 1), with probability at least 1 − exp −cmθ 3 n −3 log −3 m the following happens.

Suppose we run the Riemannian subgradient descent algorithm independently for R times, each with a uniformly random initialization on S n−1 , and choose the step size as DISPLAYFORM1 Then, provided that R ≥ C n log n, all standard basis vectors will be recovered up to accuracy with probability at least 1 − exp (−cR/n) in C Rθ −16/3 −8/3 n 4 log 8/3 n iterations.

Here C, C , c ≥ 0 are universal constants.

When the dictionary A is not the identity matrix, we can apply the rotation argument sketched in the beginning of this section to get the same result, which leads to our main result in Theorem 3.1.

A key technical challenge is establishing the uniform convergence of subdifferential sets in Proposition 3.5, which we now elaborate.

Recall that the population and empirical subdifferentials are DISPLAYFORM0 and we wish to show that the difference between ∂f (q) and E [∂f ] (q) is small uniformly over q ∈ Q = S n−1 .

Two challenges stand out in showing such a uniform convergence:1.

The subdifferential is set-valued and random, and it is unclear a-priori how one could formulate and analyze the concentration of random sets.2.

The usual covering argument won't work here, as the Lipschitz gradient property does not hold: ∂f (q) and E [∂f ] (q) are not Lipschitz in q. Therefore, no matter how fine we cover the sphere in Euclidean distance, points not in this covering can have radically different subdifferential sets.

We state and analyze concentration of random sets in the Hausdorff distance (defined in Section 2).

We now illustrate how the Hausdorff distance is the "right" distance to consider for concentration of subdifferentials-the reason is that the Hausdorff distance is closely related to the support function of sets, which for any set S ∈ R n is defined as DISPLAYFORM0 For convex compact sets, the sup difference between their support functions is exactly the Hausdorff distance.

Lemma 4.1 (Section 1.3.2, Molchanov (2013)).

For convex compact sets X, Y ⊂ R n , we have DISPLAYFORM1 Lemma 4.1 is convenient for us in the following sense.

Suppose we wish to upper bound the difference of ∂f (q) and E [∂f ] (q) along some direction u ∈ S n−1 (as we need in proving the key empirical geometry result Theorem 3.6).

As both subdifferential sets are convex and compact, by Lemma 4.1 we immediately have DISPLAYFORM2 Therefore, as long as we are able to bound the Hausdorff distance, all directional differences between the subdifferentials are simultaneously bounded, which is exactly what we want to show to carry the benign geometry from the population to the empirical objective.

We argue that the absence of gradient Lipschitzness is because the Euclidean distance is not the "right" metric in this problem.

Think of the toy example f (x) = |x|, whose subdifferential set ∂f (x) = sign(x) is not Lipschitz across x = 0.

However, once we partition R into R >0 , R <0 and {0} (i.e. according to the sign pattern), the subdifferential set is Lipschitz on each subset.

The situation with the dictionary learning objective is quie similar: we resolve the gradient nonLipschitzness by proposing a stronger metric d E on the sphere which is sign-pattern aware and averages all "subset angles" between two points.

Formally, we define d E as DISPLAYFORM0 (the second equality shown in Lemma C.1.)

Our plan is to perform the covering argument in d E , which requires showing gradient Lipschitzness in d E and bounding the covering number.

(4.7)As long as d E (p, q) ≤ , the indicator is non-zero with probability at most , and thus the above expectation should also be small -we bound it by O( log(1/ )) in Lemma F.5.To show the same for the empirical subdifferential ∂f , one only needs to bound the observed proportion of sign differences for all p, q such that d E (p, q) ≤ , which by a VC dimension argument is uniformly bounded by 2 with high probability (Lemma C.5).Bounding the covering number in d E Our first step is to reduce d E to the maximum length-2 angle (the d 2 metric) over any consistent support pattern.

This is achieved through the following vector angle inequality (Lemma C.2): for any p, DISPLAYFORM0 Therefore, as long as sign(p) = sign(q) (coordinate-wise) and DISPLAYFORM1 By Eq. (4.5), the above implies that d E (p, q) ≤ /π, the desired result.

Hence the task reduces to constructing an η = /n 2 covering in d 2 over any consistent sign pattern.

Our second step is a tight bound on this covering number: the η-covering number in d 2 is bounded by exp(Cn log(n/η)) (Lemma C.3).

For bounding this, a first thought would be to take the covering in all size-2 angles (there are n 2 of them) and take the common refinement of all their partitions, which gives covering number (C/η) O(n 2 ) = exp(Cn 2 log(1/η)).

We improve upon this strategy by sorting the coordinates in p and restricting attentions in the consecutive size-2 angles after the sorting (there are n − 1 of them).

We show that a proper covering in these consecutive size-2 angles by η/n will yield a covering for all size-2 angles by η.

The corresponding covering number in this case is thus (Cn/η) O(n) = exp(Cn log(n/η)), which modulo the log n factor is the tightest we can get.

Setup We set the true dictionary A to be the identity and random orthogonal matrices, respectively.

For each choice, we sweep the combinations of (m, n) with n ∈ {30, 50, 70, 100} and m = 10n {0.5,1,1.5,2,2.5} , and fix the sparsity level at θ = 0.1, 0.3, 0.5, respectively.

For each (m, n) pair, we generate 10 problem instances, corresponding to re-sampling the coefficient matrix X for 10 times.

Note that our theoretical guarantee applies for m = Ω(n 4 ), and the sample complexity we experiment with here is lower than what our theory requires.

To recover the dictionary, we run the Riemannian subgradient descent algorithm Eq. (3.16) with decaying step size η (k) = 1/ √ k, corresponding to the boundary case α = 1/2 in Theorem 3.8 with a much better base size.

Metric As Theorem 3.1 guarantees recovering the entire dictionary with R ≥ Cn log n independent runs, we perform R = round (5n log n) runs on each instance.

For each run, a true dictionary element a i is considered to be found if a i − q best ≤ 10 −3 .

For each instance, we regard it a successful recovery if the R = round (5n log n) runs have found all the dictionary elements, and we report the empirical success rate over the 10 instances.

Result From our simulations, Riemannian subgradient descent succeeds in recovering the dictionary as long as m ≥ Cn 2 FIG7 , across different sparsity level θ.

The dependency on n is consistent with our theory and suggests that the actual sample complexity requirement for guaranteed recovery might be even lower than O(n 4 ) we established.

3 The O(n 2 ) rate we observe also matches the results based on the SOS method BID6 Ma et al., 2016; Schramm & Steurer, 2017) .

Moreover, the problem seems to become harder when θ grows, evident from the observation that the success transition threshold being pushed to the right.

Additional experiments A faster alternative algorithm for large-scale instances is tested in Appendix H. A complementary experiment on real images is included as Appendix I.

This paper presents the first theoretical guarantee for orthogonal dictionary learning using subgradient descent on a natural 1 minimization formulation.

Along the way, we develop tools for analyzing the optimization landscape of nonconvex nonsmooth functions, which could be of broader interest.

For futute work, there is an O(n 2 ) sample complexity gap between what we established in Theorem 3.1, and what we observed in the simulations alongside previous results based on the SOS method BID6 Ma et al., 2016; Schramm & Steurer, 2017) .

As our main geometric result Theorem 3.6 already achieved tight bounds on the directional derivatives, further sample complexity improvement could potentially come out of utilizing second-order information such as the strong negative curvature (Lemma B.2), or careful algorithm-dependent analysis.

While our result applies only to (complete) orthogonal dictionaries, a natural question is whether we can generalize to overcomplete dictionaries.

To date the only known provable algorithms for learning overcomplete dictionaries in the linear sparsity regime are based on the SOS method BID6 Ma et al., 2016; Schramm & Steurer, 2017) .

We believe that our nonsmooth analysis has the potential of handling over-complete dictionaries, as for reasonably well-conditioned overcomplete dictionaries A, each a i (columns of A) makes a A approximately 1-sparse and so a i AX gives noisy estimate of a certain row of X.

So the same formulation as Eq. (1.1) intuitively still works.

We would like to leave that to future work.

Nonsmooth phase retrieval and deep networks with ReLU mentioned in Section 1.1 are examples of many nonsmooth, nonconvex problems encountered in practice.

Most existing theoretical results on these problems tend to be technically vague about handling the nonsmooth points: they either prescribe a rule for choosing a subgradient element, which effectively disconnects theory and practice because numerical testing of nonsmooth points is often not reliable, or ignore the nonsmooth points altogether, assuming that practically numerical methods would never touch these points-this sounds intuitive but no formalism on this appears in the relevant literature yet.

Besides our work, (Laurent & von Brecht, 2017; Kakade & Lee, 2018 ) also warns about potential problems of ignoring nonsmooth points when studying optimization of nonsmooth functions in machine learning.

We need the Hausdorff metric to measure differences between nonempty sets.

For any set X and a point p in R n , the point-to-set distance is defined as DISPLAYFORM0 For any two sets X 1 , X 2 ∈ R n , the Hausdorff distance is defined as DISPLAYFORM1 Moreover, for any sets DISPLAYFORM2 (A.4) On the sets of nonempty, compact subsets of R n , the Hausdorff metric is a valid metric; particularly, it obeys the triangular inequality: for nonempty, compact subsets X, Y, Z ⊂ R n , DISPLAYFORM3 (A.5) See, e.g., Sec. 7.1 of Sternberg (2013) for a proof.

Lemma A.1 (Restatement of Lemma A.1).

For convex compact sets X, Y ⊂ R n , we have DISPLAYFORM4 where h S (u) .

= sup x∈S x, u is the support function associated with the set S.

Proposition A.2 (Talagrand's comparison inequality, Corollary 8.6.3 and Exercise 8.6.5 of Vershynin FORMULA1 ).

Let {X x }

x∈T be a zero-mean random process on a subset T ⊂ R n .

Assume that for all x, y ∈ T we have DISPLAYFORM0 Then, for any t > 0 sup DISPLAYFORM1 with probability at least 1 − 2 exp −t 2 .

Here w(T ) .

= E g∼N (0,I) sup x∈T x, g is the Gaussian width of T and rad(T ) = sup x∈T x is the radius of T .

Proposition A.3 (Deviation inequality for sub-Gaussian matrices, Theorem 9.1.1 and Exercise 9.1.8 of Vershynin FORMULA1 ).

Let A be an n × m matrix whose rows A i 's are independent, isotropic, and sub-Gaussian random vectors in R m .

Then for any subset T ⊂ R m , we have DISPLAYFORM2 We have DISPLAYFORM3 where the last equality is obtained by conditioning on Ω and the fact that DISPLAYFORM4 The subdifferential expression comes from DISPLAYFORM5 and the fact that We first show that points in the claimed set are indeed stationary points by taking the choice v Ω = 0 in Eq. (3.5), giving the subgradient choice DISPLAYFORM6 DISPLAYFORM7 .

Let q ∈ S and such that q 0 = k. For all j ∈ supp(q), we have DISPLAYFORM8 On the other hand, for all j / ∈ supp(q), we always have [q Ω ] j = 0, so e j E [∂f ] (q) = 0.

Therefore, we have that E [∂f ] (q) = c(θ, k)q, and so DISPLAYFORM9 (B.6) Therefore q ∈ S is stationary.

To see that {±e i : i ∈ [n]} are the global minima, note that for all q ∈ S n−1 , we have DISPLAYFORM10 Equality holds if and only if q Ω 2 ∈ {0, 1} almost surely, which is only satisfied at q ∈ {±e i : i ∈ [n]}.To see that the other q's are saddles, we only need to show that there exists a tangent direction along which q is local max.

Indeed, for any other q, there exists at least two non-zero entries (with equal absolute value): WLOG assume that q 1 = q n > 0.

Using the reparametrization in Appendix B.3 and applying Lemma B.2, we get that E [f ] (q) is directionally differentiable along [−q −n ; 1−q 2 n qn ], with derivative zero (necessarily, because 0 ∈ E [∂ R f ] (q)) and strictly negative second derivative.

Therefore E [f ] (q) is locally maximized at q along this tangent direction, which shows that q is a saddle point.

The other direction (all other points are not stationary) is implied by Theorem 3.4, which guarantees that 0 / ∈ E [∂ R f ] (q) whenever q / ∈ S. Indeed, as long as q / ∈ S, q has a max absolute value coordinate (say n) and another non-zero coordinate with strictly smaller absolute value (say j).

For this pair of indices, the proof of Theorem 3.4(a) goes through for index j (even if q ∈ S (n+) ζ0 does not necessarily hold because the max index might not be unique), which implies that 0 / ∈ E [∂ R f ] (q).

For analysis purposes, we introduce the reparametrization w = q 1:(n−1) in the region S (n+) 0 , following (Sun et al., 2015) .

With this reparametrization, the problem becomes DISPLAYFORM0 The constraint comes from the fact that q n ≥ 1/ √ n and thus w ≤ (n − 1)/n.

Lemma B.1.

We have DISPLAYFORM1 Proof.

Direct calculation gives DISPLAYFORM2 as claimed.

Lemma B.2 (Negative-curvature region).

For all unit vector v ∈ S n−1 and all s ∈ (0, 1), let DISPLAYFORM3 it holds that DISPLAYFORM4 In other words, for all w = 0, ±w/ w is a direction of negative curvature.

Proof.

By Lemma B.1, DISPLAYFORM5 For s ∈ (0, 1), h v (s) is twice differentiable, and we have DISPLAYFORM6 completing the proof.

Proof.

For any unit vector v ∈ R n−1 , define h v (t) .

= E [g] (tv) for t ∈ (0, 1).

We have from Lemma B.1 DISPLAYFORM0 (B.20)Moreover, DISPLAYFORM1 We are interested in the regime of t so that DISPLAYFORM2 (B.28) DISPLAYFORM3 For any w, applying the above result to the unit vector w/ w and recognizing that ∇ t h w/ w (t) = D w/ w g (w) = D c w/ w g (w), we complete the proof.

We first show Eq. (3.9) using the reparametrization in Appendix B.3.

We have DISPLAYFORM0 where the second equality follows by differentiating g via the chain rule.

Now, by Lemma B.3, DISPLAYFORM1 For each radial direction v .

= w/ w , consider points of the form tv with t ≤ 1/ 1 + v 2 ∞ .

Obviously, the function DISPLAYFORM2 is monotonically decreasing wrt t. Thus, to derive a lower bound, it is enough to consider the largest t allowed.

In S (n+) ζ0, the limit amounts to requiring q 2 n / w DISPLAYFORM3 So for any fixed v and all allowed t for points in S DISPLAYFORM4 , a uniform lower bound is DISPLAYFORM5 So we conclude that for all q ∈ S DISPLAYFORM6 We now turn to showing Eq. (3.8).

For e j with q j = 0, DISPLAYFORM7 (B.37)So for all j with q j = 0, we have DISPLAYFORM8 completing the proof.

C PROOFS FOR SECTION 3.2 DISPLAYFORM9 We stress that this notion always depend on θ, and we will omit the subscript θ when no confusion is expected.

This indeed defines a metric on subsets of S n−1 .Lemma C.1.

Over any subset of S n−1 with a consistent support pattern, d E is a valid metric.

Proof.

Recall that (x, y) .

= arccos x, y defines a valid metric on S n−1 .

4 In particular, the triangular inequality holds.

For d E and p, q ∈ S n−1 with the same support pattern, we have DISPLAYFORM10 where we have adopted the convention that (0, v) .

= 0 for any v.

It is easy to verify that d E (p, q) = 0 ⇐⇒ p = q, and d E (p, q) = d E (q, p).

To show the triangular inequality, note that for any p, q and r with the same support pattern, p Ω , q Ω , and r Ω are either identically zero, or all nonzero.

For the former case, DISPLAYFORM11 holds trivially.

For the latter, since (·, ·) obeys the triangular inequality uniformly over the sphere, DISPLAYFORM12 Proof.

The inequality holds trivially when either of u, v is zero.

Suppose they are both nonzero and wlog assume both are normalized, i.e., u = v = 1.

Then, DISPLAYFORM13 2 ) (u Ω , v Ω ) > π/2, the claimed inequality holds trivially, as (u, v) ≤ π/2 by our assumption.

Suppose Ω∈( DISPLAYFORM14 by recursive application of the following inequality: DISPLAYFORM15 So we have that when Ω∈( DISPLAYFORM16 as claimed.

Lemma C.3 (Covering in maximum length-2 angles).

For any η ∈ (0, 1/3), there exists a subset Q ⊂ S n−1 of size at most (5n log(1/η)/η) 2n−1 satisfying the following: for any p ∈ S n−1 , there exists some q ∈ Q such that (p Ω , q Ω ) ≤ η for all Ω ⊂ [n] with |Ω|≤ 2.

DISPLAYFORM17 our goal is to give an η-covering of S n−1 in the d 2 metric.

Step 1 We partition S n−1 according to the support, the sign pattern, and the ordering of the non-zero elements.

For each configuration, we are going to construct a covering with the same configuration of support, sign pattern, and ordering.

There are no more than 3 n · n!

such configurations.

Note that we only need to construct one such covering for each support size, and for each support size we can ignore the zero entries -the angle (p Ω , q Ω ) is always zero when p, q have matching support and Ω contains at least one zero index.

Therefore, the task reduces to bounding the covering number of DISPLAYFORM18 Step 2 We bound the covering number of A n by induction.

Suppose that DISPLAYFORM19 holds for all n ≤ n − 1.

(The base case m = 2 clearly holds.) Let C n ⊂ S n −1 be the correpsonding covering sets.

We now construct a covering for A n .

Let R .

= 1/η = r k for some r ≥ 1 and k to be determined.

Consider the set DISPLAYFORM20 We claim that Q r,k with properly chosen (r, k) gives a covering of DISPLAYFORM21 .

Each consecutive ratio p i+1 /p i falls in one of these intervals, and we choose q so that q i+1 /q i is the left endpoint of this interval.

Such a q satisfies q ∈ Q r,k and DISPLAYFORM22 By multiplying these bounds, we obtain that for all 1 ≤ i < j ≤ n, DISPLAYFORM23 Take r = 1 + η/2n, we have r n−1 = (1 + η/2n) n−1 ≤ exp(η/2) ≤ 1 + η.

Therefore, for all i, j,we have pj /pi qj /qi ∈ [1, 1 + η), which further implies that ((p i , p j ), (q i , q j )) ≤ η by Lemma F.4.

Thus we have for all |Ω|≤ 2 that (p Ω , q Ω ) ≤ η. (The size-1 angles are all zero as we have sign match.)For this choice of r, we have k = log R/log r and thus DISPLAYFORM24 (C.28) and we have N (A n,R ) ≤ N n .Step 3 We now construct the covering of A n \ A n,R .

For any p ∈ A n \ A n,R , there exists some i such that p i+1 /p i ∈ [R, ∞), which means that the angle of the ray (p i , p i+1 ) is in between [arctan(R), π/2) = [π/2 − η, π/2).

As p is sorted, we have that DISPLAYFORM25 So if we take q such that q i+1 /q i ∈ [R, ∞), q also has the above property, which gives that DISPLAYFORM26 Therefore to obtain the cover in d 2 , we only need to consider the angles for Ω ⊂ {1, . . .

, i} and Ω ⊂ {i + 1, . . .

, n}, which can be done by taking the product of the covers in A i and A n−i .By considering all i ∈ {1, . . .

, n − 1}, we obtain the bound DISPLAYFORM27 Step 4 Putting together Step 2 and Step 3 and using the inductive assumption, we get that DISPLAYFORM28 This shows the case for m = n and completes the induction.

Step 5 Considering all configurations of {support, sign pattern, ordering}, we have DISPLAYFORM29 Lemma C.4 (Covering number in the d E metric).

Assume n ≥ 3.

There exists a numerical constant C > 0 such that for any ∈ (0, 1), S n−1 admits an -net of size exp(Cn log n ) w.r.t.

d E defined in Eq. (C.1): for any p ∈ S n−1 , there exists a q in the net with supp (q) = supp (p) and d E (p, q) ≤ .

We say such nets are admissible for S n−1 wrt d E .Proof.

Let η = /n 2 .

By Lemma C.3, there exists a subset Q ⊂ S n−1 of size at most 5n log(1/η) η DISPLAYFORM30 such that for any p ∈ S n−1 , there exists q ∈ S n−1 such that supp(p) = supp(q) and (p Ω , q Ω ) ≤ η for all |Ω|≤ 2.

In particular, the |Ω|= 1 case says that sign(p) = sign(q), which implies that DISPLAYFORM31 Thus, applying the vector angle inequality (Lemma C.2), for any p ∈ S n−1 and the corresponding q ∈ Q, we have DISPLAYFORM32 Summing up, we get DISPLAYFORM33 Below we establish the "Lipschitz" property in terms of d E distance.

Lemma C.5.

Fix θ ∈ (0, 1).

For any ∈ (0, 1), let N be an admissible -net for S n−1 wrt d E .

Let x 1 , . . . , x m be iid copies of x ∼ iid BG(θ) in R n .

When m ≥ C −2 n, the inequality DISPLAYFORM34 holds with probability at least 1 − exp −c 2 m .

Here C and c are universal constants independent of .

Proof.

We call any pair of p, q ∈ S n−1 with q ∈ N , supp (p) = supp (q), and d E (p, q) ≤ an admissible pair.

Over any admissible pair (p, q), E [R] = d E (p, q).

We next bound the deviation R − E [R] uniformly over all admissible (p, q) pairs.

Observe that the process R is the sample average of m indicator functions.

Define the hypothesis class H = x → 1 sign p x = sign q x : (p, q) is an admissible pair .(C.42) and let d vc (H) be the VC-dimension of H. From concentration results for VC-classes (see, e.g., Eq (3) and Theorem 3.4 of BID7 ), we have DISPLAYFORM35 for any t > 0.

It remains to bound the VC-dimension d vc (H).

First, we have DISPLAYFORM36 Observe that each set in the latter hypothesis class can be written as DISPLAYFORM37 the union of intersections of two halfspaces.

Thus, letting DISPLAYFORM38 be the class of halfspaces, we have DISPLAYFORM39 Note that H 0 has VC-dimension n + 1.

Applying bounds on the VC-dimension of unions and intersections (Theorem 1.1, Van Der Vaart & Wellner (2009)), we get that DISPLAYFORM40 Plugging this bound into Eq. (C.43), we can set t = /2 and make m large enough so that C 0 √ C 3 n/m ≤ /2, completing the proof.

Proposition C.6 (Pointwise convergence).

For any fixed q ∈ S n−1 , DISPLAYFORM0 Here C a , C b ≥ 0 are universal constants.

DISPLAYFORM1 and consider the zero-mean random process {X u } defined on S n−1 .

For any u, v ∈ S n−1 , we have DISPLAYFORM2 where we write DISPLAYFORM3 where we have used Lemma F.1 to obtain the last upper bound.

If DISPLAYFORM4 and we can use similar argument to conclude that DISPLAYFORM5 Thus, {X u } is a centered random process with sub-Gaussian increments with a parameter C 4 / √ m. We can apply Proposition A.2 to conclude that DISPLAYFORM6 which implies the claimed result.

Throughout the proof, we let c, C denote universal constants that could change from step to step.

Fix an ∈ (0, 1/2) to be decided later.

Let N be an admissible net for S n−1 wrt d E , with |N | ≤ exp(Cn log(n/ )) (Lemma C.4).

By Proposition C.6 and the union bound, DISPLAYFORM0 For any p ∈ S n−1 , let q ∈ N satisfy supp (q) = supp (p) and d E (p, q) ≤ .

Then we have DISPLAYFORM1 by the triangular inequality for the Hausdorff metric.

By the preceding union bound, term I is bounded by t/3 as long as the bad event does not happen.

For term II, we have DISPLAYFORM2 where the last line follows from Lemma F.5.

As long as ≤ ct/ log(1/t), the above term is upper bounded by t/3.

For term III, we have DISPLAYFORM3 By Lemma C.5, with probability at least 1 − exp(−c 2 m), the number of different signs is upper bounded by 2m for all p, q such that d E (p, q) ≤ .

On this good event, the above quantity can be upper bounded as follows.

Define a set T .

= {s ∈ R m : s i ∈ {+1, −1, 0} , s 0 ≤ 2m } and consider the quantity sup s∈T Xs , where DISPLAYFORM4 uniformly (i.e., indepdent of p, q and u).

We have DISPLAYFORM5 Noting that 1/ √ θ · X has independent, isotropic, and sub-Gaussian rows with a parameter C/ √ θ, we apply Proposition A.3 and obtain that DISPLAYFORM6 with probability at least 1 − 2 exp −t 2 0 .

So we have over all admissible (p, q) pairs, DISPLAYFORM7 Setting t 0 = ct √ m and = ct θ/log m, we have that DISPLAYFORM8 provided that m ≥ C t −2 n = Ct −1 n θ/log m, which is subsumed by the earlier requirement m ≥ Ct −2 n.

Putting together the three bounds Eq. (C.62), Eq. (C.67), Eq. (C.80), we can choose DISPLAYFORM9 .

A sufficient condition is that m ≥ Cnt −2 log 2 (n/t) for sufficiently large C. When this is satisfied, the probability is further lower bounded by 1 − exp(−cmθt 2 /log m).

Define DISPLAYFORM0 (C.85) By Proposition 3.5, with probability at least 1 − exp −cmθ 3 ζ 2 0 n −3 log −1 m we have DISPLAYFORM1 0 log (n/ζ 0 ).

We now show the properties Eq. (3.12) and Eq. (3.13) on this good event, focusing on S (n+) ζ0 but obtaining the same results for all other 2n − 1 subsets by the same arguments.

For Eq. (3.12), we have ∂ R f (q) , e j /q j − e n /q n = ∂f (q) , I − qq (e j /q j − e n /q n ) = ∂f (q) , e j /q j − e n /q n .(C.87) Now sup ∂f (q) , e n /q n − e j /q j = h ∂f (q) (e n /q n − e j /q j ) (C.88) = Eh ∂f (q) (e n /q n − e j /q j ) − Eh ∂f (q) (e n /q n − e j /q j ) + h ∂f (q) (e n /q n − e j /q j ) (C.89)≤ Eh ∂f (q) (e n /q n − e j /q j ) + e n /q n − e j /q j sup DISPLAYFORM2 By Theorem 3.4(a), DISPLAYFORM3 Moreover, e n /q n − e j /q j = 1/q 2 n + 1/q 2 j ≤ 1/q 2 n + 3/q 2 n ≤ 2 √ n. Meanwhile, we have DISPLAYFORM4 We conclude that inf ∂f (q) , e j /q j − e n /q n = − sup ∂f (q) , e n /q n − e j /q j (C.94) DISPLAYFORM5 as claimed.

For Eq. (3.13), we have by Theorem 3.4(b) that sup ∂f (q) , e n − q n q = h ∂f (q) (e n − q n q) (C.97) = Eh ∂f (q) (e n − q n q) −

Eh ∂f (q) (e n − q n q) + h ∂f (q) (e n − q n q) (C.98)≤ Eh ∂f (q) (e n − q n q) + e n − q n q sup DISPLAYFORM6 (C.100)As we are on the good event DISPLAYFORM7 we have inf ∂f (q) , q n q − e n = − sup ∂f (q) , e n − q n q (C.102) DISPLAYFORM8 q − e n for all q with q n ≥ 0 completes the proof.

C.5 PROOF OF PROPOSITION 3.7For any q ∈ S n−1 , DISPLAYFORM9 by the metric property of the Hausdorff metric.

On one hand, we have DISPLAYFORM10 On the other hand, by Proposition 3.5, DISPLAYFORM11 (C.107) with probability at least 1 − exp −c 1 mθ log −1 m , provided that m ≥ C 2 n 2 log n (simplified using θ ≥ 1/n).

Combining the two results complete the proof.

For w with w = 1/2, DISPLAYFORM12 So, back to the q space, DISPLAYFORM13 Combining the results in Eq. (C.115) and Eq. (C.121), we conclude that with high probability DISPLAYFORM14 , which is equivalent to w ≤ 1/2 in the w space.

Under this constraint, by Lemma B.3, DISPLAYFORM15 So, emulating the proof of Eq. (3.9) in Theorem 3.4, we have that for q ∈ S (n+) ζ0with q −n ≤ 1/2, (C.125) where at the last inequality we use q n = 1 − w 2 ≥ √ 3/2 when w ≤ 1/2.

Moreover, we emulate the proof of Eq. (3.13) in Theorem 3.6 to obtain that C.126) with probability at least 1 − exp −cmθ 3 log −1 m , provided that m ≥ Cθ −2 n log n. DISPLAYFORM16 DISPLAYFORM17 The last step of our proof is invoking the mean value theorem, similar to the proof of Proposition C.7.

For any q, we have DISPLAYFORM18 for a certain t ∈ (0, 1) and a certain v ∈ ∂g (tw).

We have ).

Set η = t 0 /(100 √ n) for t 0 ∈ (0, 1).

For any ζ 0 ∈ (0, 1), on the good events stated in Proposition 3.7 and Theorem 3.6, we have for all q ∈ S (n+) ζ0 \ S (n+) 1 DISPLAYFORM19 and q + being the next step of Riemannian subgradient descent that DISPLAYFORM20 In particular, we have q + ∈ S (n+) ζ0.Proof.

We divide the index set [n − 1] into three sets DISPLAYFORM21 We perform different arguments on different sets.

We let g (q) ∈ ∂ R f (q) be the subgradient taken at q and note by Proposition 3.7 that g ≤ 2, and so |g i | ≤ 2 for all i ∈ [n].

We have DISPLAYFORM22 Provided that η ≤ 1/(4 √ n), 1 − 2η √ n ≥ 1/2, and so DISPLAYFORM23 where the last inequality holds when η ≤ 1/ √ 40n.

For any j ∈ I 1 , DISPLAYFORM24 where the very last inequality holds when η ≤ 1/(26 √ n).

DISPLAYFORM25 , I 2 is nonempty.

For any j ∈ I 2 , q 2 +,n q 2 DISPLAYFORM26 Since g j /q j ≤ 2 √ 3n, 1 − ηg j /q j ≥ 1/2 when η ≤ 1/ 4 √ 3n .

Conditioned on this and due to that g j /q j − g n /q n ≥ 0, it follows DISPLAYFORM27 , we have q 2 n / q −n 2 ∞ ≤ 2, so there must be a certain j ∈ I 2 satisfying q 2 n /q 2 j ≤ 2.

We conclude that when DISPLAYFORM28 the index of largest entries of q +,−n remains in I 2 .On the other hand, when η ≤ 1/(100 √ n), for all j ∈ I 2 , DISPLAYFORM29 So when η = t/(100 √ n) for any t ∈ (0, 1), DISPLAYFORM30 completing the proof.

Proposition D.2.

For any ζ 0 ∈ (0, 1), on the good events stated in Proposition 3.7 and Theorem 3.6, if the step sizes satisfy DISPLAYFORM31 the iteration sequence will stay in S DISPLAYFORM32 where the last inequality holds provided that η ≤ (1 − ζ 0 ) /(9 √ n).

Combining the two cases finishes the proof.

D.2 PROOF OF THEOREM 3.8As we have DISPLAYFORM33 , the entire sequence q DISPLAYFORM34 will stay in S For any q and any v ∈ ∂ R f (q), we have v, q = 0 and therefore DISPLAYFORM35 So q − ηv is not inside B n .

Since projection onto B n is a contraction, we have DISPLAYFORM36 where we have used the bounds in Proposition 3.7 and Theorem 3.6 to obtain the last inequality.

Further applying Proposition C.7, we have DISPLAYFORM37 Summing up the inequalities until step K (assumed ≥ 5), we have DISPLAYFORM38 Substituting the following estimates DISPLAYFORM39 and noting 16 q (K) − e n 2 ≤ 32, we have DISPLAYFORM40 and when K ≥ 1, DISPLAYFORM41 (D.27) So we conclude that when DISPLAYFORM42 When this happens, by Proposition C.8, DISPLAYFORM43 Plugging in the choice ζ 0 = 1/(5 log n) in Eq. (D.28) gives the desired bound on the number of iterations.

E PROOFS FOR SECTION 3.4E.1 PROOF OF LEMMA 3.9Lemma E.1.

For all n ≥ 3 and ζ ≥ 0, it holds that DISPLAYFORM44 We note that a similar result appears in (Gilboa et al., 2018) but our definitions of the region S ζ are slightly different.

For completeness we provide a proof in Lemma F.3.We now prove Lemma 3.9.

Taking ζ = 1/(5 log n) in Lemma E.1, we obtain DISPLAYFORM45 By symmetry, all the 2n sets S1/(5 log n) , S DISPLAYFORM46 have the same volume which is at least 1/(4n).

As q (0) ∼ Uniform(S n−1 ), it falls into their union with probability at least 2n · 1/(4n) = 1/2, on which it belongs to a uniformly random one of these 2n sets.

Assume that the good event in Proposition 3.7 happens and that in Theorem 3.6 happens to all the 2n sets S (i+) 1/(5 log n) , S DISPLAYFORM0 , which by setting ζ 0 = 1/(5 log n) has probability at least DISPLAYFORM1 By Lemma 3.9, random initialization will fall these 2n sets with probability at least 1/2.

When it falls in one of these 2n sets, by Theorem 3.8, one run of the algorithm will find a signed standard basis vector up to accuracy.

With R independent runs, at least S .

= 1 4 R of them are effective with probability at least 1 − exp −(R/4) 2 /(R/4 · 2) = 1 − exp (−R/8), due to Bernstein's inequality.

After these effective runs, the probability any standard basis vector is missed (up to sign) is bounded by DISPLAYFORM2 where the second inequality holds whenever S ≥ 2n log n.

Lemma F.1.

For x ∼ BG(θ), x ψ2 ≤ C a .

For any vector u ∈ R n and x ∼ iid BG(θ), DISPLAYFORM0 Proof.

For any λ ∈ R, DISPLAYFORM1 So x ψ2 is bounded by a universal constant.

Moreover, DISPLAYFORM2 for any t ≥ 0.

Here C a , C b ≥ 0 are universal constants.

Consider the zero-centered random process defined on S n−1 : DISPLAYFORM0 where we use the estimate in Lemma F.1 to obtain the last inequality.

Note that X q is a mean-zero random process, and we can invoke Proposition A.2 with w(S n−1 ) = C 4 √ n and rad S n−1 = 2 to get the claimed result.

Lemma F.3.

For all n ≥ 3 and ζ ≥ 0, it holds that DISPLAYFORM1 Proof.

We have DISPLAYFORM2 where we write ψ(t) .

= 1 √ 2π t −t exp −s 2 /2 ds.

Now we derive a lower bound of the volume ratio by considering a first-order Taylor expansion of the last equation around ζ = 0 (as we are mostly interested in small ζ).

By symmetry,h (0) = 1/(2n).

Moreover, we have DISPLAYFORM3 2 /2 x 2 ψ n−1 (x) dx (F.15) and combining the above integral results, we conclude thath (ζ) ≥ 0 and complete the proof.

Lemma F.4.

Let (x 1 , y 1 ), (x 2 , y 2 ) ∈ R 2 >0 be two points in the first quadrant satisfying y 1 ≥ x 1 and y 2 ≥ x 2 , and y2/x2 y1/x1 ∈ [1, 1 + η] for some η ≤ 1, then we have ((x 1 , y 1 ), (x 2 , y 2 )) ≤ η.

DISPLAYFORM4 Proof.

For i = 1, 2, let θ i be the angle between the ray (x i , y i ) and the x-axis.

Our assumption implies that θ i ∈ [π/4, π/2) and θ 2 ≥ θ 1 , thus ((x 1 , y 1 ), (x 2 , y 2 )) = θ 2 − θ 1 , so we have (F.30) tan ((x 1 , y 1 ), (x 2 , y 2 )) = tan θ 2 − tan θ 1 1 + tan θ 2 tan θ 1 = y 2 /x 2 − y 1 /x 1 1 + y 2 y 1 /(x 2 x 1 ) = y2/x2 y1/x1 − 1 y 2 /x 2 + x 1 /y 1 ≤ y 2 /x 2 y 1 /x 1 − 1 ≤ η.

Therefore ((x 1 , y 1 ), (x 2 , y 2 )) ≤ arctan(η) ≤ η.

where we have used θ ≤ 1/2 and ≤ 1/2.

The Riemannian subgradient descent is cheap per iteration but slow in overall convergence, similar to many other first-order methods.

We also test a faster quasi-Newton type method, GRANSO, 6 that employs BFGS for solving constrained nonsmooth problems based on sequential quadratic optimization (Curtis et al., 2017) .

For a large dictionary of dimension n = 400 and sample complexity m = 10n 2 (i.e., 1.6 × 10 6 ), GRANSO successfully identifies a basis after 1500 iterations with CPU time 4 hours on a two-socket Intel Xeon E5-2640v4 processor (10-core Broadwell, 2.40 GHz)-this is approximately 10× faster than the Riemannian subgradient descent method, showing the potential of quasi-Newton type methods for solving large-scale problems.

To experiment with images, we follow a typical setup for dictionary learning as used in image processing (Mairal et al., 2014) .

We focus on testing if complete (i.e., square and invertible) dictionaries are reasonable sparsification bases for real images, instead on any particular image processing or vision tasks.

so that nonvanishing singular values of Y are identically one.

We then solve formulation (1.1) round (5n log n) times with n = 64 using the BFGS solver based on GRANSO, obtaining round (5n log n) vectors.

Negative equivalent copies are pruned and vectors with large correlations with other remaining vectors are sequentially removed until only 64 vectors are left.

This forms the final complete dictionary.

Results The learned complete dictionaries for the two test images are displayed in the second row of FIG1 .

Visually, the dictionaries seem reasonably adaptive to the image contents: for the left image with prevalent sharp edges, the learned dictionary consists of almost exclusively oriented sharp corners and edges, while for the right image with blurred textures and occasional sharp features, the learned dictionary does seem to be composed of the two kinds of elements.

Let the learned dictionary be A. We estimate the representation coefficients as A −1 Y .

The third row of FIG1 contains the histograms of the coefficients.

For both images, the coefficients are sharply concentrated around zero (see also the fourth row for zoomed versions of the portions around zero), and the distribution resembles a typical zero-centered Laplace distribution-which is a good indication of sparsity.

Quantitatively, we calculate the mean sparsity level of the coefficient vectors (i.e., columns of A −1 Y ) by the metric · 1 / · 2 : for a vector v ∈ R n , v 1 / v 2 ranges from 1 (when v is one-sparse) to √ n (when v is fully dense with elements of equal magnitudes), which serves as a good measure of sparsity level for v. For our two images, the sparsity levels by the norm-ratio metric are 5.9135 and 6.4339, respectively, while the fully dense extreme would have a value √ 64 = 8, suggesting the complete dictionaries we learned are reasonable sparsification bases for the two natural images, respectively.

@highlight

Efficient dictionary learning by L1 minimization via a novel analysis of the non-convex non-smooth geometry.
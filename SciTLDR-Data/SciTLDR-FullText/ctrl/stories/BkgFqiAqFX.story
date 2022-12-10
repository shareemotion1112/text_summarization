Giving provable guarantees for learning neural networks is a core challenge of machine learning theory.

Most prior work gives parameter recovery guarantees for one hidden layer networks, however, the networks used in practice have multiple non-linear layers.

In this work, we show how we can strengthen such results to deeper networks -- we address the problem of uncovering the lowest layer in a deep neural network under the assumption that the lowest layer uses a high threshold before applying the activation, the upper network can be modeled as a well-behaved polynomial and the input distribution is gaussian.

Understanding the landscape of learning neural networks has been a major challege in machine learning.

Various works gives parameter recovery guarantees for simple one-hidden-layer networks where the hidden layer applies a non-linear activation u after transforming the input x by a matrix W, and the upper layer is the weighted sum operator: thus f (x) = a i u(w T i x).

However, the networks used in practice have multiple non-linear layers and it is not clear how to extend these known techniques to deeper networks.

We consider a multilayer neural network with the first layer activation u and the layers above represented by an unknown polynomial P such that it has non-zero non-linear components.

More precisely, the function f computed by the neural network is as follows: f W (x) = P (u(w We assume that the input x is generated from the standard Gaussian distribution and there is an underlying true network (parameterized by some unknown W * ) 1 from which the labels are generated.

In this work we strengthen previous results for one hidden layer networks to a larger class of functions representing the transform made by the upper layer functions if the lowest layer uses a high threshold (high bias term) before applying the activation: u(a − t) instead of u(a).

Intuitively, a high threshold is looking for a high correlation of the input a with a direction w * i .

Thus even if the function f is applying a complex transform after the first layer, the identity of these high threshold directions may be preserved in the training data generated using f .Learning with linear terms in P .

Suppose P has a linear component then we show that increasing the threshold t in the lowest layer is equivalent to amplifying the coefficients of the linear part.

Instead of dealing with the polynomial P it turns out that we can roughly think of it as P (µX 1 , ..., µX d ) where µ decreases exponentially in t (µ ≈ e −t 2 ).

As µ decreases it has the effect of diminishing the non-linear terms more strongly so that relatively the linear terms stand out.

Taking advantage of this effect we manage to show that if t exceeds a certain threshold the non linear terms drop in value enough so that the directions w i can be learned by relatively simple methods.

We show that we can get close to the w i applying a simple variant of PCA.

While an application of PCA can be thought of as finding principal directions as the local maxima of max ||z||=1 E[f (x)(z T x) 2 ], DISPLAYFORM0 If W * has a constant condition number then the local maxima can be used to recover directions that are transforms of w i .

Theorem 1 (informal version of Claim 2, Theorem 11).

If t > c √ log d for large enough constant c > 0 and P has linear terms with absolute value of coefficients at least 1/poly(d) and all coefficients at most O(1), we can recover the weight vector w i within error 1/poly(d) in time poly(d).These approximations of w i obtained collectively can be further refined by looking at directions along which there is a high gradient in f ; for monotone functions we show how in this way we can recover w i exactly (or within any desired precision.

Theorem 2. (informal version of Theorem 5) Under the conditions of the previous theorem, for monotone P , there exists a procedure to refine the angle to precision in time poly(1/ , d) starting from an estimate that is 1/poly(d) close.

The above mentioned theorems hold for u being sign and ReLU.

3 When P is monotone and u is the sign function, learning W is equivalent to learning a union of half spaces.

We learn W * by learning sign of P which is exactly the union of halfspaces w T i x = t. Thus our algorithm can also be viewed as a polynomial time algorithm for learning a union of large number of half spaces that are far from the origin -to our knowledge this is the first polynomial time algorithm for this problem but with this extra requirement (see earlier work BID12 for an exponential time algorithm).

Refer to Appendix B.6 for more details.

Such linear components in P may easily be present: consider for example the case where P (X) = u(v T X − b) where u is say the sigmoid or the logloss function.

The taylor series of such functions has a linear component -note that since the linear term in the taylor expansion of u(x) has coefficient u (0), for expansion of u(x−b) it will be u (−b) which is Θ(e −b ) in the case of sigmoid.

In fact one may even have a tower (deep network) or such sigmoid/logloss layers and the linear components will still be present -unless they are made to cancel out precisely; however, the coefficients will drop exponentially in the depth of the networks and the threshold b.

Sample complexity with low thresholds and no explicit linear terms.

Even if the threshold is not large or P is not monotone, we show that W * can be learned with a polynomial sample complexity (although possibly exponential time complexity) by finding directions that maximize the gradient of f .

Theorem 3 (informal version of Corollary 1).

If u is the sign function and w i 's are orthogonal then in poly(1/ , d) samples one can determine W * within precision if the coefficient of the linear terms in P (µ(X 1 + 1), µ(X 2 + 1), µ(X 3 + 1), . . .) is least 1/poly(d)Learning without explicit linear terms.

We further provide evidence that P may not even need to have the linear terms -under some restricted cases (section 4), we show how such linear terms may implicitly arise even though they may be entirely apparently absent.

For instance consider the case when P = X i X j that does not have any linear terms.

Under certain additional assumptions we show that one can recover w i as long as the polynomial P (µ(X 1 + 1), µ(X 2 + 1), µ(X 3 + 1), ..) (where µ is e −t has linear terms components larger than the coefficients of the other terms).

Note that this transform when applied to P automatically introduces linear terms.

Note that as the threshold increases applying this transform on P has the effect of gathering linear components from all the different monomials in P and penalizing the higher degree monomials.

We show that if W * is a sparse binary matrix then we can recover W * when activation u(a) = e ρa under certain assumptions about the structure of P .

When we assume the coefficients are positive then these results extend for binary low l 1 -norm vectors without any threshold.

Lastly, we show that for even activations (∀a, u(a) = u(−a)) under orthogonal weights, we can recover the weights with no threshold.

Learning with high thresholds at deeper layers.

We also point out how such high threshold layers could potentially facilitate learning at any depth, not just at the lowest layer.

If there is any cut in the network that takes inputs X 1 , . . .

, X d and if the upper layers operations can be modelled by a polynomial P , then assuming the inputs X i have some degree of independence we could use this to modularly learn the lower and upper parts of the network separately (Appendix E) Related Work.

Various works have attempted to understand the learnability of simple neural networks.

Despite known hardness results BID8 ; BID2 , there has been an array of positive results under various distributional assumptions on the input and the underlying noise in the label.

Most of these works have focused on analyzing one hidden layer neural networks.

A line of research has focused on understanding the dynamics of gradient descent on these networks for recovering the underlying parameters under gaussian input distribution Du et al. FIG1 ; BID10 ; BID16 ; BID14 ; BID17 .

Another line of research borrows ideas from kernel methods and polynomial approximations to approximate the neural network by a linear function in a high dimensional space and subsequently learning the same BID15 ; BID8 ; BID7 a) .

Tensor decomposition methods BID0 BID9 have also been applied to learning these simple architectures.

The complexity of recovering arises from the highly non-convex nature of the loss function to be optimized.

The main result we extend in this work is by BID5 .

They learn the neural network by designing a loss function that allows a "well-behaved" landscape for optimization avoiding the complexity.

However, much like most other results, it is unclear how to extend to deeper networks.

The only known result for networks with more than one hidden layer is by BID7 .

Combining kernel methods with isotonic regression, they show that they can provably learn networks with sigmoids in the first hidden layer and a single unit in the second hidden layer in polynomial time.

We however model the above layer as a multivariate polynomial allowing for larger representation.

Another work BID1 deals with learning a deep generative network when several random examples are generated in an unsupervised setting.

By looking at correlations between input coordinates they are able to recover the network layer by layer.

We use some of their ideas in section 4 when W is a sparse binary matrix.

Notation.

We denote vectors and matrices in bold face.

|| · || p denotes the l p -norm of a vector.

|| · || without subscript implies the l 2 -norm.

For matrices || · || denotes the spectral norm and || · || F denotes the forbenius norm.

N (0, Σ) denotes the multivariate gausssian distribution with mean 0 and covariance Σ. For a scalar x we will use φ(x) to denote the p.d.f.

of the univariate standard normal distribution with mean zero and variance 1 .For a vector x we will use φ(x) to denote the p.d.f.

of the multivariate standard normal distribution with mean zero and variance 1 in each direction.

Φ denotes the c.d.f.

of the standard gausssian distribution.

Also define Φ c = 1 − Φ. Let h i denote the ith normalized Hermite polynomial Wikipedia contributors (2018).

For a function f , letf i denote the ith coefficient in the hermite expansion of f , that is, DISPLAYFORM1 For a given function f computed by the neural network, we assume that the training samples (x, y) are such that x ∈ R n is distributed according to N (0, 1) and label has no noise, that is, y = f (x).Note: Most proofs are deferred to the Appendix due to lack of space.

In this section we consider the case when P has a positive linear component and we wish to recover the parameters of true parameters W * .

The algorithm has two-steps: 1) uses existing one-hidden layer learning algorithm (SGD on carefully designed loss BID5 ) to recover an approximate solution , 2) refine the approximate solution by performing local search (for monotone P ).

The intuition behind the first step is that high thresholds enable P to in expectation be approximately close to a one-hidden-layer network which allows us to transfer algorithms with approximate guarantees.

Secondly, with the approximate solutions as starting points, we can evaluate the closeness of the estimate of each weight vector to the true weight vector using simple correlations.

The intuition of this step is to correlate with a function that is large only in the direction of the true weight vectors.

This equips us with a way to design a local search based algorithm to refine the estimate to small error.

For simplicity in this section we will work with P where the highest degree in any X i is 1.

The degree of the overall polynomial can still be n. See Appendix B.8 for the extension to general P .

More formally, Assumption 1 (Structure of network).

We assume that P has the following structure DISPLAYFORM0 to be the linear part of f .Next we will upper bound expected value of u(x): for "high-threshold" ReLU, that is, DISPLAYFORM1 2σ 2 (see Lemma 10).

We also get a lower bound on |û 4 | in terms of ρ (t, σ) 5 This enables us to make the following assumption.

Assumption 2.

Activation function u is a positive high threshold activation with threshold t, that is, the bias term is t. DISPLAYFORM2 where ρ is a positive decreasing function of t. Also, DISPLAYFORM3 Assumption 3 (Value of t).

t is large enough such that ρ(t, ||W DISPLAYFORM4 with for large enough constant η > 0 and p ∈ (0, 1].For example, for high threshold ReLU, ρ(t, 1) = e −t 2 /2 and µ = ρ(t, ||W DISPLAYFORM5 , thus t = √ 2η log d for large enough d suffices to get the above assumption (κ(W * ) is a constant).These high-threshold activation are useful for learning as in expectation, they ensure that f is close to f lin since the product terms have low expected value.

This is made clear by the following lemmas: Lemma 1.

For |S| > 1, under Assumption 2 we have, DISPLAYFORM6 .

Under Assumptions 1, 2 and 3, if t is such that dρ(t, ||W * ||) ≤ c for some small enough constant c > 0 we have, DISPLAYFORM7 Note: We should point out that f (x) and f lin (x) are very different point wise; they are just close in expectation under the distribution of x. In fact, if d is some constant then even the difference in expectation is some small constant.

This closeness suggests that algorithms for recovering under the labels from f lin can be used to recover with labels from f approximately.

Learning One Layer Neural Networks using Landscape Design.

BID5 proposed an algorithm for learning one-hidden-layer networks.

Intuitively, the approach of BID5 is to design a well behaved loss function based on correlations to recover the underlying weight vectors.

They show that the local minima of the following optimization corresponds to some transform of each of the w * i -thus it can be used to recover a transform of w * i , one at a time.

max DISPLAYFORM8 which they optimize using the Lagrangian formulation (viewed as a minimization): DISPLAYFORM9 where DISPLAYFORM10 and DISPLAYFORM11 (see Appendix A.1 for more details).

Using properties 4 We can handle DISPLAYFORM12 for some constant C by changing the scaling on t. 5 For similar bounds for sigmoid and sign refer to Appendix B.7.We previously showed that f is close to f lin in expectation due to the high threshold property.

This also implies that G lin and G are close and so are the gradients and (eignevalues of) hessians of the same.

This closeness implies that the landscape properties of one approximately transfers to the other function.

More formally, Theorem 4.

Let Z be an ( , τ )-local minimum of function A. If ||∇(B −A)(Z)|| ≤ ρ and ||∇ 2 (B − A)(Z)|| ≤ γ then Z is an ( + ρ, τ + γ)-local minimum of function B and vice-versa.

We will now apply above lemma on our G lin (z) and G(z).

DISPLAYFORM13 where w i are columns of (TW * ) −1 (ignoring log d factors).Note: For ReLU, setting t = √ C log d for large enough C > 0 we can get closeness 1/poly(d) to the columns of (TW * ) −1 .

Refer Appendix B.7 for details for sigmoid.

The paper BID5 also provides an alternate optimization that when minimized simultaneously recovers the entire matrix W * instead of having to learn columns of (TW * ) −1 separately.

We show how applying our methods can also be applied to that optimization in Appendix B.4 to recover W * by optimizing a single objective.

Assuming P is monotone, we can show that the approximate solution from the previous analysis can be refined to arbitrarily closeness using a random search method followed by approximately finding the angle of our current estimate to the true direction.

The idea at a high level is to correlate with δ (z T x − t) where δ is the Dirac delta function.

It turns out that the correlation is maximized when z is equal to one of the w i .

Correlation with δ (z T x−t) is checking how fast the correlation of f with δ(z T x−t) is changing as you change t. To understand this look at the case when our activation u is the sign function then note that correlation of u t (w T x − t) with δ (w T x − t) is very high as its correlation with δ(w T x − t ) is 0 when t < t and significant when t > t.

So as we change t' slightly from t − to t + there is a sudden increase.

If z and w differ then it can be shown that correlation of u t (w T x − t) with δ (z T x − t) essentially depends on cot(α) where α is the angle between w and z (for a quick intuition note that one can DISPLAYFORM0 .

See Lemma 16 in Appendix).

In the next section we will show how the same ideas work for non-monotone P even if it may not have any linear terms but we only manage to prove polynomial sample complexity for finding w instead of polynomial time complexity.

In this section we will not correlate exactly with δ (z T x − t) but instead we will use this high level idea to estimate how fast the correlation with δ(z T x − t ) changes between two specific values as one changes t , to get an estimate for cot(α).

Secondly since we can't to a smooth optimization over z, we will do a local search by using a random perturbation and iteratively check if the correlation has increased.

We can assume that the polynomial P doesn't have a constant term c 0 as otherwise it can easily be determined and cancelled out 6 .We will refine the weights one by one.

WLOG, let us assume that w * 1 = e 1 and we have z such that DISPLAYFORM1 Algorithm 1 RefineEstimate 1: Run EstimateT anAlpha on z to get s = tan(α) where α is the angle between z and w * 1 .

2: Perturb current estimate z by a vector along the d − 1 dimensional hyperplane normal to z with the distribution n(0, DISPLAYFORM2 Run EstimateT anAlpha on z to get s = tan(α ) where α is the angle between z and w * DISPLAYFORM3 Algorithm 2 EstimateTanAlpha 1: Find t 1 and t 2 such that P r[sgn(f (x))|x ∈ l(z, t , )] at t 1 is 0.4 and at t 2 is 0.6.

2: Return t2−t1 DISPLAYFORM4 The algorithm (Algorithm 1) estimates the angle of the current estimate with the true vector and then subsequently perturbs the vector to get closer after each successful iteration.

Theorem 5.

Given a vector z ∈ S d−1 such that it is 1/poly(d)-close to the underlying true vector DISPLAYFORM5 We prove the correctness of the algorithm by first showing that EstimateT anAlpha gives a multiplicative approximation to tan(α).

The following lemma captures this property.

Lemma 3.

EstimateT anAlpha(z) outputs y such that y = (1 ± O(η)) tan(α) where α is the angle between z and w * 1 .Proof.

We first show that the given probability when computed with sgn(x T w * 1 −t) is a well defined function of the angle between the current estimate and the true parameter up to multiplicative error.

Subsequently we show that the computed probability is close to the one we can estimate using f (x) since the current estimate is close to one direction.

The following two lemmas capture these properties.

Lemma 4.

For t, t and ≤ 1/t , we have DISPLAYFORM6 6 for example with RELU activation, f will be c0 most of the time as other terms in P will never activate.

So c0 can be set to say the median value of f .Using the above, we can show that, DISPLAYFORM7 where η 1 , η 2 > 0 are the noise due to estimating using f and DISPLAYFORM8 The following lemma bounds the range of t 1 and t 2 .Lemma 6.

We have 0 ≤ t 1 ≤ t 2 ≤ t cos(α1) .Thus, we have, DISPLAYFORM9 as long as η 2 +O( )t 2 ≤ c for some constant c > 0.

Thus, we can get a multiplicative approximation to tan(α) up to error η ( can be chosen to make its contribution smaller than η).Finally we show (proof in Appendix ??) that with constant probability, a random perturbation reduces the angle by a factor of (1 − 1/d) of the current estimate hence the algorithm will halt after DISPLAYFORM10 Lemma 7.

By applying a random Gaussian perturbation along the d − 1 dimensional hyperplane normal to z with the distribution n(0, Θ(α/d)) d−1 and scaling back to the unit sphere, with constant probability, the angle α (< π/2) with the fixed vector decreases by at least Ω(α/d).

We extend the methods of the previous section to a broader class of polynomials but only to obtain results in terms of sample complexity.

The main idea as in the previous section is to correlate with δ (z T x−t) (the derivative of the dirac delta function) and find arg max ||z||2=1 E[f (x)δ (z T x−t)].

We will show that the correlation goes to infinity when z is one of w * i and bounded if it is far from all of them.

From a practical standpoint we calculate δ (z T x − s) by measuring correlation with DISPLAYFORM0 , as in the previous section, for an even smaller ; however, for ease of exposition, in this section, we will assume that correlations with δ(z T x − s) can be measured exactly.

DISPLAYFORM1 If u = sgn then P has degree at most 1 in each X i .

Let ∂P ∂Xi denote the symbolic partial derivative of P with respect to X i ; so, it drops monomials without X i and factors off X i from the remaining ones.

Let us separate dependence on X i in P as follows: DISPLAYFORM2 We will overload the polynomial P such that P [x] to denote the polynomial computed by substituting X i = u((w * 1 ) T x) and similarly for Q and R. Under this notation f (x) = P [x].

We will also assume that |P (X)| ≤ ||X|| O(1) = ||X|| c1 (say).

By using simple correlations we will show: DISPLAYFORM3 ) samples one can determine the w * i 's within error 2 .

Note that if all the w * i 's are orthogonal then X i are independent and E Q i [x] (w * i ) T x = t is just value of Q i evaluated by setting X i = 1 and setting all the the remaining X j = µ where µ = E[X j ].

This is same as 1/µ times the coefficient of X i in P (µ(X 1 + 1), . . .

, µ(X d + 1)). ) one can determine W * within error 2 in each entry, if the coefficient of the linear terms in DISPLAYFORM0 The main point behind the proof of Theorem 6 is that the correlation is high when z is along one of w * i and negligible if it is not close to any of them.

DISPLAYFORM1 .

Otherwise if all angles α i between z and w * i are at least 2 it is at most DISPLAYFORM2 We will use the notation g(x) x=s to denote g(x) evaluated at x = s. Thus Cauchy's mean value theorem can be stated as g( DISPLAYFORM3 .

We will over load the notation a bit: φ(z T x = s) will denote the probability density that vz T x = s; so if z is a unit vector this is just φ(s); φ(z DISPLAYFORM4 denotes the probability density that both z DISPLAYFORM5 The following claim interprets correlation with δ(z T x − s) as the expected value along the corresponding plane z T x = s. DISPLAYFORM6 The following claim computes the correlation of P with δ (z T x − s).

DISPLAYFORM7 We use this to show that the correlation is bounded if all the angles are lower bounded.

Claim 5.

If P (X) ≤ ||X|| c1 and if z has an angle of at least 2 with all the w * DISPLAYFORM8 Above claims can be used to prove main Lemma 8.

Refer to the Appendix C for proofs.

Proof of Theorem 6.

If we wish to determine w * i within an angle of accuracy 2 let us set to be O( 3 2 φ(t)d −c ).

From Lemma 8, for some large enough c, this will ensure that if all α i > 2 the correlation is o(φ(t) 3 ).

Otherwise it is φ(t) 3 (1±o(1)).

Since φ(t) = poly(1/d), given poly( ) samples, we can test if a given direction is within accuracy 2 of a w * i or not.

Under additional structural assumptions on W * such as the weights being binary, that is, in {0, 1}, sparsity or certain restrictions on activation functions, we can give stronger recovery guarantees.

Proofs have been deferred to Appendix D.Theorem 7.

For activation u t (a) = e ρ(a−t) .

Let the weight vectors w * i be 0, 1 vectors that select the coordinates of x. For each i, there are exactly d indices j such that w ij = 1 and the coefficient of the linear terms in P (µ(X 1 + 1), µ(X 2 + 1), µ(X 3 + 1), ..) for µ = e −ρt is larger than the coefficient of all the product terms (constant factor gap) then we can learn the W * .In order to prove the above, we will construct a correlation graph over x 1 , . . .

, x n and subsequently identify cliques in the graph to recover w * i '

s.

With no threshold, recovery is still possible for disjoint, low l 1 -norm vector.

The proof uses simple correlations and shows that the optimization landscape for maximizing these correlations has local maximas being w * i '

s. Theorem 8.

For activation u(a) = e a .

If all w * i ∈ {0, 1} n are disjoint, then we can learn w * i as long as P has all positive coefficients and product terms have degree at most 1 in each variable.

For even activations, it is possible to recover the weight vectors even when the threshold is 0.

The technique used is the PCA like optimization using hermite polynomials as in Section 2.

Denote DISPLAYFORM0 Theorem 9.

If the activation is even and for every i, j: DISPLAYFORM1 u0û4 C({i, j},û 0 ) then there exists an algorithm that can recover the underlying weight vectors.

In this work we show how activations in a deep network that have a high threshold make it easier to learn the lowest layer of the network.

We show that for a large class of functions that represent the upper layers, the lowest layer can be learned with high precision.

Even if the threshold is low we show that the sample complexity is polynomially bounded.

An interesting open direction is to apply these methods to learn all layers recursively.

It would also be interesting to obtain stronger results if the high thresholds are only present at a higher layer based on the intuition we discussed.

Hermite polynomials form a complete orthogonal basis for the gaussian distribution with unit variance.

For more details refer to Wikipedia contributors (2018).

Let h i be the normalized hermite polynomials.

They satisfy the following, DISPLAYFORM0 This can be extended to the following:Fact 2.

For a, b with marginal distribution N (0, 1) and correlation ρ, DISPLAYFORM1 Consider the following expansion of u into the hermite basis (h i ), DISPLAYFORM2 Proof.

Observe that v T x and w T x have marginal distribution N (0, 1) and correlation v T w. Thus using Fact 2, DISPLAYFORM3 For gaussians with mean 0 and variance σ 2 define weighted hermite polynomials H σ l (a) = |σ| l h l (a/σ).

Given input v T x for x ∼ N (0, I), we suppress the superscript σ = ||v||.Corollary 2.

For a non-zero vector v (not necessarily unit norm) and a unit norm vector w, DISPLAYFORM4 Proof.

It follows as the proof of the previous lemma, DISPLAYFORM5

Consider matrix A ∈ R m×m .

Let σ i (A) to be the ith singular value of A such that DISPLAYFORM0 Fact 7.

Let B be a (mk) × (mk) principal submatrix of A, then κ(B) ≤ κ(A).

Lemma 10.

For u being a high threshold ReLU, that is, u t (a) = max(0, a − t) we have for t ≥ C for large enough constant DISPLAYFORM0 Proof.

We have DISPLAYFORM1 Also,û DISPLAYFORM2 To upper bound,û DISPLAYFORM3 Similar analysis holds forû 2 .Observe that sgn can be bounded very similarly replacing g − t by 1 which can affect the bounds up to only a polynomial in t factor.

Lemma 11.

For u being a high threshold sgn, that is, u t (a) = sgn(a − t) we have for t ≥ C for DISPLAYFORM4 For sigmoid, the dependence varies as follows: Lemma 12.

For u being a high threshold sigmoid, that is, u t (a) = 1 1+e −(a−t) we have for t ≥ C for large enough constant DISPLAYFORM5 Proof.

We have DISPLAYFORM6 Also,û DISPLAYFORM7 = Ω(e −t ).We can upper bound similarly and boundû 2 .

Let us consider the linear case with w * i 's are orthonormal.

Consider the following maximization problem for even l ≥ 4, max DISPLAYFORM0 where h l is the lth hermite polynomial.

Then we have, DISPLAYFORM1 It is easy to see that for z ∈ S n−1 , the above is maximized at exactly one of the w i 's (up to sign flip for even l) for l ≥ 3 as long as u l = 0.

Thus, each w i is a local minima of the above problem.

DISPLAYFORM2 For constraint ||z|| 2 = 1, we have the following optimality conditions (see BID11 for more details).

For all w = 0 such that w DISPLAYFORM3 For our function, we have: DISPLAYFORM4 The last follows from using the first order condition.

For the second order condition to be satisfied we will show that |S| = 1.

Suppose |S| > 2, then choosing w such that w i = 0 for i ∈ S and such that w T z = 0 (it is possible to choose such a value since |S| > 2), we get w T (∇ 2 L(z) − 2λI)w = 2(l − 2)λ||w|| 2 which is negative since λ < 0, thus these cannot be global minima.

However, for |S| = 1, we cannot have such a w, since to satisfy w T z = 0, we need w i = 0 for all i ∈ S, this gives us w T (∇ 2 L(z) − 2λI)w = −2λ||w|| 2 which is always positive.

Thus z = ±e i are the only local minimas of this problem.

Lemma 13 BID5 ).

If z is an ( , τ )-local minima of F (z) = − i α i z • (Derived from Proposition 5.7) z max = ±1 ± O(dτ /α min ) ± O( /λ) where |z| max is the value of the largest entry in terms of magnitude of z.

Proof of Lemma 1.

Let O ∈ R d×d be the orthonormal basis (row-wise) of the subspace spanned by w * i for all i ∈ [d] generated using Gram-schmidt (with the procedure done in order with elements of |S| first).

Now let O S ∈ R |S|×d be the matrix corresponding to the first S rows and let O ⊥ S ∈ R (d−|S|)×n be that corresponding to the remaining rows.

Note that OW * (W * also has the same ordering) is an upper triangular matrix under this construction.

DISPLAYFORM0 Now observe that O S W * S is also an upper triangular matrix since it is a principal sub-matrix of OW * .

Thus using Fact 6 and 7, we get the last equality.

Also, the single non-zero entry row has non-zero entry being 1 (||w * i || = 1 for all i).

This gives us that the inverse will also have the single non-zero entry row has non-zero entry being 1.

WLOG assume index 1 corresponds to this row.

Thus we can split this as following DISPLAYFORM1 Proof of Claim 1.

Consider the SVD of matrix M = UDU T .

Let W = UD −1/2 and y i = √ c i W T w * i for all i.

It is easy to see that y i are orthogonal.

Let F (z) = G(Wz): DISPLAYFORM2 Since y i are orthogonal, for means of analysis, we can assume that y i = e i , thus the formulation reduces to max z |û 4 | i 1 ci (z i ) 4 − λ ||z|| 2 − 1 2 up to scaling of λ = λû 2 2 .

Note that this is of the form in Lemma 13 hence using that we can show that the approximate local minimas of F (z) are close to y i and thus the local maximas of G(z) are close to DISPLAYFORM3 due to the linear transformation.

This can alternately be viewed as the columns of (TW DISPLAYFORM4 Proof of Theorem 4.

Let Z be an ( , τ )-local minimum of A, then we have ||∇A(Z)|| ≤ and DISPLAYFORM5 Also observe that DISPLAYFORM6 Here we use |λ min (M)| ≤ ||M|| for any symmetric matrix.

To prove this, we have ||M|| = max x∈S n−1 ||Mx||.

We have x = i x i v i where v i are the eigenvectors.

Thus we have Mx = DISPLAYFORM7 Proof of Lemma 2.

Expanding f , we have DISPLAYFORM8 Proof.

We have DISPLAYFORM9

, for c = Θ( √ η log d we get the required result.

Lemma 15.

For ||z|| = Ω(1) and λ = Θ(|û 4 |/û DISPLAYFORM0 Proof.

Let K = κ(W * ) which by assumption is θ(1).

We will argue that local minima of G cannot have z with large norm.

First lets argue this for G lin (z).

We know that DISPLAYFORM1 2 where α = |û 4 | and β =û 2 .

We will argue that z T ∇G lin (z) is large if z is large.

DISPLAYFORM2 Let y = W * z then K||z|| ≥ ||y|| ≥ ||z||/K since K is the condition number of W * .

Then this implies DISPLAYFORM3 Now we need to argue for G. DISPLAYFORM4 We know that E[f lin (x)h 2 (z T x/||z||)] has a factor of β giving us using Lemma 14: DISPLAYFORM5 Proof of Claim 2.

We have G − G lin as follows, DISPLAYFORM6 Thus we have, DISPLAYFORM7 Observe that H 2 and H 4 are degree 2 and 4 (respectively) polynomials thus norm of gradient and hessian of the same can be bounded by at most O(||z||||x|| 4 ).

Using Lemma 14 we can bound each term by roughly O(log d)d −(1+p)η+3 ||z|| 4 .

Note that λ being large does not hurt as it is scaled appropriately in each term.

Subsequently, using Lemma 15, we can show that ||z|| is bounded by a constant since ||G(z)|| ≤ d −2η .

Similar analysis holds for the hessian too.

≥ .

Now using Claim 1, we get the required result.

BID5 also showed simultaneous recovery by minimizing the following loss function G lin defined below has a well-behaved landscape.

They gave the following result.

Theorem 10 (Ge et al. (2017) We show that this minimization is robust.

Let us consider the corresponding function G to G lin with the additional non-linear terms as follows: DISPLAYFORM0 Now we can show that G and G lin are close as in the one-by-one case.

DISPLAYFORM1 Using similar analysis as the one-by-one case, we can show the required closeness.

It is easy to see that ||∇L|| and ||∇ 2 L|| will be bounded above by a constant degree polynomial in DISPLAYFORM2 No row can have large weight as if any row is large, then looking at the gradient for that row, it reduces to the one-by-one case, and there it can not be larger than a constant.

Thus we have the same closeness as in the one-by-one case.

Combining this with Theorem 10 and 4, we have the following theorem:Theorem 11.

Let c be a sufficiently small universal constant (e.g. c = 0.01 suffices), and under Assumptions 1, 2 and 3.

Assume γ ≤ c, λ = Θ(d η ), and W * be the true weight matrix.

The function G satisfies the following 1.

Any saddle point W has a strictly negative curvature in the sense that DISPLAYFORM3 −Ω(1) )-approximate local minimum, then W can be written as DISPLAYFORM4 )}, P is a permutation matrix, and the error term ||E|| ≤ O(log d)d−Ω(1) .Using standard optimization techniques we can find a local minima.

Lemma 16.

If u is the sign function then E[u(w T x)δ (z T x)] = c| cot(α)| where w, z are unit vectors and α is the angle between them and c is some constant.

Proof.

WLOG we can work the in the plane spanned by z and w and assume that z is the vector i along and w = i cos α + j sin α.

Thus we can replace the vector x by ix + jy where x, y are normally distributed scalars.

Also note that u = δ (Dirac delta function).

DISPLAYFORM0 Using the fact that x δ (x)h(x)dx = h (0) this becomes DISPLAYFORM1 Substituting s = y sin α this becomes DISPLAYFORM2 Proof of Lemma 4.

Let us compute the probability of lying in the -band for any t: DISPLAYFORM3 where the last equality follows from the mean-value theorem for somet ∈ [t − , t].Next we compute the following: DISPLAYFORM4 where the last equality follows from the mean-value theorem for some t * ∈ [t − , t ].

Combining, we get: DISPLAYFORM5 Proof of Lemma 5.

Recall that P is monotone with positive linear term, thus for high threshold u (0 unless input exceeds t and positive after) we have sgn(f (x)) = ∨sgn(x T w * i − t).

This is because, for any i, P applied to X i > 0 and ∀j = i, X j = 0 gives us c i which is positive.

Also, P (0) = 0.

Thus, sgn(P ) is 1 if any of the inputs are positive.

Using this, we have, DISPLAYFORM6 We will show that η is not large since a z is close to one of the vectors, it can not be close to the others thus α i will be large for all i =

j. Let us bound η, DISPLAYFORM7 DISPLAYFORM8 | sin(αi)| .

The above follows since γ i ≥ 0 by assumption on t .

Under the assumption, let β = max i =1 cos(α i ) we have DISPLAYFORM9 under our setting.

Thus we have, DISPLAYFORM10 for small enough .Proof of Lemma 6.

Let us assume that < c/t for sufficiently small constant c, then we have that DISPLAYFORM11 Similarly for t 1 .

Now we need to argue that t 1 , t 2 ≥ 0.

Observe that DISPLAYFORM12 Thus for sufficiently large t = Ω( √ log d), this will be less than 0.4.

Hence there will be some t 1 , t 2 ≥ 0 with probability evaluating to 0.4 since the probability is an almost increasing function of t up to small noise in the given range (see proof of Lemma 5).Proof of Lemma 7.

Let V be the plane spanned by w * 1 and z and let v 1 = w * 1 and v 2 be the basis of this space.

Thus, we can write z = cos(α)v 1 + sin(α)v 2 .Let us apply a Gaussian perturbation ρ along the tangential hyperplane normal to z. Say it has distribution N (0, 1) along any direction tangential to the vector z. Let 1 be the component of ρ on to V and let 2 be the component perpendicular to it.

We can write the perturbation as ρ = 1 (sin(α)v 1 − cos(α)v 2 ) + 2 v 3 where v 3 is orthogonal to both v 1 and v 2 .

So the new angle α of z after the perturbation is given by DISPLAYFORM13 Note that with constant probability 1 ≥ as ρ is a Gaussian variable with standard deviation .

And with high probability ||ρ|| < O( DISPLAYFORM14 Thus with constant probability: DISPLAYFORM15 Thus change in cos(α) is given by ∆ cos(α) ≥ Ω( sin(α)).

Now change in the angle α satisfies by the Mean Value Theorem: DISPLAYFORM16

Theorem 12.

Given non-noisy labels from a union of halfspaces that are at a distance Ω( √ log d) and are each a constant angle apart, there is an algorithm to recover the underlying weights to closeness in polynomial time.

Proof.

Observe that X i is equivalent to P (X 1 , ·, DISPLAYFORM0 Since P and sgn here satisfies our assumptions 1, 2, for t = Ω( √ log d) (see Lemma 11)

we can apply Theorem 11 to recover the vectors w * i approximately.

Subsequently, refining to arbitrarily close using Theorem 5 is possible due to the monotonicity.

Thus we can recover the vectors to arbitrary closeness in polynomial time.

Observe that for sigmoid activation, Assumption 2 is satisfied for ρ(t, σ) = e −t+σ 2 /2 .

Thus to satisfy Assumption 3, we need t = Ω(η log d).Note that for such value of t, the probability of the threshold being crossed is small.

To avoid this we further assume that f is non-negative and we have access to an oracle that biases the samples towards larger values of f ; that after x is drawn from the Gaussian distribution, it retains the sample (x, f (x)) with probability proportional to f (x) -so P r [x] in the new distribution.

This enables us to compute correlations even if E xÑ (0,I [f (x)] is small.

In particular by computing E[h(x)] from this distribution, we are obtaining E[f (x)h(x)]/E[f (x)] in the original distribution.

Thus we can compute correlations that are scaled.

We get our approximate theorem: Theorem 13.

For t = Ω(log d), columns of (TW * ) −1 can be recovered within error 1/poly(d) using the algorithm in polynomial time.

In the main section we assumed that the polynomial has degree at most 1 in each variable.

Let us give a high level overview of how to extend this to the case where each variable is allowed a large degree.

P now has the following structure, DISPLAYFORM0 If P has a higher degree in X i then Assumption 2 changes to a more complex (stronger) condition.

Let q i (x) = r∈Z d + |∀j =i,rj =0 c r x ri , that is q i is obtained by setting all X j for j = i to 0.

DISPLAYFORM1 The last assumption holds for the case when the degree is a constant and each coefficient is upper bounded by a constant.

It can hold for decaying coefficients.

Let us collect the univariate terms DISPLAYFORM2 Corresponding to the same we get f uni .

This will correspond to the f lin we had before.

Note that the difference now is that instead of being the same activation for each weight vector, now we have different ones q i for each.

Using H 4 correlation as before, now we get that: DISPLAYFORM3 where q i •

u t are hermite coefficients for q i • u t .

Now the assumption guarantees that these are positive which is what we had in the degree 1 case.

Second we need to show that even with higher degree, E[|f (x) − f uni (x)|] is small.

Observe that Lemma 17.

For r such that ||r|| 0 > 1, under Assumption 4 we have, DISPLAYFORM4 The proof essentially uses the same idea, except that now the dependence is not on ||r|| 1 but only the number of non-zero entries (number of different weight vectors).

With this bound, we can now bound the deviation in expectation.

DISPLAYFORM5 Proof.

We have, DISPLAYFORM6 |c r |ρ(t, 1) (ρ(t, ||W * ||)) DISPLAYFORM7 Thus as before, if we choose t appropriately, we get the required results.

Similar ideas can be used to extend to non-constant degree under stronger conditions on the coefficients.

Proof of Lemma 3.

DISPLAYFORM0 dx Let x 0 be the component of x along z and y be the component along z ⊥ .

So x = x 0ẑ + yz ⊥ .

Interpreting x as a function of x 0 and y: DISPLAYFORM1 where the second equality follows from DISPLAYFORM2 Proof of Claim 4.

Let x 0 be the component of x along z and y be the component of x in the space orthogonal to z. Letẑ denote a unit vector along z.

We have x = x 0ẑ + y and ∂x ∂x0 =ẑ.

So, correlation can be computed as follows: DISPLAYFORM3 ](x = a) this implies: DISPLAYFORM4 If u is the sign function then u (x) = δ(x).

So focusing on one summand in the sum we get DISPLAYFORM5 Again let y = y 0 (w * i ) + z where z is perpendicular to w * i and z. And (w * i ) is perpendicular component of w * i to z. Interpreting x = tẑ + y 0 (w * 1 ) + z as a function of y 0 , z we get: DISPLAYFORM6 Note that by substituting v = ax we get DISPLAYFORM7 .

So this becomes: DISPLAYFORM8 Under review as a conference paper at ICLR 2019 DISPLAYFORM9 Let α i be the angle between z and w * i .

Then this is DISPLAYFORM10 Thus, overall correlation DISPLAYFORM11 Proof of Claim 5.

Note that for small α, DISPLAYFORM12 which is a decreasing function of α i in the range [0, π]

So if all α i are upper bounded by 2 then by above corollary, DISPLAYFORM13 Observe that the above proof does not really depend on P and holds for for any polynomial of u((w * i ) T x) as long as the polynomial is bounded and the w * i are far off from z. DISPLAYFORM14 Since u((w * i ) T x) = 0 for z T x = t − and 1 for z T x = t + , and using the Cauchy mean value theorem for the second term this is DISPLAYFORM15 The last step follows from Claim 5 applied on Q i and R i as all the directions of w * j are well separated from z = w * i and w * i is absent from both Q i and R i .

Also the corresponding Q i and R i are bounded.

If u is the RELU activation, the high level idea is to use correlation with the second derivative δ of the Dirac delta function instead of δ .

More precisely we will compute DISPLAYFORM0 Although we show the analysis only for the RELU activation, the same idea works for any activation that has non-zero derivative at 0.Note that now u = sgn and u = δ.

For ReLU activation, Lemma 8 gets replaced by the following Lemma.

The rest of the argument is as for the sgn activation.

We will need to assume that P has constant degree and sum of absolute value of all coefficients is poly(d) Lemma 19.

Assuming polynomial P has constant degree, and sum of the magnitude of all coefficients is at most DISPLAYFORM1 .

Otherwise if all angles α i between z and w i are at least 2 it is at most DISPLAYFORM2 We will prove the above lemma in the rest of this section.

First we will show that z is far from any of the w * DISPLAYFORM3 Lemma 20.

If the sum of the absolute value of the coefficients of P is bounded by poly(d), its degree is at most constant, DISPLAYFORM4 Proof.

Let x 0 be the component of x along z and y be the component of x in the space orthogonal to z as before.

We have x = x 0ẑ + y and ∂x ∂x0 =ẑ.

We will look at monomials M l in P = l M l .

As before since x δ (x − a)f (x)dx = To construct this correlation graph, we will run the following Algorithm 3 Denote T i := {j : w ij = 1}. Let us compute E[f (x)x i x j ]: DISPLAYFORM5 c S e −ρt|S| E e ρ p∈S x T w * p x i x j Lemma 22.

At the local maxima, for all i ∈ [n], z is such that for all j, k ∈ S i , z j = z k at local maxima.

Proof.

We prove by contradiction.

Suppose there exists j, k such that z j < z k .

Consider the following perturbation: z + (z k − z j )(e j − e k ) for 1 ≤ > 0.

Observe that g(z) depends on only r∈Si z r and since that remains constant by this update g(z) does not change.

Also note that ||z|| 1 does not change.

However ||z|| 2 2 decreases by 2 (1 − )(z k − z j ) 2 implying that overall h(z) increases.

Thus there is a direction of improvement and thus it can not be a local maxima.

Lemma 23.

At the local maxima, ||z|| 1 ≥ α for λ < S c S |∪ i∈S Si| n e |∪ i∈S S i | 2 − γ(2α + 1).Proof.

We prove by contradiction.

Suppose ||z|| 1 < α, consider the following perturbation, z + 1.Then we have h(z + 1) − h(z) = c S e |∪ i∈S S i | 2 + p∈∪ i∈S S izp (e |∪ i∈S Si| − 1) − nλ − nγ (2||z|| 1 + ) > c S e |∪ i∈S S i | 2 | ∪ i∈S S i | − nλ − nγ (2α + 1)For given λ there is a direction of improvement giving a contradiction that this is the local maxima.

Combining the above, we have that we can choose λ, γ = poly(n, 1/ , s) where s is a paramater that depends on structure of f such that at any local maxima there exists i such that for all j ∈ S i , z j ≥ 1 and for all k ∈ ∪ j∈Si , z k ≤ .

Let us consider correlation with h 4 (z T x).

This above can be further simplified by observing that when we correlate with h 4 , i∈S u (x i )h 4 (z T x) = 0 for |S | ≥ 2.

Observe that h 4 (z T x) = d1,...,dn∈[4]: di≤4 c(d 1 , . . . , d n ) h di (x i ) for some coefficients c which are functions of z. Thus when we correlate i∈S u (x i )h 4 (z T x) for |S | ≥ 3 then we can only get a non-zero term if we have at least h 2k (x i ) with k ≥ 1 for all i ∈ S .

This is not possible for |S | ≥ 3, hence, these terms are 0.

Thus, DISPLAYFORM6 Lets compute these correlations.

DISPLAYFORM7

<|TLDR|>

@highlight

We provably recover the lowest layer in a deep neural network assuming that the lowest layer uses a "high threshold" activation and the above network is a "well-behaved" polynomial.
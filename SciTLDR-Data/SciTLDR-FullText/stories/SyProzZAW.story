It is well-known that neural networks are universal approximators, but that deeper networks tend in practice to be more powerful than shallower ones.

We shed light on this by proving that the total number of neurons m required to approximate natural classes of multivariate polynomials of n variables grows only linearly with n for deep neural networks, but grows exponentially when merely a single hidden layer is allowed.

We also provide evidence that when the number of hidden layers is increased from 1 to k, the neuron requirement grows exponentially not with n but with n^{1/k}, suggesting that the minimum number of layers required for practical expressibility grows only logarithmically with n.

Deep learning has lately been shown to be a very powerful tool for a wide range of problems, from image segmentation to machine translation.

Despite its success, many of the techniques developed by practitioners of artificial neural networks (ANNs) are heuristics without theoretical guarantees.

Perhaps most notably, the power of feedforward networks with many layers (deep networks) has not been fully explained.

The goal of this paper is to shed more light on this question and to suggest heuristics for how deep is deep enough.

It is well-known BID7 BID11 BID15 BID1 BID23 that neural networks with a single hidden layer can approximate any function under reasonable assumptions, but it is possible that the networks required will be extremely large.

Recent authors have shown that some functions can be approximated by deeper networks much more efficiently (i.e. with fewer neurons) than by shallower ones.

Often, these results admit one or more of the following limitations: "existence proofs" without explicit constructions of the functions in question; explicit constructions, but relatively complicated functions; or applicability only to types of network rarely used in practice.

It is important and timely to extend this work to make it more concrete and actionable, by deriving resource requirements for approximating natural classes of functions using today's most common neural network architectures.

BID17 recently proved that it is exponentially more efficient to use a deep network than a shallow network when Taylor-approximating the product of input variables.

In the present paper, we move far beyond this result in the following ways: (i) we use standard uniform approximation instead of Taylor approximation, (ii) we show that the exponential advantage of depth extends to all general sparse multivariate polynomials, and (iii) we address the question of how the number of neurons scales with the number of layers.

Our results apply to standard feedforward neural networks and are borne out by empirical tests.

Our primary contributions are as follows:• It is possible to achieve arbitrarily close approximations of simple multivariate and univariate polynomials with neural networks having a bounded number of neurons (see §3).• Such polynomials are exponentially easier to approximate with deep networks than with shallow networks (see §4).• The power of networks improves rapidly with depth; for natural polynomials, the number of layers required is at most logarithmic in the number of input variables, where the base of the logarithm depends upon the layer width (see §5).

Deeper networks have been shown to have greater representational power with respect to various notions of complexity, including piecewise linear decision boundaries BID22 and topological invariants BID2 .

Recently, and showed that the trajectories of input variables attain exponentially greater length and curvature with greater network depth.

Work including BID8 ; BID10 ; BID23 ; BID24 ; Telgarsky (2016) shows that there exist functions that require exponential width to be approximated by a shallow network.

BID1 provides bounds on the error in approximating general functions by shallow networks.

BID20 and BID24 show that for compositional functions (those that can be expressed by recursive function composition), the number of neurons required for approximation by a deep network is exponentially smaller than the best known upper bounds for a shallow network.

BID20 ask whether functions with tight lower bounds must be pathologically complicated, a question which we answer here in the negative.

Various authors have also considered the power of deeper networks of types other than the standard feedforward model.

The problem has also been posed for sum-product networks BID9 and restricted Boltzmann machines BID19 .

showed, using tools from tensor decomposition, that shallow arithmetic circuits can express only a measure-zero set of the functions expressible by deep circuits.

A weak generalization of this result to convolutional neural networks was shown in .

In this paper, we will consider the standard model of feedforward neural networks (also called multilayer perceptrons).

Formally, the network may be considered as a multivariate function DISPLAYFORM0 . .

, A k are constant matrices and σ denotes a scalar nonlinear function applied element-wise to vectors.

The constant k is referred to as the depth of the network.

The neurons of the network are the entries of the vectors σ(A · · · σ(A 1 σ(A 0 x)) · · · ), for = 1, . . .

, k − 1.

These vectors are referred to as the hidden layers of the network.

Two notions of approximation will be relevant in our results: -approximation, also known as uniform approximation, and Taylor approximation.

Definition 3.1.

For constant > 0, we say that a network N (x) -approximates a multivariate function f (x) (for x in a specified domain DISPLAYFORM1 Definition 3.2.

We say that a network N (x) Taylor-approximates a multivariate polynomial p(x) of degree d if p(x) is the dth order Taylor polynomial (about the origin) of N (x).The following proposition shows that Taylor approximation implies -approximation for homogeneous polynomials.

The reverse implication does not hold.

Proposition 3.3.

Suppose that the network N (x) Taylor-approximates the homogeneous multivariate polynomial p(x).

Then, for every , there exists a network N (x) that -approximates p(x), such that N (x) and N (x) have the same number of neurons in each layer. (This statement holds for x ∈ (−R, R) n for any specified R.) DISPLAYFORM2 is a Taylor series with each E i (x) homogeneous of degree i.

Since N (x) is the function defined by a neural network, it converges for every x ∈ R n .

Thus, E(x) converges, as does DISPLAYFORM3 .

By picking δ sufficiently small, we can make each term DISPLAYFORM4 d , and therefore: DISPLAYFORM5 We conclude that N (x) is an -approximation of p(x), as desired.

For a fixed nonlinear function σ, we consider the total number of neurons (excluding input and output neurons) needed for a network to approximate a given function.

Remarkably, it is possible to attain arbitrarily good approximations of a (not necessarily homogeneous) multivariate polynomial by a feedforward neural network, even with a single hidden layer, without increasing the number of neurons past a certain bound.

(See also Corollary 1 in BID24 .)

Theorem 3.4.

Suppose that p(x) is a degree-d multivariate polynomial and that the nonlinearity σ has nonzero Taylor coefficients up to degree d. Let m k (p) be the minimum number of neurons in a depth-k network that -approximates p.

Then, the limit lim →0 m k (p) exists (and is finite). (Once again, this statement holds for x ∈ (−R, R) n for any specified R.)Proof.

We show that lim →0 m 1 (p) exists; it follows immediately that lim →0 m k (p) exists for every k, since an -approximation to p with depth k can be constructed from one with depth 1.

DISPLAYFORM6 .

We claim that each p i (x) can be Taylor-approximated by a network N i (x) with one hidden layer.

This follows, for example, from the proof in BID17 that products can be Taylor-approximated by networks with one hidden layer, since each monomial is the product of several inputs (with multiplicity); we prove a far stronger result about N i (x) later in this paper (see Theorem 4.1).Suppose now that N i (x) has m i hidden neurons.

By Proposition 3.3, we conclude that since p i (x) is homogeneous, it may be δ-approximated by a network N DISPLAYFORM7 This theorem is perhaps surprising, since it is common for -approximations to functions to require ever-greater complexity, approaching infinity as → 0.

For example, the function exp(| − x|) may be approximated on the domain (−π, π) by Fourier sums of the form m k=0 a m cos(kx).

However, in order to achieve -approximation, we need to take m ∼ 1/ √ terms.

By contrast, we have shown that a finite neural network architecture can achieve arbitrarily good approximations merely by altering its weights.

Note also that the assumption of nonzero Taylor coefficients cannot be dropped from Theorem 3.4.

For example, the theorem is false for rectified linear units (ReLUs), which are piecewise linear and do not admit a Taylor series.

This is because -approximating a non-linear polynomial with a piecewise linear function requires an ever-increasing number of pieces as → 0.

In this section, we compare the efficiency of shallow networks (those with a single hidden layer) and deep networks at approximating multivariate polynomials.

Proofs of our main results are included in the Appendix.

Our first result shows that uniform approximation of monomials requires exponentially more neurons in a shallow than a deep network.

DISPLAYFORM0 Suppose that the nonlinearity σ has nonzero Taylor coefficients up to degree 2d.

Then, we have: DISPLAYFORM1 where x denotes the smallest integer that is at least x.

We can prove a comparable result for m Taylor under slightly weaker assumptions on σ.

Note that by setting r 1 = r 2 = . . .

= r n = 1, we recover the result of BID17 that the product of n numbers requires 2 n neurons in a shallow network but can be Taylor-approximated with linearly many neurons in a deep network.

DISPLAYFORM2 It is worth noting that neither of Theorems 4.1 and 4.2 implies the other.

This is because it is possible for a polynomial to admit a compact uniform approximation without admitting a compact Taylor approximation.

It is natural now to consider the cost of approximating general polynomials.

However, without further constraint, this is relatively uninstructive because polynomials of degree d in n variables live within a space of dimension n+d d , and therefore most require exponentially many neurons for any depth of network.

We therefore consider polynomials of sparsity c: that is, those that can be represented as the sum of c monomials.

This includes many natural functions.

Theorem 4.3.

Let p(x) be a multivariate polynomial of degree d and sparsity c, having monomials q 1 (x), q 2 (x), . . .

, q c (x).

Suppose that the nonlinearity σ has nonzero Taylor coefficients up to degree 2d.

Then, we have: DISPLAYFORM3 These statements also hold if m uniform is replaced with m Taylor .As mentioned above with respect to ReLUs, some assumptions on the Taylor coefficients of the activation function are necessary for the results we present.

However, it is possible to loosen the assumptions of Theorem 4.1 and 4.2 while still obtaining exponential lower bounds on m DISPLAYFORM4 Hence, A is invertible, which means that multiplying its columns by nonzero values gives another invertible matrix.

Suppose that we multiply the jth column of A by σ j to get A , where σ(x) = j σ j x j is the Taylor expansion of σ(x).

Now, observe that the ith row of A is exactly the coefficients of σ(a i x), up to the degree-d term.

Since A is invertible, the rows must be linearly independent, so the polynomials σ(a i x), restricted to terms of degree at most d, must themselves be linearly independent.

Since the space of degree-d univariate polynomials is (d + 1)-dimensional, these d + 1 linearly independent polynomials must span the space.

Hence, m Taylor 1 (p) ≤ d + 1 for any univariate degree-d polynomial p.

In fact, we can fix the weights from the input neuron to the hidden layer (to be a 0 , a 1 , . . .

, a d , respectively) and still represent any polynomial p with d + 1 hidden neurons.

Proposition 4.6.

Let p(x) = x d , and suppose that the nonlinearity σ(x) has nonzero Taylor coefficients up to degree 2d.

Then, we have: DISPLAYFORM5 These statements also hold if m uniform is replaced with m Taylor .Proof.

Part (i) follows from part (i) of Theorems 4.1 and 4.2 by setting n = 1 and r 1 = d.

For part (ii), observe that we can Taylor-approximate the square x 2 of an input x with three neurons in a single layer: DISPLAYFORM6 We refer to this construction as a square gate, and the construction of Lin et al. FORMULA15 as a product gate.

We also use identity gate to refer to a neuron that simply preserves the input of a neuron from the preceding layer (this is equivalent to the skip connections in residual nets BID14

We now consider how m uniform k (p) scales with k, interpolating between exponential in n (for k = 1) and linear in n (for k = log n).

In practice, networks with modest k > 1 are effective at representing natural functions.

We explain this theoretically by showing that the cost of approximating the product polynomial drops off rapidly as k increases.

By repeated application of the shallow network construction in Lin et al. FORMULA15 , we obtain the following upper bound on m uniform k (p), which we conjecture to be essentially tight.

Our approach leverages the compositionality of polynomials, as discussed e.g. in BID20 and BID24 , using a tree-like neural network architecture.

Theorem 5.1.

Let p(x) equal the product x 1 x 2 · · · x n , and suppose σ has nonzero Taylor coefficients up to degree n. Then, we have: DISPLAYFORM0 Proof.

We construct a network in which groups of the n inputs are recursively multiplied up to Taylor approximation.

The n inputs are first divided into groups of size b 1 , and each group is multiplied in the first hidden layer using 2 b1 neurons (as described in Lin et al. FORMULA15 ).

Thus, the first hidden layer includes a total of 2 b1 n/b 1 neurons.

This gives us n/b 1 values to multiply, which are in turn divided into groups of size b 2 .

Each group is multiplied in the second hidden layer using 2 b2 neurons.

Thus, the second hidden layer includes a total of 2 b2 n/(b 1 b 2 ) neurons.

We continue in this fashion for b 1 , b 2 , . . .

, b k such that b 1 b 2 · · · b k = n, giving us one neuron which is the product of all of our inputs.

By considering the total number of neurons used, we conclude In fact, we can solve for the choice of b i such that the upper bound in (2) is minimized, under the condition b 1 b 2 · · · b k = n. Using the technique of Lagrange multipliers, we know that the optimum occurs at a minimum of the function as n varies are shown for k = 1, 2, 3.

Observe that the b i converge to n 1/k for large n, as witnessed by a linear fit in the log-log plot.

The exact values are given by equations (4) and (5).

for n = 20 is shown in black.

In the region above and to the right of the curve, it is possible to effectively approximate the product function (Theorem 5.1).

DISPLAYFORM1 DISPLAYFORM2 Differentiating L with respect to b i , we obtain the conditions DISPLAYFORM3 Dividing (3) by k j=i+1 b j and rearranging gives us the recursion DISPLAYFORM4 Thus, the optimal b i are not exactly equal but very slowly increasing with i (see FIG6 ).The following conjecture states that the bound given in Theorem 5.1 is (approximately) optimal.

Conjecture 5.2.

Let p(x) equal to the product x 1 x 2 · · · x n , and suppose that σ has all nonzero Taylor coefficients.

Then, we have: DISPLAYFORM5 i.e., the exponent grows as n 1/k for n → ∞.We empirically tested Conjecture 5.2 by training ANNs to predict the product of input values x 1 , . . .

, x n with n = 20 (see FIG7 .

The rapid interpolation from exponential to linear width aligns with our predictions.

In our experiments, we used feedforward networks with dense connections between successive layers.

In the figure, we show results for σ(x) = tanh(x) (note that this behavior is even better than expected, since this function actually has numerous zero Taylor coefficients).

Similar results were also obtained for rectified linear units (ReLUs) as the nonlinearity, despite the fact that this function does not even admit a Taylor series.

The number of layers was varied, as was the number of neurons within a single layer.

The networks were trained using the AdaDelta optimizer (Zeiler, 2012) to minimize the absolute value of the difference between the predicted and actual values.

Input variables x i were drawn uniformly at random from the interval [0, 2], so that the expected value of the output would be of manageable size.

Eq.

(6) provides a helpful rule of thumb for how deep is deep enough.

Suppose, for instance, that we wish to keep typical layers no wider than about a thousand (∼ 2 10 ) neurons.

Eq. (6) then implies n 1/k ∼ < 10, i.e., that the number of layers should be at least k ∼ > log 10 n.

It would be very interesting if one could show that general polynomials p in n variables require a superpolynomial number of neurons to approximate for any constant number of hidden layers.

The analogous statement for Boolean circuits -whether the complexity classes T C 0 and T C 1 are equal -remains unresolved and is assumed to be quite hard.

Note that the formulations for Boolean circuits and deep neural networks are independent statements (neither would imply the other) due to the differences between computation on binary and real values.

Indeed, gaps in expressivity have already been proven to exist for real-valued neural networks of different depths, for which the analogous results remain unknown in Boolean circuits (see e.g. BID21 BID4 BID3 ; Montufar et al. FORMULA15 ; ; Telgarsky FORMULA15 ).

We have shown how the power of deeper ANNs can be quantified even for simple polynomials.

We have proved that arbitrarily good approximations of polynomials are possible even with a fixed number of neurons and that there is an exponential gap between the width of shallow and deep networks required for approximating a given sparse polynomial.

For n variables, a shallow network requires size exponential in n, while a deep network requires at most linearly many neurons.

Networks with a constant number k > 1 of hidden layers appear to interpolate between these extremes, following a curve exponential in n 1/k .

This suggests a rough heuristic for the number of layers required for approximating simple functions with neural networks.

For example, if we want no layers to have more than 2 10 neurons, say, then the minimum number of layers required grows only as log 10 n. To further improve efficiency using the O(n) constructions we have presented, it suffices to increase the number of layers by a factor of log 2 10 ≈ 3, to log 2 n.

The key property we use in our constructions is compositionality, as detailed in BID24 .

It is worth noting that as a consequence our networks enjoy the property of locality mentioned in , which is also a feature of convolutional neural nets.

That is, each neuron in a layer is assumed to be connected only to a small subset of neurons from the previous layer, rather than the entirety (or some large fraction).

In fact, we showed (e.g. Prop.

4.6) that there exist natural functions computable with linearly many neurons, with each neuron is connected to at most two neurons in the preceding layer, which nonetheless cannot be computed with fewer than exponentially many neurons in a single layer, no matter how may connections are used.

Our construction can also be framed with reference to the other properties mentioned in : those of sharing (in which weights are shared between neural connections) and pooling (in which layers are gradually collapsed, as our construction essentially does with recursive combination of inputs).

This paper has focused exclusively on the resources (neurons and synapses) required to compute a given function for fixed network depth.

(Note also results of BID18 ; BID13 ; BID12 for networks of fixed width.)

An important complementary challenge is to quantify the resources (e.g. training steps) required to learn the computation, i.e., to converge to appropriate weights using training data -possibly a fixed amount thereof, as suggested in Zhang et al. (2017) .

There are simple functions that can be computed with polynomial resources but require exponential resources to learn (Shalev-Shwartz et al., 2017) .

It is quite possible that architectures we have not considered increase the feasibility of learning.

For example, residual networks (ResNets) BID14 and unitary nets (see e.g. BID0 BID16 ) are no more powerful in representational ability than conventional networks of the same size, but by being less susceptible to the "vanishing/exploding gradient" problem, it is far easier to optimize them in practice.

We look forward to future work that will help us understand the power of neural networks to learn.

Without loss of generality, suppose that r i > 0 for i = 1, . . .

, n. Let X be the multiset in which x i occurs with multiplicity r i .We first show that n i=1 (r i + 1) neurons are sufficient to approximate p(x).

Appendix A in Lin et al. (2017) demonstrates that for variables y 1 , . . .

, y N , the product y 1 · · · · ·

y N can be Taylorapproximated as a linear combination of the 2 N functions σ(±y 1 ± · · · ± y d ).Consider setting y 1 , . . .

, y d equal to the elements of multiset X. Then, we conclude that we can approximate p(x) as a linear combination of the functions σ(±y 1 ± · · · ± y d ).

However, these functions are not all distinct: there are r i + 1 distinct ways to assign ± signs to r i copies of x i (ignoring permutations of the signs).

Therefore, there are DISPLAYFORM0

We now show that this number of neurons is also necessary for approximating p(x).

Suppose that N (x) is an -approximation to p(x) with depth 1, and let the Taylor series of N (x) be p(x)+E(x).

Let E k (x) be the degree-k homogeneous component of E(x), for 0 ≤ k ≤ 2d.

By the definition of -approximation, sup x E(x) goes to 0 as does, so by picking small enough, we can ensure that the coefficients of each E k (x) go to 0.Let m = m uniform 1 (p) and suppose that σ(x) has the Taylor expansion ∞ k=0 σ k x k .

Then, by grouping terms of each order, we conclude that there exist constants a ij and w j such that DISPLAYFORM0 For each S ⊆ X, let us take the derivative of this equation by every variable that occurs in S, where we take multiple derivatives of variables that occur multiple times.

This gives DISPLAYFORM1 DISPLAYFORM2 Observe that there are r ≡ n i=1 (r i + 1) choices for S, since each variable x i can be included anywhere from 0 to r i times.

Define A to be the r × m matrix with entries A S,j = h∈S a hj .

We claim that A has full row rank.

This would show that the number of columns m is at least the number of rows r = n i=1 (r i + 1), proving the desired lower bound on m. Suppose towards contradiction that the rows A S ,• admit a linear dependence: DISPLAYFORM3 where the coefficients c are all nonzero and the S denote distinct subsets of X. Let S * be such that |c * | is maximized.

Then, take the dot product of each side of the above equation by the vector with entries (indexed by j) equal to w j ( DISPLAYFORM4 We can use (7) to simplify the first term and (8) (with k = d + |S | − |S * |) to simplify the second term, giving us: DISPLAYFORM5 DISPLAYFORM6 Consider the coefficient of the monomial ∂ ∂S * p(x), which appears in the first summand with coefficient c * · |S * |! σ d ·d! .

Since the S are distinct, this monomial does not appear in any other term ∂ ∂S p(x), but it could appear in some of the terms DISPLAYFORM7 By definition, |c * | is the largest of the values |c |, and by setting small enough, all coefficients of ∂ ∂S E k (x) can be made negligibly small for every k. This implies that the coefficient of the monomial ∂ ∂S * p(x) can be made arbitrarily close to c * · |S * |! σ d ·d! , which is nonzero since c * is nonzero.

However, the left-hand side of equation FORMULA27 tells us that this coefficient should be zero -a contradiction.

We conclude that A has full row rank, and therefore that m uniform 1 DISPLAYFORM8 This completes the proof of part (i).We now consider part (ii) of the theorem.

It follows from Proposition 4.6, part (ii) that, for each i, we can Taylor-approximate x ri i using 7 log 2 (r i ) neurons arranged in a deep network.

Therefore, we can Taylor-approximate all of the x ri i using a total of i 7 log 2 (r i ) neurons.

From BID17 , we know that these n terms can be multiplied using 4n additional neurons, giving us a total of i (7 log 2 (r i ) +4).

As above, suppose that r i > 0 for i = 1, . . .

, n, and let X be the multiset in which x i occurs with multiplicity r i .It is shown in the proof of Theorem 4.1 that n i=1 (r i + 1) neurons are sufficient to Taylorapproximate p(x).

We now show that this number of neurons is also necessary for approximating p(x).

Let m = m Taylor 1 (p) and suppose that σ(x) has the Taylor expansion DISPLAYFORM9 Then, by grouping terms of each order, we conclude that there exist constants a ij and w j such that DISPLAYFORM10 For each S ⊆ X, let us take the derivative of equations FORMULA15 and FORMULA15 by every variable that occurs in S, where we take multiple derivatives of variables that occur multiple times.

This gives DISPLAYFORM11 DISPLAYFORM12 for |S| ≤ k ≤ d − 1.

Observe that there are r = n i=1 (r i + 1) choices for S, since each variable x i can be included anywhere from 0 to r i times.

Define A to be the r × m matrix with entries A S,j = h∈S a hj .

We claim that A has full row rank.

This would show that the number of columns m is at least the number of rows r = n i=1 (r i + 1), proving the desired lower bound on m. Suppose towards contradiction that the rows A S ,• admit a linear dependence: DISPLAYFORM13 where the coefficients c are nonzero and the S denote distinct subsets of X. Set s = max |S |.

Then, take the dot product of each side of the above equation by the vector with entries (indexed by DISPLAYFORM14 We can use (12) to simplify the first term and (13) (with k = d + |S | − s) to simplify the second term, giving us: DISPLAYFORM15 Since the distinct monomials ∂ ∂S p(x) are linearly independent, this contradicts our assumption that the c are nonzero.

We conclude that A has full row rank, and therefore that m Our proof in Theorem 4.1 relied upon the fact that all nonzero partial derivatives of a monomial are linearly independent.

This fact is not true for general polynomials p; however, an exactly similar argument shows that m uniform 1 (p) is at least the number of linearly independent partial derivatives of p, taken with respect to multisets of the input variables.

Consider the monomial q of p such that m uniform 1 (q) is maximized, and suppose that q(x) = x (q) is equal to the number n i=1 (r i + 1) of distinct monomials that can be obtained by taking partial derivatives of q. Let Q be the set of such monomials, and let D be the set of (iterated) partial derivatives corresponding to them, so that for d ∈ D, we have d(q) ∈ Q.Consider the set of polynomials P = {d(p) | d ∈ D}. We claim that there exists a linearly independent subset of P with size at least |D|/c.

Suppose to the contrary that P is a maximal linearly independent subset of P with |P | < |D|/c.

Since p has c monomials, every element of P has at most c monomials.

Therefore, the total number of distinct monomials in elements of P is less than |D|.

However, there are at least |D| distinct monomials contained in elements of P , since for d ∈ D, the polynomial d(p) contains the monomial d(q), and by definition all d(q) are distinct as d varies.

We conclude that there is some polynomial p ∈ P \P containing a monomial that does not appear in any element of P .

But then p is linearly independent of P , a contradiction since we assumed that P was maximal.

We conclude that some linearly independent subset of P has size at least |D|/c, and therefore that the space of partial derivatives of p has rank at least |D|/c = m We will prove the desired lower bounds for m uniform 1 (p); a very similar argument holds for m Taylor 1 (p).

As above, suppose that r i > 0 for i = 1, . . .

, n. Let X be the multiset in which x i occurs with multiplicity r i .Suppose that N (x) is an -approximation to p(x) with depth 1, and let the degree-d Taylor polynomial of N (x) be p(x) + E(x).

Let E d (x) be the degree-d homogeneous component of E(x).

Observe that the coefficients of the error polynomial E d (x) can be made arbitrarily small by setting sufficiently small.

Let m = m uniform 1 (p) and suppose that σ(x) has the Taylor expansion ∞ k=0 σ k x k .

Then, by grouping terms of each order, we conclude that there exist constants a ij and w j such that DISPLAYFORM16 For each S ⊆ X, let us take the derivative of this equation by every variable that occurs in S, where we take multiple derivatives of variables that occur multiple times.

This gives DISPLAYFORM17 Consider this equation as S ⊆ X varies over all C s multisets of fixed size s.

The left-hand side represents a linear combination of the m terms (

@highlight

We prove that deep neural networks are exponentially more efficient than shallow ones at approximating sparse multivariate polynomials.
Some recent work has shown separation between the expressive power of depth-2 and depth-3 neural networks.

These separation results are shown by constructing functions and input distributions, so that the function is well-approximable by a depth-3 neural network of polynomial size but it cannot be well-approximated under the chosen input distribution by any depth-2 neural network of polynomial size.

These results are not robust and require carefully chosen functions as well as input distributions.



We show a similar separation between the expressive power of depth-2 and depth-3 sigmoidal neural networks over a large class of input distributions, as long as the weights are polynomially bounded.

While doing so, we also show that depth-2 sigmoidal neural networks with small width and small weights can be well-approximated by low-degree multivariate polynomials.

Understanding the remarkable success of deep neural networks in many domains is an important problem at present (e.g., BID10 ).

This problem has many facets such as understanding generalization, expressive power, optimization algorithms in deep learning.

In this paper, we focus on the question of understanding the expressive power of neural networks.

In other words, we study what functions can and cannot be represented and approximated by neural networks of bounded size, depth, width and weights.

The early results on the expressive power of neural networks showed that the depth-2 neural networks are universal approximators; that is to say, with only mild restrictions on the activation functions or neurons, the depth-2 neural networks are powerful enough to uniformly approximate arbitrary continuous functions on bounded domains in R d , e.g., BID2 ; BID9 ; BID0 .

However, the bounds that they provide on the size or width of these neural networks are quite general, and therefore, weak.

Understanding what functions can be represented or wellapproximated by neural networks with bounded parameters is a general direction in the study of expressive power of neural networks.

Here the parameters could mean the number of neurons, the width of hidden layers, the depth, and the magnitude of its weights etc.

Natural signals (images, speech etc.) tend to be representable as compositional hierarchies BID10 , and deeper networks can be thought of as representing deeper hierarchies.

The power of depth has been a subject of investigation in deep learning, e.g., BID8 .

We are interested in understanding the effect of depth on the expressive power.

In particular, one may ask whether having more depth allows representation of more functions if the size bound remains the same.

BID5 show a separation between depth-2 and depth-3 neural networks.

More precisely, they exhibit a function g : R d → R and a probability distribution µ on R d such that g is bounded and supported on a ball of radius O( √ d) and expressible by a depth-3 network of size polynomially bounded in d. But any depth-2 network approximating g in L 2 -norm (or squared error) within a small constant under the distribution µ must be of size exponentially large in d. Their separation works for all reasonable activation functions including ReLUs (Rectified Linear Units) and sigmoids.

The function and the input distribution in BID5 are carefully constructed and their proof techniques seem to crucially rely on the specifics of these constructions.

Building upon this result, BID14 show that while the indicator function of the L 2 -ball can be well-approximated by depth-3 networks of polynomial size, any good approximation to it by depth-2 networks must require exponential size.

Here, the notion of approximation in the lower bound is the same as in BID5 and a carefully constructed distribution that is arguably not quite natural.

Daniely (2017) (see also BID12 ) also gave a separation between depth-2 and depth-3 networks by exhibiting a function g : S d−1 × S d−1 → R which can be well-approximated by a depth-3 ReLU neural network of polynomially bounded size and weights but cannot be approximated by any depth-2 (sigmoid, ReLU or more general) neural network of polynomial size with (exponentially) bounded weights.

This separation holds under uniform distribution on S d−1 × S d−1 , which is more natural than the previous distributions.

However, the proof technique crucially uses harmonic analysis on the unit sphere, and does not seems robust or applicable to other distributions.

Telgarsky (2016) shows a separation between depth-2k 3 + 8 and depth-k ReLU neural networks, for any positive integer k, when the input is uniformly distributed over [−1, 1] d .

BID11 (see also BID14 BID21 ) show that there are univariate functions on a bounded interval such that neural networks of constant depth require size at least Ω (poly(1/ )) for a uniform -approximation over the interval, whereas deep networks (the depth can depend on ) can have size O (polylog(1/ )).The above separation results all fit the following template: certain carefully constructed functions can be well approximated by deep networks, but are hard to approximate by shallow networks using a notion of error that uses a carefully defined distribution. (Only Liang & Srikant (2017) is distribution-independent as it deals with uniform approximation everywhere in the domain).

Thus these results do not tell us the extent to which deeper networks are more expressive than the shallow ones.

We would like to understand whether there are large classes of functions and distributions that witness the separation between deep and shallow networks.

An answer to this question is also more likely to shed light on practical applications of neural networks.

BID17 ; BID16 ; BID18 show that even functions computed by a depth-2 neural network of polynomial size can be hard to learn using gradient descent type of algorithms for a wide class of distributions.

These results address questions about learnability rather than the expressive power of deep neural networks.

BID7 shows that piecewise affine functions on [0, 1] d with N pieces can be exactly represented by a width (d + 3) network of depth at most N .

Lower bound of Ω((N + d − 1)/(d + 1)) on the depth is proven for functions of the above type when the network has width at most (d + 1) and very closely approximates the function.

Our depth separation results apply to neural networks with bounds on the magnitudes of the weights.

While we would prefer to prove our results without any weight restrictions, we now argue that small weights are natural.

In training neural networks, often weights are not allowed to be too large to avoid overfitting.

Weight decay is a commonly used regularization heuristic in deep learning to control the weights.

Early stopping can also achieve this effect.

Another motivation to keep the weights low is to keep the Lipschitz constant of the function computed by the network (w.r.t.

changes in the input, while keeping the network parameters fixed) small.

BID6 contains many of these references.

One of the surprising discoveries about neural networks has been the existence of adversarial examples BID19 ).

These are examples obtained by adding a tiny perturbation to input from class so that the resulting input is misclassified by the network.

The perturbations are imperceptible to humans.

Existence of such examples for a network suggests that the Lipschitz constant of the network is high as noted in BID19 .

This lead them to suggest regularizing training of neural nets by penalizing high Lipschitz constant to improve the generalization error and, in particular, eliminate adversarial examples.

This is carried out in BID1 , who find a way to control the Lipschitz constant by enforcing an orthonormality constraint on the weight matrices along with other tricks.

They report better resilience to adversarial examples.

On the other hand, BID13 suggest that Lipschitz constant cannot tell the full story about generalization.

We exhibit a simple function (derived from BID3 ) over the unit ball B d in d-dimensions can be well-approximated by a depth-3 sigmoidal neural network with size and weights polynomially bounded in d. However, its any reasonable approximation using a depth-2 sigmoidal neural network with polynomially bounded weights must have size exponentially large in d.

Our separation is robust and works for a general class of input distributions, as long as their density is at least 1/poly(d) on some small ball of radius 1/poly(d) in Bd .

The function we use can also be replaced by many other functions that are polynomially-Lipschitz but not close to any low-degree polynomial.

As a by-product of our argument, we also show that constant-depth sigmoidal neural networks are well-approximated by low-degree multivariate polynomials (with a degree bound that allows the depth separation mentioned above).

In this section, we show that a sigmoid neuron can be well-approximated by a low-degree polynomial.

As a corollary, we show that depth-2 (and in genenral, small-depth) sigmoidal neural networks can be well-approximated by low-degree multivariate polynomials.

The main idea is to use Chebyshev polynomial approximation as in Shalev-Shwartz et al. (2011) , which closely approximates the minimax polynomial (or the polynomial that has the smallest maximum deviation) to a given function.

For the simplicity of presentation and arguments, we drop the bias term b in the activation function σ( w, x + b).

This is without loss of generality, as explained at the end of the last section.

The activation function of a sigmoid neuron σ : R → R is defined as DISPLAYFORM0 .Chebyshev polynomials of the first kind {T j (t)} j≥0 are defined recursively as T 0 (t) = 1, T 1 (t) = t, and T j+1 (t) = 2t · T j (t) − T j−1 (t).

They form an orthonormal basis of polynomials over [−1, 1] with respect to the density 1/ DISPLAYFORM1 Proposition 1 (see Lemma B.1 in Shalev-Shwartz et al. (2011) ) bounds the magnitude of coefficients c j in the Chebyshev expansion of σ(wt) = ∞ j=0 c j T j (t).

Proposition 1.

For any j > 1, the coefficient c j in the Chebyshev expansion of a sigmoid neuron σ(wt) is bounded by DISPLAYFORM2 Proposition 1 implies low-degree polynomial approximation to sigmoid neurons as follows.

This observation appeared in Shalev-Shwartz et al. (2011) (see equation (B.7) in their paper).

For completeness, we give the proof in Appendix A. Proposition 2.

Given any w ∈ R with |w| ≤ B, there exists a polynomial p of degree DISPLAYFORM3 We use this O (log(1/ )) dependence in the above bound crucially in some of our results, e.g., a weaker version of Daniely's separation result for depth-2 and depth-3 neural networks.

Notice that this logarithmic dependence does not hold for a ReLU neuron; it is O(1/ ) instead.

A depth-2 sigmoidal neural network on input t ∈ [−1, 1] computes a linear combination of sigmoidal neurons σ(w 1 t), σ(w 2 t), . . .

, σ(w n t), for w 1 , w 2 , . . . , w n ∈ R, and computes a function DISPLAYFORM0 Here are a few propositions on polynomial approximations to small-depth neural networks.

For completeness, their proofs are included in Appendix A.Proposition 3 shows that a depth-2 sigmoidal neural network of bounded weights and width is close to a low-degree polynomial.

Proposition 3.

Let f : [−1, 1] → R be a function computed by a depth-2 sigmoidal neural network of width n and weights bounded by DISPLAYFORM1 Now consider a depth-2 sigmoidal neural network on input x ∈ B d , where DISPLAYFORM2 It is given by a linear combination of sigmoidal activations applied to linear functions w 1 , x , w 2 , x , . . . , w n , x (or affine functions when we have biases), for w 1 , w 2 , . . .

, w n ∈ R d and it computes a function F : DISPLAYFORM3 Proposition 4 below is a multivariate version of Proposition 3.Proposition 4.

Let F : B d → R be a function computed by a depth-2 sigmoidal neural network with width n and bounded weights, that is, |a i | ≤ B and DISPLAYFORM4 Note that its proof crucially uses the fact that Proposition 2 guarantees a low-degree polynomial that approximates a sigmoid neuron everywhere in [−1, 1].A depth-k sigmoidal neural network can be thought of as a composition -a depth-2 sigmoidal neural network on top, whose each input variable is a sigmoid applied to a depth-(k − 2) sigmoidal neural network.

In other words, it computes a function F : DISPLAYFORM5 where y = (y 1 , y 2 , . . .

, y m ) has each coordinate y j = σ(F j (x)), for 1 ≤ j ≤ m, such that each DISPLAYFORM6 Now we show an interesting consequence, namely, any constant-depth sigmoidal neural network with polynomial width and polynomially bounded weights can be well-approximated by a lowdegree multivariate polynomial.

The bounds presented in Proposition 5 are not optimal but the qualitative statement is interesting in contrast with the depth separation result.

The growth of the degree of polynomial approximation is dependent on the widths of hidden layers and it is also the subtle reason why a depth separation result is still possible (when the weights are bounded).Proposition 5.

Let F : B d → R be a function computed by a depth-k sigmoidal neural network of width at most n in each layer and weights bounded by B, then DISPLAYFORM7 Note that when n and B are polynomial in d and the depth k is constant, then this low-degree polynomial approximation also has degree polynomial in d. DISPLAYFORM8 y ) cannot be approximated by any depth-2 neural network of polynomial size and (exponentially) bounded weights.

Daniely shows this lower bound for a general neuron or activation function that includes sigmoids and ReLUs.

Daniely then uses G(x, y) = g( x, y ) = sin(πd 3 x, y ) which, on the other hand, is approximable by a depth-3 ReLU neural network with polynomial size and polynomially bounded weights.

This gives a separation between depth-2 and depth-3 ReLU neural networks w.r.t.

uniform distribution over DISPLAYFORM9 Daniely's proof uses harmonic analysis on the unit sphere, and requires the uniform distribution on DISPLAYFORM10 We show a simple proof of separation between depth-2 and depth-3 sigmoidal neural networks that compute functions F : B d → R. Our proof works for a large class of distributions on B d but requires the weights to be polynomially bounded.

The following lemma appears in BID4 .

Assumption 1 in BID5 and their version of this lemma for ReLU networks was used by BID3 in the proof of separation between the expressive power of depth-2 and depth-3 ReLU networks.

Lemma 6.

Let f : [−1, 1] → R be any L-Lipschitz function.

Then there exists a function g : [−1, 1] → R computed by a depth-2 sigmoidal neural network such that DISPLAYFORM11 the width n as well as the weights are bounded by poly(L, 1/ ), and |f (t) − g(t)| ≤ , for all t ∈ [−1, 1].

Now we are ready to show the separation between depth-2 and depth-3 sigmoidal neural networks.

The main idea, similar to BID3 , is to exhibit a function that is Lipschitz but far from any low-degree polynomial.

The Lipschitz property helps in showing that our function can be wellapproximated by a depth-3 neural network of small size and small weights.

However, being far from any low-degree polynomial, it cannot be approximated by any depth-2 neural network.

By modifying the function to G(x) = sin(πN x 2 ), this lower bound with L ∞ -norm holds for any distribution over B d whose support contains a radial line segment of length at least 1/poly(d), by making N = poly(d), for a large enough polynomial.

Remark:

Given any distribution µ over B d whose probability density is at least 1/poly(d) on some small ball of radius 1/poly(d), the lower bound or inapproximability by any depth-2 sigmoidal neural network can be made to work with L 2 -norm (squared error), for a large enough N = poly(d).Proof.

First, we will show that G(x) can be well-approximated by a depth-3 sigmoidal neural network of polynomial size and weights.

The idea is similar to Daniely's construction for ReLU networks in BID3 .

By Lemma 6, there exists a function f : [−1, 1] → R computed by a depth-2 sigmoidal neural network of size and weights bounded by poly(d, 1/ ) such that DISPLAYFORM0 6 , for all t ∈ [−1, 1].

Thus, we can compute x 2 i for each coordinate of x and add them up to get an -approximation to x 2 over B d .

That is, there exists a function S : B d → R computed by a depth-2 sigmoidal neural network of size and weights bounded by poly(d, 1/ ) such that S(x) − x 2 ≤ /10d 5 , for all x ∈ B d .

Again, by Lemma 6, we can approximate sin(πd 3 t) over [0, 1] using f : [−1, 1] → R computed by another depth-2 sigmoidal neural network with size and weights bounded by poly(d, 1/ ) such that sin(πd 3 t) − f (t) ≤ /2, for all t ∈ [0, 1].

Note that the composition of these two depth-2 neural networks f (N (x)) gives a depth-3 neural network as the output of the hidden layer of the bottom network can be fed into the top network as inputs.

DISPLAYFORM1 using that f that approximates sin(πd 5 t) closely must also be 4d 5 -Lipschitz DISPLAYFORM2 Now we will show the lower bound.

Consider any function F : B d → R computed by a depth-2 sigmoidal neural network whose weights are bounded by B = O(d 2 ) and width is n. Proposition 4 shows that there exists a d-variate polynomial P (x) of degree O B log(nB .

Consider S = {t ∈ [a, a + l] : t = −1 + (i + 1/2)/N, for some integer i}. Then S contains at least N l − 2 points where sin(πN t) alternates as ±1.

Any polynomial p of degree D cannot match the sign of sin(πN t) on all the points in S. Otherwise, by intermediate value theorem, p must have at least N l − 3 roots between the points of S, which means D ≥ N l − 3, a contradiction.

Thus, there exists t 0 ∈ S such that p(t 0 ) and sin(πN t 0 ) have opposite signs.

Since sin(πN t) = ±1, for any t ∈ S, the sign mismatch implies |sin(πN DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 An important remark on biases: Even though we handled the case of sigmoid neurons without biases, the proof technique carries over to the sigmoid neurons with biases σ( w, x + b).

The idea is to consider a new DISPLAYFORM6 ) with x d+1 = 1, and consider the new weight vector w new = (w, b).

Thus, w new , x new = w, x + b. The new input lies on a d-dimensional hyperplane slice of B d+1 , so we need to look at the restriction of the input distribution µ to this slice.

Most of the ideas in our proofs generalize without any technical modifications.

We defer the details to the full version.

In this section we show lower bounds under the L 2 -norm.

The theorem below gives a technical condition on the class of densities µ on B d for which our lower bound holds.

Let's give an example to illustrate that the condition on density is reasonable: Let K ⊂ B d be a convex set such that every point in K is at least r away from the boundary of B d (where r = 1/poly(d) is a parameter).

Further assume that (1) the probability mass of K is at least a constant and (2) for every point in K the probability density is within a constant factor of the uniform density on K. Then our lower bound applies to µ.Theorem 9.

Consider the function G : B d → R given by G(x) = sin(πN x 2 ).

Let µ be any probability density over B d such that there exists a subset C ⊆ B d satisfying the following two conditions:• The r-interior of C defined as C = {x ∈ C : B(x, r) ⊆ C} contains at least γ fraction of the total probability mass for some γ > 0, i.e., C µ(x)dx ≥ γ.• For any affine line , the induced probability density on every segment of length at least r in the intersection ∩ C is (α, β)-uniform, i.e., it is at least α times and at most β times the uniform density on that segment.

Let F : B d → R be any function computed by a depth-2 sigmoidal neural network with weights bounded by B and width n.

Then for any 0 < δ αγ/3β and N (B/r 2 ) log(nB 2 /δ), the function F cannot δ-approximate G on B d under L 2 -norm (squared error) under the probability density µ.In particular, if α, β, γ are constants, B = poly(d), n = 2 d , and r = 1/poly(d), then it suffices to choose N = poly(d) for a sufficiently large degree polynomial.

Proof.

We show a lower bound on L 2 -error of approximating G(x) with any multivariate polynomial P : B d → R of degree D under the distribution given by µ on B d .

For any fixed unit vector v, consider u ∈ B d−1 orthogonal to v and let u be the affine line going through u and parallel to the direction v given by u = {x = u + tv : t ∈ R}. DISPLAYFORM0 where IC(u, t) = 1, if u ∈ B d−1 and u + tv ∈ {u + t v ∈ C : t ∈ I} ⊆ C for some interval I of length at least r, and IC(u, t) = 0, otherwise DISPLAYFORM1 because for any x = u + tv ∈ C we have B(x, r) ⊆ C, therefore IC(u, t) = 1 DISPLAYFORM2 because for any line , the distribution induced by µ(x) along any line segment of length at least r in the intersection ∩ C is (α, β)-uniform, for any line DISPLAYFORM3 The last inequality is using the condition C µ(x)dx ≥ γ given in Theorem 9 and an adaptation of the following idea from Lemma 5 of BID3 .

For any fixed u and v, G(u + tv) = sin(πN ( u 2 + t 2 )) and P (u + tv) is a polynomial of degree at most D in t. The function sin(πN ( u 2 +t 2 )) alternates its sign as u 2 +t 2 takes values that are successive integer multiples of 1/N .

Consider s = t 2 ∈ [0, 1] and divide [0, 1] into N disjoint segments using integer grid of step size 1/N .

For any polynomial p(s) of degree at most D and any interval I ⊆ [0, 1] of length r D/N , there exists at least N r − D − 2 segments of length 1/N each on which sin(πN s) and p(s) do not change signs and have opposite signs.

Now using (sin(πN s) − p(s)) 2 ≥ sin 2 (πN s), integrating we get that I (sin(πN s) − p(s)) 2 ds ≥ r/2.

Extending this proof to t instead of s = t 2 , using sin 2 (πN t 2 )t ≤ sin 2 (πN t) for all t ∈ [0, 1], and incorporating the shift πN u 2 , we can similarly show that I sin 2 (πN ( u 2 + t 2 )) − P (u + tv)) 2 dt ≥ r/3.

Summing up over multiple such intervals gives the final inequality.

The L 2 separation between depth-2 and depth-3 neural networks under probability density µ now follows by taking a small enough δ, and combining the following ingredients (i) Proposition 4 says that any depth-2 sigmoid neural networks of width n = 2 d and weights bounded by B = poly(d) can be δ-approximated in L ∞ (and hence, also L 2 ) by a multivariate polynomials of degree DISPLAYFORM4 (ii) proof of Theorem 7 (initial part) says that G(x) can be δ-approximated in L ∞ (and hence, also L 2 ) by a depth-3 sigmoid neural network of width and size poly(d), but (iii) Theorem 9 says that, for N = poly(d) of large enough degree, G(x) cannot be 3δ-approximated in L 2 by any multivariate polynomial of degree D, and (iv) triangle inequality.

Proof.

Consider the degree-D approximation to σ(wt) given by the first D terms in its Chebyshev expansion.

The error of this approximation for any t ∈ [−1, 1] is bounded by DISPLAYFORM0 using Proposition 1, |w| ≤ B, and D = O (B log (B/ )).

Proof.

Let f be computed by a depth-2 sigmoidal neural network given by f (t) = n i=1 a i σ(w i t).

Define a parameter = δ/nB. Proposition 2 guarantees polynomial p 1 , p 2 , . . . , p n of degree O (B log(B/ )) such that |σ(w i t) − p i (t)| ≤ , for all t ∈ [−1, 1].

Thus, the polynomial p(t) = n i=1 a i p i (t) has degree O (B log(B/ )) = O B log(nB 2 /δ) , and for any t ∈ [−1, 1], DISPLAYFORM0

Proof.

Let F be computed by a depth-2 neural network given by DISPLAYFORM0 Define a parameter = δ/nB. Proposition 2 guarantees polynomial p 1 , p 2 , . . .

, p n of degree O (B log(B/ )) such that |σ( w i t) − p i (t)| ≤ , for all t ∈ [−1, 1].

Consider the following polynomial P (x) = P (x 1 , x 2 , . . .

, x d ) = n i=1 a i p i ( w i / w i , x ).

P (x) is a d-variate polynomial of degree O (B log(B/ )) = O B log(nB 2 /δ) in each variable x 1 , x 2 , . . .

, x d .

For any DISPLAYFORM1 |σ( w i t i ) − p i (t i )| using t i = w i / w i , x and |a i | ≤ B ≤ nB using |σ( w i t) − p i (t)| ≤ , for all t ∈ [−1, 1] = δ.

Proof.

We prove this by induction on the depth k.

By induction hypothesis each F j (x) can be 1 -approximated (in L ∞ -norm) by a d-variate polynomial Q j (x) of degree O (nB) k−2 log (k−2) (nB/ 1 ) in each variable.

Thus, |F j (x) − Q j (x)| = 1 , for any x ∈ B d and 1 ≤ j ≤ m. Because a sigmoid neuron is Lipschitz, DISPLAYFORM0 for any x ∈ B d and 1 ≤ j ≤ m.

Since F j (x) is the output of a depth-(k−2) sigmoidal neural network of width at most n and weights at most B, we must have |F j (x)| ≤ nB, for all x ∈ B d .

Thus, |Q j (x)| ≤ nB + 1 ≤ 2nB. By Proposition 2, there exists a polynomial q(t) of degree at most O (nB log(nB/ 2 )) such that |σ(Q j (x)) − q(Q j (x))| ≤ 2 , for all x ∈ B d and 1 ≤ j ≤ m.

Consider q ∈ R m as q = (q(Q 1 (x)), q(Q 2 (x)), . . .

, q(Q m (x))).

Then, for any x ∈ B d , we have DISPLAYFORM1 Again by Proposition 2, there is a polynomial p of degree at most O (nB log(nB/ )) such that |σ( w i , q ) − p( w i , q )| ≤ , for all x ∈ B d and 1 ≤ i ≤ n.

This is because | w i , q | = O(nB).Let's define P (x) = n i=1 a i p( w i , q ).

Therefore, for any x ∈ B d , DISPLAYFORM2 if we use 1 = 2 = δ/3n 3/2 B 2 and = δ/3nB.P (x) is a d-variate polynomial of degree DISPLAYFORM3 in each variable.

<|TLDR|>

@highlight

depth-2-vs-3 separation for sigmoidal neural networks over general distributions
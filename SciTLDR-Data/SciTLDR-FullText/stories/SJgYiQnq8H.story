The goal of standard compressive sensing is to estimate an unknown vector from linear measurements under the assumption of sparsity in some basis.

Recently, it has been shown that significantly fewer measurements may be required if the sparsity assumption is replaced by the assumption that the unknown vector lies near the range of a suitably-chosen generative model.

In particular, in (Bora {\em et al.}, 2017) it was shown that roughly $O(k\log L)$ random Gaussian measurements suffice for accurate recovery when the $k$-input generative model is bounded and $L$-Lipschitz, and that $O(kd \log w)$ measurements suffice for $k$-input ReLU networks with depth $d$ and width $w$.  In this paper, we establish corresponding algorithm-independent lower bounds on the sample complexity using tools from minimax statistical analysis.

In accordance with the above upper bounds, our results are summarized as follows: (i) We construct an $L$-Lipschitz generative model capable of generating group-sparse signals, and show that the resulting necessary number of measurements is $\Omega(k \log L)$; (ii) Using similar ideas, we construct two-layer ReLU networks of high width requiring $\Omega(k \log w)$ measurements, as well as lower-width deep ReLU networks requiring $\Omega(k d)$ measurements.

As a result, we establish that the scaling laws derived in (Bora {\em et al.}, 2017) are optimal or near-optimal in the absence of further assumptions.

The problem of sparse estimation via linear measurements (commonly referred to as compressive sensing) is well-understood, with theoretical developments including sharp performance bounds for both practical algorithms [1, 2, 3, 4] and (potentially intractable) information-theoretically optimal algorithms [5, 6, 7, 8] .

Following the tremendous success of deep generative models in a variety of applications [9] , a new perspective on compressive sensing was recently introduced, in which the sparsity assumption is replaced by the assumption of the underlying signal being well-modeled by a generative model (typically corresponding to a deep neural network) [10] .

This approach was seen to exhibit impressive performance in experiments, with reductions in the number of measurements by large factors such as 5 to 10 compared to sparsity-based methods.

In addition, [10] provided theoretical guarantees on their proposed algorithm, essentially showing that an L-Lipschitz generative model with bounded k-dimensional inputs leads to reliable recovery with m = O(k log L) random Gaussian measurements (see Section 2 for a precise statement).

Moreover, for a ReLU network generative model from R k to R n with width w and depth d, it suffices to have m = O(kd log w).

A variety of follow-up works provided additional theoretical guarantees (e.g., for more specific optimization algorithms [11, 12] , more general models [13] , or under random neural network weights [14, 15] ) for compressive sensing with generative models, but the main results of [10] are by far the most relevant to ours.

In this paper, we address a prominent gap in the existing literature by establishing algorithmindependent lower bounds on the number of measurements needed (e.g., this is explicitly posed as an open problem in [15] ).

Using tools from minimax statistical analysis, we show that for generative models satisfying the assumptions of [10] , the above-mentioned dependencies m = O(k log L) and m = O(kd log w) cannot be improved (or in the latter case, cannot be improved by more than a log n factor) without further assumptions.

Our argument is essentially based on a reduction to compressive sensing with a group sparsity model (e.g., see [16] ), i.e., forming a neural network that is capable of producing such signals.

The proofs are presented in the full paper [17] .

We begin by stating a simple corollary of a main result of Bora et al. [10] .

As we show in [17] , this is obtained by extending [10, Thm.

1.2] from spherical to rectangular domains, and then converting the high-probability bound to an average one.

In [17] , we also handle the case of spherical domains.

for a universal constant C, we have for a universal constant C that

In the following, we construct a Lipschitz-continuous generative model that can generate bounded kgroup-sparse vectors.

Then, by making use of minimax statistical analysis for group-sparse recovery, we provide an information-theoretic lower bounds that matches the upper bound in Corollary 1.

More precisely, we say that a signal in R n is k-group-sparse if, when divided into k blocks of size n k , each block contains at most one non-zero entry.

We define

Our construction of the generative function G :

is given as follows:

. . . figure shows the mapping from z1 → (x1, . . .

, x n/k ), and the same relation holds for z2 → (x n/k+1 , . . . , x 2n/k ), etc. up to z k → (x n−k+1 , . . .

, xn).

• The output x ∈ R n is divided into k sub-sequences of length

is only a function of the corresponding input z i , for i = 1, . . .

, k.

• The mapping from z i to x (i) is as shown in Figure 1 .

The interval [−r, r] is divided into n k intervals of length 2rk n , and the jth entry of x (i) can only be nonzero if z i takes a value in the j-th interval.

Within that interval, the mapping takes a "doubletriangular" shape -the endpoints and midpoint are mapped to zero, the points It is easy to show that the generative model G :

kr .

In addition, using similar steps to the case of k-sparse recovery [7, 18] , we are able to obtain a minimax lower bound for k-group-sparse recovery, which holds when x max is not too small.

Based on these results, the following sample complexity lower bound is proved in [17] .

n (and associated output dimension n) such that, for any A ∈ R m×n satisfying A 2 F = C A n, any algorithm that produces somex satisfying sup with a sufficiently large implied constant, which is a very mild assumption since for fixed r and α, the right-hand side tends to zero as k grows large (whereas typical Lipschitz constants are at least equal to one, if not much higher).

In this section, as opposed to considering general Lipschitz-continuous generative models, we focus on generative models given by neural networks with ReLU activations.

Similar to the derivation of Corollary 1, we have the following corollary for ReLU-based networks from [10, Thm.

2 such that for a universal constant C , sup

Note that this result holds even when the domain D = R k , so we do not need to distinguish between the rectangular and spherical domains.

Moreover, this result makes no assumptions about the neural network weights (nor domain size), but rather, only the input size, width, and depth.

Thus far, we have considered forming a generative model G : R k → R n capable of producing k-group-sparse signals, which leads to a lower bound of m = Ω(k log n).

While this precise approach does not appear to be suited to properly understanding the dependence on width and depth in Corollary 2, we now show that a simple variant indeed suffices: We form a wide and/or deep ReLU network G : R k → R n capable of producing all (kk 0 )-group-sparse signals having non-zero entries ±ξ, where k 0 is a certain positive integer that may be much larger than one.

The idea of the construction is illustrated in Figure 2 , which shows the mappings for k = 1 (the general case simply repeats this structure in parallel to get an output dimension n = n 0 k).

Note also that we need to replace the rectangular shapes by trapeziums (with high-gradient diagonals) to make them implementable with a ReLU network.

Again using the minimax lower bound for group-sparse recovery and a suitable choice of ξ, the following is proved in [17] .

Theorem 2.

Fix C 1 , C A > 0, and consider the problem of compressive sensing with generative models under i.i.d.

N 0, α m noise, a measurement matrix A ∈ R m×n satisfying A 2 F = C A n, and the above-described generative model G : R k → R n with parameters k, k 0 , n 0 , and ξ.

Then, if n 0 ≥ C 0 k 0 for an absolute constant C 0 , then there exists a constant C 2 = Θ(1) such that the choice ξ = C2α k yields the following:

• Any algorithm producing somex satisfying sup x * ∈Range(G) E x − x * 2 2 ≤ C 1 α must also have m = Ω kk 0 log n kk0 (or equivalently m = Ω kk 0 log n0 k0 , since n = n 0 k).

• The generative function G can be implemented as a ReLU network with a single hidden layer (i.e., d = 2) of width at most w = O(k( n0 k0 ) k0 ).

• Alternatively, if n0 k0 is an integer power of two, the generative function G can be implemented as a ReLU network with depth d = O k 0 log n0 k0 and width w = O(n).

In the settings described in the second and third dot points, the sample complexity from Corollary 2 behaves as kd log w = O kk 0 log n0 k0 + k log k and kd log w = O kk 0 · log n0 k0 · log n respectively.

While we do not claim a lower bound for every possible combination of depth and width, the final statement of Theorem 2 reveals that the upper and lower bounds match up to a constant factor (high-width case with log k ≤ O k 0 log n0 k0 ) or up to a log n factor (high-depth case).

The proof of the claim for the high-width case is based on the fact that in Figure 2 , upon replacing the rectangles by trapeziums, each mapping is piecewise linear, and at the -th scale the number of pieces is O n0 k0 −1 , which we sum over = 1, . . .

, k 0 to get the overall width.

In the high-depth case, we exploit the periodic nature of the signals in Figure 2 , and use the fact that depth-d neural networks can be used to produce periodic signals with O(2 d ) repetitions [19] .

In our case, the maximum number of repetitions is O n0 k0 k0 (at the finest scale in Figure 2 ).

We have established, to our knowledge, the first lower bounds on the sample complexity for compressive sensing with generative models.

To achieve these, we constructed generative models capable of producing group-sparse signals, and then applied a minimax lower bound for group-sparse recovery.

For bounded Lipschitz-continuous generative models we matched the O(k log L) scaling law derived in [10] , and for ReLU-based generative models, we showed that the dependence of the O(kd log w) bound from [10] has an optimal or near-optimal dependence on both the width and depth.

A possible direction for future research is to understand what additional assumptions could be placed on the generative model to further reduce the sample complexity.

@highlight

We establish that the scaling laws derived in (Bora et al., 2017) are optimal or near-optimal in the absence of further assumptions.
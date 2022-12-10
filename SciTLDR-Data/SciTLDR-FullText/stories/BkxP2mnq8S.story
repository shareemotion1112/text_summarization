The goal of compressed sensing is to learn a structured signal $x$   from a limited number of noisy linear measurements $y \approx Ax$.  In   traditional compressed sensing, ``structure'' is represented by   sparsity in some known basis.

Inspired by the success of deep   learning in modeling images, recent work starting with~\cite{BDJP17}   has instead considered structure to come from a generative model   $G: \R^k \to \R^n$.  We present two results establishing the   difficulty of this latter task, showing that existing bounds are   tight.

First, we provide a lower bound matching the~\cite{BDJP17} upper   bound for compressed sensing from $L$-Lipschitz generative models   $G$.  In particular, there exists such a function that requires   roughly $\Omega(k \log L)$ linear measurements for sparse recovery   to be possible.

This holds even for the more relaxed goal of   \emph{nonuniform} recovery.

Second, we show that generative models generalize sparsity as a   representation of structure.

In particular, we construct a   ReLU-based neural network $G: \R^{2k} \to \R^n$ with $O(1)$ layers   and $O(kn)$ activations per layer, such that the range of $G$   contains all $k$-sparse vectors.

In compressed sensing, one would like to learn a structured signal x ∈ R n from a limited number of linear measurements y ≈ Ax.

This is motivated by two observations: first, there are many situations where linear measurements are easy, in settings as varied as streaming algorithms, single-pixel cameras, genetic testing, and MRIs.

Second, the unknown signals x being observed are structured or "compressible": although x lies in R n , it would take far fewer than n words to describe x. In such a situation, one can hope to estimate x well from a number of linear measurements that is closer to the size of the compressed representation of x than to its ambient dimension n.

In order to do compressed sensing, you need a formal notion of how signals are expected to be structured.

The classic answer is to use sparsity.

Given linear measurements 1 y = Ax of an arbitrary vector x ∈ R n , one can hope to recover an estimate x * of x satisfying

for some constant C and norm · .

In this paper, we will focus on the 2 norm and achieving the guarantee with 3/4 probability.

Thus, if x is well-approximated by a k-sparse vector x , it should be accurately recovered.

Classic results such as [CRT06] show that (1) is achievable when A consists of m = O(k log n k ) independent Gaussian linear measurements.

This bound is tight, and in fact no distribution of matrices with fewer rows can achieve this guarantee in either 1 or 2 [DIPW10] .

Although compressed sensing has had success, sparsity is a limited notion of structure.

Can we learn a richer model of signal structure from data, and use this to perform recovery?

In recent years, deep convolutional neural networks have had great success in producing rich models for representing the manifold of images, notably with generative adversarial networks (GANs) [GPAM + 14] and variational autoencoders (VAEs) [KW14] .

These methods produce generative models G : R k → R n that allow approximate sampling from the distribution of images.

So a natural question is whether these generative models can be used for compressed sensing.

In [BJPD17] it was shown how to use generative models to achieve a guarantee analogous to (1): for any L-Lipschitz G : R k → R n , one can achieve

where r, δ > 0 are parameters, B k (r) denotes the radius-r 2 ball in R k and Lipschitzness is defined with respect to the 2 -norms, using only m = O(k log Lr δ ) measurements.

Thus, the recovered vector is almost as good as the nearest point in the range of the generative model, rather than in the set of k-sparse vectors.

We will refer to the problem of achieving the guarantee in (2) as "functionsparse recovery".

Our main theorem is that the [BJPD17] result is tight: for any setting of parameters n, k, L, r, δ, there exists an L-Lipschitz function G : R k → R n such that any algorithm achieving (2) with 3/4 probability must have Ω(min(k log Lr δ , n)) linear measurements.

Notably, the additive error δ that was unnecessary in sparse recovery is necessary for general Lipschitz generative model recovery.

A concurrent paper [LS19] proves a lower bound for a restricted version of (2).

They show a lower bound when the vector that x lies in the image of G and for a particular value of δ.

Our results, in comparison, apply to the most general version of the problem and are proven using a simpler communication complexity technique.

The second result in this paper is to directly relate the two notions of structure: sparsity and generative models.

We produce a simple Lipschitz neural network G sp : R 2k → R n , with ReLU activations, 2 hidden layers, and maximum width O(kn), so that the range of G contains all k-sparse vectors.

A second result of [BJPD17] is that for ReLU-based neural networks, one can avoid the additive δ term and achieve a different result from (2):

using O(kd log W ) measurements, if d is the depth and W is the maximum number of activations per layer.

Applying this result to our sparsity-producing network G sp implies, with O(k log n) measurements, recovery achieving the standard sparsity guarantee (1).

So the generative-model representation of structure really is more powerful than sparsity.

As described above, this paper contains two results: an Ω(min(k log Lr δ , n)) lower bound for compressed sensing relative to a Lipschitz generative model, and an O(1)-layer generative model whose range contains all sparse vectors.

These results are orthogonal, and we outline each in turn.

Over the last decade, lower bounds for sparse recovery have been studied extensively.

The techniques in this paper are most closely related to the techniques used in [DIPW10] .

Similar to [DIPW10] , our proof is based on communication complexity.

We will exhibit an LLipschitz function G and a large finite set Z ⊂ Im(G) ⊂ B n (R) of points that are well-separated.

Then, given a point x that is picked uniformly at random from Z, we show how to identify it from Ax using the function-sparse recovery algorithm.

This implies Ax also contains a lot of information, so m must be fairly large.

Formally, we produce a generative model whose range includes a large, well-separated set:

(4) log(|X|) = Ω min(k log( Lr R )), n Now, suppose we have an algorithm that can perform function-sparse recovery with respect to G from Theorem 2.1, with approximation factor C, and error δ < R/8 within the radius r ball in k-dimensions.

Set t = Θ(log n), and for any z 1 , z 2 , . . .

, z t ∈ Z = G(X) take

The idea of the proof is the following: given y = Az, we can recover z such that

and so, because Z has minimum distance R/ √ 6, we can exactly recover z t by rounding z to the nearest element of Z.

But then we can repeat the process on (Az − Az t ) to find z t−1 , then z t−2 , up to z 1 , and learn t lg |Z| = Ω(tk log(Lr/R)) bits total.

Thus Az must contain this many bits of information; but if the entries of A are rational numbers with poly(n) bounded numerators and (the same) poly(n) bounded denominator, then each entry of Az can be described in O(t + log n) bits, so

There are two issues that make the above outline not totally satisfactory, which we only briefly address how to resolve here.

First, the theorem statement makes no supposition on the entries of A being polynomially bounded.

To resolve this, we perturb z with a tiny (polynomially small) amount of additive Gaussian noise, after which discretizing Az at an even tinier (but still polynomial) precision has negligible effect on the failure probability.

The second issue is that the above outline requires the algorithm to recover all t vectors, so it only applies if the algorithm succeeds with 1 − 1/t probability rather than constant probability.

This is resolved by using a reduction from the augmented indexing problem, which is a one-way communication problem where Alice has z 1 , z 2 , . . .

, z t ∈ Z, Bob has i ∈ [Z] and z i+1 , · · · , z n , and Alice must send Bob a message so that Bob can output z i with 2/3 probability.

This still requires Ω(t log |Z|) bits of communication, and can be solved in O(m(t + log n)) bits of communication by sending Az as above.

Formally, our lower bound states:

A is an algorithm which picks a matrix A ∈ R m×n , and given Ax returns an x * satisfying (2) with probability ≥ 3/4, then m = Ω(min(k log(Lr/δ), n)).

Constructing the set.

The above lower bound approach, relies on finding a large, well-separated set Z as in Theorem 2.1.

We construct this aforementioned set Z within the n-dimensional 2 ball of radius R such that any two points in the set are at least Ω(R) apart.

Furthermore, since we wish to use a function-sparse recovery algorithm, we describe a function G : R k → R n and set the radius R such that G is LLipschitz.

In order to get the desired lower bound, the image of G needs to contain a subset of at least (Lr) Ω(k) points.

First, we construct a mapping as described above from R to R n/k i.e we need to find (Lr)

points in B n/k (R) that are mutually far apart.

We show that certain binary linear codes over the alphabet {±R/ √ n} yield such points that are mutually R/ √ 3k apart.

We construct a O(L)-Lipschitz mapping of O( √ Lr) points in the interval [0, r/ √ k] to a subset of these points.

In order to extend this construction to a mapping from R k to R n , we apply the above function in a coordinate-wise manner.

This would result in a mapping with the same Lipschitz parameter.

The points in R n that are images of these points lie in a ball of radius R but could potentially be R/ √ 3k close.

To get around this, we use an error correcting code over a large alphabet to choose a subset of these points that is large enough and such that they are still mutually R/ √ 6 far apart.

2.2 Sparsity-producing generative model.

To produce a generative model whose range consists of all k-sparse vectors, we start by mapping R 2 to the set of positive 1-sparse vectors.

For any pair of angles θ 1 , θ 2 , we can use a constant number of unbiased ReLUs to produce a neuron that is only active at points whose representation (r, θ) in polar coordinates has θ ∈ (θ 1 , θ 2 ).

Moreover, because unbiased ReLUs behave linearly, the activation can be made an arbitrary positive real by scaling r appropriately.

By applying this n times in parallel, we can produce n neurons with disjoint activation ranges, making a network R 2 → R n whose range contains all 1-sparse vectors with nonnegative coordinates.

By doing this k times and adding up the results, we produce a network R 2k → R n whose range contains all k-sparse vectors with nonnegative coordinates.

To support negative coordinates, we just extend the k = 1 solution to have two ranges within which it is non-zero: for one range of θ the output is positive, and for another the output is negative.

This results in the following theorem:

Theorem 2.3.

There exists a 2 layer neural network

In this section, we prove a lower bound for the sample complexity of function-sparse recovery by a reduction from a communication game.

We show that the communication game can be won by sending a vector Ax and then performing function-sparse recovery.

A lower bound on the communication complexity of the game implies a lower bound on the number of bits used to represent Ax if Ax is discretized.

We can then use this to lower bound the number of measurements in A.

Since we are dealing in bits in the communication game and the entries of a sparse recovery matrix can be arbitrary reals, we will need to discretize each measurement.

We show first that discretizing the measurement matrix by rounding does not change the resulting measurement too much and will allow for our reduction to proceed.

Matrix conditioning.

We first show that, without loss of generality, we may assume that the measurement matrix A is well-conditioned.

In particular, we may assume that the rows of A are orthonormal.

We can multiply A on the left by any invertible matrix to get another measurement matrix with the same recovery characteristics.

If we consider the singular value decomposition A = U ΣV * , where U and V are orthonormal and Σ is 0 off the diagonal, this means that we can eliminate U and make the entries of Σ be either 0 or 1.

The result is a matrix consisting of m orthonormal rows.

Discretization.

For well-conditioned matrices A, we use the following lemma (similar to one from [DIPW10] ) to show that we can discretize the entries without changing the behavior by much:

Lemma 3.1.

Let A ∈ R m×n be a matrix with orthonormal rows.

Let A be the result of rounding A to b bits per entry.

Then for any v ∈ R n there exists an s ∈ R n with A v = A(v − s) and

Proof.

Let A = A − A be the error when discretizing A to b bits, so each entry of A is less than 2 −b .

Then for any v and s = A T A v, we have As = A v and

The Augmented Indexing problem.

As in [DIPW10] , we use the Augmented Indexing communication game which is defined as follows: There are two parties, Alice and Bob.

Alice is given a string y ∈ {0, 1} d .

Bob is given an index i ∈ [d], together with y i+1 , y i+2 , . . .

, y d .

The parties also share an arbitrarily long common random string r. Alice sends a single message M (y, r) to Bob, who must output y i with probability at least 2/3, where the probability is taken over r. We refer to this problem as Augmented Indexing.

The communication cost of Augmented Indexing is the minimum, over all correct protocols, of length |M (y, r)| on the worst-case choice of r and y.

The following theorem is well-known and follows from Lemma 13 of [MNSW98] (see, for example, an explicit proof in [DIPW10])

A well-separated set of points.

We would like to prove Theorem 2.1, getting a large set of wellseparated points in the image of a Lipschitz generative model.

Before we do this, though, we prove a k = 1 analog:

There is a set of points P in B n (1) ⊂ R n of size 2 Ω(n) such that for each pair of points x, y ∈ P x − y ∈ 1 3 , 2 3

Proof.

Consider a τ -balanced linear code over the alphabet {± 1 √ n } with message length M .

It is known that such codes exist with block length O(M/τ 2 ) [BATS09] .

Setting the block length to be n and τ = 1/6, we get that there is a set of 2 Ω(n) points in R n such that the pairwise hamming distance is between Now we wish to extend this result to arbitrary k while achieving the parameters in Theorem 2.1.

Proof of Theorem 2.1.

We first define an O(L)-Lipschitz map g : R → R n/k that goes through a set of points that are pairwise Θ R √ k apart.

Consider the set of points P from Lemma 3.3 scaled

, Lr/R).

Choose subset P that such that it contains exactly min (Lr/R, exp(Ω(n/k))) points and let g 1 : [0, r/ √ k]

→ P be a piecewise linear function that goes through all the points in P in order.

Then, we define g : R → R n/k as:

Also, for every point (x 1 , . . .

,

However, there still exist distinct points x, y ∈ I k (for instance points that differ at exactly one coordinate)

We construct a large subset of the points in I k such that any two points in this subset are far apart using error correcting codes.

Consider the A ⊂ P s.t.

|A| > |P | /2 is a prime.

For any integer z > 0, there is a prime between z and 2z, so such a set A exists.

Consider a Reed-Solomon code of block length k, message length k/2, distance k/2 and alphabet A. The existence of such a code implies that there is a subset X of (P ) k of size at least (|P | /2) k/2 such that every pair of distinct elements from this set disagree in k/2 coordinates.

This translates into a distance of

in 2-norm.

So, if we set G = g ⊗k and

we get have a set of (|P | /2) k/2 ≥ (min(exp(Ω(n/k)), Lr/R)) k/2 points which are

apart in 2-norm, lie within the 2 ball of radius R.

Lower bound.

We now prove the lower bound for function-sparse recovery.

Proof of Theorem 2.2.

An application of Theorem 2.1 with R = √ Lrδ gives us a set of points Z and G such that Z = G(X) ⊆ R n such that log(|Z|) = Ω(min(k log( Lr δ ), n)), and for all x ∈ Z, x 2 ≤ √ Lrδ and for all x, x ∈ Z, x − x 2 ≥ √ Lrδ/ √ 6.

Let d = log |X| log n, and let D = 16 √ 3(C + 1).

We will show how to solve the Augmented Indexing problem on instances of size d = log(|Z|) · log(n) = Ω(k log(Lr) log n) with communication cost O(m log n).

The theorem will then follow by Theorem 3.2.

Alice is given a string y ∈ {0, 1} d , and Bob is given i ∈ [d] together with y i+1 , y i+2 , . . .

, y d , as in the setup for Augmented Indexing.

Alice splits her string y into log n contiguous chunks y 1 , y 2 , . . .

, y log n , each containing log |X| bits.

She uses y j as an index into the set X to choose x j .

Alice defines

Alice and Bob use the common randomness R to agree upon a random matrix A with orthonormal rows.

Both Alice and Bob round A to form A with b = Θ(log(n)) bits per entry.

Alice computes A x and transmits it to Bob.

Note that, since x ∈ ± 1 √ n the x's need not be discretized.

From Bob's input i, he can compute the value j = j(i) for which the bit y i occurs in y j .

Bob's input also contains y i+1 , . . .

, y n , from which he can reconstruct x j+1 , . . .

, x log n , and in particular can compute

Bob then computes A z, and using A x and linearity, he can compute

So from Lemma 3.1, there exists some s with A w = A(w − s) and

Ideally, Bob would perform recovery on the vector A(w −s) and show that the correct point x j is recovered.

However, since s is correlated with A and w, Bob needs to use a slightly more complicated technique.

Bob first chooses another vector u uniformly from B n (R/D j ) and computes A(w − s − u) = A w − Au.

He then runs the estimation algorithm A on A and A(w − s − u), obtainingŵ.

We have that u is independent of w and s, and that

so as a distribution over u, the ranges of the random variables w − s − u and w − u overlap in at least a 1 − 1/n fraction of their volumes.

Therefore w − s − u and w − u have statistical distance at most 1/n.

The distribution of w − u is independent of A, so running the recovery algorithm on A(w − u) would work with probability at least 3/4.

Hence with probability at least 3/4 − 1/n ≥ 2/3 (for n large enough),ŵ satisfies the recovery criterion for w − u, meaning

Now,

Since δ < Lr/4, this distance is strictly bounded by R/2 √ 6.

Since the minimum distance in X is R/ √ 6, this means D j x j −ŵ 2 < D j x −ŵ 2 for all x ∈ X, x = x j .

So Bob can correctly identify x j with probability at least 2/3.

From x j he can recover y j , and hence the bit y i that occurs in y j .

Hence, Bob solves Augmented Indexing with probability at least 2/3 given the message A x. Each entry of A x takes O(log n) bits to describe because A is discretized to up to log(n) bits and

Hence, the communication cost of this protocol is O(m · log n).

By Theorem 3.2, m log n = Ω(min(k log(Lr/δ), n) · log n), or m = Ω(min(k log(Lr/δ), n)).

We show that the set of all k-sparse vectors in R n is contained in the image of a 2 layer neural network.

This shows that function-sparse recovery is a generalization of sparse recovery.

Lemma 4.1.

There exists a 2 layer neural network G :

Our construction is intuitively very simple.

We define two gadgets G

.

Then, we set the i th output node (G(x 1 , x 2 )

Varying the distance of (x 1 , x 2 ) from the origin will allow us to get the desired value at the output node i.

In a similar manner, G − i which produces negative values at output node i of G with the internal nodes defined as:

The last ReLU activation preserves only negative values.

Since G

This is positive only when θ ∈ (β, π + β).

Similarly, cos(β + α/2)x 1 − sin(β + α/2)x 2 = t sin(θ − (β + α/2)) and is positive only when θ ∈ (β + α/2, π + β + α/2).

So, a + (i),1 and a + (i),2 are both non-zero when θ ∈ (β + α/2, π + β).

Using some elementary trigonometry, we may see that:

In Fact A.1, we show a proof of the above identity.

Observe that when θ > β + α, this term is negative and hence b i = 0.

So, we may conclude that G + i ((x 1 , x 2 )) = 0 if and only if (x 1 , x 2 ) = (t sin(θ), t cos(θ)) with θ ∈ ((i−1)α, iα).

Also, observe that G + i (t sin(β +α/2), t cos(β +α/2)) = t. Similarly G Proof of Theorem 2.3.

Given a vector z that is non-zero at k coordinates, let i 1 < i 2 < · · · < i k be the indices at which z is non-zero.

We may use copies of G from Lemma 4.1 to generate 1-sparse vectors v 1 , . . .

, v k such that (v j ) ij = z ij .

Then, we add these vectors to obtain z.

It is clear that we only used k copies of G to create G sp .

So, G sp can be represented by a neural network with 2 layers.

Theorem 1 provides a reduction which uses only 2 layers.

Then, using the algorithm from Theorem 3, we can recover the correct k-sparse vector using O(kd log(nk)) measurements.

Since d = 4 and ≤ n, this requires only O(k log n) linear measurements to perform 2 / 2 (k, C)-sparse recovery.

@highlight

Lower bound for compressed sensing w/ generative models that matches known upper bounds
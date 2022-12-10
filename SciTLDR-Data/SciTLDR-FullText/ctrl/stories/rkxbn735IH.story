We study the robust one-bit compressed sensing problem whose goal is to design an algorithm that faithfully recovers any sparse target vector $\theta_0\in\mathbb{R}^d$ \emph{uniformly} from $m$ quantized noisy measurements.

Under the assumption that the measurements are sub-Gaussian,  to recover any $k$-sparse $\theta_0$ ($k\ll d$) \emph{uniformly} up to an error $\varepsilon$ with high probability, the best known computationally tractable algorithm requires\footnote{Here, an algorithm is ``computationally tractable'' if it has provable convergence guarantees.

The notation $\tilde{\mathcal{O}}(\cdot)$ omits a logarithm factor of $\varepsilon^{-1}$.} $m\geq\tilde{\mathcal{O}}(k\log d/\varepsilon^4)$. In this paper, we consider a new framework for the one-bit sensing problem where the sparsity is implicitly enforced via mapping a low dimensional representation $x_0$ through a known $n$-layer ReLU generative network $G:\mathbb{R}^k\rightarrow\mathbb{R}^d$. Such a framework poses low-dimensional priors on $\theta_0$ without a known basis.

We propose to recover the target $G(x_0)$ via an unconstrained empirical risk minimization (ERM) problem under a much weaker \emph{sub-exponential  measurement assumption}.  For such a problem, we establish a joint statistical and computational analysis.

In particular, we prove that the ERM estimator in this new framework achieves an improved statistical rate of $m=\tilde{\mathcal{O}} (kn\log d /\epsilon^2)$ recovering any $G(x_0)$ uniformly up to an error $\varepsilon$. Moreover, from the lens of computation, we prove that under proper conditions on the ReLU weights, our proposed empirical risk, despite non-convexity, has no stationary point outside of small neighborhoods around the true representation $x_0$ and its negative multiple.

Furthermore, we show that the global minimizer of the empirical risk stays within the neighborhood around $x_0$ rather than its negative multiple.

Our analysis sheds some light on the possibility of inverting a deep generative model under partial and quantized measurements, complementing the recent success of using deep generative models for inverse problems.

Quantized compressed sensing investigates how to design the sensing procedure, quantizer and reconstruction algorithm so as to recover a high dimensional vector from a limited number of quantized measurements.

The problem of one-bit compressed sensing, which aims at recovering a target vector θ 0 ∈ R d from single-bit observations y i = sign( a i , θ 0 ), i ∈ {1, 2, · · · , m}, m d and random sensing vectors a i ∈ R d , is particularly challenging.

Previous theoretical successes on this problem (e.g. Jacques et al. (2013) ; Plan and Vershynin (2013) ) mainly rely on two key assumptions: (1) The Gaussianity of the sensing vector a i , (2) The sparsity of the vector θ 0 on a given basis.

However, the practical significance of these assumptions are rather limited in the sense that it is difficult to generate Gaussian vectors and high dimensional targets in practice are often distributed * Equal Contribution 1 Here, an algorithm is "computationally tractable" if it has provable convergence guarantees.

The notatioñ O(·) omits a logarithm factor of ε −1 .

near a low-dimensional manifold rather than sparse on some given basis.

The goal of this work is to make steps towards addressing these two limitations.

Specifically, we introduce a new framework for robust dithered one-bit compressed sensing where the structure of target vector θ 0 is represented via a ReLU network G :

Building upon this framework, we propose a new recovery algorithm by solving an unconstrained ERM.

We show this algorithm enjoys the following favorable properties:

• Statistically, when taking measurements a i to be sub-exponential random vectors, with high probability and uniformly for any

is the ball of radius R > 0 centered at the origin, the solution G( x m ) to the ERM recovers the true vector G(x 0 ) up to error ε when the number of samples m ≥ O(kn log 4 (ε −1 )(log d + log(ε −1 ))/ε 2 ).

In particular, our result does not require REC type assumptions adopted in previous analysis of generative signal recovery works and at the same time weakens the known sub-Gaussian assumption adopted in previous one-bit compressed sensing works.

When the number of layers n is small, this result meets the minimax optimal rate (up to a logarithm factor) for sparse recovery and simultaneously improves upon the best knownÕ(k log d/ε 4 ) statistical rate for computationally tractable algorithms.

• Computationally, we show that solving the ERM and approximate the true representation x 0 ∈ R k is tractable.

More specifically, we prove with high probability, there always exists a descent direction outside two small neighborhoods around x 0 and its negative multiple with radius O(ε 1/4 ), uniformly for any x 0 ∈ B k 2 (R ) with R = (0.5+ε) −n/2 R, when the ReLU network satisfies a weight distribution condition with parameter ε > 0 and m ≥ O(kn log 4 (ε −1 )(log d + log(ε −1 ))/ε 2 ).

Furthermore, when ε is small enough, one guarantees that the solution x m stays within the neighborhood around x 0 (rather than its negative multiple).

Our result is achieved without assuming the REC type conditions and under quantization errors, thereby improving upon previously known computational guarantees for ReLU generative signal recovery in linear models with small noise.

From a technical perspective, our proof makes use of the special piecewise linearity property of ReLU network.

The merits of such a property in the current scenario are two folds: (1) It allows us to replaces the generic chaining type bounds commonly adopted in previous works (e.g. Dirksen and Mendelson (2018a) ) by novel arguments that are "sub-Gaussian free".

(2) From a hyperplane tessellation point of view, we show that for a given accuracy level, a binary embedding of

2 (R) into Euclidean space is "easier" in that it requires less random hyperplanes than that of a bounded k sparse set.

Notations.

Throughout the paper, let S d−1 and B(x, r) denotes the unit sphere and the ball of radius r centered at

We say a random variable is sub-exponential if its ψ 1 -norm is bounded.

A random vector x ∈ R d is sub-exponential if there exists a a constant C > 0 such that sup t∈S d−1 x, t ψ1 ≤ C. We use x ψ1 to denote the minimal C such that this bound holds.

Furthermore, C, C , c, c 1 , c 2 , c 3 , c 4 , c 5 denote absolute constants, their actual values can be different per appearance.

In this paper, we focus on one-bit recovery model in which one observes quantized measurements of the following form

where a ∈ R d is a random measurement vector, ξ ∈ R is a random pre-quantization noise with an unknown distribution, τ is a random quantization threshold (i.e. dithering noise), and x 0 ∈ R k is the unknown representation to be recovered.

We are interested the high-dimensional scenario where the dimension of the representation space k is potentially much less than the ambient dimension d. The function G : R k → R d is a fixed ReLU neural network of the form:

where σ(x) = max(x, 0) and σ • (x) denotes the entry-wise application of σ(·).

We consider a scenario where the number of layers n is smaller than d,

Throughout the paper, we assume that G(x 0 ) is bounded, i.e. there exists an R ≥ 1 such that G(x 0 ) 2 ≤ R, and we take τ ∼ Uni[−λ, +λ], i.e. a uniform distribution bounded by a chosen parameter λ > 0.

Let {(a i , y i )} m i=1 be i.i.d.

copies of (a, y).

Our goal is to compute an

We propose to solve the following ERM forx m :

where

It is worth mentioning, in general, there is no guarantee that the minimizer of L(x) is unique.

Nevertheless, in Section §2.2, we will show that any solutionx m to this problem must stay inside small neighborhoods around the true signal x 0 and its negative multiple with high probability.

Our statistical guarantee relies on the following assumption on the measurement vector and noise: Assumption 2.1.

The measurement vector a ∈ R d is mean 0, isotropic and sub-exponential.

The noise ξ is also a sub-exponential random variable.

Under this assumption, we have the following main statistical performance theorem: Theorem 2.1.

Suppose Assumption 2.1 holds and consider any ε ∈ (0, 1).

Set the constant C a,ξ,R = max{c 1 (R a ψ1 + ξ ψ1 ), 1}, λ ≥ 4C a,ξ,R · log(64C ψ,ξ,R · ε −1 ) and

Then, with probability at least 1 − c 3 exp(−u), ∀u ≥ 0, any solutionx m to (3) satisfies

for all x 0 such that G(x 0 ) 2 ≤ R, where c 1 , c 2 , c 3 ≥ 1 are absolute constants.

Remark 2.1.

It is easy to verify that the sample complexity enforced by (4) holds when

where C is a large enough absolute constant.

This gives the m = O(kn log 4 (ε −1 )(log d + log(ε −1 ))/ε 2 ) statistical rate.

The question whether or not the dependency on n is redundant (comparing to that of sparse recovery guarantees) warrants further studies.

Remark 2.2.

Note that our result is a uniform recovery result in the sense that the bound G(x m ) − G(x 0 ) 2 ≤ ε holds with high probability uniformly for any target x 0 ∈ R k such that G(x 0 ) 2 ≤ R. This should be distinguished from known bounds (Plan and Vershynin (2013); Zhang et al. (2014) ; ; Thrampoulidis and Rawat (2018) ) on sparse one-bit sensing which hold only for a fixed sparse vector.

Furthermore, though assuming boundedness of G(x 0 ), our recovery algorithm solves for the minimizer without knowing this bound, which is favorable for practice.

Before presenting the results on the global landscape, we first introduce some notations used in the rest of this paper.

For any fixed x, we define W +,x := diag(W x > 0)W , in which we set the rows of W having negative product with x to be zeros.

We further define W i,+,x := diag(W i W i−1,+,x · · · W 1,+,x x > 0)W i , where only active rows of W i are kept.

Thus, we can represent the RuLU network by G(x) = (Π n i=1 W i,+,x )x := W n,+,x W n−1,+,x · · · W 1,+,x x. Definition 2.1 (Weighted Distribution Condition (WDC) ).

The matrix W ∈ R d ×k satisfies the Weighted Distribution Condition with constant ε wdc if for any nonzero vectors x 1 , x 2 ∈ R k ,

where we have θ x,z = ∠(x, z) and Mx ↔ẑ is the matrix that transformsx toẑ,ẑ tox, and ϑ to 0 for any ϑ ∈ span({x, z}) ⊥ .

Here we definex := x/ x 2 ,ẑ := z/ z 2 .

Before presenting Theorem 2.2 and 2.3, we define the directional derivative along non-zero z as

where {x N } is a sequence such that x N → x and L(x) is differentiable at any x N .

Such sequence must exist due to the piecewise linearity of G(x).

For any x such that L(x) is differentiable, the gradient of L(x) is can be easily computed as

Next, we will present Theorem 2.2 to show that under certain conditions, local minimum can only lie in small neighborhoods of two points x 0 and its negative multiple −ρ n x 0 .

Theorem 2.2.

Suppose that G is a ReLU network with W i satisfying WDC with error ε wdc for all i = 1, . . .

, n where n > 1.

With probability 1 − c 1 exp(−u), for any nonzero x 0 satisfying x 0 2 ≤ R(1/2 + ε wdc ) −n/2 , if we set 88πn

wdc , for any nonzero x, set v x = lim x N →x ∇L(x N ) where {x N } is the sequence such that ∇L(x N ) exists for all x N (and v x = ∇L(x) if L(x) is differentiable at x), then, there exists a constant ρ n ≤ 1 such that the directional derivative satisfies:

where ρ n = n−1 i=0

Note that in the above theorem, case 1 indicates that the when the magnitude of the true representation x 0 2 2 is larger than the accuracy level ε wdc , the global minimum lies in small neighborhoods around x 0 and its scalar multiple −ρ n x 0 , while for any point outside the neighborhoods of x 0 and −ρ n x 0 , one can always find a direction with a negative directional derivative.

Note that x = 0 is a local maximum due to D w L(0) < 0 along any non-zero directions w. One the other hand, case 2 implies that when x 0 2 2 is smaller than ε wdc , the global minimum lies in the neighborhood around 0 (and thus around x 0 ).

Moreover, in the following theorem, we further pin down the global minimum of the loss function for case 1.

Theorem 2.3.

Suppose that G is a ReLU network with W i satisfying WDC with error ε wdc for all i = 1, . . .

, n where n > 1.

Assume that c 1 n 3 ε 1/4 wdc ≤ 1 , and x 0 is any nonzero vector satisfying x 0 2 ≤ R(1/2 + ε wdc ) −n/2 .

Then, with probability 1 − 2c 4 exp(−u), for any x 0 such that x 0 2 2 ≥ 2 n ε wdc , setting λ ≥ 4C a,ξ,R · log(64C ψ,ξ,R · ε −1 wdc ), and m ≥ c 2 a 2 ψ1 λ 2 log 2 (λm)(kn log(ed) + k log(2R) + k log m + u)/ε 2 wdc , we have L(x) < L(z), ∀x ∈ B(φ n x 0 , c 3 n −5 x 0 2 ), and ∀z ∈ B(−ζ n x 0 , c 3 n −5 x 0 2 ), where c 1 , c 2 , c 3 , c 4 are absolute constants, φ n , ζ n are any scalars in [ρ n , 1].

Particularly, we have c 3 n −5 < min n≥2 ρ n such that the radius c 3 n −5 x 0 2 < ρ n x 0 2 for any n. Remark 2.3.

The significance of Theorem 2.3 are two folds: first, it shows that the value of ERM is always smaller around x 0 compared to its negative multiple −ρ n x 0 ; second, when the accuracy level ε wdc is small, one can guarantee that the global minimum of L(x) stays around x 0 .

In particular, by Theorem 2.2 and 2.3, our theory implies that if ε wdc ≤ cn −76 for some constant c, then the global minimum of the proposed ERM (3) is in B(φ n x 0 , c 3 n −5 x 0 2 ).

Since we do not focus on optimizing the order of n here, further improvement of such a dependency will be one of our future works.

<|TLDR|>

@highlight

We provide statistical and computational analysis of one-bit compressed sensing problem with a generative prior. 
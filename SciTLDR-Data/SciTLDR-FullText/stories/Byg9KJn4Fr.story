Particle-based inference algorithm is a promising method to efficiently generate samples for an intractable target distribution by iteratively updating a set of particles.

As a noticeable example, Stein variational gradient descent (SVGD) provides a deterministic and computationally efficient update, but it is known to underestimate the variance in high dimensions, the mechanism of which is poorly understood.

In this work we explore a connection between SVGD and MMD-based inference algorithm via Stein's lemma.

By comparing the two update rules, we identify the source of bias in SVGD as a combination of high variance and deterministic bias, and empirically demonstrate that the removal of either factors leads to accurate estimation of the variance.

In addition, for learning high-dimensional Gaussian target, we analytically derive the converged variance for both algorithms, and confirm that only SVGD suffers from the "curse of dimensionality".

The Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016 ) is a deterministic particle-based inference algorithm that iteratively transports the particles by the functional gradient in the reproducing kernel Hilbert space (RKHS) of KL-divergence, which takes the form of a kernelized Stein's operator.

In contrast to the empirical successes (Liu et al., 2017; Haarnoja et al., 2017; Kim et al., 2018) , very few convergence guarantees have been established for SVGD except for the mean-field regime (Liu and Wang, 2018; Lu et al., 2019) .

Moreover, it has been observed that the variance estimated by SVGD scales inversely with the dimensionality of the problem.

This is a highly undesirable property for two reasons: 1) underestimating the variance leads to failures of explaining the uncertainty of model predictions; 2) modern inference problems are usually high-dimensional.

For example, Bayesian neural networks (MacKay, 1992) could be more than millions of dimensions.

We study the algorithmic bias of SVGD that leads to the variance underestimation in high dimensions.

We construct another kernel-based inference algorithm termed MMDdescent, which closely resembles SVGD but estimate the variance accurately.

By comparing their updates, we identify the cause of variance collapse in SVGD as a combination of high variance due to Stein's lemma, and deterministic bias, i.e. the inability to resample particles.

We empirically verify that removing either of these two factors, while computationally expensive, leads to accurate variance estimation.

Then, under mild assumptions, we derive the equilibrium variance of SVGD and MMD-descent in matching high-dimensional Gaussians, and confirm that variance estimated by SVGD scales inversely with the dimensionality.

Particle variational inference approximates an intractable distribution p(x) with a set of particles X = {x i } n i=1 .

Specifically, we iteratively optimize a set of particles

under the deterministic update:

, where is the stepsize, and ∆(·) : R d → R d represents the update direction.

SVGD defines the update direction as

driving force

repulsive force

where k is a positive definite kernel.

Intuitively, the log derivative term S 1 (x i , x) corresponds to a driving force that guides particles towards high likelihood regions, whereas the kernel derivative term S 2 (x i , x) provides a repulsive force to prevent the particles from collapsing.

We now introduce another particle inference algorithm MMD-descent, motivated from kernel herding (Welling, 2009) .

Instead of selecting particles one by one to minimize the mean maximum discrepancy (MMD) (Gretton et al., 2012) with respect to the target, we jointly optimize all particles together to minimize MMD via gradient descent.

For symmetric kernel, such as the Euclidean distance kernel (Definition 3), the update can be written as:

Note that MMD-descent is not practical since integration under the target distribution is usually infeasible.

When the kernel k is in the Stein class of p, we have the equivalence:

SVGD vs. MMD-descent.

Observe that 1) the repulsive force term in SVGD and MMD-descent is identical; 2) in MMD-descent, the driving force is integrated under the target distribution p, whereas in SVGD under the current particle distribution q.

It is clear that at the infinite particle limit, q = p is the fixed point for both updates.

However, such asymptotic property does not entail 1) the two algorithms reliably approximate the target distribution in high dimensions, i.e. when n, d are both large ; 2) the two algorithms converges to similar stationary points under finite samples.

Given the similar updates, it is natural to ask: do SVGD and MMD-descent approximate the target distribution reliably in high dimensions and are their approximations similar?

The answer is in the negative: SVGD and MMD-descent converge to different stationary points.

Specifically, SVGD underestimates the marginal variance in high dimensions (Zhuo et al., 2017) .

For unit Gaussian targets, although both algorithms correctly estimates the mean, Figure 1 (a) illustrates that SVGD particles have decreasing marginal variance as dimensionality increases whereas MMD-descent particles approximate the marginal variance accurately.

In the following we characterize the sources of this pitfall of SVGD.

Variance from Integration by Parts The convergence of SVGD crucially depends on the equality E y∼p [S 1 (y, x)] = −E y∼p [S 2 (y, x)] obtained via integration by parts.

Nevertheless, the variance of S 1 and S 2 may differ drastically, hence invoking convergence issues under finite particles.

In general, S 1 can have much larger magnitude, resulting in large variance of the driving force term in SVGD.

In contrast, in MMD-descent the driving force can be computed via integration S 2 , and thus the high variance term is not involved.

We visualize the difference in the variance of estimating S 1 and S 2 in Figure 1(b) , and provide the following characterization for the Gaussian RBF kernel and Gaussian target.

Proposition 1 Define the mean squared error as: High variance of S 1 entails that larger number of samples is required to accurately estimate S 1 .

Due to this discrepancy, we expect SVGD to better approximate the target if more samples are used to estimate the driving force.

Assume we keep rn (r > 1) particles, use all of them to estimate S 1 and n particles for S 2 .

The modified update is given as

Where

.

This is to say, at each step the transport map is constructed via estimating the repulsive force with n particles and the driving force with rn particles.

As shown in Figure 1(b) , even though the repulsive force S 2 is calculated with few samples, when r ∈ Ω(d) the algorithm accurately estimates the variance (independent of d).

We comment that although the modification corrects the variance collapse, it is not practical since the required number of particles scales with the dimensionality.

Bias from Deterministic Update.

The analysis above suggests that the pitfall of SVGD relates to the high variance of the driving force S 1 , which is not present in MMD-descent.

However in scenarios like gradient estimation of variational inference, high variance usually results in slower convergence, but not necessarily variance collapse.

In SVGD, the particles {x i } n i=1 used to compute the update are assumed as random samples from an underlying continuous distribution.

However, due to the deterministic update, the distribution q is entirely represented by the same set of particles and drawing random samples is not possible.

We now demonstrate that this deterministic bias, when combined with high variance estimators, may cause the algorithm to converge to biased target or even diverge.

We start from an illustrative experiment of deterministic bias with MMD-descent.

Given target samples {y i } m i=1 ∼ p(y), we have two forms of update that differs in the driving force.

As argued above ∆ MMD 1 tends to have higher variance due to S 1 .

Note that the estimation of E p(y) [S 2 (y, x)] is unbiased when y i is resampled at each iteration, and empirically the converged variance is unbiased indeed (log derivative (resampled) in Figure 2 (a)).

To simulate the deterministic bias, we sample {y i } m i=1 ∼ p(y) and keep them f ixed throughout optimization.

As shown in Figure 2 can estimate the variance accurately (kernel derivative (fixed)), but ∆ MMD 1 diverges (log derivative (fixed)).

As expected, the deterministic bias of estimating S 1 is more significant in ∆ MMD 1 due to high variance.

This experiment shows that the deterministic bias in SVGD arises from the algorithm not being able to resample from q. We now design an algorithm to achieve random "resampling".

Let q 0 be the continuous distribution where initial samples are drawn from.

Let the particles at t-th iteration beq t .

We randomly draw new samplesq 0 from q 0 .

At the i-th iteration, we updateq i with ∆ SVGD using the map defined by particlesq i .

Because bothq T andq T are initially sampled from q 0 and transported using the same map defined by {q i } T −1 i=0 ,q T and q T has the same distribution.

Soq T can be seen as "resampled" from the same distribution asq T , and we can useq T for updatingq T in SVGD without the deterministic bias.

The algorithm is reminiscent of flow-based variational inference (Rezende and Mohamed, 2015) and transport-based particle gradient descent (Nitanda and Suzuki, 2017) algorithms.

As shown in Figure 2 (b), SVGD with this resampling scheme accurately estimates the target variance with a small number of particles being updated at each iteration (n = 10).

We expect similar outcomes in estimating higher order moments since the algorithm is completely unbiased.

But the computational cost of such resampled updates scales quadratically with the number of iterations, thus rendering the method impractical in real applications.

Analytically Deriving the Variance.

We now quantitatively characterize the variance collapse in SVGD by deriving the variance of the converged particles in learning a unit Gaussian in high dimensions.

Specifically, we consider the setup where n, d tend to infinity at the same rate.

Various works have shown that in this regime the kernel matrix can be asymptotically decomposed into a weighted sum of the data covariance matrix and a scaled identity (El Karoui et al., 2010; Cheng and Singer, 2013; Bordenave et al., 2013) .

We perform a similar decomposition via Taylor expansion and obtain the following characterization:

Proposition 2 (Informal) For unit Gaussian target and Gaussian RBF kernel, assume that particles at the fixed point of both algorithms correlate weakly and have concentrated norm, then as d,n → ∞ and lim n→∞ n/d = γ ∈ (0, 1), particles driven by SVGD (7) equilibrates at the marginal variance v SVGD →

1 e−1 γ, whereas MMD-descent (2) leads to v MMD → 1.

Empirical results in Figure 2 (c) align with the prediction: when d > n, the equilibrium variance of SVGD scales linearly with n but is also inverse to d. This indicates that as the dimensionality increases, more particles is required to reliably estimate the true variance.

When γ > 1, the variance empirically approaches the target variance from below as γ increases.

On the other hand, in this regime MMD-descent does not underestimate the variance for all γ.

To measuring how well a set of samples approximates a target distribution p, one may consider the maximum discrepancy between the target p and sample distribution q over some function class F:

which is known as the integral probability metric (IPM) (Müller, 1997).

In particular, if F is a unit ball in the reproducing kernel Hilbert space (RKHS) H, the resulting D F is termed the maximum mean discrepancy (MMD) (Gretton et al., 2012) , and its squared value MMD 2 (p, q) is given as µ p − µ q 2 H , which equals to:

where x, x ∼ p, y, y ∼ q, and k : Sriperumbudur et al., 2011) , which includes the commonly-used Gaussian RBF kernel, then MMD defines a proper metric.

In this work we mainly focus on the Euclidean distance kernel defined as Definition 3 (Euclidean Distance Kernel) A positive semi-definite kernel function is called Euclidean Distance kernel if it can be represented as:

In particular, the commonly-used Gaussian kernels and IMQ kernels are both Euclidean Distance kernels.

In practice, σ 2 scales with d for normalization.

Stein's Lemma.

When integration under p is difficult, Stein's method (Stein et al., 1972) can be used to construct zero-mean test functions w.r.t p.

Specifically, for differentiable function f in the Stein Class of p, i.e.,

The following identity holds:

where A p is termed the Langevin Stein operator (Gorham and Mackey, 2015) , as it arises from applying the generator method (Barbour, 1988) to the overdamped Langevin diffusion.

This identity can be easily verified via integration by parts, given that (f · p) vanishes at boundary.

This modified IPM is called the Stein's discrepancy:

Note that the Stein's discrepancy only involves the score of p and thus the normalization constant is not required.

When f is restricted in the product RKHS H d with inner product

the maximum discrepancy, known as kernel Stein discrepancy (KSD), can be estimated efficiently from samples (Liu et al., 2016 )(Chwialkowski et al., 2016 (Gorham et al., 2016) .

We now consider the approximation of an intractable distribution p(x) with a set of particles X = {x i } n i=1 representing a Dirac mixture.

To generate these particles, kernel herding was introduced by (Welling, 2009) for minimizing the MMD between the particles and the target distribution.

The herding algorithm proceeds in a greedy manner; Assume the algorithm already selects {x 1 , · · · , x n−1 }, the next particle is chosen based on:

Intuitively, the first term encourages sampling in high density areas for the target distribution.

The second term discourages sampling at points close to existing samples.

It is shown (Welling, 2009; Bach et al., 2012; Huszár and Duvenaud, 2012 ) that the kernel herding algorithm reduces the MMD at a rate O( 1 N ), for finite-dimensional Hilbert spaces H.

The herding procedure selects particles greedily to minimize its objective MMD 2 , adding particles one at a time.

One can also jointly optimize all particles to decrease some notion of distance.

Let q [ ∆] be the distribution of particles after update ∆. SVGD constructs the update direction that maximally decreases the KL divergence:

Constrain ∆ in terms of RKHS norm, the update for each particle x can be computed as:

Note that this update rule can also be interpreted as a fixed-point iteration on Stein's discrepancy (6).

The typically-used kernels include Gaussian kernel k(x, x ) = exp(−

Stein's Lemma.

Stein's lemma provides powerful tools in approximating probability distributions and specifying convergence rates (Erdogdu, 2016; Chen et al., 2018) .

In particular, via Stein's lemma, SVGD (Liu and Wang, 2016) derives an explicit particle updating formula by minimizing the KL divergence with unnormalized targets.

With the research of implicit variational inference (Huszár, 2017), Stein's lemma also flourishes score estimation methods (Li and Turner, 2017; Shi et al., 2018 ) using only random samples from an implicit distribution.

Interestingly, Erdogdu et al. (2016) observed that algorithms that are equivalent in expectation via Stein's lemma might have different convergence properties, which aligns with our analysis in Section 3.

The "curse of dimensionality" of Stein's lemma-based kernel algorithm has also been studied in Oates et al. (2016) .

A.3.

Detail of SVGD with resampling scheme:

Proposition 4 (Fixed-Sample MMD Convergence) Let Y = {y i } m i=1 be n independent random samples from target distribution p. Assume the kernel is bounded by 0 ≤ k(z, z ) ≤ K. Let X = {x i } n i=1 be the optimum performing MMD updates based on ∆ MMD 2 using samples Y .

We have

Proof:

Where A is because that X attains smallest MMD with Y for all m particles, thus its MMD is no-smaller than m random samples Z from distribution p. B follows from Gretton et al. (2012) .

Now based on Markov's Inequality, we have for any > 0,

Similarly for the kernel derivative S 2 we have,

The simplification above largely follows from E x∼N (µ,Σ) [ x 2 2 ] = µ T µ + T r(Σ).

Given the bandwidth heuristic σ ∈ Θ( √ d) and x 2 = d, one can easily obtain:

In this section we aim to calculate the variance of SVGD and MMD-Descent in learning unit Gaussian target under the scaling of n, d → ∞ with lim d,n→∞ n/d = γ ∈ (0, ∞).

Since both SVGD and MMD-descent form an interacting particle system, one can no longer treat the converged particles as i.i.d.

samples from some distribution.

We therefore assume the following on the converged fixed point, which essentially entails that the particles spread evenly in the space, have concentrated norm and only correlate weakly.

Assumption A1.

Unit Gaussian Target Distribution: p(y) ∝ exp − 1 2 y y ; Gaussian RBF Kernel: k(x, y) = exp −

Under assumption A1 and A2, we are able to compute the asymptotic variance of both SVGD and MMD-descent.

We first calculate the SVGD variance with d, n → ∞, n/d → γ.

In this subsection we consider the asymptotical scaling of n, d where n, d → ∞ and n/d → γ.

We solve the stationary point of SVGD update Eq (7), where

Therefore for Eq (12) we have for LHS

where

e. the k-th column of kernel K ∈ R n×n ).

As for the RHS of Eq (12), note that assumption A2 ensures the following Taylor expansion around its concentrated value for i = k

and for i = k we have k(x k , x k ) = 1.

This immediately gives

Equating the RHS and LHS of Eq (12) in matrix form (over all k) we have

where m = (n + e − 1)/(dv + 1)e and diag( ) is a square matrix where the i-th diagonal is i with i = O( ).

Therefore X · (K − mI n − diag( )) = 0.

Denote A = K − mI n − diag( ), Note that the K is an Euclidean random kernel matrix with K ij = k(x i , x k ) = exp −(2dv) −1 x i − x k 2 2 , from Theorem 4 in (Bordenave et al., 2013) it follows that the empirical spectrum density of A converges weakly to

where the empirical spectrum of a random matrix A ∈ R n×n is defined as µ = n −1 n i=1 δ λ i (A) .

Moreover, denote S = n/(n−1)I n −1/(n−1)1 n 1 n , then by the Hoffman-Wielandt inequality one has

where W 2 (·, ·) is the 2-Wasserstein distance.

Hence

→ 1 − 2 e + n e(n − 1) − m.

When γ > 1 i.e. d > n, Equation (17) The stationary point of MMD-descent satisfies for ∀k,

i.e.

Under assumption A1, similar to the SVGD case, we have the matrix form of the equilibrium particles 1 + e −1 (n − 1) − dv 1 + dv

@highlight

Analyze the underlying mechanisms of variance collapse of SVGD in high dimensions.
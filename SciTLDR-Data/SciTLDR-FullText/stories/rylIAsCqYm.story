In this paper, we propose the Asynchronous Accelerated Nonuniform Randomized Block Coordinate Descent algorithm (A2BCD).

We prove A2BCD converges linearly to a solution of the convex minimization problem at the same rate as NU_ACDM, so long as the maximum delay is not too large.

This is the first asynchronous Nesterov-accelerated algorithm that attains any provable speedup.

Moreover, we then prove that these algorithms both have optimal complexity.

Asynchronous algorithms complete much faster iterations, and A2BCD has optimal complexity.

Hence we observe in experiments that A2BCD is the top-performing coordinate descent algorithm, converging up to 4-5x faster than NU_ACDM on some data sets in terms of wall-clock time.

To motivate our theory and proof techniques, we also derive and analyze a continuous-time analog of our algorithm and prove it converges at the same rate.

In this paper, we propose and prove the convergence of the Asynchronous Accelerated Nonuniform Randomized Block Coordinate Descent algorithm (A2BCD), the first asynchronous Nesterovaccelerated algorithm that achieves optimal complexity.

No previous attempts have been able to prove a speedup for asynchronous Nesterov acceleration.

We aim to find the minimizer x * of the unconstrained minimization problem: DISPLAYFORM0 f (x) = f x (1) , . . .

, x (n) ( FORMULA52 where f is σ-strongly convex for σ > 0 with L-Lipschitz gradient ∇f = (∇ 1 f, . . . , ∇ n f ).

x ∈ R d is composed of coordinate blocks x (1) , . . .

, x (n) .

The coordinate blocks of the gradient ∇ i f are assumed L i -Lipschitz with respect to the ith block.

That is, ∀x, h ∈ R d : DISPLAYFORM1 where P i is the projection onto the ith block of R d .

LetL 1 n n i=1 L i be the average block Lipschitz constant.

These conditions on f are assumed throughout this whole paper.

Our algorithm can also be applied to non-strongly convex objectives (σ = 0) or non-smooth objectives using the black box reduction techniques proposed in BID1 .

Hence we consider only the coordinate smooth, strongly-convex case.

Our algorithm can also be applied to the convex regularized ERM problem via the standard dual transformation (see for instance Lin et al. (2014) ): DISPLAYFORM2 Hence A2BCD can be used as an asynchronous Nesterov-accelerated finite-sum algorithm.

Coordinate descent methods, in which a chosen coordinate block i k is updated at every iteration, are a popular way to solve equation 1.1.

Randomized block coordinate descent (RBCD, Nesterov (2012) ) updates a uniformly randomly chosen coordinate block i k with a gradient-descent-like step: DISPLAYFORM3 The complexity K( ) of an algorithm is defined as the number of iterations required to decrease the error E(f (x k ) − f (x * )) to less than (f (x 0 ) − f (x * )).

Randomized coordinate descent has a complexity of K( ) = O(n(L/σ) ln(1/ )).Using a series of averaging and extrapolation steps, accelerated RBCD Nesterov (2012) improves RBCD's iteration complexity K( ) to O(n L /σ ln(1/ )), which leads to much faster convergence whenL σ is large.

This rate is optimal when all L i are equal Lan & Zhou (2015) .

Finally, using a special probability distribution for the random block index i k , the non-uniform accelerated coordinate descent method BID2 (NU_ACDM) can further decrease the complexity to O( DISPLAYFORM4 L i /σ ln(1/ )), which can be up to √ n times faster than accelerated RBCD, since some L i can be significantly smaller than L. NU_ACDM is the current state-of-the-art coordinate descent algorithm for solving equation 1.1.Our A2BCD algorithm generalizes NU_ACDM to the asynchronous-parallel case.

We solve equation 1.1 with a collection of p computing nodes that continually read a shared-access solution vector y into local memory then compute a block gradient ∇ i f , which is used to update shared solution vectors (x, y, v) .

Proving convergence in the asynchronous case requires extensive new technical machinery.

A traditional synchronous-parallel implementation is organized into rounds of computation: Every computing node must complete an update in order for the next iteration to begin.

However, this synchronization process can be extremely costly, since the lateness of a single node can halt the entire system.

This becomes increasingly problematic with scale, as differences in node computing speeds, load balancing, random network delays, and bandwidth constraints mean that a synchronous-parallel solver may spend more time waiting than computing a solution.

Computing nodes in an asynchronous solver do not wait for others to complete and share their updates before starting the next iteration.

They simply continue to update the solution vectors with the most recent information available, without any central coordination.

This eliminates costly idle time, meaning that asynchronous algorithms can be much faster than traditional ones, since they have much faster iterations.

For instance, random network delays cause asynchronous algorithms to complete iterations Ω(ln(p)) time faster than synchronous algorithms at scale.

This and other factors that influence the speed of iterations are discussed in Hannah & Yin (2017a) .

However, since many iterations may occur between the time that a node reads the solution vector, and the time that its computed update is applied, effectively the solution vector is being updated with outdated information.

At iteration k, the block gradient ∇ i k f is computed at a delayed iterateŷ k defined as 1 : DISPLAYFORM5 for delay parameters j(k, 1), . . .

, j(k, n) ∈ N. Here j(k, i) denotes how many iterations out of date coordinate block i is at iteration k. Different blocks may be out of date by different amounts, which is known as an inconsistent read.

We assume 2 that j(k, i) ≤ τ for some constant τ < ∞.Asynchronous algorithms were proposed in Chazan & Miranker (1969) to solve linear systems.

General convergence results and theory were developed later in BID5 ; Bertsekas & Tsitsiklis (1997); Tseng et al. (1990); Luo & Tseng (1992; 1993); Tseng (1991) There is also a rich body of work on asynchronous SGD.

In the distributed setting, Zhou et al. (2018) showed global convergence for stochastic variationally coherent problems even when the delays grow at a polynomial rate.

In Lian et al. (2018) , an asynchronous decentralized SGD was proposed with the same optimal sublinear convergence rate as SGD and linear speedup with respect to the number of workers.

In Liu et al. (2018) , authors obtained an asymptotic rate of convergence for asynchronous momentum SGD on streaming PCA, which provides insight into the tradeoff between asynchrony and momentum.

In Dutta et al. (2018) , authors prove convergence results for asynchronous SGD that highlight the tradeoff between faster iterations and iteration complexity.

Further related work is discussed in Section 4.

In this paper, we prove that A2BCD attains NU_ACDM's state-of-the-art iteration complexity to highest order for solving equation 1.1, so long as delays are not too large (see Section 2).

The proof is very different from that of BID2 , and involves significant technical innovations and complexity related to the analysis of asynchronicity.

We also prove that A2BCD (and hence NU_ACDM) has optimal complexity to within a constant factor over a fairly general class of randomized block coordinate descent algorithms (see Section 2.1).

This extends results in Lan & Zhou (2015) to asynchronous algorithms with L i not all equal.

Since asynchronous algorithms complete faster iterations, and A2BCD has optimal complexity, we expect A2BCD to be faster than all existing coordinate descent algorithms.

We confirm with numerical experiments that A2BCD is the current fastest coordinate descent algorithm (see Section 5).We are only aware of one previous and one contemporaneous attempt at proving convergence results for asynchronous Nesterov-accelerated algorithms.

However, the first is not accelerated and relies on extreme assumptions, and the second obtains no speedup.

Therefore, we claim that our results are the first-ever analysis of asynchronous Nesterov-accelerated algorithms that attains a speedup.

Moreover, our speedup is optimal for delays not too large 3 .The work of Meng et al. claims to obtain square-root speedup for an asynchronous accelerated SVRG.In the case where all component functions have the same Lipschitz constant L, the complexity they obtain reduces to (n + κ) ln(1/ ) for κ = O τ n 2 (Corollary 4.4).

Hence authors do not even obtain accelerated rates.

Their convergence condition is τ < In a contemporaneous preprint, authors in Fang et al. (2018) skillfully devised accelerated schemes for asynchronous coordinate descent and SVRG using momentum compensation techniques.

Although their complexity results have the improved √ κ dependence on the condition number, they do not prove any speedup.

Their complexity is τ times larger than the serial complexity.

Since τ is necessarily greater than p, their results imply that adding more computing nodes will increase running time.

The authors claim that they can extend their results to linear speedup for asynchronous, accelerated SVRG under sparsity assumptions.

And while we think this is quite likely, they have not yet provided proof.

We also derive a second-order ordinary differential equation (ODE), which is the continuous-time limit of A2BCD (see Section 3).

This extends the ODE found in Su et al. (2014) to an asynchronous accelerated algorithm minimizing a strongly convex function.

We prove this ODE linearly converges to a solution with the same rate as A2BCD's, without needing to resort to the restarting techniques.

The ODE analysis motivates and clarifies the our proof strategy of the main result.

We should consider functions f where it is efficient to calculate blocks of the gradient, so that coordinate-wise parallelization is efficient.

That is, the function should be "coordinate friendly" Peng et al. (2016b) .

This is a very wide class that includes regularized linear regression, logistic regression, etc.

The L 2 -regularized empirical risk minimization problem is not coordinate friendly in general, however the equivalent dual problem is, and hence can be solved efficiently by A2BCD (see Lin et al. (2014) , and Section 5).To calculate the k + 1'th iteration of the algorithm from iteration k, we use only one block of the gradient ∇ i k f .

We assume that the delays j(k, i) are independent of the block sequence i k , but otherwise arbitrary (This is a standard assumption found in the vast majority of papers, but can be relaxed Sun et al. (2017) ; Leblond et al. (2017); Cannelli et al. (2017) ).

Definition 1.

Asynchronous Accelerated Randomized Block Coordinate Descent (A2BCD).

Let f be σ-strongly convex, and let its gradient ∇f be L-Lipschitz with block coordinate Lipschitz parameters L i as in equation 1.2.

We define the condition number κ = L/σ, and let L = min i L i .

Using these parameters, we sample i k in an independent and identically distributed (IID) fashion according to DISPLAYFORM0 Let τ be the maximum asynchronous delay.

We define the dimensionless asynchronicity parameter ψ, which is proportional to τ , and quantifies how strongly asynchronicity will affect convergence: DISPLAYFORM1 We use the above system parameters and ψ to define the coefficients α, β, and γ via eqs. (2.3) to (2.5).

Hence A2BCD algorithm is defined via the iterations: eqs. (2.6) to (2.8).

DISPLAYFORM2 See Section A for a discussion of why it is practical and natural to have the gradient DISPLAYFORM3 DISPLAYFORM4 Here we define y k = y 0 for all k < 0.

The determination of the coefficients c i is in general a very involved process of trial and error, intuition, and balancing competing requirements.

The algorithm doesn't depend on the coefficients, however; they are only an analytical tool.

We define DISPLAYFORM5 To simplify notation 4 , we assume that the minimizer x * = 0, and that f (x * ) = 0 with no loss in generality.

We define the Lyapunov function: DISPLAYFORM6 We now present this paper's first main contribution.

Theorem 1.

Let f be σ-strongly convex with a gradient ∇f that is L-Lipschitz with block Lipschitz constants DISPLAYFORM7 ).

Then for A2BCD we have: DISPLAYFORM8 To obtain E[ρ k ]

≤ ρ 0 , it takes K A2BCD ( ) iterations for: 13) where O(·) is asymptotic with respect to σ −1/2 S → ∞, and uniformly bounded.

DISPLAYFORM9 This result is proven in Section B. A stronger result for L i ≡ L can be proven, but this adds to the complexity of the proof; see Section E for a discussion.

In practice, asynchronous algorithms are far more resilient to delays than the theory predicts.

τ can be much larger without negatively affecting the convergence rate and complexity.

This is perhaps because we are limited to a worst-case analysis, which is not representative of the average-case performance.

Allen-Zhu et al. FORMULA15 (Theorem 5.1) shows a linear convergence rate of 1 − 2/ 1 + 2σ −1/2 S for NU_ACDM, which leads to the corresponding iteration complexity of DISPLAYFORM10 .

Hence, we have: DISPLAYFORM11 We can assume x * = 0 with no loss in generality since we may translate the coordinate system so that x * is at the origin.

We can assume f (x * ) = 0 with no loss in generality, since we can replace f (x) with f (x)−f (x * ).

Without this assumption, the Lyapunov function simply becomes: DISPLAYFORM12 Published as a conference paper at ICLR 2019 DISPLAYFORM13 , the complexity of A2BCD asymptotically matches that of NU_ACDM.

Hence A2BCD combines state-of-the-art complexity with the faster iterations and superior scaling that asynchronous iterations allow.

We now present some special cases of the conditions on the maximum delay τ required for good complexity.

Remark 1.

Reduction to synchronous case.

Notice that when τ = 0, we have ψ = 0, c i ≡ 0 and hence A k ≡ 0.

Thus A2BCD becomes equivalent to NU_ACDM, the Lyapunov function 5 ρ k becomes equivalent to one found in BID2 (pg.

9), and Theorem 1 yields the same complexity.

The maximum delay τ will be a function τ (p) of p, number of computing nodes.

Clearly τ ≥ p, and experimentally it has been observed that τ = O(p) Leblond et al. (2017) .

Let gradient complexity K( , τ ) be the number of gradients required for an asynchronous algorithm with maximum delay τ to attain suboptimality .

τ (1) = 0, since with only 1 computing node there can be no delay.

This corresponds to the serial complexity.

We say that an asynchronous algorithm attains a complexity speedup if DISPLAYFORM0 is increasing in p.

We say it attains linear complexity speedup if DISPLAYFORM1 In Theorem 1, we obtain a linear complexity speedup (for p not too large), whereas no other prior attempt can attain even a complexity speedup with Nesterov acceleration.

In the ideal scenario where the rate at which gradients are calculated increases linearly with p, algorithms that have linear complexity speedup will have a linear decrease in wall-clock time.

However in practice, when the number of computing nodes is sufficiently large, the rate at which gradients are calculated will no longer be linear.

This is due to many parallel overhead factors including too many nodes sharing the same memory read/write bandwidth, and network bandwidth.

However we note that even with these issues, we obtain much faster convergence than the synchronous counterpart experimentally.

NU_ACDM and hence A2BCD are in fact optimal in some sense.

That is, among a fairly wide class of coordinate descent algorithms A, they have the best-possible worst-case complexity to highest order.

We extend the work in Lan & Zhou (2015) to encompass algorithms are asynchronous and have unequal L i .

For a subset S ∈ R d , we let IC(S) (inconsistent read) denote the set of vectors v whose components are a combination of components of vectors in the set S. DISPLAYFORM0 Definition 4.

Asynchronous Randomized Incremental Algorithms.

Consider the unconstrained minimization problem equation 1.1 for function f satisfying the conditions stated in Section 1.

We define the class A as algorithms G on this problem such that: DISPLAYFORM1 This is a rather general class: x k+1 can be constructed from any inconsistent reading of past iterates IC(X k ), and any past gradient of an inconsistent read ∇ ij f (IC(X j )).

DISPLAYFORM2 Hence A has a complexity lower bound: DISPLAYFORM3 Our proof in Section D follows very similar lines to Lan & Zhou (2015) ; Nesterov (2013).

In this section we present and analyze an ODE which is the continuous-time limit of A2BCD.

This ODE is a strongly convex, and asynchronous version of the ODE found in Su et al. (2014) .

For simplicity, assume L i = L, ∀i.

We rescale (I.e. we replace f (x) with 1 σ f .) f so that σ = 1, and hence κ = L/σ = L. Taking the discrete limit of synchronous A2BCD (i.e. accelerated RBCD), we can derive the following ODE 6 (see Section equation C.1): DISPLAYFORM0 We define the parameter η nκ 1/2 , and the energy: DISPLAYFORM1 .

This is very similar to the Lyapunov function discussed in equation 2.11, with DISPLAYFORM2 the role of v k 2 , and A k = 0 (since there is no delay yet).

Much like the traditional analysis in the proof of Theorem 1, we can derive a linear convergence result with a similar rate.

See Section C.2.

We may also analyze an asynchronous version of equation 3.1 to motivate the proof of our main theorem.

HereŶ (t) is a delayed version of Y (t) with the delay bounded by τ .

DISPLAYFORM0 Unfortunately, this energy satisfies (see Section equation C.4, equation C.7): DISPLAYFORM1 Hence this energy E(t) may not be decreasing in general.

But, we may add a continuous-time asynchronicity error (see Sun et al. (2017) ), much like in Definition 2, to create a decreasing energy.

Let c 0 ≥ 0 and r > 0 be arbitrary constants that will be set later.

Define: DISPLAYFORM2 Lemma 6.

When rτ ≤ 1 2 , the asynchronicity error A(t) satisfies: DISPLAYFORM3 DISPLAYFORM4 Hence f (Y (t)) convergence linearly to f (x * ) with rate O exp −t/(nκ 1/2 ) Notice how this convergence condition is similar to Corollary 3, but a little looser.

The convergence condition in Theorem 1 can actually be improved to approximately match this (see Section E).Proof.

DISPLAYFORM5 The preceding should hopefully elucidate the logic and general strategy of the proof of Theorem 1.

We now discuss related work that was not addressed in Section 1.

Nesterov acceleration is a method for improving an algorithm's iteration complexity's dependence the condition number κ.

FORMULA15 showed that many of the assumptions used in prior work (such as bounded delay τ < ∞) were unrealistic and unnecessary in general.

In Hannah & Yin (2017a) the authors showed that asynchronous iterations will complete far more iterations per second, and that a wide class of asynchronous algorithms, including asynchronous RBCD, have the same iteration complexity as their synchronous counterparts.

Hence certain asynchronous algorithms can be expected to significantly outperform traditional ones.

In Xiao et al. (2017) authors propose a novel asynchronous catalyst-accelerated BID6 primal-dual algorithmic framework to solve regularized ERM problems.

They structure the parallel updates so that the data that an update depends on is up to date (though the rest of the data may not be).

However catalyst acceleration incurs a log(κ) penalty over Nesterov acceleration in general.

In BID0 , the author argues that the inner iterations of catalyst acceleration are hard to tune, making it less practical than Nesterov acceleration.

To investigate the performance of A2BCD, we solve the ridge regression problem.

Consider the following primal and corresponding dual objective (see for instance Lin et al. (2014) ): DISPLAYFORM0 where A ∈ R d×n is a matrix of n samples and d features, and l is a label vector.

We let A = [A 1 , . . .

, A m ] where A i are the column blocks of A. We compare A2BCD (which is asynchronous accelerated), synchronous NU_ACDM (which is synchronous accelerated), and asynchronous RBCD (which is asynchronous non-accelerated).

Nodes randomly select a coordinate block according to equation 2.1, calculate the corresponding block gradient, and use it to apply an update to the shared solution vectors.

synchronous NU_ACDM is implemented in a batch fashion, with batch size p (1 block per computing node).

Nodes in synchronous NU_ACDM implementation must wait until all nodes apply their computed gradients before they can start the next iteration, but the asynchronous algorithms simply compute with the most up-to-date information available.

We use the datasets w1a (47272 samples, 300 features), wxa which combines the data from from w1a to w8a (293201 samples, 300 features), and aloi (108000 samples, 128 features) from LIBSVM Chang & Lin (2011) .

The algorithm is implemented in a multi-threaded fashion using C++11 and GNU Scientific Library with a shared memory architecture.

We use 40 threads on two 2.5GHz 10-core Intel Xeon E5-2670v2 processors.

See Section A.1 for a discussion of parameter tuning and estimation.

The parameters for each algorithm are tuned to give the fastest performance, so that a fair comparison is possible.

A critical ingredient in the efficient implementation of A2BCD and NU_ACDM for this problem is the efficient update scheme discussed in Lee & Sidford (2013b; a) .

In linear regression applications such as this, it is essential to be able to efficiently maintain or recover Ay.

This is because calculating block gradients requires the vector A T i Ay, and without an efficient way to recover Ay, block gradient evaluations are essentially 50% as expensive as full-gradient calculations.

Unfortunately, every accelerated iteration results in dense updates to y k because of the averaging step in equation 2.6.

Hence Ay must be recalculated from scratch.

However Lee & Sidford (2013a) introduces a linear transformation that allows for an equivalent iteration that results in sparse updates to new iteration variables p and q. The original purpose of this transformation was to ensure that the averaging steps (e.g. equation 2.6) do not dominate the computational cost for sparse problems.

However we find a more important secondary use which applies to both sparse and dense problems.

Since the updates to p and q are sparse coordinate-block updates, the vectors Ap, and Aq can be efficiently maintained, and therefore block gradients can be efficiently calculated.

The specifics of this efficient implementation are discussed in Section A.2.In Table 5 , we plot the sub-optimality vs. time for decreasing values of λ, which corresponds to increasingly large condition numbers κ.

When κ is small, acceleration doesn't result in a significantly better convergence rate, and hence A2BCD and async-RBCD both outperform sync-NU_ACDM since they complete faster iterations at similar complexity.

Acceleration for low κ has unnecessary overhead, which means async-RBCD can be quite competitive.

When κ becomes large, async-RBCD is no longer competitive, since it has a poor convergence rate.

We observe that A2BCD and sync-NU_ACDM have essentially the same convergence rate, but A2BCD is up to 4 − 5× faster than sync-NU_ACDM because it completes much faster iterations.

We observe this advantage despite the fact that we are in an ideal environment for synchronous computation: A small, homogeneous, high-bandwidth, low-latency cluster.

In large-scale heterogeneous systems with greater synchronization overhead, bandwidth constraints, and latency, we expect A2BCD's advantage to be much larger.

TAB4 : Sub-optimality f (y k ) − f (x * ) (y-axis) vs time in seconds (x-axis) for A2BCD, synchronous NU_ACDM, and asynchronous RBCD for data sets w1a, wxa and aloi for various values of λ.

An efficient implementation will have coordinate blocks of size greater than 1.

This to ensure the efficiency of linear algebra subroutines.

Especially because of this, the bulk of the computation for each iteration is computing ∇ i k f (ŷ k ), and not the averaging steps.

Hence the computing nodes only need a local copy of y k in order to do the bulk of an iteration's computation.

Given this gradient ∇ i k f (ŷ k ), updating y k and v k is extremely fast (x k can simply be eliminated).

Hence it is natural to simply store y k and v k centrally, and update them when the delayed gradients ∇ i k f (ŷ k ).

Given the above, a write mutex over (y, v) has minuscule overhead (which we confirm with experiments), and makes the labeling of iterates unambiguous.

This also ensures that v k and y k are always up to date when (y, v) are being updated.

Whereas the gradient ∇ i k f (ŷ k ) may at the same time be out of date, since it has been calculated with an outdated version of y k .

However a write mutex is not necessary in practice, and does not appear to affect convergence rates or computation time.

Also it is possible to prove convergence under more general asynchronicity.

When defining the coefficients, σ may be underestimated, and L, L 1 , . . .

, L n may be overestimated if exact values are unavailable.

Notice that x k can be eliminated from the above iteration, and the block gradient ∇ i k f (ŷ k ) only needs to be calculated once per iteration.

A larger (or overestimated) maximum delay τ will cause a larger asynchronicity parameter ψ, which leads to more conservative step sizes to compensate.

To estimate ψ, one can first performed a dry run with all coefficient set to 0 to estimate τ .

All function parameters can be calculated exactly for this problem in terms of the data matrix and λ.

We can then use these parameters and this tau to calculate ψ.

ψ and τ merely change the parameters, and do not change execution patterns of the processors.

Hence their parameter specification doesn't affect the observed delay.

Through simple tuning though, we found that ψ = 0.25 resulted in good performance.

In tuning for general problems, there are theoretical reasons why it is difficult to attain acceleration without some prior knowledge of σ, the strong convexity modulus BID3 .

Ideally σ is pre-specified for instance in a regularization term.

If the Lipschitz constants L i cannot be calculated directly (which is rarely the case for the classic dual problem of empirical risk minimization objectives), the line-search method discussed in Roux et al. (2012) Section 4 can be used.

As mentioned in Section 5, authors in Lee & Sidford (2013a) proposed a linear transformation of an accelerated RBCD scheme that results in sparse coordinate updates.

Our proposed algorithm can be given a similar efficient implementation.

We may eliminate x k from A2BCD, and derive the equivalent iteration below: DISPLAYFORM0

where C and Q k are defined in the obvious way.

Hence we define auxiliary variables p k , q k defined via: DISPLAYFORM0 These clearly follow the iteration: DISPLAYFORM1 Since the vector Q k is sparse, we can evolve variables p k , and q k in a sparse manner, and recover the original iteration variables at the end of the algorithm via A.1.The gradient of the dual function is given by: DISPLAYFORM2 As mentioned before, it is necessary to maintain or recover Ay k to calculate block gradients.

Since Ay k can be recovered via the linear relation in equation A.1, and the gradient is an affine function, we maintain the auxiliary vectors Ap k and Aq k instead.

Hence we propose the following efficient implementation in Algorithm 1.

We used this to generate the results in Table 5 .

We also note also that it can improve performance to periodically recover v k and y k , reset the values of p k , q k , and C to v k , y k , and I respectively, and restarting the scheme (which can be done cheaply in time O(d)).We let B ∈ R 2×2 represent C k , and b represent B −1 .

⊗ is the Kronecker product.

Each computing node has local outdated versions of p, q, Ap, Aq which we denotep,q,Âp,Âq respectively.

We also find it convenient to define: DISPLAYFORM3 Algorithm 1 Shared-memory implementation of A2BCD Randomly select block i via equation 2.1.

Read shared data into local memory:p ← p,q ← q,Âp ← Ap,Âq ← Aq,B ← B.

Compute block gradient: DISPLAYFORM0 10: DISPLAYFORM1

11: DISPLAYFORM0 12: DISPLAYFORM1 Increase iteration count: k ← k + 1 14: end while 15: Recover original iteration variables: DISPLAYFORM2

We first recall a couple of inequalities for convex functions.

Lemma 7.

Let f be σ-strongly convex with L-Lipschitz gradient.

Then we have: DISPLAYFORM0 We also find it convenient to define the norm: DISPLAYFORM1 B.1 Starting point

First notice that using the definition equation 2.8 of v k+1 we have: DISPLAYFORM0 We have the following general identity: DISPLAYFORM1 It can also easily be verified from equation 2.6 that we have: DISPLAYFORM2 DISPLAYFORM3 This inequality is our starting point.

We analyze the terms on the second line in the next section.

To analyze these terms, we need a small lemma.

This lemma is fundamental in allowing us to deal with asynchronicity.

Lemma 8.

Let χ, A > 0.

Let the delay be bounded by τ .

Then: DISPLAYFORM0 Proof.

See Hannah & Yin (2017a).

We have: DISPLAYFORM0 The terms in bold in equation B.8 and equation B.9 are a result of the asynchronicity, and are identically 0 in its absence.

Our strategy is to separately analyze terms that appear in the traditional analysis of Nesterov FORMULA15 , and the terms that result from asynchronicity.

We first prove equation B.8: FIG5 equation B.10 follows from strong convexity (equation B.2 with x = y k and y = x * ), and the fact that ∇f is L-Lipschitz.

The term due to asynchronicity becomes: DISPLAYFORM0 DISPLAYFORM1 using Lemma 8 with χ = κψ −1 , A = y k .

Combining this with equation B.10 completes the proof of equation B.8.We now prove equation B.9: DISPLAYFORM2 Here the last line follows from Lemma 8 with χ = κψ DISPLAYFORM3 We can complete the proof using the following identity that can be easily obtained from equation 2.6: DISPLAYFORM4

Much like Nesterov (2012), we need a f (x k ) term in the Lyapunov function (see the middle of page 357).

However we additionally need to consider asynchronicity when analyzing the growth of this term.

Again terms due to asynchronicity are emboldened.

Lemma 10.

We have: DISPLAYFORM0 Proof.

From the definition equation 2.7 of x k+1 , we can see that x k+1 − y k is supported on block i k .

Since each gradient block ∇ i f is L i Lipschitz with respect to changes to block i, we can use equation B.1 to obtain: DISPLAYFORM1 Here the last line followed from the definition equation B.3 of the norm · * 1/2 .

We now analyze the middle term: DISPLAYFORM2 We then apply Lemma 8 to this with χ = 2h DISPLAYFORM3 Finally to complete the proof, we combine equation B.11, with equation B.12.

The previous inequalities produced difference terms of the form y k+1−j − y k−j 2 .

The following lemma shows how these errors can be incorporated into a Lyapunov function.

Lemma 11.

Let 0 < r < 1 and consider the asynchronicity error and corresponding coefficients: DISPLAYFORM0

Remark 2.

Interpretation.

This result means that an asynchronicity error term A k can negate a series of difference terms − ∞ j=1 s j y k+1−j − y k−j 2 at the cost of producing an additional error c 1 E k y k+1 − y k 2 , while maintaining a convergence rate of r. This essentially converts difference terms, which are hard to deal with, into a y k+1 − y k 2 term which can be negated by other terms in the Lyapunov function.

The proof is straightforward.

Proof.

DISPLAYFORM0 Noting the following completes the proof: DISPLAYFORM1 Given that A k allows us to negate difference terms, we now analyze the cost c 1 E k y k+1 − y k 2 of this negation.

We have: DISPLAYFORM0 Proof.

DISPLAYFORM1 Here equation B.13 following from equation 2.8, the definition of v k+1 .

equation B.14 follows from the inequality x + y 2 ≤ 2 x 2 + 2 y 2 .

The rest is simple algebraic manipulation.

DISPLAYFORM2 (definitions of h and α: equation 2.3, and equation 2.5) = 1 DISPLAYFORM3 Rearranging the definition of ψ, we have: DISPLAYFORM4 Using this on equation B.15, we have: DISPLAYFORM5 This completes the proof.

We are finally in a position to bring together all the all the previous results together into a master inequality for the Lyapunov function ρ k (defined in equation 2.11).

After this lemma is proven, we will prove that the right hand size is negative, which will imply that ρ k linearly converges to 0 with rate β.

Lemma 13.

Master inequality.

We have: DISPLAYFORM0 Proof.

DISPLAYFORM1 We now collect and organize the similar terms of this inequality.

DISPLAYFORM2 Now finally, we add the function-value and asynchronicity terms to our analysis.

We use Lemma 11 is with r = 1 − σ 1/2 S −1 , and DISPLAYFORM3 Notice that this choice of s i will recover the coefficient formula given in equation 2.9.

Hence we have: DISPLAYFORM4 (Lemmas 11 and 12) + c 1 2α In the next section, we will prove that every coefficient on the right hand side of equation B.16 is 0 or less, which will complete the proof of Theorem 1.

DISPLAYFORM5 DISPLAYFORM6 Here the last line followed since ψ ≤ 1 2 and σ 1/2 S −1 ≤ 1.

We now analyze the coefficient of DISPLAYFORM7 Proof.

DISPLAYFORM8 in Lemma 13 is non-positive.

Proof.

We first need to bound c 1 .(equation B.18 and equation 2.9) c 1 = s DISPLAYFORM9 It can be easily verified that if x ≤ 1 2 and y ≥ 0, then (1 − x) −y ≤ exp(2xy).

Using this fact with x = σ 1/2 S −1 and y = τ , we have: DISPLAYFORM10 (since ψ ≤ 3/7 and hence τ σ DISPLAYFORM11 We now analyze the coefficient of ∇f (ŷ k ) DISPLAYFORM12 Proof.

DISPLAYFORM13 Here the last inequality follows since β ≤ 1 and α ≤ σ 1/2 S −1 .

We now rearrange the definition of ψ to yield the identity: DISPLAYFORM14 Using this, we have: DISPLAYFORM15 Here the last line followed since L ≤ L, ψ ≤ 3 7 , and τ ≥ 1.

Hence the proof is complete.

Proof of Theorem 1.

Using the master inequality 13 in combination with the previous Lemmas 14, 15, 16, and 17, we have: DISPLAYFORM16 When we have: DISPLAYFORM17 then the Lyapunov function ρ k has decreased below ρ 0 in expectation.

Hence the complexity K( ) satisfies: DISPLAYFORM18 Now it can be shown that for 0 < x ≤ 1 2 , we have: DISPLAYFORM19 Since n ≥ 2, we have σ 1/2 S −1 ≤ 1 2 .

Hence: DISPLAYFORM20 An expression for K NU_ACDM ( ), the complexity of NU_ACDM follows by similar reasoning.

DISPLAYFORM21 Finally we have: DISPLAYFORM22 which completes the proof.

C.1 Derivation of ODE for synchronous A2BCDIf we take expectations with respect to E k , then synchronous (no delay) A2BCD becomes: DISPLAYFORM0 We find it convenient to define η = nκ 1/2 .

Inspired by this, we consider the following iteration: DISPLAYFORM1 C.1 Derivation of ODE for synchronous A2BCDfor coefficients: DISPLAYFORM2 s is a discretization scale parameter that will be sent to 0 to obtain an ODE analogue of synchronous A2BCD.

We first use equation DISPLAYFORM3 The proof of convergence is completed in Section 3.

For parameter set σ, L 1 , . . .

, L n , n, we construct a block-separable function f on the space R

As mentioned, a stronger result than Theorem 1 is possible.

In the case when L i = L for all i, we can consider a slight modification of the coefficients: DISPLAYFORM0 (E.1) DISPLAYFORM1 for the asynchronicity parameter: DISPLAYFORM2 This leads to complexity: DISPLAYFORM3 Here there is no restriction on ψ as in Theorem 1, and hence there is no restriction on τ .

Assuming ψ ≤ 1 gives optimal complexity to within a constant factor.

Notice then that the resulting condition of τ τ ≤ 1 6 nκ −1/2 (E.9) now essentially matches the one in Theorem 3 in Section 3.

While this result is stronger, it increases the complexity of the proof substantially.

So in the interests of space and simplicity, we do not prove this stronger result.

@highlight

We prove the first-ever convergence proof of an asynchronous accelerated algorithm that attains a speedup.
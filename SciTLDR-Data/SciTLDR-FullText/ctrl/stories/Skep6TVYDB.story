Zeroth-order optimization is the process of minimizing an objective $f(x)$, given oracle access to evaluations at adaptively chosen inputs $x$. In this paper, we present two simple yet powerful GradientLess Descent (GLD) algorithms that do not rely on an underlying gradient estimate and are numerically stable.

We analyze our algorithm from a novel geometric perspective and we show that for {\it any monotone transform} of a smooth and strongly convex objective with latent dimension $k \ge n$, we present a novel analysis that shows convergence within an $\epsilon$-ball of the optimum in $O(kQ\log(n)\log(R/\epsilon))$ evaluations, where the input dimension is $n$, $R$ is the diameter of the input space and $Q$ is the condition number.

Our rates are the first of its kind to be both 1) poly-logarithmically dependent on dimensionality and 2) invariant under monotone transformations.

We further leverage our geometric perspective to show that our analysis is optimal.

Both monotone invariance and its ability to utilize a low latent dimensionality are key to the empirical success of our algorithms, as demonstrated on synthetic and MuJoCo benchmarks.

We consider the problem of zeroth-order optimization (also known as gradient-free optimization, or bandit optimization), where our goal is to minimize an objective function f : R n → R with as few evaluations of f (x) as possible.

For many practical and interesting objective functions, gradients are difficult to compute and there is still a need for zeroth-order optimization in applications such as reinforcement learning (Mania et al., 2018; Salimans et al., 2017; Choromanski et al., 2018) , attacking neural networks Papernot et al., 2017) , hyperparameter tuning of deep networks (Snoek et al., 2012) , and network control (Liu et al., 2017) .

The standard approach to zeroth-order optimization is, ironically, to estimate the gradients from function values and apply a first-order optimization algorithm (Flaxman et al., 2005) .

Nesterov & Spokoiny (2011) analyze this class of algorithms as gradient descent on a Gaussian smoothing of the objective and gives an accelerated O(n √ Q log((LR 2 + F )/ )) iteration complexity for an LLipschitz convex function with condition number Q and R = x 0 − x * and F = f (x 0 ) − f (x * ).

They propose a two-point evaluation scheme that constructs gradient estimates from the difference between function values at two points that are close to each other.

This scheme was extended by (Duchi et al., 2015) for stochastic settings, by (Ghadimi & Lan, 2013) for nonconvex settings, and by (Shamir, 2017) for non-smooth and non-Euclidean norm settings.

Since then, first-order techniques such as variance reduction (Liu et al., 2018) , conditional gradients (Balasubramanian & Ghadimi, 2018) , and diagonal preconditioning (Mania et al., 2018) have been successfully adopted in this setting.

This class of algorithms are also known as stochastic search, random search, or (natural) evolutionary strategies and have been augmented with a variety of heuristics, such as the popular CMA-ES (Auger & Hansen, 2005) .

These algorithms, however, suffer from high variance due to non-robust local minima or highly non-smooth objectives, which are common in the fields of deep learning and reinforcement learn-ing.

Mania et al. (2018) notes that gradient variance increases as training progresses due to higher variance in the objective functions, since often parameters must be tuned precisely to achieve reasonable models.

Therefore, some attention has shifted into direct search algorithms that usually finds a descent direction u and moves to x + δu, where the step size is not scaled by the function difference.

The first approaches for direct search were based on deterministic approaches with a positive spanning set and date back to the 1950s (Brooks, 1958) .

Only recently have theoretical bounds surfaced, with Gratton et al. (2015) giving an iteration complexity that is a large polynomial of n and Dodangeh & Vicente (2016) giving an improved O(n 2 L 2 / ).

Stochastic approaches tend to have better complexities: Stich et al. (2013) uses line search to give a O(nQ log(F/ )) iteration complexity for convex functions with condition number Q and most recently, Gorbunov et al. (2019) uses importance sampling to give a O(nQ log(F/ )) complexity for convex functions with average condition numberQ, assuming access to sampling probabilities.

Stich et al. (2013) notes that direct search algorithms are invariant under monotone transforms of the objective, a property that might explain their robustness in high-variance settings.

In general, zeroth order optimization suffers an at least linear dependence on input dimension n and recent works have tried to address this limitation when n is large but f (x) admits a low-dimensional structure.

Some papers assume that f (x) depends only on k coordinates and Wang et al. (2017) applies Lasso to find the important set of coordinates, whereas Balasubramanian & Ghadimi (2018) simply change the step size to achieve an O(k(log(n)/ )

2 ) iteration complexity.

Other papers assume more generally that f (x) = g(P A x) only depends on a k-dimensional subspace given by the range of P A and Djolonga et al. (2013) apply low-rank approximation to find the low-dimensional subspace while Wang et al. (2013) use random embeddings.

Hazan et al. (2017) assume that f (x) is a sparse collection of k-degree monomials on the Boolean hypercube and apply sparse recovery to achieve a O(n k ) runtime bound.

We will show that under the case that f (x) = g(P A x), our algorithm will inherently pick up any low-dimensional structure in f (x) and achieve a convergence rate that depends on k log(n).

This initial convergence rate survives, even if we perturb f (x) = g(P A x) + h(x), so long as h(x) is sufficiently small.

We will not cover the whole variety of black-box optimization methods, such as Bayesian optimization or genetic algorithms.

In general, these methods attempt to solve a broader problem (e.g. multiple optima), have weaker theoretical guarantees and may require substantial computation at each step: e.g. Bayesian optimization generally has theoretical iteration complexities that grow exponentially in dimension, and CMA-ES lacks provable complexity bounds beyond convex quadratic functions.

In addition to the slow runtime and weaker guarantees, Bayesian optimization assumes the success of an inner optimization loop of the acquisition function.

This inner optimization is often implemented with many iterations of a simpler zeroth-order methods, justifying the need to understand gradient-less descent algorithms within its own context.

In this paper, we present GradientLess Descent (GLD), a class of truly gradient-free algorithms (also known as direct search algorithms) that are parameter free and provably fast.

Our algorithms are based on a simple intuition: for well-conditioned functions, if we start from a point and take a small step in a randomly chosen direction, there is a significant probability that we will reduce the objective function value.

We present a novel analysis that relies on facts in high dimensional geometry and can thus be viewed as a geometric analysis of gradient-free algorithms, recovering the standard convergence rates and step sizes.

Specifically, we show that if the step size is on the order of O(

, we can guarantee an expected decrease of 1 − Ω( 1 n ) in the optimality gap, based on geometric properties of the sublevel sets of a smooth and strongly convex function.

Our results are invariant under monotone transformations of the objective function, thus our convergence results also hold for a large class of non-convex functions that are a subclass of quasi-convex functions.

Specifically, note that monotone transformations of convex functions are not necessarily convex.

However, a monotone transformation of a convex function is always quasi-convex.

The maximization of quasi-concave utility functions, which is equivalent to the minimization of quasiconvex functions, is an important topic of study in economics (e.g. Arrow & Enthoven (1961) ).

. '

Monotone' column indicates the invariance under monotone transformations (Definition 4).

'k-Sparse' and 'k-Affine' columns indicate that iteration complexity is poly(k, log(n)) when f (x) depends only on a k-sparse subset of coordinates or on a rank-k affine subspace.

Iteration Complexity Monotone k-Sparse k-Affine Nesterov & Spokoiny (2011) n log((

Intuition suggests that the step-size dependence on dimensionality can be improved when f (x) admits a low-dimensional structure.

With a careful choice of sampling distribution we can show that if f (x) = g(P A x), where P A is a rank k matrix, then our step size can be on the order of O(

) as our optimization behavior is preserved under projections.

We call this property affine-invariance and show that the number of function evaluations needed for convergence depends logarithmically on n. Unlike most previous algorithms in the high-dimensional setting, no expensive sparse recovery or subspace finding methods are needed.

Furthermore, by novel perturbation arguments, we show that our fast convergence rates are robust and holds even under the more realistic assumption when f (x) = g(P A x) + h(x) with h(x) being sufficiently small.

Theorem 1 (Convergence of GLD: Informal Restatement of Theorem 7 and Theorem 14).

Let f (x) be any monotone transform of a convex function with condition number Q and R = x 0 − x * .

Let y be a sample from an appropriate distribution centered at x. Then, with constant probability,

Therefore, we can find x T such that x T −x * ≤ after T = O(nQ log(R/ )) function evaluations.

Furthermore, for functions f (x) = g(P A x) + h(x) with rank k matrix P A and sufficiently small h(x), we only require O(kQ log(n) log(R/ )) evaluations.

Another advantage of our non-standard geometric analysis is that it allows us to deduce that our rates are optimal with a matching lower bound (up to logarithmic factors), presenting theoretical evidence that gradient-free inherently requires Ω(nQ) function evaluations to converge.

While gradient-estimation algorithms can achieve a better theoretical iteration complexity of O(n √ Q), they lack the monotone and affine invariance properties.

Empirically, we see that invariance properties are important to successful optimization, as validated by experiments on synthetic BBOB and MuJoCo benchmarks that show the competitiveness of GLD against standard optimization procedures.

We first define a few notations for the rest of the paper.

Let X be a compact subset of R n and let · denote the Euclidean norm.

The diameter of X , denoted X = max x,x ∈X x − x , is the maximum distance between elements in X .

Let f : X → R be a real-valued function which attains its minimum at x * .

We use f (X ) = {f (x) : x ∈ X } to denote the image of f on a subset X of R n , and B(c, r) = {x ∈ R n : c − x ≤ r} to denote the ball of radius r centered at c.

When the function f is clear from the context, we omit it.

Definition 3.

We say that f is α-strongly convex for

2 for all x, y ∈ X and β-smooth for

for all x, y ∈ X .

Definition 4.

We say that g•f is a monotone transformation of f if g : f (X ) → R is a monotonically (and strictly) increasing function.

Monotone transformations preserve the level sets of a function in the sense that

.

Because our algorithms depend only on the level set properties, our results generalize to any monotone transformation of a strongly convex and strongly smooth function.

This leads to our extended notion of condition number.

Definition 5.

A function f has condition number Q ≥ 1 if it is the minimum ratio β/α over all functions g such that f is a monotone transformation of g and g is α-strongly convex and β smooth.

When we work with low rank extensions of f , we only care about the condition number of f within a rank k subspace.

Indeed, if f only varies along a rank k subspace, then it has a strong convexity value of 0, making its condition number undefined.

If f is α-strongly convex and β-smooth, then its Hessian matrix always has eigenvalues bounded between α and β.

Therefore, we need a notion of a projected condition number.

Let A ∈ R d×k be some orthonormal matrix and let P A = AA be the projection matrix onto the column space of A. Definition 6.

For some orthonormal A ∈ R d×k with d > k, a function f has condition number restricted to A, Q(A) ≥ 1, if it is the minimum ratio β/α over all functions g such that f is a monotone transformation of g and h(y) = g(Ay) is α-strongly convex and β smooth.

The GLD template can be summarized as follows: given a sampling distribution D, we start at x 0 and in iteration t, we choose a scalar radii r t and we sample y t from a distribution r t D centered around x t , where r t provides the scaling of D. Then, if f (x t+1 ) < x t , we update x t+1 = y t ; otherwise, we set x t+1 = x t .

The analysis of GLD follows from the main observation that the sublevel set of a monotone transformation of a strongly convex and strongly smooth function contains a ball of sufficiently large radius tangent to the level set (Lemma 15).

In this section, we show that this property, combined with facts of high-dimensional geometry, implies that moving in a random direction from any point has a good chance of significantly improving the objective.

As we mentioned before, the key to fast convergence is the careful choice of step sizes, which we describe in Theorem 7.

The intuition here is that we would like to take as large steps as possible while keeping the probability of improving the objective function reasonably high, so by insights in high-dimensional geometry, we choose a step size of Θ(1/ √ n).

Also, we show that if f (x) admits a latent rank-k structure, then this step size can be increased to Θ(1/ √ k) and is therefore only dependent on the latent dimensionality of f (x), allowing for fast high-dimensional optimization.

Lastly, our geometric understanding allows us to show that our convergence rates are optimal with a matching lower bound.

Without loss of generality, this section assumes that f (x) is strongly convex and smooth with condition number Q.

Theorem 7.

For any x such that

such that if r = 2 k1 C 1 or r = 2 −k2 C 2 , then a random sample y from uniform distribution over

with probability at least 1 4 .

Proving the above theorem requires the following lemma about the intersection of balls in high dimensions and it is proved in the appendix.

Lemma 8.

Let B 1 and B 2 be two balls in R n of radii r 1 and r 2 respectively.

Let be the distance between the centers.

If

where c n is a dimension-dependent constant that is lower bounded by 1 4 at n = 1.

A direct application of Lemma 8 seems to imply that uniform sampling of a high-dimensional ball is necessary.

Upon further inspection, this can be easily replaced with a much simpler Gaussian sampling procedure that concentrates the mass close to the surface to the ball.

This procedure lends itself to better analysis when f (x) admits a latent low-dimensional structure since any affine projection of a Gaussian is still Gaussian.

Lemma 9.

Let B 1 and B 2 be two balls in R n of radii r 1 and r 2 respectively.

Let be the distance between the centers.

If

and r 2 ≥ − n and X = (X 1 , ..., X n ) are independent Gaussians with mean centered at the center of B 1 and variance

where c is a dimension-independent constant.

Assume that there exists some rank k projection matrix P A such that f (x) = g(P A x), where k is much smaller than n. Because Gaussians projected on a k-dimensional subspace are still Gaussians, we show that our algorithm has a dimension dependence on k. We let Q g (A) be the condition number of g restricted to the subspace A that drives the dominant changes in f (x).

Theorem 10.

Let f (x) = g(P A x) for some unknown rank k matrix P A with k < n and suppose

with constant probability.

Note that the speed-up in progress is due to the fact that we can now tolerate the larger sampling radius of Ω(1/ √ k), while maintaining a high probability of making progress.

If k is unknown, we can simply use binary search to find the correct radius with an extra factor of log(n) in our runtime.

The low-rank assumption is too restrictive to be realistic; however, our fast rates still hold, at least for the early stages of the optimization, even if we assume that f (x) = g(P A x) + h(x) and |h(x)| ≤ δ is a full-rank function that is bounded by δ.

In this setting, we can show that convergence remains fast, at least until the optimality gap approaches δ.

Theorem 11.

Let f (x) = g(P A x) + h(x) for some unknown rank k matrix P A with k < n where g, h are convex and |h| ≤ δ.

Suppose

with constant probability whenever

We show that our upper bounds given in the previous section are tight up to logarithmic factors for any symmetric sampling distribution D. These lower bounds are easily derived from our geometric perspective as we show that a sampling distribution with a large radius gives an extremely low probability of intersection with the desired sub-level set.

Therefore, while gradient-approximation algorithms can be accelerated to achieve a runtime that depends on the square-root of the condition number Q, gradient-less methods that rely on random sampling are likely unable to be accelerated according to our lower bound.

However, we emphasize that monotone invariance allows these results to apply to a broader class of objective functions, beyond smooth and convex, so the results can be useful in practice despite the seemingly worse theoretical bounds.

Algorithm 1: Gradientless Descent with Binary Search (GLD-Search) Input: function: f : R n → R, T ∈ Z + : number of iterations, x 0 : starting point, D: sampling distribution, R: maximum search radius, r: minimum search radius 1 Set K = log(R/r) 2 for t = 0, . . .

, T do 3 Ball Sampling Trial i:

Update:

9 end 10 return x t Theorem 12.

Let y = x + v, where v is a random sample from rD for some radius r > 0 and D is standard Gaussian or any rotationally symmetric distribution.

Then, there exist a region X with positive measure such that for any x ∈ X,

with probability at least 1 − 1 poly(nQ) .

In this section, we present two algorithms that follow the same Gradientless Descent (GLD) template: GLD-Search and GLD-Fast, with the latter being an optimized version of the former when an upper bound on the condition number of a function is known.

For both algorithms, since they are monotone-invariant, we appeal to the previous section to derive fast convergence rates for any monotone transform of convex f (x) with good condition number.

We show the efficacy of both algorithms experimentally in the Experiments section.

Although the sampling distribution D is fixed, we have a choice of radii for each iteration of the algorithm.

We can apply a binary search procedure to ensure progress.

The most straightforward version of our algorithm is thus with a naive binary sweep across an interval in [r, R] that is unchanged throughout the algorithm.

This allows us to give convergence guarantees without previous knowledge of the condition number at a cost of an extra factor of log(n/ ).

Theorem 13.

Let x 0 be any starting point and f a blackbox function with condition number Q. Running Algorithm 1 with r = √ n , R = X and D = N (0, I) as a standard Gaussian returns a

2 ) function evaluations with high probability.

Furthermore, if f (x) = g(P A x) admits a low-rank structure with P A a rank k matrix, then we only require O(kQ g (A) log(n X / )

2 ) function evaluations to guarantee P A (x T − x * ) ≤ .

This holds analogously even if f (x) = g(P A x) + h(x) is almost low-rank where |h| ≤ δ and > 60δkQ g (A).

GLD-Search (Algorithm 1) uses a naive lower and upper bound for the search radius x t − x * , which incurs an extra factor of log(1/ ) in the runtime bound.

In GLD-Fast, we remove this extra factor dependence on log(1/ ) by drastically reducing the range of the binary search.

This is done by exploiting the assumption that f has a good condition number upper boundQ and by slowly halfing the diameter of the search space every few iterations since we expect x t → x * as t → ∞.

Algorithm 2: Gradientless Descent with Fast Binary Search (GLD-Fast) Input: function f : R n → R, T ∈ Z + : number of iterations, x 0 : starting point, D: sampling distribution, R: diameter of search space, Q: condition number bound 1 Set K = log(4

Ball Sampling Trial i:

Update: x t+1 = arg min i f (y) y = x t , y = x t + v i 10 end 11 return x t Theorem 14.

Let x 0 be any starting point and f a blackbox function with condition number upper bounded by Q. Running Algorithm 2 with suitable parameters returns a point

) function evaluations with high probability.

Furthermore, if f (x) = g(P A x) admits a low-rank structure with P A a rank k matrix, then we only

is almost low-rank where |h| ≤ δ and > 60δkQ g (A).

We tested GLD algorithms on a simple class of objective functions and compare it to Accelerated Random Search (ARS) by Nesterov & Spokoiny (2011) , which has linear convergence guarantees on strongly convex and strongly smooth functions.

To our knowledge, ARS makes the weakest assumption among the zeroth-order algorithms that have linear convergence guarantees and perform only a constant order of operations per iteration.

Our main conclusion is that GLD-Fast is comparable to ARS and tends to achieve a reasonably low error much faster than ARS in high dimensions (≥ 50).

In low dimensions, GLD-Search is competitive with GLD-Fast and ARS though it requires no information about the function.

We let H α,β,n ∈ R n×n be a diagonal matrix with its i-th diagonal equal to α + (β − α)

i−1 n−1 .

In simple words, its diagonal elements form an evenly space sequence of numbers from α to β.

Our objective function is then f α,β,n : R n → R as f α,β,n (x) = 1 2 x H α,β,n x, which is α-strongly convex and β-strongly smooth.

We always use the same starting point x = 1 √ n (1, . . . , 1), which requires X = √ Q for our algorithms.

We plot the optimality gap f (b t ) − f (x * ) against the number of function evaluations, where b t is the best point observed so far after t evaluations.

Although all tested algorithms are stochastic, they have a low variance on the objective functions that we use; hence we average the results over 10 runs and omit the error bars in the plots.

We ran experiments on f 1,8,n with imperfect curvature informationα andβ (see Figure 3 in appendix).

GLD-Search is independent of the condition number.

GLD-Fast takes only one parameter, which is the upper bound on the condition number; if approximation factor is z, then we pass 8z as the upper bound.

ARS requires both strong convexity and smoothness parameters.

We test three different distributions of the approximation error; when the approximation factor is z, then ARS-alpha gets (α/z, β), ARS-beta gets (α, zβ), and ARS-even gets (α/ √ z, √ zβ) as input.

GLD-Fast is more robust and faster than ARS when the condition number is over-approximated.

When the condition number is underestimated, GLD-Fast still steadily converges.

In Figure 1 , we ran experiments on f 1,8,n for different settings of dimensionality n, and its monotone transformation with g(y) = − exp(− √ y).

For this experiment, we assume a perfect oracle for the strong convexity and smoothness parameters of f .

The convergence of GLD is totally unaffected by the monotone transformation.

For the low-dimension cases of a transformed function (bottom half of the figure), we note that there are inflection points in the convergence curve of ARS.

This means that ARS initially struggles to gain momentum and then struggles to stop the momentum when it gets close to the optimum.

Another observation is that unlike ARS that needs to build up momentum, GLD-Fast starts from a large radius and therefore achieves a reasonably low error much faster than ARS, especially in higher dimensions.

To show that practicality of GLD on practical and non-convex settings, we also test GLD algorithms on a variety of BlackBox Optimization Benchmarking (BBOB) functions (Hansen et al., 2009 ).

For each function, the optima is known and we use the log optimality gap as a measure of competance.

Because each function can exhibit varying forms of non-smoothness and convexity, all algorithms are ran with a smoothness constant of 10 and a strong convexity constant of 0.1.

All other setup details are same as before, such as using a fixed starting point.

The plots, given in Appendix C, underscore the superior performance of GLD algorithms on various BBOB functions, demonstrating that GLD can successfully optimize a diverse set of functions even without explicit knowledge of condition number.

We note that BBOB functions are far from convex and smooth, many exhibiting high conditioning, multi-modal valleys, and weak global structure.

Due to our radius search produce, our algorithm appears more robust to non-ideal settings with non-convexity and ill conditioning.

As expected, we note that GLD-Fast tend to outperform GLDSearch, especially as the dimension increases, matching our theoretical understanding of GLD.

We also ran experiments on the Mujoco benchmarks with varying architectures, both linear and nonlinear.

This demonstrates the viability of our approach even in the non-convex, high dimensional setting.

We note that however, unlike e.g. ES which uses all queries to form a gradient direction, our algorithm removes queries which produce less reward than using the current arg-max, which can be an information handicap.

Nevertheless, we see that our algorithm still achieves competitive performance on the maximum reward.

We used a horizon of 1000 for all experiments.

We further tested the affine invariance of GLD on the policy parameters from using Gaussian ball sampling, under the HalfCheetah benchmark by projecting the state s of the MDP with linear policy to a higher dimensional state W s, using a matrix multiplication with an orthonormal W .

Specifically, in this setting, for a linear policy parametrized by matrix K, the objective function is thus J(KW ) where π K (W s) = KW s. Note that when projecting into a high dimension, there is a slowdown factor of log dnew d old where d new , d old are the new high dimension and previous base dimension, respectively, due to the binary search in our algorithm on a higher dimensional space.

For our HalfCheetah case, we projected the 17 base dimension to a 200-length dimension, which suggests that the slowdown factor is a factor log 200 17 ≈ 3.5.

This can be shown in our plots in the appendix (Figure 15 ).

We introduced GLD, a robust zeroth-order optimization algorithm that is simple, efficient, and we show strong theoretical convergence bounds via our novel geometric analysis.

As demonstrated by our experiments on BBOB and MuJoCo benchmarks, GLD performs very robustly even in the non-convex setting and its monotone and affine invariance properties give theoretical insight on its practical efficiency.

GLD is very flexible and allows easy modifications.

For example, it could use momentum terms to keep moving in the same direction that improved the objective, or sample from adaptively chosen ellipsoids similarly to adaptive gradient methods. (Duchi et al., 2011; McMahan & Streeter, 2010) .

Just as one may decay or adaptively vary learning rates for gradient descent, one might use a similar change the distribution from which the ball-sampling radii are chosen, perhaps shrinking the minimum radius as the algorithm progresses, or concentrating more probability mass on smaller radii.

Likewise, GLD could be combined with random restarts or other restart policies developed for gradient descent.

Analogously to adaptive per-coordinate learning rates Duchi et al. (2011); McMahan & Streeter (2010) , one could adaptively change the shape of the balls being sampled into ellipsoids with various length-scale factors.

Arbitrary combinations of the above variants are also possible.

Lemma 15.

If h has condition number Q, then for all x ∈ X , there is a ball of radius Q −1 x − x * that is tangent at x and inside the sublevel set L ↓ x (h).

Proof.

Write h = g • f such that f is α-strongly convex and β-smooth for some β = Qα and g is monotonically increasing.

From the smoothness assumption, we have for any s,

Consider the ball B = B(x − 1 β ∇f (x), 1 β ∇f (x) ).

For any y ∈ B, the above inequality implies f (y) ≤ f (x).

Hence, when we apply g on both sides, we still have h(y) ≤ h(x) for all y ∈ B. Therefore, B ⊆ L ↓ h(y) .

By strong convexity, ∇f (x) ≥ α x−x * .

It follows that the radius of B is at least

Proof of Lemma 8.

Without loss of generality, consider the unit distance case where = 1.

Furthermore, it suffices to prove for the smallest possible radius r 2 = 1 − 1 4n .

Since |r 1 −r 2 | ≤ ≤ r 1 +r 2 , the intersection B 1 ∩B 2 is composed of two hyperspherical caps glued end to end.

We lower bound vol (B 1 ∩ B 2 ) by the volume of the cap C 1 of B 1 that is contained in the intersection.

Consider the triangle with sides r 1 , r 2 and .

From classic geometry, the height of

The volume of a spherical cap is Li (2011) ,

.

where I is the regularized incomplete beta function defined as

where x ∈ [0, 1] and a, b ∈ (0, ∞).

Note that for any fixed a and b, I x (a, b) is increasing in x. Hence, in order to obtain a lower bound on vol (C 1 ), we want to lower bound 1 − Write

(1),

Hence,

.

To complete the proof, note that

2 ) is increasing in n, and V 1 = 1 4 .

As n goes to infinity, this value converges to 1 as B 1 ⊂ B 2 .

Proof of Lemma 7.

Let ν = 1 5nQ .

Let q = (1 − ν)x + νx * .

Let B q = B(c q , r q ) be a ball that has q on its surface, lies inside L ↓ q , and has radius r q = Q −1 x−x * .

Lemma 15 guarantees its existence.

and that a random sample y from B x belongs to B q , which happens with probability at least 1 4 .

Then, our guarantee follows by

where the first line follows from Lemma 15 and second line from convexity of f .

Therefore, it now suffices to prove Eq. 2.

To do so, we will apply Lemma 8 after showing that the radius of B x and B q are in the proper ranges.

Let = x − c q and note that

Since x is outside of B q , we also have

It follows that

In the log 2 space, our choice of k 1 is equivalent to starting from log 2 C 1 and sweeping through the range [log 2 C 1 , log 2 C 2 ] at the interval of size 1.

This is guaranteed to find a point between 2 and , which is also an interval of size 1.

Therefore, there exists a k 1 satisfying the theorem statement, and similarly, we can prove the existence of k 2 .

Finally, it remains to show that r q ≥ (1 − 1/(4n)) .

From Eq. (3), it suffices to show that x − q ≤ 4n or equivalently ν x − x * ≤ 4n .

From Eq. (4),

and the proof is complete.

Proof of Lemma 9.

Without loss of generality, let = 1 and B 2 is centered at the origin with radius r 2 and B 1 is centered at e 1 = (1, 0, ..., 0).

Then, we simply want to show that

By Markov's inequality, we see that n i=2 X 2 i ≤ 2r 2 1 = 2/n with probability at most 1/2.

And since X 1 is independent and r 2 ≥ 1 − 1/n, it suffices to show that

Since X 1 has standard deviation at least r 1 / √ n ≥ 1/(2n), we see that the probability of deviating at least a few standard deviation below is at least a constant.

Proof of Theorem 10.

We can consider the projection of all points onto the column space of A and since the Gaussian sampling process is preserved, our proof follows from applying Theorem 7 restricted onto the k-dimensional subspace and using Lemma 9 in place of Lemma 8.

By Lemma 9, we see that if we sample from a Gaussian distribution y ∼ N (x, r 2 k I), then if z * is the minimum of g(x) restricted to the column space of A, then

with constant probability.

By boundedness on h, we know that h(y) ≤ h(x) + 2δ.

Furthermore, this also implies that g(P A x * ) ≤ g(z * ) + 2δ.

Therefore, we know that the decrease is at least

10kQg(A) and our proof is complete.

Proof of Theorem 12.

Our main proof strategy is to show that progress can only be made with a radius size of O( log(nQ)/(nQ)); larger radii cannot find descent directions with high probability.

Consider a simple ellipsoid function f (x) = x Dx, where D is a diagonal matrix and D 11 ≤ D 22 ≤ ... ≤ D nn , where WLOG we let D 11 = 1 and D ii = Q for i > 1.

The optima is x * = 0 with f (x * ) = 0.

Then, if we let v ∼ N (0, I) be a standard Gaussian vector, then for some radius r, we see that the probability of finding a descent direction is:

By standard concentration bounds for sub-exponential variables, we have

Therefore, with exponentially high probability, i>1 X 2 i ≥ n/2.

Also, since |x i | ≤ 0.1/(Q √ n), Chernoff bounds give:

Therefore, with probability at least 1 − 1/(nQ)

If Qrn ≥ Ω( log(nQ)), then we have

We conclude that the probability of descent is upper bounded by Pr

This probability is exactly Φ(−l), where Φ is the cumulative density of a standard normal and l = Ω( log(nQ)).

By a naive upper bound, we see that

Since l = Ω( log(nQ)), we conclude that with probability at least 1 − 1/poly(nQ), we have

Otherwise, we are in the case that Qrn ≤ O( log(nQ)).

Arguing simiarly as before, with high probability, our objective function and each coordinate can change by at most O( log(nQ)/(Qn)).

Next, we extend our proof to any symmetric distribution D. Since D is rotationally symmetric, if we parametrize v = (r, θ) is polar-coordinates, then the p.d.f.

of any scaling of D must take the form p(v) = p r (r)u(θ), where u(θ) induces the uniform distribution over the unit sphere.

Therefore, if Y is a random variable that follows D, then we may write Y = Rv/ v , where R is a random scalar with p.d.f p r (r) and v is a standard Gaussian vector and R, X are independent.

As previously argued, v ∈ [0.5n, 1.5n] with exponentially high probability.

Therefore, if R ≥ Ω( log(nQ)/Q), the same arguments will imply that Y is a descent direction with polynomially small probability.

Thus, when Y is a descent direction, it must be that R ≤ Ω( log(nQ)/Q) and as argued previously, our lower bound follows similarly.

Proof of Theorem 13.

By the Gaussian version of Theorem 7 (full rank version of Theorem 10), as long as our binary search sweeps between minimum search radius r ≤

√ n

x − x * and maximum search radius of the diameter of the whole space R = X , the objective value will decrease multiplicatively by 1 − 1 5nQ in each iteration with constant probability.

Therefore, if x t − x * ≥ 2Q and we set r = √ n and R = X , then with high probability, we expect f (

Otherwise, if there exists some x t such that

.

Therefore, by strong convexity, we conclude that in either case,

3/2 .

Finally note that each iteration uses a binary search that requires O(log(R/r)) = O(log(n X / )) function evaluations.

Therefore, by combining these bounds, we derive our result.

The low-rank result follows from applying Theorem 10 and Theorem 11 instead.

Proof of Theorem 14.

Let H = O(nQ log(Q)) be the number of iterations between successive radius halving and we initialize R = X and half R every H iterations.

We call the iterations between two halving steps an epoch.

We claim that x i − x 0 ≤ R for all iterations and proceed with induction on the epoch number.

The base case is trivial.

Assume that x i − x 0 ≤ R for all iterations in the previous epoch and let iteration i s be the start of the epoch and iteration i s + H be the end of the epoch.

Then, since x is − x * ≤ R, we see that

QR for all i in the previous epoch, then by the Gaussian version of Theorem 7 (Theorem 10), since we do a binary sweep from R 4Q to 4 √ QR, we can choose D accordingly so that we are guaranteed that our objective value will decrease multiplicatively by 1 − 1 5nQ with constant probability at a cost of O(log(Q)) function evaluations per iteration.

This implies that with high probability, after O(nQ log(Q)) iterations, we conclude

2 , which contradicts the fact that f (x is )−f (x * ) ≤ βR 2 by smoothness.

If it is the latter, then by smoothness, we reach the same conclusion:

Therefore, by strong convexity, we have

And our induction is complete.

Therefore, we conclude that after log( X / ) epochs, we have x T − x * ≤ .

Each epoch has H iterations, each with O(log(Q)) function evaluations and so our result follows.

The low-rank result follows from applying Theorem 10 and Theorem 11 instead.

However, note that since we do not know the latent dimension k, we must extend the binary search to incur an extra log(n) factor in the binary search cost.

<|TLDR|>

@highlight

Gradientless Descent is a provably efficient gradient-free algorithm that is monotone-invariant and fast for high-dimensional zero-th order optimization.

@highlight

This paper proposes stable GradientLess Descent (GLD) algorithms that do not rely on gradient estimate.
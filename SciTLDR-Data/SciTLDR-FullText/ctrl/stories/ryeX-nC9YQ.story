Low-precision training is a promising way of decreasing the time and energy cost of training machine learning models.

Previous work has analyzed low-precision training algorithms, such as low-precision stochastic gradient descent, and derived theoretical bounds on their convergence rates.

These bounds tend to depend on the dimension of the model $d$ in that the number of bits needed to achieve a particular error bound increases as $d$ increases.

This is undesirable because a motivating application for low-precision training is large-scale models, such as deep learning, where $d$ can be huge.

In this paper, we prove dimension-independent bounds for low-precision training algorithms that use fixed-point arithmetic, which lets us better understand what affects the convergence of these algorithms as parameters scale.

Our methods also generalize naturally to let us prove new convergence bounds on low-precision training with other quantization schemes, such as low-precision floating-point computation and logarithmic quantization.

As machine learning models continue to scale to target larger problems on bigger data, the task of training these models quickly and efficiently becomes an ever-more-important problem.

One promising technique for doing this is low-precision computation, which replaces the 32-bit or 64-bit floating point numbers that are usually used in ML computations with smaller numbers, often 8-bit or 16-bit fixed point numbers.

Low-precision computation is a broadly applicable technique that has received a lot of attention, especially for deep learning, and specialized hardware accelerators have been developed to support it (Jouppi et al., 2017; Burger, 2017; Caulfield et al., 2017) .A major application for low-precision computation is the training of ML models using empirical risk minimization.

This training is usually done using stochastic gradient descent (SGD), and most research in low-precision training has focused on low-precision versions of SGD.

While most of this work is empirical (Wu et al., 2018; Das et al., 2018; Zhu et al., 2016; Köster et al., 2017; Lee et al., 2017; Hubara et al., 2016; Rastegari et al., 2016; Zhou et al., 2016; Gupta et al., 2015; Courbariaux et al., 2014; 2015; De Sa et al., 2017) , significant research has also been done in the theoretical analysis of low-precision training.

This theoretical work has succeeded in proving bounds on the convergence rate of low-precision SGD and related low-precision methods in various settings, including for convex (De Sa et al., 2018; Zhang et al., 2017) and non-convex objectives (De Sa et al., 2015; Li et al., 2017; Alistarh et al., 2017) .

One common characteristic of these results is that the bounds tend to depend on the dimension d of the model being learned (equivalently, d is the number of parameters).

For example, (Li et al., 2017) gives the convergence bound DISPLAYFORM0 where the objective f is strongly convex with parameter µ, low-precision SGD outputsw T after T iterations, w * is the true global minimizer of the objective, σ 2 max is an upper bound on the second moment of the stochastic gradient samples E[ f (w) 2 2 ] ≤ σ 2 max , and δ is the quantization step, the difference between adjacent numbers in the low-precision format.

Notice that, as T → ∞, this bound shows convergence down to a level of error that increases with the dimension d. Equivalently, in order to achieve the same level of error as d increases, we would need to use more bits of quantization to make δ smaller.

Similar dimension-dependent results, where either the error or the number of bits needed increases with d, can also be seen in other work on low-precision training algorithms (Alistarh et al., 2017; Zhang et al., 2017; De Sa et al., 2018) .

This dependence on d is unsatisfying because the motivation for low-precision training is to tackle large-scale problems on big data, where d can range up to 10 8 or more for commonly used models (Simonyan and Zisserman, Table 1 : Summary of our dimension-free results compared with prior work.

The values report the number of bits needed, according to the theoretical bound, for the LP-SGD (Li et al., 2017) algorithm to achieve an expected objective gap (f (w) − f (w * )) of when we let step size α → 0, epoch length T → ∞. Here we let R denote the radius of the range of numbers representable in the low-precision format and assume w * 2 = Θ(R).

The rest of the parameters can be found in the assumptions to be introduced later.

log 2 O (Rσ/ε) · log 1 + σ1/σ 2014).

For example, to compensate for a factor of d = 10 8 in (1), we could add bits to decrease the quantization step δ by a factor of √ d, but this would require adding log 2 (10 4 ) ≈ 13 bits, which is significant compared to the 8 or 16 bits that are commonly used in low-precision training.

−In this paper, we address this problem by proving dimension-free bounds on the convergence of LP-SGD Li et al. (2017) .

Our main technique for doing so is a tight dimension-independent bound on the expected quantization error of the low-precision stochastic gradients in terms of the 1 -norm.

Our results are summarized in Table 1 , and we make the following contributions:• We describe conditions under which we can prove a dimension-free bound on the convergence of SGD with fixed-point, quantized iterates on strongly convex problems.• We study non-linear quantization schemes, in which the representable low-precision numbers are distributed non-uniformly.

We prove dimension-free convergence bounds for SGD using logarithmic quantization (Lee et al., 2017) , and we show that using logarithmic quantization can reduce the number of bits needed for LP-SPG to provably converge.• We study quantization using low-precision floating-point numbers, and we present theoretical analyis that suggests how to assign a given number of bits to exponent and mantissa to optimize the accuracy of training algorithms.

We validate our results experimentally.

Motivated by the practical implications of faster machine learning, much work has been done on low-precision training.

This work can be roughly divided into two groups.

The first focuses on training deep models with low-precision weights, to be later used for faster inference.

For some applications, methods of this type have achieved good results with very low-precision models: for example, binarized (Courbariaux et al., 2015; Hubara et al., 2016; Rastegari et al., 2016) and ternary networks (Zhu et al., 2016) have been observed to be effective (although as is usual for deep learning they lack theoretical convergence results).

However, these approaches are still typically trained with full-precision iterates: the goal is faster inference, not faster training (although faster training is often achieved as a bonus side-effect).A second line of work on low-precision training, which is applied to both DNN training and nondeep-learning tasks, focuses on making various aspects of SGD low-precision, while still trying to solve the same optimization problem as the full-precision version.

The most common way to do this is to make the iterates of SGD (the w t in the SGD update step w t+1 = w t − α t ∇f t (w t )) stored and computed in low-precision arithmetic (Courbariaux et al., 2014; Gupta et al., 2015; De Sa et al., 2018; 2015; Li et al., 2017) .

This is the setting we will focus on most in this paper, because it has substantial theoretical prior work which exhibits the dimension-dependence we set out to study (Li et al., 2017; Zhang et al., 2017; Alistarh et al., 2017; De Sa et al., 2018) .

The only paper we found with a bound that was not dimension-dependent was De Sa et al. (2015) , but in that paper the authors required that the gradient samples be 1-sparse (have only one nonzero entry), which is not a realistic assumption for most ML training tasks.

In addition to quantizing the iterates, other work has studied quantizing the training set (Zhang et al., 2017) and numbers used to communicate among parallel workers (Alistarh et al., 2017) .

We expect that our results on dimension-free bounds will be complementary with these existing theoretical approaches, and we hope that they can help to explain the success of the exciting empirical work in this area.

In this section, we analyze the performance of stochastic gradient descent (SGD) using low-precision training.

Though there are numerous variants of this algorithm, SGD remains the de facto algorithm used most for machine learning.

We will start by describing SGD and how it can be made lowprecision.

Suppose we are trying to solve the problem minimize: DISPLAYFORM0 SGD solves this problem iteratively by repeatedly running the update step w t+1 = w t − α∇f it (w t ) (3) where α is the step size 1 or learning rate, and i t is the index of a component function chosen randomly and uniformly at each iteration from {1, . . .

, n}. To make this algorithm low-precision, we quantize the iterates (the vectors w t ) and store them in a low-precision format.

The standard format to use lets us represent numbers in a set DISPLAYFORM1 with δ > 0 being the quantization gap, the distance between adjacent representable numbers, and b ∈ N being the number of bits we use (De Sa et al., 2018) .

Usually, δ is a power of 2, and this scheme is called fixed-point arithmetic.

It is straightforward to encode numbers in this set as b-bit signed integers, by just multiplying or dividing by δ to convert to or from the encoded formatand we can even do many arithmetic computations on these numbers directly as integers.

This is sometimes called linear quantization because the representable points are distributed uniformly throughout their range.

However, as the gradient samples will produce numbers outside this set during iteration, we need some way to map these numbers to the set of numbers that we can represent.

The standard way to do this is with a quantization function Q(x) : R → dom(δ, b).

While many quantization functions have been proposed, the one typically used in theoretical analysis (which we will continue to use here) is randomized rounding.

Randomized rounding, also known as unbiased rounding or stochastic rounding, rounds up or down at random such that E [Q(x)] = x whenever x is within the range of representable numbers (i.e. when −δ · 2 DISPLAYFORM2 .

When x is outside that range, we quantize it to the closest representable point.

When we apply Q (δ,b) to a vector argument, it quantizes each of its components independently.

Using this quantization function, we can write the update step for low-precision SGD (LP-SGD), which is a simple quantization of (3), DISPLAYFORM3 As mentioned before, one common feature of prior bounds on the convergence of LP-SGD is that they depend on the number of dimensions d, whereas bounds on full precision SGD under the same conditions do not do so.

This difference is due to the fact that, when we use a quantization function Q to quantize a number w, it increases its variance by E (Q(w) − w) 2 ≤ δ 2 /4.

Observe that this inequality is tight since it holds as an equality when w is in the middle of two quantization points, e.g. w = δ/2, as illustrated in FIG1 (a).

When quantizing a vector w ∈ R d , the squared error can be increased by DISPLAYFORM4 and this bound is again tight.

This variance inequality is the source of the d term in analyses of LP-SGD, and the tightness of the bound leads to the natural belief that the d term is inherent, and that low-precision results are inevitably dimension-dependent.

However, we propose that if we can instead bound the variance in (5) with some properties of the problem itself that do not change as d changes, we can achieve a result that is dimensionindependent.

One way to do this is to look at the variance graphically.

FIG4 plots the quantization error as a function of w along with the bound in (5).

Notice that the squared error looks like a series of parabolas, and the bound in (5) is tight at the top of those parabolas, but loose elsewhere.

Instead, suppose we want to do the opposite and produce a bound that is tight when the error is zero (at points in dom(δ, b)).

To do this, we observe that E (Q(w) − w) 2 ≤ δ|w −z| for any z ∈ dom(δ, b).

This bound is also tight when z is adjacent to w, and we plot it in FIG4 as well.

The natural vector analog of this is where · 1 denotes the L1 norm.

This is a dimension-independent bound we can use to replace (5) to bound the convergence of LP-SGD and other algorithms.

However, this replacement is nontrivial as our bound is now non-constant: it depends on w, which is a variable updated each iteration.

Also, in order to bound this new L1 norm term, we will need some new assumptions about the problem.

Next, we will state these assumptions, along with the standard assumptions used in the analysis of SGD for convex objectives, and then we will use them to present our dimension-free bound on the convergence of SGD.

Assumption 1.

All the loss functionsf i are differentiable, and their gradients are L-Lipschitz continuous in the sense of 2-norm, that is, DISPLAYFORM5 DISPLAYFORM6 All the gradients of the loss functionsf i are L 1 -Lipschitz continuous in the sense of 1-norm to 2-norm, that is, DISPLAYFORM7 These two assumptions are simply expressing of Lipschitz continuity in different norms.

Assumption 1 is a standard assumption in the analysis of SGD on convex objectives, and has been applied in the low-precision case as well in prior work (De Sa et al., 2018) .

Assumption 2 is analogous to 1, except we are bounding the L1 norm instead of the L2 norm.

This holds naturally (with a reasonable value of L 1 ) for many problems, in particular problems for which the gradient samples are sparse.

Assumption 3.

The gradient of the total loss function f is µ-strongly convex for some µ > 0: DISPLAYFORM8 Assumption 3 is a standard assumption that bounds the curvature of the loss function f , and is satisfied for many classes of convex objectives.

For example, any convex loss with L2 regularization will always be strongly convex.

When an objective is strongly convex and Lipschitz continuous, it is standard to say it has condition number κ = L/µ, and here we extend this to say it has L1 condition number κ 1 = L 1 /µ. Assumption 4.

The gradient of each loss function is bounded by some constant σ near the optimal point in the sense of l 1 and l 2 norm, that is, DISPLAYFORM9 This assumption constrains the gradient for each loss function at the optimal point.

We know ∇f (w * ) = 1 n i∇ f i (w * ) = 0, so it is intuitive that each ∇f i (w * ) can be bounded by some value.

Therefore this is a natural assumption to make and it has been used in a lot of other work in this area.

Note that this assumption only needs to hold under the expectation over allf i .

With these assumptions, we proved the following theorem for low-precision SGD: Theorem 1.

Suppose that we run LP-SGD on an objective that satisfies Assumptions 1-4, and with step size α < 1/(2κ 2 µ).

After T LP-SGD update steps (4), selectw T uniformly at random from {w 0 , w 1 , . . . , w T −1 }.

Then, the expected objective gap ofw T is bounded by DISPLAYFORM10 This theorem shows a bound of the expected distance between the result we get at K-th iteration and the optimal value.

By choosing an appropriate step size we can achieve convergence at a 1/T rate, while the limit we converge to is only dependent on dimension-free factors.

Meanwhile, as mentioned in the first section, previous work gives a dimension-dependent bound (1) for the problem, which also converges at a 1/T rate.

2 Therefore our result guarantees a dimension-independent convergence limit without weakening the convergence rate.

It is important to note that, because the dimension-dependent bound in (5) was tight, we should not expect our new result to improve upon the previous theory in all cases.

In the worst case, κ 1 = √ d·κ and similarly σ 1 = √ d · σ; this follows from the fact that for vectors in R d , the norms are related by the inequality DISPLAYFORM11 Substituting this into our result produces a dimension-dependent bound again.

This illustrates the importance of introducing the new parameters κ 1 and σ 1 and requiring that they be bounded; if we could not express our bound in terms of these parameters, the best we could do here is recover a dimension-dependent bound.

Experiments Next, we validate our theoretical results experimentally.

To do this, we analyzed how the size of the noise floor of convergence of SGD and LP-SGD varies as the dimension is changed for a class of synthetic problems.

Importantly, we needed to pick a class of problems for which the parameters L, L 1 , µ, σ, and σ 1 , did not change as we changed the dimension d. To do this, we chose a class of synthetic linear regression models with loss components sampled independently and identically asf DISPLAYFORM12 wherex is a sparse vector sampled to have s nonzero entries each of which is sampled uniformly from {−1, 1}, andỹ is sampled from N (x T w * , β 2 ) for some variance parameter β.

Importantly, the nonzero entries ofx were chosen non-uniformly such that Pr[x i = 0] = p i for some probabilities p i which decrease as i increases; this lets us ensure that µ remains constant as d is increased.

For simplicity, we sampled a fresh loss component of this form at each SGD iteration, which is sometimes called the online setting.

It is straightforward to derive that for this problem DISPLAYFORM13 We set α = 0.01, β = 0.2, p 1 = 0.9, p d = 0.001, and s = 16, we chose each entry of w * uniformly from [−1/2, 1/2], and we set δ such that the low-precision numbers would range from −1 to 1.

FIG1 (b) shows the convergence of SGD and LP-SGD as the dimension d is changed, for both 8-bit and 6-bit quantization.

Notice that while changing d has an effect on the initial convergence rate for both SGD and LP-SGD, it has no effect on the noise ball size, the eventual loss gap that the algorithm converges to.

FIG3 (a) measures this noise ball size more explicitly as the dimension is changed: it reports the loss gap averaged across the second half of the iterates.

Notice that as the dimension d is changed, the average loss gap is almost unchanged, even for very low-precision methods for which the precision does significantly affect the size of the noise ball.

This validates our dimension-free bounds, and shows that they can describe the actual dependence on d in at least one case.

Figure 2(b) validates our results in the opposite way: it looks at how this gap changes as our new parameters σ 1 and L 1 change while d, µ, and σ are kept fixed.

To do this, we fixed d = 1024 and changed s across a range, setting β = 0.8/ √ s, which keeps σ 2 constant as s is changed: this has the effect of changing σ 1 (and, as a side effect, L 1 and L).

We can see from figure 2(b) that changing σ 1 in this way has a much greater effect on LP-SGD than it does on SGD.

This validates our theoretical results, and suggests that σ 1 and L 1 can effectively determine the effect of low-precision compute on SGD.

in some settings (Lee et al., 2017) .

In general, we can quantize to a set of points DISPLAYFORM14 and, just like with linear quantization, we can still use a quantization function Q(w) with randomized rounding that rounds up or down to a number in D in such a way that E [Q(w)] = w for w ∈ [−q n , q n−1 ].

When we consider the quantization variance here, the natural dimension-dependent bound would be DISPLAYFORM15 This is still a tight bound since it holds with equality for a number in the middle of two most distant quantization points.

However, when applied in the analysis of LP-SGD, this bound induces poor performance and often under-represents the actual result.

Here we discuss a specific NLQ method and use it to introduce a tight bound on the quantization variance.

This method has been previously studied as logarithmic quantization or µ−law quantization, and is defined recursively by DISPLAYFORM16 where δ > 0 and ζ > 0 are fixed parameters.

Note that this includes linear quantization as a special case by setting ζ = 0.

It turns out that we can prove a tight dimension-independent bound on the quantization variance of this scheme.

First, we introduce the following definition.

Definition 1.

An unbiased quantization function Q satisfies the dimension-free variance bound with parameters δ, ζ, and η if for all w ∈ [−q n , q n−1 ] and all z ∈ D, DISPLAYFORM17 We can prove that our logarithmic quantization scheme satisfies this bound.

Lemma 1.

The logarithmic quantization scheme (7) satisfies the dimension-free variance bound with parameters δ, ζ, and η = ζ 2 4(ζ+1) < ζ 4 .

Notice that this bound becomes identical to the linear quantization bound (6) when ζ = 0, so this result is a strict generalization of our results from the linear quantization case.

With this setup, we can apply NLQ to the low-precision training algorithms we have studied earlier in this paper.

Theorem 2.

Suppose that we run LP-SGD on an objective that satisfies Assumptions 1-4, and using a quantization scheme that satisfies the dimension-free variance bound.

If ζ < 1 κ , then DISPLAYFORM18 This theorem is consistent with Theorem 1 in that, if we set ζ = η = 0, which makes logarithmic quantization linear, they would have an identical result.

If we fix the representable range R (the largest-magnitude values representable in the low-precision format) and choose our quantization parameters optimally, we get the result that the number of bits we need to achieve objective gap is log 2 O (Rσ/ε)·log 1+σ 1 /σ .

This bound is notable because even in the worst case where we do not have a bound on σ 1 and must use σ 1 ≤ √ d·σ, this bound gives us log 2 O (Rσ/ε)·log 1+ √ d .

That is, while a dimension-dependent factor still remains, it is now "hidden" within a log term.

This greatly decreases the effect of the dimension, and suggests that NLQ may be a promising technique to use for low-precision training at scale.

Also note that, although this bound holds only when we set ζ < 1 κ = µ L , which to some extent limits the acceleration of the strides in logarithmic quantization, the bound µ L is independent of σ and σ 1 , thus this effect of "pushing " σ 1 into a log term is independent of the setting of ζ.

Floating point.

Next, we look at another type of non-linear quantization that is of great practical use: floating-point quantization (FPQ).

Here, the quantization points are simply floating-point numbers with some fixed number of exponential bits b e and mantissa bits b m .

Floating-point numbers are represented in the form DISPLAYFORM19 where "exponent" is a b e -bit unsigned number, the m i are the b m bits of the mantissa, and "bias" is a term that sets the range of the representable numbers by determining the range of the exponent.

In standard floating point numbers, the exponent ranges from [−2 be−1 + 2, 2 be−1 − 1], which corresponds to a bias of 2 be−1 − 1.

To make our results more general, we also consider non-standard bias by defining a scaling factor s = 2 −(bias−standard bias) ; the standard bias setting corresponds to s = 1.

We also consider the case of denormal floating point numbers, which tries to address underflow by replacing the 1 in (8) with a 0 for the smallest exponent value.

Under these conditions, we can prove that floating-point quantization satisfies the bound in Definition 1.

Lemma 2.

The FPQ scheme using randomized rounding satisfies the dimension-free variance bound with parameters δ normal , ζ, and η for normal FPQ and δ denormal , ζ, and η for denormal FPQ where DISPLAYFORM20 This bound can be immediately combined with Theorem 2 to produce dimension-independent bounds on the convergence rate of low-precision floating-point SGD.

If we are given a fixed number of total bits b = b e + b m , we can minimize this upper bound on the objective gap to try to predict the best way to allocate our bits between the exponent and the mantissa.

Unfortunately, there is no analytical expression for this optimal choice of b e .

To give a sense of the asymptotic behavior of this optimal allocation, we present upper and lower bounds on it.

Theorem 3.

When using FPQ without denormal numbers, given b total bits, the optimal number of exponential bits b e such that the asymptotic upper bound on the objective gap given by Theorem 2 is minimized is in the interval between: Theorem 4.

When using denormal FPQ, given b total bits, the optimal number of exponential bits b e such that the asymptotic upper bound on the objective gap, as T → ∞ and α → 0, given by Theorem 2 is minimized is in the interval between: DISPLAYFORM21 DISPLAYFORM22 and DISPLAYFORM23 where e denotes the base of the natural logarithm and W stands for the Lambert W function.

In cases where neither of these two values exists, the noise ball size increases as b e , thus b e = 2 would be the optimal setting, which is equivalent to linear quantization.

These theorems give us an idea of where the optimal setting of b e lies such that the theoretical asymptotic error is minimized.

When using normal FPQ, this optimal assignment of b e is O(log(b)), and for denormal FPQ the result is independent of b. This suggests that once the total number of bits grows past a threshold, we should assign most of or all the extra bits to the mantissa.

Experiments For FPQ, we ran experiments on two different data sets.

First, we ran LP-SGD on the same synthetic data set that we used for linear regression.

Here we used normal FPQ with 20 bits in total, and we get the result in FIG5 (a).

In this diagram, we plotted the empirical noise ball size, its theoretical upper bound, and the optimal interval for b e as Theorem 3 predicts.

As the figure shows, our theorem accurately predicts the optimal setting of exponential bits, which is 5 in this case, to minimize both the theoretical upper bound and the actual empirical result of the noise ball size, despite the theoretical upper bound being loose.

Second, we ran LP-SGD on the MNIST dataset (Deng, 2012) .

To set up the experiment, we normalized the MNIST data to be in [0, 1] by dividing by 255, then subtracted out the mean for each features.

We ran multiclass logistic regression using an L2 regularization constant of 10 −4 and a step size of α = 10 −4 , running for 500 total epochs (passes through the dataset) to be sure we converged.

For this task, our (measured) problem parameters were L = 37.41, L 1 = 685.27, σ = 2.38, σ 1 = 29.11, and d = 784.

In FIG5 (b), we plotted the observed loss gap, averaged across the last ten epochs, for LP-SGD using various 16-bit floating point formats.

We also plot our theoretical bound on the loss gap, and the predicted optimal number of exponential bits to use based on that bound.

Our results show that even though our bound is very loose for this task, it still predicts the right number of bits to use with reasonable accuracy.

This experiment also validates the use of IEEE standard half-precision floating-point numbers, which have 5 exponential bits, for this sort of task.

In this paper, we present dimension-independent bounds on the convergence of SGD when applied to low-precision training.

We point out the conditions under which such bounds hold.

We further extend our results to non-linear methods of quantization: logarithmic quantization and floating point quantization.

We analyze the performance of SGD under logarithmic quantization and demonstrate that NLQ is a promising method for reducing the number of bits required in low-precision training.

We also presented ways in which our theory could be used to suggest how to allocate bits between exponent and mantissa when FPQ is used.

We hope that our work will encourage further investigation of non-linear quantization techniques.

A ALGORITHMIn our work, we presented dimension-free bounds on the performance of low-precision SGD, here we present the algorithm in detail.

given: n loss functionsfi, number of epochs T , step size α, and initial iterate w0.given: low-precision quantization function Q. for t = 0 to T − 1 do sample it uniformly from {1, 2, · · · , n}, quantize wt+1 ← Q wt − α∇fi t (wt) end for return wT B PROOF FOR RESULTS IN TABLE 1 As mentioned in the caption of Table 1 , here only we consider the convergence limit, that is, we assume α → 0, T → ∞, and we compute the minimum number of bits b we would require in order for the limit to be less than some small positive ε.

Meanwhile, we denote the radius of the representable range by R and we assume R = w * 2 without loss of generality, as this is the worst case for all our bounds that depend on w * 2 .

Then in linear quantization, we have: DISPLAYFORM0 and in non-linear quantization, we need: DISPLAYFORM1 In the following proof we'll take the equality for these two inequalities.

In previous work Li et al. FORMULA0 , we have DISPLAYFORM0 here we re-denote G as σ max for concordance with our result.

Here σ 2 max is an upper bound on the second moment of the stochastic gradient samples E f (w) and set the limit (as α → 0 and T → ∞) to be ≤ ε, and notice that 2 b−1 − 1 > 2 b−2 , then we have: DISPLAYFORM1

In Theorem 1, we know that DISPLAYFORM0 2 1 µ 4 Set the limit (as α → 0 and T → ∞) to be ≤ ε, then we need: DISPLAYFORM1 Then for sufficiently small ε, more explicitly, ε that satisfies DISPLAYFORM2 will satisfy the requirements, and we will get DISPLAYFORM3 This is the expression that we wanted.

Notice that even if we did not invoke small ε in the above big-O analysis, we can set DISPLAYFORM4 L 1 Then our number of bits would look like DISPLAYFORM5 which shows explicitly that we have replaced the dimension factor with parameters of the loss functions.

In Theorem 2, we know that, if ζ < 1 κ , then DISPLAYFORM0 Set the limit (as α → 0 and T → ∞) to be ≤ ε and replace w * 2 with R; then we get DISPLAYFORM1 So, in addition to our requirement that ζ ≤ κ −1 , it suffices to have DISPLAYFORM2 If we set DISPLAYFORM3 then all our other requirements will be satisfied for sufficiently small ε.

Specifically, we need ε to be small enough that DISPLAYFORM4 As is standard in big-O analysis, we assume that ε is small enough that these requirements are satisfied, in which case our assignment of δ and ζ, combined with the results of Theorem 2, is sufficient to ensure an objective gap of ε.

Next, starting from (9), the number of bits we need for non-linear quantization must satisfy DISPLAYFORM5 Since we know that 0 ≤ ζ < 1, it follows that log(1 + ζ) ≥ ζ/2.

So in order for the above to be true, it suffices to have DISPLAYFORM6 Since 2 b−1 − 1 > 2 b−2 , it follows that it suffices to have DISPLAYFORM7 Finally, using our assignment of δ and ζ gives us DISPLAYFORM8 This is the expression that we wanted.

Notice that even if we did not invoke small ε in the above big-O analysis, we would still get a rate in which all of our 1 -dependent terms are inside the doublelogarithm, because none of the requirements above that constrain ζ are 1 -dependent.

To be explicit, to do this we would set δ and ζ to be DISPLAYFORM9 Then our number of bits would look like DISPLAYFORM10 which shows explicitly that any 1 -dependent terms are inside the double logarithm.

Before we prove the main theorems presented in the paper, we will prove the following lemmas that will be useful later, as well as the lemmas we presented before.

The proof of lemma 1 can be extracted from the proof of lemma 5 that we will show later.

Proof of Lemma 2.

Here we consider the positive case first, then symmetrically the negative case also holds.

First, for normal FPQ, the set of quantization points are: DISPLAYFORM0 and we set the parameters for the nonlinear quantization bound to be: DISPLAYFORM1 For any w within representable range, we can assume it is in [

q i , q i+1 ), then DISPLAYFORM2 So now we only need to prove that DISPLAYFORM3 First, we consider a special case where q i = 0.

In this case, DISPLAYFORM4 and similarly for v = δ, DISPLAYFORM5 Next, we consider the case where q i = 0.

In this case, we can assume DISPLAYFORM6 observe that ζ − η > 0, so the right hand side is a concave function of v, thus it achieves minimum at either v = 0 or v = q i .

At v = q i : DISPLAYFORM7 and at v = 0, since q i+1 ≤ (1 + ζ)q i and q i ≤ w, DISPLAYFORM8 which is a positive parabola.

Recall that η = DISPLAYFORM9 , therefore RHS − LHS ≥ 0.

Now we extend this conclusion to the case where v ≤ 0.

In this case, DISPLAYFORM10 since w, ζ, δ, η are all positive, this is apparently a decreasing function of v, thus it achieves minimum at v = 0, which is what we have already proven.

So far, we've proven the lemma in the case of w ≥ 0, v ≥ 0 and w ≥ 0, v ≤ 0, and symmetrically it holds for w ≤ 0, v ≤ 0 and w ≤ 0, v ≥ 0, which indicates that we can extend D to be a set containing both positive and negative numbers.

In the de-normal FPQ case, the set of quantization points are: DISPLAYFORM11 and we set the parameters for the nonlinear quantization bound to be: DISPLAYFORM12 The proof for this case follows the exact same structure as the normal FPQ case.

Lemma 3.

Under condition of linear quantization when using low-precision representation (δ, b), DISPLAYFORM13 where Q is the linear quantization function.

Proof of Lemma 3. (This proof follows the same structure as the proof for lemma 1 in (De Sa et al., 2018) ) First, observe that this lemma holds if it holds for each dimension, so we only need to prove that for any w, v ∈ R where Q (δ,b) (w) = w, i.e. w ∈ dom(δ, b), DISPLAYFORM14 then we can sum up all the dimensions to get the result.

Now we consider the problem in two situations.

First, if w + v is within the range representable by (δ, b), then E Q (δ,b) (w + v) = w + v. In this case, DISPLAYFORM15

Observe that this trivially holds for v = 0, and is symmetrical for positive and negative v. Without loss of generality we assume v > 0, let z be the rounded-down quantization of v, then we have z ≥ 0.

Then Q (δ,b) (v) will round to z + δ (the rounded-up quantization of v) with probability v−z δ , and it will round to z with probability DISPLAYFORM0 .

This quantization is unbiased because DISPLAYFORM1 Thus, its variance will be DISPLAYFORM2 In the other case, when w + v is on the exterior of the representable region, the quantization function Q (δ,b) just maps it to the nearest representable value.

Since w * is in the interior of the representable region, this operation will make w + v closer to w * .

Thus, DISPLAYFORM3 and so it will certainly be the case that DISPLAYFORM4 Now that we've proven the inequality for one dimension, we can sum up all d dimensions and get DISPLAYFORM5 For completeness, we also re-state the proof of following lemma, which was presented as equation FORMULA20 in (Johnson and Zhang, 2013) , and here we present the proof for this lemma used in (De Sa et al., 2018) .Lemma 4.

Under the standard condition of Lipschitz continuity, if i is sampled uniformly at random from {1, . . .

, N }, then for any w, DISPLAYFORM6 Proof of Lemma 4.

For any i, define DISPLAYFORM7 Clearly, if i is sampled randomly as in the lemma statement, E [g i (w)] = f (w).

But also, w * must be the minimizer of g i , so for any w DISPLAYFORM8 where the second inequality follows from the Lipschitz continuity property.

Re-writing this in terms of f i and averaging over all the i now proves the lemma statement.

Lemma 5.

Under the condition of logarithmic quantization, for any DISPLAYFORM9 where Q is the non-linear quantization function.

Note that the proof this lemma naturally extends to lemma 1, thus we omitted the proof for lemma 1 and just present the proof for lemma 5.Proof of Lemma 5.

Here we only consider the positive case first, where DISPLAYFORM10 with [0, q n−1 ] being the representable range of D. As for the negative case, we will show later that it holds symmetrically.

Observe that this lemma holds if it holds for each dimension, so we only need to prove that for any w, v ∈ R where v ∈ D, DISPLAYFORM11 then we can sum up all the dimensions and use Cauchy-Schwarz inequality to get the result.

Now we consider the problem in two situations.

First, if w is outside the representable range, the quantization function Q just maps it to the nearest representable value.

Since w * is in the interior of the representable range, this operation will make w closer to w * .

Thus, DISPLAYFORM12 and so it will certainly be the case that DISPLAYFORM13 Second, if w is within the representable range, then E [Q(w)] = w. In this case, DISPLAYFORM14 Since w is within representable range, we can assume it is in [q i , q i+1 ), then DISPLAYFORM15 So now we only need to prove that DISPLAYFORM16 observe that ζ − η > 0, so the right hand side is a concave function of v, thus it achieves minimum at either v = 0 or v = q i .

At v = q i : DISPLAYFORM17 and at v = 0: DISPLAYFORM18 which is a positive parabola.

Recall that η = DISPLAYFORM19 , therefore RHS − LHS ≥ 0.

Now we extend this conclusion to the case where v ≤ 0.

In this case, DISPLAYFORM20 since w, ζ, δ, η are all positive, this is apparently a decreasing function of v, thus it achieves minimum at v = 0, which is what we have already proven.

So far, we've proven the lemma in the case of w ≥ 0, v ≥ 0 and w ≥ 0, v ≤ 0, and symmetrically it holds for w ≤ 0, v ≤ 0 and w ≤ 0, v ≥ 0, which indicates that we can extend D to be a set containing both positive and negative numbers, and we can reset D to be DISPLAYFORM21 Now we have proven all the lemmas we need.

Next, we make some small modifications to the assumptions (weakening them) so that our theorems are shown in a more general sense.

For assumption 2, we change it to:Assumption 5.

All the gradients of the loss functions f i are L 1 -Lipschitz continuous in the sense of 1-norm to p-norm, that is, ∀i ∈ {1, 2, · · · n}, ∀x, y, DISPLAYFORM22 While in the body of the paper and in our experiments we choose p = 2 for simplicity, here we are going to prove that a generalization of Theorem 1 holds for all real numbers p.

We also need a similar generalization of Assumption 3.

Assumption 6.

The average of the loss functions f = 1 n i f i is µ 1 − strongly convex near the optimal point in the sense of p-norm, that is, DISPLAYFORM23 with p being any real number.

This assumption is essentially the same as the assumption for strong convexity that we stated before, since in practice we would choose p = 2 and then µ 1 and µ would be the same.

But here we are actually presenting our result in a stronger sense in that we can choose any real number p and the proof goes the same.

Now we are ready to prove the theorems.

Note that the result of the following proof contains µ 1 since we are proving a more general version of our theorems; substituting them with µ will lead to the same result that we stated before.

Proof of Theorem 1.

In low-precision SGD, we have: DISPLAYFORM24 ) by lemma 3, we know that DISPLAYFORM25 where the second inequality holds due to the strongly convexity assumption.

According to the assumptions we had, we have: DISPLAYFORM26 where the last inequality holds due to assumption 2 where we let p = 2.

Applying this result to the previous formula and we will have: DISPLAYFORM27 Here we introduce a positive constant C that we'll set later, and by basic inequality we get DISPLAYFORM28 one setting C to be αµ − α 2 L 2 , we will have: DISPLAYFORM29 since we can set α to be small enough such that αL 2 ≤ µ 2 , then the result will become: DISPLAYFORM30 now we sum up this inequality from t = 0 to t = T − 1 and divide by 2αT , then we get: DISPLAYFORM31 and since we samplew uniformly from (w o , w 1 , · · · , w T −1 ), we get DISPLAYFORM32 Proof of Theorem 2 .

In low-precision SGD, we have: DISPLAYFORM33 Proof of Theorem 3.

In the normal FPQ case, the set of quantization points are: D = {0} ∪ s · 1 + x n m · 2 y | x = 0, 1, · · · , n m − 1, y = − n e 2 + 2, · · · , n e 2 − 1 then the parameters for the nonlinear quantization bound can be computed as: 1 n m = 4sσ 1 1 √ 2 ne + σ w * 2 n e C let the derivative over n e to be 0 and we get: DISPLAYFORM34 ∂A ∂n e = −2(ln 2)sσ 1 1 √ 2 ne + σ w * 2 1 C = 0, √ 2 ne = 2(ln 2)sσ 1 C σ w * 2 n e = 2 log 2 2(ln 2)sσ 1 C σ w *

, b e = log 2 2b + 2 log 2 2(ln 2)sσ 1 σ w *

And when b is small, δ, ζ, η are large and the dominating term for the noise ball is B = δL 1 +ζL w * 2 +ζσ = 4sL 1 DISPLAYFORM0 n e C let the derivative of n e to be 0 and we get:∂B ∂n e = −2(ln 2)sL 1 1 √ 2 ne + (L w * 2 + σ) 1 C = 0, √ 2 ne = 2(ln 2)sL 1 C L w * 2 + σ n e = 2 log 2 2(ln 2)sL 1 C L w * 2 + σ , b e = log 2 2b + 2 log 2 2(ln 2)sL 1 L w * 2 + σ For b such that neither the terms dominates the result, we know the noise ball size is: ∂A ∂n e ne=2 log 2 2(ln 2)sσ 1 C σ w * 2 = 0, ∂B ∂n e ne=2 log 2 2(ln 2)sL 1 C L w * 2 +σ = 0 then we know the solution of .

<|TLDR|>

@highlight

we proved dimension-independent bounds for low-precision training algorithms

@highlight

This paper discusses conditions under which the convergence of training models with low-precision weights do not rely on model dimension.
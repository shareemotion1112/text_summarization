Various gradient compression schemes have been proposed to mitigate the communication cost in distributed training of large scale machine learning models.

Sign-based methods, such as signSGD (Bernstein et al., 2018), have recently been gaining popularity because of their simple compression rule and connection to adaptive gradient methods, like ADAM.

In this paper, we perform a general analysis of sign-based methods for non-convex optimization.

Our analysis is built on intuitive bounds on success probabilities and does not rely on special noise distributions nor on the boundedness of the variance of stochastic gradients.

Extending the theory to distributed setting within a parameter server framework, we assure exponentially fast variance reduction with respect to number of nodes, maintaining 1-bit compression in both directions and using small mini-batch sizes.

We validate our theoretical findings experimentally.

One of the key factors behind the success of modern machine learning models is the availability of large amounts of training data (Bottou & Le Cun, 2003; Krizhevsky et al., 2012; Schmidhuber, 2015) .

However, the state-of-the-art deep learning models deployed in industry typically rely on datasets too large to fit the memory of a single computer, and hence the training data is typically split and stored across a number of compute nodes capable of working in parallel.

Training such models then amounts to solving optimization problems of the form

where f m : R d → R represents the non-convex loss of a deep learning model parameterized by x ∈ R d associated with data stored on node m.

Arguably, stochastic gradient descent (SGD) (Robbins & Monro, 1951; Vaswani et al., 2019; Qian et al., 2019) in of its many variants (Kingma & Ba, 2015; Duchi et al., 2011; Schmidt et al., 2017; Zeiler, 2012; Ghadimi & Lan, 2013 ) is the most popular algorithm for solving (1).

In its basic implementation, all workers m ∈ {1, 2, . . .

, M } in parallel compute a random approximation g m (x k ) of ∇f m (x k ), known as the stochastic gradient.

These approximations are then sent to a master node which performs the aggregation

The aggregated vector is subsequently broadcast back to the nodes, each of which performs an update of the form x k+1 = x k − γ kĝ (x k ), thus updating their local copies of the parameters of the model.

Typically, communication of the local gradient estimatorsĝ m (x k ) to the master forms the bottleneck of such a system (Seide et al., 2014; Zhang et al., 2017; Lin et al., 2018) .

In an attempt to alleviate this communication bottleneck, a number of compression schemes for gradient updates have been proposed and analyzed Wen et al., 2017; Khirirat et al., 2018;  (Bernstein et al., 2019) signSGD, Theorem 1

Step size

Weak noise assumptions?

ρi > Mishchenko et al., 2019) .

A compression scheme is a (possibly randomized) mapping Q : R d → R d , applied by the nodes toĝ m (x k ) (and possibly also by the master to aggregated update in situations when broadcasting is expensive as well) in order to reduce the number of bits of the communicated message.

Sign-based compression.

Although most of the existing theory is limited to unbiased compression schemes, i.e., on operators Q satisfying EQ(x) = x, biased schemes such as those based on communicating signs of the update entries only often perform much better (Seide et al., 2014; Strom, 2015; Wen et al., 2017; Carlson et al., 2015; Balles & Hennig, 2018; Bernstein et al., 2018; Zaheer et al., 2018; Liu et al., 2019) .

The simplest among these sign-based methods is signSGD (see also Algorithm 1; Option 1), whose update direction is assembled from the component-wise signs of the stochastic gradient.

Adaptive methods.

While ADAM is one of the most popular adaptive optimization methods used in deep learning (Kingma & Ba, 2015) , there are issues with its convergence (Reddi et al., 2019) and generalization (Wilson et al., 2017) properties.

It was noted in Balles & Hennig (2018) that the behaviour of ADAM is similar to a momentum version of signSGD.

Connection between sign-based and adaptive methods has long history, originating at least in Rprop (Riedmiller & Braun, 1993) and RMSprop (Tieleman & Hinton, 2012) .

Therefore, investigating the behavior of signSGD can improve our understanding on the convergence of adaptive methods such as ADAM.

We now summarize the main contributions of this work.

Our key results are summarized in Table 1.

1 In fact, bounded variance assumption, being weaker than bounded second moment assumption, is stronger (or, to be strict, more curtain) than SPB assumption in the sense of differential entropy, but not in the direct sense.

The entropy of probability distribution under the bounded variance assumption is bounded, while under the SPB assumption it could be arbitrarily large.

This observation is followed by the fact that for continuous random variables, the Gaussian distribution has the maximum differential entropy for a given variance (see https://en.wikipedia.org/wiki/Differential_entropy).

• 2 methods for 1-node setup.

In the M = 1 case, we study two general classes of sign based methods for minimizing a smooth non-convex function f .

The first method has the standard form

while the second has a new form not considered in the literature before:

• Key novelty.

The key novelty of our methods is in a substantial relaxation of the requirements that need to be imposed on the gradient estimatorĝ(x k ) of the true gradient ∇f (x k ).

In sharp contrast with existing approaches, we allowĝ(x k ) to be biased.

Remarkably, we only need one additional and rather weak assumption onĝ(x k ) for the methods to provably converge: we require the signs of the entries ofĝ(x k ) to be equal to the signs of the entries of ∇f (x k ) with a probability strictly larger than 1 /2 (see Section 2; Assumption 1).

We show through a counterexample (see Section 2.2) that this assumption is necessary.

• Geometry.

As a byproduct of our analysis, we uncover a mixed l 1 -l 2 geometry of sign descent methods (see Section 3).

• Convergence theory.

We perform a complexity analysis of methods (2) and (3) (see Section 4.1; Theorem 1).

While our complexity bounds have the same O( 1 / √ K) dependence on the number of iterations, they have a better dependence on the smoothness parameters associated with f .

Theorem 1 is the first result on signSGD for non-convex functions which does not rely on mini-batching, and which allows for step sizes independent of the total number of iterations K. Finally, Theorem 1 in Bernstein et al. (2019) can be recovered from our general Theorem 1.

Our bounds are cast in terms of a novel norm-like function, which we call the ρ-norm, which is a weighted l 1 norm with positive variable weights.

• Distributed setup.

We extend our results to the distributed setting with arbitrary M (Section 4.2), where we also consider sign-based compression of the aggregated gradients.

In this section we describe our key (and weak) assumption on the gradient estimatorĝ(x) of the true gradient ∇f (x), and give an example which shows that without this assumption, method (2) can fail.

Assumption 1 (SPB: Success Probability Bounds).

For any x ∈ R d , we have access to an independent (and not necessarily unbiased) estimatorĝ(x) of the true gradient g(x) := ∇f (x) that satisfies

for all x ∈ R d and all i ∈ {1, 2, . . .

, d}.

We will refer to the probabilities ρ i as success probabilities.

As we will see, they play a central role in the convergence of sign based methods.

We stress that Assumption 1 is the only assumption on gradient noise in this paper.

Moreover, we argue that it is reasonable to require from the sign of stochastic gradient to show true gradient direction more likely than the opposite one.

Extreme cases of this assumption are the absence of gradient noise, in which case ρ i = 1, and an overly noisy stochastic gradient, in which case ρ i ≈ 1 2 .

Remark 1.

Assumption 1 can be relaxed by replacing bounds (4) with

However, if Prob(signĝ i (x) = 0) = 0 (e.g. in the case ofĝ i (x) has continuous distributions), then these two bounds are identical.

Under review as a conference paper at ICLR 2020 Extension to stochastic sign oracle.

Notice that we do not requireĝ to be unbiased.

Moreover, we do not assume uniform boundedness of the variance, or of the second moment.

This observation allows to extend existing theory to more general sign-based methods with a stochastic sign oracle.

By a stochastic sign oracle we mean an oracle that takes x k ∈ R d as an input, and outputs a random vectorŝ k ∈ R d with entries in ±1.

However, for the sake of simplicity, in the rest of the paper we will work with the signSGD formulation, i.e., we letŝ k = signĝ(x k ).

Here we analyze a counterexample to signSGD discussed in Karimireddy et al. (2019) .

Consider the following least-squares problem with unique minimizer x * = (0, 0):

, where ε ∈ (0, 1) and stochastic gradientĝ(x) = ∇ a i , x 2 = 2 a i , x a i with probabilities 1/2 for i = 1, 2.

Let us take any point from the line l = {(z 1 , z 2 ) : z 1 + z 2 = 2} as initial point x 0 for the algorithm and notice that signĝ(x) = ±(1, −1) for any x ∈ l.

Therefore, signSGD with any step-size sequence remains stuck along the line l, whereas the problem has a unique minimizer at the origin.

We now investigate the cause of the divergence.

In this counterexample, Assumption 1 is violated.

Indeed, note that for i = 1, 2.

By S := {x ∈ R 2 : a 1 , x · a 2 , x > 0} = ∅ denote the open cone of points having either an acute or an obtuse angle with both a i '

s.

Then for any x ∈ S, the sign of the stochastic gradient is ±(1, −1) with probabilities 1 /2.

Hence for any x ∈ S, we have low success probabilities:

So, in this case we have an entire conic region with low success probabilities, which clearly violates (4).

Furthermore, if we take a point from the complement open coneS c , then the sign of stochastic gradient equals to the sign of gradient, which is perpendicular to the axis of S (thus in the next step of the iteration we get closer to S).

For example, if a 1 , x < 0 and a 2 , x > 0, then signĝ(x) = (1, −1) with probability 1, in which case x − γ signĝ(x) gets closer to low success probability region S.

In summary, in this counterexample there is a conic region where the sign of the stochastic gradient is useless (or behaves adversarially), and for any point outside that region, moving direction (which is the opposite of the sign of gradient) leads toward that conic region.

To justify our SPB assumption, we show that it holds under general assumptions on gradient noise.

Lemma 1 (see B.1).

Assume that for any point x ∈ R d , we have access to an independent and unbiased estimatorĝ(x) of the true gradient g(x).

Assume further that each coordinateĝ i has a unimodal and symmetric distribution with variance σ

Next, we remove the distribution condition and add a strong growth condition (Schmidt & Le Roux, 2013; Vaswani et al., 2019) together with fixed mini-batch size.

Lemma 2 (see B.2).

Assume that for any point x ∈ R d , we have access to an independent, unbiased estimatorĝ(x) of the true gradient g(x), with coordinate-wise bounded variances σ

for some constant c. Then, choosing a mini-batch size τ > 2c, we get

Finally, we give an adaptive condition on mini-batch size for the SPB assumption to hold.

Lemma 3 (see B.3).

Assume that for any point x ∈ R d we have access to an independent and unbiased estimatorĝ(x) of the true gradient g(x).

Let σ 2 i = σ 2 i (x) be the variance and ν

Under review as a conference paper at ICLR 2020

In this section we introduce the concept of a norm-like function, which call ρ-norm, induced from success probabilities.

Used to measure gradients in our convergence rates, ρ-norm is a technical tool enabling the analysis.

be the collection of probability functions from the SPB assumption.

We define the ρ-norm of gradient

Note that ρ-norm is not a norm as it may not satisfy the triangle inequality.

However, under SPB assumption, ρ-norm is positive definite as it is a weighted l 1 norm with positive (and variable) weights

, and g ρ = 0 if and only if g = 0.

Under the assumptions of Lemma 2, ρ-norm can be lower bounded by a weighted l 1 norm with positive constant weights 1 − 2c

Under the assumptions of Lemma 1, ρ-norm can be lower bounded by a mixture of the l 1 and squared l 2 norms:

Note that l 1,2 -norm is again not a norm.

However, it is positive definite, continuous and order preserving, i.e., for any g k , g,g ∈ R d we have: i) g l 1,2 ≥ 0 and g l 1,2 = 0 if and only if g = 0;

.

From these three properties it follows that g k l 1,2 → 0 implies g k → 0.

These properties are important as we will measure convergence rate in terms of the l 1,2 norm in the case of unimodal and symmetric noise assumption.

To understand the nature of the l 1,2 norm, consider the following two cases when σ i (x) ≤ c|g i (x)| +c for some constants c,c ≥ 0.

If the iterations are in ε-neighbourhood of a minimizer x * with respect to the l ∞ norm (i.e., max 1≤i≤d |g i | ≤ ε), then the l 1,2 norm is equivalent to scaled l 2 norm squared:

On the other hand, if iterations are away from a minimizer (i.e., min 1≤i≤d |g i | ≥ L), then the l 1,2 -norm is equivalent to scaled l 1 norm:

g 1 .

These equivalences are visible in Figure 1 , where we plot the level sets of g → g l 1,2 at various distances from the origin.

Similar mixed norm observation was also noted in Bernstein et al. (2019) .

Now we turn to our theoretical results of sign based methods.

First we give our general convergence results under the SPB assumption.

Afterwards, we present convergence result in the distributed setting under the unimodal and symmetric noise assumptions.

Throughout the paper we assume that f :

and is L-smooth with some non-negative constants

That is, we assume that

We allow f to be nonconvex.

We now state our convergence result for Algorithm 1 under the general SPB assumption.

Under review as a conference paper at ICLR 2020

Theorem 1 (Non-convex convergence of signSGD, see B.4).

Under the SPB assumption, signSGD (Algorithm 1 with Option 1) with step sizes γ k = γ 0 / √ k + 1 converges as follows

If γ k ≡ γ > 0, we get 1 /K convergence to a neighbourhood of the solution:

We now comment on the above result:

• Generalization.

Theorem 1 is the first general result on signSGD for non-convex functions without mini-batching, and with step sizes independent of the total number of iterations K. Known convergence results (Bernstein et al., 2018; on signSGD use mini-batches and/or step sizes dependent on K. Moreover, they also use unbiasedness and unimodal symmetric noise assumptions, which are stronger assumptions than our SPB assumption (see Lemma 1).

Finally, Theorem 1 in Bernstein et al. (2019) can be recovered from Theorem 1 (see Section D for the details).

• Convergence rate.

Rates (6) and (7) can be arbitrarily slow, depending on the probabilities ρ i .

This is to be expected.

At one extreme, if the gradient noise was completely random, i.e., if ρ i ≡ 1/2, then the ρ-norm would become identical zero for any gradient vector and rates would be trivial inequalities, leading to divergence as in the counterexample.

At other extreme, if there was no gradient noise, i.e., if ρ i ≡ 1, then the ρ-norm would be just the l 1 norm and from (6) we get the rateÕ(1/ √ K) with respect to the l 1 norm.

However, if we know that ρ i > 1/2, then we can ensure that the method will eventually converge.

• Geometry.

The presence of the ρ-norm in these rates suggests that there is no particular geometry (e.g., l 1 or l 2 ) associated with signSGD.

Instead, the geometry is induced from the success probabilities.

For example, in the case of unbiased and unimodal symmetric noise, the geometry is described by the mixture norm l 1,2 .

• Practicality.

The rate (7) (as well as (30)) supports the common learning schedule practice of using a constant step size for a period of time, and then halving the step-size and continuing this process.

For a reader interested in comparing Theorem 1 with a standard result for SGD, we state the standard result in the Section C. We now state a general convergence rate for Algorithm 1 with Option 2.

Theorem 2 (see B.5).

Under the SPB assumption, Algorithm 1 (Option 2) with step sizes γ k = γ 0 / √ k + 1 converges as follows:

In the case of constant step size γ k = γ > 0, the same rate as (7) is achieved.

Comparing Theorem 2 with Theorem 1, notice that a small modification in Algorithm 1 can remove the log-dependent factor from (6); we then bound the average of past gradient norms instead of the minimum.

On the other hand, in a big data regime, function evaluations in Algorithm 1 (Option 2, line 4) are infeasible.

Clearly, Option 2 is useful only when one can afford function evaluations and has rough estimates about the gradients (i.e., signs of stochastic gradients).

This option should be considered within the framework of derivative-free optimization.

In this part we present the convergence result of distributed signSGD (Algorithm 2) with majority vote introduced in Bernstein et al. (2018) .

Majority vote is considered within a parameter server framework, where for each coordinate parameter server receives one sign from each node and sends Under review as a conference paper at ICLR 2020 back the sign sent by the majority of nodes.

Known convergence results (Bernstein et al., 2018; use O(K) mini-batch size as well as O(1/K) constant step size.

In the sequel we remove this limitations extending Theorem 1 to distributed training.

In distributed setting the number of nodes M get involved in geometry introducing new ρ M -norm, which is defined by the regularized incomplete beta function I (see B.6).

Now we can state the convergence rate of distributed signSGD with majority vote.

Theorem 3 (Non-convex convergence of distributed signSGD, see B.6).

Under SPB assumption, distributed signSGD (Algorithm 2) with step sizes γ k = γ 0 / √ k + 1 converges as follows

For constant step sizes γ k ≡ γ > 0, we have convergence up to a level proportional to step size γ:

Variance Reduction.

Using Hoeffding's inequality, we show that

1 , where ρ(x) = min 1≤i≤d ρ i (x) > 1 /2.

Hence, in some sense, we have exponential variance reduction in terms of number of nodes (see B.7).

Number of nodes.

Notice that theoretically there is no difference between 2l−1 and 2l nodes, and this in not a limitation of the analysis.

Indeed, as it is shown in the proof, expected sign vector at the master with M = 2l − 1 nodes is the same as with M = 2l nodes: E sign(ĝ

is the sum of stochastic sign vectors aggregated from nodes.

The intuition behind this phenomenon is that majority vote with even number of nodes, e.g. M = 2l, fails to provide any sign Under review as a conference paper at ICLR 2020 with little probability (it is the probability of half nodes voting for +1, and half nodes voting for −1).

However, if we remove one node, e.g. M = 2l − 1, then master receives one sign-vote less but gets rid of that little probability of failing the vote (sum of odd number of ±1 cannot vanish).

So, somehow this two things cancel each other and we gain no improvement in expectation adding one more node to parameter server framework with odd number of nodes.

We verify our theoretical results experimentally using the MNIST dataset with feed-forward neural network (FNN) and the well known Rosenbrock (non-convex) function with d = 10 variables:

Stochastic formulation of minimization problem for Rosenbrock function is as follows: at any point x ∈ R d we have access to biased stochastic gradientĝ(x) = ∇f i (x) + ξ, where index i is chosen uniformly at random from {1, 2, . . .

, d − 1} and ξ ∼ N (0, ν 2 I) with ν > 0.

Figure 2 illustrates the effect of multiple nodes in distributed training with majority vote.

As we see increasing the number of nodes improves the convergence rate.

It also supports the claim that in expectation there is no improvement from 2l − 1 nodes to 2l nodes.

Figure 4 shows the robustness of SPB assumption in the convergence rate (7) with constant step size.

We exploited four levels of noise in each column to demonstrate the correlation between success probabilities and convergence rate.

In the first experiment (first column) SPB assumption is violated strongly and the corresponding rate shows divergence.

In the second column, probabilities still violating SPB assumption are close to the threshold and the rate shows oscillations.

Next columns show the improvement in rates when success probabilities are pushed to be close to 1.

Under review as a conference paper at ICLR 2020 Figure 5: Performance of signSGD with variable step size (γ 0 = 0.25) under four different noise levels (mini-batch size 1, 2, 5, 7) using Rosenbrock function.

As in the experiments of Figure 4 with constant step size, these plots show the relationship between success probabilities and the convergence rate (6).

In low success probability regime (first and second columns) we observe oscillations, while in high success probability regime (third and forth columns) oscillations are mitigated substantially.

Under review as a conference paper at ICLR 2020 (7) to a neighborhood of the solution.

We fixed gradient noise level by setting mini-batch size 2 and altered the constant step size.

For the first column we set bigger step size γ = 0.25 to detect the divergence (as we slightly violated SPB assumption).

Then for the second and third columns we set γ = 0.1 and γ = 0.05 to expose the convergence to a neighborhood of the minimizer.

For the forth column we set even smaller step size γ = 0.01 to observe a slower convergence.

Here we state the well-known Gauss's inequality on unimodal distributions 3 .

Theorem 4 (Gauss's inequality).

Let X be a unimodal random variable with mode m, and let σ 2 m be the expected value of (X − m)

2 .

Then for any positive value of r,

Applying this inequality on unimodal and symmetric distributions, direct algebraic manipulations give the following bound:

where m = µ and σ 2 m = σ 2 are the mean and variance of unimodal, symmetric random variable X, and r ≥ 0.

Now, using the assumption that eachĝ i (x) has unimodal and symmetric distribution, we apply this bound for

and get a bound for success probabilities

Improvment on Lemma 1 and l 1,2 norm: The bound after Gauss inequality can be improved including a second order term

Hence, continuing the proof of Lemma 1, we get

and we could have defined l 1,2 -norm in a bit more complicated form as

Under review as a conference paper at ICLR 2020 B.2 SUFFICIENT CONDITIONS FOR SPB: PROOF OF LEMMA 2 Letĝ (τ ) be the gradient estimator with mini-batch size τ .

It is known that the variance forĝ (τ ) is dropped by at least a factor of τ , i.e.

Hence, estimating the failure probabilities of signĝ (τ ) when g i = 0, we have

which imples

We will split the derivation into three lemmas providing some intuition on the way.

The first two lemmas establish success probability bounds in terms of mini-batch size.

Essentially, we present two methods: one works well in the case of small randomness, while the other one in the case of non-small randomness.

In the third lemma, we combine those two bounds to get the condition on mini-batch size ensuring SPB assumption.

Lemma 4.

Let X 1 , X 2 , . . .

, X τ be i.i.d.

random variables with non-zero mean µ := EX 1 = 0, finite variance σ 2 := E|X 1 − µ| 2 < ∞. Then for any mini-batch size τ ≥ 1

Proof.

Without loss of generality, we assume µ > 0.

Then, after some adjustments, the proof follows from the Chebyshev's inequality:

where in the last step we used independence of random variables X 1 , X 2 , . . .

, X τ .

Obviously, bound (11) is not optimal for big variance as it becomes a trivial inequality.

In the case of non-small randomness a better bound is achievable additionally assuming the finitness of 3th central moment.

Lemma 5.

Let X 1 , X 2 , . . .

, X τ be i.i.d.

random variables with non-zero mean µ := EX 1 = 0, positive variance σ 2 := E|X 1 − µ| 2 > 0 and finite 3th central moment ν 3 := E|X 1 − µ| 3 < ∞. Then for any mini-batch size τ ≥ 1

where error function erf is defined as

Proof.

Again, without loss of generality, we may assume that µ > 0.

Informally, the proof goes as follows.

As we have an average of i.i.d.

random variables, we approximate it (in the sense of distribution) by normal distribution using the Central Limit Theorem (CLT).

Then we compute success probabilities for normal distribution with the error function erf.

Finally, we take into account the approximation error in CLT, from which the third term with negative sign appears.

More formally, we apply Berry-Esseen inequality 4 on the rate of approximation in CLT (Shevtsova, 2011):

where N ∼ N (0, 1) has the standard normal distribution.

Setting t = −µ √ τ /σ, we get

It remains to compute the second probability using the cumulative distribution function of normal distribuition and express it in terms of the error function:

Clearly, bound (12) is better than (11) when randomness is high.

On the other hand, bound (12) is not optimal for small randomness (σ ≈ 0).

Indeed, one can show that in a small randomness regime, while both variance σ 2 and third moment ν 3 are small, the ration ν/σ might blow up to infinity producing trivial inequality.

For instance, taking X i ∼ Bernoulli(p) and letting p → 1 gives ν/σ = O (1 − p) − 1 /6 .

This behaviour stems from the fact that we are using CLT: less randomness implies slower rate of approximation in CLT.

As a result of these two bounds on success probabilities, we conclude a condition on mini-batch size for the SPB assumption to hold.

Under review as a conference paper at ICLR 2020 Lemma 6.

Let X 1 , X 2 , . . .

, X τ be i.i.d.

random variables with non-zero mean µ = 0 and finite variance σ 2 < ∞. Then

where ν 3 is (possibly infinite) 3th central moment.

Proof.

First, if σ = 0 then the lemma holds trivially.

If ν = ∞, then it follows immediately from Lemma 4.

Assume both σ and ν are positive and finite.

In case of τ > 2σ 2 /µ 2 we apply Lemma 4 again.

Consider the case τ ≤ 2σ 2 /µ 2 , which implies

we get

which together with (12) gives

Hence, SPB assumption holds if

It remains to show that erf(1)

Lemma (3) follows from Lemma (6) applying it to i.i.d.

dataĝ

First, from L-smoothness assumption we have

where g k = g(x k ),ĝ k =ĝ(x k ),ĝ k,i is the i-th component ofĝ k andL is the average value of L i '

s.

Taking conditional expectation given current iteration x k gives

Under review as a conference paper at ICLR 2020

Using the definition of success probabilities ρ i we get

Plugging this into (15) and taking full expectation, we get

Therefore

Now, in case of decreasing step sizes

where we have used the following standard inequalities

In the case of constant step size γ k = γ

Under review as a conference paper at ICLR 2020 B.5 CONVERGENCE ANALYSIS: PROOF OF THEOREM 2

Clearly, the iterations {x k } k≥0 of Algorithm 1 (Option 2) do not increase the function value in any iteration, i.e. E[f (x k+1 )|x k ] ≤ f (x k ).

Continuing the proof of Theorem 1 from (20), we get

where we have used the following inequality

The proof for constant step size is the same as in Theorem 1.

The proof of Theorem 3 goes with the same steps as in Theorem 1, except the derivation (16)-(19) is replaced by

where we have used the following lemma.

Lemma 7.

Assume that for some point x ∈ R d and some coordinate i ∈ {1, 2, . . .

, d}, master node receives M independent stochastic signs signĝ m i (x), m = 1, . . .

, M of true gradient g i (x) = 0.

Let g (M ) (x) be the sum of stochastic signs aggregated from nodes:

where l = [ (M +1) /2] and ρ i > 1 /2 is the success probablity for coordinate i.

@highlight

General analysis of sign-based methods (e.g. signSGD) for non-convex optimization, built on intuitive bounds on success probabilities.
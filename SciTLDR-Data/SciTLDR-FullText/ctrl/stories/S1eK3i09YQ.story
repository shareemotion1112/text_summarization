One of the mysteries in the success of neural networks is randomly initialized first order methods like gradient descent can achieve zero training loss even though the objective function is non-convex and non-smooth.

This paper demystifies this surprising phenomenon for two-layer fully connected ReLU activated neural networks.

For an $m$ hidden node shallow neural network with ReLU activation and $n$ training data, we show as long as $m$ is large enough and no two inputs are parallel, randomly initialized gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function.



Our analysis relies on the following observation: over-parameterization and random initialization jointly restrict every weight vector to be close to its initialization for all iterations, which allows us to exploit a strong convexity-like property to show that gradient descent converges at a global linear rate to the global optimum.

We believe these insights are also useful in analyzing deep models and other first order methods.

Neural networks trained by first order methods have achieved a remarkable impact on many applications, but their theoretical properties are still mysteries.

One of the empirical observation is even though the optimization objective function is non-convex and non-smooth, randomly initialized first order methods like stochastic gradient descent can still find a global minimum.

Surprisingly, this property is not correlated with labels.

In BID37 , authors replaced the true labels with randomly generated labels, but still found randomly initialized first order methods can always achieve zero training loss.

A widely believed explanation on why a neural network can fit all training labels is that the neural network is over-parameterized.

For example, Wide ResNet (Zagoruyko and Komodakis) uses 100x parameters than the number of training data.

Thus there must exist one such neural network of this architecture that can fit all training data.

However, the existence does not imply why the network found by a randomly initialized first order method can fit all the data.

The objective function is neither smooth nor convex, which makes traditional analysis technique from convex optimization not useful in this setting.

To our knowledge, only the convergence to a stationary point is known BID5 .In this paper we demystify this surprising phenomenon on two-layer neural networks with rectified linear unit (ReLU) activation.

Formally, we consider a neural network of the following form.

DISPLAYFORM0 a r σ w r x (1) where x ∈ R d is the input, w r ∈ R d is the weight vector of the first layer, a r ∈ R is the output weight and σ (·) is the ReLU activation function: σ (z) = z if z ≥ 0 and σ (z) = 0 if z < 0 .We focus on the empirical risk minimization problem with a quadratic loss.

Given a training data set {(x i , y i )} n i=1 , we want to minimize DISPLAYFORM1 Our main focus of this paper is to analyze the following procedure.

We fix the second layer and apply gradient descent (GD) to optimize the first layer DISPLAYFORM2 where η > 0 is the step size.

Here the gradient formula for each weight vector is 2 ∂L(W, a) DISPLAYFORM3 (f (W, a, x i ) − y i )a r x i I w r x i ≥ 0 .Though this is only a shallow fully connected neural network, the objective function is still nonsmooth and non-convex due to the use of ReLU activation function.

3 Even for this simple function, why randomly initialized first order method can achieve zero training error is not known.

Many previous works have tried to answer this question or similar ones.

Attempts include landscape analysis BID28 , partial differential equations (Mei et al.) , analysis of the dynamics of the algorithm BID20 , optimal transport theory BID3 , to name a few.

These results often make strong assumptions on the labels and input distributions or do not imply why randomly initialized first order method can achieve zero training loss.

See Section 2 for detailed comparisons between our result and previous ones.

In this paper, we rigorously prove that as long as no two inputs are parallel and m is large enough, with randomly initialized a and W(0), gradient descent achieves zero training loss at a linear convergence rate, i.e., it finds a solution DISPLAYFORM4 Thus, our theoretical result not only shows the global convergence but also gives a quantitative convergence rate in terms of the desired accuracy.

Analysis Technique Overview Our proof relies on the following insights.

First we directly analyze the dynamics of each individual prediction f (W, a, x i ) for i = 1, . . .

, n. This is different from many previous work BID8 BID20 which tried to analyze the dynamics of the parameter (W) we are optimizing.

Note because the objective function is non-smooth and non-convex, analysis of the parameter space dynamics is very difficult.

In contrast, we find the dynamics of prediction space is governed by the spectral property of a Gram matrix (which can vary in each iteration, c.f.

Equation (6)) and as long as this Gram matrix's least eigenvalue is lower bounded, gradient descent enjoys a linear rate.

It is easy to show as long as no two inputs are parallel, in the initialization phase, this Gram matrix has a lower bounded least eigenvalue. (c.f.

Theorem 3.1).

Thus the problem reduces to showing the Gram matrix at later iterations is close to that in the initialization phase.

Our second observation is this Gram matrix is only related to the activation patterns (I w r x i ≥ 0 ) and we can use matrix perturbation analysis to show if most of the patterns do not change, then this Gram matrix is close to its initialization.

Our third observation is we find over-parameterization, random initialization, and the linear convergence jointly restrict every weight vector w r to be close to its initialization.

Then we can use this property to show most of the patterns do not change.

Combining these insights we prove the first global quantitative convergence result of gradient descent on ReLU activated neural networks for the empirical risk minimization problem.

Notably, our proof only uses linear algebra and standard probability bounds so we believe it can be easily generalized to analyze deep neural networks.

Notations We let [n] = {1, 2, . . .

, n}. Given a set S, we use unif {S} to denote the uniform distribution over S. Given an event E, we use I {A} to be the indicator on whether this event happens.

We use N (0, I) to denote the standard Gaussian distribution.

For a matrix A, we use A ij to denote its (i, j)-th entry.

We use · 2 to denote the Euclidean norm of a vector, and use · F to denote the Frobenius norm of a matrix.

If a matrix A is positive semi-definite, we use λ min (A) to denote its smallest eigenvalue.

We use ·, · to denote the standard Euclidean inner product between two vectors.

In this section, we survey an incomplete list of previous attempts in analyzing why first order methods can find a global minimum.

Landscape Analysis A popular way to analyze non-convex optimization problems is to identify whether the optimization landscape has some good geometric properties.

Recently, researchers found if the objective function is smooth and satisfies (1) all local minima are global and (2) for every saddle point, there exists a negative curvature, then the noise-injected (stochastic) gradient descent BID17 BID12 BID7 can find a global minimum in polynomial time.

This algorithmic finding encouraged researchers to study whether the deep neural networks also admit these properties.

For the objective function defined in Equation (2), some partial results were obtained.

BID28 showed if md ≥ n, then at every differentiable local minimum, the training error is zero.

However, since the objective is non-smooth, it is hard to show gradient descent convergences to a differentiable local minimum.

BID33 studied the same problem and related the loss to the gradient norm through the least singular value of the "extended feature matrix" D at the stationary points.

However, they did not prove the convergence rate of the gradient norm.

Interestingly, our analysis relies on the Gram matrix which is DD .Landscape analyses of ReLU activated neural networks for other settings have also been studied in many previous works BID24 BID38 BID11 BID15 BID23 .

These works establish favorable landscape properties but none of them implies that gradient descent converges to a global minimizer of the empirical risk.

More recently, some negative results have also been discovered BID25 BID34 and new procedures have been proposed to test local optimality and escape strict saddle points at non-differentiable points BID35 .

However, the new procedures cannot find global minima as well.

For other activation functions, some previous works showed the landscape does have the desired geometric properties BID6 BID27 BID22 BID18 BID14 BID0 BID31 BID34 .

However, it is unclear how to extend their analyses to our setting.

Another way to prove convergence result is to analyze the dynamics of first order methods directly.

Our paper also belongs to this category.

Many previous works assumed (1) the input distribution is Gaussian and (2) the label is generated according to a planted neural network.

Based on these two (unrealistic) conditions, it can be shown that randomly initialized (stochastic) gradient descent can learn a ReLU BID29 BID26 , a single convolutional filter BID2 , a convolutional neural network with one filter and one output layer BID10 and residual network with small spectral norm weight matrix BID20 .

5 Beyond Gaussian input distribution, BID8 showed for learning a convolutional filter, the Gaussian input distribution assumption can be relaxed but they still required the label is generated from an underlying true filter.

Comparing with these work, our paper does not try to recover the underlying true neural network.

Instead, we focus on providing theoretical justification on why randomly initialized gradient descent can achieve zero training loss, which is what we can observe and verify in practice.

BID16 established an asymptotic result showing for the multilayer fully-connected neural network with a smooth activation function, if every layer's weight matrix is infinitely wide, then for finite training time, the convergence of gradient descent can be characterized by a kernel.

Our proof technique relies on a Gram matrix which is the kernel matrix in their paper.

Our paper focuses on the two-layer neural network with ReLU activation function (non-smooth) and we are able to prove the Gram matrix is stable for infinite training time.

The most related paper is by BID19 who observed that when training a two-layer full connected neural network, most of the patterns (I w r x i ≥ 0 ) do not change over iterations, which we also use to show the stability of the Gram matrix.

They used this observation to obtain the convergence rate of GD on a two-layer over-parameterized neural network for the cross-entropy loss.

They need the number of hidden nodes m scales with poly(1/ ) where is the desired accuracy.

Thus unless the number of hidden nodes m → ∞, their result does not imply GD can achieve zero training loss.

We improve by allowing the amount of over-parameterization to be independent of the desired accuracy and show GD can achieve zero training loss.

Furthermore, our proof is much simpler and more transparent so we believe it can be easily generalized to analyze other neural network architectures.

Other Analysis Approaches BID3 used optimal transport theory to analyze continuous time gradient descent on over-parameterized models.

They required the second layer to be infinitely wide and their results on ReLU activated neural network is only at the formal level.

Mei et al. analyzed SGD for optimizing the population loss and showed the dynamics can be captured by a partial differential equation in the suitable scaling limit.

They listed some specific examples on input distributions including mixture of Gaussians.

However, it is still unclear whether this framework can explain why first order methods can minimize the empirical risk.

BID4 built connection between neural networks with kernel methods and showed stochastic gradient descent can learn a function that is competitive with the best function in the conjugate kernel space of the network.

Again this work does not imply why first order methods can achieve zero training loss.

In this section, we present our result for gradient flow, i.e., gradient descent with infinitesimal step size.

The analysis of gradient flow is a stepping stone towards understanding discrete algorithms and this is the main topic of recent work BID1 BID9 .

In the next section, we will modify the proof and give a quantitative bound for gradient descent with positive step size.

Formally, we consider the ordinary differential equation 6 defined by: DISPLAYFORM0 We denote u i (t) = f (W(t), a, x i ) the prediction on input x i at time t and we let u(t) = (u 1 (t), . . .

, u n (t)) ∈ R n be the prediction vector at time t.

We state our main assumption.

DISPLAYFORM1 H ∞ is the Gram matrix induced by the ReLU activation function and the random initialization.

Later we will show that during the training, though the Gram matrix may change (c.f.

Equation (6)), it is still close to H ∞ .

Furthermore, as will be apparent in the proof (c.f.

Equation FORMULA12 ), H ∞ is the fundamental quantity that determines the convergence rate.

Interestingly, various properties of this H ∞ matrix has been studied in previous works BID33 BID30 .

Now to justify this assumption, the following theorem shows if no two inputs are parallel the least eigenvalue is strictly positive.

Theorem 3.1.

If for any i = j, x i x j , then λ 0 > 0.Note for most real world datasets, no two inputs are parallel, so our assumption holds in general.

Now we are ready to state our main theorem in this section.

Theorem 3.2 (Convergence Rate of Gradient Flow).

Suppose Assumption 3.1 holds and for all i ∈ [n], x i 2 = 1 and |y i | ≤ C for some constant C.

Then if we set the number of hidden nodes DISPLAYFORM2 , then with probability at least 1 − δ over the initialization, we have DISPLAYFORM3 This theorem establishes that if m is large enough, the training error converges to 0 at a linear rate.

Here we assume x i 2 = 1 only for simplicity and it is not hard to relax this condition.

7 The bounded label condition also holds for most real world data set.

The number of hidden nodes m required is Ω n 6 λ 4 0 δ 3 , which depends on the number of samples n, λ 0 , and the failure probability δ.

Over-parameterization, i.e., the fact m = poly(n, 1/λ 0 , 1/δ), plays a crucial role in guaranteeing gradient descent to find the global minimum.

In this paper, we only use the simplest concentration inequalities (Hoeffding's and Markov's) in order to have the cleanest proof.

We believe using a more advanced concentration analysis we can further improve the dependency.

Lastly, we note the specific convergence rate depends on λ 0 but independent of the number of hidden nodes m.

Our first step is to calculate the dynamics of each prediction.

DISPLAYFORM0 where H(t) is an n × n matrix with (i, j)-th entry DISPLAYFORM1 With this H(t) matrix, we can write the dynamics of predictions in a compact way: DISPLAYFORM2 Remark 3.1.

Note Equation (7) completely describes the dynamics of the predictions.

In the rest of this section, we will show (1) at initialization DISPLAYFORM3 Therefore, according to Equation (7), as m → ∞, the dynamics of the predictions are characterized by H ∞ .

This is the main reason we believe H ∞ is the fundamental quantity that describes this optimization process.

H(t) is a time-dependent symmetric matrix.

We first analyze its property when t = 0.

The following lemma shows if m is large then H(0) has a lower bounded least eigenvalue with high probability.

The proof is by the standard concentration bound so we defer it to the appendix.

7 More precisely, if 0 < c low ≤ xi 2 ≤ c high for all i ∈ [n], we only need to change Lemma 3.1-3.3 to make them depend on c low and c high and the amount of over-parameterization m will depend on c high c low .

We assume xi 2 = 1 so we can present the cleanest proof and focus on our main analysis technique.

Our second step is to show H(t) is stable in terms of W(t).

Formally, the following lemma shows for any W close to W(0), the induced Gram matrix H is close to H(0) and has a lower bounded least eigenvalue.

Lemma 3.2.

If w 1 , . . . , w m are i.i.d.

generated from N (0, I), then with probability at least 1 − δ, the following holds.

For any set of weight vectors w 1 , . . . , w m ∈ R d that satisfy for any r ∈ [m], w r (0) − w r 2 ≤ cδλ0 n 2 R for some small positive constant c, then the matrix H ∈ R n×n defined by DISPLAYFORM4 2 .

This lemma plays a crucial role in our analysis so we give the proof below.

Note this event happens if and only if w r (0) x i < R. Recall w r (0) ∼ N (0, I).

By anticoncentration inequality of Gaussian, we have P ( DISPLAYFORM0 .

Therefore, for any set of weight vectors w 1 , . . . , w m that satisfy the assumption in the lemma, we can bound the entry-wise deviation on their induced matrix H: for any DISPLAYFORM1 where the expectation is taken over the random initialization of w 1 (0), . . .

, w m (0).

Summing over DISPLAYFORM2 .

Thus by Markov's inequality, with proba- DISPLAYFORM3 .

Next, we use matrix perturbation theory to bound the deviation from the initialization DISPLAYFORM4 Lastly, we lower bound the smallest eigenvalue by plugging in R DISPLAYFORM5 The next lemma shows two facts if the least eigenvalue of H(t) is lower bounded.

First, the loss converges to 0 at a linear convergence rate.

Second, w r (t) is close to the initialization for every r ∈ [m].

This lemma clearly demonstrates the power of over-parameterization.

DISPLAYFORM6 R .Proof of Lemma 3.3 Recall we can write the dynamics of predictions as d dt u(t) = H(y − u(t)).

We can calculate the loss function dynamics DISPLAYFORM7 Thus we have DISPLAYFORM8 2 is a decreasing function with respect to t. Using this fact we can bound the loss DISPLAYFORM9 Therefore, u(t)

→ y exponentially fast.

Now we bound the gradient norm.

Recall for 0 ≤ s ≤ t, DISPLAYFORM10 Integrating the gradient, we can bound the distance from the initialization DISPLAYFORM11 The next lemma shows if R < R, the conditions in Lemma 3.2 and 3.3 hold for all t ≥ 0.

The proof is by contradiction and we defer it to appendix.

DISPLAYFORM12 Thus it is sufficient to show R < R which is equivalent to m = Ω DISPLAYFORM13 .

We bound DISPLAYFORM14 Thus by Markov's inequality, we have with probability at least 1 − δ, y − u(0) DISPLAYFORM15 .

Plugging in this bound we prove the theorem.

In this subsection, we showcase our proof technique can be applied to analyze the convergence of gradient flow for jointly training both layers.

Formally, we consider the ordinary differential equation defined by: DISPLAYFORM0 for r = 1, . . .

, m. The following theorem shows using gradient flow to jointly train both layers, we can still enjoy linear convergence rate towards zero loss.

, with probability at least 1 − δ over the initialization we have DISPLAYFORM1 Theorem 3.3 shows under the same assumptions as in Theorem 3.2, we can achieve the same convergence rate as that of only training the first layer.

The proof of Theorem 3.3 relies on the same arguments as the proof of Theorem 3.2.

Again we consider the dynamics of the predictions and this dynamics is characterized by a Gram matrix.

We can show for all t > 0, this Gram matrix is close to the Gram matrix at the initialization phase.

We refer readers to appendix for the full proof.

In this section, we show randomly initialized gradient descent with a constant positive step size converges to the global minimum at a linear rate.

We first present our main theorem. , and we set the step size η = O λ0 n 2 then with probability at least 1 − δ over the random initialization we have for k = 0, 1, 2, . . .

DISPLAYFORM0 Theorem 4.1 shows even though the objective function is non-smooth and non-convex, gradient descent with a constant step size still enjoys a linear convergence rate.

Our assumptions on the least eigenvalue and the number of hidden nodes are exactly the same as the theorem for gradient flow.

We prove Theorem 4.1 by induction.

Our induction hypothesis is just the following convergence rate of the empirical loss.

Condition 4.1.

At the k-th iteration, we have y − u(k) DISPLAYFORM0 A directly corollary of this condition is the following bound of deviation from the initialization.

The proof is similar to that of Lemma 3.3 so we defer it to appendix.

DISPLAYFORM1 Now we show Condition 4.1 holds for every k = 0, 1, . .

..

For the base case k = 0, by definition Condition 4.1 holds.

Suppose for k = 0, . . .

, k, Condition 4.1 holds and we want to show Condition 4.1 holds for k = k + 1.Our strategy is similar to the proof of Theorem 3.2.

We define the event DISPLAYFORM2 where R = Lemma 4.1.

With probability at least 1 − δ over the initialization, we have DISPLAYFORM3 for some positive constant C > 0.Next, we calculate the difference of predictions between two consecutive iterations, analogue to dui(t) dt term in Section 3.

DISPLAYFORM4 Here we divide the right hand side into two parts.

I i 1 accounts for terms that the pattern does not change and I i 2 accounts for terms that pattern may change.

DISPLAYFORM5 We view I i 2 as a perturbation and bound its magnitude.

Because ReLU is a 1-Lipschitz function and |a r | = 1, we have DISPLAYFORM6 To analyze I i 1 , by Corollary 4.1, we know w r (k) − w r (0) ≤ R and w r (k) − w r (0) ≤ R for all r ∈ [m].

Furthermore, because R < R, we know I w r (k + 1) x i ≥ 0 = I w r (k) x i ≥ 0 for r ∈ S i .

Thus we can find a more convenient expression of I i 1 for analysis DISPLAYFORM7 where DISPLAYFORM8

n × n matrix with (i, j)-th entry being H ⊥ ij (k).

Using Lemma 4.1, we obtain an upper bound of the operator norm DISPLAYFORM0 Similar to the classical analysis of gradient descent, we also need bound the quadratic term.

DISPLAYFORM1 With these estimates at hand, we are ready to prove the induction hypothesis.

DISPLAYFORM2 The third equality we used the decomposition of u(k + 1) − u(k).

The first inequality we used the Lemma 3.2, the bound on the step size, the bound on I 2 , the bound on H(k) DISPLAYFORM3 and the bound on u(k + 1) − u(k) 2 2 .

The last inequality we used the bound of the step size and the bound of R. Therefore Condition 4.1 holds for k = k + 1.

Now by induction, we prove Theorem 4.1.

In this section, we use synthetic data to corroborate our theoretical findings.

We use the initialization and training procedure described in Section 1.

For all experiments, we run 100 epochs of gradient descent and use a fixed step size.

We uniformly generate n = 1000 data points from a d = 1000 dimensional unit sphere and generate labels from a one-dimensional standard Gaussian distribution.

We test three metrics with different widths (m).

First, we test how the amount of overparameterization affects the convergence rates.

Second, we test the relation between the amount of over-parameterization and the number of pattern changes.

Formally, at a given iteration k, we check DISPLAYFORM0 (there are mn patterns).

This aims to verify Lemma 3.2.

Last, we test the relation between the amount of over-parameterization and the maximum of the distances between weight vectors and their initializations.

Formally, at a given iteration k, we check max r∈[m] w r (k) − w r (0) 2 .

This aims to verify Lemma 3.3 and Corollary 4.1.

FIG3 shows as m becomes larger, we have better convergence rate.

We believe the reason is as m becomes larger, H(t) matrix becomes more stable, and thus has larger least eigenvalue.

FIG3 and FIG3 show as m becomes larger, the percentiles of pattern changes and the maximum distance from the initialization become smaller.

These empirical findings are consistent with our theoretical results.

In this paper we show with over-parameterization, gradient descent provable converges to the global minimum of the empirical loss at a linear convergence rate.

The key proof idea is to show the over-parameterization makes Gram matrix remain positive definite for all iterations, which in turn guarantees the linear convergence.

Here we list some future directions.

First, we believe our approach can be generalized to deep neural networks.

We elaborate the main idea here for gradient flow.

Consider a deep neural network of the form DISPLAYFORM0 where x ∈ R d is the input, W (1) ∈ R m×d is the first layer, W (h) ∈ R m×m for h = 2, . . .

, H are the middle layers and a ∈ R m is the output layer.

Recall u i is the i-th prediction.

If we use the quadratic loss, we can compute DISPLAYFORM1 Similar to Equation (5), we can calculate DISPLAYFORM2 where DISPLAYFORM3 .

Therefore, similar to Equation FORMULA12 , we can write du(t) dt =

Proof of Theorem 3.1.

The proof of this lemma just relies on standard real and functional analysis.

Let H be the Hilbert space of integrable d-dimensional vector fields on DISPLAYFORM0 The inner product of this space is then f, g H = E w∼N (0,I) f (w) g(w) .ReLU activation induces an infinite-dimensional feature map φ which is defined as for any x ∈ R d , (φ(x))(w) = xI w x ≥ 0 where w can be viewed as the index.

Now to prove H ∞ is strictly positive definite, it is equivalent to show φ(x 1 ), . . .

, φ(x n ) ∈ H are linearly independent.

Suppose that there are α 1 , · · · , α n ∈ R such that DISPLAYFORM1 This means that α 1 φ(x 1 )(w) + · · · + α n φ(x n )(w) = 0 a.e.

Now we prove α i = 0 for all i.

We define D i = w ∈ R d : w x i = 0 .

This is set of discontinuities of φ(x i ).

The following lemma characterizes the basic property of these discontinuity sets.

DISPLAYFORM2 For j = i, φ(x j )(w) is continuous in a neighborhood of z, then for any > 0 there is a small enough r > 0 such that ∀w ∈ B(z, r), |φ(x j )(w) − φ(x j )(z)| < .Let µ be the Lebesgue measure on R d .

We have DISPLAYFORM3 Thus, we have DISPLAYFORM4 Therefore, as r → 0+, by continuity, we have DISPLAYFORM5 Next recall that (φ(x))(w) = xI x w > 0 , so for w ∈ B + r and x i , (φ(x i ))(w) = x i .

Then, we have DISPLAYFORM6 For w ∈ B − r and x i , we know (φ(x i ))(w) = 0.

Then we have DISPLAYFORM7 Now recall i α i φ(x i ) ≡ 0.

Using Equation FORMULA59 , FORMULA60 and FORMULA61 , we have DISPLAYFORM8 Since x i = 0, we must have α i = 0.

We complete the proof.

Proof of Lemma A.1.

Let µ be the canonical Lebesgue measure on DISPLAYFORM9 This implies our desired result.

Proof of Lemma 3.1.

For every fixed (i, j) pair, H ij (0) is an average of independent random variables.

Therefore, by Hoeffding inequality, we have with probability 1 − δ , DISPLAYFORM10 Setting δ = n 2 δ and applying union bound over (i, j) pairs, we have for every (i, j) pair with probability at least 1 − δ DISPLAYFORM11 Thus we have DISPLAYFORM12 we have the desired result.

Proof of Lemma 3.4.

Suppose the conclusion does not hold at time t. If there exists r ∈ [m], DISPLAYFORM13 , then by Lemma 3.3 we know there exists s ≤ t such that λ min (H(s)) < 1 2 λ 0 .

By Lemma 3.2 we know there exists DISPLAYFORM14 Thus at t 0 , there exists r ∈ [m], w r (t 0 ) − w r (0) 2 2 = R. Now by Lemma 3.2, we know H(t 0 ) ≥ 1 2 λ 0 for t ≤ t 0 .

However, by Lemma 3.3, we know w r (t 0 ) − w r (0) 2 < R < R. Contradiction.

For the other case, at time t, λ min (H(t)) < 1 2 λ 0 we know there exists DISPLAYFORM15 The rest of the proof is the same as the previous case.

A.1 PROOF OF THEOREM 3.3 In this section we show using gradient flow to jointly train both the first layer and the output layer we can still achieve 0 training loss.

We follow the same approach we used in Section 3.

Recall the gradient for a. DISPLAYFORM16 We compute the dynamics of an individual prediction.

DISPLAYFORM17 Recall we have found a convenient expression for the first term.

DISPLAYFORM18 where DISPLAYFORM19 For the second term, it easy to derive DISPLAYFORM20 where DISPLAYFORM21 Therefore we have du(t) dt = (H(t) + G(t)) (y − u(t)) .First use the same concentration arguments as in Lemma 3.1, we can show λ min (H(0)) ≥ 3λ0 4with 1 − δ probability over the initialization.

In the following, our arguments will base on that λ min (H(0)) ≥ Proof of Lemma A.2.

We can calculate the loss function dynamics d dt y − u(t) 2 2 = − 2 (y − u(t)) (H(t) + G(t)) (y − u(t)) ≤ − 2 (y − u(t)) (H(t)) (y − u(t)) DISPLAYFORM22 where in the first inequality we use the fact that G(t) is Gram matrix thus it is positive.

Lemma A.6.

If R w < R w and R a < R a , we have for all t ≥ 0, λ min (H(t)) ≥ 1 2 λ 0 , for all r ∈ [m], w r (t) − w r (0) 2 ≤ R w , |a r (t) − a r (0)| ≤ R a and y − u(t) Proof of Lemma A.6.

We prove by contradiction.

Let t > 0 be the smallest time that the conclusion does not hold.

Then either λ min (H(t)) < 2 .

However, since R w < R w and R a < R a .

This contradicts with the minimality of t. The last case is similar for which we can simply apply Lemma A.5.Based on Lemma A.2, we only need to ensure R w < R w and R a < R a .

By the proof in Section 3, we know with probability at least δ, y − u(0) 2 ≤ C √ n δ for some large constant C. Note our section on m in Theorem 3.3 suffices to ensure R w < R w and R a < R a .

We now complete the proof.

Proof of Corollary 4.1.

We use the norm of gradient to bound this distance.

DISPLAYFORM0

<|TLDR|>

@highlight

We prove gradient descent achieves zero training loss with a linear rate on over-parameterized neural networks.

@highlight

This work considers optimizing a two-layer over-parameterized ReLU network with the squared loss and given a data set with arbituary labels.

@highlight

This paper studies one hidden layer neural networks with square loss, where they show that in over-parameterized setting, random initialization and gradient descent gets to zero loss.
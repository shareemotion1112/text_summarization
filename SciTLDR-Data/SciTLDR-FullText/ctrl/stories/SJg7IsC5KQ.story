Despite its empirical success, the theoretical underpinnings of the stability, convergence and acceleration properties of batch normalization (BN) remain elusive.

In this paper, we attack this problem from a modelling approach, where we perform thorough theoretical analysis on BN applied to simplified model: ordinary least squares (OLS).

We discover that gradient descent on OLS with BN has interesting properties, including a scaling law, convergence for arbitrary learning rates for the weights, asymptotic acceleration effects, as well as insensitivity to choice of learning rates.

We then demonstrate numerically that these findings are not specific to the OLS problem and hold qualitatively for more complex supervised learning problems.

This points to a new direction towards uncovering the mathematical principles that underlies batch normalization.

Batch normalization BID7 (BN) is one of the most important techniques for training deep neural networks and has proven extremely effective in avoiding gradient blowups during back-propagation and speeding up convergence.

In its original introduction BID7 , the desirable effects of BN are attributed to the so-called "reduction of covariate shift".

However, it is unclear what this statement means in precise mathematical terms.

To date, there lacks a comprehensive theoretical analysis of the effect of batch normalization.

In this paper, we study the convergence and stability of gradient descent with batch normalization (BNGD) via a modeling approach.

More concretely, we consider a simplified supervised learning problem: ordinary least squares regression, and analyze precisely the effect of BNGD when applied to this problem.

Much akin to the mathematical modeling of physical processes, the least-squares problem serves as an idealized "model" of the effect of BN for general supervised learning tasks.

A key reason for this choice is that the dynamics of GD without BN (hereafter called GD for simplicity) in least-squares regression is completely understood, thus allowing us to isolate and contrast the additional effects of batch normalization.

The modeling approach proceeds in the following steps.

First, we derive precise mathematical results on the convergence and stability of BNGD applied to the least-squares problem.

In particular, we show that BNGD converges for any constant learning rate ε ∈ (0, 1], regardless of the conditioning of the regression problem.

This is in stark contrast with GD, where the condition number of the problem adversely affect stability and convergence.

Many insights can be distilled from the analysis of the OLS model.

For instance, we may attribute the stability of BNGD to an interesting scaling law governing ε and the initial condition; This scaling law is not present in GD.

The preceding analysis also implies that if we are allowed to use different learning rates for the BN rescaling variables (ε a ) and the remaining trainable variables (ε), we may conclude that BNGD on our model converges for any ε > 0 as long as ε a ∈ (0, 1].

Furthermore, we discover an acceleration effect of BNGD and moreover, there exist regions of ε such that the performance of BNGD is insensitive to changes in ε, which help to explain the robustness of BNGD to the choice of learning rates.

We reiterate that contrary to many previous works, all the preceding statements are precise mathematical results that we derive for our simplified model.

The last step in our modeling approach is also the most important: we need to demonstrate that these insights are not specific features of our idealized model.

Indeed, they should be true characteristics, at least in an approximate sense, of BNGD for general supervised learning problems.

We do this by numerically investigating the convergence, stability and scaling behaviors of BNGD on various datasets and model architectures.

We find that the key insights derived from our idealized analysis indeed correspond to practical scenarios.

Batch normalization was originally introduced in BID7 and subsequently studied in further detail in BID6 .

Since its introduction, it has become an important practical tool to improve stability and efficiency of training deep neural networks BID5 BID2 .

Initial heuristic arguments attribute the desirable features of BN to concepts such as "covariate shift", which lacks a concrete mathematical interpretation and alternative explanations have been given BID17 .

Recent theoretical studies of BN includes BID13 , where the authors proposed a variant of BN, the diminishing batch normalization (DBN) algorithm and analyzed the convergence of the DBN algorithm, showing that it converges to a stationary point of the loss function.

More recently, BID1 demonstrated that the higher learning rates of batch normalization induce a regularizing effect.

Another related work is BID8 , where the authors also considered the convergence properties of BNGD on linear networks (similar to the least-squares problem), as well as other special problems, such as learning halfspaces and extensions.

In the OLS case, the authors showed that for a particularly adaptive choice of dynamic learning rate schedule, which can be seen as a fixed effective step size in our terminology (see equation (11) and the discussion that immediately follows), BNGD converges linearly if λ max is known.

Moreover, the analysis also requires setting the rescaling parameter a every step to satisfy a stationarity condition, instead of simply performing gradient descent on a, as is done in the original BNGD.The present research differs from these previous analysis in an important way -we study the BNGD algorithm itself, and not a special variant.

More specifically, we consider constant learning rates (without knowledge of properties of the OLS loss function) and we perform gradient descent on rescaling parameters.

We prove that the convergence occurs for even in this case (and in fact, for arbitrarily large learning rates for ε, as long as 0 < ε a ≤ 1).

This poses more challenges in the analysis and contrasts our work with previous analysis on modified versions of BNGD.

This is an important distinction; While a decaying or dynamic learning rate is sometimes used in practice, in the case of BN it is critical to analyze the non-asymptotic, constant learning rate case, precisely because one of the key practical advantages of BN is that a bigger learning rate can be used than that in GD.

Hence, it is desirable, as in the results presented in this work, to perform our analysis in this regime.

Finally, through the lens of the least-squares example, BN can be viewed as a type of overparameterization, where additional parameters, which do not increase model expressivity, are introduced to improve algorithm convergence and stability.

In this sense, this is related in effect to the recent analysis of the implicit acceleration effects of over-parameterization on gradient descent BID0 ).

Our paper is organized as follows.

In Section 2, we outline the ordinary least squares (OLS) problem and present GD and BNGD as alternative means to solve this problem.

In Section 3, we demonstrate and analyze the convergence of the BNGD for the OLS model, and in particular contrast the results with the behavior of GD, which is completely known for this model.

We also discuss the important insights to BNGD that these results provide us with.

We then validate these findings on more general supervised learning problems in Section 4.

Finally, we conclude in Section 5.

Consider the simple linear regression model where x ∈ R d is a random input column vector and y is the corresponding output variable.

Since batch normalization is applied for each feature separately, in order to gain key insights it is sufficient to the case of y ∈ R. A noisy linear relationship is assumed between the dependent variable y and the independent variables x, i.e. y = x T w + noise where w ∈ R d is the parameters.

Denote the following moments: DISPLAYFORM0 (1) To simplify the analysis, we assume the covariance matrix H of x is positive definite and the mean E[x] of x is zero.

The eigenvalues of H are denoted as λ i (H), i = 1, 2, ...d,.

Particularly, the maximum and minimum eigenvalue of H is denoted by λ max and λ min respectively.

The condition number of H is defined as κ := λmax λmin .

Note that the positive definiteness of H allows us to define the vector norms .

H and .

DISPLAYFORM1

The ordinary least squares (OLS) method for estimating the unknown parameters w leads to the following optimization problem min DISPLAYFORM0 The gradient of J 0 with respect to w is ∇ w J 0 (w) = Hw − g, and the unique minimizer is w = u := H −1 g. The gradient descent (GD) method (with step size or learning rate ε) for solving the optimization problem (2) is given by the iterating sequence, DISPLAYFORM1 which converges if 0 < ε < 2 λmax =: ε max , and the convergence rate is determined by the spectral radius DISPLAYFORM2 It is well known (for example see Chapter 4 of BID16 ) that the optimal learning rate is ε opt = 2 λmax+λmin , where the convergence estimate is related to the condition number κ(H): DISPLAYFORM3

Batch normalization is a feature-wise normalization procedure typically applied to the output, which in this case is simply z = x T w. The normalization transform is defined as follows: DISPLAYFORM0 where σ := √ w T Hw.

After this rescaling, N BN (z) will be order 1, and hence in order to reintroduce the scale BID7 , we multiply N BN (z) with a rescaling parameter a (Note that the shift parameter can be set zero since E[w T x|w] = 0).

Hence, we get the BN version of the OLS problem (2): DISPLAYFORM1 The objective function J(a, w) is no longer convex.

In fact, it has trivial critical points, {(a * , w * )|a * = 0, w * T g = 0}, which are saddle points of J(a, w).We are interested in the nontrivial critical points which satisfy the relations, DISPLAYFORM2 It is easy to check that the nontrivial critical points are global minimizers, and the Hessian matrix at each critical point is degenerate.

Nevertheless, the saddle points are strict (Details can be found in Appendix), which typically simplifies the analysis of gradient descent on non-convex objectives BID12 BID14 .Consider the gradient descent method to solve the problem (7), which we hereafter call batch normalization gradient descent (BNGD).

We set the learning rates for a and w to be ε a and ε respectively.

These may be different, for reasons which will become clear in the subsequent analysis.

We thus have the following discrete-time dynamical system DISPLAYFORM3 DISPLAYFORM4 We now begin a concrete mathematical analysis of the above iteration sequence.

In this section, we discuss several mathematical results one can derive concretely for BNGD on the OLS problem (7).

First, we establish a simple but useful scaling property, which an important ingredient in allowing us to prove a linear convergence result for arbitrary constant learning rates.

We also derive the asymptotic properties of the "effective" learning rate of BNGD (to be precisely defined subsequently), which shows some interesting sensitivity behavior of BNGD on the chosen learning rates.

Detailed proofs of all results presented here can be found in the Appendix.

In this section, we discuss a straightforward, but useful scaling property that the BNGD iterations possess.

Note that the dynamical properties of the BNGD iteration are governed by a set of numbers, or a configuration {H, u, a 0 , w 0 , ε a , ε}.Definition 3.1 (Equivalent configuration).

Two configurations, {H, u, a 0 , w 0 , ε a , ε} and {H , u , a 0 , w 0 , ε a , ε }, are said to be equivalent if for iterates {w k }, {w k } following these configurations respectively, there is an invertible linear transformation T and a nonzero constant t such that DISPLAYFORM0 The scaling property ensures that equivalent configurations must converge or diverge together, with the same rate up to a constant multiple.

Now, it is easy to check the system has the following scaling law.

Proposition 3.2 (Scaling property).

DISPLAYFORM1 (1) The configurations {µQ T HQ, γ √ µ Qu, γa 0 , γQw 0 , ε a , ε} and {H, u, a 0 , w 0 , ε a , ε} are equivalent.(2) The configurations {H, u, a 0 , w 0 , ε a , ε} and {H, u, a 0 , rw 0 , ε a , r 2 ε} are equivalent.

It is worth noting that the scaling property (2) in Proposition 3.2 originates from the batchnormalization procedure and is independent of the specific structure of the loss function.

Hence, it is valid for general problems where BN is used (Lemma A.9).

Despite being a simple result, the scaling property is important in determining the dynamics of BNGD, and is useful in our subsequent analysis of its convergence and stability properties (see the sketch of the proof of Theorem 3.3).

We have the following convergence result.

Theorem 3.3 (Convergence for BNGD).

The iteration sequence (a k , w k ) in equation FORMULA9 -(10) converges to a stationary point for any initial value (a 0 , w 0 ) and any ε > 0, as long as ε a ∈ (0, 1].

Particularly, we have the following sufficient conditions of converging to global minimizers.( DISPLAYFORM0 , ε a ∈ (0, 1] and ε is sufficiently small (the smallness is quantified by Lemma A.13), then (a k , w k ) converges to a global minimizer.(2) If ε a = 1 and ε > 0, then (a k , w k ) converges to global minimizers for almost all initial values (a 0 , w 0 ).

We first prove that the algorithm converges for any ε a ∈ (0, 1] and small enough ε, with any initial value (a 0 , w 0 ) such that w 0 ≥ 1 (Lemma A.13).

Next, we observe that the sequence { w k } is monotone increasing, and thus either converges to a finite limit or diverges.

The scaling property is then used to exclude the divergent case if { w k } diverges, then at some k the norm w k should be large enough, and by the scaling property, it is equivalent to a case where w k =1 and ε is small, which we have proved converges.

This shows that w k converges to a finite limit, from which the convergence of w k and the loss function value can be established, after some work.

The proof is fully presented in Theorem A.17 and preceding Lemmas.

In addition, using the 'strict saddle point' arguments in BID12 BID14 , we can prove the set of initial value for which (a k , w k ) converges to saddle points has Lebesgue measure 0, provided some conditions, such as when ε a = 1, ε > 0 (Lemma A.20).

It is important to note that BNGD converges for all step size ε > 0 of w k , independent of the spectral properties of H. This is a significant advantage and is in stark contrast with GD, where the step size is limited by λ max (H), and the condition number of H intimately controls the stability and convergence rate.

Although we only prove the almost sure convergence to global minimizer for the case of ε a = 1, we have not encountered convergence to saddles in the OLS experiments even for ε a ∈ (0, 2) with initial values (a 0 , w 0 ) drawn from typical distributions.

Now, let us consider the convergence rate of BNGD when it converges to a minimizer.

Compared with GD, the update coefficient before Hw k in equation FORMULA10 changed from ε to a complicated term which we named as the effective step size or learning rateε k DISPLAYFORM0 and the recurrence relation in place of u − w k is DISPLAYFORM1 Consider the dynamics of the residual DISPLAYFORM2 k )w k , which equals 0 if and only if w k is a global minimizer.

Using the property of H-norm (see section A.1), we observe that the effective learning rateε k determines the convergence rate of e k via DISPLAYFORM3 where ρ(I −ε k H) is spectral radius of the matrix I −ε k H. The inequality FORMULA3 shows that the convergence of e k (and hence the loss function, see Lemma A.23) is linear providedε k ∈ (δ, 2/λ max − δ) for some positive number δ.

In fact, if we enforceε k = 1/λ max for each k, which is done in the analysis in BID8 , then one immediately obtains the same linear convergence rate.

But this requires knowledge of λ max (problem-dependent) and a modification the BNGD algorithm.

We instead focus our analysis on the original BNGD algorithm.

Next, let us discuss below an acceleration effect of BNGD over GD.

When (a k , w k ) is close to a minimizer, we can approximate the iteration (9)-(10) by a linearized system.

The Hessian matrix for BNGD at a minimizer (a * , w * ) is diag(1, H * / w * 2 ), where the matrix H * is DISPLAYFORM4 The matrix H * is positive semi-definite (H * u = 0) and has better spectral properties than H, such as a lower pseudo-condition number κ * = λ * max λ * min ≤ κ, where λ * max and λ * min are the maximal and minimal nonzero eigenvalues of H * respectively.

Particularly, κ * < κ for almost all u (see section A.1 ).

This property leads to acceleration effects of BNGD: When e k H is small, the contraction coefficient ρ in (13) can be improved to a lower coefficient.

More precisely, we have the following result: Proposition 3.4.

For any positive number δ ∈ (0, 1), if (a k , w k ) is close to a minimizer, such that DISPLAYFORM5 where ρ DISPLAYFORM6 Generally, we have ρ DISPLAYFORM7 , and the optimal rate is ρ * opt := DISPLAYFORM8 where the inequality is strict for almost all u. Hence, the estimate (15) indicates that the optimal BNGD could have a faster convergence rate than the optimal GD, especially when κ * is much smaller than κ.

Finally, we discuss the dependence of the effective learning rateε k (and by extension, the effective convergence rate FORMULA3 or FORMULA5 ) on ε.

This is in essence a sensitivity analysis on the performance of BNGD with respect to the choice of learning rate.

The explicit dependence ofε k on ε is quite complex, but we can nevertheless give the following asymptotic estimates.

DISPLAYFORM9 (1) When ε is small enough, ε 1, the effective step size has a same order with ε, i.e. there are two positive constants, C 1 , C 2 , independent on ε and k, such that C 1 ≤ε k /ε ≤ C 2 .(2) When ε is large enough, ε 1, the effective step size has order O(ε −1 ), i.e. there are two positive constants, C 1 , C 2 , independent on ε and k, such that C 1 ≤ε k ε ≤ C 2 .Observe that for finite k,ε k is a differentiable function of ε.

Therefore, the above result implies, via the mean value theorem, the existence of some ε 0 > 0 such that dε k /dε|

ε=ε0 = 0.

Consequently, there is at least some small interval of the choice of learning rates ε where the performance of BNGD is insensitive to this choice.

In fact, empirically this is one commonly observed advantage of BNGD over GD, where the former typically allows for a variety of (large) learning rates to be used without adversely affecting performance.

The same is not true for GD, where the convergence rate depends sensitively on the choice of learning rate.

We will see later in Section 4 that although we only have a local insensitivity result above, the interval of this insensitivity is actually quite large in practice.

Let us first summarize our key findings and insights from the analysis of BNGD on the OLS problem.1.

A scaling law governs BNGD, where certain configurations can be deemed equivalent 2.

BNGD converges for any learning rate ε > 0, provided that ε a ∈ (0, 1].

In particular, different learning rates can be used for the BN variables (a) compared with the remaining trainable variables (w) 3.

There exists intervals of ε for which the performance of BNGD is not sensitive to the choice of εIn the subsequent sections, we first validate numerically these claims on the OLS model, and then show that these insights go beyond the simple OLS model we considered in the theoretical framework.

In fact, much of the uncovered properties are observed in general applications of BNGD in deep learning.

Here we test the convergence and stability of BNGD for the OLS model.

Consider a diagonal matrix H = diag(h) where h = (1, ..., κ) is a increasing sequence.

The scaling property (Proposition 3.2) allows us to set the initial value w 0 having same 2-norm with u, w 0 = u = 1.

Of course, one can verify that the scaling property holds strictly in this case.

FIG0 gives examples of H with different condition numbers κ.

We tested the loss function of BNGD, compared with the optimal GD (i.e. GD with the optimal step size ε opt ), in a large range of step sizes ε a and ε, and with different initial values of a 0 .

Another quantity we observe is the effective step sizeε k of BN.

The results are encoded by four different colors: whetherε k is close to the optimal step size ε opt , and whether loss of BNGD is less than the optimal GD.

The results indicate that the optimal convergence rate of BNGD can be better than GD in some configurations.

This acceleration phenomenon is ascribed to the pseudo-condition number of H * (discard the only zero eigenvalue) being less than κ(H).

This advantage of BNGD is significant when the (pseudo)-condition number discrepancy between H and H * is large.

However, if this difference is small, the acceleration is imperceptible.

This is consistent with our analysis in section 3.3.Another important observation is a region such thatε is close to ε opt , in other words, BNGD significantly extends the range of 'optimal' step sizes.

Consequently, we can choose step sizes in BNGD at greater liberty to obtain almost the same or better convergence rate than the optimal GD.

However, the size of this region is inversely dependent on the initial condition a 0 .

Hence, this suggests that small a 0 at first steps may improve robustness.

On the other hand, small ε a will weaken the performance of BN.

The phenomenon suggests that improper initialization of the BN parameters weakens the power of BN.

This experience is encountered in practice, such as BID3 , where higher initial values of BN parameter are detrimental to the optimization of RNN models.

whetherε k is close to the optimal step size ε opt of GD, characterized by the inequality 0.8ε opt <

ε k < ε opt /0.8, and whether loss of BNGD is less than the optimal GD.

Parameters: H = diag(logspace(0,log10(κ),100)), u is randomly chosen uniformly from the unit sphere in R 100 , w 0 is set to Hu/ Hu .

The GD and BNGD iterations are executed for k = 2000 steps with the same w 0 .

In each image, the range of ε a (x-axis) is 1.99 * logspace (-10,0,41) , and the range of ε (y-axis) is logspace (-5,16,43) .

We conduct experiments on deep learning applied to standard classification datasets: MNIST (LeCun et al., 1998), Fashion MNIST BID19 and CIFAR-10 (Krizhevsky & Hinton, 2009 ).

The goal is to explore if the key findings outlined at the beginning of this section continue to hold for more general settings.

For the MNIST and Fashion MNIST dataset, we use two different networks:(1) a one-layer fully connected network (784 × 10) with softmax mean-square loss; (2) a fourlayer convolution network (Conv-MaxPool-Conv-MaxPool-FC-FC) with ReLU activation function and cross-entropy loss.

For the CIFAR-10 dataset, we use a five-layer convolution network (ConvMaxPool-Conv-MaxPool-FC-FC-FC).

All the trainable parameters are randomly initialized by the Glorot scheme BID4 before training.

For all three datasets, we use a minibatch size of 100 for computing stochastic gradients.

In the BNGD experiments, batch normalization is performed on all layers, the BN parameters are initialized to transform the input to zero mean/unit variance distributions, and a small regularization parameter =1e-3 is added to variance √ σ 2 + to avoid division by zero.

Scaling property Theoretically, the scaling property 3.2 holds for any layer using BN.

However, it may be slightly biased by the regularization parameter .

Here, we test the scaling property in practical settings.

Figure 2 gives the loss of network-(2) (2CNN+2FC) at epoch=1 with different learning rate.

The norm of all weights and biases are rescaled by a common factor η.

We observe that the scaling property remains true for relatively large η.

However, when η is small, the norm of weights are small.

Therefore, the effect of the -regularization in √ σ 2 + becomes significant, causing the curves to be shifted.

Stability for large learning rates We use the loss value at the end of the first epoch to characterize the performance of BNGD and GD methods.

Although the training of models have generally not converged at this point, it is enough to extract some relative rate information.

FIG5 shows the loss value of the networks on the three datasets.

It is observed that GD and BNGD with identical learning rates for weights and BN parameters exhibit a maximum allowed learning rate, beyond which the iterations becomes unstable.

On the other hand, BNGD with separate learning rates exhibits a much larger range of stability over learning rate for non-BN parameters, consistent with our theoretical results in Theorem 3.3.Insensitivity of performance to learning rates Observe that BN accelerates convergence more significantly for deep networks, whereas for one-layer networks, the best performance of BNGD and Figure 2 : Tests of scaling property of the 2CNN+2FC network on MNIST dataset.

BN is performed on all layers, and =1e-3 is added to variance √ σ 2 + .

All the trainable parameters (except the BN parameters) are randomly initialized by the Glorot scheme, and then multiplied by a same parameter η.

GD are similar.

Furthermore, in most cases, the range of optimal learning rates in BNGD is quite large, which is in agreement with the OLS analysis (Proposition 3.5).

This phenomenon is potentially crucial for understanding the acceleration of BNGD in deep neural networks.

Heuristically, the "optimal" learning rates of GD in distinct layers (depending on some effective notion of "condition number") may be vastly different.

Hence, GD with a shared learning rate across all layers may not achieve the best convergence rates for all layers at the same time.

In this case, it is plausible that the acceleration of BNGD is a result of the decreased sensitivity of its convergence rate on the learning rate parameter over a large range of its choice.

Figure 3: Performance of BNGD and GD method on MNIST (network-(1), 1FC), Fashion MNIST (network-(2), 2CNN+2FC) and CIFAR-10 (2CNN+3FC) datasets.

The performance is characterized by the loss value at ephoch=1.

In the BNGD method, both the shared learning rate schemes and separated learning rate scheme (learning rate lr a for BN parameters) are given.

The values are averaged over 5 independent runs.

In this paper, we adopted a modeling approach to investigate the dynamical properties of batch normalization.

The OLS problem is chosen as a point of reference, because of its simplicity and the availability of convergence results for gradient descent.

Even in such a simple setting, we saw that BNGD exhibits interesting non-trivial behavior, including scaling laws, robust convergence properties, acceleration, as well as the insensitivity of performance to the choice of learning rates.

Although these results are derived only for the OLS model, we show via experiments that these are qualitatively valid for general scenarios encountered in deep learning, and points to a concrete way in uncovering the reasons behind the effectiveness of batch normalization.

Interesting future directions include the extension of the results for the OLS model to more general settings of BNGD, where we believe the scaling law (Proposition 3.2) should play a significant role.

In addition, we have not touched upon another empirically observed advantage of batch normalization, which is better generalization errors.

It will be interesting to see how far the current approach takes us in investigating such probabilistic aspects of BNGD.

The objective function in problem (7) has an equivalent form: DISPLAYFORM0 where u = H −1 g.

The gradients are: DISPLAYFORM1 DISPLAYFORM2 The Hessian matrix is DISPLAYFORM3 where DISPLAYFORM4 DISPLAYFORM5 The objective function J(a, w) has trivial critical points, {(a * , w * )|a DISPLAYFORM6 It is obvious that a * is the minimizer of J(a, w * ), but (a * , w * ) is not a local minimizer of J(a, w) unless g = 0, hence (a * , w * ) are saddle points of J(a, w).

The Hessian matrix at those saddle points has at least a negative eigenvalue, i.e. the saddle points are strict.

In fact, the eigenvalues at the saddle point (a * , w * ) are On the other hand, the nontrivial critical points satisfies the relations, DISPLAYFORM7 where the sign of a * depends on the direction of u, w * , i.e. sign(a * ) = sign(u T w * ).

It is easy to check that the nontrivial critical points are global minimizers.

The Hessian matrix at those minimizers is diag(1, H * / w * 2 ) where the matrix H * is DISPLAYFORM8 which is positive semi-definite and has a zero eigenvalue corresponding to the eigenvector u, i.e. H * u = 0.

The following lemma, similar to the well known Cauchy interlacing theorem, gives an estimate of eigenvalues of H * .Lemma A.1.

If H is positive definite and H * is defined as DISPLAYFORM9 , then the eigenvalues of H and H * satisfy the following inequalities: DISPLAYFORM10 Here λ i (H) means the i-th smallest eigenvalue of H.Proof.(1) According to the definition, we have H * u = 0, and for any x ∈ R d , DISPLAYFORM11 which implies H * is semi-positive definite, and λ i (H * ) ≥ λ 1 (H * ) = 0.

Furthermore, we have the following equality: DISPLAYFORM12 (2) We will prove DISPLAYFORM13 In fact, using the Min-Max Theorem, we have DISPLAYFORM14 (3) We will prove λ i (H * ) ≥ λ i−1 (H) for all i, 2 ≤ i ≤ d. In fact, using the Max-Min Theorem, we have DISPLAYFORM15 where we have used the fact that DISPLAYFORM16 There are several corollaries related to the spectral property of H * .

We first give some definitions.

Since H * is positive semi-definite, we can define the H * -seminorm.

Definition A.2.

The H * -seminorm of a vector x is defined as x H * := x T H * x. x H * = 0 if and only if x is parallel to u. DISPLAYFORM17 λ2(H * ) .

Definition A.4.

For any real number ε, the pseudo-spectral radius of the matrix I − εH * is defined as ρ DISPLAYFORM18 The following corollaries are direct consequences of Lemma A.1, hence we omit the proofs.

Corollary A.5.

The pseudo-condition number of H * is less than or equal to the condition number of H : DISPLAYFORM19 where the equality holds up if and only if u ⊥ span{v 1 , v d }, v i is the eigenvector of H corresponding to eigenvalue λ i (H).Corollary A.6.

For any vector x ∈ R d and any real number ε, we have (I − εH DISPLAYFORM20 Corollary A.7.

For any positive number ε > 0, we have DISPLAYFORM21 where the inequality is strict if u DISPLAYFORM22 It is obvious that the inequality in FORMULA2 and FORMULA2 is strict for almost all u.

The dynamical system defined in equation FORMULA9 - FORMULA10 is completely determined by a set of configurations {H, u, a 0 , w 0 , ε a , ε}. It is easy to check the system has the following scaling property: Lemma A.8 (Scaling property).

Suppose µ = 0, γ = 0, r = 0, Q T Q = I, then(1) The configurations {µQ T HQ, γ √ µ Qu, γa 0 , γQw 0 , ε a , ε} and {H, u, a 0 , w 0 , ε a , ε} are equivalent.(2) The configurations {H, u, a 0 , w 0 , ε a , ε} and {H, u, a 0 , rw 0 , ε a , r 2 ε} are equivalent.

The scaling property is valid for general loss functions provided batch normalization is used.

Consider a general problem DISPLAYFORM0 and its BN version DISPLAYFORM1 Then the gradient descent method gives the following iteration, DISPLAYFORM2 DISPLAYFORM3 whereh = h(a k w k /σ k ), and h is the gradient of original problem: DISPLAYFORM4 It is easy to check the general BNGD has the following property: Lemma A.9 (General scaling property).

Suppose r = 0, then the configurations {w 0 , ε, * } and {rw 0 , r 2 ε, * } are equivalent.

Here the sign * means other parameters.

Recall the BNGD iterations DISPLAYFORM0 Hw k .The scaling property simplify our analysis by allowing us to set, for example, u = 1 and w 0 = 1.

In the rest of this section, we only set u = 1.For the step size of a, it is easy to check that a k tends to infinity with ε a > 2 and initial value a 0 = 1, w 0 = u.

Hence we only consider 0 < ε a < 2, which make the iteration of a k bounded by some constant C a .

Lemma A.10 (Boundedness of a k ).

If the step size 0 < ε a < 2, then the sequence a k is bounded for any ε > 0 and any initial value (a 0 , w 0 ).

DISPLAYFORM1 According to the iterations (34), we have DISPLAYFORM2 Define DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 and using the property DISPLAYFORM6 u − tw H , and the property of H-norm, we have DISPLAYFORM7 Therefore we have the following lemma to make sure the iteration converge: Lemma A.11.

Let 0 < ε a < 2.

If there are two positive numbers ε − andε + , and the effective step sizeε k satisfies DISPLAYFORM8 for all k large enough, then the iterations (34) converge to a minimizer.

Proof.

Without loss of generality, we assume FORMULA3 is satisfied for all k ≥ 0.

We will prove w k converges and the direction of w k converges to the direction of u.(1) Since w k is always increasing, we only need to prove it is bounded.

We have, DISPLAYFORM9 The inequality in last lines are based on the fact that He i 2 ≤ λ max e i 2 H , and |a k | are bounded by a constant C a .

Next, we will prove ∞ i=0 qi wi 2 < ∞, which implies w k are bounded.

According to the estimate (38), we have DISPLAYFORM10 where DISPLAYFORM11 .

Using the definition of q k , we have DISPLAYFORM12 Since q k is bounded in [0, u T Hu], summing both side of the inequality, we get the bound of the infinite series DISPLAYFORM13 (2) Since w k is bounded, we denoteε − := ε − w∞ 2 , and define ρ := max DISPLAYFORM14 then the inequality (38) implies q k+1 ≤ ρ 2 q k .

As a consequence, q k tends to zero, which implies the direction of w k converges to the direction of u.(3) The convergence of a k is a consequence of w k converging.

Since a k is bounded, we assume |a k | <C a √ u T Hu,C a ≥ 1, and define ε 0 := 1 2Caκλmax.

The following lemma gives the convergence for small step size.

Lemma A.12.

If the initial values (a 0 , w 0 ) satisfies a 0 w T 0 g > 0, and step size satisfies ε a ∈ (0, 1], ε/ w 0 2 < ε 0 , then the sequence (a k , w k ) converges to a global minimizer.

Remark 1:

If we set a 0 = 0, then we have DISPLAYFORM15 Remark 2: For the case of ε a ∈ (1, 2), if the initial value satisfies an additional condition 0 < |a 0 | ≤ ε a |w T 0 g| σ0 , then we have (a k , w k ) converges to a global minimizer as well.

Proof.

Without loss of generality, we only consider the case of a 0 > 0, w T 0 g > 0, w 0 ≥ 1.(1) We will prove a k > 0, w DISPLAYFORM16 On the one hand, if a k > 0, 0 < y k < 2δ, then DISPLAYFORM17 On the other hand, when a k > 0, y k > 0, ε < ε 0 , we have DISPLAYFORM18 DISPLAYFORM19 As a consequence, we have a k > 0, y k ≥ δ y := min{y 0 , δ} for all k by induction.(2) We will prove the effective step sizeε k satisfying the condition in Lemma A.11.Since a k is bounded, ε < ε 0 , we havê DISPLAYFORM20 and DISPLAYFORM21 which implies DISPLAYFORM22 and there is a positive constant ε − > 0 such that DISPLAYFORM23 (3) Employing the Lemma A.11, we conclude that (a k , w k ) converges to a global minimizer.

Lemma A.13.

If step size satisfies ε a ∈ (0, 1], ε/ w 0 2 < ε 0 , then the sequence (a k , w k ) converges.

Proof.

Thanks to Lemma A.12, we only need to consider the case of a k w T k g ≤ 0 for all k, and we will prove the iteration converges to a saddle point in this case.

Since the case of a k = 0 or w T k g = 0 is trivial, we assume a k w T k g < 0 below.

More specifically , we will prove |a k+1 | < r|a k | for some constant r ∈ (0, 1), which implies convergence to a saddle point.(1) If a k and a k+1 have same sign, hence different sign with w T k g, then we have DISPLAYFORM24 (2) If a k and a k+1 have different signs, then we have DISPLAYFORM25 Consequently, we get DISPLAYFORM26 (3) Setting r := max(|1 − ε a |, 2εε a κλ max − (1 − ε a )), we finish the proof.

To simplify our proofs for Theorem 3.3, we give two lemmas which are obvious but useful.

Lemma A.14.

If positive series f k , h k satisfy f k+1 ≤ rf k + h k , r ∈ (0, 1) and lim DISPLAYFORM27 Proof.

It is obvious, because the series b k defined by b k+1 = rb k + h k , b 0 > 0, tends to zeros.

Lemma A.15 (Separation property).

For δ 0 small enough, the set S := {w|y 2 q < δ 0 , w ≥ 1} is composed by two separated parts: S 1 and S 2 , dist(S 1 , S 2 ) > 0, where in the set S 1 one has y 2 < δ 1 , q > δ 2 , and in S 2 one has q < δ 2 , y 2 > δ 1 for some δ 1 > 0, δ 2 > 0.

Here y := w T g, q := DISPLAYFORM28 Proof.

The proof is based on H being positive.

The geometric meaning is illustrated in FIG4 .

Proof.

Denote y k := w T k g. According to the separation property (Lemma A.15), we can chose a δ 0 > 0 small enough such that the separated parts of the set S := {w|y 2 q < δ 0 , w ≥ 1}, S 1 and S 2 , have dist(S 1 , S 2 ) > 0.Because y 2 k q k tends to zero, we have w k belongs to S for k large enough, for instance k > k 1 .

On the other hand, because w k+1 − w k tends to zero, we have w k+1 −

w k < dist(S 1 , S 2 ) for k large enough, for instance k > k 2 .

Then consider k > k 3 := max(k 1 , k 2 ), we have all w k belongs to the same part S 1 or S 2 .

DISPLAYFORM29 On the other hand, if w k ∈ S 2 , (y DISPLAYFORM30 Theorem A.17.

Let ε a ∈ (0, 1] and ε > 0.

The sequence (a k , w k ) converges for any initial value (a 0 , w 0 ).Proof.

We will prove w k converges, then prove (a k , w k ) converges as well.(1) We will prove that w k is bounded and hence converges.

In fact, according to the Lemma A.13, once w k 2 ≥ ε/ε 0 for some k, the rest of the iteration will converge, hence w k is bounded.(2) We will prove lim k→∞ w k+1 − w k = 0, and lim DISPLAYFORM31 The convergence of w k implies k a 2 k q k is summable.

As a consequence, lim DISPLAYFORM32 and lim k→∞ w k+1 − w k = 0.

In fact, we have DISPLAYFORM33 Consider the iteration of series |a k − w DISPLAYFORM34 The constant C in (57) can be chosen as C = ελmax u H λmin w0 2 .

Since a k e k H tends to zero, we can use Lemma A.14 to get lim DISPLAYFORM35 Combine the equation FORMULA5 , then we have lim case, the iteration of (a k , w k ) converges to a saddle point.

However, in the latter case, (a k , w k ) converges to a global minimizer.

In both cases we have (a k , w k ) converges.

DISPLAYFORM36 To finish the proof of Theorem 3.3, we have to demonstrate the special case of ε a = 1 where the set of initial values such that BN iteration converges to saddle points is Lebeguse measure zero.

We leave this demonstration in next section where we consider the case of ε a ≥ 1.

In this section, we will prove the set of initial values such that BN iteration converges to saddle points is (Lebeguse) measure zero, as long as ε a ≥ 1.

The tools in our proof is similar to the analysis of gradient descent on non-convex objectives BID12 BID14 .

In addition, we used the real analytic property of the BN loss function (16).For brevity, here we denote x := (a, w) and let ε a = ε, then the BN iteration can be rewrote as DISPLAYFORM0 ) is a measure zero set, then the preimage T −1 (A) is of measure zero as well.

Proof.

Since T is smooth enough, according to Theorem 3 of BID15 , we only need to prove the Jacobian of T (x) is nonzero for almost all x ∈ R d .

In other words, the set {x : det(I − ε∇ 2 J(x)) = 0} is of measure zero.

This is true because the function det(I − ε∇ 2 J(x)) is a real analytic function of x ∈ R d /{0}. (Details of properties of real analytic functions can be found in BID9

Lemma A.19.

Let f : X → R be twice continuously differentiable in an open set X ⊂ R d and x * ∈ X be a stationary point of f .

If ε > 0, det(I − ε∇ 2 f (x * )) = 0 and the matrix ∇ 2 f (x * ) has at least a negative eigenvalue, then there exist a neighborhood U of x * such that the following set B has measure zero, DISPLAYFORM0 Proof.

The detailed proof is similar to BID12 BID14 .Define the transform function as F (x) := x − ε∇f (x).

Since det(I − ε∇ 2 f (x * )) = 0, accorded to the inverse function theorem, there exist a neighborhood U of x * such that T has differentiable inverse.

Hence T is a local C 1 diffeomorphism, which allow us to use the central-stable manifold theorem BID18 .

The negative eigenvalues of ∇ 2 f (x * ) indicates λ max (I − ε∇ 2 f (x * )) > 1 and the dimension of the unstable manifold is at least one, which implies the set B is on a lower dimension manifold hence B is of measure zero.

Lemma A.20.

If ε a = ε ≥ 1, then the set of initial values such that BN iteration converges to saddle points is of Lebeguse measure zero.

Proof.

We will prove this argument using Lemma A.18 and Lemma A.19 .

Denote the saddle points set as W := {(a * , w * ) : a * = 0, w * T g = 0}. The basic point is that the saddle point x * := (a * , w * )of the BN loss function (16) has eigenvalues (1) For each saddle point x * := (a * , w * ) of BN loss function, ε ≥ 1 is enough to allow us to use Lemma A.19.

Hence there exist a neighborhood U x * of x * such that the following set B x * is of measure zero, DISPLAYFORM1 (2) The neighborhoods U x * of all x * ∈ W forms a cover of W , hence, accorded to Lindelöf's open cover lemma, there are countable neighborhoods {U i : i = 1, 2, ...} cover W , i.e. U := ∪ i U i ⊇ W .

As a consequence, the following set A 0 is of measure zero, DISPLAYFORM2

In the last section, we encountered the following estimate for e k = u − DISPLAYFORM0 We can improve the convergence rate of the above if H * has better spectral property.

This is the content of Proposition 3.4 and the following lemma is enough to prove it.

Lemma A.22.

The following inequality holds, DISPLAYFORM1 DISPLAYFORM2

Through our analysis, we discovered that a modification of the BNGD, which we call MBNGD, becomes much easier to analyze and possesses better convergence properties.

Note that the results in the main paper do not depend on the results in this section.

The modification is simply to enforce a k = w Theorem B.4.

The iteration sequence w k in equation FORMULA8 converges for any initial value w 0 and any step size ε > 0.

Furthermore, w k will converge to a global minimizer unless w T k g = 0 for some k.

Proof.

Obviously, if w T k g = 0 for some k = k 0 , then w k = w k0 for all k ≥ k 0 , hence w k converges to w k0 .

Without losing generality, we consider w T k g = 0 for all k and w 0 ≥ 1 below.(1) Firstly, we will prove that w k is bounded and hence converges.

In fact, according to the Lemma B.2, once w k 2 ≥ ε/ε 0 for some k, the rest of the iteration will converge, hence w k is bounded.(2) Secondly, we will prove w k converges to a vector parallel to u. and the above tends to zero, i.e. lim k→∞ w k+1 − w k = 0.According to the separation property (Lemma A.15), we can chose a δ 0 > 0 small enough such that the separated parts of the set S := {w|y 2 q < δ 0 , w ≥ 1}, S 1 and S 2 , have dist(S 1 , S 2 ) > 0.Because y 2 k q k tends to zero, we have w k belongs to S for k large enough, for instance k > k 1 .

On the other hand, because w k+1 − w k tends to zero, we have w k+1 −

w k < dist(S 1 , S 2 ) for k large enough, for instance k > k 2 .

Then consider k > k 3 := max(k 1 , k 2 ), we have all w k belongs to the same part S 1 or S 2 .However, Lemma B.3 says ∞ k=0 y 2 k = ∞, hence w k ∈ S 1 (q k > δ 2 ) for all k > k 3 is not true.

Therefore w k ∈ S 2 (y 2 k > δ 1 ) for all k > k 3 .

Consequently, we can claim that ∞ k=0 q k is summable and w k converges to a vector parallel to u.

Here we test the convergence and stability of MBNGD for OLS model.

Consider the diagonal matrix H = diag(h), where h = (1, ..., κ) is an increasing sequence.

The scaling property allows us to set the initial value w 0 having same 2-norm with u, w 0 = u = 1.

FIG9 gives an example of a 5-dimensional H with condition number κ = 2000.

The GD and MBNGD iteration are executed k = 5000 times where u and w 0 are randomly chosen from the unit sphere.

The values of effective step size, loss e k 2 H and error e k are plotted.

Furthermore, to explore the performance of GD and MBNGD, the mean values over 300 random tests are given.

It is worth to note that, the geometric mean (G-mean) is more reliable than the arithmetic mean (Amean), where the geometric mean of x can be defined as exp(E(ln x)).

Here the reliability means that the G-mean converges quickly when the number of tests increase, however the A-mean does not converge as quickly.

In this example, the optimal convergence rate of MBNGD is observably better than GD.

This acceleration phenomenon is ascribed to the pseudo-condition number of κ * (H * ) being less than κ(H).

However, if the difference between (pseudo-)condition number of H and H * is small, the acceleration is imperceptible.

Another important observation is that the BN significantly extends the range of 'optimal' step size, which is embodied by the effective step sizeε k having a large constant C inε = O(Cε −1 ).

This means we can chose step size in BN at a large interval to get almost same or better convergence rate than that of the best choice for GD.

FIG10 gives an example of 100-dimension H with condition number κ = 2000.

Similar results as those in the 5-dimensional case are obtained.

However, the best optimal convergence rate of MBNGD here has not noticeably improved compared with GD with the optimal learning rate, which is due to the fact that large d decrease the difference between eigenvalues of H and H * .Additional tests indicate that: (1) larger dimensions leads to larger intervals of 'optimal' step size, FIG11 (2) the effect of condition number on the 'optimal' interval is small FIG12 ).

FIG0 ).

<|TLDR|>

@highlight

We mathematically analyze the effect of batch normalization on a simple model and obtain key new insights that applies to general supervised learning.
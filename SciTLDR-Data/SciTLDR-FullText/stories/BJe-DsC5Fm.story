In this paper, we design and analyze a new zeroth-order (ZO) stochastic optimization algorithm, ZO-signSGD, which enjoys dual advantages of gradient-free operations and signSGD.

The latter requires only the sign information of  gradient estimates but is able to achieve a comparable  or even better convergence speed than SGD-type algorithms.

Our study  shows that ZO signSGD requires $\sqrt{d}$ times more iterations than signSGD, leading to a convergence rate of  $O(\sqrt{d}/\sqrt{T})$ under mild conditions, where $d$ is the number of optimization variables, and $T$ is the number of iterations.

In addition, we analyze the effects of different types of gradient estimators on the convergence of ZO-signSGD, and propose two variants of ZO-signSGD that  at least  achieve $O(\sqrt{d}/\sqrt{T})$ convergence rate.

On the application side we explore the connection between ZO-signSGD and  black-box adversarial attacks in robust deep learning.

Our empirical evaluations on image classification datasets MNIST and CIFAR-10 demonstrate the superior performance of ZO-signSGD on the generation of   adversarial examples from black-box neural networks.

Zeroth-order (gradient-free) optimization has attracted an increasing amount of attention for solving machine learning (ML) problems in scenarios where explicit expressions for the gradients are difficult or infeasible to obtain.

One recent application of great interest is to generate prediction-evasive adversarial examples, e.g., crafted images with imperceptible perturbations to deceive a well-trained image classifier into misclassification.

However, the black-box optimization nature limits the practical design of adversarial examples, where internal configurations and operating mechanism of public ML systems (e.g., Google Cloud Vision API) are not revealed to practitioners and the only mode of interaction with the system is via submitting inputs and receiving the corresponding predicted outputs BID31 BID27 BID36 BID17 BID3 .

It was observed in both white-box and black-box settings 1 that simply leveraging the sign information of gradient estimates of an attacking loss can achieve superior empirical performance in generating adversarial examples BID13 BID28 BID16 .

Spurred by that, this paper proposes a zeroth-order (ZO) sign-based descent algorithm (we call it 'ZO-signSGD') for solving black-box optimization problems, e.g. design of black-box adversarial examples.

The convergence behavior and algorithmic stability of the proposed ZO-signSGD algorithm are carefully studied in both theory and practice.

In the first-order setting, a sign-based stochastic gradient descent method, known as signSGD, was analyzed by BID2 BID1 .

It was shown in BID2 that signSGD not only reduces the per iteration cost of communicating gradients, but also could yield a faster empirical convergence speed than SGD BID19 .

That is because although the sign operation compresses the gradient using a single bit, it could mitigate the negative effect of extremely components of gradient noise.

Theoretically, signSGD achieves O(1/ √ T ) convergence rate under the condition of a sufficiently large mini-batch size, where T denotes the total number of iterations.

The work in BID1 established a connection between signSGD and Adam with restrictive convex analysis.

Prior to BID2 BID1 , although signSGD was not formally defined, the fast gradient sign method BID13 to generate white-box adversarial examples actually obeys the algorithmic protocol of signSGD.

The effectiveness of signSGD has been witnessed by robust adversarial training of deep neural networks (DNNs) BID28 .

Given the advantages of signSGD, one may wonder if it can be generalized for ZO optimization and what the corresponding convergence rate is.

In this paper, we answer these questions affirmatively.

Contributions We summarize our key contributions as follows.• We propose a new ZO algorithm, 'ZO-signSGD', and rigorously prove its convergence rate of O( √ d/ √ T ) under mild conditions.

• Our established convergence analysis applies to both mini-batch sampling schemes with and without replacement.

In particular, the ZO sign-based gradient descent algorithm can be treated as a special case in our proposed ZO-signSGD algorithm.• We carefully study the effects of different types of gradient estimators on the convergence of ZO-signSGD, and propose three variants of ZO-signSGD for both centralized and distributed ZO optimization.• We conduct extensive synthetic experiments to thoroughly benchmark the performance of ZO-signSGD and to investigate its parameter sensitivity.

We also demonstrate the superior performance of ZO-signSGD for generating adversarial examples from black-box DNNs.

Related work Other types of ZO algorithms have been developed for convex and nonconvex optimization, where the full gradient is approximated via a random or deterministic gradient estimate BID18 BID29 BID11 BID9 BID10 BID34 BID15 BID12 BID22 BID25 .

Examples include ZO-SGD BID11 , ZO stochastic coordinate descent (ZO-SCD) BID22 , and ZO stochastic variance reduced gradient descent (ZO-SVRG) BID26 a; BID14 .

Both ZO-SGD and ZO-SCD can achieve O( DISPLAYFORM0 And ZO-SVRG can further improve the iteration complexity to O(d/T ) but suffers from an increase of function query complexity due to the additional variance reduced step, known as 'gradient blending' BID26 ), compared to ZO-SGD.

The existing work showed that ZO algorithms align with the iteration complexity of their first-order counterparts up to a slowdown effect in terms of a small-degree polynomial of the problem size d.

In this section, we provide a background on signSGD, together with the problem setup of our interest.

In particular, we show that the commonly-used methods for generating adversarial attacks fall into the framework of signSGD.Preliminaries on signSGD Consider a nonconvex finite-sum problem of the form DISPLAYFORM0 where x ∈ R d are optimization variables, and {f i } are n individual nonconvex cost functions.

The finite-sum form (1) encompasses many ML problems, ranging from generalized linear models to neural networks.

If the gradients of {f i } are available, then problem (1) can be solved by many first-order methods such as SGD, SCD, and signSGD.

The method of our interest is signSGD, which differs from SGD and SCD, takes the sign of gradient (or its estimate) as the descent direction.

It was recently shown in BID2 that signSGD is quite robust to gradient noise and yields fast empirical convergence.

Algorithm 1 provides a generic sign-based gradient descent framework that encapsulates different variants of signSGD.

In Algorithm 1, GradEstimate(·) signifies a general gradient estimation procedure, which adopts either a stochastic gradient estimate in the first-order setting BID2 or a function difference based random gradient estimate in the ZO setting BID29 BID9 .

We call the ZO variant of signSGD 'ZO-signSGD', which will be elaborated on in Sec. 3.Adversarial attacks meet signSGD It is now widely known that ML models (e.g., deep neural networks) are vulnerable to adversarial attacks, which craft inputs (e.g., images) with imperceptible perturbations to cause incorrect classification BID35 BID13 BID20 BID23 .

The resulting inputs crafted by adversaries are known as adversarial examples.

Investigating adversarial examples not only helps to understand the limitation of learning models, but also provides opportunities to improve the models' robustness BID30 Algorithm 1 Generic sign-based gradient descent 1:

Input: learning rate {δ k }, initial value x 0 , and number of iterations T 2: for k = 0, 1, . . .

, T − 1 do 3:ĝ k ←− GradEstimate(x k ) # applies to both first and zeroth order gradient estimates 4: sign-gradient update DISPLAYFORM1 , where sign(x) takes element-wise signs of x5: end for BID0 BID28 .

In what follows, we show that the generation of adversarial examples in BID13 BID20 can be interpreted through signSGD.Let x 0 denote the natural (legitimate) input of an ML model associated with the true label t 0 , and x = x 0 + δ be the adversarial example to be designed, where δ are adversarial perturbations.

If f (x, t 0 ) is the training loss of a learning model, then the goal of (white-box) adversarial attack is to find minimal perturbation δ that is sufficient to mislead the learning model, namely, to maximize the loss f (x 0 + δ, t 0 ).

Taking the first-order approximation of f (x , t 0 ) around x 0 , we obtain DISPLAYFORM2 By constraining the strength of perturbation in the ∞ ball of small radius (i.e., δ ∞ ≤ ), the linear approximation of f (x , t 0 ) is then maximized at δ = sign(∇ x f (x 0 , t 0 )) BID33 .

Therefore, generation of adversarial examples proposed in BID13 obeys the sign-gradient update rule in (2), DISPLAYFORM3 Such a connection between adversarial example generation and signSGD also holds in other attacks, e.g., the iterative target attack method BID20 .

Similarly, a so-called black-box attack BID16 BID3 is associated with our proposed ZO-signSGD algorithm.

One limitation of signSGD BID2 is the need of first-order information, i.e., stochastic gradients.

However, there exists a large practical demand for solving ML problems where explicit expressions of the gradients are difficult or infeasible to obtain, e.g., the generation of adversarial examples from black-box neural networks as discussed in Sec. 1 and 2.Gradient estimation via ZO oracle In the ZO setting where the first-order information is unavailable, the gradient estimator at Step 3 of Algorithm 1 has only access to function values of {f i (x)} given a query point x. Based on that, we construct a ZO gradient estimate through a forward difference of two function values BID29 BID10 BID9 .

In Algorithm 1, GradEstimate(x) is then specified as DISPLAYFORM0 where DISPLAYFORM1 are i.i.d.

random directions drawn from a uniform distribution over a unit sphere, and∇f i (x; u i,j ) gives a two-point based random gradient estimate with direction u i,j and smoothing parameter µ > 0.

We remark that the random direction vectors in (3) can also be drawn from the standard Gaussian distribution BID29 .

However, the uniform distribution could be more useful in practice since it is defined in a bounded space rather than the whole real space required for Gaussian.

We highlight that unlike the first-order stochastic gradient estimate, the ZO gradient estimate (3) is a biased approximation to the true gradient of f .

Instead, it becomes unbiased to the gradient of the randomized smoothing function f µ BID8 BID10 , DISPLAYFORM2 where f i,µ gives the randomized smoothing version of f i , and the random variable v follows a uniform distribution over the unit Euclidean ball.

Clearly, there exists a gap between a ZO gradient estimate and the true gradient of f , but as will be evident later, such a gap can be measured through the smoothing function f µ .Motivations of ZO-signSGD.

Compared to SGD-type methods, the fast empirical convergence of signSGD and ZO-signSGD has been shown in the application of generating white-box and black-box adversarial examples BID13 BID28 BID16 .

As mentioned in BID2 , the sign operation could mitigate the negative effect of (coordinate-wise) gradient noise of large variance.

Recall that the ZO gradient estimate is a biased approximation to the true gradient, and thus, could suffer from having larger noise variance than (first-order) stochastic gradients.

In this context, one could benefit from ZO-signSGD due to its robustness to gradient noise.

In Appendix 1, we provide two concrete examples ( FIG1 and Fig. A2 ) to confirm the aforementioned analysis.

In FIG1 , we show the robustness of ZO-signSGD against sparse noise perturbation through a toy quadratic optimization problem, originally introduced in BID2 to motivate the fast convergence of signSGD against SGD.

In Fig. A2 , we show that gradient estimation via ZO oracle indeed encounters gradient noise of large variance.

Thus, taking the sign of a gradient estimate might scale down the extremely noisy components.

ZO-signSGD & technical challenges beyond signSGD Algorithm 1 becomes ZO-signSGD as the ZO gradient estimate (3) is applied.

We note that the extension from first order to ZO is nontrivial, as the proposed ZO-signSGD algorithm yields three key differences to signSGD.First, ZO-signSGD has milder assumption on the choice of mini-batch sampling.

Recall that signSGD in BID2 achieves O(1/ √ T ) convergence rate given the condition that the mini-batch size is sufficiently large, b = O(T ).

However, this condition only becomes true when the mini-batch sample is randomly selected from [n] with replacement, which is unusual when n ≤ T .

Here [n] represents the integer set {1, 2, . . .

, n}. And signSGD fails to cover signGD when b = n, since sampling with replacement leads to I k = [n] even if b = n. In the proposed ZO-signSGD algorithm, we will relax the assumption on mini-batch sampling.

Second, in ZO-signSGD both the ZO gradient estimator and the sign operator give rise to approximation errors to the true gradient.

Although the statistical properties of ZO gradient estimates can be acquired with the aid of the randomized smoothing function (4), the use of mini-batch sampling without replacement introduces extra difficulty to bound the variance of ZO gradient estimates since mini-batch samples are no longer independent.

Moreover, the sign-based descent algorithm evaluates the convergence error in the 1 -norm geometry, leading to a mismatch with the 2 -norm based gradient variance.

Besides translating the the gradient norm from 1 to 2 , the probabilistic convergence method BID11 ) is used to bound the eventual convergence error of ZO-signSGD.Finally, beyond the standard ZO gradient estimator (3), we will cover multiple variants of ZOsignSGD for centralized or distributed optimization.

In this section, we begin by stating assumptions used in our analysis.

We then derive the convergence rate of ZO-signSGD for nonconvex optimization.

Assumptions of problem (1) are listed as follows.

DISPLAYFORM0 Both A1 and A2 are the standard assumptions used in nonconvex optimization literature BID2 BID32 .

A1 implies the L-smoothness of f i , namely, for any x and y we obtain DISPLAYFORM1 i .

A2 implies the bounded variance of ∇f i in BID2 , Assumption 3), namely, DISPLAYFORM2 2 , where we have used the fact that ∇f (x) 2 ≤ σ under A2.

Throughout the paper, we assume that problem (1) is solvable, namely, f (x * ) > −∞ where x * is an optimal solution.

We recall that Algorithm 1 becomes ZO-signSGD when the gradient estimation step (3) is applied.

For nonconvex problems, the convergence of an algorithm is typically measured by stationarity, e.g., using ∇f (x) 2 2 in SGD BID11 and ∇f (x) 1 in signSGD BID2 .

For the latter, the 1 geometry is met when quantifying the stochasticity through the (non-linear) sign operation.

Different from signSGD, ZO-signSGD only obtains a biased estimate to the true gradient.

In Proposition 1, we bypass such a bias by leveraging the randomized smoothing technique used for ZO optimization BID10 BID29 BID9 .Proposition 1 Under A1, the outputs {x k } T −1 k=0 of ZO-signSGD, i.e., Algorithm 1 with (3), satisfies DISPLAYFORM3 where the expectation is taken with respect to all the randomness of ZO-signSGD, f µ is the randomized smoothing function of f in (4), andĝ k = GradEstimate(x k ) in (3).Proof: See Appendix 2.In Proposition 1, the rationale behind introducing the smoothing function f µ is that ∇f µ (x k ) is the mean of ZO gradient estimateĝ k .

And thus, the convergence of ZO-signSGD is now linked with the variance ofĝ DISPLAYFORM4 ].

This crucial relationship presented in Proposition 1 holds for a general class of signSGD-type algorithms that use different ZO gradient estimators.

Spurred by (5), we next investigate the second-order moment ofĝ k in Proposition 2.Proposition 2 Under A1 and A2, the variance of ZO gradient estimateĝ k is upper bounded by DISPLAYFORM5 where DISPLAYFORM6 In FORMULA14 , α b and β b are Boolean variables depending on the choice of mini-batch sampling, DISPLAYFORM7 for mini-batch with replacement DISPLAYFORM8 where I(x > a) is the indicator function of x with respect to the constraint x > a, and I(x > a) = 1 if x > a and 0 otherwise.

Proof: See Appendix 3.Compared to the variance bound (σ 2 /b) of the stochastic gradient estimate of f in signSGD BID2 , Proposition 2 provides a general result for the ZO gradient estimateĝ k .

It is clear that the bound in (6) contains two parts: DISPLAYFORM9 , where the former h 1 = O(σ 2 /b) characterizes the reduced variance (using b mini-batch samples) for the stochastic gradient estimate of the smoothing function f µ , and the latter h 2 = O(C(d, µ)/(bq)) reveals the dimension-dependent variance induced by ZO gradient estimate using b mini-batch samples and q random directions.

If a stochastic gradient estimate of f is used in signSGD, then h 2 is eliminated and the variance bound in (6) is reduced to (σ 2 /b).Furthermore, Proposition 2 covers mini-batch sampling with and without replacement, while signSGD only considers the former case.

For the latter case, Proposition 2 implies that if b = n (i.e., DISPLAYFORM10 , corresponding to α b = 0 and β b = 1 in (7).

In the other extreme case of b = 1, both the studied mini-batch schemes become identical, corresponding to α b = 1 and β b = 0.

Proposition 2 also implies that the use of large b and q reduces the variance of the gradient estimate, and will further improve the convergence rate.

With the aid of Proposition 1 and 2, we can then show the convergence rate of ZO-signSGD in terms of stationarity of the original function f .

The remaining difficulty is how to bound the gap between f and its smoothed version f µ .

It has been shown in BID10 BID29 that there exists a tight relationship between f µ and f given the fact that the former is a convolution of the latter and the density function of a random perturbation v in (4).

We demonstrate the convergence rate of ZO-signSGD in Theorem 1.Theorem 1 Under A1 and A2, if we randomly pick x R from {x k } T −1 k=0 with probability P (R = k) = DISPLAYFORM11 , then the convergence rate of ZO-signSGD is given by DISPLAYFORM12 where f * denotes the minimum value.

Proof: See Appendix 4.In Theorem 1, we translate the gradient norm from 1 to 2 , and adopt a probabilistic output x R BID11 BID21 to avoid exhaustive search over {x k } for min k ∇f (x k ) 2 .

Note that the convergence rate of ZO-signSGD relies on the learning rate δ k , the problem size d, the smoothing parameter µ, the mini-batch size b, and the number of random perturbations q for ZO gradient estimation.

We next obtain explicit dependence on these parameters by specifying Theorem 1.

DISPLAYFORM13 ), then the convergence in (8) simplifies to DISPLAYFORM14 where α b and β b were defined in FORMULA17 , and 1 ≤ (α b + β b ) ≤ 2.

We provide several key insights on the convergence rate of ZO-signSGD through (9).First, the convergence rate of ZO-signSGD is measured through ∇f (x R ) 2 rather than its squared counterpart ∇f (x R ) 2 2 , where the latter was used in measuring the convergence of ZO-SGD.

We recall from (Ghadimi & Lan, 2013, Theorem 3.2 & Corollary 3.

3) that ZO-SGD yields the convergence DISPLAYFORM15 2 ≤ 1, the convergence of ZO-signSGD meets a stricter criterion than that of ZO-SGD.

The possible downside of ZO-signSGD is that it suffers an additional error of order O( DISPLAYFORM16 ) in the worst case.

The aforementioned results imply that ZO-signSGD could only converge to a neighborhood of a stationary point but with a fast convergence speed.

Here the size of the neighborhood is controlled by the mini-batch size b and the number of random direction vectors q.

Also, our convergence analysis applies to mini-batch sampling both with and without replacement.

DISPLAYFORM17 ) convergence rate regardless of the choice of mini-batch sampling.

When b = n, it is known from (9) that the use of mini-batch without replacement recovers ZO-signGD, yielding the convergence rate O( DISPLAYFORM18 .

By contrast, the use of mini-batch with replacement leads to the worse convergence rate O( DISPLAYFORM19 ).

Clearly, as b = n and n < T , ZO-signSGD using mini-batch with replacement fails to achieve the rate O( DISPLAYFORM20 ) regardless of the choice of q. By contrast, ZO-signSGD using mini-batch without replacement recovers O( DISPLAYFORM21 When b > n, ZO-signSGD is restricted to using mini-batch sampling with replacement.

Similar to signSGD BID2 , we can obtain O( DISPLAYFORM22 , where the dependence on q is induced by the use of ZO gradient estimation.

Here we study three variants of ZO-signSGD, where the gradient will be estimated using a) the central difference of function values, b) the sign of ZO gradient estimates with majority vote, or c) the sign of ZO gradient estimates with majority vote for distributed optimization.

That is, DISPLAYFORM0 DISPLAYFORM1 where {u i,j } and∇f i (x; u i,j ) have been defined in (3).

The gradient estimator FORMULA1 The ZO gradient estimator (10) was used in BID34 for bandit convex optimization and in BID16 for designing black-box adversarial attacks.

Compared to the form of forward difference (3), the central difference (10) requires b(q − 1) times more function queries in gradient estimation.

At the cost of more function queries, one may wonder if the convergence rate of ZO-signSGD can be further improved.

Corollary 1 Suppose that the conditions in Theorem 1 hold, ZO-signSGD with gradient estimator (10) yields the same convergence rate of ZO-signSGD that uses the estimator (3).Proof: Recall that Proposition 1 is independent of specific forms of gradient estimators, and thus holds for (10).

Although Proposition 2 relies on the second-order moments of each gradient estimator, we prove that under A1 and A2, both (3) and (10) maintain the same statistical properties.

As a result, Proposition 2 and Theorem 1 also hold for (10); see more details in Appendix 5.We next study the gradient estimator (11), whose sign is equivalent to the majority vote (i.e., the element-wise median) of signs of individual gradient estimates {∇f i (x; u i,j )}.

It was shown in BID2 ) that signSGD with majority vote has a better convergence rate under additional assumptions of unimodal symmetric noise distribution of coordinate-wise gradient estimates.

In Corollary 2, we show that such a speed-up in convergence can also be achieved by ZO-signSGD with majority vote, which we refer to as 'ZO-M-signSGD'.Corollary 2 Suppose that the conditions in Theorem 1 hold, and the distribution of gradient noise is unimodal and symmetric.

Then, ZO-M-signSGD with δ k = O( DISPLAYFORM2 Proof: See Appendix 6.We recall from Theorem 1 that under the same parameter setting of Corollary 2, ZO-signSGD yields O( DISPLAYFORM3 ) convergence rate in the worst case.

It is clear from (13) that the error correction term of order DISPLAYFORM4 is eliminated in ZO-M-signSGD.

Such an improvement in convergence is achieved under the condition of unimodal symmetric gradient noise.

We remark that different from the stochastic gradient noise studied in BID2 , the ZO gradient estimation noise could violate this assumption.

For example, in a scalar case, if the gradient estimate g follows the distribution where g = 1 with probability 0.9, g = −10 with probability 0.1, then E[g] < 0 and sign(E[g]) < 0.

However, E[sign(g)] > 0.

This implies that without the assumption of symmetry, the sign of gradient estimates with majority vote (E[sign(g)]) can be in the opposite direction of the sign of averaged gradients (sign(E[g])).

Our results in the next section show that ZO-M-signSGD may not outperform ZO-signSGD.Lastly, we focus on the gradient estimator (12), whose sign can be interpreted as the major vote of M distributed agents about the sign of the true gradient BID2 .

The resulting variant of ZO-signSGD is called 'ZO-D-signSGD', and its convergence rate is illustrated in Corollary 3.

Compared to ZO-M-signSGD for centralized optimization, ZO-D-signSGD suffers an extra error correction term O( DISPLAYFORM5 ) in the distributed setting.

It is also worth mentioning that if M = n and q = 1, then the gradient estimator (12) reduces to (11) with I k = [n].

In this case, Corollary 2 and 3 reach a consensus on O( DISPLAYFORM6 ) convergence error.

Proof: See Appendix 7.

In this section, we empirically show the effectiveness of ZO-signSGD, and validate its convergence behavior on both synthetic and real-world datasets such as MNIST and CIFAR-10.

For the synthetic experiment, we study the problem of binary classification in the least squared formulation.

For the real-world application, we design adversarial examples from black-box neural networks as mentioned in Sec. 2.

Throughout this section, we compare ZO-signSGD and its variants with SGD, signSGD BID2 , ZO-SGD BID11 , and ZO-SCD BID22 .

We consider the least squared problem with a nonconvex loss function (Xu et al., 2017; BID25 DISPLAYFORM0 2 , which satisfies Assumption A2 by letting σ = max i {2 a i 2 }.

Here instead of using the conventional cost function of logistic regression (a convex function), the considered least squared formulation is introduced to align with our nonconvex theoretical analysis.

For generating the synthetic dataset, we randomly draw samples {a i } from N (0, I), and obtain the label y i = 1 if 1/(1 + e −a T i x ) > 0.5 and 0 otherwise.

The number of training samples {a i , y i } is set by n = 2000 against 200 testing samples.

We find the best constant learning rate for algorithms via a greedy search over η ∈ [0.001, 0.1] (see Appendix 8.1 for more details), and we choose the smoothing parameter µ = 10/ √ T d. Unless specified otherwise, let b = q = 10, T = 5000 and d = 100.In FIG1 , we report the training loss, the test accuracy, as well as the effects of algorithmic parameters on the convergence of the studied algorithms.

We observe from FIG1 and (b) that ZO-signSGD outperforms other ZO algorithms, and signSGD yields the best convergence performance once the first-order information is available.

In FIG1 and (d) , we observe that the convergence performance of ZO algorithms is improved as b and q increase.

In particular, ZO-signSGD and ZO-M-signSGD at b = q = 30 approach to the best result provided by signSGD.

In FIG1 -(e) and (f), the convergence of all algorithms degrades as the problem size d increases.

However, ZO-signSGD and ZO-M-signSGD converge faster than ZO-SGD and ZO-SCD.

In Fig. 2 , we demonstrate the convergence trajectory of different variants of ZO-signSGD for b ∈ {40, 400}. To make a fair comparison between ZOsignSGD and ZO-D-signSGD, let each of M = 40 agents use a mini-batch of size b/M .

As we can see, ZO-signSGD outperforms ZO-M-signSGD and ZO-D-signSGD.

And the convergence is improved as the mini-batch size increases.

However, we observe that in all examples, ZO-signSGD and its variants converge to moderate accuracy much faster than ZO-SGD, only within a few tens of iterations.

Generating black-box adversarial examples Here we study adversarial robustness by generating adversarial examples from a black-box image classifier trained by a deep neural network (DNN) model; see details on problem formulation in Appendix 8.2.

We recall from Sec. 2 that the task of black-box adversarial attack falls within the category of ZO optimization as one can only access to the input-output relation of the DNN while crafting adversarial examples.

The DNN models trained on MNIST and CIFAR-10 BID4 are performed as the zeroth-order oracle 2 .

We select one image from each class of MNIST and CIFAR-10 and separately implement black-box attacks using the same attacking loss function (see Appendix 8.2) but with different ZO optimization algorithms (ZO-SGD, ZO-signSGD and ZO-M-signSGD).

We also set the same parameters for each method, i.e., µ = 0.01, q = 9, and δ = 0.05 for MNIST and δ = 0.0005 for CIFAR-10, to accommodate to the dimension factor d. Moreover, we benchmark their performance with the natural evolution strategy (NES) based two-point gradient estimator in BID16 for solving the same attacking loss function, where the sign of gradient estimate is also used in the FORMULA1 , NES computes the ZO gradient estimate using the central difference of two function values.

Thus, one iteration of ZO-NES requires 2q function queries and thus we set q = 5 to align with the number of function queries used in other ZO methods.

All methods use the the same natural image as the initial point for finding adversarial examples.

Fig. 3 shows the plots of black-box attacking loss versus iterations (more results are shown in Appendix 8.3).

We find that ZO-signSGD usually takes significantly less iterations than other methods to find the first successful adversarial example with a similar attacking loss.

For MNIST, the average iteration over all attacked images in TAB0 to find the first successful adversarial example is 184 for ZO-SGD, 103 for ZO-signSGD, 151 for ZO-M-signSGD, and 227 for ZO-NES.

Their corresponding average 2 distortion is 2.345 for ZO-SGD, 2.381 for ZO-signSGD, 2.418 for ZO-MsignSGD, and 2.488 for ZO-NES.

For CIFAR-10, the average iteration over all attacked images in TAB3 to find the first successful adversarial example is 302 for ZO-SGD, 250 for ZO-signSGD, 389 for ZO-M-signSGD, and 363 for ZO-NES.

Their corresponding average 2 distortion is 0.177 for ZO-SGD, 0.208 for ZO-signSGD, 0.219 for ZO-M-signSGD, and 0.235 for ZO-NES.

As a visual illustration, we compare the adversarial examples of a hand-written digit "1" of each attacking method at different iterations in TAB0 , corresponding to Fig. 3-(a) .

As we can see, ZO-signSGD and ZO-M-signSGD can reduce roughly 54% of iterations (around 600 less model queries) than ZO-SGD to find the first successful adversarial example.

Given the first successful adversarial example, we observe that ZO-signSGD yields slightly higher 2 distortion than ZO-SGD.

This is not surprising since Theorem 1 suggests that ZO-signSGD might not converge to a solution of very high accuracy but it can converge to moderate accuracy sufficient for black-box attacks at a very fast speed.

Note that the first successful adversarial examples generated by different ZO methods are all visually similar to the original ones but lead to different top-1 predictions; see more results in Appendix 8.3.

In addition, we observe that ZO-NES is not as effective as ZO-signSGD in either query efficiency (given by the number of iterations to achieve the first successful attack) or attack distortion.

Thus, compared to ZO-NES, ZO-signSGD offers a provable and an efficient black-box adversarial attacking method.

Motivated by the impressive convergence behavior of (first-order) signSGD and the empirical success in crafting adversarial examples from black-box ML models, in this paper we rigorously prove the O( √ d/ √ T ) convergence rate of ZO-signSGD and its variants under mild conditions.

Compared to signSGD, ZO-signSGD suffers a slowdown (proportional to the problem size d) in convergence rate, however, it enjoys the gradient-free advantages.

Compared to other ZO algorithms, we corroborate the superior performance of ZO-signSGD on both synthetic and real-word datasets, particularly for its application to black-box adversarial attacks.

In the future, we would like to generalize our analysis to nonsmooth and nonconvex constrained optimization problems.

BID2 FIG1 , we assume that the ZO gradient estimate of f (x) and its first-order gradient ∇f (x) = x suffer from a sparse noise vector v, where v1 ∈ N (0, 1002 ), and vi = 0 for i ≥ 2.

As a result, the used descent direction at iteration t is given bŷ ∇f (xt) + v or ∇f (xt) + v. FIG1 presents the convergence performance of 5 algorithms: SGD, signSGD, ZO-SGD, ZO-signSGD and its variant using the central difference based gradient estimator (10).

Here we tune a constant learning rate finding 0.001 best for SGD and ZO-SGD and 0.01 best for signSGD and its ZO variants.

As we can see, sign-based first-order and ZO algorithms converge much faster than the stochastic gradient-based descent algorithms.

This is not surprising since the presence of extremely noisy component v1 leads to an inaccurate gradient value, and thus degrades the convergence of SGD and ZO-SGD.

By contrast, the sign information is more robust to outliers and thus leads to better convergence performance of sign SGD and its variants.

We also note that the convergence trajectory of ZO-signSGD using the gradient estimator FORMULA1 coincides with that using the gradient estimator FORMULA6 given by the forward difference of two function values.

FIG1 : Comparison of different gradient-based and gradient sign-based first-order and ZO algorithms in the example of sparse noise perturbation.

The solid line represents the loss averaged over 10 independent trials with random initialization, and the shaded region indicates the standard deviation of results over random trials.

Left: Loss value against iterations for SGD, signSGD, ZO-SGD, ZO-signSGD and ZO-signSGD using the central difference based gradient estimator (10).

Right: Local regions to highlight the effect of the gradient estimators (3) and (10) on the convergence of ZO-signSGD.

The intuition behind why ZO-signSGD could outperform ZO-SGD is that the sign operation can mitigate the negative effect of (coordinate-wise) gradient noise of large variance.

To confirm this point, we examine the coordinate-wise variance of gradient noises during an entire training run of the binary classifier provided in the first experiment of Sec. 6.

At each iteration, we perform an additional 100 random trials to obtain the statistics of gradient estimates.

In Fig. A2 -(a), we present the 1 norm of the mean of gradient estimates (over 100 trials) versus the number of iterations.

As we can see, both signSGD and ZO-signSGD outperform SGD and ZO-SGD, evidenced by a fast decrease of the 1 norm of gradient estimate.

In Fig. A2-(b) , we present the coordinate-wise gradient noise variance (over 100 trails at each coordinate) against the number of iterations.

It is not surprising that compared to first-order methods, ZO methods suffer gradient noise of larger variance.

In this scenario, we could benefit from ZO-signSGD since taking the sign of gradient estimates might scale down extremely noisy components.

Indeed, we observe a significant decrease of the noise variance while performing ZO-signSGD compared to ZO-SGD.(a) (b) Figure A2 : Statistics of gradient estimates during an entire training run of the binary classifier provided in the first experiment of Sec. 6.

a) The 1 norm of the mean of gradient estimates versus iteration.

b) Coordinate-wise gradient noise variance versus iteration.

The solid line represents the variance averaged over all coordinates, and the shaded region indicates the corresponding standard deviation with respect to all coordinates at each iteration.

Based on the definition of the smoothing function fµ, for any x and y we have DISPLAYFORM0 where the first inequality holds due to Jensen's inequality, and the second inequality holds due to A1.

It is known from (15) that fµ has L-Lipschitz continuous gradient.

By the L-smoothness of fµ, we obtain that DISPLAYFORM1 where (∇fµ(x))i denotes the ith element of ∇fµ(x).Taking expectation for both sides of FORMULA1 , we obtain that DISPLAYFORM2 Similar to BID2 , Theorem 1), we relax Prob [sign(ĝ k,i ) = sign((∇fµ(x k ))i)] by Markov's inequality, DISPLAYFORM3 Substituting FORMULA1 into FORMULA1 , we obtain DISPLAYFORM4 where the second inequality holds due to x 1 ≤ √ d x 2, and the last inequality holds by applying Jensen's inequality to the concave function DISPLAYFORM5 ].

Taking sum of both sides of (19), we then obtain (5).

We recall from (3) thatĝ DISPLAYFORM0 Let zi :=∇f i(xk ) − ∇fµ(x k ) and zi,j =∇fi(x k ; ui,j) − ∇fµ(x k ).

Thus, DISPLAYFORM1 where there are two sources of randomness: a) minibatch sampling i ∈ I k , and b) the random direction sampling u = ui,j.

Note that these two sources of randomness are independent, and the random direction samples {ui,j} are i.i.d..

Next, we discuss two types of mini-batch sampling: a) mini-batch samples without replacement, and b) minibatch samples with replacement.

Suppose that I k is a uniform random subset of [n] (no replacement), motivated by BID21 , Lemma A.1) we introduce a new variable Wi = I(i ∈ I k ), where I is an indicator function, and I(i ∈ I k ) = 1 if i ∈ I k , and 0 otherwise.

As a result, we have DISPLAYFORM2 From (21), the variance ofĝ k is given by DISPLAYFORM3 In FORMULA3 , the equality (a) holds since FORMULA8 , where we have used the fact that Eu[∇fi(x k )] =∇fi,µ(x k ) (Liu et al., 2018c, Lemma.

1) , and recall that fi,µ denotes the smoothing function of fi.

The above implies that DISPLAYFORM4 DISPLAYFORM5 And the equality (b) holds due to Eu[ zi DISPLAYFORM6 On the other hand, suppose that the mini-batch I k contains i.i.d.

samples (namely, with replacement), the vectors {zi} are then i.i.d.

under both mini-batch sampling and random direction sampling.

Therefore, we obtain that DISPLAYFORM7 where the second equality holds since FORMULA3 and FORMULA3 , we obtain that DISPLAYFORM8 DISPLAYFORM9 In ( In FORMULA3 , we next bound DISPLAYFORM10 DISPLAYFORM11 where for ease of notation, let∇fi :=∇f i(xk ; ui,1),∇fi,µ :=∇f i,µ(xk ) and ∇fµ := ∇fµ(x k ).

According to BID26 , Lemma 1), the first term at RHS of (27) yields DISPLAYFORM12 where the last inequality holds due to A2.

Based on the definition of fµ, the second term at RHS of (27) yields DISPLAYFORM13 where we have used the Jensen's inequality and DISPLAYFORM14 Substituting FORMULA3 and FORMULA3 into (27), we have DISPLAYFORM15 where C(d, µ) was defined in (28).We are now ready to bound (26).

Based on In ( DISPLAYFORM16 where the equality (a) holds since Eu[∇fi(x; ui,j)] = ∇fi,µ(x) for any j, given by BID26 , Lemma 1).Substituting FORMULA1 and FORMULA3 into (25) DISPLAYFORM17 4 PROOF OF THEOREM 1Substituting FORMULA14 into FORMULA12 , we obtain It is known from BID26 , Lemma 1) that DISPLAYFORM18 From FORMULA6 , where f * µ = minx fµ(x) and f * = minx f (x).This yields fµ(x0) − f (x0) + f * − f * µ ≤ µ 2 L, and thus DISPLAYFORM19 Substituting FORMULA6 into FORMULA6 , we obtain DISPLAYFORM20 Due to ∇fµ(x k ) 2 ≤ ∇fµ(x k ) 1 and dividing T −1 k=0 δ k for both sides of (37), we obtain that DISPLAYFORM21 where ξ l is finite since E ĝ i,j k − ∇fµ(x k ) 2 2 is upper bounded.

Substituting (44) into (43), we have |(∇fµ(x k )) l | Prob sign(ĝ i,j k,l ) = sign((∇fµ(x k )) l ) ≤ξ l .With the new gradient estimateḡ k = i∈I k q j=1 sign(ĝ i,j k ) in (11), we require to bound Prob [sign(ḡ k,l ) = sign ((∇fµ(x k DISPLAYFORM22 whereḡ k,l is the lth coordinate ofḡ k .We recall thatĝ i,j k,l is an unbiased stochastic approximation to gradient component (∇fµ(x k )) l with variance ξ 7 PROOF OF COROLLARY 3

@highlight

We design and analyze a new zeroth-order stochastic optimization algorithm, ZO-signSGD, and demonstrate its connection and application to black-box adversarial attacks in robust deep learning
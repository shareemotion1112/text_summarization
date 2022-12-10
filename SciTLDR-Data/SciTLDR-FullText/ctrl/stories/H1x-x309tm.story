This paper studies a class of adaptive gradient based momentum algorithms that update the  search directions and learning rates simultaneously using past gradients.

This class, which we refer to as the ''``Adam-type'', includes the popular algorithms such as Adam, AMSGrad, AdaGrad.

Despite their popularity in training deep neural networks (DNNs), the convergence of these algorithms for solving  non-convex problems remains an open question.

In this paper, we develop an analysis framework and a set of mild sufficient conditions that guarantee the convergence of the Adam-type methods, with a convergence rate of order   $O(\log{T}/\sqrt{T})$ for non-convex stochastic optimization.

Our convergence analysis applies to a new algorithm called AdaFom (AdaGrad with First Order Momentum).

We show that the conditions are essential, by identifying concrete examples in which violating the conditions makes an algorithm diverge.

Besides providing one of the first comprehensive analysis for Adam-type methods in the non-convex setting, our results can also help the practitioners to easily  monitor the progress of algorithms and determine their convergence behavior.

First-order optimization has witnessed tremendous progress in the last decade, especially to solve machine learning problems BID3 .

Almost every first-order method obeys the following generic form BID4 ), x t+1 = x t − α t ∆ t , where x t denotes the solution updated at the tth iteration for t = 1, 2, . . .

, T , T is the number of iterations, ∆ t is a certain (approximate) descent direction, and α t > 0 is some learning rate.

The most well-known first-order algorithms are gradient descent (GD) for deterministic optimization (Nesterov, 2013; BID5 and stochastic gradient descent (SGD) for stochastic optimization (Zinkevich, 2003; Ghadimi & Lan, 2013) , where the former determines ∆ t using the full (batch) gradient of an objective function, and the latter uses a simpler but more computationally-efficient stochastic (unbiased) gradient estimate.

Recent works have proposed a variety of accelerated versions of GD and SGD (Nesterov, 2013) .

These achievements fall into three categories: a) momentum methods (Nesterov, 1983; Polyak, 1964; BID10 which carefully design the descent direction ∆ t ; b) adaptive learning rate methods BID1 BID9 Zeiler, 2012; BID7 which determine good learning rates α t , and c) adaptive gradient methods that enjoy dual advantages of a) and b).

In particular, Adam (Kingma & Ba, 2014) , belonging to the third type of methods, has become extremely popular to solve deep learning problems, e.g., to train deep neural networks.

Despite its superior performance in practice, theoretical investigation of Adam-like methods for non-convex optimization is still missing.

Very recently, the work (Reddi et al., 2018) pointed out the convergence issues of Adam even in the convex setting, and proposed AMSGrad, a corrected version of Adam.

Although AMSGrad has made a positive step towards understanding the theoretical behavior of adaptive gradient methods, the convergence analysis of (Reddi et al., 2018) was still very restrictive because it only works for convex problems, despite the fact that the most successful applications are for non-convex problems.

Apparently, there still exists a large gap between theory and practice.

To the best of our knowledge,• (Practicality) The sufficient conditions we derive are simple and easy to check in practice.

They can be used to either certify the convergence of a given algorithm for a class of problem instances, or to track the progress and behavior of a particular realization of an algorithm.• (Tightness and Insight) We show the conditions are essential and "tight", in the sense that violating them can make an algorithm diverge.

Importantly, our conditions provide insights on how oscillation of a so-called "effective stepsize" (that we define later) can affect the convergence rate of the class of algorithms.

We also provide interpretations of the convergence conditions to illustrate why under some circumstances, certain Adam-type algorithms can outperform SGD.

Notations We use z = x/y to denote element-wise division if x and y are both vectors of size d; x y is element-wise product, x 2 is element-wise square if x is a vector, √ x is element-wise square root if x is a vector, (x) j denotes jth coordinate of x, x is x 2 if not otherwise specified.

We use [N ] to denote the set {1, · · · , N }, and use O(·), o(·), Ω(·), ω(·) as standard asymptotic notations.

Stochastic optimization is a popular framework for analyzing algorithms in machine learning due to the popularity of mini-batch gradient evaluation.

We consider the following generic problem where we are minimizing a function f , expressed in the expectation form as follows DISPLAYFORM0 where ξ is a certain random variable representing randomly selected data sample or random noise.

In a generic first-order optimization algorithm, at a given time t we have access to an unbiased noisy gradient g t of f (x), evaluated at the current iterate x t .

The noisy gradient is assumed to be bounded and the noise on the gradient at different time t is assumed to be independent.

An important assumption that we will make throughout this paper is that the function f (x) is continuously differentiable and has Lipschitz continuous gradient, but could otherwise be a non-convex function.

The non-convex assumption represents a major departure from the convexity that has been assumed in recent papers for analyzing Adam-type methods, such as (Kingma & Ba, 2014) and (Reddi et al., 2018) .Our work focuses on the generic form of exponentially weighted stochastic gradient descent method presented in Algorithm 1, for which we name as generalized Adam due to its resemblance to the original Adam algorithm and many of its variants.

DISPLAYFORM1 In Algorithm 1, α t is the step size at time t, β 1,t > 0 is a sequence of problem parameters, m t ∈ R d denotes some (exponentially weighted) gradient estimate, andv t = h t (g 1 , g 2 , ..., g t ) ∈ R d takes all the past gradients as input and returns a vector of dimension d, which is later used to inversely weight the gradient estimate m t .

And note that m t / √v t ∈ R d represents element-wise division.

Throughout the paper, we will refer to the vector α t / √v t as the effective stepsize.

We highlight that Algorithm 1 includes many well-known algorithms as special cases.

We summarize some popular variants of the generalized Adam algorithm in TAB0 .

DISPLAYFORM2 RMSProp N/A Adam * N/A stands for an informal algorithm that was not defined in literature.

We present some interesting findings for the algorithms presented in TAB0 .•

Adam is often regarded as a "momentum version" of AdaGrad, but it is different from AdaFom which is also a momentum version of AdaGrad 1 .

The difference lies in the form of v t .

Intuitively, Adam adds momentum to both the first and second order moment estimate, while in AdaFom we only add momentum to the first moment estimate and use the same second moment estimate as AdaGrad.

These two methods are related in the following way: if we let β 2 = 1 − 1/t in the expression ofv t in Adam, we obtain AdaFom.

We can view AdaFom as a variant of Adam with an increasing sequence of β 2 , or view Adam as a variant of AdaFom with exponentially decaying weights of g assumptions (see Corollary 3.2) , while Adam is shown to possibly diverge (Reddi et al., 2018) .•

The convergence of AMSGrad using a fast diminishing β 1,t such that β 1,t ≤ β 1,t−1 , β 1,t −−−→ t→∞ b, b = 0 in convex optimization was studied in (Reddi et al., 2018) .However, the convergence of the version with constant β 1 or strictly positive b and the version for non-convex setting are unexplored before our work.

We notice that an independent work (Zhou et al., 2018) has also proved the convergence of AMSGrad with constant β 1 .It is also worth mentioning that Algorithm 1 can be applied to solve the popular "finite-sum" problems whose objective is a sum of n individual cost functions.

That is, DISPLAYFORM3 where each f i : R d → R is a smooth and possibly non-convex function.

If at each time instance the index i is chosen uniformly randomly, then Algorithm 1 still applies, with g t = ∇f i (x t ).

It can also be extended to a mini-batch case with g t = 1 b i∈It ∇f i (x t ), where I t denotes the minibatch of size b at time t.

It is easy to show that g t is an unbiased estimator for ∇f (x).In the remainder of this paper, we will analyze Algorithm 1 and provide sufficient conditions under which the algorithm converges to first-order stationary solutions with sublinear rate.

We will also discuss how our results can be applied to special cases of generalized Adam.

The main technical challenge in analyzing the non-convex version of Adam-type algorithms is that the actually used update directions could no longer be unbiased estimates of the true gradients.

Furthermore, an additional difficulty is introduced by the involved form of the adaptive learning rate.

Therefore the biased gradients have to be carefully analyzed together with the use of the inverse of exponential moving average while adjusting the learning rate.

The existing convex analysis (Reddi et al., 2018) does not apply to the non-convex scenario we study for at least two reasons: first, non-convex optimization requires a different convergence criterion, given by stationarity rather than the global optimality; second, we consider constant momentum controlling parameter.

In the following, we formalize the assumptions required in our convergence analysis.

DISPLAYFORM0 It is also lower bounded, i.e. f (x * ) > −∞ where x * is an optimal solution.

A2: At time t, the algorithm can access a bounded noisy gradient and the true gradient is bounded, i.e. ∇f (x t ) ≤ H, g t ≤ H, ∀t > 1.

The noisy gradient is unbiased and the noise is independent, i.e. DISPLAYFORM0 Reference (Reddi et al., 2018 ) uses a similar (but slightly different) assumption as A2, i.e., the bounded elements of the gradient g t ∞ ≤ a for some finite a. The bounded norm of ∇f (x t ) in A2 is equivalent to Lipschitz continuity of f (when f is differentiable) which is a commonly used condition in convergence analysis.

This assumption is often satisfied in practice, for example it holds for the finite sum problem (2) when each f i has bounded gradient, and g t = ∇f i (x t ) where i is sampled randomly.

A3 is also standard in stochastic optimization for analyzing convergence.

Our main result shows that if the coordinate-wise weighting term √v t in Algorithm 1 is properly chosen, we can ensure the global convergence as well as the sublinear convergence rate of the algorithm (to a first-order stationary solution).

First, we characterize how the effective stepsize parameters α t andv t affect convergence of Adam-type algorithms.

Theorem 3.1.

Suppose that Assumptions A1-A3 are satisfied, β 1 is chosen such that β 1 ≥ β 1,t , β 1,t ∈ [0, 1) is non-increasing, and for some constant DISPLAYFORM1 where C 1 , C 2 , C 3 are constants independent of d and T , C 4 is a constant independent of T , the expectation is taken with respect to all the randomness corresponding to {g t }.

DISPLAYFORM2 α t /( √v t ) j denote the minimum possible value of effective stepsize at time t over all possible coordinate and past gradients {g i } t i=1 .

Then the convergence rate of Algorithm 1 is given by DISPLAYFORM3 where s 1 (T ) is defined through the upper bound of RHS of (3), namely, O(s 1 (T )), and DISPLAYFORM4 Proof: See Appendix 6.2.In Theorem 3.1, α t m t / √v t ≤ G is a mild condition.

Roughly speaking, it implies that the change of x t at each each iteration should be finite.

As will be evident later, with g t ≤ H, the condition α t m t / √v t ≤ G is automatically satisfied for both AdaGrad and AMSGrad.

Besides, instead of bounding the minimum norm of ∇f in (4), we can also apply a probabilistic output (e.g., select an output x R with probability p( & Lan, 2013; Lei et al., 2017) .

It is worth mentioning that a small number could be added tov t for ensuring the numerical stability.

In this case, our Theorem 3.1 still holds given the fact the resulting algorithm is still a special case of Algorithm 1.

Accordingly, our convergence results for AMSGrad and AdaFom that will be derived later also hold as α t m t /( √v t + ) ≤ α t m t / √v t ≤ G when is added tô v t .

We will provide a detailed explanation of Theorem 3.1 in Section 3.1.

DISPLAYFORM5 Theorem 3.1 implies a sufficient condition that guarantees convergence of the Adam-type methods: s 1 (T ) grows slower than s 2 (T ).

We will show in Section 3.2 that the rate s 1 (T ) can be dominated by different terms in different cases, i.e. the non-constant quantities Term A and B below DISPLAYFORM6 where the growth of third term at LHS of (5) can be directly related to growth of Term B via the relationship between 1 and 2 norm or upper boundedness of (α t / √v t ) j .

From (4) in Theorem 3.1, it is evident that s 1 (T ) = o(s 2 (T )) can ensure proper convergence of the algorithm.

This requirement has some important implications, which we discuss below.• (The Bounds for s 1 (T ) and s 2 (T )) First, the requirement that DISPLAYFORM0 .

This is a common condition generalized from SGD.

Term A in (5) is a generalization of the term T t=1 a t g t 2 for SGD (where {α t } is the stepsize sequence for SGD), and it quantifies possible increase in the objective function brought by higher order curvature.

The term T t=1 γ t is the lower bound on the summation of effective stepsizes, which reduces to T t=1 α t when Algorithm 1 is simplified to SGD.• (Oscillation of Effective Stepsizes) Term B in (5) characterizes the oscillation of effective stepsizes α t / √v t .

In our analysis such an oscillation term upper bounds the expected possible ascent in objective induced by skewed update direction g t / √v t ("skewed" in the sense that E[g t / √v t ] is not parallel with ∇f (x t )), therefore it cannot be too large.

Bounding this term is critical, and to demonstrate this fact, in Section 3.2.2 we show that large oscillation can result in non-convergence of Adam for even simple unconstrained non-convex problems.• (Advantage of Adaptive Gradient).

One possible benefit of adaptive gradient methods can be seen from Term A. When this term dominates the convergence speed in Theorem 3.1, it is possible that proper design ofv t can help reduce this quantity compared with SGD (An example is provided in Appendix 6.1.1 to further illustrate this fact.) in certain cases.

Intuitively, adaptive gradient methods like AMSGrad can provide a flexible choice of stepsizes, sincev t can have a normalization effect to reduce oscillation and overshoot introduced by large stepsizes.

At the same time, flexibility of stepsizes makes the hyperparameter tuning of an algorithm easier in practice.

In the next, we show our bound FORMULA8 is tight in the sense that there exist problems satisfying Assumption 1 such that certain algorithms belonging to the class of Algorithm 1 can diverge due to the high growth rate of Term A or Term B.

We demonstrate the importance of Term A in this subsection.

Consider a simple one-dimensional optimization problem min x f (x), with f (x) = 100x 2 if |x| <= b, and f (x) = 200b|x| − 100b 2 if |x| > b, where b = 10.

In FIG1 , we show the growth rate of different terms given in Theorem 3.1, where α 0 0, α t = 0.01 for t ≥ 1, and β 1,t = 0, β 2,t = 0.9 for both Adam and AMSGrad.

We observe that both SGD and Adam are not converging to a stationary solution (x = 0), which is because

Next, we use an example to demonstrate the importance of the Term B for the convergence of Adam-type algorithms.

DISPLAYFORM0 and DISPLAYFORM1 It is easy to verify that the only point with ∇f (x) = 0 is x = 0.

The problem satisfies the assumptions in Theorem 3.1 as the stochastic gradient g t = ∇f i (x t ) is sampled uniformly for i ∈ [11].

We now use the AMSGrad and Adam to optimize x, and the results are given in FIG2 , where we set α t = 1, β 1,t = 0, and β 2,t = 0.1.

We observe that FORMULA8 , implying the non-convergence of Adam.

Our theoretical analysis matches the empirical results in FIG2 .

In contrast, AMSGrad converges in FIG2 because of its smaller oscillation in effective stepsizes, associated with Term B. We finally remark that the importance of the quantity DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 is also noticed by (Huang et al., 2018) .

However, they did not analyze its effect on convergence, and their theory is only for convex optimization.

3.3 CONVERGENCE OF AMSGRAD AND ADAFOM Theorem 3.1 provides a general approach for the design of the weighting sequence {v t } and the convergence analysis of Adam-type algorithms.

For example, SGD specified by TAB0 with stepsizes α t = 1/ √ t yields O(log T / √ T ) convergence speed by Theorem 3.1.

Moreover, the explanation on the non-convergence of Adam in (Reddi et al., 2018) is consistent with our analysis in Section 3.2.

That is, Term B in (5) can grow as fast as s 2 (T ) so that s 1 (T )/s 2 (T ) becomes a constant.

Further, we notice that Term A in (5) can also make Adam diverge which is unnoticed before.

Aside from checking convergence of an algorithm, Theorem 3.1 can also provide convergence rates of AdaGrad and AMSGrad, which will be given as corollaries later.

Our proposed convergence rate of AMSGrad matches the result in (Reddi et al., 2018) for stochastic convex optimization.

However, the analysis of AMSGrad in (Reddi et al., 2018 ) is constrained to diminishing momentum controlling parameter β 1,t .

Instead, our analysis is applicable to the more popular constant momentum parameter, leading to a more general non-increasing parameter setting.

In Corollary 3.1 and Corollary 3.2, we derive the convergence rates of AMSGrad (Algorithm 3 in Appendix 6.2.3) and AdaFom (Algorithm 4 in Appendix 6.2.4), respectively.

Note that AdaFom is more general than AdaGrad since when β 1,t = 0, AdaFom becomes AdaGrad.

DISPLAYFORM5 , for AMSGrad (Algorithm 3 in Appendix 6.2.3) with β 1,t ≤ β 1 ∈ [0, 1) and β 1,t is non-increasing, α t = 1/ √ t, we have for any T , DISPLAYFORM6 where Q 1 and Q 2 are two constants independent of T .Proof: See Appendix 6.2.3.

DISPLAYFORM7 , for AdaFom (Algorithm 4 in Appendix 6.2.4) with β 1,t ≤ β 1 ∈ [0, 1) and β 1,t is non-increasing, α t = 1/ √ t, we have for any T , DISPLAYFORM8 where Q 1 and Q 2 are two constants independent of T .Proof: See Appendix 6.2.4.The assumption |(g 1 ) i | ≥ c, ∀i is a mild assumption and it is used to ensurev 1 ≥ r for some constant r.

It is also usually needed in practice for numerical stability (for AMSGrad and AdaGrad, if (g 1 ) i = 0 for some i, division by 0 error may happen at the first iteration).

In some implementations, to avoid numerical instability, the update rule of algorithms like Adam, AMSGrad, and AdaGrad take the form of x t+1 = x t − α t m t /( √v t + ) with being a positive number.

These modified algorithms still fall into the framework of Algorithm 1 since can be incorporated into the definition ofv t .

Meanwhile, our convergence proof for Corollary 3.1 and Corollary 3.2 can go through without assuming |(g 1 ) i | ≥ c, ∀i because √v t ≥ .

In addition, can affect the worst case convergence rate by a constant factor in the analysis.

We remark that the derived convergence rate of AMSGrad and AdaFom involves an additional log T factor compared to the fastest rate of first order methods (1/ √ T ).

However, such a slowdown can be mitigated by choosing an appropriate stepsize.

To be specific, the log T factor for AMSGrad would be eliminated when we adopt a constant rather than diminishing stepsize, e.g., α t = 1/ √ T .

It is also worth mentioning that our theoretical analysis focuses on the convergence rate of adaptive methods in the worst case for nonconvex optimization.

Thus, a sharper convergence analysis that can quantify the benefits of adaptive methods still remains an open question in theory.

In this section, we compare the empirical performance of Adam-type algorithms, including AMSGrad, Adam, AdaFom and AdaGrad, on training two convolutional neural networks (CNNs).

In the first example, we train a CNN of 3 convolutional layers and 2 fully-connected layers on MNIST.

In the second example, we train a CIFARNET on CIFAR-10.

We refer readers to Appendix 6.1.2 for more details on the network model and the parameter setting.

In Figure 3 , we present the training loss and the classification accuracy of Adam-type algorithms versus the number of iterations.

As we can see, AMSGrad performs quite similarly to Adam which confirms the result in (Reddi et al., 2018) .

The performance of AdaGrad is worse than other algorithms, because of the lack of momentum and/or the significantly different choice ofv t .

We also observe that the performance of AdaFom lies between AMSGrad/Adam and AdaGrad.

This is not surprising, since AdaFom can be regarded as a momentum version of AdaGrad but uses a simpler adaptive learning rate (independent on β 2 ) compared to AMSGrad/Adam.

In Figure 4 , we consider to train a larger network (CIFARNET) on CIFAR-10.

As we can see, Adam and AMSGrad perform similarly and yield the best accuracy.

AdaFom outperforms AdaGrad in both training and testing, which agrees with the results obtained in the MNIST experiment.

We provided some mild conditions to ensure convergence of a class of Adam-type algorithms, which includes Adam, AMSGrad, AdaGrad, AdaFom, SGD, SGD with momentum as special cases.

Apart from providing general convergence guarantees for algorithms, our conditions can also be checked in practice to monitor empirical convergence.

To the best of our knowledge, the convergence of Adam-type algorithm for non-convex problems was unknown before.

We also provide insights on how oscillation of effective stepsizes can affect convergence rate for the class of algorithms which could be beneficial for the design of future algorithms.

This paper focuses on unconstrained non-convex optimization problems, and one future direction is to study a more general setting of constrained non-convex optimization.

Momentum methods take into account the history of first-order information (Nesterov, 2013; 1983; Nemirovskii et al., 1983; Ghadimi & Lan, 2016; Polyak, 1964; BID10 Ochs et al., 2015; Yang et al., 2016; Johnson & Zhang, 2013; Reddi et al., 2016; Lei et al., 2017) .

A well-known method, called Nesterov's accelerated gradient (NAG) originally designed for convex deterministic optimization (Nesterov, 2013; 1983; Nemirovskii et al., 1983) , constructs the descent direction ∆ t using the difference between the current iterate and the previous iterate.

A recent work (Ghadimi & Lan, 2016 ) studied a generalization of NAG for non-convex stochastic programming.

Similar in spirit to NAG, heavy-ball (HB) methods (Polyak, 1964; BID10 Ochs et al., 2015; Yang et al., 2016) form the descent direction vector through a decaying sum of the previous gradient information.

In addition to NAG and HB methods, stochastic variance reduced gradient (SVRG) methods integrate SGD with GD to acquire a hybrid descent direction of reduced variance (Johnson & Zhang, 2013; Reddi et al., 2016; Lei et al., 2017) .

Recently, certain accelerated version of perturbed gradient descent (PAGD) algorithm is also proposed in (Jin et al., 2017) , which shows the fastest convergence rate among all Hessian free algorithms.

Adaptive learning rate methods accelerate ordinary SGD by using knowledge of the past gradients or second-order information into the current learning rate α t BID1 BID9 Zeiler, 2012; BID7 .

In BID1 , the diagonal elements of the Hessian matrix were used to penalize a constant learning rate.

However, acquiring the second-order information is computationally prohibitive.

More recently, an adaptive subgradient method (i.e., AdaGrad) penalized the current gradient by dividing the square root of averaging of the squared gradient coordinates in earlier iterations BID9 .

Although AdaGrad works well when gradients are sparse, its convergence is only analyzed in the convex world.

Other adaptive learning rate methods include Adadelta (Zeiler, 2012) and ESGD BID7 , which lacked theoretical investigation although some convergence improvement was shown in practice.

Adaptive gradient methods update the descent direction and the learning rate simultaneously using knowledge in the past, and thus enjoy dual advantages of momentum and adaptive learning rate methods.

Algorithms of this family include RMSProp (Tieleman & Hinton, 2012), Nadam BID8 , and Adam (Kingma & Ba, 2014) .

Among these, Adam has become the most widely-used method to train deep neural networks (DNNs).

Specifically, Adam adopts exponential moving averages (with decaying/forgetting factors) of the past gradients to update the descent direction.

It also uses inverse of exponential moving average of squared past gradients to adjust the learning rate.

The work (Kingma & Ba, 2014) showed Adam converges with at most O(1/ √ T ) rate for convex problems.

However, the recent work (Reddi et al., 2018) pointed out the convergence issues of Adam even in the convex setting, and proposed a modified version of Adam (i.e., AMSGrad), which utilizes a non-increasing quadratic normalization and avoids the pitfalls of Adam.

Although AMSGrad has made a significant progress toward understanding the theoretical behavior of adaptive gradient methods, the convergence analysis of (Reddi et al., 2018) only works for convex problems.

In this section, we provide some additional experiments to demonstrate how specific Adam-type algorithms can perform better than SGD and how SGD can out perform Adam-type algorithms in different situations.

One possible benefit of adaptive gradient methods is the "sparse noise reduction" effect pointed out in BID2 .

Below we illustrate another possible practical advantage of adaptive gradient methods when applied to solve non-convex problems, which we refer to as flexibility of stepsizes.

To highlight ideas, let us take AMSGrad as an example, and compare it with SGD.

First, in nonconvex problems there can be multiple valleys with different curvatures.

When using fixed stepsizes (or even a slowly diminishing stepsize), SGD can only converge to local optima in valleys with small curvature while AMSGrad and some other adaptive gradient algorithms can potentially converge to optima in valleys with relative high curvature (this may not be beneficial if one don't want to converge to a sharp local minimum).

Second, the flexible choice of stepsizes implies less hyperparameter tuning and this coincides with the popular impression about original Adam.

We empirically demonstrate the flexible stepsizes property of AMSGrad using a deterministic quadratic problem.

Consider a toy optimization problem min x f (x), f (x) = 100x 2 , the gradient is given by 200x.

For SGD (which reduces to gradient descent in this case) to converge, we must have α t < 0.01; for AMSGrad,v t has a strong normalization effect and it allows the algorithm to use larger α t '

s. We show the growth rate of different terms given in Theorem 3.1 for different stepsizes in FIG1 to Figure A4 (where we choose β 1,t = 0, β 2,t = 0.9 for both Adam and AMSGrad).

In FIG1 , α t = 0.1 and SGD diverges due to large α t , AMSGrad converges in this case, Adam is oscillating between two non-zero points.

In FIG2 , stepsizes α t is set to 0.01, SGD and Adam are oscillating, AMSGrad converges to 0.

For Figure A3 , SGD converges to 0 and AMSGrad is converging slower than SGD due to its smaller effective stepsizes, Adam is oscillating.

One may wonder how diminishing stepsizes affects performance of the algorithms, this is shown in Figure A4 where α t = 0.1/ √ t, we can see SGD is diverging until stepsizes is small, AMSGrad is converging all the time, Adam appears to get stuck but it is actually converging very slowly due to diminishing stepsizes.

This example shows AMSGrad can converge with a larger range of stepsizes compared with SGD.From the figures, we can see that the term T t=1 α t g t / √v t 2 is the key quantity that limits the convergence speed of algorithms in this case.

In FIG1 , FIG2 , and early stage of Figure A4 , the quantity is obviously a good sign of convergence speed.

In Figure A3 , since the difference of quantity between AMSGrad and SGD is compensated by the larger effective stepsizes of SGD and some problem independent constant, SGD converges faster.

In fact, Figure A3 provides a case where AMSGrad does not perform well.

Note that the normalization factor √v t can be understood as imitating the largest Lipschitz constant along the way of optimization, so generally speaking dividing by this number makes the algorithm converge easier.

However when the Lipschitz constant becomes smaller locally around a local optimal point, the stepsizes choice of AMSGrad dictates that √v t does not change, resulting a small effective stepsizes.

This could be mitigated by AdaGrad and its momentum variants which allowsv t to decrease when g t keeps decreasing.

FIG1 : Comparison of algorithms with α t = 0.1, we defined α 0 = 0 FIG2 : Comparison of algorithms with α t = 0.01, we defined α 0 = 0 Figure A3 : Comparison of algorithms with α t = 0.001, we defined α 0 = 0 Figure A4 : Comparison of algorithms with α t = 0.1/ √ t, we defined α 0 = 0 DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3

In the experiment on MNIST, we consider a convolutional neural network (CNN), which includes 3 convolutional layers and 2 fully-connected layers.

In convolutional layers, we adopt filters of sizes 6 × 6 × 1 (with stride 1), 5 × 5 × 6 (with stride 2), and 6 × 6 × 12 (with stride 2), respectively.

In both AMSGrad 2 and Adam, we set β 1 = 0.9 and β 2 = 0.99.

In AdaFom, we set β 1 = 0.9.

We choose 50 as the mini-batch size and the stepsize is choose to be α t = 0.0001 + 0.003e −t/2000 .The architecture of the CIFARNET that we are using is as below.

The model starts with two convolutional layers with 32 and 64 kernels of size 3 x 3, followed by 2 x 2 max pooling and dropout with keep probability 0.25.

The next layers are two convolutional layers with 128 kernels of size 3 x 3 and 2 x 2, respectively.

Each of the two convolutional layers is followed by a 2 x 2 max pooling layer.

The last layer is a fully connected layer with 1500 nodes.

Dropout with keep probability 0.25 is added between the fully connected layer and the convolutional layer.

All convolutional layers use ReLU activation and stride 1.

The learning rate α t of Adam and AMSGrad starts with 0.001 and decrease 10 times every 20 epochs.

The learning rate of AdaGrad and AdaFom starts with 0.05 and decreases to 0.001 after 20 epochs and to 0.0001 after 40 epochs.

These learning rates are tuned so that each algorithm has its best performance.

In this section, we present the convergence proof of Algorithm 1.

We will first give several lemmas prior to proving Theorem 3.1.

Lemma 6.1.

Let x 0 x 1 in Algorithm 1, consider the sequence DISPLAYFORM0 Then the following holds true DISPLAYFORM1 Proof.[Proof of Lemma 6.1]

By the update rules S1-S3 in Algorithm 1, we have when t > 1, DISPLAYFORM2 Since DISPLAYFORM3 Divide both sides by 1 − β 1,t , we have DISPLAYFORM4 Define the sequence DISPLAYFORM5 Then FORMULA0 can be written as DISPLAYFORM6 where the second equality is due to x t+1 − x t = −α t m t / √v t .

For t = 1, we have z 1 = x 1 (due to x 1 = x 0 ), and DISPLAYFORM7 where the forth equality holds due to (S1) and (S3) of Algorithm 1.The proof is now complete.

Without loss of generality, we initialize Algorithm 1 as below to simplify our analysis in what follows, DISPLAYFORM0 Lemma 6.2.

Suppose that the conditions in Theorem 3.1 hold, then DISPLAYFORM1 where DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 Proof.[Proof of Lemma 6.2] By the Lipschitz smoothness of ∇f , we obtain DISPLAYFORM8 where d t = z t+1 − z t , and Lemma 6.1 together with (12) yield DISPLAYFORM9 Based on FORMULA3 and FORMULA0 , we then have DISPLAYFORM10 where {T i } have been defined in FORMULA0 - FORMULA0 .

Further, using inequality a + b + c 2 ≤ 3 a 2 + 3 b 2 + 3 c 2 and FORMULA3 , we have DISPLAYFORM11 Substituting the above inequality into FORMULA3 , we then obtain (13).

The next series of lemmas separately bound the terms on RHS of (13).

Lemma 6.3.

Suppose that the conditions in Theorem 3.1 hold, T 1 in (14) can be bounded as DISPLAYFORM0 [Proof of Lemma 6.3]

Since g t ≤ H, by the update rule of m t , we have m t ≤ H, this can be proved by induction as below.

Recall that m t = β 1,t m t−1 + (1 − β 1,t )g t , suppose m t−1 ≤ H, we have DISPLAYFORM1 then since m 0 = 0, we have m 0 ≤ H which completes the induction.

Given m t ≤ H, we further have DISPLAYFORM2 where the first equality holds due to (12), and the last inequality is due to β 1 ≥ β 1,i .The proof is now complete.

DISPLAYFORM3 Proof.[Proof of Lemma 6.6] DISPLAYFORM4 where the fist inequality is due to β 1 ≥ β 1,t and (12)

, the second inequality is due to m i < H.This completes the proof.

Q.E.D. Lemma 6.7.

Suppose the assumptions in Theorem 3.1 hold.

For T 2 in (15), we have DISPLAYFORM5 Proof.[Proof of Lemma 6.7]

Recall from the definition (9), we have DISPLAYFORM6 Further we have z 1 = x 1 by definition of z 1 .

We have DISPLAYFORM7 The second term of (26) can be bounded as DISPLAYFORM8 where the first inequality is because a, b ≤ 1 2 a 2 + b 2 and the fact that z 1 = x 1 , the second inequality is because DISPLAYFORM9 and T 7 is defined as DISPLAYFORM10 We next bound the T 7 in (28), by update rule DISPLAYFORM11 Based on that, we obtain DISPLAYFORM12 where the first inequality is due to β 1,t ≤ β 1 , the second equality is by substituting expression of m t , the last inequality is because (a + b) 2 ≤ 2( a 2 + b 2 ), and we have introduced T 8 and T 9 for ease of notation.

In (29), we first bound T 8 as below DISPLAYFORM13 where (i) is due to ab < 1 2 (a 2 + b 2 ) and follows from β 1,t ≤ β 1 and β 1,t ∈ [0, 1), (ii) is due to symmetry of p and k in the summation, (iii) is because of DISPLAYFORM14 is exchanging order of summation, and the second-last inequality is due to the similar reason as (iii).For the T 9 in (29), we have DISPLAYFORM15 where the first inequality holds due to β 1,k < 1 and |(g k ) j | ≤ H, the second inequality holds due to β 1,k ≤ β 1 , and the last inequality applied the triangle inequality.

For RHS of (31), using Lemma 6.8(that will be proved later) with DISPLAYFORM16 , we further have DISPLAYFORM17 Based on (27), (29), (30) and (32), we can then bound the second term of (26) as DISPLAYFORM18 Let us turn to the first term in (26).

Reparameterize g t as g t = ∇f (x t ) + δ t with E[δ t ] = 0, we have DISPLAYFORM19 It can be seen that the first term in RHS of (34) is the desired descent quantity, the second term is a bias term to be bounded.

For the second term in RHS of (34), we have DISPLAYFORM20 where the last equation is because given x i ,v i−1 , E δ i (1/ v i−1 )|x i ,v i−1 = 0 and δ i ≤ 2H due to g i ≤ H and ∇f (x i ) ≤ H based on Assumptions A2 and A3.

Further, we have DISPLAYFORM21 Substituting FORMULA11 and FORMULA13 into FORMULA8 , we then bound the first term of FORMULA3 as DISPLAYFORM22 We finally apply (37) and (33) to obtain (24).

The proof is now complete.

Q.E.D. Q.E.D.

Proof.[Proof of Theorem 3.1]

We combine Lemma 6.2, Lemma 6.3, Lemma 6.4, Lemma 6.5, Lemma 6.6, and Lemma 6.7 to bound the overall expected descent of the objective.

First, from Lemma 6.2, we have DISPLAYFORM0 ≤ 1 − log a 1 + log DISPLAYFORM1 Proof.[Proof of Lemma 6.9]

We will prove it by induction.

Suppose DISPLAYFORM2 we have DISPLAYFORM3 + 1 − log a 1 + log DISPLAYFORM4 Applying the definition of concavity to log(x), with f (z) log(z), we have f (z) ≤ f (z 0 ) + f (z 0 )(z − z 0 ), then substitute z = x − b, z 0 = x, we have f (x − b) ≤ f (x) + f (x)(−b) which is equivalent to log(x) ≥ log(x − b) + b/x for b < x, using x = T i=1 a i , b = a T , we have DISPLAYFORM5 Now it remains to check first iteration.

We have a 1 a 1 = 1 ≤ 1 − log(a 1 ) + log(a 1 ) = 1This completes the proof.

Q.E.D.

<|TLDR|>

@highlight

We analyze convergence of Adam-type algorithms and provide mild sufficient conditions to guarantee their convergence, we also show  violating the conditions can makes an algorithm diverge.

@highlight

Presents a convergence analysis in the non-convex setting for a family of optimization algorithms.

@highlight

This paper investigates the convergence condition of Adam-type optimizers in the unconstrained non-convex optimization problems.
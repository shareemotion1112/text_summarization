Adaptive gradient algorithms perform gradient-based updates using the history of gradients and are ubiquitous in training deep neural networks.

While adaptive gradient methods theory is well understood for minimization problems, the underlying factors driving their empirical success in min-max problems such as GANs remain unclear.

In this paper, we aim at bridging  this gap from both theoretical and empirical perspectives.

First, we analyze a variant of Optimistic Stochastic Gradient (OSG) proposed in~\citep{daskalakis2017training} for solving a class of non-convex non-concave min-max problem and establish $O(\epsilon^{-4})$ complexity for finding $\epsilon$-first-order stationary point, in which the algorithm only requires invoking one stochastic first-order oracle while enjoying state-of-the-art iteration complexity achieved by stochastic extragradient method by~\citep{iusem2017extragradient}.

Then we propose an adaptive variant of OSG named Optimistic Adagrad (OAdagrad) and reveal an \emph{improved} adaptive complexity $\widetilde{O}\left(\epsilon^{-\frac{2}{1-\alpha}}\right)$~\footnote{Here $\widetilde{O}(\cdot)$ compresses a logarithmic factor of $\epsilon$.}, where $\alpha$ characterizes the growth rate of the cumulative stochastic gradient and $0\leq \alpha\leq 1/2$. To the best of our knowledge, this is the first work for establishing adaptive complexity in non-convex non-concave min-max optimization.

Empirically, our experiments show that indeed adaptive gradient algorithms outperform their non-adaptive counterparts in GAN training.

Moreover, this observation can be explained by the slow growth rate of the cumulative stochastic gradient, as observed empirically.

Adaptive gradient algorithms (Duchi et al., 2011; Tieleman & Hinton, 2012; Kingma & Ba, 2014; Reddi et al., 2019) are very popular in training deep neural networks due to their computational efficiency and minimal need for hyper-parameter tuning (Kingma & Ba, 2014) .

For example, Adagrad (Duchi et al., 2011) automatically adjusts the learning rate for each dimension of the model parameter according to the information of history gradients, while its computational cost is almost the same as Stochastic Gradient Descent (SGD).

However, in supervised deep learning (for example, image classification tasks using a deep convolutional neural network), there is not enough evidence showing that adaptive gradient methods converge faster than its non-adaptive counterpart (i.e., SGD) on benchmark datasets.

For example, it is argued in (Wilson et al., 2017 ) that adaptive gradient methods often find a solution with worse performance than SGD.

Specifically, Wilson et al. (2017) observed that Adagrad has slower convergence than SGD in terms of both training and testing error, while using VGG (Simonyan & Zisserman, 2014) on CIFAR10 data.

GANs (Goodfellow et al., 2014) are a popular class of generative models.

In a nutshell, they consist of a generator and a discriminator, both of which are defined by deep neural networks.

The generator and the discriminator are trained under an adversarial cost, corresponding to a non-convex non-concave min-max problem.

GANs are known to be notoriously difficult to train.

In practice, Adam (Kingma & Ba, 2014 ) is the defacto optimizer used for GAN training.

The common optimization strategy is to alternatively update the discriminator and the generator (Arjovsky et al., 2017; Gulrajani et al., 2017) .

Using Adam is important in GAN training, since replacing it with non-adaptive methods (e.g. SGD) would significantly deteriorate the performance.

This paper studies and attempts to answer the following question:

We analyze a variant of Optimistic Stochastic Gradient (OSG) in (Daskalakis & Panageas, 2018) and propose an adaptive variant named Optimistic Adagrad (OAdagrad) for solving a class of nonconvex non-concave min-max problems.

Both of them are shown to enjoy state-of-the-art complexities.

We further prove that the convergence rate of OAdagrad to an -first-order stationary point depends on the growth rate of the cumulative stochastic gradient.

In our experiments, we observed an interesting phenomenon while using adaptive gradient methods for training GANs: the cumulative stochastic gradient grows at a slow rate.

This observation is in line with the prediction of our theory suggesting improved convergence rate for OAdagrad in GAN training, when the growth rate of the cumulative stochastic gradient is slow.

Since GAN is a min-max optimization problem in nature, our problem of interest is to solve the following stochastic optimization problem:

where U, V are closed and convex sets, F (u, v) is possibly non-convex in u and non-concave in v. ξ is a random variable following an unknown distribution D. In GAN training, u, v represent the parameters of generator and discriminator respectively.

The ideal goal for solving (1) is to find a saddle point (u * , v * ) ∈ U × V such that F (u * , v) ≤ F (u * , v * ) ≤ F (u, v * ) for ∀u ∈ U, ∀v ∈ V.

To achieve this goal, the typical assumption usually made is that the objective function is convexconcave.

When F (u, v) is convex in u and concave in v, non-asymptotic guarantee in terms of the duality gap is well established by a series of work (Nemirovski & Yudin, 1978; Nemirovski, 2004; Nesterov, 2007; Nemirovski et al., 2009; Juditsky et al., 2011) .

However, when F (u, v) is non-convex in u and non-concave in v, finding the saddle point is NP-hard in general.

Instead, we focus on finding the first-order stationary point provided that the objective function is smooth.

I.e. we aim to find (u, v) ∈ U × V such that ∇ u F (u, v) = 0, ∇ v F (u, v) = 0.

Note that this is a necessary condition for finding the (local) saddle point.

Related Work.

Several works designed iterative first-order deterministic (Dang & Lan, 2015) and stochastic (Iusem et al., 2017; Lin et al., 2018) algorithms for achieving the -first-order stationary point with non-asymptotic guarantee.

The goal is to find x such that T (x) ≤ or (u, v) and the first-order stochastic oracle is the noisy observation of Dang & Lan (2015) focuses on the deterministic setting.

On the other hand, (Iusem et al., 2017 ) develops a stochastic extra-gradient algorithm that enjoys O( −4 ) iteration complexity.

The extra-gradient method requires two stochastic first-order oracles in one iteration, which can be computationally expensive in deep learning applications such as GANs.

The inexact proximal point method developed in (Lin et al., 2018) has iteration complexity O( −6 ) for finding an -first-order stationary point 2 .

To avoid the cost of an additional oracle call in extragradient step, several studies (Chiang et al., 2012; Rakhlin & Sridharan, 2013; Daskalakis et al., 2017; Gidel et al., 2018; Xu et al., 2019) proposed single-call variants of the extragradient algorithm.

Some of them focus on the convex setting (e.g. (Chiang et al., 2012; Rakhlin & Sridharan, 2013) ), while others focus on the non-convex setting (Xu et al., 2019) .

The closest to our work is the work by (Daskalakis et al., 2017; Gidel et al., 2018) , where the min-max setting and GAN training are considered.

However, the convergence of those algorithms is only shown for a class of bilinear problems in (Daskalakis et al., 2017) and for monotone variational inequalities in (Gidel et al., 2018) .

Hence a big gap remains between the specific settings studied in (Daskalakis et al., 2017; Gidel et al., 2018) (Chavdarova et al., 2019) strong-monotonicity finite sum stochastic finite sum

Extragradient (Azizian et al., 2019) strong-monotonicity

, or -optim ( -close to the set of optimal solution).

T g stands for the time complexity for invoking one stochastic first-order oracle.

non-concave min-max problems.

Table 1 provides a complete overview of our results and existing results.

It is hard to give justice to the large body of work on min-max optimization, so we refer the interested reader to Appendix B that gives a comprehensive survey of related previous methods that are not covered in this Table.

Our main goal is to design stochastic first-order algorithms with low iteration complexity, low periteration cost and suitable for a general class of non-convex non-concave min-max problems.

The main tool we use in our analysis is variational inequality.

be an operator and X ⊂ R d is a closed convex set.

The Stampacchia Variational Inequality (SVI) problem (Hartman & Stampacchia, 1966) is defined by the operator T and X and denoted by SVI(T, X ).

It consists of finding x * ∈ X such that T (x * ), x − x * ≥ 0 for ∀x ∈ X .

A similar one is Minty Variational Inequality (MVI) problem (Minty et al., 1962) denoted by MVI(T, X ), which consists of finding x * such that T (x), x − x * ≥ 0 for ∀x ∈ X .

Min-max optimization is closely related to variational inequalities.

The corresponding SVI and MVI for the min-max problem are defined through

Our main contributions are summarized as follows:

• Following (Daskalakis et al., 2017) , we extend optimistic stochastic gradient (OSG) analysis beyond the bilinear and unconstrained case, by assuming the Lipschitz continuity of the operator T and the existence of a solution for the variational inequality MVI(T, X ).

These conditions were considered in the analysis of the stochastic extragradient algorithm in (Iusem et al., 2017) .

We analyze a variant of Optimistic Stochastic Gradient (OSG) under these conditions, inspired by the analysis of (Iusem et al., 2017) .

We show that OSG achieves state-of-the-art iteration complexity O(1/ 4 ) for finding an -first-order stationary point.

Note that our OSG variant only requires invoking one stochastic first-order oracle while enjoying the state-of-the-art iteration complexity achieved by stochastic extragradient method (Iusem et al., 2017) .

• Under the same conditions, we design an adaptive gradient algorithm named Optimistic Adagrad (OAdagrad), and show that it enjoys better adaptive complexity O

where α characterizes the growth rate of cumulative stochastic gradient and 0 ≤ α ≤ 1/2.

Similar to Adagrad (Duchi et al., 2011) , our main innovation is in considering variable metrics according to the geometry of the data in order to achieve potentially faster convergence rate for a class of nonconvex-nonconcave min-max games.

Note that this adaptive complexity improves upon the non-adaptive one (i.e. O(1/ 4 )) achieved by OSG.

To the best of our knowledge, we establish the first known adaptive complexity for adaptive gradient algorithms in a class of non-convex non-concave min-max problems.

• We demonstrate the effectiveness of our algorithms in GAN training on CIFAR10 data.

Empirical results identify an important reason behind why adaptive gradient methods behave well in GANs, which is due to the fact that the cumulative stochastic gradient grows in a slow rate.

We also show that OAdagrad outperforms Simultaneous Adam in sample quality in ImageNet generation using self-attention GANs (Zhang et al., 2018) .

This confirms the superiority of OAdagrad in min-max optimization.

In this section, we fix some notations and give formal definitions of variational inequalities, and their relationship to the min-max problem (1).

Notations.

Let X ⊂ R d be a closed convex set, and · the euclidean norm.

We note Π X the projec-

.

At every point x ∈ X , we don't have access to T (x) and have only access to a noisy observations of T (x).

That is, T (x; ξ), where ξ is a random variable with distribution D. For the ease of presentation, we use the terms stochastic gradient and stochastic first-order oracle interchangeably to stand for T (x; ξ) in the min-max setting.

Definition 1 (Monotonicity).

An operator T is monotone if T (x)−T (y), x−y ≥ 0 for ∀x, y ∈ X .

An operator T is pseudo-monotone if

We give here formal definitions of monotonic operators T and the -first-order stationary point.

Definition 2 ( -First-Order Stationary Point).

A point x ∈ X is called -first-order stationary point if T (x) ≤ .

We make the following observations:

(a).

From the definition, it is evident that strong-monotonicity ⇒ monotonicity ⇒pseudo-monotonicity.

Assuming SVI has a solution and pseudo-monotonicity of the operator T imply that MVI(T, X ) has a solution.

To see that, assume that SVI has a nonempty solution set, i.e. there exists x * such that T (x * ), y − x * ≥ 0 for any y. Noting that pseudomonotonicity means that for every y, x, T (x), y − x ≥ 0 implies T (y), y − x ≥ 0, we have T (y), y − x * ≥ 0 for any y, which means that x * is the solution of Minty variational inequality.

Note that the reverse may not be true and an example is provided in Appendix G.

(b).

For the min-max problem (1), when F (u, v) is convex in u and concave in v, T is monotone.

And, therefore solving SVI(T, X ) is equivalent to solving (1).

When T is not monotone, by assuming T is Lipschitz continuous, it can be shown that the solution set of (1) is a subset of the solution set of SVI(T, X ).

However, even solving SVI(T, X ) is NP-hard in general and hence we resort to finding an -first-order stationary point.

Throughout the paper, we make the following assumption:

(ii).

MVI(T, X ) has a solution, i.e. there exists

Remark: Assumptions (i) and (iii) are commonly used assumptions in the literature of variational inequalities and non-convex optimization (Juditsky et al., 2011; Ghadimi & Lan, 2013; Iusem et al., 2017) .

Assumption (ii) is used frequently in previous work focusing on analyzing algorithms that solve non-monotone variational inequalities (Iusem et al., 2017; Lin et al., 2018; Mertikopoulos et al., 2018) .

Assumption (ii) is weaker than other assumptions usually considered, such as pseudomonotonicity, monotonicity, or coherence as assumed in (Mertikopoulos et al., 2018) .

For nonconvex minimization problem, it has been shown that this assumption holds while using SGD to learn neural networks (Li & Yuan, 2017; Kleinberg et al., 2018; Zhou et al., 2019) .

This section serves as a warm-up and motivation of our main theoretical contribution presented in the next section.

Inspired by (Iusem et al., 2017) , we present an algorithm called Optimistic Stochastic Gradient (OSG) that saves the cost of the additional oracle call as required in (Iusem et al., 2017) and maintains the same iteration complexity.

The main algorithm is described in Algorithm 1, where m t denotes the minibatch size for estimating the first-order oracle.

It is worth mentioning that Algorithm 1 becomes stochastic extragradient method if one changes

.

Stochastic extragradient method requires to compute stochastic gradient over both sequences {x k } and {z k }.

In contrast, {x k } is an ancillary sequence in OSG and the stochastic gradient is only computed over the sequence of {z k }.

Thus, stochastic extragradient method is twice as expensive as OSG in each iteration.

In some tasks (e.g. training GANs) where the stochastic gradient computation is expensive, OSG is numerically more appealing.

4:

Remark: When X = R d , the update in Algorithm 1 becomes the algorithm in (Daskalakis et al., 2017) , i.e.

The detailed derivation of (2) can be found in Appendix F.

and run Algorithm 1 for N iterations.

Then we have

Corollary 1.

Consider the unconstrained case where X = R d .

Let η ≤ 1/9L, and we have

Remark: There are two implications of Corollary 1.

•

, and the total complexity is

, where O(·) hides a logarithmic factor of .

•

2 ≤ 2 , the total number of iterations is N = O( −2 ), and the total complexity is

Before introducing Optimistic Adagrad, we present here a quick overview of Adagrad (Duchi et al., 2011) .

The main objective in Adagrad is to solve the following minimization problem:

where w is the model parameter, and ζ is an random variable following distribution P. The update rule of Adagrad is

where

Our second algorithm named Optimistic Adagrad (OAdagrad) is an adaptive variant of OSG, which also updates minimization variable and maximization variable simultaneously.

The key difference between OSG and OAdagrad is that OAdagrad inherits ideas from Adagrad to construct variable metric based on history gradients information, while OSG only utilizes a fixed metric.

This difference helps us establish faster adaptive convergence under some mild assumptions.

Note that in OAdagrad we only consider the unconstrained case, i.e. X = R d .

(ii).

There exists a universal constant D > 0 such that x k 2 ≤ D for k = 1, . . .

, N , and

Remark: Assumption 2 (i) is a standard one often made in literature (Duchi et al., 2011) .

Assumption 2 (ii) holds when we use normalization layers in the discriminator and generator such as spectral normalization of weights (Miyato et al., 2018; Zhang et al., 2018) , that will keep the norms of the weights bounded.

Regularization techniques such as weight decay also ensure that the weights of the networks remain bounded throughout the training.

Hx .

Denote g 0:k by the concatenation of g 0 , . . . , g k , and denote g 0:k,i by the i-th row of g 0:k .

4:

Theorem 2.

Suppose Assumption 1 and 2 hold.

Suppose g 1:k,i 2 ≤ δk α with 0 ≤ α ≤ 1/2 for every i = 1, . . .

, d and every k = 1, . . .

, N .

When η ≤ δ 9L , after running Algorithm 2 for N iterations, we have

(6) To make sure

, where O(·) hides a logarithmic factor of .

• We denote g 1:k by the cumulative stochastic gradient, where g 1:k,i 2 ≤ δk α characterizes the growth rate of the gradient in terms of i-th coordinate.

In our proof, a key quantity is d i=1 g 1:k,i 2 that crucially affects the computational complexity of Algorithm 2.

Since

.

But in practice, the stochastic gradient is usually sparse, and hence α can be strictly smaller than 1 2 .

• As shown in Theorem 2, the minibatch size used in Algorithm 2 for estimating the firstorder oracle can be any positive constant and independent of .

This is more practical than the results established in Theorem 1, since the minibatch size in Theorem 1 does either increase in terms of number of iterations or is dependent on .

When α = 1 2 , the complexity of Algorithm 2 is O(1/ 4 ), which matches the complexity stated in Theorem 1.

When α < 1 2 , the complexity of OAdagrad given in Algorithm 2 is O − 2 1−α , i.e., strictly better than that of OSG given in Algorithm 1.

Alternating Adam is very popular in GAN training (Goodfellow et al., 2014; Arjovsky et al., 2017; Gulrajani et al., 2017; Brock et al., 2018) .

In Alternating Adam, one alternates between multiple steps of Adam on the discriminator and a single step of Adam on the generator.

The key difference between OAdagrad and Alternating Adam is that OAdagrad updates the discriminator and generator simultaneously.

It is worth mentioning that OAdagrad naturally fits into the framework of Optimistic Adam proposed in (Daskalakis et al., 2017) .

Taking β 1 = 0, β 2 → 1 in their Algorithm 1 reduces to OAdagrad with annealing learning rate.

To the best of our knowledge, there is no convergence proof for Alternating Adam for non-convex non-concave problems.

Our convergence proof for OAdagrad provides a theoretical justification of a special case of Optimistic Adam.

WGAN-GP on CIFAR10 In the first experiment, we verify the effectiveness of the proposed algorithms in GAN training using the PyTorch framework (Paszke et al., 2017) .

We use Wasserstein GAN with gradient penalty (WGAN-GP) (Gulrajani et al., 2017) and CIFAR10 data in our experiments.

The architectures of discriminator and generator, and the penalty parameter in WGAN-GP are set to be same as in the original paper.

We compare Alternating Adam, OSG and OAdagrad, where the Alternating Adam is to run 5 steps of Adam on the discriminator before performing 1 step of Adam on the generator.

We try different batch sizes (64, 128, 256) for each algorithm.

For each algorithm, we tune the learning rate in the range of {1×10 when using batch size 64, and use the same learning rate for batch size 128 and 256.

We report Inception Score (IS) (Salimans et al., 2016) as a function of number of iterations.

Figure 1 suggests that OAdagrad performs better than OSG and Alternating Adam, and OAdagrad results in higher IS.

We compare the generated CIFAR10 images associated with these three methods, which is included in Appendix A. We also provide experimental results to compare the performance of different algorithms using different minibatch sizes, which are included in Appendix E.

In the second experiment, we employ OAdagrad to train GANs and study the growth rate of the cumulative stochastic gradient (i.e., d i=1 g 1:N,i 2 ).

We tune the learning rate from {1×10 −3 , 2×10 −4 , 1×10 −4 , 2×10 −5 , 1×10 −5 } and choose batch size to be 64.

In Figure 2 , the blue curve and red curve stand for the growth rate for OAdagrad and its corresponding tightest polynomial growth upper bound respectively.

N is the number of iterations, and c is a multiplicative constant such that the red curve and blue curve overlaps at the starting point of the training.

The degree of the polynomial is determined using binary search.

We can see that the growth rate of cumulative stochastic gradient grows very slowly in GANs (the worst-case polynomial degree is 0.5, but it is 0.2 for WGAN-GP on CIFAR10 and 0.07 for WGAN on LSUN Bedroom dataset).

As predicted by our theory, this behavior explains the faster convergence of OAdagrad versus OSG, consistent with what is observed empirically in Figure 1 .

In the third experiment, we consider GAN training on largescale dataset.

We use the model from Self-Attention GAN (Zhang et al., 2018) (SA-GAN) and ImageNet as our dataset.

Note that in this setting the boundedness of both generator (G) and discriminator (D) is ensured by spectral normalization of both G and D. For both OAdagrad and Simultaneous Adam, we use different learning rate for generator and discriminator, as suggested in (Heusel et al., 2017) .

Specifically, the learning rates used are 10 −3 for the generator and 4 × 10 −5 for the discriminator.

We report both Inception Score (IS) and Fréchet Inception Distance (Heusel et al., 2017)

We compare the generated ImageNet images associated with the three optimization methods in Appendix A. Since Alternating Adam collapsed we don't report its Inception Score or FID.

As it can be seen in Figure 3 and Appendix A, OAdagrad outperforms simultaneous Adam in quantitative metrics (IS and FID) and in sample quality generation.

Future work will include investigating whether OAdagrad would benefit from training with larger batch size, in order to achieve state-of-the-art results.

In this paper, we explain the effectiveness of adaptive gradient methods in training GANs from both theoretical and empirical perspectives.

Theoretically, we provide two efficient stochastic algorithms for solving a class of min-max non-convex non-concave problems with state-of-the-art computational complexities.

We also establish adaptive complexity results for an Adagrad-style algorithm by using coordinate-wise stepsize according to the geometry of the history data.

The algorithm is proven to enjoy faster adaptive convergence than its non-adaptive counterpart when the gradient is sparse, which is similar to Adagrad applied to convex minimization problem.

We have conducted extensive empirical studies to verify our theoretical findings.

In addition, our experimental results suggest that the reason why adaptive gradient methods deliver good practical performance for GAN training is due to the slow growth rate of the cumulative stochastic gradient.

Comparison of Generated CIFAR10 Images by Different Optimization Methods In this section, we report the generated CIFAR10 images during the training of WGAN-GP by three optimization methods (OSG, OAdagrad, Alternating Adam).

Every method uses batch size 64, and 1 iteration represents calculating the stochastic gradient with minibatch size 64 once.

Figure 4 consists of images by three optimization methods at iteration 8000.

Visually we can see that OAdagrad is better than Alternating Adam, and both of them are significantly better than OSG.

It is consistent with the inception score results reported in Figure 1 , and it also illustrates the tremendous benefits delivered by adaptive gradient methods when training GANs.

Figure 5: Self-Attention GAN (SA-GAN): Generated ImageNet images using different optimization methods at iteration 135000.

OAdagrad produces better quality images than simultaneous Adam.

For both Oadagrad and simultaneous Adam we use the same learning rates: 0.001 for generator and 0.00004 for the discriminator.

Alternating Adam in our experience with same learning rate as in SA-GAN 0.0001 for generator and 0.0004 for discriminator collapsed.

Note that our setting is different from SA-GAN since our batchsize is 128 while it is 256 in SA-GAN.

It was also noted in SA-GAN that alternating Adam is hard to train.

Figure 6: Self-Attention GAN on ImageNet, with evaluation using Unoffical PyTorch Inception Score and Unoffical Pytorch FID.

We see that OAdagard indeed outperforms Simultaneous Adam in terms of the (PyTorch) Inception score (higher is better), and in terms of (PyTorch) Fréchet Inception Distance (lower is better).

We don't report here Alternating Adam since in our run it has collapsed.

Min-max Optimization and GAN Training For convex-concave min-max optimization, the extragradient method was first proposed by (Korpelevich, 1976) .

Later on, under gradient Lipschitz condition, Nemirovski (2004) extended the idea of extragradient to mirror-prox and obtained the O(1/N ) convergence rate in terms of the duality gap (see also (Nesterov, 2007) ), where N is the number of iterations.

When only the stochastic first-order oracle is available, the stochastic mirrorprox was analyzed by (Juditsky et al., 2011 ).

The convergence rates for both deterministic and stochastic mirror-prox are optimal (Nemirovsky & Yudin, 1983) .

Recently, Zhao (2019) developed a nearly-optimal stochastic first-order algorithm when the primal variable is strongly convex in the primal variable.

Bach & Levy (2019) proposed a universal algorithm that is adaptive to smoothness and noise, and simultaneously achieves optimal convergence rate.

There is a plethora of work analyzing one-sided nonconvex min-max problem, where the objective function is nonconvex in the minimization variable but concave in maximization variable.

When the function is weakly-convex in terms of the minimization variable, Rafique et al. (2018) propose a stage-wise stochastic algorithm that approximately solves a convex-concave subproblem by adding a quadratic regularizer and show the first-order convergence of the equivalent minimization problem.

Under the same setting, Lu et al. (2019) utilize block-based optimization strategy and show the convergence of the stationarity gap.

By further assuming that the function is smooth in the minimization variable, Lin et al. (2019) show that (stochastic) gradient descent ascent is able to converge to the first-order stationary point of the equivalent minimization problem.

Liu et al. (2020) cast the problem of stochastic AUC maximization with deep neural networks into a nonconvex-concave min-max problem, show the PL (Polyak-Łojasiewicz) condition holds for the objective of the outer minimization problem, and propose an algorithm and establish its fast convergence rate.

A more challenging problem is the non-convex non-concave min-max problem.

Dang & Lan (2015) demonstrate that the deterministic extragradient method is able to converge to -first-order stationary point with non-asymptotic guarantee.

Under the condition that the objective function is weaklyconvex and weakly-concave, Lin et al. (2018) designs a stage-wise algorithm, where in each stage a strongly-convex strongly-concave subproblem is constructed by adding quadratic terms and appropriate stochastic algorithms can be employed to approximately solve it.

They also show the convergence to the stationary point.

Sanjabi et al. (2018) design an alternating deterministic optimization algorithm, in which multiple steps of gradient ascent for dual variable are conducted before one step of gradient descent for primal variable is performed.

They show the convergence to stationary point based on the assumption that the inner maximization problem satisfies PL condition (Polyak, 1969) .

Our work is different from these previous methods in many aspects.

In comparison to (Lin et al., 2018) , our result does not need the bounded domain assumption.

Furthermore, our iteration complexity is O(1/ 4 ) to achieve -first-order stationary point while the corresponding complexity in (Lin et al., 2018) is O(1/ 6 ).

When comparing to (Sanjabi et al., 2018) , we do not assume that the PL (Polyak-Łojasiewicz) condition holds.

Additionally, our algorithm is stochastic and not restricted to the deterministic case.

Apparently the most related work to the present one is (Iusem et al., 2017) .

The stochastic extragradient method analyzed in (Iusem et al., 2017) requires calculation of two stochastic gradients per iteration, while the present algorithm only needs one since it memorizes the stochastic gradient in the previous iteration to guide the update in the current iteration.

Nevertheless, we achieve the same iteration complexity as in (Iusem et al., 2017) .

There are a body of work analyzing the convergence behavior of min-max optimization algorithms and its application in training GANs (Heusel et al., 2017; Daskalakis & Panageas, 2018; Nagarajan & Kolter, 2017; Grnarova et al., 2017; Yadav et al., 2017; Gidel et al., 2018; Mertikopoulos et al., 2018; Mazumdar et al., 2019) .

A few of them (Heusel et al., 2017; Daskalakis & Panageas, 2018; Mazumdar et al., 2019) only have asymptotic convergence.

Others (Nagarajan & Kolter, 2017; Grnarova et al., 2017; Daskalakis et al., 2017; Yadav et al., 2017; Gidel et al., 2018; Mertikopoulos et al., 2018) focus on more restricted settings.

For example, Nagarajan & Kolter (2017); Grnarova et al. (2017) require the concavity of the objective function in terms of dual variable.

Yadav et al. (2017); Gidel et al. (2018) assume the objective to be convex-concave.

Mertikopoulos et al. (2018) imposes the so-called coherence condition which is stronger than our assumption.

Daskalakis et al. (2017) analyze the last-iteration convergence for bilinear problem.

Recently, analyze the benefits of using negative momentum in alternating gradient descent to improve the training of a bilinear game.

Chavdarova et al. (2019) develop a variance-reduced extragradient method and shows its linear convergence under strong monotonicity and finite-sum structure assumptions.

Azizian et al. (2019) provide a unified analysis of extragradient for bilinear game, strongly monotone case, and their intermediate cases.

However, none of them give non-asymptotic convergence results for the class of non-convex non-concave min-max problem considered in our paper.

C PROOF OF THEOREM 1 C.1 FACTS Suppose X ⊂ R d is closed and convex set, then we have

Proof.

Let x * ∈ X * , where X * is the set of optimal solutions of MVI(T, X ), i.e. T (x), x − x * ≥ 0 holds for ∀x ∈ X .

where (a) holds by using Fact 1.

Note that

where the last inequality holds by the fact that

Define Λ k = 2 x * − z k , η k .

Taking x = x * in (8) and combining (9) and (10), we have

we rearrange terms in (11), which yields

Take summation over k = 1, . . .

, N in (12) and note that x 0 = z 0 , which yields

By taking η ≤ 1 9L , we have 1 − 36η 2 L 2 ≥ 1 2 , and we have the result.

Proof.

Define r η (z k ) = z k − Π X (z k − ηT (z k )) .

Our goal is to get a bound on r η (z k ).

We have:

where (a) holds since (a + b) 2 ≤ 2a 2 + 2b 2 , (b) holds by the non-expansion property of the projection operator and (a + b) 2 ≤ 2a 2 + 2b 2 .

Let x * ∈ X * , where X * is the set of optimal solutions of MVI(T, X ), i.e. T (x), x − x * ≥ 0 holds for ∀x ∈ X .

By summing over k in Equation (14) and using Equation (7) in Lemma 1, we have

D PROOF OF THEOREM 2

In this section, we define g k = T (z k ), k = g k − g k .

Lemma 2.

For any positive definite diagonal matrix H satisfying H δI with δ > 0, if T (x 1 ) − T (x 2 ) 2 ≤ L x 1 − x 2 2 for x 1 , x 2 ∈ X , then

Proof.

Note that H δI, we have 0 < H −1 1 δ I. Noting that x H = √ x Hx, we have

Lemma 3.

When η ≤ δ 9L , we have 1 2

Proof.

Our goal is to bound

E MORE EXPERIMENTAL RESULTS ON CIFAR10

In Figure 7 , we compare the performance of OSG, Alternating Adam (AlterAdam) and OAdagrad under the same minibatch size setting on CIFAR10 dataset, where one epoch means one pass of the dataset.

We can see that OAdagrad and Alternating Adam behave consistently better than OSG.

When the minibatch size is small (e.g., 64), OAdagrad and Alternating Adam have comparable performance, but when the minibatch size is large (e.g., 128, 256), OAdagrad converges faster than Alternating Adam.

This phenomenon shows the benefits of OAdagrad when large minibatch size is used.

@highlight

This paper provides novel analysis of adaptive gradient algorithms for solving non-convex non-concave min-max problems as GANs, and explains the reason why adaptive gradient methods outperform its non-adaptive counterparts by empirical studies.

@highlight

Develops algorithms for the solution of variational inequalities in the stochastic setting, proposing a variation of the extragradient method.
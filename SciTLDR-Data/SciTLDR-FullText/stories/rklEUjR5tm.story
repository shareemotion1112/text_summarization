Derivative-free optimization (DFO) using trust region methods is frequently used for machine learning applications, such as (hyper-)parameter optimization without the derivatives of objective functions known.

Inspired by the recent work in continuous-time minimizers, our work models the common trust region methods with the exploration-exploitation using a dynamical system coupling a pair of dynamical processes.

While the first exploration process searches the minimum of the blackbox function through minimizing a time-evolving surrogation function, another exploitation process updates the surrogation function time-to-time using the points traversed by the exploration process.

The efficiency of derivative-free optimization thus depends on ways the two processes couple.

In this paper, we propose a novel dynamical system, namely \ThePrev---\underline{S}tochastic \underline{H}amiltonian \underline{E}xploration and \underline{E}xploitation, that surrogates the subregions of blackbox function using a time-evolving quadratic function, then explores and tracks the minimum of the quadratic functions using a fast-converging Hamiltonian system.

The \ThePrev\ algorithm is later provided as a discrete-time numerical approximation to the system.

To further accelerate optimization, we present \TheName\ that parallelizes multiple \ThePrev\ threads for concurrent exploration and exploitation.

Experiment results based on a wide range of machine learning applications show that \TheName\ outperform a boarder range of derivative-free optimization algorithms with faster convergence speed under the same settings.

Derivative-free optimization (DFO) techniques BID31 , such as Bayesian optimization algorithms BID43 BID24 , non-differentiable coordinate descent BID4 , natural gradient method BID12 BID14 , and natural evolution strategies BID41 , have been widely used for black-box function optimization.

DFO techniques have been viewed as one of promising solutions, when the first-order/higher-order derivatives of the objective functions are not available.

For example, to train large-scale machine learning models, parameter tuning is sometimes required.

The problem to find the best parameters from the high-dimensional parameter space is frequently formalized as a black-box optimization problem, as the function that maps the specific parameter settings to the performance of models is not known BID11 BID9 BID48 BID21 .

The evaluation of the black-box function is often computationally expensive, and there thus needs DFO algorithms to converge fast with global/local minimum guarantee.

Backgrounds.

To ensure the performance of DFO algorithms, a series of pioneering work has been done BID5 BID36 BID16 BID2 BID11 .

Especially, Powell et al. (Powell, 1964; BID33 proposed Trust-Region methods that intends to "surrogate" the DFO solutions through exploring the minimum in the trust regions of the blackbox objective functions, where the trust regions are tightly approximated using model functions (e.g., quadratic functions or Gaussian process) via interpolation.

Such two processes for exploration and exploitation are usually alternatively iterated, so as to pursue the global/local minimum BID2 .

With exploration and exploitation BID7 , a wide range of algorithms have been proposed using trust region for DFO surrogation BID34 BID38 BID43 BID45 BID42 BID46 BID39 BID0 BID30 BID23 BID1 .Technical Challenges.

Though trust region methods have been successfully used for derivative-free optimization for decades, the drawbacks of these methods are still significant:??? The computational and storage complexity for (convex) surrogates is extremely high.

To approximate the trust regions of blackbox functions, quadratic functions BID34 BID39 and Gaussian process BID43 BID46 BID23 are frequently used as (convex) surrogates.

However, fitting the quadratic functions and Gaussian process through interpolation is quite time-consuming with high sample complexity.

For example, using quadratic functions as surrogates (i.e., approximation to the second-order Taylor's expansion) needs to estimate the gradient and inverse Hessian matrix BID34 BID39 , where a large number of samples are required to avoid ill-conditioned inverse Hessian approximation; while the surrogate function in GP is nonconvex, which is even more sophisticated to optimize.??? The convergence of trust region methods cannot be guaranteed for high-dimensional nonconvex DFO.

Compared to the derivative-based algorithms such as stochastic gradient descent and accelerated gradient methods BID3 BID44 , the convergence of DFO algorithms usually are not theoretically guaranteed.

Jamieson et al. BID16 provided the lower bound for algorithms based on boolean-based comparison of function evaluation.

It shows that DFO algorithms can converge at ???(1/ ??? T ) rate in the best case (T refers to the total number of iterations), without assumptions on convexity and smoothness, even when the evaluation of black-box function is noisy.

Our Intuitions.

To tackle the technical challenges, we are motivated to study novel trust region methods with following properties 1.

Low-complexity Quadratic Surrogates with Limited Memory.

To lower the computational complexity, we propose to use quadratic functions with identity Hessian matrices as surrogates.

Rather than incorporating all evaluated samples in quadratic form approximation, our algorithm only works with the most-recently evaluated sample points.

In this way, the memory consumption required can be further reduced.

However, the use of identity Hessian matrices for quadratic form loses the information about the distribution (e.g., Fisher information or covariance BID13 ) of evaluated sample points.

2.

Fast Quadratic Exploration with Stochastic Hamiltonian Dynamical Systems.

Though it is difficult to improve the convergence rate of the DFO algorithms in general nonconvex settings with less oracle calls (i.e., times of function evaluation), one can make the exploration over the quadratic trust region even faster.

Note that exploration requires to cover a trust region rather than running on the fastest path (e.g., the gradient flow BID15 ) towards the minimum of trust region.

In this case, there needs an exploration mechanism traversing the whole quadratic trust region in a fast manner and (asymptotically) approaching to the minimum.

FIG0 illustrates the examples of exploration processes over the quadratic region via its gradient flows (i.e., gradient descent) or using Hamiltonian dynamics with gradients BID25 as well as their stochastic variants with explicit perturbation, all in the same length of time.

It shows that the stochastic Hamiltonian dynamics (shown in FIG0 (d)) can well balance the needs of fast-approaching the minimum while sampling the quadratic region with its trajectories.

Compared to the (stochastic) gradient flow, which leads to the convergence to the minimum in the fast manner, the stochastic Hamiltonian system are expected to well explore the quadratic trust region with the convergence kept.

Inspired by theoretical convergence consequences of Hamiltonian dynamics with Quadratic form BID44 BID25 , we propose to use stochastic Hamiltonian dynamical system for exploring the quadratic surrogates.

3.

Multiple Quadratic Trust Regions with Parallel Exploration-Exploitation.

Instead of using one quadratic cone as the surrogate, our method constructs the trust regions using multiple quadratic surrogates, where every surrogate is centered by one sample point.

In this way, the information of multiple sample points can be still preserved.

Further, to enjoy the speedup of parallel computation, the proposed method can be accelerated through exploring the minimum from multiple trust regions (using multiple Hamiltonian dynamical sys- Our work is inspired by the recent progress in the continuous-time convex minimizers BID44 BID15 BID47 on convex functions, where the optimization algorithms are considered as the discrete-time numerical approximation to some (stochastic) ordinary differential equations (ODEs) or dynamics, such as It?? processes for SGD algorithms BID15 or Hamiltonian systems for Nesterov's accelerated SGD BID44 .

We intend to first study the new ODE and dynamical system as a continuous-time DFO minimizer that addresses above three research issues.

With the new ODE, we aim at proposing the discrete-time approximation as the algorithms for black-box optimization.

Our Contributions.

Specifically, we make following contributions.

(1) To address the three technical challenges, a continuous-time minimizer for derivative-free optimization based on a Hamiltonian system coupling two processes for exploration and exploitation respectively.

(2) Based on the proposed dynamical system, an algorithm, namely SHE 2 -Stochastic Hamiltonian Exploration and Exploitation, as a discrete-time version of the proposed dynamical system, as well as P-SHE 2 that parallelizes SHE 2 for acceleration.

(3) With the proposed algorithms, a series of experiments to evaluate SHE 2 and P-SHE 2 using real-world applications.

The two algorithms outperform a wide range of DFO algorithms with better convergence.

To the best of our knowledge, this work is the first to use a Hamiltonian system with coupled process for DFO algorithm design and analysis.

In this section, we first review the most relevant work of trust region methods for DFO problem, then present the preliminaries of this work.

The trust region algorithms can be categorized by the model functions used for surrogates.

Generally, there are two types of algorithms adopted: Gaussian Process (GP) BID43 BID45 BID46 BID23 or Quadratic functions BID34 BID39 BID30 for surrogation.

Blessed by the power of Bayesian nonparameteric statistics, Gaussian process can well fit the trust regions, with confidence bounds measured, using samples evaluated by the blackbox function.

However, the GP-based surrogation cannot work in high dimension and cannot scale-up with large number of samples.

To solved this problem, GP-based surrogation algorithms using the kernel gradients BID46 and mini-batch BID23 have been recently studied.

On the other hand, the quadratic surrogation BID34 indeed approximates the trust region through interpolating the second-order Taylor expansion of the blackbox objective.

With incoming points evaluated, there frequently needs to numerically estimate and adapt the inverse Hessian matrix and gradient vector, which is extremely time-consuming and sample-inefficiency (with sample BID34 ).

Following such settings, BID39 proposed to a second-order algorithm for blackbox variational inference based on quadratic surrogation, while BID30 ) leveraged a Gaussian Mixture Model (multiple quadratic surrogations) to fit the policy search space over blackbox probabilistic distribution for policy optimization.

A novel convex model generalizing the quadratic surrogation has been recently proposed to characterize the loss for structured prediction BID1 .

DISPLAYFORM0 In addition, some evolutionary strategies, such as Covariance Matrix Adaptation Evolution Strategy (CMA-ES) BID13 BID0 , indeed behave as a sort of quadratic surrogate as well.

Compared to the common quadratic surrogate, CMA-ES models the energy of blackbox function using a multivariate Gaussian distribution.

For every iteration, CMA-ES draws a batch of multiple samples from the distribution, then statistically updates parameters of the distribution using the samples with blackbox evaluation.

CMA-ES can be further accelerated with parallel blackbox function evaluation and has been used for hyperparameter optimization of deep learning BID22 .

Here, we review the Nesterov's accelerated method for quadratic function minimization.

We particularly are interested in the ODE of Nesterov's accelerated method and interpret behavior of the ODE as a Hamiltonian dynamical system.

Corollary 1 (ODE of Nesterov's Accelerated Method).

According to BID44 , the discretetime numerical format of the Nesterov's accelerated method BID27 can be viewed as an ODE as follow.

Z DISPLAYFORM0 where f (X) is defined as the objective function for minimization and ??? X f (Z(t)) refers to the gradient of the function on the point Z(t).

Above ODE can converge with strongly theoretical consequences if the function f (X) is convex with some smoothness assumptions BID44 .

Corollary 2 (Convergence of Eq 1 over Quadratic Loss).

Let's set f (X) = DISPLAYFORM1 According to the ODE analysis of Nestereov's accelerated method BID44 , the ODE listed in Eq 1 converges with increasing time t at the following rate: DISPLAYFORM2 where X(0) refers to the initial status of the ODE.

The proof has been given in BID44 .

In this section, we present the proposed Hamiltonian system for Black-Box minimization via exploration and exploitation.

Then, we introduce the algorithms and analyze its approximation to the dynamical systems.

Given a black-box objective function f (X) and X ??? R d , we propose to search the minimum of f (X) in the d-dimensional vector space R d , using a novel Hamiltonian system, derived from the ODE of Nesterov's accelerated method and Eq 1, yet without the derivative of f (X) needed.

Definition 1 (Quadratic Loss Function).

Given two d-dimensional vectors X and Y , we characterizes the Euclid distance between the two vectors using the function as follow.

DISPLAYFORM0 where the partial derivative of Q on X should be DISPLAYFORM1 Definition 2 (Stochastic Hamiltonian Exploration and Exploitation).

As was shown in Eq. 4, a Hamiltonian system is designed with following two coupled processes: exploration process X(t) and exploitation process Y (t), where t refers to the searching time.

These two processes are coupled withn each other.

Specifically, the exploration process X(t) in Eq 4 uses a second order ODE to track the dynamic process Y (t), while the exploiting process Y (t) always memorizes the minimum point (i.e., X(?? )) that have been reached by X(t) from time 0 to t. DISPLAYFORM2 where (1) DISPLAYFORM3 indicates the fastest direction to track Y (t) from X(t); and (2) the perturbation term ??(t) referring to an unbiased random noise with controllable bound ?? t ??? ???(?? t + 3/t) and ?? t ??? 3/t; 5: DISPLAYFORM4 6: DISPLAYFORM5 else 12: DISPLAYFORM6 end if 14: end for 15: return Y T ; ??(t) 2 ??? would help the system escape from an unstable stationary point in even shorter time.

In the above dynamical system, we treat Y (t) as the minimizer of the black-box function f (X).

2 approximates the black-box function f (X) using a simple yet effective quadratic function, then leverages the ODE listed in Eq 1 to approximate the minimum with the quadratic function.

With the new trajectories traversed by Eq 1, the quadratic function would be updated.

Through repeating such surrogation-approximation-updating procedures, the ODE continuously tracks the time-dependent evolution of quadratic loss functions and finally stops at a stationary point when the quadratic loss functions is no longer updated (even with new trajectories traversed).

Remark 1.

We can use the analytical results BID44 to interpret the dynamical system (in Eq 4) as an adaptive perturbated dynamical system that intends to minimize the Euclid distance between X(t) and Y (t) at each time t. The memory complexity of this continuous-time minimizer is O(1), where a Markov process Y (t) is to used to memorize the status quo of local minimum during exploration and exploitation.

Theorem 1 (Convergence of SHE 2 Dynamics).

Let's denote x * as a possible local minimum point of the landscape function f (x).

We have as t ??? ???, with high probability, that X(t) ??? x * , where X(t) is the solution to (4).Please refer to the Lemma 1 and Lemma 2 in the Appendix for the proof of above theorems.

We will discuss the rate of convergence, when introducing SHE 2 algorithm as a discrete-time approximation to SHE 2 .3.2 SHE 2 : ALGORITHM DESIGN AND ANALYSIS Given a black-box function f (x) and a sequence of non-negative step-size ?? t (t=0, 1, 2, . . .

, T ), which is small enough, as well as the scale of perturbation ??, we propose to implement SHE 2 as Algorithm 1.

The output of algorithm Y T refers to the value of Y t in the last iteration (i.e., the t th iteration).

The whole algorithm only uses the evaluation of function f (x) for comparisons, without computing its derivatives.

In each iteration, only the variable Y t is dedicated to memorize the local minimum in the sequence of X 1 , X 2 , . . .

, X t .

Thus the memory complexity of SHE 2 is O(1).In terms of convergence, Jamieson et al BID16 provided an universal lower bound on the convergence rate of DFO based on the "boolean-valued" comparison of (noisy) function evaluation.

SHE 2 should enjoy the same convergence rate ???(1/ ??? T ) without addressing any further assumptions.

Here, we would demonstrate that the proposed algorithm behaves as a discrete-time approximation to the dynamical systems of X(t) and Y (t) addressed in Eq 4, while as ?? t ??? 0 the sequence of X t and Y t (for 1 ??? t ??? T ) would converge to the behavior of continuous-time minimizer -coupled processes X(t) and Y (t).Given an appropriate constant step-size ?? t ??? 0 for t = 1, 2..., T , we can rewrite the the sequences X t described in lines 4-7 of Algorithm 1 as the following Stochastic Differential Equation (SDE) of X ?? (t) with the random noise ??(t): DISPLAYFORM0 where ??(t) refers to the continuous-time dynamics of sequence ?? 1 , ?? 2 , . . .

, ?? T and |??(t)| 2 = ?? for every time t. Through combining above two ODEs and Lemma 1, we can obtain the SDE of X(t) based on the perturbation ??(t) as: DISPLAYFORM1 The sequence Y t (t=0, 1, 2, ?? ?? ?? T ) always exploits the minimum point that has been already found by X t at time t. Thus, we can consider Y t is the discrete-time of Y (t) that exploits the minimum traversed by X(t).

In this way, we can consider the coupled sequences of X t and Y t (for 1 ??? t ??? T ) as the discrete-time form of the proposed dynamical system with X(t) and Y (t).

To enjoy the speedup of parallel computation, we propose a new Hamiltonian dynamical system with a set of ODEs that leverage multiple pairs of coupled processes for exploration and exploitation in parallel.

Then, we present the algorithm design as a discrete-time approximation to the ODEs.

2 DYNAMICAL SYSTEM Given a black-box objective function f (X) and X ??? R d , we propose to search the minimum of f (X) in the d-dimensional vector space R d , using following systems.

Definition 3 (Parallel Stochastic Hamiltonian Exploration and Exploitation).

As was shown in Eq. 7, a Hamiltonian system is designed with (1) N pairs of coupled exploration-exploitation processes: X i (t) and Y i (t) for 1 ??? i ??? N that explores and exploits the minimum in-parallel from N (random/unique) starting points, and (2) an overall exploitation process Y (t) memorizing the local minimum traversed by the all N pairs of coupled processes.

Specifically, for each pair of coupled processes, a new surrogation model Q ?? (X i (t), Y i (t), Y (t)) has been proposed to measure the joint distance from X i (t) to Y i (t) and Y (t) respectively, where ?? > 0 refers to a trade-off factor weighted-averaging the two distances.

DISPLAYFORM0 where DISPLAYFORM1 indicates the fastest direction to track Y i (t) and Y (t), jointly, from X(t).

In the above dynamical system, we treat Y (t) as the minimizer of the black-box function f (X).

Remark 2.

We understand the dynamical system listed in Eq 7 as a perturbated dynamical system with multiple state variables, where all variables are coupled to search the minimum of f (X) through X i (t) (for 1 ??? i ??? N ).

The memory complexity of this continuous-time minimizer is O(N ), where every Markov process Y i (t) is to used to memorize the status quo of local minimum traversed by the corresponding processes.

6: for t = 1, 2 . . .

, T do 7:for j = 1, 2, 3, . . .

, N in Parallel do 8:/* X(t) update for the j th SHE 2 thread*/ 9:?? t ??? ???(?? t + 3/t) and ?? t ??? 3/t;10: DISPLAYFORM0 11: DISPLAYFORM1 12: DISPLAYFORM2 13: DISPLAYFORM3 DISPLAYFORM4 where ?? j (t) refers to the continuous-time dynamics of sequence ?? j 1 , ?? j 2 , . . . , ?? j T and |?? j (t)| 2 = ?? for every time t. Through combining above three ODEs and Eq. 8, we can obtain the ODE of X(t) as: DISPLAYFORM5 Using same the settings, we can conclude that X j t would have similar behavior as X i (t) (for 1 ??? i ??? N in Eq 7).

Thus, Algorithm 2 can be viewed as a discrete-time approximation of dynamical systems in Eq 7.

Since the sequence Y t always exploits the minimum point that has been found by all N threads at every time t, we can use the algorithm output Y T as the minimizer of f (x).

The proposed P-SHE 2 algorithm can be viewed as a particle swarm optimizer Kennedy FORMULA1 with inverse-scale step-size settings.

Compared to Particle Swarm, which usually adopts constant stepsize settings (i.e., ?? t , ?? t and ?? t are fixed as a constant value), P-SHE 2 proposes to use a small ?? t , while setting ?? t = ???(?? t + 3/t) and ?? t = 3/t for each (the t th ) iteration.

Such settings help the optimizer approximates to the Nesterov's scheme, so as to enjoy faster convergence speed, under certain assumption.

In terms of contribution, our research made as yet an rigorous analysis for Particle Swarm through linking it to to Nesterov's scheme Nesterov (2013); BID44 .

We provide three sets of experiments to validate our algorithms.

In the first set of experiments, we demonstrate the performance of SHE 2 and P-SHE 2 to minimize two non-convex functions through the comparisons to a set of DFO optimizers, including Gaussian Process optimization algorithms (GP-UCB) (Martinez-Cantin, 2014), Powell's BOBYQA methods BID35 , Limited Memory-BFGS-B (L-BFGS) BID50 , Covariance Matrix Adaptation Evolution Strategy (CMA-ES) BID13 , and Particle Swarm optimizer (PSO) BID18 .

For the second set of experiments, we use the same set of algorithms to train logistic regression BID20 and support vector machine BID6 classifiers, on top of benchmark datasets, for supervised learning tasks.

In the third set, we use P-SHE 2 to optimize the hyper-parameters of ResNet-50 for the performance tuning on Flower 102 and MIT Indoor 67 benchmark datasets under transfer learning settings.

Figure 2 presents the performance comparison between P-SHE 2 , SHE 2 and the baseline algorithms using two 2D benchmark nonconvex functions-Franke's function and Peaks function.

Figure 2.a and c present the landscape of these two functions, while Figure 2 .b and d present the performance evaluation of P-SHE 2 , SHE 2 and baseline algorithms on these two functions.

All these algorithms are tuned with best parameters and evaluated for 20 times, while averaged performance is presented.

Specifically, we illustrate how these algorithms would converge with increasing number of iterations.

Obviously, on Franke's function, only P-SHE 2 (10), i.e., the P-SHE 2 algorithm with N = 10 search threads, CMA-ES, GP-UCB algorithms and BOBYQA converge to the global minimum, while the rest of algorithms, including SHE 2 and PSO, converge to the local minimum.

Though P-SHE 2 (10) needs more iterations to converge to the global minimum, its per iteration time consumption is significantly lower than the other three convergeable algorithms (shown in Figure 2.e) .

The same comparison result can be also observed from the comparison using Peaks function (in Figures 2.b and 2 .c, only CMA-ES and P-SHE 2 (10) converge to global minimum in the given number of iterations).

Compared P-SHE 2 (10) to PSO(10), they both use 10 search threads with the same computational/memory complexity, while P-SHE 2 (10) converges much faster than PSO(10).

The same phenomena can be also observed from the comparison between SHE 2 and PSO(1), both of which search with single thread.

We can suggest that the adaptive step-size settings inherent from Nesterove's scheme accelerate the convergence speed.

We use above algorithms to train logistic regression (LR) and SVM classifiers using Iris (4 features, 3 classes and 150 instances), Breast (32 features, 2 classes and 569 instances) and Wine (13 features, 3 classes and 178 instances) datasets.

We treat the loss functions of logistic regression and SVM as black-box functions and parameters (e.g., projection vector ?? for logistic regression) as optimization outcomes.

Note that the number of parameters for multi-class (#class ??? 3) classification is #class ?? #f eatures, e.g., 39 for wine data.

We don't include GP-UCB in the comparison, as it is extremely time-consuming to scale-up in high-dimensional settings.

FIG2 demonstrates how loss function could be minimized by above algorithms with iterations by iterations.

For both classifiers on all three datasets, P-SHE 2 (100)-the P-SHE 2 algorithms with N = 100 search threads, outperforms all rest algorithms with the most significant loss reduction and the best convergence performance.

We also test the accuracy of trained classifiers using the testing datasets.

Table.

1 shows that both classifiers trained by P-SHE 2 (100) enjoys the best accuracy among all above DFO algorithms and the accuracy is comparable to those trained using gradientbased optimizers.

All above experiments are carried out under 10-folder cross-validation.

Note that the accuracy of the classifiers trained by P-SHE 2 is closed to/or even better than some fine-tuned gradient-based solutions BID8 BID40 ).

To test the performance of P-SHE 2 for derivative-free optimization with noisy black-box function evaluation, We use P-SHE 2 to optimize the hyper-parameter of ResNet-50 networks for Flower BID29 and MIT Indoor 67 classification BID37 ) tasks.

The two networks are pre-trained using ImageNet BID19 and Place365 datasets BID49 , respectively.

Specifically, we design a black-box function to package the training procedure of the ResNet-50, where 12 continuous parameters, including the learning rate of procedure, type of optimizers (after simple discretization), the probabilistic distribution of image pre-processing operations for randomized data augmentation and so on, are interfaced as the input of the function while the validation loss of the network is returned as the output.

We aim at searching the optimal parameters with the lowest validation loss.

The experiments are all based on a Xeon E5 cluster with many available TitanX, M40x8, and 1080Ti GPUs.

Our experiments compare P-SHE 2 with a wide range of solvers and hyper-parameter tuning tools, including PSO, CMA-ES BID22 , GP-UCB BID17 and BOBYQA under the same pre-training/computing settings.

Specifically, we adopt the vanilla implementation of GP-UCB and BOBYQA (with single search thread), while P-SHE 2 , PSO and CMA-ES are all with 10 search threads for parallel optimization.

The experimental results show that all these algorithms can well optimize the hyer-parameters of ResNet for the better performance under the same settings, while P-SHE 2 has ever searched the hyperparameters with the lowest validation loss in our experiments (shown in FIG4 ).

Due to the restriction of PyBOBYQA API, we can only provide the function evaluation of the final solution obtained by BOBYQA as a flatline in FIG4 .

In fact, P-SHE 2 , PSO and CMA-ES may spend more GPU hours than GP-UCB and BOBYQA due to the parallel search.

For the fair comparison, we also evaluate GP-UCB and BOBYQA with more than 100 iterations til the convergence, where GP-UCB can achieve 0.099854 validation error (which is comparable to the three parallel solvers) for Flower 102 task.

Note that we only claim that P-SHE 2 can be used for hyerparameter optimization with decent performance.

We don't intend to state that P-SHE 2 is the best for hyerparameter tuning, as the performance of the three parallel solvers are sometimes randon and indeed close to each other.

In this paper, we present SHE 2 and P-SHE 2 -two derivative-free optimization algorithms that leverage a Hamiltonian exploration and exploitation dynamical systems for black-box function optimization.

Under mild condition SHE 2 algorithm behaves as a discrete-time approximation to a Nestereov's scheme ODE BID44 over the quadratic trust region of the blackbox function.

Moreover, we propose P-SHE 2 to further accelerate the minimum search through parallelizing multiple SHE 2 -alike search threads with simple synchronization.

Compared to the existing trust region methods, P-SHE 2 uses multiple quadratic trust regions with multiple (coupled) stochastic Hamiltonian dynamics to accelerate the exploration-exploitation processes, while avoiding the needs of Hessian matrix estimation for quadratic function approximation.

Instead of interpolating sampled points in one quadratic function, P-SHE 2 defacto constructs one quadratic surrogate (with identity Hessian) for each sampled point and leverages parallel search threads with parallel black-box function evaluation to boost the performance.

Experiment results show that P-SHE 2 can compete a wide range of DFO algorithms to minimize nonconvex benchmark functions, train supervised learning models via parameter optimization, and fine-tune deep neural networks via hyperparameter optimization.

2 DYNAMICAL SYSTEM Our goal is to show that in the system (4) we have X(t) ??? x * as t ??? ???, where x * is a local minimum point of the landscape function f (x).

Definition 4.

We say the point x * is a local minimum point of the function f (x) if and only if f (x * ) ??? f (x) for any x ??? U (x * ), where U (x * ) which is any open neighborhood around the point x * .Let us first remove the noise ??(t) in our system (4).

Thus we obtain the following deterministic dynamical system (X 0 (t), Y 0 (t)): DISPLAYFORM0 In the equations (a) and (b) of (10), the pair of processes (X 0 (t), Y 0 (t)) is a pair of coupled processes.

In the next Lemma, we show that X 0 (t) converges to the minimum point of f (x) along the trajectory of X 0 (t) as t ??? ???.Lemma 1.

For the deterministic system (10), we have that DISPLAYFORM1 Proof.

Set??? 0 (t) = P (t), we can write equation (a) in (10) in Hamiltonian form DISPLAYFORM2 Set H(X, Y, P ) = P 2 2 + Q(X, Y ).

Then we have DISPLAYFORM3 As we have DISPLAYFORM4 , we see from (12) that we have DISPLAYFORM5 Notice that by our construction part (b) of the coupled process (10), we have f (Y 0 (t)) ??? f (X 0 (s)) for 0 ??? s ??? t. If X 0 (t) ??? Y 0 (t) = 0, then (X 0 (t) ??? Y 0 (t)) ????? 0 (t) = 0.

If X 0 (t) ??? Y 0 (t) = 0, then we see that Y 0 (t) = X 0 (s 0 ) for some 0 ??? s 0 < t, and f (X 0 (s 0 )) < f (X 0 (t)).

By continuity of the trajectory of X 0 (t) as well as the function f (x), we see that in this case??? 0 (t) = 0, so that (X 0 (t) ??? Y 0 (t)) ????? 0 (t) = 0.

Thus we see that (13) actually gives d dt H(X 0 (t), Y 0 (t), P (t)) = ??? 3 t P (t) 2 2 ??? 0 .From here, we know that H(X 0 (t), Y 0 (t), P (t)) keeps decaying until P (t) ??? 0 and X 0 (t) ??? Y 0 (t) 2 ??? 0, as desired.

Since f (Y 0 (t)) ??? min 0???s???t f (X 0 (s)), Lemma 1 tells us that as t ??? ???, the deterministic process X 0 (t) in (10) approaches the minimum of f along the trajectory traversed by itself.

Let us now add the noise ??(t) to part (a) of (10), so that we come back to our original system (4).

We would like to argue that with the noise ??(t), we actually have lim t?????? min 0???s???t f (X(s)) = f (x * ), and thus X(t) ??? x * when t ??? ??? as desired.

Lemma 2.

For the process X(t) in part (a) of the system (4), we have X(t)

??? x * as t ??? ???, where x * is a local minimum of the landscape function f (x).Proof.

We first notice that we have ??(t) = ?? ?? t ?? t 2 , where ?? t ??? N (0, I) is a sequence of i.i.d.normal, so that ??(t) 2 = ??, E??(t) = 0.

Viewing (10) as a small random perturbation (see BID10 , Chapter 2, Section 1)) of the system (4) we know that for any ?? > 0 fixed, we have DISPLAYFORM6 as ?? ??? 0.

From here we know that the process X(t) behaves close to X 0 (t) with high probability, so that by Lemma 1 we know that with high probability we have lim t?????? f (X(t)) ??? min 0???s???t f (X(s)) = 0 .Our next step is to improve the above asymptotic to X(t) ??? x * as t ??? ???. Comparing with (16), we see that it suffices to show DISPLAYFORM7 To demonstrate (17), we note that when t is large, we can ignore in (4) the damping term 3 t??? (t) and obtain a friction-less dynamics f (X(?? ))

.Combining Lemma 1, (15) and ??? ???X Q(X, Y ) = X ??? Y we further see that the term ??? ???X Q(X, Y ) also contribute little in (18).

Thus part (a) of (18) reduces to a very simple equation DISPLAYFORM8 Equation FORMULA1 enables the process X(t) to explore locally in an ergodic way its neighborhood points, so that if FORMULA1 is not valid, then X(t + dt) will explore a nearby point at which f (X(t + dt)) is less that min 0???s???t f (X(t)), and thus will move to that point.

This leads to a further decay in the value of f (X(t)), which demonstrates that in he limit t ??? ??? we must have (17), and the Lemma concludes.

Summarizing, we have the Theorem 1.

Here we provide a short discussion on the convergence rate of the algorithm SHE2.

In the previous appendix we have demonstrated that the system (4) converges via two steps.

Step 1 in Lemma 1 shows that the differential equation modeling Nesterov's accelerated gradient descent (see BID44 ) helps the process X(t) to "catch up" with the minimum point Y (t) on its path.

Step 2 in Lemma 2 shows that when t ??? ??? the noise term ??(t) helps the process X(t) to reach local

@highlight

a new derivative-free optimization algorithms derived from Nesterov's accelerated gradient methods and Hamiltonian dynamics
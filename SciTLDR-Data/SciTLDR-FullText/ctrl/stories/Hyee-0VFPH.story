We develop a novel and efficient algorithm for optimizing neural networks inspired by a recently proposed geodesic optimization algorithm.

Our algorithm, which we call Stochastic  Geodesic Optimization (SGeO), utilizes an adaptive coefficient on top of Polyak's Heavy Ball method effectively controlling the amount of weight put on the previous update to the parameters based on the change of direction in the optimization path.

Experimental results on strongly convex functions with Lipschitz gradients and deep Autoencoder benchmarks show that SGeO reaches lower errors than established first-order methods and competes well with lower or similar errors to a recent second-order method called K-FAC (Kronecker-Factored Approximate Curvature).

We also incorporate Nesterov style lookahead gradient into our algorithm (SGeO-N) and observe notable improvements.

First order methods such as Stochastic Gradient Descent (SGD) with Momentum (Sutskever et al., 2013) and their variants are the methods of choice for optimizing neural networks.

While there has been extensive work on developing second-order methods such as Hessian-Free optimization (Martens, 2010) and Natural Gradients (Amari, 1998; Martens & Grosse, 2015) , they have not been successful in replacing them due to their large per-iteration costs, in particular, time and memory.

Although Nesterov's accelerated gradient and its modifications have been very effective in deep neural network optimization (Sutskever et al., 2013) , some research have shown that Nesterov's method might perform suboptimal for strongly convex functions (Aujol et al., 2018) without looking at local geometry of the objective function.

Further, in order to get the best of both worlds, search for optimization methods which combine the the efficiency of first-order methods and the effectiveness of second-order updates is still underway.

In this work, we introduce an adaptive coefficient for the momentum term in the Heavy Ball method as an effort to combine first-order and second-order methods.

We call our algorithm Geodesic Optimization (GeO) and Stochastic Geodesic Optimization (SGeO) (for the stochastic case) since it is inspired by a geodesic optimization algorithm proposed recently (Fok et al., 2017) .

The adaptive coefficient effectively weights the momentum term based on the change in direction on the loss surface in the optimization process.

The change in direction can contribute as implicit local curvature information without resorting to the expensive second-order information such as the Hessian or the Fisher Information Matrix.

Our experiments show the effectiveness of the adaptive coefficient on both strongly-convex functions with Lipschitz gradients and general non-convex problems, in our case, deep Autoencoders.

GeO can speed up the convergence process significantly in convex problems and SGeO can deal with illconditioned curvature such as local minima effectively as shown in our deep autoencoder benchmark experiments.

SGeO has similar time-efficiency as first-order methods (e.g. Heavy Ball, Nesterov) while reaching lower reconstruction error.

Compared to second-order methods (e.g., K-FAC), SGeO has better or similar reconstruction errors while consuming less memory.

The structure of the paper is as follows: In section 2, we give a brief background on the original geodesic and contour optimization introduced in Fok et al. (2017) , neural network optimization methods and the conjugate gradient method.

In section 3, we introduce our adaptive coefficient specifically designed for strongly-convex problems and then modify it for general non-convex cases.

In section 4, we discuss some of the related work in the literature.

Section 5 illustrates the algorithm's performance on convex and non-convex benchmarks.

More details and insights regarding the algorithm and the experiments can be found in the Appendix.

The goal is to solve the optimization problem min θ∈R f (θ) where f : R D → R is a differentiable function.

Fok et al. (2017) approach the problem by following the geodesics on the loss surface guided by the gradient.

In order to solve the geodesic equation iteratively, the authors approximate it using a quadratic.

In the neighbourhood of θ t , the solution of the geodesic equation can be approximated as:

Clearly, one can see the conjugate gradient method as a momentum method where t is the learning rate and t γ t−1 is the momentum parameter:

We added the prime notation to avoid confusion with d t = θ t+1 − θ t throughout the paper).

To avoid calculating the Hessian ∇ 2 f which can be very expensive in terms of computation and memory, is usually determined using a line search, i.e. by approximately calculating t = arg min f (θ t + d t ) and several approximations to γ have been proposed.

For example, Fletcher & Reeves (1964) have proposed the following:

Note that γ F R (with an exact line search) is equivalent to the original conjugate gradient algorithm in the quadratic case.

The adaptive coefficient that appears before the unit tangent vector in equation 2 has an intuitive geometric interpretation:

where φ t is the angle between the previous update d t = θ t − θ t−1 and the negative of the current gradient −g t .

Since 0 ≤ φ ≤ π, thus −1 ≤ cos (π − φ t ) ≤ 1.

The adaptive coefficient embeds a notion of direction change on the path of the algorithm which can be interpreted as implicit second-order information.

The change in direction at the current point tells us how much the current gradient's direction is different from the previous gradients which is similar to what second-order information (e.g. the Hessian) provide.

For more details on the adaptive coefficient we refer the reader to Appendix C.

We propose to apply this implicit second-order information to the Heavy Ball method of Polyak (1964) as an adaptive coefficient for the momentum term such that, in strongly-convex functions with Lipschitz gradients, we reinforce the effect of the previous update when the directions align, i.e. in the extreme case: φ = 0 and decrease when they don't, i.e. the other extreme case: φ = π.

Thus, we write the coefficient as γ C t = 1 −ḡ t ·d t (11) with C indicating "convex".

It's obvious that 0 ≤ γ C t ≤ 2.

Note that we will use the bar notation (e.g.d) throughout the paper indicating normalization by magnitude.

Aμ-strongly convex function f with L-Lipschitz gradients has the following properties:

Applying the proposed coefficient to the Heavy Ball method, we have the following algorithm which we call GeO (Geodesic Optimization):

Algorithm 1: GEO (STRONGLY CONVEX AND LIPSCHITZ)

Calculate the gradient g t = ∇f (θ t )

Calculate adaptive coefficient γ

where T is total number of iterations and α is a tunable parameter set based on the function being optimized and is the learning rate.

Incorporating Nesterov's momentum We can easily incorporate Nesterov's lookahead gradient into GeO by modifying line 4 to g t = ∇f (θ t + µd t ) which we call GeO-N. In GeO-N the gradient is taken at a further point θ t + µd t where µ is a tunable parameter usually set to a value close to 1.

However, the algorithm proposed in the previous section would be problematic for non-convex functions such as the loss function when optimizing neural networks.

Even if the gradient information was not partial (due to minibatching), the current direction of the gradient cannot be trusted because of non-convexity and poor curvature (such as local minima, saddle points, etc).

To overcome this issue, we propose to alter the adaptive coefficient to

with N C indicating "non-convex".

By applying this small change we are reinforcing the previous direction when the directions do not agree thus avoiding sudden and unexpected changes of direction (i.e. gradient).

In other words, we choose to trust the previous history of the path already taken more, thus acting more conservatively.

To increase efficiency, we use minibatches, calling the following algorithm SGeO (Stochastic Geodesic Optimization):

Draw minibatch from training set Calculate the gradient g t = ∇f (θ t )

Calculate adaptive coefficient γ

Further we found that using the unit vectors for the gradientḡ and the previous updated, when calculating the next update in the non-convex case makes the algorithm more stable.

In other words, the algorithm behaves more robustly when we ignore the magnitude of the gradient and the momentum term and only pay attention to their directions.

Thus, the magnitudes of the updates are solely determined by the corresponding step sizes, which are in our case, the learning rate and the adaptive geodesic coefficient.

Same as the strongly convex case, we can integrate Nesterov's lookahead gradient into SGeO by replacing line 5 with g t = ∇f (θ t + µd t ) which we call SGeO-N.

There has been extensive work on large-scale optimization techniques for neural networks in recent years.

A good overview can be found in Bottou et al. (2018) .

Here, we discuss some of the work more related to ours in three parts.

Adagrad (Duchi et al., 2011) is an optimization technique that extends gradient descent and adapts the learning rate according to the parameters.

Adadelta (Zeiler, 2012) and RMSprop (Tieleman & Hinton, 2012) improve upon Adagrad by reducing its aggressive deduction of the learning rate.

Adam (Kingma & Ba, 2014) improves upon the previous methods by keeping an additional average of the past gradients which is similar to what momentum does.

Adaptive Restart (Odonoghue & Candes, 2015) proposes to reset the momentum whenever rippling behaviour is observed in accelerated gradient schemes.

AggMo (Lucas et al., 2018) keeps several velocity vectors with distinct parameters in order to damp oscillations.

AMSGrad (Reddi et al., 2018) on the other hand, keeps a longer memory of the past gradients to overcome the suboptimality of the previous algorithms on simple convex problems.

We note that these techniques are orthogonal to our approach and can be adapted to our geodesic update to further improve performance.

Several recent works have been focusing on acceleration for gradient descent methods.

Meng & Chen (2011) propose an adaptive method to accelerate Nesterov's algorithm in order to close a small gap in its convergence rate for strongly convex functions with Lipschitz gradients adding a possibility of more than one gradient call per iteration.

In Su et al. (2014) , the authors propose a differential equation for modeling Nesterov inspired by the continuous version of gradient descent, a.k.a.

gradient flow.

Wibisono et al. (2016) take this further and suggest that all accelerated methods have a continuous time equivalent defined by a Lagrangian functional, which they call the Bregman Lagrangian.

They also show that acceleration in continuous time corresponds to traveling on the same curve in spacetime at different speeds.

It would be of great interest to study the differential equation of geodesics in the same way.

In a recent work, Defazio (2018) proposes a differential geometric interpretation of Nesterov's method for strongly-convex functions with links to continuous time differential equations mentioned earlier and their Euler discretization.

Second-order methods are desirable because of their fine convergence properties due to dealing with bad-conditioned curvature by using local second-order information.

Hessian-Free optimization (Martens, 2010 ) is based on the truncated-Newton approach where the conjugate gradient algorithm is used to optimize the quadratic approximation of the objective function.

The natural gradient method (Amari, 1998) reformulates the gradient descent in the space of the prediction functions instead of the parameters.

This space is then studied using concepts in differential geometry.

K-FAC (Martens & Grosse, 2015) approximates the Fisher information matrix which is based on the natural gradient method.

Our method is different since we are not using explicit second-order information but rather implicitly deriving curvature information using the change in direction.

We evaluated SGeO on strongly convex functions with Lipschitz gradients and benchmark deep autoencoder problems and compared with the Heavy-Ball and Nesterov's algorithms and K-FAC.

We borrow these three minimization problems from Meng & Chen (2011) where they try to accelerate Nesterov's method by using adaptive step sizes.

The problems are Anisotropic Bowl, Ridge Regression and Smooth-BPDN.

The learning rate for all methods is set to 1 L except for Nesterov which is set to 4 3L+μ and the momentum parameter µ for Heavy Ball, Nesterov and GeO-N is set to the following:

where L is the Lipschitz parameter andμ is the strong-convexity parameter.

The adaptive parameter γ t for Fletcher-Reeves is set to γ

and for GeO and GeO-N is γ C t = 1 −ḡ t ·d t .

The functionspecific parameter α is set to 1, 0.5 and 0.9 in that order for the following problems.

It's important to note that the approximate conjugate gradient method is only exact when an exact line search is used, which is not the case in our experiments with a quadratic function (Ridge Regression).

Anisotropic Bowl The Anisotropic Bowl is a bowl-shaped function with a constraint to get Lipschitz continuous gradients:

As in Meng & Chen (2011) , we set n = 500, τ = 4 and

and µ = 1.

Figure 1 shows the convergence results for our algorithms and the baselines.

The algorithms terminate when f (θ) − f * < 10 −12 .

GeO-N and GeO take only 82 and 205 iterations to converge, while the closest result is that of Heavy-Ball and Fletcher-Reeves which take approximately 2500 and 3000 iterations respectively.

Ridge Regression The Ridge Regression problem is a linear least squares function with Tikhonov regularization:

where A ∈ R m×n is a measurement matrix, b ∈ R m is the response vector and γ > 0 is the ridge parameter.

The function f (θ) is a positive definite quadratic function with the unique solution of

2 + λ and strong convexity parameterμ = λ.

Following Meng & Chen (2011) , m = 1200, n = 2000 and λ = 1.

A is generated from U ΣV T where U ∈ R m×m and V ∈ R n×m are random orthonormal matrices and Σ ∈ R m×m is diagonal with entries linearly distanced in [100, 1] while b = randn(m, 1) is drawn (i.i.d) from the standard normal distribution.

Thusμ = 1 and L ≈ 1001.

Figure 2 shows the results where Fletcher-Reeves, which is a conjugate gradient algorithm, performs better than other methods but we observe similar performances overall except for gradient descent.

The tolerance is set to f (θ) − f * < 10 −13 .

Smooth-BPDN Smooth-BPDN is a smooth and strongly convex version of the BPDN (basis pursuit denoising) problem:

where

Since we cannot find the solution analytically, Nesterov's method is used as an approximation to the solution (f * N ) and the tolerance is set to f (θ) − f * N < 10 −12 .

Figure 3 shows the results for the algorithms.

GeO-N and GeO converge in 308 and 414 iterations respectively, outperforming all other methods.

Closest to these two is Fletcher-Reeves with 569 iterations and Nesterov and Heavy Ball converge similarly in 788 iterations.

To evaluate the performance of SGeO, we apply it to 3 benchmark deep autoencoder problems first introduced in Hinton & Salakhutdinov (2006) which use three datasets, MNIST, FACES and CURVES.

Due to the difficulty of training these networks, they have become standard benchmarks for neural network optimization.

To be consistent with previous literature (Martens, 2010; Sutskever et al., 2013; Martens & Grosse, 2015) , we use the same network architectures as in Hinton & Salakhutdinov (2006) and also report the reconstruction error instead of the log-likelihood objec- Our baselines are the Heavy Ball algorithm (SGD-HB) (Polyak, 1964) , SGD with Nesterov's Momentum (SGD-N) (Sutskever et al., 2013) and K-FAC (Martens & Grosse, 2015) , a second-order method utilizing natural gradients using an approximation of the Fisher information matrix.

Both the baselines and SGeO were implemented using MATLAB on GPU with single precision on a single machine with a 3.6 GHz Intel CPU and an NVIDIA GeForce GTX 1080 Ti GPU with 11 GBs of memory.

The results are shown in Figures 4 to 6.

Since we are mainly interested in optimization and not in generalization, we only report the training error, although we have included the test set performances in the Appendix B. We report the reconstruction relative to the computation time to be able to compare with K-FAC, since each iteration of K-FAC takes orders of magnitude longer than SGD and SGeO. The per-iteration graphs can be found in the Appendix A.

All methods use the same parameter initialization scheme known as "sparse initialization" introduced in Martens (2010) .

The experiments for the Heavy Ball algorithm and SGD with Nesterov's momentum follow Sutskever et al. (2013) which were tuned to maximize performance for these problems.

For SGeO, we chose a fixed momentum parameter and used a simple multiplicative schedule for the learning rate:

where the initial value ( 1 ) was chosen from {0.1,0.15,0.2,0.3,0.4,0.5} and is decayed (K) every 2000 iterations (parameter updates).

The decay parameter (β ) was set to 0.95.

For the momentum parameter µ, we did a search in {0.999,0.995,0.99}. The minibatch size was set to 500 for all methods except K-FAC which uses an exponentially increasing schedule for the minibatch size.

For K-FAC we used the official code provided 1 by the authors with default parameters to reproduce the results.

The version of K-FAC we ran was the Blk-Tri-Diag approach which achieves the best results in all three cases.

To do a fair comparison with other methods, we disabled iterate averaging for K-FAC.

It is also worth noting that K-FAC uses a form of momentum (Martens & Grosse, 2015) .

In all three experiments, SGeO-N is able to outperform the baselines (in terms of reconstruction error) and performs similarly as (if not better than) K-FAC.

We can see the effect of the adaptive coefficient on the Heavy Ball method, i.e. SGeO, which also outperforms SGD with Nesterov's momentum in two of the experiments, MNIST and FACES, and also outperforms K-FAC in the MNIST experiment.

Use of Nesterov style lookahead gradient significantly accelerates training for the MNIST and CURVES dataset, while we see this to a lesser extent in the FACES dataset.

This is also the case for the other baselines (Sutskever et al., 2013; Martens & Grosse, 2015) .

Further, we notice an interesting phenomena for the MNIST dataset (Figure 4) .

Both SGeO and SGeO-N reach very low error rates, after only 900 seconds of training, SGeO and SGeO-N arrive at an error of 0.004 and 0.0002 respectively.

We proposed a novel and efficient algorithm based on adaptive coefficients for the Heavy Ball method inspired by a geodesic optimization algorithm.

We compared SGeO against SGD with Nesterov's Momentum and regular momentum (Heavy Ball) and a recently proposed second-order method, K-FAC, on three deep autoencoder optimization benchmarks and three strongly convex functions with Lipschitz gradients.

We saw that SGeO is able to outperform all first-order methods that we compared to, by a notable margin.

SGeO is easy to implement and the computational overhead it has over the first-order methods, which is calculating the dot product, is marginal.

It can also perform as effectively as or better than second-order methods (here, K-FAC) without the need for expensive higher-order operations in terms of time and memory.

We believe that SGeO opens new and promising directions in high dimensional optimization research and in particular, neural network optimization.

We are working on applying SGeO to other machine learning paradigms such as CNNs, RNNs and Reinforcement Learning.

It remains to analyse the theoretical properties of SGeO such as its convergence rate in convex and non-convex cases which we leave for future work.

Here we include the per-iteration results for the autoencoder experiments in Figures 7 to 9.

We reported the reconstruction error vs. running time in the main text to make it easier to compare to K-FAC.

K-FAC, which is a second-order algorithm, converges in fewer iterations but has a high per-iteration cost.

All other methods are first-order and have similar per-iteration costs.

We include generalization experiments on the test set here.

However, as mentioned before, our focus is optimization and not generalization, we are aware that the choice of optimizer can have a significant effect on the performance of a trained model in practise.

Results are shown in Figures 10 to 12.

SGeO-N shows a significant better predictive performance than SGeO on the CURVES data set and both perform similarly on the two other datasets..

Note that the algorithms are tuned for best performance on the training set.

Overfitting can be dealt with in various ways such as using appropriate regularization during training and using a small validation set to tune the parameters.

C ADAPTIVE COEFFICIENT BEHAVIOUR C.1 GEOMETRIC INTERPRETATION Figure 13 (b) shows the dot product valueḡ ·d which is equivalent to cos (π − φ) (where φ is the angle between the previous update and the negative of the current gradient) for different values of φ.

Figure 13 (a) shows the adaptive coefficient (γ) behaviour for different values of φ for both convex and non-convex cases.

Recall that the adaptive coefficient is used on top the Heavy Ball method.

For strongly convex function with Lipschitz gradients we set γ C = 1 −ḡ ·d and for non-convex cases γ N C = 1 +ḡ ·d.

Here we include the values of the adaptive coefficient during optimization from our experiments.

where w i = 1 + θi−1 4 for i = 1, 2.

We initialize all three methods at (9, 10).

Scaled Goldstein-Price Function The scaled Godstein-Price function (Surjanovic & Bingham; Picheny et al., 2013) features several local minima, ravines and plateaus which can be representative

whereθ i = 4θ i − 2 for i = 1, 2.

We initialize all methods at (1.5, 1.5).

Details The momentum parameter µ for both Nesterov and Geodesic-N was set to 0.9.

The learning rate for all methods is fixed and is tuned for best performance.

The results from both experiments indicate that Geodesic is able to effectively escape local minima and recover from basins of attraction, while Nesterov's method gets stuck at local minima in both cases.

We can also observe the effect of lookahead gradient on our method where the path taken by Geodesic-N is much smoother than Geodesic.

<|TLDR|>

@highlight

We utilize an adaptive coefficient on top of regular momentum inspired by geodesic optimization which significantly speeds up training in both convex and non-convex functions.
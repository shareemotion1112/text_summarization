Careful tuning of the learning rate, or even schedules thereof, can be crucial to effective neural net training.

There has been much recent interest in gradient-based meta-optimization, where one tunes hyperparameters, or even learns an optimizer, in order to minimize the expected loss when the training procedure is unrolled.

But because the training procedure must be unrolled thousands of times, the meta-objective must be defined with an orders-of-magnitude shorter time horizon than is typical for neural net training.

We show that such short-horizon meta-objectives cause a serious bias towards small step sizes, an effect we term short-horizon bias.

We introduce a toy problem, a noisy quadratic cost function, on which we analyze short-horizon bias by deriving and comparing the optimal schedules for short and long time horizons.

We then run meta-optimization experiments (both offline and online) on standard benchmark datasets, showing that meta-optimization chooses too small a learning rate by multiple orders of magnitude, even when run with a moderately long time horizon (100 steps) typical of work in the area.

We believe short-horizon bias is a fundamental problem that needs to be addressed if meta-optimization is to scale to practical neural net training regimes.

The learning rate is one of the most important and frustrating hyperparameters to tune in deep learning.

Too small a value causes slow progress, while too large a value causes fluctuations or even divergence.

While a fixed learning rate often works well for simpler problems, good performance on the ImageNet BID23 benchmark requires a carefully tuned schedule.

A variety of decay schedules have been proposed for different architectures, including polynomial, exponential, staircase, etc.

Learning rate decay is also required to achieve convergence guarantee for stochastic gradient methods under certain conditions BID2 .

Clever learning rate heuristics have resulted in large improvements in training efficiency BID5 BID28 .

A related hyperparameter is momentum; typically fixed to a reasonable value such as 0.9, careful tuning can also give significant performance gains BID29 .

While optimizers such as Adam BID8 are often described as adapting coordinate-specific learning rates, in fact they also have global learning rate and momentum hyperparameters analogously to SGD, and tuning at least the learning rate can be important to good performance.

In light of this, it is not surprising that there have been many attempts to adapt learning rates, either online during optimization BID26 BID25 , or offline by fitting a learning rate schedule BID16 .

More ambitiously, others have attempted to learn an optimizer BID1 BID12 BID4 BID14 BID32 BID20 .

All of these approaches are forms of meta-optimization, where one defines a meta-objective (typically the expected loss after some number of optimization steps) and tunes the hyperparameters to minimize this meta-objective.

But because gradient-based meta-optimization can require thousands of updates, each of which unrolls the entire base-level optimization procedure, the meta-optimization is thousands of times more expensive than the baselevel optimization.

Therefore, the meta-objective must be defined with a much smaller time horizon (e.g. hundreds of updates) than we are ordinarily interested in for large-scale optimization.

The hope is that the learned hyperparameters or optimizer will generalize well to much longer time horizons.

Unfortunately, we show that this is not achieved in this paper.

This is because of a strong tradeoff between short-term and long-term performance, which we refer to as short-horizon bias.

In this work, we investigate the short-horizon bias both mathematically and empirically.

First, we analyze a quadratic cost function with noisy gradients based on BID25 .

We consider this a good proxy for neural net training because secondorder optimization algorithms have been shown to train neural networks in orders-of-magnitude fewer iterations BID17 , suggesting that much of the difficulty of SGD training can be explained by quadratic approximations to the cost.

In our noisy quadratic problem, the dynamics of SGD with momentum can be analyzed exactly, allowing us to derive the greedy-optimal (i.e. 1-step horizon) learning rate and momentum in closed form, as well as to (locally) minimize the long-horizon loss using gradient descent.

We analyze the differences between the short-horizon and long-horizon schedules.

Interestingly, when the noisy quadratic problem is either deterministic or spherical, greedy schedules are optimal.

However, when the problem is both stochastic and badly conditioned (as is most neural net training), the greedy schedules decay the learning rate far too quickly, leading to slow convergence towards the optimum.

This is because reducing the learning rate dampens the fluctuations along high curvature directions, giving it a large immediate reduction in loss.

But this comes at the expense of long-run performance, because the optimizer fails to make progress along low curvature directions.

This phenomenon is illustrated in FIG0 , a noisy quadratic problem in 2 dimensions, in which two learning rate schedule are compared: a small fixed learning rate (blue), versus a larger fixed learning rate (red) followed by exponential decay (yellow).

The latter schedule initially has higher loss, but it makes more progress towards the optimum, such that it achieves an even smaller loss once the learning rate is decayed.

Figure 2 shows this effect quantitatively for a noisy quadratic problem in 1000 dimensions (defined in Section 2.3).

The solid lines show the loss after various numbers of steps of lookahead with a fixed learning rate; if this is used as the meta-objective, it favors small learning rates.

The dashed curves show the loss if the same trajectories are followed by 50 steps with an exponentially decayed learning rate; these curves favor higher learning rates, and bear little obvious relationship to the solid ones.

This illustrates the difficulty of selecting learning rates based on short-horizon information.

Steps= 10 Steps= 30 Steps= 100Figure 2: Short-horizon metaobjectives for the noisy quadratic problem.

Solid: loss after k updates with fixed learning rate.

Dashed: loss after k updates with fixed learning rate, followed by exponential decay.

The second part of our paper empirically investigates gradientbased meta-optimization for neural net training.

We consider two idealized meta-optimization algorithms: an offline algorithm which fits a learning rate decay schedule by running optimization many times from scratch, and an online algorithm which adapts the learning rate during training.

Since our interest is in studying the effect of the meta-objective itself rather than failures of meta-optimization, we give the metaoptimizers sufficient time to optimize their meta-objectives well.

We show that short-horizon meta-optimizers, both online and offline, dramatically underperform a hand-tuned fixed learning rate, and sometimes cause the base-level optimization progress to slow to a crawl, even with moderately long time horizons (e.g. 100 or 1000 steps) similar to those used in prior work on gradient-based meta-optimization.

In short, we expect that any meta-objective which does not correct for short-horizon bias will probably fail when run for a much longer time horizon than it was trained on.

There are applications where short-horizon meta-optimization is directly useful, such as few-shot learning BID24 BID22 .

In those settings, short-horizon bias is by definition not an issue.

But much of the appeal of meta-optimization comes from the possibility of using it to speed up or simplify the training of large neural networks.

In such settings, short-horizon bias is a fundamental obstacle that must be addressed for meta-optimization to be practically useful.

In this section, we consider a toy problem which demonstrates the short-horizon bias and can be analyzed analytically.

In particular, we borrow the noisy quadratic model of BID25 ; the true function being optimized is a quadratic, but in each iteration we observe a noisy version with the correct curvature but a perturbed minimum.

This can be equivalently viewed as noisy observations of the gradient, which are intended to capture the stochasticity of a mini-batch-based optimizer.

We analyze the dynamics of SGD with momentum on this example, and compare the long-horizon-optimized and greedy-optimal learning rate schedules.

Approximating the cost surface of a neural network with a quadratic function has led to powerful insights and algorithms.

Second-order optimization methods such as Newton-Raphson and natural gradient (Amari, 1998) iteratively minimize a quadratic approximation to the cost function.

Hessianfree (H-F) optimization BID17 is an approximate natural gradient method which tries to minimize a quadratic approximation using conjugate gradient.

It can often fit deep neural networks in orders-of-magnitude fewer updates than SGD, suggesting that much of the difficulty of neural net optimization is captured by quadratic models.

In the setting of Bayesian neural networks, quadratic approximations to the log-likelihood motivated the Laplace approximation BID15 and variational inference BID6 BID33 .

BID9 used quadratic approximations to analyze the sensitivity of a neural network's predictions to particular training labels, thereby yielding insight into adversarial examples.

Such quadratic approximations to the cost function have also provided insights into learning rate and momentum adaptation.

In a deterministic setting, under certain conditions, second-order optimization algorithms can be run with a learning rate of 1; for this reason, H-F was able to eliminate the need to tune learning rate or momentum hyperparameters.

BID19 observed that for a deterministic quadratic cost function, greedily choosing the learning rate and momentum to minimize the error on the next step is equivalent to conjugate gradient (CG).

Since CG achieves the minimum possible loss of any gradient-based optimizer on each iteration, the greedily chosen learning rates and momenta are optimal, in the sense that the greedy sequence achieves the minimum possible loss value of any sequence of learning rates and momenta.

This property fails to hold in the stochastic setting, however, and as we show in this section, the greedy choice of learning rate and momentum can do considerably worse than optimal.

Our primary interest in this work is to adapt scalar learning rate and momentum hyperparameters shared across all dimensions.

Some optimizers based on diagonal curvature approximations BID8 have been motivated in terms of adapting dimension-specific learning rates, but in practice, one still needs to tune scalar learning rate and momentum hyperparameters.

Even K-FAC BID19 , which is based on more powerful curvature approximations, has scalar learning rate and momentum hyperparameters.

Our analysis applies to all of these methods since they can be viewed as performing SGD in a preconditioned space.

We will primarily focus on the SGD with momentum algorithm in this paper.

The update is written as follows: DISPLAYFORM0 DISPLAYFORM1 where L is the loss function, t is the training step, and α (t) is the learning rate.

We call the gradient trace v (t) "velocity", and its decay constant µ (t) "momentum".

We denote the ith coordinate of a vector v as v i .

When we focus on a single dimension, we sometimes drop the dimension subscripts.

We also denote DISPLAYFORM2 , where E and V denote expectation and variance respectively.

We now define the noisy quadratic model, where in each iteration, the optimizer is given the gradient for a noisy version of a quadratic cost function, where the curvature is correct but the minimum is sampled stochastically from a Gaussian distribution.

We assume WLOG that the Hessian is diagonal because SGD is a rotation invariant algorithm, and therefore the dynamics can be analyzed in a coordinate system corresponding to the eigenvectors of the Hessian.

We make the further (nontrivial) assumption that the noise covariance is also diagonal.

1 Mathematically, the stochastic cost function is written as:L DISPLAYFORM0 where c is the stochastic minimum, and each c i follows a Gaussian distribution with mean θ * i and variance σ 2 i .

The expected loss is given by: DISPLAYFORM1 The optimum of L is given by θ * = E[c]; we assume WLOG that θ * = 0.

The stochastic gradient is given by DISPLAYFORM2 Since the deterministic gradient is given by ∂L ∂θi = h i θ i , the stochastic gradient can be viewed as a noisy Gaussian observation of the deterministic gradient with variance h We treat the iterate θ (t) as a random variable (where the randomness comes from the sampled c's); the expected loss in each iteration is given by DISPLAYFORM3

We are interested in adapting a global learning rate α (t) and a global momentum decay parameter µ (t) for each time step t. We first derive a recursive formula for the mean and variance of the iterates at each step, and then analyze the greedy-optimal schedule for α (t) and µ (t) .Several observations allow us to compactly model the dynamics of SGD with momentum on the noisy quadratic model.

First, E[L(θ (t) )] can be expressed in terms of E[θ i ] and V[θ i ] using Eqn.

5.

Second, due to the diagonality of the Hessian and the noise covariance matrix, each coordinate evolves independently of the others.

Third, the means and variances of the parameters θ i are functions of those statistics at the previous step.

Because each dimension evolves independently, we now drop the dimension subscripts.

Combining these observations, we model the dynamics of SGD with momentum as a deterministic recurrence relation with sufficient statistics DISPLAYFORM0 ).

The dynamics are as follows:Theorem 1 (Mean and variance dynamics).

The expectations of the parameter θ and the velocity v are updated as, DISPLAYFORM1 The variances of the parameter θ and the velocity v are updated as DISPLAYFORM2 By applying Theorem 1 recursively, we can obtain E[θ (t) ] and V[θ (t) ], and hence E[L(θ (t) )], for every t. Therefore, using gradient-based optimization, we can fit a locally optimal learning rate and momentum schedule, i.e. a sequence of values {( DISPLAYFORM3 ] at some particular time T .

We refer to this as the optimized schedule.

Furthermore, there is a closed-form solution for one-step lookahead, i.e., we can solve for the optimal learning rate α (t) * and momentum DISPLAYFORM4 given the statistics at time t.

We call this as the greedy-optimal schedule.

Theorem 2 (Greedy-optimal learning rate and momentum).

The greedy-optimal learning rate and momentum schedule is given by DISPLAYFORM5 Note that BID25 derived the greedy optimal learning rate for SGD, and Theorem 2 extends it to the greedy optimal learning rate and momentum for SGD with momentum.

As noted in Section 2.1, BID19 found the greedy choice of α and µ to be optimal for gradient descent on deterministic quadratic objectives.

We now show that the greedy schedule is also optimal for SGD without momentum in the case of univariate noisy quadratics, and hence also for multivariate ones with spherical Hessians and gradient covariances.

In particular, the following holds for SGD without momentum on a univariate noisy quadratic: Theorem 3 (Optimal learning rate, univariate).

For all T ∈ N, the sequence of learning rates DISPLAYFORM0 Moreover, this agrees with the greedy-optimal learning rate schedule as derived by BID25 .If the Hessian and the gradient covariance are both spherical, then each dimension evolves identically and independently according to the univariate dynamics.

Of course, one is unlikely to encounter an optimization problem where both are exactly spherical.

But some approximate secondorder optimizers, such as K-FAC, can be viewed as preconditioned SGD, i.e. SGD in a transformed Figure 3 : Comparisons of the optimized learning rates and momenta trained by gradient descent (red), greedy learning rates and momenta (blue), and the optimized fixed learning rate and momentum (green) in both noisy (a) and deterministic (b) quadratic settings.

In the deterministic case, our optimized schedule matched the greedy one, just as the theory predicts.space where the Hessian and the gradient covariance are better conditioned BID19 .

In principle, with a good enough preconditioner, the Hessian and the gradient covariance would be close enough to spherical that a greedy choice of α and µ would perform well.

It will be interesting to investigate whether any practical optimization algorithms demonstrate this behavior.

In this section, we compare the optimized and greedy-optimal schedules on a noisy quadratic problem.

We chose a 1000 dimensional quadratic cost function with the curvature distribution from BID13 , on which CG achieves its worst-case convergence rate.

We assume that h i = V[ ∂L ∂θi ], and hence σ 2 i = 1 hi ; this choice is motivated by the observations that under certain assumptions, the Fisher information matrix is a good approximation to the Hessian matrix, but also reflects the covariance structure of the gradient noise BID18 .

We computed the greedy-optimal schedules using Theorem 3.

For the optimized schedules, we minimized the expected loss at time T = 250 using Adam using Adam BID8 , with a learning rate 0.003 and 500 steps.

We set an upper bound for the learning rate which prevented the loss component for any dimension from becoming larger than its initial value; this was needed because otherwise the optimized schedule allowed the loss to temporarily grow very large, a pathological solution which would be unstable on realistic problems.

We also considered fixed learning rate and momentum, with the two hyperparameters fit using Adam.

The training curves and the corresponding learning rates and momenta are shown in Figure 3(a) .

The optimized schedule achieved a much lower final expected loss value (4.25) than was obtained by the greedy-optimal schedule (63.86) or fixed schedule (42.19).We also show the sums of the losses along the 50 highest curvature directions and 50 lowest curvature directions.

We find that under the optimized schedule, the losses along the high curvature directions hardly decrease initially.

However, because it maintains a high learning rate, the losses along the low curvature directions decrease significantly.

After 50 iterations, it begins decaying the learning rate, at which point it achieves a large drop in both the high-curvature and total losses.

On the other hand, under the greedy-optimal schedule, the learning rates and momenta become small very early on, which immediately reduces the losses on the high curvature directions, and hence also the total loss.

However, in the long term, since the learning rates are too small to make substantial progress along the low curvature directions, the total loss converged to a much higher value in the end.

This gives valuable insight into the nature of the short-horizon bias in meta-optimization: shorthorizon objectives will often encourage the learning rate and momentum to decay quickly, so as to achieve the largest gain in the short term, but at the expense of long-run performance.

It is interesting to compare this behavior with the deterministic case.

We repeated the above experiment for a deterministic quadratic cost function (i.e. σ 2 i = 0) with the same Hessian; results are shown in Figure 3(b) .

The greedy schedule matches the optimized one, as predicted by the analysis of BID19 .

This result illustrates that stochasticity is necessary for short-horizon bias to manifest.

Interestingly, the learning rate and momentum schedules in the deterministic case are nearly flat, while the optimized schedules for the stochastic case are much more complex, suggesting that stochastic optimization raises a different set of issues for hyperparameter adaptation.

We now turn our attention to gradient-based hyperparameter optimization.

A variety of approaches have been proposed which tune hyperparameters by doing gradient descent on a meta-objective BID26 BID16 BID1 .

We empirically analyze an idealized version of a gradient-based meta-optimization algorithm called stochastic meta-descent (SMD) BID26 .

Our version of SMD is idealized in two ways: first, we drop the algorithmic tricks used in prior work, and instead allow the meta-optimizer more memory and computation than would be economical in practice.

Second, we limit the representational power of our meta-model: whereas BID1 aimed to learn a full optimization algorithm, we focus on the much simpler problem of adapting learning rate and momentum hyperparameters, or schedules thereof.

The aim of these two simplifications is that we would like to do a good enough job of optimizing the meta-objective that any base-level optimization failures can be attributed to deficiencies in the meta-objective itself (such as short-horizon bias) rather than incomplete metaoptimization.

Despite these simplifications, we believe our experiments are relevant to practical meta-optimization algorithms which optimize the meta-objective less thoroughly.

Since the goal of the metaoptimizer is to adapt two hyperparameters, it's possible that poor meta-optimization could cause the hyperparameters to get stuck in regions that happen to perform well; indeed, we observed this phenomenon in some of our early explorations.

But it would be dangerous to rely on poor meta-optimization, since improved meta-optimization methods would then lead to worse base-level performance, and tuning the meta-optimizer could become a roundabout way of tuning learning rates and momenta.

We also believe our experiments are relevant to meta-optimization methods which aim to learn entire algorithms.

Even if the learned algorithms don't have explicit learning rate parameters, it's possible for a learning rate schedule to be encoded into an algorithm itself; for instance, Adagrad BID3 implicitly uses a polynomial decay schedule because it sums rather than averages the squared derivatives in the denominator.

Hence, one would need to worry about whether the metaoptimizer is implicitly fitting a learning rate schedule that's optimized for short-term performance.

The high-level idea of stochastic meta-descent (SMD) BID26 is to perform gradient descent on the learning rate, or any other differentiable hyperparameters.

This is feasible since any gradient based optimization algorithm can be unrolled as a computation graph (see FIG5 , and automatic differentiation is readily available in most deep learning libraries.

There are two basic types of automatic differentiation (autodiff) methods: forward mode and reverse mode.

In forward mode autodiff, directional derivatives are computed alongside the forward computation.

In contrast, reverse mode autodiff (a.k.a.

backpropagation) computes the gradients moving backwards through the computation graph.

Meta-optimization using reverse mode can be computationally demanding due to memory constraints, since the parameters need to be stored at every step.

BID16 got around this by cleverly exploiting approximate reversibility to minimize the memory cost of activations.

Since we are optimizing only two hyperparameters, however, forward mode autodiff can be done cheaply.

Here, we provide the forward differentiation equations for obtaining the gradient of vanilla SGD learning rate.

Let dθt dα be u t , and dLt dα be α , and the Hessian at step t to be H t .

By chain rule, we get, DISPLAYFORM0 While the Hessian is infeasible to construct explicitly, the Hessian-vector product in Equation 9 can be computed efficiently using reverse-on-reverse BID31 or forward-on-reverse automatic differentiation BID21 , in time linear in the cost of the forward pass.

See Schraudolph FORMULA1 for more details.

DISPLAYFORM1 Using the gradients with respect to hyperparameters, as given in Eq. 9, we can apply gradient based meta-optimization, just like optimizing regular parameters.

It is worth noting that, although SMD was originally proposed for optimizing vanilla SGD, in practice it can be applied to other optimization algorithms such as SGD with momentum or Adam BID8 .

Moreover, gradient-based optimizers other than SGD can be used for the meta-optimization as well.

The basic SMD algorithm is given as Algorithm 1.

Here, α is a set of hyperparameters (e.g. learning rate), and α 0 are inital hyperparameter values; θ is a set of optimization intermediate variables, such as weights and velocities; η is a set of metaoptimizer hyperparameters (e.g. meta learning rate).

BGrad(y, x, dy) is the backward gradient function that computes the gradients of the loss function wrt.

θ, and FGrad(y, x, dx) is the forward gradient function that accumulates the gradients of θ with respect to α.

Step and MetaStep optimize regular parameters and hyperparameters, respectively, for one step using gradient-based methods.

Additionally, T is the lookahead window size, and M is the number of meta updates.

Simplifications from the original SMD algorithm.

The original SMD algorithm BID26 fit coordinate-wise adaptive learning rates with intermediate gradients (u t ) accumulated throughout the process of training.

Since computing separate directional derivatives for each coordinate using forward mode autodiff is computationally prohibitive, the algorithm used approximate updates.

Both features introduced bias into the meta-gradients.

We make several changes to the original algorithm.

First, we tune only a global learning rate parameter.

Second, we use exact forward mode accumulation because this is feasible for a single learning rate.

Third, rather than accumulate directional derivatives during training, we compute the meta-updates on separate SGD trajectories simulated using fixed network parameters.

Finally, we compute multiple meta-updates in order to ensure that the meta-objective is optimized sufficiently well.

Together, these changes ensure unbiased meta-gradients, as well as careful optimization of the meta-objective, at the cost of high computational overhead.

We do not recommend this approach as a practical SMD implementation, but rather as a way of understanding the biases in the meta-objective itself.

To understand the sensitivity of the optimized hyperparameters to the horizon, we first carried out an offline experiment on a multi-layered perceptron (MLP) on MNIST BID11 .

Specifically, we fit learning rate decay schedules offline by repeatedly training the network, and a single meta-gradient was obtained from each training run.

Learnable decay schedule.

We used a parametric learning rate decay schedule known as inverse time decay BID30 : α t = α0 (1+ t K ) β , where α 0 is the initial learning rate, t is the number of training steps, β is the learning rate decay exponent, and K is the time constant.

We jointly optimized α 0 and β.

We fixed µ = 0.9, K = 5000 for simplicity.

Experimental details.

The network had two layers of 100 hidden units, with ReLU activations.

Weights were initialized with a zero-mean Gaussian with standard deviation 0.1.

We used a warm start from a network trained for 50 SGD with momentum steps, using α = 0.1, µ = 0.9.

(We used a warm start because the dynamics are generally different at the very start of training.)

For SMD optimization, we trained all hyperparameters in log space using Adam optimizer, with 5k meta steps.

FIG6 shows SMD optimization trajectories on the meta-objective surfaces, initialized with multiple random hyperparameter settings.

The SMD trajectories appear to have converged to the global optimum.

Importantly, the meta-objectives with longer horizons favored a much smaller learning rate decay exponent β, leading to a more gradual decay schedule.

The meta-objective surfaces were very different depending on the time horizon, and the final β value differed by over two orders of magnitude between 100 and 20k step horizons.

We picked the best learning rate schedules from meta-objective surfaces (in FIG6 , and obtained the training curves of a network shown in FIG7 .

The resulting training loss at 20k steps with the 100 step horizon was over three orders of magnitude larger than with the 20k step horizon.

In general, short horizons gave better performance initially, but were surpassed by longer horizons.

The differences in error were less drastic, but we see that the 100 step network was severely undertrained, and the 1k step network achieved noticeably worse test error than the longer-horizon ones.

In this section, we study whether online adaptation also suffers from short-horizon bias.

Specifically, we used Algorithm 1) to adapt the learning rate and momentum hyperparameters online while a network is trained.

We experimented with an MLP on MNIST and a CNN on CIFAR-10 (Krizhevsky, 2009).Experimental details.

For the MNIST experiments, we used an MLP network with two hidden layers of 100 units, with ReLU activations.

Weights were initialized with a zero-mean Gaussian with standard deviation 0.1.

For CIFAR-10 experiments, we used a CNN network adapted from Caffe BID7 , with 3 convolutional layers of filter size 3 × 3 and depth [32, 32, 64] , and 2 × 2 max pooling with stride 2 after every convolution layer, and follwed by a fully connected hidden layer of 100 units.

Meta-optimization was done with 100 steps of Adam for every 10 steps of regular training.

We adapted the learning rate α and momentum µ. After 25k steps, adaptation was stopped, and we trained for another 25k steps with an exponentially decaying learning rate such that it reached 1e-4 on the last time step.

We re-parameterized the learning rate with the effective learning rate α eff = α 1−µ , and the momentum with 1 − µ, so that they can be optimized more smoothly in the log space.

FIG8 shows training curves both with online SMD and with hand-tuned fixed learning rate and momentum hyperparameters.

We show several SMD runs initialized from widely varying hyperparameters; all the SMD runs behaved similarly, suggesting it optimized the meta-objective efficiently enough.

Under SMD, learning rates were quickly decreased to very small values, leading to slow progress in the long term, consistent with the noisy quadratic and offline adaptation experiments.

As online SMD can be too conservative in the choice of learning rate, it is natural to ask whether removing the stochasticity in the lookahead sequence can fix the problem.

We therefore considered online SMD where the entire lookahead trajectory used a single mini-batch, hence removing the stochasticity.

As shown in FIG9 , this deterministic lookahead scheme led to the opposite problem: the adapted learning rates were very large, leading to instability.

We conclude that the stochasticity of mini-batch training cannot be simply ignored in meta-optimization.

In this paper, we analyzed the problem of short-horizon bias in meta-optimization.

We presented a noisy quadratic toy problem which we analyzed mathematically, and observed that the optimal learning rate schedule differs greatly from a greedy schedule that minimizes training loss one step ahead.

While the greedy schedule tends to decay the learning rate drastically to reduce the loss on high curvature directions, the optimal schedule keeps a high learning rate in order to make steady progress on low curvature directions, and eventually achieves far lower loss.

We showed that this bias stems from the combination of stochasticity and ill-conditioning: when the problem is either deterministic or spherical, the greedy learning rate schedule is globally optimal; however, when the problem is both stochastic and ill-conditioned (as is most neural net training), the greedy schedule performs poorly.

We empirially verified the short-horizon bias in the context of neural net training by applying gradient based meta-optimization, both offline and online.

We found the same pathological behaviors as in the noisy quadratic problem -a fast learning rate decay and poor long-run performance.

While our results suggest that meta-optimization should not be applied blindly, our noisy quadratic analysis also provides grounds for optimism: by removing ill-conditioning (by using a good preconditioner) and/or stochasticity (with large batch sizes or variance reduction techniques), it may be possible to enter the regime where short-horizon meta-optimization works well.

It remains to be seen whether this is achievable with existing optimization algorithms.

We calculate the mean of the parameter θ (t+1) , DISPLAYFORM0 Let's assume the following initial conditions: DISPLAYFORM1 Then Eq.(10) and Eq.(11) describes how E θ (t) , E v (t) changes over time t.

We calculate the variance of the velocity v (t+1) , DISPLAYFORM0 The variance of the parameter θ (t+1) is given by, DISPLAYFORM1 We also need to derive how the covariance of θ and v changes over time: DISPLAYFORM2 Let's assume the following initial conditions: DISPLAYFORM3 Combining Eq. FIG0 , we obtain the following dynamics (from t = 0, . . .

, T − 1): DISPLAYFORM4 Published as a conference paper at ICLR 2018Thus we can write A We now generalize the above derivation.

First rewrite L min in terms of A T −k min and calculate the optimal learning rate at time step T − k. Theorem 4.

For all T ∈ N, and k ∈ N, 1 ≤ k ≤ T , we have, DISPLAYFORM5 Therefore, the optimal learning α (t) at timestep t is given as, DISPLAYFORM6 h(A (t) + σ 2 ) .Proof.

The form of L min can be easily proven by induction on k, and use the identity that, and setting it to zero.

Note that the subscript min is omitted from A (t) in Eq. FORMULA0 as we assume all A (t) are obtained using optimal α * , and hence minimum.

<|TLDR|>

@highlight

We investigate the bias in the short-horizon meta-optimization objective.

@highlight

This paper proposes a simplified model and problem to demonstrate the short-horizon bias of the learning rate meta-optimization.

@highlight

This paper studies the issue of truncated backpropagation for meta-optimization through a number of experiments on a toy problem
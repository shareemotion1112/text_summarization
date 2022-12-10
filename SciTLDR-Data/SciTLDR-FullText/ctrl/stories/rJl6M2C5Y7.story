Effective performance of neural networks depends critically on effective tuning of optimization hyperparameters, especially learning rates (and schedules thereof).

We present Amortized Proximal Optimization (APO), which takes the perspective that each optimization step should approximately minimize a proximal objective (similar to the ones used to motivate natural gradient and trust region policy optimization).

Optimization hyperparameters are adapted to best minimize the proximal objective after one weight update.

We show that an idealized version of APO (where an oracle minimizes the proximal objective exactly) achieves global convergence to stationary point and locally second-order convergence to global optimum for neural networks.

APO incurs minimal computational overhead.

We experiment with using APO to adapt a variety of optimization hyperparameters online during training, including (possibly layer-specific) learning rates, damping coefficients, and gradient variance exponents.

For a variety of network architectures and optimization algorithms (including SGD, RMSprop, and K-FAC), we show that with minimal tuning, APO performs competitively with carefully tuned optimizers.

Tuning optimization hyperparameters can be crucial for effective performance of a deep learning system.

Most famously, carefully selected learning rate schedules have been instrumental in achieving state-of-the-art performance on challenging datasets such as ImageNet BID6 and WMT BID36 .

Even algorithms such as RMSprop BID34 and Adam (Kingma & Ba, 2015) , which are often interpreted in terms of coordinatewise adaptive learning rates, still have a global learning rate parameter which is important to tune.

A wide variety of learning rate schedules have been proposed BID24 BID14 BID2 .

Seemingly unrelated phenomena have been explained in terms of effective learning rate schedules BID35 .

Besides learning rates, other hyperparameters have been identified as important, such as the momentum decay factor BID31 , the batch size BID28 , and the damping coefficient in second-order methods BID20 BID19 .There have been many attempts to adapt optimization hyperparameters to minimize the training error after a small number of updates BID24 BID1 BID2 .

This approach faces two fundamental obstacles: first, learning rates and batch sizes have been shown to affect generalization performance because stochastic updates have a regularizing effect BID5 BID18 BID27 BID35 .

Second, minimizing the short-horizon expected loss encourages taking very small steps to reduce fluctuations at the expense of long-term progress BID37 .

While these effects are specific to learning rates, they present fundamental obstacles to tuning any optimization hyperparameter, since basically any optimization hyperparameter somehow influences the size of the updates.

In this paper, we take the perspective that the optimizer's job in each iteration is to approximately minimize a proximal objective which trades off the loss on the current batch with the average change in the predictions.

Specifically, we consider proximal objectives of the form J(φ) = h(f (g(θ, φ))) + λD(f (θ), f (g(θ, φ))), where f is a model with parameters θ, h is an approximation to the objective function, g is the base optimizer update with hyperparameters φ, and D is a distance metric.

Indeed, approximately solving such a proximal objective motivated the natural gradient algorithm BID0 , as well as proximal reinforcement learning algorithms BID26 .

We introduce Amortized Proximal Optimization (APO), an approach which adapts optimization hyperparameters to minimize the proximal objective in each iteration.

We use APO to tune hyperparameters of SGD, RMSprop, and K-FAC; the hyperparameters we consider include (possibly layer-specific) learning rates, damping coefficients, and the power applied to the gradient covariances.

Notice that APO has a hyperparameter λ which controls the aggressiveness of the updates.

We believe such a hyperparameter is necessary until the aforementioned issues surrounding stochastic regularization and short-horizon bias are better understood.

However, in practice we find that by performing a simple grid search over λ, we can obtain automatically-tuned learning rate schedules that are competitive with manual learning rate decay schedules.

Furthermore, APO can automatically adapt several optimization hyperparameters with only a single hand-tuned hyperparameter.

We provide theoretical justification for APO by proving strong convergence results for an oracle which solves the proximal objective exactly in each iteration.

In particular, we show global linear convergence and locally quadratic convergence under mild assumptions.

These results motivate the proximal objective as a useful target for meta-optimization.

We evaluate APO on real-world tasks including image classification on MNIST, CIFAR-10, CIFAR-100, and SVHN.

We show that adapting learning rates online via APO yields faster training convergence than the best fixed learning rates for each task, and is competitive with manual learning rate decay schedules.

Although we focus on fast optimization of the training objective, we also find that the solutions found by APO generalize at least as well as those found by fixed hyperparameters or fixed schedules.

We view a neural network as a parameterized function z = f (x, θ), where x is the input, θ are the weights and biases of the network, and z can be interpreted as the output of a regression model or the un-normalized log-probabilities of a classification model.

Let the training dataset be {( DISPLAYFORM0 , where input x i is associated with target t i .

Our goal is to minimize the loss function: DISPLAYFORM1 where Z is the matrix of network outputs on all training examples x 1 , . . .

, x N , and T is the vector of labels.

We design an iterative optimization algorithm to minimize Eq. 1 under the following framework: in the kth iteration, one aims to update θ to minimize the following proximal objective: DISPLAYFORM2 where x is the data used in the current iteration, P is the distribution of data, θ k is the parameters of the neural network at the current iteration, h(·) is some approximation of the loss function, and D(·, ·) represents the distance between network outputs under some metric (for notational convenience, we use mini-batch size of 1 to describe the algorithm).

We first provide the motivation for this proximal objective in Section 2.1; then in Section 2.2, we propose an algorithm to optimize it in an online manner.

In this section, we show that by approximately minimizing simple instances of Eq. 2 in each iteration (similar to BID25 ), one can recover the classic Gauss-Newton algorithm and Natural Gradient Descent BID0 .

In general, updating θ so as to minimize the proximal objective is impractical due to the complicated nonlinear relationship between θ and z. However, one can find an approximate solution by linearizing the network function: DISPLAYFORM0 where J = ∇ θ f (x, θ) is the Jacobian matrix.

We consider the following instance of Eq. 2: DISPLAYFORM1 where ∆z f (x, θ) − f (x, θ k ) is the change of network output, t is the label of current data x. Here h(·) is defined as the first-order Taylor approximation of the loss function.

Using the linear approximation (Eq. 3), and a local second-order approximation of D, this proximal objective can be written as: DISPLAYFORM2 DISPLAYFORM3 is the Hessian matrix of the dissimilarity measured atz = f (x, θ k ).Solving Eq. 5 yields: DISPLAYFORM4 where G Ex ∼P J ∇ 2DJ is the pre-conditioning matrix.

Different settings for the dissimilarity DISPLAYFORM5 (7) is defined as the squared Euclidean distance, Eq. 6 recovers the classic Gauss-Newton algorithm.

When DISPLAYFORM6 is defined as the Bregman divergence, Eq. 6 yields the Generalized Gauss-Newton (GGN) method.

When the output of neural network parameterizes an exponential-family distribution, the dissimilarity term can be defined as Kullback-Leibler divergence: DISPLAYFORM7 in which case Eq. 6 yields Natural Gradient Descent BID0 .

Since different versions of our proximal objective lead to various efficient optimization algorithms, we believe it is a useful target for meta-optimization.

Although optimizers including the Gauss-Newton algorithm and Natural Gradient Descent can be seen as ways to approximately solve Eq. 2, they rely on a local linearization of the neural network and usually require more memory and more careful tuning in practice.

We propose to instead directly minimize Eq. 2 in an online manner.

Finding good hyperparameters (e.g., the learning rate for SGD) is a challenging problem in practice.

We propose to adapt these hyperparameters online in order to best optimize the proximal objective.

Consider any optimization algorithm (base-optimizer) of the following form: θ ← g(x, t, θ, ξ, φ).(10) Here, θ is the set of model parameters, x is the data used in this iteration, t is the corresponding label, ξ is a vector of statistics computed online during optimization, and φ is a vector of optimization hyperparameters to be tuned.

For example, ξ contains the exponential moving averages of the squared gradients of the parameters in RMSprop.

φ usually contains the learning rate (global or layer-specific), and possibly other hyperparameters dependent on the algorithm.

For each step, we formulate the meta-objective from Eq. 2 as follows (for notational convenience we omit variables other than θ and φ of g): DISPLAYFORM0 Here,x is a random mini-batch sampled from the data distribution P. We compute the approximation to the loss, h, using the same mini-batch as the gradient of the base optimizer, to avoid the short horizon bias problem BID37 ; we measure D on a different mini-batch to avoid instability that would result if we took a large step in a direction that is unimportant for the current batch, but important for other batches.

The hyperparameters φ are optimized using a stochastic gradient-based algorithm (the meta-optimizer) using the gradient ∇ φ J(φ) (similar in spirit to BID24 BID17 ).

We refer to our framework as Amortized Proximal Optimization (APO).

The simplest version of APO, which uses SGD as the meta-optimizer, is shown in Algorithm 1.

One can choose any meta-optimizer; we found that RMSprop was the most stable and best-performing meta-optimizer in practice, and we used it for all our experiments.

DISPLAYFORM1

When considering optimization meta-objectives, it is useful to analyze idealized versions where the meta-objective is optimized exactly (even when doing so is prohibitively expensive in practice).

For instance, BID37 analyzed an idealized SMD algorithm, showing that even the idealized version suffered from short-horizon bias.

In this section, we analyze two idealized versions of APO where an oracle is assumed to minimize the proximal objective exactly in each iteration.

In both cases, we obtain strong convergence results, suggesting that our proximal objective is a useful target for meta-optimization.

We view the problem in output space (i.e., explicitly designing an update schedule for z i ).

Consider the space of outputs on all training examples; when we train a neural network, we are optimizing over a manifold in this space: DISPLAYFORM0 We assume that f is continuous, so that M is a continuous manifold.

Given an oracle that for each iteration exactly minimizes the expectation of proximal objective Eq. 2 over the dataset, we can write one iteration of APO in output space as: DISPLAYFORM1 where z i is ith column of Z, corresponding to the network output on data x i after update, z k,i is the current network output on data x i .

We first define the proximal objective as Eq. 4, using the Euclidean distance as the dissimilarity measure, which corresponds to Gauss-Newton algorithm under the linearization of network.

With an oracle, this proximal objective leads to projected gradient descent: DISPLAYFORM0 Consider a loss function on one data point (z) : R d → R, where d is the dimension of neural network's output.

1 We say the gradient is L-Lipschitz if: DISPLAYFORM1 When the manifold M is smooth, a curve in M is called geodesic if it is the shortest curve connecting the starting and the ending point.

We say M have a C-bounded curvature if for each trajectory DISPLAYFORM2 going along some geodesic and v(t) 2 = 1, there is v(t) ≤ C with spectral norm.

For each point Z ∈ M, consider the tangent space at point Z as T Z M. We call the projection of ∇ L(Z) onto the hyperplane T Z M as the effective gradient of L at Z ∈ M.

It is worth noting that zero effective gradient corresponds to stationary point of the neural network.

We have the following theorem stating the global convergence of Eq. 14 to stationary point: Theorem 1.

Assume the loss satisfies A1.

Furthermore, assume L is lower bounded by L * and has gradient norm upper bound G. Let g * T be the effective gradient in the first T iterations with minimal norm.

When the manifold is smooth with C-bounded curvature, with λ ≥ {CG, L 4 }, the norm of g * T converges with rate O 1 T as: DISPLAYFORM3 This convergence result differs from usual neural network convergence results, because here the Lipschitz constants are defined for the output space, so they are known and generally nice.

For instance, L = 1 when we use a quadratic loss.

In contrast, the gradient is in general not Lipschitz continuous in weight space for deep networks.

We further replace the dissimilarity term with: DISPLAYFORM0 which is the second-order approximation of Eq. 8.

With a proximal oracle, this variant of APO turns out to be Proximal Newton Method in the output space, if we set λ = 1 2 : DISPLAYFORM1 where DISPLAYFORM2 H is the norm with local Hessian as metric.

In general, Newton's method can't be applied directly to neural nets in weight space, because it is nonconvex BID4 .

However, Proximal Newton Method in output space can be efficient given a strongly convex loss function.

Consider a loss (z) with µ-strongly convex: DISPLAYFORM3 where z * is the unique minimizer and µ is some positive real number, and L H -smooth Hessian: for any vector v ∈ R d such that v = 1, there is: DISPLAYFORM4 The following theorem suggests the locally fast convergence rate of iteration Eq. 17: Theorem 2.

Under assumptions A2 and A3, if the unique minimum Z * ∈ M, then whenever iteration (17) converges to Z * , it converges locally quadratically 2 : DISPLAYFORM5 Hence, the proximal oracle achieves second-order convergence for neural network training under fairly reasonable assumptions.

Of course, we don't expect practical implementations of APO (or any other practical optimization method for neural nets) to achieve the second-order convergence rates, but we believe the second-order convergence result still motivates our proximal objective as a useful target for meta-optimization.

Finding good optimization hyperparameters is a longstanding problem BID3 .

Classic methods for hyperparameter optimization, such as grid search, random search, and Bayesian optimization BID29 BID33 , are expensive, as they require performing many complete training runs, and can only find fixed hyperparameter values (e.g., a constant learning rate).

Hyperband can reduce the cost by terminating poorly-performing runs early, but is still limited to finding fixed hyperparameters.

Population Based Training (PBT) BID10 ) trains a population of networks simultaneously, and throughout training it terminates poorly-performing networks, replaces their weights by a copy of the weights of a better-performing network, perturbs the hyperparameters, and continues training from that point.

PBT can find a coarse-grained learning rate schedule, but because it relies on random search, it is far less efficient than gradient-based meta-optimization.

There have been a number of approaches to gradient-based adaptation of learning rates.

Gradientbased optimization algorithms can be unrolled as computation graphs, allowing the gradients of hyperparameters such as learning rates to be computed via automatic differentiation.

BID17 propagate gradients through the full unrolled training procedure to find optimal learning rate schedules offline.

Stochastic meta-descent (SMD) (Schraudolph, 1999) adapts hyperparameters online.

Hypergradient descent (HD) BID2 takes the gradient of the learning rate with respect to the optimizer update in each iteration, to minimize the expected loss in the next iteration.

In particular, HD suffers from short horizon bias BID37 , while in Appendix F we show that APO does not.

Some authors have proposed learning entire optimization algorithms BID14 BID35 BID1 .

BID14 view this problem from a reinforcement learning perspective, where the state consists of the objective function L and the sequence of prior iterates {θ t } and gradients {∇ θ L(θ t )}, and the action is the step ∆θ.

In this setting, the update rule φ is a policy, which can be found via policy gradient methods BID32 .

Approaches that learn optimizers must be trained on a set of objective functions {f 1 , . . .

, f n } drawn from a distribution F; this setup can be restrictive if we only have one instance of an objective function.

In addition, the initial phase of training the optimizer on a distribution of functions can be expensive.

APO requires only the objective function of interest and finds learning rate schedules in a single training run.

In principle, APO could be used to learn a full optimization algorithm; however, learning such an algorithm would be just as hard as the original optimization problem, so one would not expect an out-of-the-box meta-optimizer (such as RMSprop with learning rate 0.001) to work as well as it does for adapting few hyperparameters.

In this section, we evaluate APO empirically on a variety of learning tasks; Table 1 gives an overview of the datasets, model architectures, and base optimizers we consider.

In our proximal objective, DISPLAYFORM0 , h can be any approximation to the loss function (e.g., a linearization); in our experiments, we directly used the loss value h = , as we found this to work well in many settings.

As the dissimilarity term D, we used the squared Euclidean norm.

We used APO to tune the optimization hyperparameters of four base-optimizers: SGD, SGD with Nesterov momentum (denoted SGDm), RMSprop, and K-FAC.

For SGD, the only hyperparameter is the learning rate; we consider both a single, global learning rate, as well as per-layer learning rates.

For SGDm, the update rule is given by: DISPLAYFORM1 where g = ∇ .

Since adapting µ requires considering long-term performance BID31 , it is not appropriate to adapt it with a one-step objective like APO.

Instead, we just adapt the learning rate with APO as if there's no momentum, but then apply momentum with µ = 0.9 on top of the updates.

For RMSprop, the optimizer step is given by: DISPLAYFORM2 We note that, in addition to the learning rate η, we can also consider adapting and the power to which s is raised in the denominator of Eq. 21-we denote this parameter ρ, where in standard RMSprop we have ρ = 1 2 .

Both and ρ can be interpreted as having a damping effect on the update.

K-FAC is an approximate natural gradient method (Amari, 1998) based on preconditioning the gradient by an approximation to the Fisher matrix, θ ← θ − F −1 ∇ .

For K-FAC, we tune the global learning rate and the damping factor.

Meta-Optimization Setup.

Throughout this section, we use the following setup for metaoptimization: we use RMSprop as the meta-optimizer, with learning rate 0.1, and perform 1 metaoptimization update for every 10 steps of the base optimization.

We show in Appendix E that with this default configuration, APO is robust to the initial learning rate of the base optimizer.

Each meta-optimization step takes approximately the same amount of computation as a base optimization step; by performing meta-updates once per 10 base optimization steps, the computational overhead of using APO is just a small fraction more than the original training procedure.

Rosenbrock.

We first validated APO on the two-dimensional Rosenbrock function, f (x, y) = (1 − x) 2 + 100(y − x 2 ) 2 , with initialization (x, y) = (1, −1.5).

We used APO to tune the learning rate of RMSprop, and compared to standard RMSprop with several fixed learning rates.

Because this problem is deterministic, we set λ = 0 for APO.

FIG0 shows that RMSprop-APO was able to achieve a substantially lower objective value than the baseline RMSprop.

The learning rates for each method are shown in FIG0 ; we found that APO first increases the learning rate to make rapid progress at the start of optimization, and then gradually decreases it as it approaches the local optimum.

In Appendix D we show that APO converges quickly from many different locations on the Rosenbrock surface, and in Appendix E we show that APO is robust to the initial learning rate of the base optimizer.

Badly-Conditioned Regression.

Next, we evaluated APO on a badly-conditioned regression problem BID22 , which is intended to be a difficult test problem for optimization algorithms.

In this problem, we consider a dataset of input/output pairs {(x, y)}, where the outputs are given by y = Ax, where A is an ill-conditioned matrix with κ(A) = 10 10 .

The task is to fit a two-layer linear model f (x) = W 2 W 1 x to this data; the loss to be minimized is FIG0 (c) compares the performance of RMSprop with a hand-tuned fixed learning rate to the performance of RMSprop-APO, with learning rates shown in FIG0 .

Again, the adaptive learning rate enabled RMSprop-APO to achieve a loss value orders of magnitude smaller than that achieved by RMSprop with a fixed learning rate.

DISPLAYFORM0

For each of the real-world datasets we consider-MNIST, CIFAR-10, CIFAR-100, SVHN, and FashionMNIST-we chose the learning rates for the baseline optimizers via grid searches: for SGD and SGDm, we performed a grid search over learning rates {0.1, 0.01, 0.001}, while for RMSprop, we performed a grid search over learning rates {0.01, 0.001, 0.0001}. For SGD-APO and SGDm-APO, we set the initial learning rate to 0.1, while for RMSprop-APO, we set the initial learning rate to 0.0001.

These initial learning rates are used for convenience; we show in Appendix E that APO is robust to the choice of initial learning rate.

The only hyperparameter we consider for APO is the value of λ: for SGD-APO and SGDm-APO, we select the best λ from a grid search over {0.1, 0.01, 1e-3}; for RMSprop, we choose λ from a grid search over {0.1, 0.01, 1e-3, 1e-4, 1e-5, 0}. Note that because each value of λ yields a learning rate schedule, performing a search over λ is much more effective than searching over fixed learning rates.

In particular, we show that the adaptive learning rate schedules discovered by APO are competitive with manual learning rate schedules.

First, we compare SGD and RMSprop with their APO-tuned variants on MNIST, and show that APO outperforms fixed learning rates.

As the classification network for MNIST, we used a twolayer MLP with 1000 hidden units per layer and ReLU nonlinearities.

We trained on mini-batches of size 100 for 100 epochs.

SGD with APO.

We used APO to tune the global learning rate of SGD and SGD with Nesterov momentum (denoted SGDm) on MNIST, where the momentum is fixed to 0.9.

For baseline SGDm, we used learning rate 0.01, while for baseline SGD, we used both learning rates 0.1 and 0.01.

The training curve of SGD with learning rate 0.1 almost coincides with that of SGDm with learning rate 0.01.

For SGD-APO, the best λ was 1e-3, while for SGDm-APO, the best λ was 0.1.

A comparison of the algorithms is shown in FIG1 .

APO substantially improved the training loss for both SGD and SGDm.

We compare K-FAC with a fixed learning rate and a manual learning rate schedule to APO, used to tune 1) the learning rate; and 2) both the learning rate and damping coefficient.

RMSprop with APO.

Next, we used APO to tune the global learning rate of RMSprop.

For baseline RMSprop, the best fixed learning rate was 1e-4, while for RMSprop-APO, the best λ was 1e-5.

FIG1 (b) compares RMSprop and its APO-tuned variant on MNIST.

RMSprop-APO achieved a training loss about three orders of magnitude smaller than the baseline.

We trained a 34-layer residual network (ResNet34) BID7 on CIFAR-10 (Krizhevsky & Hinton, 2009), using mini-batches of size 128, for 200 epochs.

We used batch normalization and standard data augmentation (horizontal flipping and cropping).

For each optimizer, we compare APO to 1) fixed learning rates; and 2) manual learning rate decay schedules.

SGD with APO.

For SGD, we used both learning rates 0.1 and 0.01 since both work well.

For SGD with momentum, we used learning rate 0.01.

We also consider a manual schedule for both SGD and SGDm: starting from learning rate 0.1, and we decay it by a factor of 5 every 60 epochs.

For the APO variants, we found that λ=1e-3 was best for SGD, while λ = 0.1 was best for SGDm.

As shown in FIG1 (c), APO not only accelerates training, but also achieves higher accuracy on the test set at the end of training.

RMSprop with APO.

For RMSprop, we use fixed learning rates 1e-3 and 1e-4, and we consider a manual learning rate schedule in which we initialize the learning rate to 1e-3 and decay by a factor of 5 every 60 epochs.

For RMSprop-APO, we used λ = 1e-3.

The training curves, test accuracies, and learning rates for RMSprop and RMSprop-APO on CIFAR-10 are shown in FIG1 (d).

We found that APO achieved substantially lower training loss than fixed learning rates, and was competitive with the manual decay schedule.

In particular, both the final training loss and final test accuracy achieved by APO are similar to those achieved by the manual schedule.

K-FAC with APO.

We also used APO to tune the learning rate and damping coefficient of K-FAC.

Similarly to the previous experiments, we use K-FAC to optimize a ResNet34 on CIFAR-10.

We used mini-batches of size 128 and trained for 100 epochs.

For the baseline, we used a fixed learning rate of 1e-3 as well as a decay schedule with initial learning rate 1e-3, decayed by a factor of 10 at epochs 40 and 80.

For APO, we used λ = 1e-2.

In experiments where the damping is not tuned, it is fixed at 1e-3.

The results are shown in FIG2 .

We see that K-FAC-APO performs competitively with the manual schedule when tuning just the global learning rate, and that both training loss and test accuracy improve when we tune both the learning rate and damping coefficient simultaneously.

Next, we evaluated APO on the CIFAR-100 dataset.

Similarly to our experiments on CIFAR-10, we used a ResNet34 network with batch-normalization and data augmentation, and we trained on minibatches of size 128, for 200 epochs.

We compared SGD-APO/SGDm-APO to standard SGD/SGDm using (1) a fixed learning rate found by grid search; (2) a custom learning rate schedule in which the learning rate is decayed by a factor of 5 at epochs 60, 120, and 180.

We set λ = 1e-3 for SGD-APO and λ = 0.1 for SGDm-APO.

Figure 4 shows the training loss, test accuracy, and the tuned learning rate.

It can be seen that APO generally achieves smaller training loss and higher test accuracy.

We also used APO to train an 18-layer residual network (ResNet18) with batch normalization on the SVHN dataset BID21 .

Here, we used the standard train and test sets, without additional training data.

We used mini-batches of size 128 and trained our networks for 160 epochs.

We compared APO to 1) fixed learning rates, and 2) a manual schedule in which we initialize the learning rate to 1e-3 and decay by a factor of 10 at epochs 80 and 120.

We show the training loss, test accuracy, and learning rates for each method in Figure 5 .

Here, RMSprop-APO achieves similar training loss to the manual schedule, and obtains higher test accuracy than the schedule.

We also see that the learning rate adapted by APO spans two orders of magnitude, similar to the manual schedule.

Figure 6: SGD with weight decay compared to SGD-APO without weight decay, on CIFAR-10.

BID9 is a widely used technique to speed up neural net training.

Networks with BN are commonly trained with weight decay.

It was shown that the effectiveness of weight decay for networks with BN is not due to the regularization, but due to the fact that weight decay affects the scale of the network weights, which changes the effective learning rate (Zhang et al., 2018; BID8 BID35 .

In particular, weight decay decreases the scale of the weights, which increases the effective learning rate; if one uses BN without regularizing the norm of the weights, then the weights can grow without bound, pushing the effective learning rate to 0.

Here, we show that using APO to tune learning rates allows for effective training of BN networks without using weight decay.

In particular, we compared SGD-APO without weight decay and SGD with weight decay 5e-4.

Figure 6 shows that SGD-APO behaved better than SGD with a fixed learning rate, and achieved comparable performance as SGD with a manual schedule.

We introduced amortized proximal optimization (APO), a method for online adaptation of optimization hyperparameters, including global and per-layer learning rates, and damping parameters for approximate second-order methods.

We evaluated our approach on real-world neural network optimization tasks-training MLP and CNN models-and showed that it converges faster and generalizes better than optimal fixed learning rates.

Empirically, we showed that our method overcomes short horizon bias and performs well with sensible default values for the meta-optimization parameters.

Guodong Zhang, Chaoqi Wang, Bowen Xu, and Roger Grosse.

Three mechanisms of weight decay regularization.

arXiv preprint arXiv:1810.12281, 2018.A PROOF OF THEOREM 1We first introduce the following lemma:Lemma 1.

Assume the manifold is smooth with C-bounded curvature, the gradient norm of loss function L is upper bounded by G. If the effective gradient at point Z k ∈ M is g k , then for any DISPLAYFORM0 Proof.

We construct the Z satisfying the above inequality.

Consider the following point in R d : DISPLAYFORM1 We show that Z is a point satisfying the inequality in the lemma.

Firstly, we notice that DISPLAYFORM2 This is because when we introduce the extra curveṽ DISPLAYFORM3 Here we use the fact thatv = 0 and v ≤ C.

Therefore we have DISPLAYFORM4 Here the first equality is by introducing the extra Y , the first inequality is by triangle inequality, the second equality is by the definition of g k being ∇ Z L(Z k ) projecting onto a plane, the second inequality is due to the above bound of Y − Z , the last inequality is due to DISPLAYFORM5 , there is therefore DISPLAYFORM6 which completes the proof.

Proof.

For the ease of notation, we denote the effective gradient at iteration k as g k .

For one iteration, there is DISPLAYFORM0 Here the first inequality is due to the Lipschitz continuity and the fact that total loss equals to the sum of all loss functions, and the second inequality is due to λ ≥ L 4 , the third inequality is due to Lemma 1 with γ =

So we have DISPLAYFORM1 Telescoping, there is DISPLAYFORM2

Proof.

For notational convenience, we think of Z as a vector rather than a matrix in this proof.

The Hessian ∇ 2 L(Z) is therefore a block diagonal matrix, where each block is the Hessian of loss on a single data.

First, we notice the following equation: DISPLAYFORM0 is the norm of vector v defined by the positive definite matrix A. DISPLAYFORM1 , therefore also positive definite.

As a result of the above equivalence, one step of Proximal Newton Method can be written as: DISPLAYFORM2 .Since Z * ∈ M by assumption, there is: DISPLAYFORM3 Now we have the following inequality for one iteration: DISPLAYFORM4 Here the first inequality is because of triangle inequality, the second inequality is due to the previous result, the equality is because ∇ L(Z * ) = 0, the last inequality is because of the strong convexity.

By the Lipschitz continuity of the Hessian, we have: DISPLAYFORM5 Therefore, we have: Here we highlight the ability of APO to tune several optimization hyperparameters simultaneously.

We used APO to adapt all of the RMSprop hyperparameters {η, ρ, }.

As shown in FIG6 (a), tuning ρ and in addition to the learning rate η can stabilize training.

We also used APO to adapt per-layer learning rates.

FIG6 (b) shows the per-layer learning rates tuned by APO, when using SGD on MNIST.

FIG6 (c) uses APO to tune per-layer learning rate of RMSprop on MNIST.

FIG7 shows the adaptation of the additional ρ and hyperparameters of RMSprop, for training an MLP on MNIST.

Tuning per-layer learning rates is a difficult optimization problem, and we found that it was useful to use a smaller meta learning rate of 0.001 and perform meta-updates more frequently.

DISPLAYFORM6

We also used APO to train a convolutional network on the FashionMNIST dataset BID38 .

The network we use consists of two convolutional layers with 16 and 32 filters respectively, both with kernel size 5, followed by a fully-connected layer.

The results are shown in FIG6 (d), where we also compare K-FAC to hand-tuned RMSprop and RMSprop-APO on the same problem.

We find that K-FAC with a fixed learning rate outperforms RMSprop-APO, while K-FAC-APO substantially outperforms K-FAC.

The results are shown in FIG6 (d).

We also show the adaptation of both the learning rate and damping coefficient for K-FAC-APO in FIG6 (d).

In this section, we present additional experiments on the Rosenbrock problem.

We show in FIG9 that APO converges quickly from different starting points on the Rosenbrock surface.

In this section we show that APO is robust to the choice of initial learning rate of the base optimizer.

With a suitable meta learning rate, APO quickly adapts many different initial learning rates to the same range, after which the learning rate adaptation follows a similar trajectory.

Thus, APO helps to alleviate the difficulty involved in selecting an initial learning rate.

First, we used RMSprop-APO to optimize Rosenbrock, starting with a wide range of initial learning rates; we see in FIG0 that the training losses and learning rates are nearly identical between all these experiments.

Next, we trained an MLP on MNIST and ResNet34 on CIFAR-10 using RMSprop-APO, with the learning rate of the base optimizer initialized to 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, and 1e-7.

We used the default meta learning rate 0.1.

As shown in FIG0 , the training loss, test accuracy, and learning rate adaptation are nearly identical using these initial learning rates, which span 5 orders of magnitude.

In this section we apply APO to the noisy quadratic problem investigated in BID37 BID23 , and demonstrate that APO overcomes the short horizon bias problem.

We optimize a quadratic function f (x) = x T Hx, where x ∈ R 1000 , H is a diagonal matrix H = diag{h 1 , h 2 , · · · , h 1000 }, with eigenvalues h i evenly distributed in interval [0.01, 1].

Initially, we set x with each dimension being 100.

For each iteration, we can access the noisy version of the function, i.e., the gradient and function value of functioñ DISPLAYFORM0 Here c is the vector of noise: each dimension of c is independently randomly sampled from a normal distribution at each iteration, and the variance of dimension i is set to be 1 hi .

For SGD, we consider the following four learning rate schedules: optimal schedule, exponential schedule, linear schedule and a fixed learning rate.

For SGD with APO, we directly use functionf as the loss approximation h, use Euclidean distance norm square as the dissimilarity term D, and consider the following schedules for λ: optimal schedule(with λ ≥ 0), exponential schedule, linear schedule and a fixed λ.

We calculate the optimal parameter for each schedule of both algorithms so as to achieve a minimal function value at the end of 300 iterations.

We optimize the schedules with 10000 steps of Adam and learning rate 0.001 after unrolling the entire 300 iterations.

The function values at the end of 300 iterations with each schedule are shown in Table 2 .

FIG0 plots the training loss and learning rate of SGD during the 300 iterations under optimal schedule, figure 13 plots the training loss and λ under optimal schedule for SGD with APO.

It can be seen that SGD with APO achieves almost the same training loss as optimal SGD for noisy quadratics task.

This indicates that APO doesn't suffer from the short-horizon bias mentioned in BID37 .

Adam BID11 is an adaptive optimization algorithm widely used for training neural networks, which can be seen as RMSProp with momentum.

The update rule is given by: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 where g t = ∇ θ t (θ t−1 ).Similar to SGD with Nesterov momentum, we fixed the β 1 and β 2 for Adam, and used APO to tune the global learning rate η.

We tested Adam-APO with a ResNet34 network on CIFAR-10 dataset, and compared it with both Adam with fixed learning rate and Adam with a learning rate schedule where the learning rate is initialized to 1e-3 and is decayed by a factor of 5 every 60 epochs.

Similarly to SGD with momentum, we found that Adam generally benefits from larger values of λ.

Thus, we recommend performing a grid search over λ values from 1e-2 to 1.

As shown in FIG0 , APO improved both the training loss and test accuracy compared to the fixed learning rate, and achieved comparable performance as the manual learning rate schedule.

Population-based training (PBT) BID10 is an approach to hyperparameter optimization that trains a population of N neural networks simultaneously: each network periodically evaluates its performance on a target measure (e.g., the training loss); poorly-performing networks can exploit better-performing members of the population by cloning the weights of the better network, copying and perturbing the hyperparameters used by the better network, and resuming training.

In this way, a single model can essentially experience multiple hyperparameter settings during training; in particular, we are interested in evaluating the learning rate schedule found using PBT.Here, we used PBT to tune the learning rate for RMSprop, to optimize a ResNet34 model on CIFAR-10.

For PBT, we used a population of size 4 (which we found to perform better than a population of size 10), and used a perturbation strategy that consists of randomly multiplying the learning rate by either 0.8 or 1.2.

In PBT, one can specify the probability with which to re-sample a hyperparameter value from an underlying distribution.

We found that it was critical to set this to 0; otherwise, the learning rate could jump from small to large values and cause instability in training.

FIG0 compares PBT with APO; we show the best training loss achieved by any of the models in the PBT population, as a function of wall-clock time.

For a fair comparison between these methods, we ran both PBT and APO using 1 GPU.

We see that APO outperforms PBT, achieving a training loss an order of magnitude smaller than PBT, and achieves the same test accuracy, much more quickly.

<|TLDR|>

@highlight

We introduce amortized proximal optimization (APO), a method to adapt a variety of optimization hyperparameters online during training, including learning rates, damping coefficients, and gradient variance exponents.
This paper proposes a new approach for step size adaptation in gradient methods.

The proposed method called step size optimization (SSO) formulates the step size adaptation as an optimization problem which minimizes the loss function with respect to the step size for the given model parameters and gradients.

Then, the step size is optimized based on alternating direction method of multipliers (ADMM).

SSO does not require the second-order information or any probabilistic models for adapting the step size, so it is efficient and easy to implement.

Furthermore, we also introduce stochastic SSO for stochastic learning environments.

In the experiments, we integrated SSO to vanilla SGD and Adam, and they outperformed state-of-the-art adaptive gradient methods including RMSProp, Adam, L4-Adam, and AdaBound on extensive benchmark datasets.

First-order gradient methods (simply gradient methods) have been widely used to fit model parameters in machine learning and data mining, such as training deep neural networks.

In the gradient methods, step size (or learning rate) is one of the most important hyperparameters that determines the overall optimization performance.

For this reason, step size adaptation has been extensively studied from various perspectives such as second-order information (Byrd et al., 2016; Schaul et al., 2013) , Bayesian approach (Mahsereci & Henning, 2015) , learning to learn paradigm (Andrychowicz et al., 2016) , and reinforcement learning (Li & Malik, 2017) .

However, they are hardly used in practice due to lack of solid empirical evidence for the step size adaptation performance, hard implementation, or huge computation.

For these reasons, some heuristically-motivated methods such as AdaGrad (Duchi et al., 2011) , RMSProp (Tieleman & Hinton, 2012) , and Adam (Kingma & Ba, 2015) are mainly used in practice to solve the large-scale optimization problems such as training deep neural networks.

Recently, two impressive methods, called L 4 (Rolinek & Martius, 2018) and AdaBdound (Luo et al., 2019) , were proposed to efficiently adapt the step size in training of models, and showed some improvement over existing methods without huge computation.

However, performance comparisons to them were conducted only on relatively simple datasets such as MNIST and CIFAR-10, even though L 4 has several newly-introduced hyperparameters, and AdaBound needs manually-desgined bound functions.

Moreover, L 4 still requires about 30% more execution time, and AdaBound lacks the time complexity analysis or empirical results on training performance against actual execution time.

This paper proposes a new optimization-based approach for the step size adaptation, called step size optimization (SSO).

In SSO, the step size adaptation is formulated as a sub-optimization problem of the gradient methods.

Specifically, the step size is adapted to minimize a linearized loss function for the current model parameter values and gradient.

The motivation of SSO and the justification for the performance improvement by SSO is clear because it directly optimizes the step size to minimize the loss function.

We also present a simple and efficient algorithm to solve this step size optimization problem based on the alternating direction method of multipliers (ADMM) (Gabay & Mercier, 1976) .

Furthermore, we provide a practical implementation of SSO on the loss function with L 2 regularization (Krogh & Hertz, 1992) and stochastic SSO for the stochastic learning environments.

SSO does not require the second-order information (Byrd et al., 2016; Schaul et al., 2013) and any probabilistic models (Mahsereci & Henning, 2015) to adapt the step size, so it is efficient and easy to implement.

We analytically and empirically show that the additional time complexity of SSO in the gradient methods is negligible in the training of the model.

To validate the practical usefulness of SSO, we made two gradient methods, SSO-SGD and SSO-Adam, by integrating SSO to vanilla SGD and Adam.

In the experiments, we compared the training performance of SSO-SGD and SSOAdam with two state-of-the-art step size adaptation methods (L 4 and AdaBdound) as well as the most commonly used gradient methods (RMSProp and Adam) on extensive benchmark datasets.

The goal of step size optimization (SSO) is to find the optimal step size that minimizes the loss function with respect to the step size η as:

where J is the loss function; Ω is a regularization term; θ is the model parameter; and v is the gradient for updating θ.

Note that v is an optimizer-dependent gradient such as the moving average of the gradients in Adam.

As gradient methods update the model by moving to the opposite direction of the gradient (θ ← θ − ηv), the loss function J (θ) and the regularization term Ω(θ) can be expressed as J (θ − ηv) and Ω(θ − ηv), respectively.

In real-world problems, however, directly solving the optimization problem in Eq. (1) is infeasible due to the severe nonlinearity of J .

To handle this difficulty, first, we linearize J around θ as:

where g = ∇ θ J is the true gradient.

Note that v is the same as g in vanilla gradient method.

However, in order for the linearization of Eq. (2) to be valid, η should be sufficiently small.

To this end, we introduce an inequality constraint for the upper bound of η.

Thus, the optimization problem of SSO is given by:

where is a positive hyperparameter that defines the upper bound of the step size.

That is, SSO adapts the step size by solving the constrained optimization problem in Eq. (3).

The augmented Lagrangian is a widely used optimization technique to handle a constrained optimization problem by transforming it into an unconstrained problem.

The objective function in the augmented Lagrangian is defined based on equality constraints.

For this reason, we need to transform the optimization problem with the inequality constraints in Eq. (3) to the problem with the equality constraints by introducing slack variables s 1 and s 2 as:

Finally, the augmented Lagrangian for the problem of Eq. (4) is given by:

where λ 1 ≥ 0 and λ 2 ≥ 0 are dual variables, and µ is a balancing parameter between the objective function and the penalty term for the equality constraints.

In general, µ is simply set to be gradually increased in optimization process to guarantee the feasibility for the equality constraints (Gabay & Mercier, 1976; Ouyan et al., 2013) .

In this section, we describe the optimization algorithm to find the optimal step size that minimizes the augmented Lagrangian in Eq. (5) using ADMM.

In mathematical optimization and machine learning, ADMM has been widely used to solve the optimization problem containing different types of primal variables x, z with the equality constraints such as:

x, z = arg min

where A and B are coefficient matrices of the equality constraints, and c is a constant.

ADMM iteratively finds the optimal variables by minimizing the augmented Lagrangian for the problem in Eq. (6), denoted by L µ (x, z, λ), because directly solving the problem can be nontrivial.

Specifically, ADMM (Algorithm 1) optimizes primal and dual variables in a one-sweep Gauss-Seidel manner:

is minimized with respect to the primal variables x and z alternatively for the fixed dual variable λ.

Then, L µ (x, z, λ) is minimized over the dual variable λ for the fixed primal variables x and z.

Algorithm 1: ADMM Output: Optimized primal variables x and

The step size optimization problem of Eq. (4) is an equality constrained problem, and also has two kinds of primal variables, the step size η and the slack variables s 1 , s 2 .

That is, for the primal variables η and s 1 , s 2 , the step size optimization problem of Eq. (4) has the same structure as the problem in Eq. (6), where

, s 2 ) = 0, and the equality constraints are η −s 1 = 0 and −η −s 2 = 0.

Thus, the primal variables of the step size optimization problem η, s 1 , s 2 , and the dual variables λ 1 , λ 2 can be optimized by ADMM as:

) .

Note that the max operation with zero is applied to Eq. (8) ∼ (11) for satisfying the nonnegative constrains of the slack and the dual variables.

Algorithm 2 shows the overall process of the gradient descent method with SSO to optimize θ.

In line 3, the true gradient g and the adaptive gradient v are computed, and the augmented Lagrangian L µ (η, s 1 , s 2 , λ 1 , λ 2 ) for optimizing the step size is determined based on the current gradients.

Then, the step size is optimized for currently given model parameters and gradients by iteratively minimizing L µ (η, s 1 , s 2 , λ 1 , λ 2 ) in line 5∼13.

Finally, θ is updated with the optimized step size and the adaptive gradient in line 14.

Algorithm 2: Gradient descent method with SSO Input : Upper bound of the step size:

In this section, we additionally introduce the upper bound decay for SSO.

Since the shape of the augmented Lagrangian for SSO can be changed by the regularization term Ω, the convergence property of the gradient methods with SSO may be different for each different regularization term.

To overcome this problem, we devise the upper bound decay method and integrate it into SSO.

One possible implementation of the upper bound decay is to use the exponential decay as follows.

where γ ∈ (0, 1) is decay factor.

That is, is exponentially decreased over the training.

It is similar to the step size decay, but there is a big difference.

The upper bound decay indirectly reduces the step size by decreasing the upper bound of the step size instead of reducing it directly.

That is, SSO with the upper bound decay automatically provides an optimal step size that will be gradually reduced over the training.

Furthermore, SSO with the upper bound decay always guarantees that the step size converges to zero regardless of the shape of the augmented Lagrangian for the valid decay factors such as γ ∈ (0, 1) in Eq (12).

Thus, the upper bound decay is more flexible than the step size decay and can be regarded as a generalized method of the step size decay.

One main advantage of SSO over the existing methods is that it can exploit such upper bound decay.

In SSO with the upper bound decay, the initial upper bound (0) is a hyperparameter.

In this section, we derive SSO with the L 2 regularization that is the most widely used regularization technique and also provide a practical implementation of the gradient method with SSO.

With the L 2 regularization term, the augmented Lagrangian of the step size optimization problem is given by:

where β is a positive hyperparameter of the L 2 regularization for balancing the loss function and the regularization term.

By applying ADMM in Algorithm 1, we can optimize the step size using the following update rules:

Note that the update rule for the dual variables λ 1 and λ 2 are the same as Eq. (10) and (11) because the update rules of the dual variables are independent of the loss function and the regularization term in SSO.

More precisely, the update rules of the dual variables depend only on the equality constraints for the step size.

If the optimal step size exists within an range [0, ), the slack variables converge as s 1 → η and s 2 → − η when ADMM in SSO is sufficiently iterated (µ → ∞).

Thus, the step size η converges to some value as:

In contrast, if the optimal step size exists over the upper bound, the step size may converge near the upper bound (η ≈ ) in ADMM.

Thus, the slack variables converge as s 1 → and s 2 → 0, and the step size consequently converge to the upper bound as follows.

Thus, the step size always converges in ADMM of SSO.

Unfortunately, the second case (η → ) is not the desired result in which the model is sufficiently trained because a relatively small step size is required in this situation to make the gradient methods converge.

However, it is not a problem in SSO with the upper bound decay because the upper bound must be reduced by the decay method over the training.

In this section, we describe SSO for the stochastic learning environments and also provide stochastic SSO with L 2 regularization.

The optimization problem in the stochastic environments can be formulated on a mini-batch with respect to the step size as follows.

where J i is the loss function for the i th sample in the mini-batch.

This problem can be rewritten as an optimization problem with a conservative penalty term as (Ouyan et al., 2013; Li et al., 2014) :

Note that the conservative penalty term is introduced to prevent undesired large change in the model parameters as mini-batch changes.

The conservative penalty term can be rewritten with respect to the step size as:

Thus, we can derive the optimization algorithm for η (k+1) in stochastic environments by applying the linearization and ADMM to the problem in Eq. (20) as described in Section 2.1 ∼ 2.3.

In addition, we provide the update rule of stochastic SSO with L 2 regularization for the practical implementation of it.

Similar to the update rule of the deterministic SSO with L 2 regularization in Section 2.5.1, the update rule for η of stochastic SSO with L 2 regularization is given by:

Note that the update rule of the slack variables are the same as Eq. (15) and (16) because they depend only on the constraints of the problem.

Due to the sub-optimization process for the step size adaptation of SSO, a gradient method with SSO inevitably requires additional complexity.

In this section, we analyze the time complexity of the gradient method with SSO and show that the additional time complexity of the gradient method with SSO is almost the same as the vanilla gradient method in large-scale optimization problems such as training deep neural networks.

The empirical time complexity analysis over the actual execution time will be conducted in the experiment section.

In the experiments, we validated the effectiveness of SSO in gradient-based optimization.

To this end, we generated two gradient methods with stochastic SSO using the upper bound decay -(1) vanilla SGD with SSO (SSO-SGD) and (2) Adam with SSO (SSO-Adam), and then their optimization performance was compared with the state-of-the-art gradient methods: 1) RMSProp; 2) Adam; 3) L 4 -Adam; 4) AdaBound.

We specified the optimization problem as training deep neural networks because it is the most appealing and challenging problem using gradient methods.

To train deep neural networks, cross-entropy loss with L 2 regularization is used as a loss function.

We conducted the experiments on four well-known benchmark datasets: MNIST, SVHN, CIFAR-10, and CIFAR-100 datasets.

For MNIST dataset, convolutional neural network (CNN) was used.

For CIFAR and SVHN datasets, ResNet (He et al., 2016) was used.

For all datasets, we measured the training loss and the highest test accuracy during the training.

We reported the mean and the standard deviation of the highest accuracies by repeating the training several times.

Specifically, we repeated the training 10 times for MNIST dataset and 5 times for the other datasets.

Table 1 summarizes the experiment results, and detailed explanations of the results for each dataset will be provided in the following sections.

For each gradient method, we selected the best initial learning rate using a grid search in a set of {0.1, 0.05, 0.01, 0.005, 0.001} for all datasets.

For the other hyperparameters of Adam, such as the exponential decay rate of Adam, we followed the settings of the original paper of Adam (Kingma & Ba, 2015) .

To set the additional hyperpameters of L 4 -Adam and the bound functions of AdaBound, we used the recommended setting in their original papers (Rolinek & Martius, 2018; Luo et al., 2019) .

For MNIST dataset, the initial upper bound of SSO, (0) , was set to 0.5.

For the other datasets, (0) was fixed to 1.

The decay factor in the upper bound decay, γ, was fixed to 0.95 for all datasets.

The number of iterations for optimizing the step size in SSO was fixed to 20 (I = 20).

We selected the best regularization coefficient (β) using a grid search for each gradient method on for all datasets.

The batch size was fixed to 128 for all datasets.

For MNIST dataset, we exploited a commonly used architecture of CNN with two convolution layer and one fully-connected output layer.

For SVHN and CIFAR datasets, we used the ResNet with three residual blocks and one fully-connected output layer .

All experiments were conducted on NVIDIA GeForce RTX 2080 Ti.

We used PyTorch to implement SSO and the author's source code for L 4 -Adam 1 and AdaBound 2 .

The source code of SSO and the experiment scripts are available at GitHubURL (open after the review).

MNIST dataset is a widely used benchmark dataset for digit recognition.

It contains 60,000 training instances and 10,000 test instances of 28×28×1 size from 10 classes.

As shown in Fig. 1-(a) , L 4 -Adam, AdaBound, SSO-SGD, and SSO-Adam rapidly reduced the training loss to zero on MNIST dataset.

Although SSO-Adam showed the highest test accuracy, other gradient methods also showed similar accuracies because MNIST dataset is too simple and easy to fit the model (easy to achieve 99% test accuracy).

For this reason, to accurately evaluate the effectiveness of each method, we compared the performance on SVHN dataset, which is also a widely used benchmark dataset for digit recognition and more realistic.

CIFAR datasets contain 50,000 training instances and 10,000 test instances of 32×32×3 size.

CIFAR-10 and CIFAR-100 contain 10 and 100 categories (classes), respectively.

We also used ResNet with three residual blocks and one fully-connected layer for CIFAR datasets.

On CIFAR-10 dataset, L 4 , SSO-SGD, and SSO-Adam also rapidly reduced the training loss to zero (Fig. 3-a) .

Furthermore, SSO-SGD and SSO-Adam outperformed all competitors in the test accuracy ( Fig. 3-a) .

Especially, SSO-SGD achieved about 3% improvement on the test accuracy compared to AdaBound that showed the highest test accuracy among the competitors.

As shown in Fig. 4-(b) , both SSO-SGD and SSO-Adam outperformed all state-of-the-tart competitors in the test accuracy again.

In particular, SSO-SGD achieved 5% improved test accuracy compared to L 4 -Adam that showed the highest test accuracy among the competitors.

In this experi- ment, SSO-SGD and SSO-Adam showed better generalization performance than L 4 -Adam because stochastic SSO is designed to the stochastic learning environments.

In this experiment, we measured the test accuracy over actual execution time to evaluate the usefulness of each method.

Fig. 5 shows the results of the experiment on SVHN and CIFAR100 datasets.

The experiment results are similar to the results in Fig. 2-(b) and 4.

It shows that SSO is as efficient as existing step size adaptation methods.

Furthermore, quantitatively, SSO-SGD and SSO-Adam required about 5,500 seconds execution time for 100 epochs like RMSProp and Adam, but L 4 -Adam required about 9,000 seconds execution time that is 30% higher than the execution time of SSO-SGD and SSO-Adam.

On CIFAR-100 dataset, SSO-SGD and SSO-Adam also required about 8,500 seconds execution time for 200 epochs like RMSProp and Adam, but L 4 -Adam spent about 13,000 seconds.

We checked the training performance on MNIST dataset by change the initial upper bound (0) to measure the sensitivity of SSO for the hyperparameter.

Fig. 6 shows the experiment results.

A APPENDIX: OPTIMIZED STEP SIZE ON MNIST DATASET.

In this experiment, we measured the optimized step size of SSO rate for each epoch.

We used SSOAdam with the upper bound decay.

Since the step size adaptation is executed by the number of mini-batches for each epoch, we presented maximum, mean, and minimum of the optimized step size for each epoch.

Fig. 7 shows the optimized step sizes for each epoch.

As shown in the result, the learning rate is strictly optimized within [0, ] and gradually reduced over the epochs.

Note that the current upper bounds are not shown because the maximums of the optimized step sizes overlap them in Fig. 7 -(a).

(a) Optimized step size over the epochs (b) Optimized step size over the epochs without presenting the maximum Figure 7 : Maximum, mean, and minimum of the optimized step sizes for each epoch on MNIST dataset.

@highlight

We propose an efficient and effective step size adaptation method for the gradient methods.

@highlight

A new step size adaptation in first-order gradient methods that establishes a new optimization problem with the first-order expansion of the loss function and regularization, where step size is treated as variable.
In this paper, we consider the problem of training neural networks (NN).

To promote a NN with specific structures, we explicitly take into consideration the nonsmooth regularization (such as L1-norm) and constraints (such as interval constraint).

This is formulated as a constrained nonsmooth nonconvex optimization problem, and we propose a convergent proximal-type stochastic gradient descent (Prox-SGD) algorithm.

We show that under properly selected learning rates, momentum eventually resembles the unknown real gradient and thus is crucial in analyzing the convergence.

We establish that with probability 1, every limit point of the sequence generated by the proposed Prox-SGD is a stationary point.

Then the Prox-SGD is tailored to train a sparse neural network and a binary neural network, and the theoretical analysis is also supported by extensive numerical tests.

In this paper, we consider the problem of training neural networks (NN) under constraints and regularization.

It is formulated as an optimization problem

where x is the parameter vector to optimize, y i is the i-th training example which consists of the training input and desired output, and m is the number of training examples.

The training loss f is assumed to be smooth (but nonconvex) with respect to x, the regularization r is assumed to be convex (but nonsmooth), proper and lower semicontinuous, and the constraint set X is convex and compact (closed and bounded).

When r(x) = 0 and X = R n , stochastic gradient descent (SGD) has been used to solve the optimization problem (1).

At each iteration, a minibatch of the m training examples are drawn randomly, and the obtained gradient is an unbiased estimate of the true gradient.

Therefore SGD generally moves along the descent direction, see Bertsekas & Tsitsiklis (2000) .

SGD can be accelerated by replacing the instantaneous gradient estimates by a momentum aggregating all gradient in past iterations.

Despite the success and popularity of SGD with momentum, its convergence had been an open problem.

Assuming f is convex, analyzing the convergence was first attempted in Kingma & Ba (2015) and later concluded in Reddi et al. (2018) .

The proof for a nonconvex f was later given in Chen et al. (2019) ; Lei et al. (2019) .

In machine learning, the regularization function r is typically used to promote a certain structure in the optimal solution, for example sparsity as in, e.g., feature selection and compressed sensing, or a zero-mean-Gaussian prior on the parameters (Bach et al., 2011; Boyd et al., 2010) .

It can be interpreted as a penalty function since at the optimal point x of problem (1), the value r(x ) will be small.

One nominant example is the Tikhonov regularization r(x) = µ x 2 2 for some predefined constant µ, and it can be used to alleviate the ill-conditioning and ensure that the magnitude of the weights will not become exceedingly large.

Another commonly used regularization, the 1 -norm where r(x) = µ x 1 = µ n j=1 |x j | (the convex surrogate of the 0 -norm), would encourage a sparse solution.

In the context of NN, it is used to (i) promote a sparse neural network (SNN) to alleviate overfitting and to allow a better generalization, (ii) accelerate the training process, and (iii) prune the network to reduce its complexity, see Louizos et al. (2018) and Gale et al. (2019) .

Technically, it is difficult to analyze the regularizations as some commonly used convex regularizers are nonsmooth, for example, 1 -norm.

In current implementations of Tensorflow, the gradient of |x| is simply set to 0 when x = 0.

This amounts to the stochastic subgradient descent method and usually exhibits slow convergence.

Other techniques to promote a SNN includes magnitude pruning and variational dropout, see Gale et al. (2019) .

Although regularization can be interpreted as a constraint from the duality theory, sometimes it may still be more desirable to use explicit constraints, for example, x 2 j ≤ α, where the summation is over the weights on the same layer.

This is useful when we already know how to choose α.

Another example is the lower and upper bound on the weights, that is, l ≤ w ≤ u for some predefined l and u. Compared with regularization, constraints do not encourage the weights to stay in a small neighborhood of the initial weight, see Chapter 7.2 of Goodfellow et al. (2016) for more details.

The set X models such explicit constraints, but it poses an additional challenge for stochastic gradient algorithms as the new weight obtained from the SGD method (with or without momentum) must be projected back to the set X to maintain its feasibility.

However, projection is a nonlinear operator, so the unbiasedness of the random gradient would be lost.

Therefore the convergence analysis for constrained problems is much more involved than unconstrained problems.

In this paper, we propose a convergent proximal-type stochastic gradient algorithm (Prox-SGD) to train neural networks under nonsmooth regularization and constraints.

It turns out momentum plays a central role in the convergence analysis.

We establish that with probability (w.p.) 1, every limit point of the sequence generated by Prox-SGD is a stationary point of the nonsmooth nonconvex problem (1).

This is in sharp contrast to unconstrained optimization, where the convergence of the vanilla SGD method has long been well understood while the convergence of the SGD method with momentum was only settled recently.

Nevertheless, the convergence rate of Prox-SGD is not derived in the current work and is worth further investigating.

To test the proposed algorithm, we consider two applications.

The first application is to train a SNN, and we leverage 1 -regularization, that is,

The second application is to train a binary neural network (BNN) where the weights (and activations) are either 1 or -1 (see Courbariaux et al. (2015; ; Hou et al. (2017) ; Yin et al. (2018) ; Bai et al. (2019) for more details).

To achieve this, we augment the loss function with a term that penalizes the weights if they are not +1 or -1:

where µ is a given penalty parameter.

The binary variable a j can be interpreted as a switch for weight x j : when a j = 0, (1 − a j )(x j − 1) 2 is activated, and there is a strong incentive for x j to be 1 (the analysis for a j = 1 is similar).

Since integer variables are difficult to optimize, we relax a j to be a continuous variable between 0 and 1.

To summarize, a BNN can be obtained by solving the following regularized optimization problem under constraints with respect to x and a

If µ is properly selected (or sufficiently large), the optimal a j will be exactly or close to 0 or 1.

Consequently, regularization and constraints offer interpretability and flexibility, which allows us to use more accurate models to promote structures in the neural networks, and the proposed convergent Prox-SGD algorithm ensures efficient training of such models.

In this section, we describe the Prox-SGD algorithm to solve (1).

Background and setup.

We make the following blanket assumptions on problem (1).

• f i (x, y (i) ) is smooth (continuously differentiable) but not necessarily convex.

•

• r(x) is convex, proper and lower semicontinuous (not necessarily smooth).

• X is convex and compact.

We are interested in algorithms that can find a stationary point of (1).

A stationary point x satisfies the optimality condition: at x = x ,

When r(x) = 0 and X = R n , the deterministic optimization problem 1 can be solved by the (batch) gradient descent method.

When m, the number of training examples, is large, it is computationally expensive to calculate the gradient.

Instead, we estimate the gradient by a minibatch of m(t) training examples.

We denote the minibatch by M(t): its elements are drawn uniformly from {1, 2, . . .

, m} and there are m(t) elements.

Then the estimated gradient is

and it is an unbiased estimate of the true gradient.

The proposed algorithm.

The instantaneous gradient g(t) is used to form an aggregate gradient (momentum) v(t), which is updated recursively as follows

where ρ(t) is the stepsize (learning rate) for the momentum and ρ(t) ∈ (0, 1].

At iteration t, we propose to solve an approximation subproblem and denote its solution as x(x(t), v(t), τ (t)), or simply x(t)

A quadratic regularization term is incorporated so that the subproblem (7) is strongly convex and its modulus is the minimum element of the vector τ (t), denoted as τ (t) and τ (t) = min j=1,...,n τ j (t).

Note that τ (t) should be lower bounded by a positive constant that is strictly larger than 0, so that the quadratic regularization in (7) will not vanish.

The difference between two vectors x(t) and x(t) specifies a direction starting at x(t) and ending at x(t).

This update direction is used to refine the weight vector

where (t) is a stepsize (learning rate) for the weight and (t) ∈ (0, 1].

Note that x(t + 1) is feasible as long as x(t) is feasible, as it is the convex combination of two feasible points x(t) and x(t) while the set X is convex.

The above steps (5)-(8) are summarized in Algorithm 1, which is termed proximal-type Stochastic Gradient Descent (Prox-SGD), for the reason that the explicit constraint x ∈ X in (7) can also be formulated implicitly as a regularization function, more specifically, the indicator function δ X (x).

If all elements of τ (t) are equal, then x(t) is exactly the proximal operator

for t = 0 : 1 : T do 1.

Compute the instantaneous gradient g(t) based on the minibatch M(t):

2.

Update the momentum:

3.

Compute x(t) by solving the approximation subproblem:

4.

Update the weight:

end for algorithm momentum weight quadratic gain in subproblem regularization constraint set Table 1 : Connection between the proposed framework and existing methods, where ρ, β, and δ are some predefined constants.

Prox-SGD in Algorithm 1 bears a similar structure as several SGD algorithms, without and with momentum, see Table 1 , and it allows to interpret some existing algorithms as special cases of the proposed framework.

For example, no momentum is used in SGD, and this amounts to setting ρ(t) = 1 in Algorithm 1.

In ADAM, the learning rate for momentum is a constant ρ and the learning rate for the weight vector is given by /(1 − ρ t ) for some , and this simply amounts to setting ρ(t) = ρ and (t) = /(1 − ρ t ) in Algorithm 1.

This interpretation also implies that the convergence conditions to be proposed shortly later are also suffcient for existing algorithms (although they are not meant to be the weakest conditions available in literature).

Solving the approximation subproblem (7).

Since (7) is strongly convex, x(t) is unique.

Generally x(t) in (7) does not admit a closed-form expression and should be solved by a generic solver.

However, some important special cases that are frequently used in practice can be solved efficiently.

• The trivial case is X = R n and r = 0, where

where the vector division is understood to be element-wise.

When X = R n and r(x) = µ x 1 , x(t) has a closed-form expression that is known as the soft-thresholding operator

where Bach et al., 2011) .

• If X = R n and r(x) = µ x 2 and τ (t) = τ I for some τ , then (Parikh & Boyd, 2014)

If x is divided into blocks x 1 , x 2 , . . .

, the 2 -regularization is commonly used to promote block sparsity (rather than element sparsity by 1 -regularization).

• When there is a bound constraint l ≤ x ≤ u, x(t) can simply be obtained by first solving the approximation subproblem (7) without the bound constraint and then projecting the optimal point onto the interval [l, u] .

For example, when X = R n and r = 0,

with

• If the constraint function is quadratic: X = {x : x 2 2 ≤ 1}, x(t) has a semi-analytical expression (up to a scalar Lagrange multiplier which can be found efficiently by the bisection method).

Approximation subproblem.

We explain why we update the weights by solving an approximation subproblem (7).

First, we denote f as the smooth part of the objective function in (7).

Clearly it depends on x(t) and v(t) (and thus M(t)), while x(t) and v(t) depend on the old weights x(0), . . .

, x(t − 1) and momentum and v(0), . . . , v(t − 1).

Define F(t) {x (0), . . .

, x(t), M(0), . . . , M(t)} as a shorthand notation for the trajectory generated by Prox-SGD.

We formally write f as

It follows from the optimality of x(t) that

After inserting (13) and reorganizing the terms, the above inequality becomes

Since ∇f (x) is Lipschitz continuous with constant L, we have

where the first inequality follows from the descent lemma (applied to f ) and the second inequality follows from the Jensen's inequality of the convex function r and the update rule (8).

If v(t) = ∇f (x(t)) (which is true asymptotically as we show shortly later), by replacing ∇f (x(t)) in (16) by v(t) and inserting (14) into (16), we obtain

The right hand side (RHS) will be negative when (t) <

L : this will eventually be satisfied as we shall use a decaying (t).

This implies that the proposed update (8) will decrease the objective value of (1) after each iteration.

Momentum and algorithm convergence.

It turns out that the momentum (gradient averaging step) in (6) is essential for the convergence of Prox-SGD.

Under some mild technical assumptions we outline now, the aggregate gradient v(t) will converge to the true (unknown) gradient ∇f (x(t)).

This remark is made rigorous in the following theorem.

Theorem 1.

Assume that the unbiased gradient g(t) has a bounded second moment

for some finite and positive constant C, and the sequence of stepsizes {ρ(t)} and { (t)} satisfy

Then lim t→∞ v(t) − ∇f (x(t)) = 0, and every limit point of the sequence {x(t)} is a stationary point of (1) w.p.1.

Proof.

Under the assumptions (18) and (19), it follows from Lemma 1 of Ruszczyński (1980) that v(t) → ∇f (x(t)).

Since the descent direction x(t) − x(t) is a descent direction in view of (14), the convergence of the Prox-SGD algorithm can be obtained by generalizing the line of analysis in Theorem 1 of Yang et al. (2016) for smooth optimization problems.

The detailed proof is included in the appendix to make the paper self-contained.

We draw some comments on the convergence analysis in Theorem 1.

The bounded second moment assumption on the gradient g in (18) and decreasing stepsizes in (19) are standard assumptions in stochastic optimization and SGD.

What is noteworthy is that (t) should decrease faster than ρ(t) to ensure that v(t) → ∇f (x(t)).

But this is more of an interest from the theoretical perspective, and in practice, we observe that (t)/ρ(t) = a for some constant a that is smaller than 1 usually yields satisfactory performance, as we show numerically in the next section.

According to Theorem 1, the momentum v(t) converges to the (unknown) true gradient ∇f (x(t)), so the Prox-SGD algorithm eventually behaves similar to the (deterministic) gradient descent algorithm.

This property is essential to guarantee the convergence of the Prox-SGD algorithm.

To guarantee the theoretical convergence, the quadratic gain τ (t) in the approximation subproblem (7) should be lower bounded by some positive constant (and it does not even have to be timevarying).

In practice, there are various rationales to define it (see Table 1 ), and they lead to different empirical convergence speed and generalization performance.

The technical assumptions in Theorem 1 may not always be fully satisfied by the neural networks deployed in practice, due to, e.g., the nonsmooth ReLU activation function, batch normalization and dropout.

Nevertheless, Theorem 1 still provides valuable guidance on the algorithm's practical performance and the choice of the hyperparameters.

In this section, we perform numerical experiments to test the proposed Prox-SGD algorithm.

In particular, we first train two SNN to compare Prox-SGD with ADAM (Kingma & Ba, 2015) , AMSGrad (Reddi et al., 2018), ADABound (Luo et al., 2019) and SGD with momentum.

Then we train a BNN to illustrate the merit of regularization and constraints.

To ensure a fair comparison, the hyperparameters of all algorithms are chosen according to either the inventors' recommendations or a hyperparameter search.

Furthermore, in all simulations, the quadratic gain τ (t) in Prox-SGD is updated in the same way as ADAM, with β = 0.999 (see Table 1 ).

We first consider the multiclass classification problem on CIFAR-10 dataset (Krizhevsky, 2009) with convolution neural network (CNN).

The network has 6 convolutional layers and each of them is followed by a batch normalization layer; the exact setting is shown in Table 2 .

Following the parameter configurations of ADAM in Kingma & Ba (2015) , AMSGrad in Reddi et al. (2018) , and ADABound in Luo et al. (2019) , we set ρ = 0.1, β = 0.999 and = 0.001 (see Table  1 ), which are uniform for all the algorithms and commonly used in practice.

Note that we have also incorporated 1 -regularization in these algorithms by the built-in function in Tensorflow/Pytorch, which amounts to adding the subgradient of the 1 -norm to the gradient of the loss function.

For the proposed Prox-SGD, (t) and ρ(t) decrease over the iterations as follows,

Recall that the 1 -norm in the approximation subproblem naturally leads to the soft-thresholding proximal mapping, see (10).

The regularization parameter µ in the soft-thresholding then permits controlling the sparsity of the parameter variable x; in this experiment we set µ = 5 · 10 −5 .

In Figure 1 , we compare the four algorithms (Prox-SGD, ADAM, AMSGrad, ADABound) in terms of three metrics, namely, the training loss, the test accuracy and the achieved sparsity.

On the one hand, Figure 1(a) shows that Prox-SGD outperforms ADAM, AMSGrad and ADABound in the achieved loss value.

On the other hand, the accuracy achieved by these algorithms is comparable, see Figure 1

The sparsity of the trained model is measured by the cumulative distribution function (CDF) of the weights' value, which specifies the percentage of weights before any given value.

For the proposed Prox-SGD in Figure 1 (c) , we can observe around 0 in the x-axis the abrupt change of the CDF in the y-axis, which implies that more than 90% of the weights are exactly zero.

By comparison, only 40%-50% are exactly zero by the other algorithms.

What is more, for this experiment, the soft-thresholding proximal operator in Prox-SGD does not increase the computation time: ADAM 17.24s (per epoch), AMSGrad 17.44s, ADABound 16.38s, Prox-SGD 16.04s.

Therefore, in this experiment, the proposed Prox-SGD with soft-thresholding proximal mapping has a clear and substantial advantage than other stochastic subgradient-based algorithms.

In this subsection, the performance of Prox-SGD is evaluated by a larger and more complex network and dataset.

In particular, we train the DenseNet-201 network (Huang et al., 2017) for CIFAR-100 (Krizhevsky, 2009) .

DenseNet-201 is the deepest topology of the DenseNet family and belongs to the state of the art networks in image classification tasks.

We train the network using Prox-SGD, ADAM and SGD with momentum.

To ensure a fair comparison among these algorithms, the learning rate is not explicitly decayed during training for all algorithms.

The ideal hyperparameters for each algorithm were computed using a grid-search and the curves are averaged over five runs for each algorithm.

All algorithms use a batch-size of 128.

For Prox-SGD, the regularization parameter is µ = 10 −5 , the learning rate for the weight factor and momentum is, respectively, (t) = 0.15 (t + 4) 0.5 , ρ(t) = 0.9 (t + 4) 0.5 .

For ADAM, = 6 · 10 −4 and ρ = 0.1.

SGD with momentum uses a learning rate of = 6 · 10 −3

and a momentum of 0.9 (so ρ = 0.1).

The regularization parameter for both ADAM and SGD with momentum is µ = 10 −4 .

In Figure 3 , we demonstrate that Prox-SGD is much more efficient in generating a SNN, irrespective of the hyperparameters (related to the learning rate).

In particular, we try many different initial learning rate of the weight vector (0) for Prox-SGD and test their performance.

From Figure  3 (a)-(b) we easily see that, as expected, the hyperparameters affect the achieved training loss and test accuracy, and many lead to a worse training loss and/or test accuracy than ADAM and SGD with momentum.

However, Figure 3 (c) shows that most of them (except when they are too small: 0.01 and 0.001) generate a much sparser NN than both ADAM and SGD with momentum.

These observations are also consistent with the theoretical framework in Section 2: interpretating ADAM and SGD with momentum as special cases of Prox-SGD implies that they have the same convergence rate, and the sparsity is due to the explicit use of the nonsmooth 1 -norm regularization.

For this experiment, the soft-thresholding proximal operator in Prox-SGD increases the training time: the average time per epoch for Prox-SGD is 3.5 min, SGD with momentum 2.8 min and ADAM 2.9 min.

In view of the higher level of sparsity achieved by Prox-SGD, this increase in computation time is reasonable and affordable.

In this subsection, we evaluate the proposed algorithm Prox-SGD in training the BNN by solving problem (3).

We train a 6-layer fully-connected deep neural network (DNN) for the MNIST dataset, and we use the tanh activation function to promote a binary activation output; see Table 3 .

The algorithm parameters are the same as Sec. 3.1, except that µ = 2 · 10 −4 .

The chosen setup is particularly suited to evaluate the merit of the proposed method, since MNIST is a simple dataset and it allows us to investigate soly the effect of the proposed model and training algorithm.

After customizing the general description in Algorithm 1 to problem (3), the approximation subproblem is

Both x(t) and a(t) have a closed-form expression (cf. (9) and (12))

where v x (t) and v a (t) are the momentum updated in the spirit of (6), with the gradients given by

, and g a (t) = µx(t).

The training loss is shown in Figure 4 (a).

We remark that during the training process of Prox-SGD, the weights are not binarized, for the reason that the penalty should regularize the problem in a way such that the optimal weights (to which Prox-SGD converges) are exactly or close to 1 or -1.

After training is completed, the CDF of the learned weights is summarized in Figure 4(c) , and then the learned weights are binarized to generate a full BNN whose test accuracy is in Figure 4 (b) .

On the one hand, we see from Figure 4 (a)-(b) that the achieved training loss and test accuracy by BNN is worse than the standard full-precision DNN (possibly with soft-thresholding).

This is expected as BNN imposes regularization and constraints on the optimization problem and reduces the search space.

However, the difference in test accuracy is quite small.

On the other hand, we see from Figure 4 (c) that the regularization in the proposed formulation (3) is very effective in promoting binary weights: 15% of weights are in the range (-1,-0.5) and 15% of weights are in the range (0.5,1), and all the other weights are either -1 or 1.

As all weights are exactly or close to 1 or -1, we could just binarize the weights to exactly 1 or -1 only once by hard thresholding, after the training is completed, and thus the incurred performance loss is small (98% versus 95% for test accuracy).

In contrast, the weights generated by the full-precision DNN (that is, without regularization) are smoothly distributed in [−2, 2].

Even though the proposed formulation (3) doubles the number of parameters to optimize (from x in full-precision DNN to (x, a) in BNN Prox-SGD), the convergence speed is equally fast in terms of the number of iterations.

The computation time is also roughly the same: full-precision DNN 13.06s (per epoch) and Prox-SGD 12.21s.

We remark that g a (t), the batch gradient w.r.t.

a, has a closedform expression and it does not involve the back-propagation.

In comparison with the algorithm in Courbariaux et al. (2016) , the proposed Prox-SGD converges much faster and achieves a much better training loss and test accuracy (95% versus 89%, the computation time per epoch for Courbariaux et al. (2016) is 13.56s).

The notable performance improvement is due to the regularization and constraints.

Naturally we should make an effort of searching for a proper regularization parameter µ, but this effort is very well paid off.

Furthermore, we observe in the simulations that the performance is not sensitive to the exact value of µ, as long as it is in an appropriate range.

In this paper, we proposed Prox-SGD, a proximal-type stochastic gradient descent algorithm with momentum, for constrained optimization problems where the smooth loss function is augmented by a nonsmooth and convex regularization.

We considered two applications, namely the stochastic training of SNN and BNN, to show that regularization and constraints can effectively promote structures in the learned network.

More generally, incorporating regularization and constraints allows us to use a more accurate and interpretable model for the problem at hand and the proposed convergent Prox-SGD algorithms ensures efficient training.

Numerical tests showed that Prox-SGD outperforms state-of-the-art algorithms, in terms of convergence speed, achieved training loss and/or the desired structure in the learned neural networks.

A APPENDIX: PROOF OF THEOREM 1

Proof.

The claim lim t→∞ v(t)−∇f (x(t)) = 0 is a consequence of (Ruszczyński, 1980, Lemma 1) .

To see this, we just need to verify that all the technical conditions therein are satisfied by the problem at hand.

Specifically, Condition (a) of (Ruszczyński, 1980 , Lemma 1) is satisfied because X is closed and bounded.

Condition (b) of (Ruszczyński, 1980 , Lemma 1) is exactly (18).

Conditions (c)-(d) of (Ruszczyński, 1980 , Lemma 1) come from the stepsize rules in (19) of Theorem 1.

Condition (e) of (Ruszczyński, 1980 , Lemma 1) comes from the Lipschitz property of ∇f and stepsize rule in (19) of Theorem 1.

We need the following intermediate result to prove the limit point of the sequence x(t) is a stationary point of (1).

Lemma 1.

There exists a constant L such that

and lim t1,t2→∞ e(t 1 , t 2 ) = 0 w.p.1.

Proof.

We assume without loss of generality (w.l.o.g.) that τ (t) = τ 1, and the approximation subproblem (7) reduces to

It is further equivalent to min x∈X,r(x)≤y

where the (unique) optimal x and y is ( x(t) and r( x(t)), respectively.

We assume w.l.o.g.

that t 2 > t 1 .

It follows from first-order optimality condition that

Setting (x, y) = ( x(t 2 ), r( x(t 2 ))) in (23a) and (x, y) = ( x(t 1 ), r( x(t 1 ))) in (23b), and adding them up, we obtain

The term on the left hand side can be lower bounded as follows:

where the inequality comes from the Lipschitz continuity of ∇f (x), with ε(t) v(t) − ∇f (x(t)) .

Combining the inequalities (24) and (25), we have

which leads to the desired (asymptotic) Lipschitz property:

with L τ −1 (L + τ ) and e(t 1 , t 2 ) τ −1 (ε(t 1 ) + ε(t 2 )), and lim t1→∞,t2→∞ e(t 1 , t 2 ) = 0 w.p.1.

Define U (x) f (x) + r(x).

Following the line of analysis from (15) to (16), we obtain

where in the last inequality we used (14) and the Cauchy-Schwarz inequality.

Let us show by contradiction that lim inf t→∞ x(t)−x(t) = 0 w.p.1.

Suppose lim inf t→∞ x(t)− x(t) ≥ χ > 0 with a positive probability.

Then we can find a realization such that at the same time x(t) − x(t) ≥ χ > 0 for all t and lim t→∞ ∇f (x(t)) − v(t) = 0; we focus next on such a realization.

Using x(t) − x(t) ≥ χ > 0, the inequality (27) is equivalent to

Since lim t→∞ ∇f (x(t)) − v(t) = 0, there exists a t 0 sufficiently large such that

Therefore, it follows from (28) and (29) that

which, in view of ∞ n=t0 n+1 = ∞, contradicts the boundedness of {U (x(t))}. Therefore it must be lim inf t→∞ x(t) − x(t) = 0 w.p.1.

Let us show by contradiction that lim sup t→∞ x(t) − x(t) = 0 w.p.1.

Suppose lim sup t→∞ x(t) − x(t) > 0 with a positive probability.

We focus next on a realization along with lim sup t→∞ x(t) − x(t) > 0, lim t→∞ ∇f (x(t)) − v(t) = 0, lim inf t→∞ x(t) − x(t) = 0, and lim ti,t2→∞ e(t 1 , t 2 ) = 0, where e(t 1 , t 2 ) is defined in Lemma 1.

It follows from lim sup t→∞ x(t) − x(t) > 0 and lim inf t→∞ x(t) − x(t) = 0 that there exists a δ > 0 such that x(t) ≥ 2δ (with x(t) x(t) − x(t)) for infinitely many t and also x(t) < δ for infinitely many t.

Therefore, one can always find an infinite set of indexes, say T , having the following properties: for any t ∈ T , there exists an integer i t > t such that x(t) < δ, x(i t ) > 2δ, δ ≤ x(n) ≤ 2δ, t < n < i t .

Given the above bounds, the following holds: for all t ∈ T ,

≤ (1 + L) x(i t ) − x(t) + e(i t , t)

it−1 n=t (n) x(n) + e(i t , t)

≤ 2δ(1 + L)

it−1 n=t (n) + e(i t , t),

implying that lim inf

Proceeding as in (32), we also have: for all t ∈ T , x(t + 1) − x(t) ≤ x(t + 1) − x(t) ≤ (1 + L) (t) x(t) + e(t, t + 1), which leads to

(1 + (1 + L) (t)) x(t) + e(t, t + 1) ≥ x(t + 1) ≥ δ,

where the second inequality follows from (31).

It follows from (34) that there exists aδ 2 > 0 such that for sufficiently large t ∈ T , x(t) ≥ δ − e(t, t + 1) 1 + (1 + L) (t) ≥δ 2 > 0.

Here after we assume w.l.o.g.

that (35) holds for all t ∈ T (in fact one can always restrict {x(t)} t∈T to a proper subsequence).

We show now that (33) is in contradiction with the convergence of {U (x(t))}. Invoking (27), we have for all t ∈ T , U (x(t + 1)) − U (x(t)) ≤ − (t) τ − L 2 (t) x(t) − x(t) 2 + (t)δ ∇f (x(t)) − v(t)

and for t < n <

i t ,

<|TLDR|>

@highlight

We propose a convergent proximal-type stochastic gradient descent algorithm for constrained nonsmooth nonconvex optimization problems

@highlight

This paper proposes Prox-SGD, a theoretical framework for stochastic optimization algorithms shown to converge asymptotically to stationarity for smooth non-convvex loss + convex constraint/regularizer.

@highlight

The paper proposes a new gradient-based stochastic optimization algorithm with gradient averaging by adapting theory for proximal algorithms to the non-convex setting.
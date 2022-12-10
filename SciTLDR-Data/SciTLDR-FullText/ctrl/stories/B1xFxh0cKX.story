Many applications in machine learning require optimizing a function whose true gradient is unknown, but where surrogate gradient information (directions that may be correlated with, but not necessarily identical to, the true gradient) is available instead.

This arises when an approximate gradient is easier to compute than the full gradient (e.g. in meta-learning or unrolled optimization), or when a true gradient is intractable and is replaced with a surrogate (e.g. in certain reinforcement learning applications or training networks with discrete variables).

We propose Guided Evolutionary Strategies, a method for optimally using surrogate gradient directions along with random search.

We define a search distribution for evolutionary strategies that is elongated along a subspace spanned by the surrogate gradients.

This allows us to estimate a descent direction which can then be passed to a first-order optimizer.

We analytically and numerically characterize the tradeoffs that result from tuning how strongly the search distribution is stretched along the guiding subspace, and use this to derive a setting of the hyperparameters that works well across problems.

Finally, we apply our method to example problems including truncated unrolled optimization and training neural networks with discrete variables, demonstrating improvement over both standard evolutionary strategies and first-order methods (that directly follow the surrogate gradient).

We provide a demo of Guided ES at: redacted URL

Optimization in machine learning often involves minimizing a cost function where the gradient of the cost with respect to model parameters is known.

When gradient information is available, firstorder methods such as gradient descent are popular due to their ease of implementation, memory efficiency, and convergence guarantees (Sra et al., 2012) .

When gradient information is not available, however, we turn to zeroth-order optimization methods, including random search methods such as evolutionary strategies (Rechenberg, 1973; Nesterov & Spokoiny, 2011; Salimans et al., 2017) .However, what if only partial gradient information is available?

That is, what if one has access to surrogate gradients that are correlated with the true gradient, but may be biased in some unknown fashion?

Naïvely, there are two extremal approaches to optimization with surrogate gradients.

On one hand, you could ignore the surrogate gradient information entirely and perform zeroth-order optimization, using methods such as evolutionary strategies to estimate a descent direction.

These methods exhibit poor convergence properties when the parameter dimension is large BID5 .

On the other hand, you could directly feed the surrogate gradients to a first-order optimization algorithm.

However, bias in the surrogate gradients will interfere with optimizing the target problem (Tucker et al., 2017) .

Ideally, we would like a method that combines the complementary strengths of these two approaches: we would like to combine the unbiased descent direction estimated with evolutionary strategies with the low-variance estimate given by the surrogate gradient.

In this work, we propose a method for doing this called guided evolutionary strategies (Guided ES).The critical assumption underlying Guided ES is that we have access to surrogate gradient information, but not the true gradient.

This scenario arises in a wide variety of machine learning problems, which typically fall into two categories: cases where the true gradient is unknown or not defined, and cases where the true gradient is hard or expensive to compute.

Examples of the former include: models with discrete stochastic variables (where straight through estimators (Bengio et al., Figure 1: (a) Schematic of guided evolutionary strategies.

We perform a random search using a distribution (white contours) elongated along a subspace (white arrow) which we are given instead of the true gradient (blue arrow).

(b) Comparison of different algorithms on a quadratic loss, where a bias is explicitly added to the gradient to mimic situations where the true gradient is unknown.

The loss (left) and correlation between surrogate and true gradient (right) are shown during optimization.

See §4.1 for experimental details.2013; van den Oord et al., 2017) or Concrete/Gumble-Softmax methods (Maddison et al., 2016; BID12 are commonly used) and learned models in reinforcement learning (e.g. for Q functions (Watkins & Dayan, 1992; Mnih et al., 2013; or value estimation (Mnih et al., 2016) ).

For the latter, examples include optimization using truncated backprop through time (Rumelhart et al., 1985; Williams & Peng, 1990; Wu et al., 2018) .

Surrogate gradients also arise in situations where the gradients are explicitly modified during training, as in feedback alignment BID17 and related methods (Nøkland, 2016; BID6 .The key idea in Guided ES is to keep track of a low-dimensional subspace, defined by the recent history of surrogate gradients during optimization, which we call the guiding subspace.

We then perform a finite difference random search (as in evolutionary strategies) preferentially within this subspace.

By concentrating our search samples in a low-dimensional subspace where the true gradient has non-negative support, we dramatically reduce the variance of the search direction.

Our contributions in this work are:• a new method for combining surrogate gradient information with random search,• an analysis of the bias-variance tradeoff underlying the technique ( §3.3),• a scheme for choosing optimal hyperparameters for the method ( §3.4), and• applications to example problems ( §4).

This work builds upon a random search method known as evolutionary strategies (Rechenberg, 1973; Nesterov & Spokoiny, 2011) , or ES for short, which generates a descent direction via finite differences over random perturbations of parameters.

ES has seen a resurgence in popularity in recent years (Salimans et al., 2017; Mania et al., 2018) .

Our method can primarily be thought of as a modification to ES where we augment the search distribution using surrogate gradients.

Extensions of ES that modify the search distribution use natural gradient updates in the search distribution (Wierstra et al., 2008) or construct non-Gaussian search distributions BID7 .

The idea of using gradients in concert with evolutionary algorithms was proposed by BID15 , who use gradients of a network with respect to its inputs (as opposed to parameters) to augment ES.

Other methods for adapting the search distribution include covariance matrix adaptation ES (CMA-ES) BID8 , which uses the recent history of descent steps to adapt the distribution over parameters, or variational optimization (Staines & Barber, 2012) , which optimizes the parameters of a probability distribution over model weights.

Guided ES, by contrast, adapts the search distribution using surrogate gradient information.

In addition, we never need to work with or compute a full n × n covariance matrix.

We wish to minimize a function f (x) over a parameter space in n-dimensions (x ∈ R n ), where ∇f is either unavailable or uninformative.

A popular approach is to estimate a descent direction with stochastic finite differences (commonly referred to as evolutionary strategies (Rechenberg, 1973) or random search (Rastrigin, 1963) ).

Here, we use antithetic sampling (Owen, 2013) (using a pair of function evaluations at x + and x − ) to reduce variance.

This estimator is defined as: DISPLAYFORM0 where i ∼ N (0, σ 2 I), and P is the number of sample pairs.

We will set P to one for all experiments, and when analyzing optimal hyperparameters.

The overall scale of the estimate (β) and variance of the perturbations (σ 2 ) are constants, to be chosen as hyperparameters.

This estimate solely relies on computing 2P function evaluations.

However, it tends to have high variance, thus requiring a large number of samples to be practical, and scales poorly with the dimension n. We refer to this estimator as vanilla evolutionary strategies (or vanilla ES) in subsequent sections.

Even when we do not have access to ∇f , we frequently have additional information about f , either from prior knowledge or gleaned from previous iterates during optimization.

To formalize this, we assume we are given a set of vectors which may correspond to biased or corrupted gradients.

That is, these vectors are correlated (but need not be perfectly aligned) with the true gradient.

If we are given a single vector or surrogate gradient for a given parameter iterate, we can generate a subspace by keeping track of the previous k surrogate gradients encountered during optimization.

We use U to denote an n × k orthonormal basis for the subspace spanned by these vectors (i.e., U T U = I k ).We leverage this information by changing the distribution of i in eq. (1) to N (0, σ 2 Σ) with DISPLAYFORM0 where k and n are the subspace and parameter dimensions, respectively, and α is a hyperparameter that trades off variance between the full parameter space and the subspace.

Setting α = 1 recovers the vanilla ES estimator (and ignores the guiding subspace), but as we show choosing α < 1 can result in significantly improved performance.

The other hyperparameter is the scale β in (1), which controls the size of the estimated descent direction.

The parameter σ 2 controls the overall scale of the variance, and will drop out of the analysis of the bias and variance below, due to the 1 σ 2 factor in (1).

In practice, if f (x) is stochastic, then increasing σ 2 will dampen noise in the gradient estimate, while decreasing σ 2 reduces the error induced by third and higher-order terms in the Taylor expansion of f below.

For an exploration of the effects of σ 2 in ES, see BID14 .Samples of i can be generated efficiently as i = σ DISPLAYFORM1 Our estimator requires 2P function evaluations in addition to the cost of computing the surrogate gradient.

Furthermore, it may be possible to parallelize the forward pass computations.

Figure 1a depicts the geometry underlying our method.

Instead of the true gradient (blue arrow), we are given a surrogate gradient (white arrow) which is correlated with the true gradient.

We use this to form a guiding distribution (denoted with white contours) and use this to draw samples (white dots) which we use as part of a random search procedure. (Figure 1b demonstrates the performance of the method on a toy problem, and is discussed in §4.1.)For the purposes of analysis, suppose ∇f exists.

We can approximate the function in the local neighborhood of x using a second order Taylor approximation: DISPLAYFORM2 .

For the remainder of §3, we take this second order Taylor expansion to be exact.

By substituting this expression into (1), we see that our estimate g is equal to DISPLAYFORM3 Note that even terms in the Taylor expansion cancel out in the expression for g due to antithetic sampling.

The computational and memory costs of using Guided ES to compute parameter updates, compared to standard (vanilla) ES and gradient descent, are outlined in Appendix D.

As we have alluded to, there is a bias-variance tradeoff lurking within our estimate g. In particular, by emphasizing the search in the full space (i.e., choosing α close to 1), we reduce the bias in our estimate at the cost of increased variance.

Emphasizing the search along the guiding subspace (i.e., choosing α close to 0) will induce a bias in exchange for a potentially large reduction in variance, especially if the subspace dimension k is small relative to the parameter dimension n. Below, we analytically and numerically characterize this tradeoff.

Importantly, regardless of the choice of α and β, the Guided ES estimator always provides a descent direction in expectation.

The mean of the estimator in eq. FORMULA4 is E[g] = βΣ∇f (x) corresponds to the gradient multiplied by a positive semi-definite (PSD) matrix, thus the update (−E[g]) remains a descent direction.

This desirable property ensures that α trades off variance for "safe" bias.

That is, the bias will never produce an ascent direction when we are trying to minimize f .The alignment between the k-dimensional orthonormal guiding subspace (U ) and the true gradient (∇f (x)) will be a key quantity for understanding the bias-variance tradeoff.

We characterize this alignment using a k-dimensional vector of uncentered correlation coefficients ρ, whose elements are the correlation between the gradient and every column of U .

That is, DISPLAYFORM0 .

This correlation ρ 2 varies between zero (if the gradient is orthogonal to the subspace) and one (if the gradient is full contained in the subspace).We can evaluate the squared norm of the bias of our estimate g as DISPLAYFORM1 We additionally define the normalized squared bias,b, as the squared norm of the bias divided by the squared norm of the true gradient (this quantity is independent of the overall scale of the gradient).

Plugging in our estimate for g from eq. (2) yields the following expression for the normalized squared bias (see Appendix A.1 for derivation): DISPLAYFORM2 where again β is a scale factor and α is part of the parameterization of the covariance matrix that trades off variance in the full parameter space for variance in the guiding subspace (Σ = DISPLAYFORM3 .

We see that the normalized squared bias consists of two terms: the first is a contribution from the search in the full space and is thus independent of ρ, whereas the second depends on the squared norm of the uncentered correlation, ρ 2 2 .

In addition to the bias, we are also interested in the variance of our estimate.

We use total variance (i.e., tr(Var(g))) to quantify the variance of our estimator DISPLAYFORM4 using an identity for the fourth moment of a Gaussian (see Appendix A.2) and the fact that the trace is linear and invariant under cyclic permutations.

We are interested in the normalized variance,ṽ, which we define as the quantity above divided by the squared norm of the gradient.

Plugging in our estimate g yields the following expression for the normalized variance (see Appendix A.2): DISPLAYFORM5 Equations (4) and FORMULA10 quantify the bias and variance of our estimate as a function of the subspace and parameter dimensions (k and n), the parameters of the distribution (α and β), and the correlation ρ 2 .

Note that for simplicity we have set the number of pairs of function evaluations, P , to one.

As P increases, the variance will decrease linearly, at the cost of extra function evaluations.

FIG1 explores the tradeoff between normalized bias and variance for different settings of the relevant hyperparameters (α and β) for example values of ρ 2 = 0.23, k = 3, and n = 100.

FIG1 shows the sum of the normalized bias plus variance, the global minimum of which (blue star) can be used to choose optimal values for the hyperparameters, discussed in the next section.

, and the sum of both (c) are shown as a function of the tradeoff (α) and scale (β) hyperparameters, for a fixed ρ 2 = 0.23.

For these plots, the subspace dimension was set to k = 3 and the parameter dimension was set to n = 100.

The blue line in (c) denotes the optimal β for every value of α, and the star denotes the global optimum.

The expressions for the normalized bias and variance depend on the subspace and parameter dimensions (k and n, respectively), the hyperparameters of the guiding distribution (α and β) and the uncentered correlation between the true gradient and the subspace ( ρ 2 ).

All of these quantities except for the correlation ρ 2 are known or defined in advance.

To choose optimal hyperparameters, we minimize the sum of the normalized bias and variance, (equivalent to the expected normalized square error in the gradient estimate,b+ṽ = DISPLAYFORM0 ).

This objective becomes: DISPLAYFORM1 subject to the feasibility constraints β ≥ 0 and 0 ≤ α ≤ 1.As further motivation for this hyperparameter objective, in the simple case that f (x) = 1 2 x 2 2 then minimizing eq. (6) also results in the hyperparameters that cause SGD to most rapidly descend f (x).

See Appendix C for a derivation of this relationship.

We can solve for the optimal tradeoff (α * ) and scale (β * ) hyperparameters as a function of ρ 2 , k, and n. FIG2 shows the optimal value for the tradeoff hyperparameter (α * ) in the 2D plane spanned by the correlation ( ρ 2 ) and ratio of the subspace dimension to the parameter dimension k n .

Remarkably, we see that for large regions of the ( ρ 2 , k n ) plane, the optimal value for α is either 0 or 1.

In the upper left (blue) region, the subspace is of high quality (highly correlated with the true gradient) and small relative to the full space, so the optimal solution is to place all of the weight in the subspace, setting α to zero (therefore Σ ∝ U U T ).

In the bottom right (orange) region, we have the opposite scenario, where the subspace is large and low-quality, thus the optimal solution is to place all of the weight in the full space, setting α to one (equivalent to vanilla ES, Σ ∝ I).

The strip in the middle is an intermediate regime where the optimal α is between 0 and 1.We can also derive an expression for when this transition in optimal hyperparameters occurs.

To do this, we use the reparameterization θ = αβ (1 − α)β .

This allows us to express the objective in (6) as a least squares problem DISPLAYFORM2 2 , subject to a non-negativity constraint (θ 0), where A and b depend solely on the problem data k, n, and ρ 2 (see Appendix B.1 for details).

In addition, A is always a positive semi-definite matrix, so the reparameterized problem is convex.

We are particularly interested in the point where the non-negativity constraint becomes tight.

Formulating the Lagrange dual of this problem and solving for the KKT conditions allows us to identify this point using the complementary slackness conditions BID3 .

This yields the equations ρ 2 = k+4 n+4 and ρ 2 = k n (see Appendix B.2), which are shown in FIG2 , and line up with the numerical solution.

FIG2 further demonstrates this tradeoff.

For fixed n = 100, we plot four curves for k ranging from 1 to 30.

As ρ 2 increases, the optimal hyperparameters sweep out a curve from α * = 1, DISPLAYFORM3 In practice, the correlation between the gradient and the guiding subspace is typically unknown.

However, we find that ignoring ρ 2 and setting β = 2 and α = 1 2 works well (these are the values used for all experiments in this paper).

A direction for future work would be to estimate the correlation ρ 2 online, and to use this to choose hyperparameters by minimizing eq. (6).

We first test our method on a toy problem where we control the bias of the surrogate gradient explicitly.

We generated random quadratic problems of the form f (x) = 1 2 Ax − b 2 2 where the entries of A and b were drawn independently from a standard normal distribution, but rather than allow the optimizers to use the true gradient, we (for illustrative purposes) added a random bias to generate surrogate gradients.

Figure 1b compares the performance of stochastic gradient descent (SGD) with standard (vanilla) evolutionary strategies (ES), CMA-ES, and Guided ES.

For this, and all of the results in this paper, we set the hyperparameters as β = 2 and α = 1 2 , as described above.

We see that Guided ES proceeds in two phases: it initially quickly descends the loss as it follows the biased gradient, and then transitions into random search.

Vanilla ES and CMA-ES, however, do not get to take advantage of the information available in the surrogate gradient, and converge more slowly.

We see this also in the plot of the uncentered correlation (ρ) between the true gradient and the surrogate gradient in Figure 1c .

Further experimental details are provided in Appendix E.1.

Another application where surrogate gradients are available is in unrolled optimization.

Unrolled optimization refers to taking derivatives through an optimization process.

For example, this approach has been used to optimize hyperparameters BID4 Maclaurin et al., 2015; BID1 , to stabilize training (Metz et al., 2016) , and even to train neural networks to act as optimizers BID0 Wichrowska et al., 2017; BID16 Lv et al., 2017) .

Taking derivatives through optimization with a large number of steps is costly, so a common approach is to instead choose a small number of unrolled steps, and use that as a target for training.

However, Wu et al. FORMULA0 recently showed that this approach yields biased gradients.

To demonstrate the utility of Guided ES here, we trained multi-layer perceptrons (MLP) to predict the learning rate for a target problem, using as input the eigenvalues of the Hessian at the current iterate.

FIG3 shows the bias induced by unrolled optimization, as the number of optimization steps ranges from one iteration (orange) to 15 (blue).

We compute the surrogate gradient of the parameters in the MLP using the loss after one SGD step.

FIG3 , we show the absolute value of the difference between the optimal learning rate and the MLP prediction for different optimization algorithms.

Further experimental details are provided in Appendix E.2.

Next, we explore using Guided ES in the scenario where the surrogate gradient is not provided, but instead we train a model to generate surrogate gradients (we call these synthetic gradients).

In real-world applications, training a model to produce synthetic gradients is the basis of model-based and actor-critic methods in RL and has been applied to decouple training across neural network layers BID11 and to generate policy gradients BID10 .

A key challenge with such an approach is that early in training, the model generating the synthetic gradients is untrained, and thus will produce biased gradients.

In general, it is unclear during training when following these synthetic gradients will be beneficial.

We define a parametric model, M (x; θ) (an MLP), which provides synthetic gradients for the target problem f .

The target model M (·) is trained online to minimize mean squared error against evaluations of f (x).

FIG5 compares vanilla ES, Guided ES, and the Adam optimizer BID13 .

We show training curves for these methods in FIG5 , and the correlation between the synthetic gradient and true gradients for Guided ES in FIG5 .

Despite the fact that the quality of the synthetic gradients varies wildly during optimization, Guided ES consistently makes progress on the target problem.

Further experimental details are provided in Appendix E.3.

Finally, we applied Guided ES to train neural networks with discrete variables.

Specifically, we trained autoencoders with a discrete latent codebook as in the VQ-VAE (van den Oord et al., 2017) on MNIST.

The encoder and decoder were fully connected networks with two hidden layers.

We use the straight-through estimator BID2 taken through the discretization step as the surrogate gradient.

For Guided ES, we computed the Guided ES update only for the encoder weights, as those are the only parameters with biased gradients (due to the straight-through estimator)-the other weights in the network were trained directly with Adam.

FIG6 shows the training loss using Adam, standard (vanilla) ES, and Guided ES (note that vanilla ES does not make progress on this timescale due to the large number of parameters (n = 152912)).

We achieve a small improvement, likely due to the biased straight-through gradient estimator leading to suboptimal encoder weights.

The correlation between the Guided ES update step and the straight-through gradient FIG6 ) can be thought of as a metric for the quality of the surrogate gradient (which is fairly high for this problem).

Overall, this demonstrates that we can use Guided ES and first-order methods together, applying the Guided ES update only to the parameters that have surrogate gradients (and using firstorder methods for the parameters that have unbiased gradients).

Further experimental details are provided in Appendix E.4.

We have introduced guided evolutionary strategies (Guided ES), an optimization algorithm which combines the benefits of first-order methods and random search, when we have access to surrogate gradients that are correlated with the true gradient.

We analyzed the bias-variance tradeoff inherent in our method analytically, and demonstrated the generality of the technique by applying it to unrolled optimization, synthetic gradients, and training neural networks with discrete variables.

The squared bias norm is defined as: DISPLAYFORM0 where ∼ N (0, Σ) and the covariance is given by: DISPLAYFORM1 This expression reduces to (recall that U is orthonormal, so U T U = I): DISPLAYFORM2 Dividing by the norm of the gradient ( ∇f (x) 2 2 ) yields the expression for the normalized bias (eq. (4) in the main text).

First, we state a useful identity.

Suppose ∼ N (0, Σ), then DISPLAYFORM0 We can see this by observing that the (i, k) entry of E[ DISPLAYFORM1 by Isserlis' theorem, and then we recover the identity by rewriting the terms in matrix notation.

The total variance is given by: DISPLAYFORM2 Using the identity above, we can express the total variance as: DISPLAYFORM3 Since the trace of the covariance matrix Σ is 1, we can expand the quantity tr(Σ)Σ + Σ 2 as: DISPLAYFORM4 Thus the expression for the total variance reduces to: DISPLAYFORM5 and dividing by the norm of the gradient yields the expression for the normalized variance (eq. FORMULA10 in the main text).B OPTIMAL HYPERPARAMETERS

We wish to minimize the sum of the normalized bias and variance, eq. (6) in the main text.

First, we use a reparameterization by using the substitution θ 1 = αβ and θ 2 = (1 − α)β.

This substitution yields:b DISPLAYFORM0 which is quadratic in θ.

Therefore, we can rewrite the problem as: DISPLAYFORM1 , where A and b are given by: DISPLAYFORM2 Note that A and b depend on the problem data (k, n, and ρ 2 ), and that A is a positive semi-definite matrix (as k and n are non-negative integers, and ρ 2 is between 0 and 1).

In addition, we can express the constraints on the original parameters (β ≥ 0 and 0 ≤ α ≤ 1) as a non-negativity constraint in the new parameters (θ 0).

The optimal hyperparameters are defined (see main text) as the solution to the minimization problem: DISPLAYFORM0 where θ = αβ (1 − α)β are the hyperparameters to optimize, and A and b are specified in eq. (7).The Lagrangian for FORMULA27 is given by L(θ, λ) = 1 2 Aθ − b 2 2 − λ T θ, and the corresponding dual problem is: maximize DISPLAYFORM1 Since the primal is convex, we have strong duality and the Karush-Kuhn-Tucker (KKT) conditions guarantee primal and dual optimality.

These conditions include primal and dual feasibility, that the gradient of the Lagrangian vanishes (∇ θ L(θ, λ) = Aθ − b − λ = 0), and complimentary slackness (which ensures that for each inequality constraint, either the constraint is satisfied or λ = 0).Solving the condition on the gradient of the Langrangian for λ yields that the lagrange multipliers λ are simply the residual λ = Aθ − b. Complimentary slackness tells us that λ i θ i = 0, for all i.

We are interested in when this constraint becomes tight.

To solve for this, we note that there are two regimes where each of the two inequality constraints is tight (the blue and orange regions in FIG2 ).

These occur for the solutions θ (1) = 0 k k+2 (when the first inequality is tight) and DISPLAYFORM2 (when the second inequality is tight).

To solve for the transition point, we solve for the point where the constraint is tight and the lagrange multiplier (λ) equals zero.

We have two inequality constraints, and thus will have two solutions (which are the two solid curves in FIG2 ).

Since the lagrange multiplier is the residual, these points occur when Aθ DISPLAYFORM3 The first solution θ (1) = 0 k k+2 yields the upper bound: DISPLAYFORM4 And the second solution θ (2) = n n+2 0 yields the lower bound: DISPLAYFORM5 These are the equations for the lines separating the regimes of optimal hyperparameters in FIG2 .

Choosing hyperparameters which most rapidly descend the simple quadratic loss in eq. FORMULA0 is equivalent to choosing hyperparameters which minimize the expected square error in the estimated gradient, as is done in §3.4.

This provides further support for the method used to choose hyperparameters in the main text.

Here we derive this equivalence.

Assume a loss function of the form DISPLAYFORM0 and that updates are performed via gradient descent with learning rate 1, x ← x − g.

The expected loss after a single training step is then DISPLAYFORM1 For this problem, the true gradient is simply ∇f (x) = x. Substituting this into eq. (11), we find DISPLAYFORM2 Up to a multiplicative constant, this is exactly the expected square error between the descent direction g and the gradient ∇f (x) used as the objective for choosing hyperparameters in §3.4.

Here, we outline the computational and memory costs of Guided ES and compare them to standard (vanilla) evolutionary strategies and gradient descent.

As elsewhere in the paper, we define the parameter dimension as n and the number of pairs of function evaluations (for evolutionary strategies) as P .

We denote the cost of computing the full loss as F 0 , and (for Guided ES and gradient descent), we assume that at every iteration we compute a surrogate gradient which has cost F 1 .

Note that for standard training of neural networks with backpropogation, these quantities have similar cost (F 1 ≈ 2F 0 ), however for some applications (such as unrolled optimization discussed in §4.2) these can be very different.

Computational cost Memory cost Gradient descent F 1 n Vanilla evolutionary strategies 2P F 0 n Guided evolutionary strategies F 1 + 2P F 0 (k + 1)n Table 1 : Per-iteration compute and memory costs for gradient descent, standard (vanilla) evolutionary strategies, and the method proposed in this paper, guided evolutionary strategies.

Here, F 0 is the cost of a function evaluation, F 1 is the cost of computing a surrogate gradient, n is the parameter dimension, k is the subspace dimension used for the guiding subspace, and P is the number of pairs of function evaluations used for the evolutionary strategies algorithms.

Below, we give detailed methods used for each of the experiments from §4.

For each problem, we specify a desired loss function that we would like to minimize (f (x)), as well as specify the method for generating a surrogate or approximate gradient (∇f (x)).

Our target problem is linear regression, DISPLAYFORM0 , where A is a random M × N matrix and b is a random M -dimensional vector.

The elements of A and b were drawn IID from a standard Normal distribution.

We chose N = 1000 and M = 2000 for this problem.

The surrogate gradient was generated by adding a random bias (drawn once at the beginning of optimization) and noise (resampled at every iteration) to the gradient.

These quantities were scaled to have the same norm as the gradient.

Thus, the surrogate gradient is given by: ∇f (x) = ∇f (x) + (b + n) ∇f (x) 2 , where b and n are unit norm random vectors that are fixed (bias) or resampled (noise) at every iteration.

The plots in Figure 1b show the loss suboptimality (f (x) − f * ), where f * is the minimum of f (x) for a particular realization of the problem.

The parameters were initialized to the zeros vector and optimized for 10,000 iterations.

Figure 1b shows the mean and spread (std. error) over 10 random seeds.

For each optimization algorithm, we performed a coarse grid search over the learning rate for each method, scanning 17 logarithmically spaced values over the range (10 −5 , 1).

The learning rates chosen were: 5e-3 for gradient descent, 0.2 for guided and vanilla ES, and 1.0 for CMA-ES.

For the two evolutionary strategies algorithms, we set the overall variance of the perturbations as σ = 0.1 and used P = 1 pair of samples per iteration.

The subspace dimension for Guided ES was set to k = 10.

The results were not sensitive to the choices for σ, P , or k.

We define the target problem as the loss of a quadratic after running T = 15 steps of gradient descent.

The quadratic has the same form as described above, DISPLAYFORM0 2 , but with M = 20 and N = 10.

The learning rate for the optimizer was taken as the output of a multilayer perceptron (MLP), with three hidden layers containing 32 hidden units per layer and with rectified linear (ReLU) activations after each hidden layer.

The inputs to the MLP were the 10 eigenvalues of the Hessian, A T A, and the output was a single scalar that was passed through a softplus nonlinearity (to ensure a positive learning rate).

Note that the optimal learning rate for this problem is 2M λmin+λmax , where λ min and λ max are the minimum and maximum eigenvalues of A T A, respectively.

The surrogate gradients for this problem were generated by backpropagation through the optimization process, but by unrolling only T = 1 optimization steps (truncated backprop).

FIG3 shows the distance between the MLP predicted learning rate and the optimal learning rate 2M λmin+λmax , during the course of optimization of the MLP parameters.

That is, FIG3 shows the progress on the meta-optimization problems (optimizing the MLP to predict the learning rate) using the three different algorithms (SGD, vanilla ES, and guided ES).As before, the mean and spread (std. error) over 10 random seeds are shown, and the learning rate for each of the three methods was chosen by a grid search over the range (10 −5 , 10).

The learning rates chosen were 0.3 for gradient descent, 0.5 for guided ES, and 10 for vanilla ES.

For the two evolutionary strategies algorithms, we set the variance of the perturbations to σ = 0.01 and used P = 1 pair of samples per iteration.

The results were not sensitive to the choices for σ, P , or k.

Here, the target problem consisted of a mean squared error objective, f (x) = 1 2 x − x * 2 2 , where x * was random sampled from a uniform distribution between [-1, 1].

The surrogate gradient was defined as the gradient of a model, M (x; θ), with inputs x and parameters θ.

We parameterize this model using a multilayered perceptron (MLP) with two 64-unit hidden layers and relu activations.

The surrogate gradients were taken as the gradients of M with respect to x: ∇f (x) = ∇ x M (x; θ).The model was optimized online during optimization of f by minimizing the mean squared error with the (true) function observations: DISPLAYFORM0 2 .

The data D used to train M were randomly sampled in batches of size 512 from the most recent 8192 function evaluations encountered during optimization.

This is equivalent to uniformly sampling from a replay buffer, a strategy commonly used in reinforcement learning.

We performed one θ update per x update with Adam with a learning rate of 1e-4.The two evolutionary strategies algorithms inherently generate samples of the function during optimization.

In order to make a fair comparison when optimizing with the Adam baseline, we similarly generated function evaluations for training the model M by sampling points around the current iterate from the same distribution used in vanilla ES (Normal with σ = 0.1).

This ensures that the amount and spread of training data for M (in the replay buffer) when optimizing with Adam is similar to the data in the replay buffer when training with vanilla or guided ES.

FIG5 shows the mean and spread (standard deviation) of the performance of the three algorithms over 10 random instances of the problem.

We set σ = 0.1 and used P = 1 pair of samples per iteration.

For Guided ES, we used a subspace dimension of k = 1.

The results were not sensitive to the number of samples P , but did vary with σ, as this controls the spread of the data used to train M , thus we tuned σ with a coarse grid search.

We trained a vector quantized variational autoencoder (VQ-VAE) as defined in van den Oord et al. (2017) on MNIST.

Our encoder and decoder networks were both fully connected neural networks with 64 hidden units per layer and ReLU nonlinearities.

For the vector quantization, we used a small codebook (twelve codebook vectors).

The dimensionality of the codebook and latent variables was 16, and we used 10 latent variables.

To train the encoder weights, van den Oord et al. (2017) proposed using a straight through estimator BID2 to bypass the discretization in the vector quantizer.

Here, we use this as the surrogate gradient passed to Guided ES.

Since the gradients are correct (unbiased) for the decoder and embedding weights, we do not use Guided ES on those variables, instead using first-order methods (Adam) directly.

For training with vanilla ES or Guided ES, we used P = 10 pairs of function evaluations per iteration to reduce variance (note that these can be done in parallel).

<|TLDR|>

@highlight

We propose an optimization method for when only biased gradients are available--we define a new gradient estimator for this scenario, derive the bias and variance of this estimator, and apply it to example problems.

@highlight

The authors propose an approach that combines random search with the surrogate gradient information and give a discussion on variance-bias trade-off as well as a discussion on hyperparameter optimization.

@highlight

 The paper proposes a method to improve random search by building a subspace of the previous k surrogate gradients.

@highlight

This paper attempts accelerating the OpenAI type evolution by introducing a non-isotrophic distribution with a covariance matrix in the form I + UU^t and external information such as a surrogate gradient to determine U
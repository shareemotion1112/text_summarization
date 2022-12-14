We propose a fast second-order method that can be used as a drop-in replacement for current deep learning solvers.

Compared to stochastic gradient descent (SGD), it only requires two additional forward-mode automatic differentiation operations per iteration, which has a computational cost comparable to two standard forward passes and is easy to implement.

Our method addresses long-standing issues with current second-order solvers, which invert an approximate Hessian matrix every iteration exactly or by conjugate-gradient methods, procedures that are much slower than a SGD step.

Instead, we propose to keep a single estimate of the gradient projected by the inverse Hessian matrix, and update it once per iteration with just two passes over the network.

This estimate has the same size and is similar to the momentum variable that is commonly used in SGD.

No estimate of the Hessian is maintained.

We first validate our method, called CurveBall, on small problems with known solutions (noisy Rosenbrock function and degenerate 2-layer linear networks), where current deep learning solvers struggle.

We then train several large models on CIFAR and ImageNet, including ResNet and VGG-f networks, where we demonstrate faster convergence with no hyperparameter tuning.

We also show our optimiser's generality by testing on a large set of randomly-generated architectures.

Stochastic Gradient Descent (SGD) and back-propagation BID16 are the algorithmic backbone of current deep network training.

The success of deep learning demonstrates the power of this combination, which has been successfully applied on various tasks with large datasets and very deep networks BID11 ).Yet, while SGD has many advantages, speed of convergence (in terms of number of iterations) is not necessarily one of them.

While individual SGD iterations are very quick to compute and lead to rapid progress at the beginning of the optimisation, it soon reaches a slower phase where further improvements are achieved slowly.

This can be attributed to entering regions of the parameter space where the objective function is poorly scaled.

In such cases, rapid progress would require vastly different step sizes for different directions in parameter space, which SGD cannot deliver.

Second-order methods, such as Newton's method and its variants, eliminate this issue by rescaling the gradient according to the local curvature of the objective function.

For a scalar loss in R, this rescaling takes the form H ???1 J where H is the Hessian matrix (second-order derivatives) or an approximation of the local curvature in the objective space, and J is the gradient of the objective.

They can in fact achieve local scale-invariance (Wright & Nocedal, 1999, p. 27) , and make provably better progress in the regions where gradient descent stalls.

While they are unmatched in other domains, there are several obstacles to their application to deep models.

First, it is impractical to invert or even store the Hessian, since it grows quadratically with the number of parameters, and there are typically millions of them.

Second, any Hessian estimate is necessarily noisy and ill-conditioned due to stochastic sampling, to which classic inversion methods such as conjugate-gradient are not robust.

In this paper, we propose a new algorithm that can overcome these difficulties and make second order optimisation practical for deep learning.

We show in particular how to avoid the storage of any estimate of the Hessian matrix or its inverse.

Instead, we treat the computation of the Newton update, H ???1 J, as solving a linear system that itself can be solved via gradient descent.

The cost of solving this system is amortized over time by interleaving its steps with the parameter update steps.

Our proposed method adds little overhead, since a Hessian-vector product can be implemented for modern networks with just two steps of automatic differentiation.

Interestingly, we show that our method is equivalent to momentum SGD (also known as the heavy-ball method) with a single additional term, accounting for curvature.

For this reason we named our method CURVEBALL.

Unlike other proposals, the total memory footprint is as small as that of momentum SGD.

This paper is structured as follows.

We introduce relevant technical background in sec. 2, and present our method in sec. 3.

We evaluate our method and show experimental results in sec. 4.

Related work is discussed in sec. 5.

Finally we summarise our findings in sec. 6.

In order to make the description of our method self-contained, we succinctly summarise a few standard concepts in optimisation.

Our goal is to find the optimal parameters of a model (e.g. a neural network) ?? : R p ??? R o , with p parameters w ??? R p and o outputs (the notation does not show the dependency on the training data, which is subsumed in ?? for compactness).

The quality of the outputs is evaluated by a loss function L : R o ??? R, so finding w is reduced to the optimisation problem: DISPLAYFORM0 Perhaps the simplest algorithm to find an optimum (or at least a stationary point) of eq. 1 is gradient descent (GD).

GD updates the parameters using the iteration w ??? w ??? ??J(w), where ?? > 0 is the learning rate and J(w) ??? R p is the gradient (or Jacobian) of the objective function f with respect to the parameters w. A useful variant is to augment GD with a momentum variable z (Polyak, 1964), which can be interpreted as a decaying average of past gradients: DISPLAYFORM1 with a momentum parameter ??.

Momentum GD, as given by eq. 2-3, can be shown to have faster convergence than GD for convex functions, remaining stable under higher learning rates, and exhibits somewhat better resistance to poor scaling of the objective function BID24 Goh, 2017) .

One important aspect is that these advantages cost almost no additional computation and only a modest additional memory, which explains why it is widely used in practice.

In neural networks, GD is usually replaced by its stochastic version (SGD), where at each iteration one computes the gradient not of the model f = L(??(w)), but of the model f t = L t (?? t (w)) assessed on a small batch of samples, drawn at random from the training set.

As mentioned in section 1, the Newton method is similar to GD, but steers the gradient by the inverse Hessian matrix, computing H ???1 J as a descent direction.

However, inverting the Hessian may be numerically unstable or the inverse may not even exist.

To address this issue, the Hessian is usually regularized with a parameter ?? > 0, obtaining what is known as the Levenberg (Mor??, 1978) method: DISPLAYFORM0 where H ??? R p??p , J ??? R p and I ??? R p??p is the identity matrix.

Note that, unlike for momentum GD (eq. 2), the new step z is independent of the previous step.

To avoid burdensome notation, we omit the w argument in H(w) and J(w), but they must be recomputed at each iteration.

Intuitively, the effect of eq. 4 is to rescale the step appropriately for different directions -directions with high curvature require small steps, while directions with low curvature require large steps to make progress.

Note also that Levenberg's regularization loses the scale-invariance of the original Newton method, meaning that rescaling the function f changes the scale of the gradient and hence the regularised descent direction chosen by the method.

An alternative that alleviates this issue is LevenbergMarquardt, which replaces I in eq. 4 with diag(H).

For non-convex functions such as deep networks, these methods only converge to a local minimum when the Hessian is positive-semidefinite (PSD).

In order to introduce fast computations involving the Hessian, we must take a short digression into how Jacobians are computed.

The Jacobian of L(??(w)) (eq. 1) is generally computed as J = J ?? J L where J ?? ??? R p??o and J L ??? R o??1 are the Jacobians of the model and loss, respectively.

In practice, a Jacobian is never formed explicitly, but Jacobian-vector products Jv are implemented with the back-propagation algorithm.

We define DISPLAYFORM0 as the reverse-mode automatic differentiation (RMAD) operation, commonly known as backpropagation.

Note that, because the loss is a scalar function, the starting projection vector v typically used in gradient descent is a scalar and we set v = 1.

For intermediate computations, however, it is generally a (vectorized) tensor of gradients (see eq. 9).A perhaps lesser known alternative is forward-mode automatic differentiation (FMAD), which computes a vector-Jacobian product, from the other direction: DISPLAYFORM1 This variant is less commonly-known in deep learning as RMAD is appropriate to compute the derivatives of a scalar-valued function, such as the learning objective, whereas FMAD is more appropriate for vector-valued functions of a scalar argument.

However, we will show later that FMAD is relevant in calculations involving the Hessian.

The only difference between RMAD and FMAD is the direction of associativity of the multiplication: FMAD propagates gradients in the forward direction, while RMAD (or back-propagation) does it in the backward direction.

For example, for the composition of functions a DISPLAYFORM2 Because of this, both operations have similar computational overhead, and can be implemented similarly.

Since the Hessian of learning objectives involving deep networks is not necessarily positive semidefinite (PSD), it is common to use a surrogate matrix with this property, which prevents second-order methods from being attracted to saddle-points (this problem is discussed by BID7 ).

One of the most widely used is the Gauss-Newton approximation BID21 BID2 Wright & Nocedal, 1999, p. 254 ): DISPLAYFORM0 When H L is PSD, which is the case for all convex losses (e.g. logistic loss, L p distance), the resultin?? H is PSD by construction.

For the method that we propose, and indeed for any method that implicitly inverts the Hessian (or its approximation), only computing Hessian-vector products??v is required.

As such, eq. 10 takes a very convenient form: DISPLAYFORM1 The cost of eq. 12 is thus equivalent to that of two back-propagation operations.

This is similar to a classic result BID28 BID32 , but written in terms of common automatic differentiation operations.

The intermediate matrix-vector product H L u has negligible cost: for example, for the squared-distance loss, DISPLAYFORM2 , where p is the vector of predictions from a softmax layer and is the element-wise product.

These products thus require only element-wise operations.

In general, a step z is found by minimizing eq. 14, either via explicit inversion?? ???1 J BID21 BID2 or the conjugate-gradient (CG) method BID20 .

The later approach, called the Hessian-free method (also Truncated Newton or Newton-CG (Wright & Nocedal, 1999, p. 168) ) is the most economical in terms of memory, since it only needs access to Hessian-vector products (section 2.3).

A high-level view is illustrated in Algorithm 2, where CG stands for one step of conjugate-gradient (a stopping condition, line search and some intermediate variables were omitted for clarity).

Note that for every update of w (outer loop), Algorithm 2 must perform several steps of CG (inner loop) to find a single search direction z.

We propose a number of changes in order to eliminate this costly inner loop.

The first is to reuse the previous search direction z to warm-start the inner iterations, instead of resetting z each time (Algorithm 2, line 2).

If z does not change abruptly, then this should help reduce the number of CG iterations, by starting closer to the solution.

The second change is to use this fact to dramatically reduce the inner loop iterations to just one (R = 1).

A different interpretation is that we now interleave the updates of the search direction z and parameters w (Algorithm 1, lines 4 and 5), instead of nesting them (Algorithm 2, lines 4 and 6).Unfortunately, this change loses the guarantees of the CG method, which depend on the starting point being z 0 = ???J(w t ) (Wright & Nocedal, 1999, p. 124) .

This loss of guarantee was verified in practice, as we found the resulting algorithm extremely unstable.

Our third change is then to replace CG with gradient descent, which has no such dependency.

Differentiating eq. 14 w.r.t.

z yields: DISPLAYFORM3 Applying these changes to the Hessian-free method (Algorithm 2) results in Algorithm 1.

By contrasting it to momentum GD (eq. 2-3), we can see that it is equivalent, except for an extra curvature term??(w)z.

In order to establish this equivalence, we introduced a factor ?? that decays z each step (Algorithm 1, line 4), whereas a pure gradient descent step on z would not include this factor.

We can obtain it formally by simply regularizing the quadratic model in eq. 13 with the term (1 ??? ??) z 2 , which is small when ?? 1 (the recommended setting for the momentum parameter BID11 ).

Due to the addition of curvature to momentum GD, which is also known as the heavy-ball method, we name our algorithm CURVEBALL.Implementation.

Using the fast Hessian-vector products from section 2.3, it is easy to implement eq. 15, including a regularization term ??I (section 2.1).

We can further improve eq. 15 by grouping Levenberg-Marquardt BID23 16 ?? 4 14 ?? 3 17 ?? 4 9 ?? 4 BFGS (Wright & Nocedal, 1999, p. 136) 19 ?? 4 44 ?? 21 63 ?? 29 43 ?? 21Exact Hessian 14 ?? 1 10 ?? 3 17 ?? 4 9 ?? 0.5 DISPLAYFORM4 the operations to minimize the number of automatic differentiation (back-propagation) steps: DISPLAYFORM5 In this way, the total number of passes over the model is two: we compute J ?? v and J T ?? v products, implemented respectively as one RMAD (back-propagation) and one FMAD operation (section 2.2).Automatic ?? and ?? hyper-parameters in closed form.

Our proposed method introduces a few hyper-parameters, which just like with SGD, would require tuning for different settings.

Ideally, we would like to have no tuning at all.

Fortunately, the quadratic minimization interpretation in eq. 14 allows us to draw on standard results in optimisation.

At any given step, the optimal ?? and ?? can be obtained by solving a 2 ?? 2 linear system BID21 , sec. 7): DISPLAYFORM6 Note that, in calculating the proposed update (eq. 16), the quantities ??? z , J T ?? z and J L have already been computed and can now be reused.

Together with the fact that?? = J ?? H L J T ?? , this means that the elements of the above 2 ?? 2 matrix can be computed with only one additional forward pass.

Automatic ?? hyper-parameter rescaling.

The regularization term ??I (eq. 4) can be interpreted as a trust-region (Wright & Nocedal, 1999, p. 68) .

When the second-order approximation holds well, ?? can be small, corresponding to an unregularized Hessian and a large trust-region.

Conversely, a poor fit requires a correspondingly large ??.

We can measure the difference (or ratio) between the objective change predicted by the quadratic fit (f ) and the real objective change (f ), by computing ?? = (f (w + z) ??? f (w)) /f (z).

This requires one additional evaluation of the objective for f (w + z), but otherwise relies only on previously computed quantities.

This makes it a very attractive estimate of the trust region, with ?? = 1 corresponding to a perfect approximation.

Following (Wright & Nocedal, 1999, p. 69) , we evaluate ?? every 5 iterations, decreasing ?? by a factor of 0.999 when ?? > 3/2, and increasing by the inverse factor when ?? < 1/2.

We noted that our algorithm is not very sensitive to the initial ??.

In experiments using batch-normalization (section 4), we simply initialize it to one, otherwise setting it to 10.

We plot the evolution of automatically tuned hyper-parameters in FIG6 (Appendix B).Convergence.

In addition to the usual absence of strong guarantees for non-convex problems, which applies in our setting (deep neural networks), there is an added difficulty due to the recursive nature of our algorithm (the interleaved w and z steps).

Our method is a variant of the heavy-ball method BID17 BID9 (by adding a curvature term), which until very recently had resisted establishing global convergence rates that improve on gradient descent without momentum BID18 , table 1), and even then only for strongly convex or quadratic functions.

For this reason, we present proofs for two more tractable cases (Appendix A).

The first is the global linear convergence of our method for convex quadratic functions, which allows a direct inspection of the region of convergence as well as its rate (Theorem A.1).

The second establishes that, for convex non-quadratic functions, CURVEBALL's steps are always in a descent direction, when using the automatic hyper-parameter tuning of eq. 18 (Theorem A.2).

We note that in practice, due to the Gauss-Newton approximation and the trust region (eq. 4), the effective Hessian is guaranteed to be positive-definite.

Similarly to momentum SGD, our main claim as to the method's suitability for non-convex deep network optimisation is necessarily empirical, based on the extensive experiments in section 4, which show strong performance on several large-scale problems with no hyper-parameter tuning.

Degenerate problems with known solutions.

While the main purpose of our optimizer is its application to large-scale deep learning architecture, we begin by applying our method to problems of limited complexity, with the goal of exploring the strengths and weaknesses of our approach in an interpretable domain.

We perform a comparison with two popular first order solvers -SGD with momentum and Adam BID12 3 , as well as with more traditional methods such as Levenberg-Marquardt, BFGS (Wright & Nocedal, 1999, p. 136 ) (with cubic line-search) and Newton's method with the exact Hessian.

The first problem we consider is the search for the minimum of the two-dimensional Rosenbrock test function, which has the useful benefit of enabling us to visualise the trajectories found by each optimiser.

Specifically, we use the stochastic variant of this function BID39 , R : R 2 ??? R: DISPLAYFORM0 where at each evaluation of the function, a noise sample i is drawn from a uniform distribution U[?? 1 , ?? 2 ] with ?? 1 , ?? 2 ??? R (we can recover the deterministic Rosenbrock function with ?? 1 = ?? 2 = 1).To assess robustness to noise, we compare each optimiser on the deterministic formulation and two stochastic variants (with differing noise regimes).

We also consider a second problem of interest, recently introduced by BID30 .

It consists of fitting a deep network with only two linear layers to a dataset where sample inputs x are related to sample outputs y by the relation y = Ax, where A is an ill-conditioned matrix (with condition number = 10 5 ).The results are shown in TAB0 .

We use a grid-search to determine the best hyper-parameters for both SGD and Adam (reported in appendix B.1).

We report the number of iterates taken to reach the solution, with a tolerance of ?? = 10 ???4 .

Statistics are computed over 100 runs of each optimiser.

We observe that first-order methods perform poorly in all cases, and moreover show a very high variance of results.

The Newton method with an exact Hessian 4 generally performs best, followed closely by Levenberg-Marquardt (LM), however they are impractical for larger-scale problems.

Our method delivers comparable (and sometimes better) performance despite avoiding a costly Hessian inversion.

On the other hand, the performance of BFGS, which approximates the Hessian with a buffer of parameter updates, seems to correlate negatively with the level of noise.

FIG0 shows example trajectories.

The slow, oscillating behaviour of first-order methods is noticeable, as well as the impact of noise on the BFGS steps.

On the other hand, CURVEBALL, Newton and LM converge in few iterations.

CIFAR.

We now turn to the task of training deep networks on more realistic datasets.

Second-order methods are typically not used in such scenarios, due to the large number of parameters and stochastic sampling.

We start with a basic 5-layer convolutional neural network (CNN).

5 We train this network for 20 epochs on CIFAR-10, with and without batch-normalization (which is known to improve conditioning BID14 ) using for every experiment a mini-batch size of 128.

To assess optimiser performance on larger models, we also train a much larger ResNet-18 model BID11 .

As baselines, we picked SGD (with momentum) and Adam, which we found to outperform the competing first-order optimisers.

Their learning rates are chosen from the set 10 ???k , k ??? N with a grid search for the basic CNN, while for the ResNet SGD uses the schedule recommended by the authors BID11 .

We focus on the training error, since it is the quantity being optimised by eq. 1 (validation error is discussed below).

The results can be seen in FIG1 .

We observe that in each setting, CURVEBALL outperforms its competitors, in a manner that is robust to normalisation and model type.

ImageNet.

To assess the practicality of our method at larger scales, we apply it to the classification task on the large-scale ImageNet dataset.

We report results of training on both a medium-scale setting using a subset formed from the images of 100 randomly sampled classes as well as the large-scale setting, by training on the full dataset.

Both experiments use the VGG-f architecture with mini-batch size of 256 and follow the settings described by BID5 .

The results are depicted in FIG1 .

We see that our method provides compelling performance against popular first order solvers in both cases, and that interestingly, its margin of improvement grows with the scale of the dataset.

Comparison to other second-order methods on MNIST.

In order to compare ours with existing second-order methods, we use the public KFAC BID21 implementation made available by the authors and run a simple experiment on the MNIST dataset.

In this scenario a four layer MLP (with output sizes 128-64-32-10) with hyperbolic tangent activations is trained on this classification task.

We closely follow the same protocol as BID21 for layer initialisation and data normalisation, with batch size 64.

We show results in FIG1 with the best learning rate for each method.

On this problem our method performs comparably to first order solvers, while KFAC makes less progress until it has stabilised its Fisher matrix estimation.

Random architecture results.

It can be argued that standard architectures are biased to favour SGD, since it was used in the architecture searches, and architectures in which it failed to optimise were discarded BID11 .

It would be useful to assess the optimisers' ability to generalise across architectures, testing how well they perform regardless of the network model.

We make an attempt in this direction by comparing the optimisers on 50 deep CNN architectures that are generated randomly (see appendix B.3 for details).

In addition to being more architecture-agnostic, this makes any hand-tuning of hyper-parameters infeasible, which we believe to be a fair requirement for a dependable optimiser.

The results on CIFAR10 are shown in FIG2 , as the median across all runs (thick lines) and 25 th -75 th percentiles (shaded regions).

CURVEBALL consistently outperforms first-order methods, with the bulk of the achieved errors below those of SGD and Adam.

Wall-clock time.

To provide an estimate of the relative efficiency of each model, FIG2 shows wall clock time on the basic CIFAR-10 model (without batch norm).

Importantly, from a practical perspective, we observe that our method is competitive with first order solvers, while not requiring any tuning.

Moreover, our prototype implementation includes custom FMAD operations which have not received the same degree of optimisation as RMAD (back-propagation), and could further benefit from careful engineering.

We also experimented with a Hessian-free optimiser (based on conjugate gradients) BID20 .

We show a comparison in logarithmic time in the appendix FIG5 .

Due to the costly CG operation, which requires several passes through the network, it is an order of magnitude slower than the first-order methods and our own second-order method.

This validates our initial motivation of designing a Hessian-free method without the inner CG loop (Section 3).Overfitting and validation error While the focus of this work is optimisation, it is also of interest to compare the validation errors attained by the trained models -these are reported TAB1 .

We observe that models trained with the proposed method exhibit better training and validation error on most models, with the exception of ResNet where overfitting plays a more significant role.

However, we note that this could be addressed with better regularisation, and we show one such example, by also reporting the validation error with a dropout rate of 0.3 in brackets.

While second order methods have proved to be highly effective tools for optimising deterministic functions BID23 Wright & Nocedal, 1999, p. 164 ) their application to stochastic optimisation, and in particular to deep neural networks remains an active area of research.

Many methods have been developed to improve stochastic optimisation with curvature information to avoid slow progress in ill-conditioned regions BID7 , while avoiding the cost of storing and inverting a Hessian matrix.

A popular approach is to construct updates from a buffer of parameter gradients and their first-and-second-order moments at previous iterates (e.g. AdaGrad BID8 , AdaDelta (Zeiler, 2012), RMSProp BID34 or Adam (Kingma & Ba, 2014) ).

These solvers benefit from needing no additional function evaluations beyond traditional mini-batch stochastic gradient descent.

Typically they set adaptive learning rates by making use of empirical estimates of the curvature with a diagonal approximation to the Hessian (e.g. Zeiler (2012)) or a rescaled diagonal Gauss-Newton approximation (e.g. BID8 ).

While the diagonal structure decreases the computational cost, their overall efficiency remains limited and in many cases can be matched by a well tuned SGD solver BID35 .Second order solvers take a different approach, investing more computation per iteration in the hope of achieving higher quality updates.

Trust-region methods BID6 and cubic regularization BID25 BID4 are canonical examples.

To achieve this higher quality, they invert the Hessian matrix H, or a tractable approximation such as the GaussNewton approximation BID20 BID22 BID2 ) (described in section 2), or other regularized BID7 or subsampled versions of the Hessian BID15 BID37 .

Another line of work belonging to the trust-region family BID6 , which has proven effective for tasks such as classification, introduces second order information with natural gradients (Amari, 1998).

In this context, it is common to derive a loss function from a Kullback-Leibler (KL) divergence.

The natural gradient makes use of the infinitesimal distance induced by the latter to follow the curvature in the Riemannian manifold equipped with this new distance.

In practice the natural gradient method amounts to replacing the Hessian H in the modified gradient formula H ???1 J with the Fisher matrix F , which facilitates traversal of the optimal path in the metric space induced by the KL-divergence.

Since the seminal work of Amari (1998) several authors have studied variations of this idea.

TONGA BID31 relies on the empirical Fisher matrix where the previous expectation over the model predictive distribution is replaced by the sample predictive distribution.

The works of BID27 and Martens established a link between Gauss-Newton methods and the natural gradient.

More recently BID21 introduced the KFAC optimiser which uses a block diagonal approximation of the Fisher matrix.

This was shown to be an efficient stochastic solver in several settings, but it remains a computationally challenging approach for larger-scale deep networks problems.

Many of the methods discussed above perform an explicit system inversion that can often prove prohibitively expensive BID38 .

Consequently, a number of works BID20 BID22 BID41 have sought to exploit the cheaper computation of Hessian-vector products via automatic differentiation BID28 BID32 , to perform system inversions with conjugate gradients (Hessian-free methods).

Other approaches BID3 BID0 have resorted to rank-1 approximations of the Hessian for efficiency.

While these methods have had some success, they have only been demonstrated on single-layer models of moderate scale compared to the state-of-the-art in deep learning.

We speculate that the main reason they are not widely adopted is their requirement of several steps (network passes) per parameter update BID33 BID13 , which would put them at a similar disadvantage w.r.t.

first-order methods as the Hessian-free method that we tested ( FIG5 in the appendix).

Perhaps more closely related to our approach, Orr (1995) uses automatic differentiation to compute Hessian-vector products to construct adaptive, per-parameter learning rates.

The closest method is LiSSA BID0 , which is built around the idea of approximating the Hessian inverse with a Taylor series expansion.

This series can be implemented as the recursion H ???1 (r) = I + (I ??? H)H ???1 (r???1) , starting with H ???1 (0) = I. Since LiSSA is a type of Hessian-free method, the core of the algorithm is similar to Algorithm 2: it also refines an estimate of the Newton step iteratively, but with a different update rule in line 4.

With some simple algebraic manipulations, we can use the Taylor recursion to write this update in a form that is similar to ours: z r+1 = z r ?????z r ???J. This looks similar to our gradient-descent-based update with a learning rate of ?? = 1 (Alg.

1, lines 3-4), with some key differences.

First, they reset the state of the step estimate for every minibatch (Alg.

2, line 2).

Reusing past solutions, like momentum-SGD, is an important factor in the performance of our algorithm, since we only have to perform one update per mini-batch.

In contrast, BID0 report a typical number of inner-loop updates (R in Alg.

2) equal to the number of samples (e.g. R = 10, 000 for a tested subset of MNIST).

While this is not a problem for their tested case of linear Support Vector Machines, since each update only requires one inner-product, the same does not apply to deep neural networks.

Second, they invert the Hessian independently for each mini-batch, while our method aggregates the implicit Hessian across all past mini-batches (with a forgetting factor of ??).

Since batch sizes are orders of magnitude smaller than the number of parameters (e.g. 256 samples vs. 60 million parameters for the VGG-f), the Hessian matrix for a mini-batch is a poor substitute for the Hessian of the full dataset in these problems, and severely ill-conditioned.

Third, while their method fixes ?? to 1, we found that setting it correctly can be used to attain convergence on ill-conditioned problems.

For example, even on quadratic problems, we show that this parameter needs to be carefully chosen to avoid divergence (Theorem A.1).

The gradient descent interpretation used in our work, contrasted to the Taylor series recursion of BID0 , thus brings an additional degree of freedom that may be useful to relax other assumptions.

Finally, while they demonstrate improved performance on convex problems with linear models, we focus on the needs of training deep networks on large datasets (millions of samples and parameters), on which no previous Newton method has been able to surpass the first-order methods that are commonly used by the deep learning community.

In this work, we have proposed a practical second-order solver that has been specifically tailored for deep-learning-scale stochastic optimisation problems.

We showed that our optimiser can be applied to a large range of datasets and reach better training error than first order method with the same number of iterations, with essentially no hyper-parameters tuning.

In future work, we intend to bring more improvements to the wall-clock time of our method by engineering the FMAD operation to the same standard as back-propagation, and study optimal trust-region strategies to obtain ?? in closed-form.

We now perform a change of variables to diagonalize the Hessian, H = Qdiag(h)Q T , with Q orthogonal and h the vector of eigenvalues.

Let w * = arg min w f (w) = H ???1 b be the optimal solution of the minimization.

Then, replacing w t = Qx t + w * in eq. 30: DISPLAYFORM0 Then, expanding H with its eigendecomposition, DISPLAYFORM1 Left-multiplying by Q T ,and canceling out Q due to orthogonality, DISPLAYFORM2 Similarly for eq. 29, replacing z t = Qy t yields DISPLAYFORM3 Note that each pair formed by the corresponding element of y t and x t is an independent system with only 2 variables, since the pairs do not interact (eq. 33 and 34 only contain element-wise operations).From now on, we will be working on the ith element of each vector.

We can thus write eq. 33 and 34 (for a single element i of each) as a vector equation: DISPLAYFORM4 The matrix on the left is necessary to express the fact that the y t+1 factor in eq. 34 must be moved to the left-hand side, which corresponds to iteration t + 1 (x t+1 ??? y t+1 = x t ).

Left-multiplying eq. 35 by the inverse, DISPLAYFORM5 This is the transition matrix R i that characterizes the iteration, and taking its power models multiple iterations in closed-form: DISPLAYFORM6 The two eigenvalues of R i are given in closed-form by: DISPLAYFORM7 The series in eq. 37 converges when |eig (R i )| < 1 simultaneously for both eigenvalues, which is equivalent to: DISPLAYFORM8 with ?? > 0 and ??h i > 0.

Note that when using the Gauss-Newton approximation of the Hessian, h i > 0 and thus the last condition simplifies to ?? > 0.Since eq. 39 has to be satisfied for every eigenvalue, we have 3 2 ??h max ??? 1 < ?? < 1 + ??h min ,with h min and h max the smallest and largest eigenvalues of the Hessian H, respectively, proving the result.

The rate of convergence is the largest of the two values |eig (R i )|.

When the argument of the square root in eq. 38 is non-negative, it does not admit an easy interpretation; however, when it is negative, eq. 38 simplifies to: The convergence rate for a single eigenvalue is illustrated in FIG3 .

Graphically, the regions of convergence for different eigenvalues will differ only by a scale factor along the ??h i axis (horizontal stretching of FIG3 ).

Moreover, the largest possible range of ??h i values is obtained when ?? = 1, and that range is 0 < ??h i < 4 3 .

We can infer that the intersection of the regions of convergence for several eigenvalues will be maximized with ?? = 1, for any fixed ??.

DISPLAYFORM9

Theorem A.2.

Let the Hessian?? t+1 be positive definite (which holds when the objective is convex or when Gauss-Newton approximation and trust region are used).

Then the update z t+1 in Algorithm 1 is a descent direction when ?? and ?? are chosen according to eq. 18, and z t+1 = 0.Proof.

To show that the update represents a descent direction, it suffices to show that J T z t+1 < 0 (where we have written J = J(w t ) to simplify notation).

Since the surrogate Hessian?? t+1 is positive definite (PD) by construction, the update z t+1 = ??z t ??? ????? zt+1 satisfies z Figure 5 : Hyper-parameter evolution during training.

Average momentum ?? (left), learning rate ?? (middle), and trust region ?? (right), for each epoch for the basic CNN on CIFAR10, with and without batch normalisation (BN).

To make their scales comparable, we plot ?? divided by its initial value (which is ?? 0 = 1 with batch normalisation and ?? 0 = 10 without).

<|TLDR|>

@highlight

A fast second-order solver for deep learning that works on ImageNet-scale problems with no hyper-parameter tuning

@highlight

Choosing direction by using a single step of gradient descent "towards Newton step" from an original estimate, and then taking this direction instead of original gradient

@highlight

A new approximate second-order optimization method with low computational cost that replaces the computation of the Hessian matrix with a single gradient step and a warm start strategy.
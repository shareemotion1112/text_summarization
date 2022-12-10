We study two types of preconditioners and preconditioned stochastic gradient descent (SGD) methods in a unified framework.

We call the first one the Newton type due to its close relationship to the Newton method, and the second one the Fisher type as its preconditioner is closely related to the inverse of Fisher information matrix.

Both preconditioners can be derived from one framework, and efficiently estimated on any matrix Lie groups designated by the user using natural or relative gradient descent minimizing certain preconditioner estimation criteria.

Many existing preconditioners and methods, e.g., RMSProp, Adam, KFAC, equilibrated SGD, batch normalization, etc., are special cases of or closely related to either the Newton type or the Fisher type ones.

Experimental results on relatively large scale machine learning  problems are reported for performance study.

This paper investigates the use of preconditioner for accelerating gradient descent, especially in large scale machine learning problems.

Stochastic gradient descent (SGD) and its variations, e.g., momentum BID11 BID9 , RMSProp and Adagrad BID5 , Adam BID6 , etc., are popular choices due to their simplicity and wide applicability.

These simple methods do not use well normalized step size, could converge slow, and might involve more controlling parameters requiring fine tweaking.

Convex optimization is a well studied field BID2 .

Many off-the-shelf methods there, e.g., (nonlinear) conjugate gradient descent, quasi-Newton methods, Hessian-free optimizations, etc., can be applied to small and middle scale machine learning problems without much modifications.

However, these convex optimization methods may have difficulty in handling gradient noise and scaling up to problems with hundreds of millions of free parameters.

For a large family of machine learning problems, natural gradient with the Fisher information metric is equivalent to a preconditioned gradient using inverse of the Fisher information matrix as the preconditioner BID1 .

Natural gradient and its variations, e.g., Kronecker-factored approximate curvature (KFAC) BID8 and the one in BID10 , all use such preconditioners.

Other less popular choices are the equilibrated preconditioner BID4 and the one proposed in BID7 .

Momentum or the heavy-ball method provides another independent way to accelerate converge BID9 BID11 .

Furthermore, momentum and preconditioner can be combined to further accelerate convergence as shown in Adam BID6 .

This paper groups the above mentioned preconditioners and preconditioned SGD methods into two classes, the Newton type and the Fisher type.

The Newton type is closely related to the Newton method, and is suitable for general purpose optimizations.

The Fisher type preconditioner relates to the inverse of Fisher information matrix, and is limited to a large subclass of stochastic optimization problems where the Fish information metric can be well defined.

Both preconditioners can be derived from one framework, and estimated on any matrix Lie groups designated by the user with almost the same natural or relative gradient descent methods minimizing specific preconditioner estimation criteria.

We consider the minimization of cost function f (θ θ θ) = E z [ (θ θ θ, z z z)]where E z takes expectation over random variable z z z, is a loss function, and θ θ θ is the model parameter vector to be optimized.

For example, in a classification problem, could be the cross entropy loss, z z z is a pair of input feature vector and class label, vector θ θ θ consists of all the trainable parameters in the classification model, and E z takes average over all samples from the training data set.

By assuming second order differentiable model and loss, we could approximate (θ θ θ, z z z) as a quadratic function of θ θ θ within a trust region around θ θ θ, i.e., (θ θ θ, z z z) = b b b T z θ θ θ + 0.5θ θ θ T H H H z θ θ θ + a z , where a z is the sum of higher order approximation errors and constant term independent of θ θ θ, H H H z is a symmetric matrix, and subscript z in b b b z , H H H z and a z reminds us that these three terms depend on z z z. Clearly, these three terms depend on θ θ θ as well, although we do not explicitly show this dependence to simplify our notations since we just consider parameter updates in the same trust region.

Now, we may rewrite DISPLAYFORM0 We do not impose any assumption, e.g., positive definiteness, on H H H except for being symmetric.

Thus, the quadratic surface in the trust region could be non-convex.

To simplify our notations, we no longer consider the higher order approximation errors included in a, and simply assume that f (θ θ θ) is a quadratic function of θ θ θ in the trust region.

Let us consider a certain iteration.

Preconditioned SGD updates θ θ θ as DISPLAYFORM0 where µ > 0 is the step size,f (θ θ θ) is an estimate of f (θ θ θ) obtained by replacing expectation with sample average, and positive definite matrix P P P could be a fixed or adaptive preconditioner.

By letting θ θ θ = P P P −0.5 θ θ θ, we can rewrite (2) as DISPLAYFORM1 where P P P −0.5 denotes the principal square root of P P P .

Hence, (3) suggests that preconditioned SGD is equivalent to SGD in a transformed parameter domain.

Within the considered trust region, let us write the stochastic gradient, ∂f (θ θ θ)/∂θ θ θ, explicitly as DISPLAYFORM2 whereĤ H H andb b b are estimates of H H H and b b b, respectively.

Combining (4) and (2) gives the following linear system θ θ θ ← (I I I − µP P PĤ H H)θ θ θ − µP P Pb b b (5) for updating θ θ θ within the assumed trust region, where I I I is the identity matrix.

A properly determined P P P could significantly accelerate convergence of the locally linear system in (5).We review a few facts shown in BID7 before introducing our main contributions.

Let δθ θ θ be a random perturbation of θ θ θ, and be small enough such that θ θ θ + δθ θ θ still resides in the same trust region.

Then, (4) suggests the following resultant perturbation of stochastic gradient, DISPLAYFORM3 where ε ε ε accounts for the error due to replacingĤ H H with H H H. Note that by definition, δĝ g g is a random vector dependent on both z z z and δθ θ θ.

The preconditioner in BID7 is pursued by minimizing criterion c(P P P ) = E z,δθ [δĝ g g T P P P δĝ g g + δθ θ θ T P P P −1 δθ θ θ] (7) where subscript δθ in E z,δθ denotes taking expectation over δθ θ θ.

Under mild conditions, criterion (7) determines a unique positive definite P P P , which is optimal in the sense that it preconditions the stochastic gradient such that DISPLAYFORM4 which is comparable to relationship H H H −1 δg g gδg g g T H H H −1 = δθ θ θδθ θ θ T , where δg g g = H H Hδθ θ θ is the perturbation of noiseless gradient, and we assume that H H H is invertible, but not necessarily positive definite.

Clearly, this preconditioner is comparable to H H H −1 .

It perfectly preconditions the gradient such that the amplitudes of parameter perturbations match that of the associated preconditioned gradient, regardless of the amount of gradient noise.

Naturally, preconditioned SGD with this preconditioner inherits the scale-invariance property of the Newton method.

Note that in the presence of gradient noise, the optimal P P P and P P P −1 given by (8) are not unbiased estimates of H H H −1 and H H H, respectively.

Actually, even if H H H is positive definite and available, H H H −1 may not always be a good preconditioner when H H H is ill-conditioned since it could significantly amplify the gradient noise along the directions of the eigenvectors of H H H associated with small eigenvalues, and lead to divergence.

More specifically, it is shown in BID7 that Preconditioner estimation criterion (7) requires δθ θ θ to be small enough such that θ θ θ and θ θ θ + δθ θ θ reside in the same trust region.

In practice, numerical error might be an issue when handling small numbers with floating point arithmetic.

This concern becomes more grave with the popularity of single and even half precision math in large scale neural network training.

Luckily, (6) relates δĝ g g to the Hessianvector product, which can be efficiently evaluated with automatic differentiation software tools.

Let v v v be a random vector with the same dimension as θ θ θ.

Then, (4) suggests the following method for Hessian-vector product evaluation, DISPLAYFORM5 DISPLAYFORM6 Now, replacing (δθ θ θ, δĝ g g) in (7) with (v v v,Ĥ H Hv v v) leads to our following new preconditioner estimation criterion, c n ( DISPLAYFORM7 where the subscript v in E z,v suggests taking expectation over v v v. We no longer have the need to assume v v v to be an arbitrarily small vector.

It is important to note that this criterion only requires the Hessian-vector product.

The Hessian itself is not of interest.

We call (10) the Newton type preconditioner estimation criterion as the resultant preconditioned SGD method is closely related to the Newton method.

We consider the machine learning problems where the empirical Fisher information matrix can be well defined by DISPLAYFORM0 whereĝ g g = ∂f (θ θ θ)/∂θ θ θ is a shorthand for stochastic gradient, and λ ≥ 0 is a damping factor.

Clearly, v v v is independent ofĝ g g. Let us further assume that v v v is drawn from standard multivariate normal distribution N (0 0 0, DISPLAYFORM1 Then, we could simplify c f (P P P ) as DISPLAYFORM2 By letting the derivative of c f (P P P ) with respect to P P P be zero, the optimal positive definite solution for c f (P P P ) is readily shown to be DISPLAYFORM3 Whenĝ g g is a gradient estimation obtained by taking average over B independent samples, E z [ĝ g gĝ g g T ] is related to the Fisher information matrix by DISPLAYFORM4 We call this preconditioner the Fisher type one due to its close relationship to the Fisher information matrix.

One can easily modify this preconditioner to obtain an unbiased estimation of F F F −1 .

Let s s s be an exponential moving average ofĝ g g. Then, after replacing theĝ g g in (13) withĝ g g − s s s + s s s/ √ B and setting λ = 0, P P P 2 /B will be an unbiased estimation of F F F −1 .

Generally, it might be acceptable to keep the bias term, (B − 1)g g gg g g T /B, in (14) for two reasons: it is nonnegative definite and regularizes the inversion in FORMULA0 ; it vanishes when the parameters approach a stationary point.

Actually, the Fisher information matrix could be singular for many commonly used models, e.g., finite mixture models, neural networks, hidden Markov models.

We might not be able to inverse F F F for these singular statistical models without using regularization or damping.

A Fisher type preconditioner with λ > 0 loses the scale-invariance property of a Newton type preconditioner.

Both P P P and P P P 2 can be useful preconditioners when the step size µ and damping factor λ are set properly.

Following the ideas in BID7 , we can show that (10) determines a unique positive definite preconditioner if and only if DISPLAYFORM0 has distinct eigenvalues.

Other minimum solutions of criterion FORMULA0 are either indefinite or negative definite, and are not interested for our purpose.

The proof itself has limited novelty.

We omit it here.

Instead, let us consider the simplest case, where θ θ θ is a scalar parameter, to gain some intuitive understandings of criterion (10).

For scalar parameter, it is trivial to show that the optimal solutions minimizing (10) are DISPLAYFORM1 whereĤ H H, H H H, P P P , and v v v are replaced with their plain lower case letters, and we have used the fact that H H H −Ĥ H H and v v v are independent.

For gradient descent, we choose the positive solution, although the negative one gives the global minimum of (10).

With the positive preconditioner, eigenvalue of the locally linear system in (5) is DISPLAYFORM2 Now, it is clear that this optimal preconditioner damps the gradient noise when E z [(h−ĥ) 2 ] is large, and preconditions the locally linear system in (5) such that its eigenvalue has unitary amplitude when the gradient noise vanishes.

Convergence is ensured when a normalized step size, i.e., 0 < µ < 1, is used.

For θ θ θ with higher dimensions, eigenvalues of the locally linear system in (5) is normalized into range [−1, 1] as well, in a way similar to (16).

Let us take the Newton type preconditioner as an example to derive its updating rule.

Updating rule for the Fisher type preconditioner is the same except for replacing the Hessian-vector product with stochastic gradient.

Here, Lie group always refers to the matrix Lie group.

It is inconvenient to optimize P P P directly as it must be a symmetric and positive definite matrix.

Instead, we represent the preconditioner as P P P = Q Q Q T Q Q Q, and estimate Q Q Q. Now, Q Q Q must be a nonsingular matrix as both c n (P P P ) and c f (P P P ) diverge when P P P is singular.

Invertible matrices with the same dimension form a Lie group.

In practice, we are more interested in Lie groups with sparse representations.

Examples of such groups are given in the next section.

Let us consider a proper small perturbation of Q Q Q, δQ Q Q, such that Q Q Q + δQ Q Q still lives on the same Lie group.

The distance between Q Q Q and Q Q Q + δQ Q Q can be naturally defined as dist(Q Q Q, Q Q Q + δQ Q Q) = tr(δQ Q QQ Q Q −1 Q Q Q −T δQ Q Q T ) BID1 .

Intuitively, this distance is larger for the same amount of perturbation when Q Q Q is closer to a singular matrix.

With the above tensor metric definition, natural gradient for optimizing Q Q Q has form DISPLAYFORM0 For example, when Q Q Q lives on the group of invertible upper triangular matrices, R R R is given by DISPLAYFORM1 where triu takes the upper triangular part of a matrix.

Another way to derive (17) is to let δQ Q Q = EQ Q Q, and consider the derivative with respect to E, where E is a proper small matrix such that Q Q Q + EQ Q Q still lives on the same Lie group.

Gradient derived in this way is known as relative gradient BID3 .

For our preconditioner learning problem, relative gradient and natural gradient have the same form.

Now, Q Q Q can be updated using natural or relative gradient descent as DISPLAYFORM2 In practice, it is convenient to use the following updating rule with normalized step size, DISPLAYFORM3 where 0 < µ 0 < 1, and · takes the norm of a matrix.

One simple choice for matrix norm is the maximum absolute value of a matrix.

Note that natural gradient can take different forms.

One should not confuse the natural gradient on the Lie group derived from a tensor metric with the natural gradient for model parameter learning derived from a Fisher information metric.

One iteration of the Newton type preconditioned SGD consists of the following steps.1.

Evaluate stochastic gradientĝ g g.

The two step sizes, µ and µ 0 , are normalized.

They should take values in range [0, 1] with typical value 0.01.

We usually initialize Q Q Q to a scaled identity matrix with proper dimension.

The specific form ofR R R depends on the Lie group to be considered.

For example, for upper triangular Q Q Q, we havê DISPLAYFORM0 , where Q Q Q −T v v v can be efficiently calculated with back substitution.

We only need to replaceĤ H Hv v v in the Newton type preconditioned SGD withĝ g g+λv v v to obtain the Fisher type one.

Thus, we do not list its main steps here.

Note that only its step size for the preconditioner updating is normalized.

There is no simple way to jointly determine the proper ranges for step size µ and damping factor λ.

Again,R R R may take different forms on different Lie groups.

For upper triangular Q Q Q, we haveR DISPLAYFORM0 , where v v v ∼ N (0 0 0, I I I).

Here, it is important to note that the natural or relative gradient for c f (P P P ) with the form given in (12) involves explicit matrix inversion.

However, matrix inversion can be avoided by using the c f (P P P ) in (11), which includes v v v as an auxiliary variable.

It is highly recommended to avoid explicit matrix inversion for large Q Q Q.

There are many ways to modify the above preconditioned SGD methods.

Since curvatures typically evolves slower than gradients, one can update the preconditioner less frequently to save computations per iteration.

With parallel computing available, one might update the preconditioner and model parameters simultaneously and asynchronously to save wall time per iteration.

Combining preconditioner and momentum may further accelerate convergence.

For recurrent neural network learning, we may need to clip the norm of preconditioned gradients to avoid excessively large parameter updates.

In general, preconditioned gradient clipping relates to the trust region method by θ θ θ[new] − θ θ θ = µP P Pĝ g g/max(1, P P Pĝ g g /Ω) ≤ µΩwhere Ω > 0 is a clipping threshold, comparable to the size of trust region.

One may set Ω to a positive number proportional to the square root of the number of model parameters.

Most importantly, we can choose different Lie groups for estimating our preconditioners to achieve a good trade off between performance and complexity.

In practice, we seldom consider the Lie group consisting of dense invertible matrices for preconditioner estimation when the problem size is large.

Lie groups with sparse structures are of more interests.

To begin with, let us recall a few facts about Lie group.

If A A A and B B B are two Lie groups, then A A A T , A A A ⊗B B B, and A A A ⊕B B B all are Lie groups, where ⊗ and ⊕ denote Kronecker product and direct sum, respectively.

Furthermore, for any matrix C C C with compatible dimensions, block matrix DISPLAYFORM0 still forms a Lie group.

We do not show proofs of the above statements here as they are no more than a few lines of algebraic operations.

These simple rules can be used to design many useful Lie groups for constructing sparse preconditioners.

We already know that invertible upper triangular matrices form a Lie group.

Here, we list a few useful ones with sparse representations.

Diagonal matrices with the same dimension and positive diagonal entries form a Lie group with reducible representation.

Preconditioners learned on this group are called diagonal preconditioners.

For matrix parameter Θ Θ Θ, we can flatten Θ Θ Θ into a vector, and precondition its gradient using a Kronecker product preconditioner with Q Q Q having form Q Q Q = Q Q Q 2 ⊗ Q Q Q 1 .

Clearly, Q Q Q is a Lie group as long as Q Q Q 1 and Q Q Q 2 are two Lie groups.

Let us check its role in learning the following affine transformation DISPLAYFORM0 where x x x is the input feature vector augmented with 1, and y y y is the output feature vector.

After reverting the flattened Θ Θ Θ back to its matrix form, the preconditioned SGD learning rule for Θ Θ Θ is DISPLAYFORM1 Similar to (3), we introduce coordinate transformation DISPLAYFORM2 2 , and rewrite (23) as DISPLAYFORM3 Correspondingly, the affine transformation in FORMULA2 is rewritten as y y y = Θ Θ Θ x x x , where y y y = Q Q Q −T 1 y y y and x x x = Q Q Q 2 x x x are the transformed input and output feature vectors, respectively.

Hence, the preconditioned SGD in (23) is equivalent to the SGD in (24) with transformed feature vectors x x x and y y y .

We know that feature whitening and normalization could significantly accelerate convergence.

A Kronecker product preconditioner plays a similar role in learning the affine transformation in (22).

This is a special Kronecker product preconditioner by constraining Q Q Q 1 to be a diagonal matrix, and Q Q Q 2 to be a sparse matrix where only its diagonal and last column can have nonzero values.

Note that Q Q Q 2 with nonzero diagonal entries forms a Lie group.

Hence, Q Q Q = Q Q Q 2 ⊗ Q Q Q 1 is a Lie group as well.

We call it a scaling and normalization preconditioner as it resembles a preconditioner that scales the output features and normalizes the input features.

Let us check the transformed features y y y = Q Q Q −T 1 y y y and x x x = Q Q Q 2 x x x. It is clear that y y y is an element-wisely scaled version of y y y as Q Q Q 1 is a diagonal matrix.

To make x x x a "normalized" feature vector, x x x needs to be an input feature vector augmented with 1.Let us check a simple example to verify this point.

We consider an input vector with two features, and write down its normalized features explicitly as below, DISPLAYFORM0 where m i and σ i are the mean and standard deviation of x i , respectively.

It is straightforward to show that the feature normalization operation in (25) forms a Lie group with four freedoms.

For the scaling-and-normalization preconditioner, we have no need to force the last diagonal entry of Q Q Q 2 to be 1.

Hence, the group of feature normalization operation is a subgroup of Q Q Q 2 .

This is another special Kronecker product preconditioner by constraining Q Q Q 1 to be a diagonal matrix, and Q Q Q 2 to be an upper triangular matrix with positive diagonal entries.

We call it a scaling-andwhitening preconditioner since it resembles a preconditioner that scales the output features and whitens the input features.

Again, the input feature vector x x x must be augmented with 1 such that the whitening operation forms a Lie group represented by upper triangular matrices with 1 being its last diagonal entry.

This is a subgroup of Q Q Q 2 as we have no need to fix Q Q Q 2 's last diagonal entry to 1.It is not possible to enumerate all kinds of Lie groups suitable for constructing preconditioners.

For example, Kronecker product preconditioner with form Q Q Q = Q Q Q 3 ⊗ Q Q Q 2 ⊗ Q Q Q 1 could be used for preconditioning gradients of a third order tensor.

The normalization and whitening groups are just two special cases of the groups with the form shown in FORMULA0 , and there are numerous more choices having sparsities between that of these two.

Regardless of the detailed form of Q Q Q, all such preconditioners share the same form of learning rule shown in FORMULA2 , and they all can be efficiently learned using natural or relative gradient descent without much tuning effort.

Adagrad, RMSProp and Adam all use the Fisher type preconditioner living on the group of diagonal matrices with positive diagonal entries.

This is a simple group.

Optimal solution for c f (P P P ) has closed-form solution P P P = diag(1 E z [ĝ g g ĝ g g] + λ 2 ), where and denote element wise multiplication and division, respectively.

In practice, simple exponential moving average is used to replace the expectation when using this preconditioner.

For diagonal preconditioner, the optimal solution minimizing c n (P P P ) has closed-form solution DISPLAYFORM0 reduces to a vector with unit entries.

Then, this optimal solution gives the equilibration preconditioner in BID4 .

The preconditioners considered in BID10 and BID8 are closely related to the Fisher type Kronecker product preconditioners.

While KFAC approximates the Fisher metric of a matrix parameter as a Kronecker product to obtain its approximated inverse in closedform solution, our method turns to an iterative solution to approximate this same inverse.

Theoretically, our method's accuracy is only limited by the expressive power of the Lie group since no intermediate approximation is made.

In practice, one distinct advantage of our method over KFAC is that explicit matrix inversion is avoided by introducing auxiliary vector v v v and using back substitution, while KFAC typically requires inversion of symmetric matrices.

On graphics processing units (GPU) and with large matrices, parallel back substitution is as computationally cheap as matrix multiplication, and could be several orders of magnitude faster than inversion of symmetric matrix.

Another advantage is that our method is derived from a unified framework.

There is no need to invent different preconditioner learning rules when we switch the Lie group representations.

Batch normalization can be viewed as preconditioned SGD using a specific scaling-andnormalization preconditioner with constraint Q Q Q 1 = I I I and Q Q Q 2 from the feature normalization Lie group.

However, we should be aware that explicit input feature normalization is only empirically shown to accelerate convergence, and has little meaning in certain scenarios, e.g., recurrent neural network learning where features may not have any stationary first or second order statistic.

Both the Newton and Fisher type preconditioned SGD methods provide a more general and principled approach to find the optimal preconditioner, and apply to a broader range of applications.

Generally, a scaling-and-normalization preconditioner does not necessarily "normalize" the input features in the sense of mean removal and variance normalization.

We use the square root Fisher type preconditioners in the following experiments since they are less picky on the damping factor, and seem to be more numerically robust on large scale problems.

Still, as shown in our Pytorch implementation package, the original Fisher type preconditioners could perform better on small scale problems like the MNIST handwritten digit recognition task.

Let us consider the minimization of Rosenbrock function, f (θ θ θ) = 100(θ 2 − θ 2 1 ) 2 + (1 − θ 1 ) 2 , starting from initial guess θ θ θ = [−1, 1].

This is a well known benchmark problem for mathematical optimization.

The compared methods use fixed step size.

For each method, the best step size is selected from sequence {. . .

, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, . . .}.

For gradient descent, the best step size is 0.002.

For momentum method, the moving average factor is 0.9, and the best step size is 0.002.

For Nesterov momentum, the best step size is 0.001.

For preconditioned SGD, Q Q Q is initialized to 0.1I I I and lives on the group of triangular matrices.

For the Fisher type method, we set λ = 0.1, and step sizes 0.01 and 0.001 for preconditioner and parameter updates, respectively.

For the Newton type method, we set step sizes 0.2 and 0.5 for preconditioner and parameter updates, respectively.

FIG1 summarizes the results.

The Newton type method performs the best, converging to the optimal solution using about 200 iterations.

The Fisher type method does not fit into this problem, and performs poorly as expected.

Mathematical optimization is not our focus.

Still, this example shows that the Newton type preconditioned SGD works well for mathematical optimization.

We consider the ImageNet ILSVRC2012 database for the image classification task.

The well known AlexNet is considered.

We follow the descriptions in BID0 as closely as possible to set up our experiment.

One main difference is that we do not augment the training data.

Another big difference is that we use a modified local response normalization (LRN).

The LRN function from TensorFlow implementation is not second order differentiable.

We have to approximate the local energy used for LRN with a properly scaled global energy to facilitate Hessian-vector product evaluation.

Note that convolution can be rewritten as correlation between the flattened input image patches and filter coefficients.

In this way, we find that there are eight matrices to be optimized in the AlexNet, and their shapes are: each matrix.

Denser preconditioners, e.g., the Kronecker product one, require hundreds of millions parameters for representations, and are expensive to run on our platform.

Each compared method is trained with 40 epochs, mini-batch size 128, step size µ for the first 20 epochs, and 0.1µ for the last 20 epochs.

We have compared several methods with multiple settings, and only report the ones with reasonably good results here.

For Adam, the initial step size is set to 0.00005.

For batch normalization, initial step size is 0.002, and its moving average factors for momentum and statistics used for feature normalization are 0.9 and 0.99, respectively.

The momentum method uses initial step size 0.002, and moving average factor 0.9 for momentum.

Preconditioned SGD performs better with the scaling-and-normalization preconditioner.

Its Q Q Q is initialized to 0.1I I I, and updated with normalized step size 0.01.

For the Fisher type preconditioner, we set λ = 0.001 and initial step size 0.00005.

For the Newton type preconditioner, its initial step size is 0.01.

FIG0 summarizes the results.

Training loss for batch normalization is only for reference purpose as normalization alters the L2-regularization term.

Batch normalization does not perform well under this setup, maybe due to its conflict with certain settings like the LRN and L2-regularization.

We see that the scalingand-normalization preconditioner does accelerate convergence, although it is super sparse.

The Newton type preconditioned SGD performs the best, and achieves top-1 validation accuracy about 56% when using only one crop for testing, while the momentum method may require 90 epochs to achieve similar performance.

We consider the world level language modeling problem with reference implementation available from https://github.com/pytorch/examples.

The Wikitext-2 database with 33278 tokens is considered.

The task is to predict the next token from history observations.

Our tested network consists of six layers, i.e., encoding layer, LSTM layer, dropout layer, LSTM layer, dropout layer, and decoding layer.

For each LSTM layer, we put all its coefficients into a single matrix Θ Θ Θ by defining output and augmented input feature vectors as in DISPLAYFORM0 , where t is a discrete time index, x x x is the input, h h h is the hidden state, and c c c is the cell state.

The encoding layer's weight matrix is the transpose of that of the decoding layer.

Thus, we totally get three matrices to be optimized.

With hidden layer size 200, shapes of these three matrices are respectively.

For all methods, the step size is reduced to one fourth of the current value whenever the current perplexity on validation set is larger than the best one ever found.

For SGD, the initial step size is 20, and the gradient is clipped with threshold 0.25.

The momentum method diverges easily without clipping.

We set momentum 0.9, initial step size 1, and clipping threshold 0.25.

We set initial step size 0.005 and damping factor λ 2 = 10 −12 for Adam and sparse Adam.

Sparse Adam updates its moments and model parameters only when the corresponding stochastic gradients are not zeros.

We have tried diagonal, scaling-and-normalization and scaling-and-whitening preconditioners for each matrix.

The encoding (decoding) matrix is too large to consider KFAC like preconditioner.

The diagonal preconditioner performs the worst, and the other two have comparable performance.

For both types of preconditioned SGD, the clipping threshold for preconditioned gradient is 100, the initial step size is 0.1, and Q Q Q is initialized to I I I.

We set λ = 0 for the Fisher Figure 3 summarizes the results when the dropout rate is 0.35.

Methods involving momentum, including Adam and sparse Adam, perform poorly.

Note that our preconditioners preserve the sparsity property of gradients from the encoding and decoding layers (Appendix A).

This saves considerable computations by avoiding update parameters with zero gradients.

Again, both preconditioners accelerate convergence significantly despite their high sparsity.

Compared with SGD, the Fisher type preconditioned SGD adds limited computational complexity when sparse preconditioners are adopted.

The Newton type preconditioned SGD requires Hessianvector product, which typically has complexity comparable to that of gradient evaluation.

Thus, using SGD as the base line, the Newton type preconditioned SGD approximately doubles the computational complexity per iteration, while the Fisher type SGD has similar complexity.

Wall time per iteration of preconditioned SGD highly depends on the implementations.

Ideally, the preconditioners and parameters could be updated in a parallel and asynchronous way such that SGD and preconditioned SGD have comparable wall time per iteration.

We have put our TensorFlow and Pytorch implementations on https://github.com/ lixilinx.

More experimental results comparing different preconditioners and optimization methods on diverse benchmark problems can be found there.

For the ImageNet experiment, all compared methods are implemented in Tensorflow, and require two days and a few hours to finish 40 epochs on a GeForce GTX 1080 Ti GPU.

The word level language modeling experiment is implemented in Pytorch.

We have rewritten the word embedding function to enable second order derivative.

For this task, SGD and the Fisher type preconditioned SGD have similar wall time per iteration, while the Newton type method requires about 80% more wall time per iteration than SGD when running on the same GPU.

Two types of preconditioners and preconditioned SGD methods are studied.

The one requiring Hessian-vector product for preconditioner estimation is suitable for general purpose optimization.

We call it the Newton type preconditioned SGD due to its close relationship to the Newton method.

The other one only requires gradient for preconditioner estimation.

We call it the Fisher type preconditioned SGD as its preconditioner is closely related to the inverse of Fisher information matrix.

Both preconditioners can be efficiently learned using natural or relative gradient descent on any matrix Lie groups designated by the user.

The Fisher type preconditioned SGD has lower computational complexity per iteration, but may require more tuning efforts on selecting its step size and damping factor.

The Newton type preconditioned SGD has higher computational complexity per iteration, but is more user friendly due to its use of normalized step size and built-in gradient noise damping ability.

Both preconditioners, even with very sparse representations, are shown to considerably accelerate convergence on relatively large scale problems.

@highlight

We propose a new framework for preconditioner learning, derive new forms of preconditioners and learning methods, and reveal the relationship to methods like RMSProp, Adam, Adagrad, ESGD, KFAC, batch normalization, etc.
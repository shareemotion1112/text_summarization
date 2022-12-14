We present Spectral Inference Networks, a framework for learning eigenfunctions of linear operators by stochastic optimization.

Spectral Inference Networks generalize Slow Feature Analysis to generic symmetric operators, and are closely related to Variational Monte Carlo methods from computational physics.

As such, they can be a powerful tool for unsupervised representation learning from video or graph-structured data.

We cast training Spectral Inference Networks as a bilevel optimization problem, which allows for online learning of multiple eigenfunctions.

We show results of training Spectral Inference Networks on problems in quantum mechanics and feature learning for videos on synthetic datasets.

Our results demonstrate that Spectral Inference Networks accurately recover eigenfunctions of linear operators and can discover interpretable representations from video in a fully unsupervised manner.

Spectral algorithms are central to machine learning and scientific computing.

In machine learning, eigendecomposition and singular value decomposition are foundational tools, used for PCA as well as a wide variety of other models.

In scientific applications, solving for the eigenfunction of a given linear operator is central to the study of PDEs, and gives the time-independent behavior of classical and quantum systems.

For systems where the linear operator of interest can be represented as a reasonably-sized matrix, full eigendecomposition can be achieved in O(n 3 ) time BID11 , and in cases where the matrix is too large to diagonalize completely (or even store in memory), iterative algorithms based on Krylov subspace methods can efficiently compute a fixed number of eigenvectors by repeated application of matrix-vector products (Golub & Van Loan, 2012) .At a larger scale, the eigenvectors themselves cannot be represented explicitly in memory.

This is the case in many applications in quantum physics and machine learning, where the state space of interest may be combinatorially large or even continuous and high dimensional.

Typically, the eigenfunctions of interest are approximated from a fixed number of points small enough to be stored in memory, and then the value of the eigenfunction at other points is approximated by use of the Nystr??m method (Bengio et al., 2004) .

As this depends on evaluating a kernel between a new point and every point in the training set, this is not practical for large datasets, and some form of function approximation is necessary.

By choosing a function approximator known to work well in a certain domain, such as convolutional neural networks for vision, we may be able to bias the learned representation towards reasonable solutions in a way that is difficult to encode by choice of kernel.

In this paper, we propose a way to approximate eigenfunctions of linear operators on highdimensional function spaces with neural networks, which we call Spectral Inference Networks (SpIN).

We show how to train these networks via bilevel stochastic optimization.

Our method finds correct eigenfunctions of problems in quantum physics and discovers interpretable representations from video.

This significantly extends prior work on unsupervised learning without a generative model and we expect will be useful in scaling many applications of spectral methods.

The outline of the paper is as follows.

Sec 2 provides a review of related work on spectral learning and stochastic optimization of approximate eigenfunctions.

Sec. 3 defines the objective function for Spectral Inference Networks, framing eigenfunction problems as an optimization problem.

Sec. 4 describes the algorithm for training Spectral Inference Networks using bilevel optimization and a custom gradient to learn ordered eigenfunctions simultaneously.

Experiments are presented in Sec. 5 and future directions are discussed in Sec. 6.

We also include supplementary materials with more in-depth derivation of the custom gradient updates (Sec. A), a TensorFlow implementation of the core algorithm (Sec. B), and additional experimental results and training details (Sec. C).

Spectral methods are mathematically ubiquitous, arising in a number of diverse settings.

Spectral clustering BID9 , normalized cuts BID14 and Laplacian eigenmaps (Belkin & Niyogi, 2002) are all machine learning applications of spectral decompositions applied to graph Laplacians.

Related manifold learning algorithms like LLE BID19 and IsoMap BID13 also rely on eigendecomposition, with a different kernel.

Spectral algorithms can also be used for asymptotically exact estimation of parametric models like hidden Markov models and latent Dirichlet allocation by computing the SVD of moment statistics BID0 BID0 .In the context of reinforcement learning, spectral decomposition of predictive state representations has been proposed as a method for learning a coordinate system of environments for planning and control (Boots et al., 2011) , and when the transition function is symmetric its eigenfunctions are also known as proto-value functions (PVFs) BID7 .

PVFs have also been proposed by neuroscientists as a model for the emergence of grid cells in the entorhinal cortex BID17 .

The use of PVFs for discovering subgoals in reinforcement learning has been investigated in BID5 and combined with function approximation in BID6 , though using a less rigorous approach to eigenfunction approximation than SpIN.

A qualitative comparison of the two approaches is given in the supplementary material in Sec. C.3.Spectral learning with stochastic approximation has a long history as well.

Probably the earliest work on stochastic PCA is that of "Oja's rule" BID10 , which is a Hebbian learning rule that converges to the first principal component, and a wide variety of online SVD algorithms have appeared since.

Most of these stochastic spectral algorithms are concerned with learning fixed-size eigenvectors from online data, while we are concerned with cases where the eigenfunctions are over a space too large to be represented efficiently with a fixed-size vector.

The closest related work in machine learning on finding eigenfunctions by optimization of parametric models is Slow Feature Analysis (SFA) (Wiskott & Sejnowski, 2002) , which is a special case of SpIN.

SFA is equivalent to function approximation for Laplacian eigenmaps BID16 , and it has been shown that optimizing for the slowness of features in navigation can also lead to the emergence of units whose response properties mimic grid cells in the entorhinal cortex of rodents (Wyss et al., 2006; Franzius et al., 2007) .

SFA has primarily been applied to train shallow or linear models, and when trained on deep models is typically trained in a layer-wise fashion, rather than end-to-end BID3 BID18 .

The features in SFA are learned sequentially, from slowest to fastest, while SpIN allows for simultaneous learning of all eigenfunctions, which is more useful in an online setting.

Spectral methods and deep learning have been combined in other ways.

The spectral networks of Bruna et al. (2014) are a generalization of convolutional neural networks to graph and manifold structured data based on the idea that the convolution operator is diagonal in a basis defined by eigenvectors of the Laplacian.

In BID2 spectral decompositions were incorporated as differentiable layers in deep network architectures.

Spectral decompositions have been used in combination with the kernelized Stein gradient estimator to better learn implicit generative models like GANs BID15 .

While these use spectral methods to design or train neural networks, our work uses neural networks to solve large-scale spectral decompositions.

In computational physics, the field of approximating eigenfunctions of a Hamiltonian operator is known as Variational Quantum Monte Carlo (VMC) (Foulkes et al., 2001) .

VMC methods are usually applied to finding the ground state (lowest eigenvalue) of electronic systems, but extensions to excited states (higher eigenvalues) have been proposed (Blunt et al., 2015) .

Typically the class of function approximator is tailored to the system, but neural networks have been used for calculating ground states (Carleo & Troyer, 2017) and excited states (Choo et al., 2018) .

Stochastic optimization for VMC dates back at least to Harju et al. (1997) .

Most of these methods use importance sampling from a well-chosen distribution to eliminate the bias due to finite batch sizes.

In machine learning we are not free to choose the distribution from which the data is sampled, and thus cannot take advantage of these techniques.

Eigenvectors of a matrix A are defined as those vectors u such that Au = ??u for some scalar ??, the eigenvalue.

It is also possible to define eigenvectors as the solution to an optimization problem.

If A is a symmetric matrix, then the largest eigenvector of A is the solution of: DISPLAYFORM0 or equivalently (up to a scaling factor in u) DISPLAYFORM1 This is the Rayleigh quotient, and it can be seen by setting derivatives equal to zero that this is equivalent to finding u such that Au = ??u, where ?? is equal to the value of the Rayleigh quotient.

We can equivalently find the lowest eigenvector of A by minimizing the Rayleigh quotient instead.

Amazingly, despite being a nonconvex problem, algorithms such as power iteration converge to the global solution of this problem (Daskalakis et al., 2018, Sec. 4) .To compute the top N eigenvectors U = (u 1 , . . . , u N ), we can solve a sequence of maximization problems: DISPLAYFORM2 If we only care about finding a subspace that spans the top N eigenvectors, we can divide out the requirement that the eigenvectors are orthogonal to one another, and reframe the problem as a single optimization problem (Edelman et al., 1998, Sec. 4.4) : DISPLAYFORM3 or, if u i denotes row i of U: DISPLAYFORM4 Note that this objective is invariant to right-multiplication of U by an arbitrary matrix, and thus we do not expect the columns of U to be the separate eigenvectors.

We will discuss how to break this symmetry in Sec. 4.1.

We are interested in the case where both A and u are too large to represent in memory.

Suppose that instead of a matrix A we have a symmetric (not necessarily positive definite) kernel k(x, x ) where x and x are in some measurable space ???, which could be either continuous or discrete.

Let the inner product on ??? be defined with respect to a probability distribution with density p(x), DISPLAYFORM0 ].

In theory this could be an improper density, such as the uniform distribution over R n , but to evaluate it numerically there must be some proper distribution over ??? from which the data are sampled.

We can construct a symmetric linear DISPLAYFORM1 To compute a function that spans the top N eigenfunctions of this linear operator, we need to solve the equivalent of Eq. 5 for function spaces.

Replacing rows i and j with points x and x and sums with expectations, this becomes: DISPLAYFORM2 where the optimization is over all functions u : ??? ??? R N such that each element of u is an integrable function under the metric above.

Also note that as u i is a row vector while u(x) is a column vector, the transposes are switched.

This is equivalent to solving the constrained optimization problem DISPLAYFORM3 For clarity, we will use ?? = E x u(x)u(x)T to denote the covariance 1 of features and DISPLAYFORM4 T to denote the kernel-weighted covariance throughout the paper, so the objective in Eq. 6 becomes Tr(?? ???1 ??).

The empirical estimate of these quantities will be denoted as?? and??.

The form of the kernel k often allows for simplification to Eq. 6.

If ??? is a graph, and k(x, x ) = ???1 if x = x and are neighbors and 0 otherwise, and k(x, x) is equal to the total number of neighbors of x, this is the graph Laplacian, and can equivalently be written as: DISPLAYFORM0 for neighboring points (Sprekeler, 2011, Sec. 4 .1).

It's clear that this kernel penalizes the difference between neighbors, and in the case where the neighbors are adjacent video frames this is Slow Feature Analysis (SFA) (Wiskott & Sejnowski, 2002) .

Thus SFA is a special case of SpIN, and the algorithm for learning in SpIN here allows for end-to-end online learning of SFA with arbitrary function approximators.

The equivalent kernel to the graph Laplacian for ??? = R n is DISPLAYFORM1 where e i is the unit vector along the axis i.

This converges to the differential Laplacian, and the linear operator induced by this kernel is DISPLAYFORM2 , which appears frequently in physics applications.

The generalization to generic manifolds is the Laplace-Beltrami operator.

Since these are purely local operators, we can replace the double expectation over x and x with a single expectation.

There are many possible ways of solving the optimization problems in Equations 6 and 7.

In principle, we could use a constrained optimization approach such as the augmented Lagrangian method (Bertsekas, 2014), which has been successfully combined with deep learning for approximating maximum entropy distributions BID4 .

In our experience, such an approach was difficult to stabilize.

We could also construct an orthonormal function basis and then learn some flow that preserves orthonormality.

This approach has been suggested for quantum mechanics problems by Cranmer et al. (2018) .

But, if the distribution p(x) is unknown, then the inner product f, g is not known, and constructing an explicitly orthonormal function basis is not possible.

Also, flows can only be defined on continuous spaces, and we are interested in methods that work for large discrete spaces as well.

Instead, we take the approach of directly optimizing the quotient in Eq. 6.

Since Eq. 6 is invariant to linear transformation of the features u(x), optimizing it will only give a function that spans the top N eigenfunctions of K. If we were to instead sequentially optimize the Rayleigh quotient for each function u i (x): DISPLAYFORM0 we would recover the eigenfunctions in order.

However, this would be cumbersome in an online setting.

It turns out that by masking the flow of information from the gradient of Eq. 6 correctly, we can simultaneously learn all eigenfunctions in order.

First, we can use the invariance of trace to cyclic permutation to rewrite the objective in Eq. 6 as DISPLAYFORM1 , this matrix has the convenient property that the upper left n ?? n block only depends on the first n functions u 1: DISPLAYFORM2 T .

This means the maximum of DISPLAYFORM3 ?? ii with respect to u 1:n (x) spans the first n < N eigenfunctions.

If we additionally mask the gradients of ?? ii so they are also independent of any u j (x) where j is less than i: DISPLAYFORM4 and combine the gradients for each i into a single masked gradient??? DISPLAYFORM5 ) which we use for gradient ascent, then this is equivalent to independently optimizing each u i (x) towards the objective ?? ii .

Note that there is still nothing forcing all u(x) to be orthogonal.

If we explicitly orthogonalize u(x) by multiplication by L ???1 , then we claim that the resulting v(x) = L ???1 u(x) will be the true ordered eigenfunctions of K. A longer discussion justifying this is given in the supplementary material in Sec. A. The closed form expression for the masked gradient, also derived in the supplementary material, is given by: DISPLAYFORM6 where triu and diag give the upper triangular and diagonal of a matrix, respectively.

This gradient can then be passed as the error from u back to parameters ??, yielding: DISPLAYFORM7 To simplify notation we can express the above as DISPLAYFORM8 Where DISPLAYFORM9 ????? are linear operators that denote left-multiplication of the Jacobian of ?? and ?? with respect to ?? by A. A TensorFlow implementation of this gradient is given in the supplementary material in Sec. B.

The expression in Eq. 14 is a nonlinear function of multiple expectations, so naively replacing ??, ??, L, ?? and their gradients with empirical estimates will be biased.

This makes learning Spectral Inference Networks more difficult than standard problems in machine learning for which unbiased gradient estimates are available.

We can however reframe this as a bilevel optimization problem, for which convergent algorithms exist.

Bilevel stochastic optimization is the problem of simultaneously solving two coupled minimization problems min x f (x, y) and min y g(x, y) for which we only have Algorithm 1 Learning in Spectral Inference Networks 1: given symmetric kernel k, decay rates ?? t , first order optimizer OPTIM 2: initialize parameters ?? 0 , average covariance?? 0 = I, average Jacobian of covarianceJ ??0 = 0 3: while not converged do

Get minibatches x t1 , . . .

, x tN and x t1 , . . . , x tN 5: DISPLAYFORM0 T , covariance of minibatches DISPLAYFORM1 Compute gradient??? ?? Tr(??(?? t ,?? t ,?? ??t ,J ??t )) according to Eq. 14 11: DISPLAYFORM2 noisy unbiased estimates of the gradient of each: DISPLAYFORM3 .

Bilevel stochastic problems are common in machine learning and include actor-critic methods, generative adversarial networks and imitation learning BID12 .

It has been shown that by optimizing the coupled functions on two timescales then the optimization will converge to simultaneous local minima of f with respect to x and g with respect to y (Borkar, 1997): DISPLAYFORM4 where DISPLAYFORM5 By replacing ?? and J ?? with a moving average in Eq. 14, we can cast learning Spectral Inference Networks as exactly this kind of bilevel problem.

Throughout the remainder of the paper, letX t denote the empirical estimate of a random variable X from the minibatch at time t, and letX t represent the estimate of X from a moving average, so?? t andJ ??t are defined as: DISPLAYFORM6 DISPLAYFORM7 This moving average is equivalent to solving DISPLAYFORM8 by stochastic gradient descent and clearly has the true ?? and J ?? as a minimum for a fixed ??.

Note that Eq. 14 is a linear function of ?? and J ?? , so plugging in?? t and?? ??t gives an unbiased noisy estimate.

By also replacing terms that depend on ?? and J ?? with?? t andJ ??t , then alternately updating the moving averages and ?? t , we convert the problem into a two-timescale update.

Here ?? t corresponds to x t ,?? t andJ ??t correspond to y t ,??? ?? Tr(??(?? t ,?? t ,?? ??t ,J ??t )) corresponds to F(x t , y t ) and (?? t???1 ????? t ,J ??t???1 ????? ??t ) corresponds to G(x t , y t ).

We can finally combine all these elements together to define what a Spectral Inference Network is.

We consider a Spectral Inference Network to be any machine learning algorithm that:1.

Minimizes the objective in Eq. 6 end-to-end by stochastic optimization 2.

Performs the optimization over a parametric function class such as deep neural networks 3.

Uses the modified gradient in Eq. 14 to impose an ordering on the learned features 4.

Uses bilevel optimization to overcome the bias introduced by finite batch sizesThe full algorithm for training Spectral Inference Networks is given in Alg.

1, with TensorFlow pseudocode in the supplementary material in Sec. B. There are two things to note about this algorithm.

First, we have to compute an explicit estimate?? ??t of the Jacobian of the covariance with respect to the parameters at each iteration.

That means if we have N eigenfunctions we are computing, each step of training will require N 2 backward gradient computations.

This will be a bottleneck in scaling the algorithm, but we found this approach to be more stable and robust than others.

Secondly, while the theory of stochastic optimization depends on proper learning rate schedules, in practice these proper learning rate schedules are rarely used in deep learning.

Asymptotic convergence is usually less important than simply getting into the neighborhood of a local minimum, and even for bilevel problems, a careful choice of constant learning rates often suffices for good performance.

We follow this practice in our experiments and pick constant values of ?? and ??.

In this section we present empirical results on a quantum mechanics problem with a known closedform solution, and an example of unsupervised feature learning from video without a generative

As a first experiment to demonstrate the correctness of the method on a problem with a known solution, we investigated the use of SpIN for solving the Schr??dinger equation for a two-dimensional hydrogen atom.

The time-independent Schr??dinger equation for a single particle with mass m in a potential field V (x) is a partial differential equation of the form: DISPLAYFORM0 whose solutions describe the wavefunctions ??(x) with unique energy E. The probability of a particle being at position x then has the density |??(x)| 2 .

The solutions are eigenfunctions of the linear operator H ???h 2 2m ??? 2 + V (x) -known as the Hamiltonian operator.

We set 2 2m to 1 and choose V (x) = 1 |x| , which corresponds to the potential from a charged particle.

In 2 or 3 dimensions this can be solved exactly, and in 2 dimensions it can be shown that there are 2n + 1 eigenfunctions with energy ???1 (2n+1) 2 for all n = 0, 1, 2, . . . (Yang et al., 1991) .

We trained a standard neural network to approximate the wavefunction ??(x), where each unit of the output layer was a solution with a different energy E. Details of the training network and experimental setup are given in the supplementary material in Sec. C.1.

We found it critical to set the decay rate for RMSProp to be slower than the decay ?? used for the moving average of the covariance in SpIN, and expect the same would be true for other adaptive gradient methods.

To investigate the effect of biased gradients and demonstrate how SpIN can correct it, we specifically chose a small batch size for our experiments.

As an additional baseline over the known closed-form solution, we computed eigenvectors of a discrete approximation to H on a 128 ?? 128 grid.

Training results are shown in FIG0 , we see the circular harmonics that make up the electron orbitals of hydrogen in two dimensions.

With a small batch size and no bias correction, the eigenfunctions FIG0 are incorrect and the eigenvalues FIG0 , ground truth in black) are nowhere near the true minimum.

With the bias correction term in SpIN, we are able to both accurately estimate the shape of the eigenfunctions FIG0 and converge to the true eigenvalues of the system FIG0 .

Note that, as eigenfunctions 2-4 and 5-9 are nearly degenerate, any linear combination of them is also an eigenfunction, and we do not expect FIG0 and FIG0 to be identical.

The high accuracy of the learned eigenvalues gives strong empirical support for the correctness of our method.

Having demonstrated the effectiveness of SpIN on a problem with a known closed-form solution, we now turn our attention to problems relevant to representation learning in vision.

We trained a convolutional neural network to extract features from videos, using the Slow Feature Analysis kernel of Eq. 8.

The video is a simple example with three bouncing balls.

The velocities of the balls are constant until they collide with each other or the walls, meaning the time dynamics are reversible, and hence the transition function is a symmetric operator.

We trained a model with 12 output eigenfunctions using similar decay rates to the experiments in Sec. 5.1.

Full details of the training setup are given in Sec. C.2, including training curves in FIG4 .

During the course of training, the order of the different eigenfunctions often switched, as lower eigenfunctions sometimes took longer to fit than higher eigenfunctions.

Analysis of the learned solution is shown in FIG2 showing whether the feature is likely to be positively activated (red) or negatively activated (blue) when a ball is in a given position.

Since each eigenfunction is invariant to change of sign, the choice of color is arbitrary.

Most of the eigenfunctions are encoding for the position of balls independently, with the first two eigenfunctions discovering the separation between up/down and left/right, and higher eigenfunctions encoding higher frequency combinations of the same thing.

However, some eigenfunctions are encoding more complex joint statistics of position.

For instance, one eigenfunction (outlined in green in FIG2 ) has no clear relationship with the marginal position of a ball.

But when we plot the frames that most positively or negatively activate that feature FIG2 we see that the feature is encoding whether all the balls are crowded in the lower right corner, or one is there while the other two are far away.

Note that this is a fundamentally nonlinear feature, which could not be discovered by a shallow model.

Higher eigenfunctions would likely encode for even more complex joint relationships.

None of the eigenfunctions we investigated seemed to encode anything meaningful about velocity, likely because collisions cause the velocity to change rapidly, and thus optimizing for slowness of features is unlikely to discover this.

A different choice of kernel may lead to different results.

We have shown that a single unified framework is able to compute spectral decompositions by stochastic gradient descent on domains relevant to physics and machine learning.

This makes it possible to learn eigenfunctions over very high-dimensional spaces from very large datasets and generalize to new data without the Nystr??m approximation.

This extends work using slowness as a criterion for unsupervised learning without a generative model, and addresses an unresolved issue with biased gradients due to finite batch size.

A limitation of the proposed solution is the requirement of computing full Jacobians at every time step, and improving the scaling of training is a promising direction for future research.

The physics application presented here is on a fairly simple system, and we hope that Spectral Inference Nets can be fruitfully applied to more complex physical systems for which computational solutions are not yet available.

The representations learned on video data show nontrivial structure and sensitivity to meaningful properties of the scene.

These representations could be used for many downstream tasks, such as object tracking, gesture recognition, or faster exploration and subgoal discovery in reinforcement learning.

Finally, while the framework presented here is quite general, the examples shown investigated only a small number of linear operators.

Now that the basic framework has been laid out, there is a rich space of possible kernels and architectures to combine and explore.

A BREAKING THE SYMMETRY BETWEEN EIGENFUNCTIONS Since Eq. 6 is invariant to linear transformation of the features u(x), optimizing it will only give a function that spans the top K eigenfunctions of K. We discuss some of the possible ways to recover ordered eigenfunctions, explain why we chose the approach of using masked gradients, and provide a derivation of the closed form expression for the masked gradient in Eq. 12.

If u * (x) is a function to R N that is an extremum of Eq. 6, then E x [k(x, x )u * (x )] = ???u * (x) for some matrix ??? ??? R N ??N which is not necessarily diagonal.

We can express this matrix in terms of quantities in the objective: DISPLAYFORM0 To transform u * (x) into ordered eigenfunctions, first we can orthogonalize the functions by multi- DISPLAYFORM1 The matrix L ???1 ??L ???T = ?? is symmetric, so we can diagonalize it: ?? = VDV T , and then DISPLAYFORM2 are true eigenfunctions, with eigenvalues along the diagonal of D. In principle, we could optimize Eq. 6, accumulating statistics on ?? and ??, and transform the functions u * into w * at the end.

In practice, we found that the extreme eigenfunctions were "contaminated" by small numerical errors in the others eigenfunctions, and that this approach struggled to learn degenerate eigenfunctions.

This inspired us to explore the masked gradient approach instead, which improves numerical robustness.

Throughout this section, let x i:j be the slice of a vector from row i to j and let A i:j,k: be the block of a matrix A containing rows i through j and columns k through .

DISPLAYFORM0 T ], L be the Cholesky decomposition of ?? and ?? = L ???1 ??L ???T .

The arguments here are not meant as mathematically rigorous proofs but should give the reader enough of an understanding to be confident that the numerics of our method are correct for optimization over a sufficiently expressive class of functions.

Claim 1.

?? 1:n,1:n is independent of u n+1:n (x).The Cholesky decomposition of a positive-definite matrix is the unique lower triangular matrix with positive diagonal such that LL T = ??. Expanding this out into blocks yields: DISPLAYFORM1 ?? 1:n,1:n ?? 1:n,n+1:N ?? n+1:N,1:n ?? n+1:N,n+1:N Inspecting the upper left block, we see that L 1:n,1:n L T 1:n,1:n = ?? 1:n,1:n .

As L 1:n,1:n is also lowertriangular, it must be the Cholesky decomposition of ?? 1:n,1:n .

The inverse of a lower triangular matrix will also be lower triangular, and a similar argument to the one above shows that the upper left block of the inverse of a lower triangular matrix will be the inverse of the upper left block, so the upper left block of ?? can be written as: DISPLAYFORM2 DISPLAYFORM3 , ordered from highest eigenvalue to lowest.

The argument proceeds by induction.

DISPLAYFORM4 , which is simply the Rayleigh quotient in Eq. 10 for i = 1.

The maximum of this is clearly proportional to the top eigenfunction, and DISPLAYFORM5 1:n,1:n u 1:n (x) are the first n eigenfunctions of K. Because u 1:n (x) span the first n eigenfunctions, and ?? ii is independent of u n+1 (x) for i < n+1, u 1:n+1 (x) is a maximum of n i=1 ?? ii no matter what the function u n+1 (x) is.

Training u n+1 (x) with the masked gradient ??? u Tr(??) is equivalent to maximizing ?? (n+1)(n+1) , so for the optimal u n+1 (x), u 1:n+1 (x) will be a maximum of n+1 i=1 ?? ii .

Therefore u 1:n (x) span the first n eigenfunctions and u 1:n+1 (x) span the first n + 1 eigenfunctions, so orthogonalizing u 1:n+1 (x) by multiplication by L ???1 1:n+1,1:n+1 will subtract anything in the span of the first n eigenfunctions off of u n+1 (x), meaning v n+1 (x) will be the (n + 1)th eigenfunction of K

The derivative of the normalized features with respect to parameters can be expressed as DISPLAYFORM0 if we flatten out the matrix-valued L, ?? and ??.The reverse-mode sensitivities for the matrix inverse and Cholesky decomposition are given by?? = ???C TC C T where DISPLAYFORM1 where L is the Cholesky decomposition of ?? and ??(??) is the operator that replaces the upper triangular of a matrix with its lower triangular transposed (Giles, 2008; BID8 .

Using this, we can compute the gradients in closed form by application of the chain rule.

To simplify notation slightly, let ??? k and ?? k be matrices defined as: DISPLAYFORM2 Then the unmasked gradient has the form: DISPLAYFORM3 DISPLAYFORM4 while the gradients of ?? and ?? with respect to u are given (elementwise) by: DISPLAYFORM5 DISPLAYFORM6 which, in combination, give the unmasked gradient with respect to u as: DISPLAYFORM7 Here the gradient is expressed as a row vector, to be consistent with Eq. 25, and a factor of 2 has been dropped in the last line that can be absorbed into the learning rate.

To zero out the relevant elements of the gradient ??? u ?? kk as described in Eq. 11, we can rightmultiply by ??? k .

The masked gradients can be expressed in closed form as: DISPLAYFORM8 where triu and diag give the upper triangular and diagonal of a matrix, respectively.

A TensorFlow implementation of this masked gradient is given below.

Here we provide a short pseudocode implementation of the updates in Alg.

1 in TensorFlow.

The code is not intended to run as is, and leaves out some global variables, proper initialization and code for constructing networks and kernels.

However all nontrivial elements of the updates are given in detail here.import tensorflow as tf from tensorflow.python.ops.parallel_for import jacobian @tf.custom_gradient def covariance(x, y): batch_size = float(x.shape [0] .value) cov = tf.matmul(x, y, transpose_a=True) / batch_size def gradient(grad): return (tf.matmul(y, grad) / batch_size, tf.matmul(x, grad) / batch_size) return cov, gradient @tf.custom_gradient def eigenvalues(sigma, pi):"""Eigenvalues as custom op so that we can overload gradients.

""" chol = tf.cholesky(sigma) choli = tf.linalg.inv(chol) rq = tf.matmul(choli, tf.matmul(pi, choli, transpose_b=True)) eigval = tf.matrix_diag_part(rq) def gradient(_): """Symbolic form of the masked gradient.

""" dl = tf.diag(tf.matrix_diag_part(choli)) triu = tf.matrix_band_part(tf.matmul(rq, dl), 0, -1) dsigma = -1.0 * tf.matmul(choli, triu, transpose_a=True) dpi = tf.matmul (choli, dl, transpose_a=True) return dsigma, dpi return eigval, gradient def moving_average(x, c):"""Creates moving average operation.

This is pseudocode for clarity!

Should actually initialize sigma_avg with tf.eye, and should add handling for when x is a list. """ ma = tf.

Variable(tf.zeros_like(x), trainable=False) ma_update = tf.assign(ma, (1-c) * ma + c * x) return ma, ma_update def spin(x1, x2, network, kernel, params, optim): """Function to create TensorFlow ops for learning in SpIN.Args: x1: first minibatch, of shape (batch size, input dimension) x2: second minibatch, of shape (batch size, input dimension) network: function that takes minibatch and parameters and returns output of neural network kernel: function that takes two minibatches and returns symmetric function of the inputs params: list of tf.

Variables with network parameters optim: an instance of a tf.train.Optimizer object Returns:step: op that implements one iteration of SpIN training update eigenfunctions: op that gives ordered eigenfunctions """ # 'u1' and 'u2' are assumed to have the batch elements along first # dimension and different eigenfunctions along the second dimension u1 = network(x1, params) u2 = network(x2, params) sigma = 0.5 * (covariance(u1, u1) + covariance(u2, u2)) sigma.set_shape ((u1.shape[1] , u1.shape[1])) # For the hydrogen examples in Sec. 4.1, 'kernel(x1, x2) * u2' # can be replaced by the result of applying the operator # H to the function defined by 'network(x1, params)'.

pi = covariance(u1, kernel(x1, x2) * u2) pi.set_shape ((u1 To solve for the eigenfunctions with lowest eigenvalues, we used a neural network with 2 inputs (for the position of the particle), 4 hidden layers each with 128 units, and 9 outputs, corresponding to the first 9 eigenfunctions.

We used a batch size of 128 -much smaller than the 16,384 nodes in the 2D grid used for the exact eigensolver solution.

We chose a softplus nonlinearity log(1 + exp(x)) rather than the more common ReLU, as the Laplacian operator ??? 2 would be zero almost everywhere for a ReLU network.

We used RMSProp (Tieleman & Hinton, 2012) with a decay rate of 0.999 and learning rate of 1e-5 for all experiments.

We sampled points uniformly at random from the box DISPLAYFORM0 2 during training, and to prevent degenerate solutions due to the boundary condition, we multiplied the output of the network by i ( 2D 2 ??? x 2 i ??? D), which forces the network output to be zero at the boundary without the derivative of the output blowing up.

We chose D = 50 for the experiments shown here.

We use the finite difference approximation of the differential Laplacian given in Sec. 3.2 with some small number (around 0.1), which takes the form: DISPLAYFORM1 when applied to ??(x).

Because the Hamiltonian operator is a purely local function of ??(x), we don't need to sample pairs of points x, x for each minibatch, which simplifies calculations.

We made one additional modification to the neural network architecture to help separation of different eigenfunctions.

Each layer had a block-sparse structure that became progressively more separated the deeper into the network it was.

For layer out of L with m inputs and n outputs, the weight w ij was only nonzero if there exists k ??? {1, . . .

, K} such that i ??? [ DISPLAYFORM2 .

This split the weight matrices into overlapping blocks, one for each eigenfunction, allowing features to be shared between eigenfunctions in lower layers of the network while separating out features which were distinct between eigenfunctions higher in the network.

We trained on 200,000 64??64 pixel frames, and used a network with 3 convolutional layers, each with 32 channels, 5??5 kernels and stride 2, and a single fully-connected layer with 128 units before outputting 12 eigenfunctions.

We also added a constant first eigenfunction, since the first eigenfunction of the Laplacian operator is always constant with eigenvalue zero.

This is equivalent to forcing the features to be zero-mean.

We used the same block-sparse structure for the weights that was used in the Schr??dinger equation experiments, with sparsity in weights between units extended to sparsity in weights between entire feature maps for the convolutional layers.

We trained with RMSProp with learning rate 1e-6 and decay 0.999 and covariance decay rate ?? = 0.01 for 1,000,000 iterations.

To make the connection to gradient descent clearer, we use the opposite convention to RMSProp: ?? = 1 corresponds to zero memory for the moving average, meaning the RMS term in RMSProp decays ten times more slowly than the covariance moving average in these experiments.

Each batch contained 24 clips of 10 consecutive frames.

So that the true state was fully observable, we used two consecutive frames as the input x t , x t+1 and trained the network so that the difference from that and the features for the frames x t+1 , x t+2 were as small as possible. , we trained a network to perform next-frame prediction on 500k frames of a random agent playing one game.

We simultaneously trained another network to compute the successor features BID1 of the latent code of the next-frame predictor, and computed the "eigenpurposes" by applying PCA to the successor features on 64k held-out frames of gameplay.

We used the same convolutional network architecture as BID6 , a batch size of 32 and RMSProp with a learning rate of 1e-4 for 300k iterations, and updated the target network every 10k iterations.

While the original paper did not mean-center the successor features when computing eigenpurposes, we found that the results were significantly improved by doing so.

Thus the baseline presented here is actually stronger than in the original publication.

On the same data, we trained a spectral inference network with the same architecture as the encoder of the successor feature network, except for the fully connected layers, which had 128 hidden units and 5 non-constant eigenfunctions.

We tested SpIN on the same 64k held-out frames as those used to estimate the eigenpurposes.

We used the same training parameters and kernel as in Sec. 5.2.

As SpIN is not a generative model, we must find another way to compare the features learned by each method.

We averaged together the 100 frames from the test set that have the largest magnitude positive or negative activation for each eigenfunction/eigenpurpose.

Results are shown in FIG5 , with more examples and comparison against PCA on pixels at the end of this section.

By comparing the top row to the bottom row in each image, we can judge whether that feature is encoding anything nontrivial.

If the top and bottom row are noticeably different, this is a good indication that something is being learned.

It can be seen that for many games, successor features may find a few eigenpurposes that encode interesting features, but many eigenpurposes do not seem to encode anything that can be distinguished from the mean image.

Whereas for SpIN, nearly all eigenfunctions are encoding features such as the presence/absence of a sprite, or different arrange- ments of sprites, that lead to a clear distinction between the top and bottom row.

Moreover, SpIN is able to learn to encode these features in a fully end-to-end fashion, without any pixel reconstruction loss, whereas the successor features must be trained from two distinct losses, followed by a third step of computing eigenpurposes.

The natural next step is to investigate how useful these features are for exploration, for instance by learning options which treat these features as rewards, and see if true reward can be accumulated faster than by random exploration.

@highlight

We show how to learn spectral decompositions of linear operators with deep learning, and use it for unsupervised learning without a generative model.

@highlight

The authors propose to use a deep learning framework to solve the computation of the largest eigenvectors.

@highlight

This paper presents a framework to learn eigenfunctions via a stochastic process and proposes to tackle the challenge of computing eigenfunctions in a large-scale context by approximating then using a two-phase stochastic optimization process.
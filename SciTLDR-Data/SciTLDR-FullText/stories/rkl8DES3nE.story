We study SGD and Adam for estimating a rank one signal planted in matrix or tensor noise.

The extreme simplicity of the problem setup allows us to isolate the effects of various factors: signal to noise ratio, density of critical points, stochasticity and initialization.

We observe a surprising phenomenon: Adam seems to get stuck in local minima as soon as polynomially many critical points appear (matrix case), while SGD escapes those.

However, when the number of critical points degenerates to exponentials (tensor case), then both algorithms get trapped.

Theory tells us that at fixed SNR the problem becomes intractable for large $d$ and in our experiments SGD does not escape this.

We exhibit the benefits of warm starting in those situations.

We conclude that in this class of problems, warm starting cannot be replaced by stochasticity in gradients to find the basin of attraction.

Reductionism consists of breaking down the study of complex systems and phenomena into their atomic components.

While the use of stochastic gradient based algorithms has shown tremendous success at minimizing complicated loss functions arising in deep learning, our understanding of why, when and how this happens is still limited.

Statements such as stochastic gradients escape from isolated critical points along the road to the best basin of attraction, or SGD generalizes better because it does not get stuck in steep local minima still need to be better understood.

Can we prove or replicate these phenomena in the simplest instances of the problem?

We study the behavior of stochastic gradient descent (SGD) BID11 and an adaptive variant (Adam) BID8 under a class of well studied non-convex problems.

The single spiked models were originally designed for studying principal component analysis on matrices BID12 BID3 BID5 and have also been extended to higher order tensors BID10 .

Adaptive stochastic optimization methods have been gaining popularity in the deep learning community thanks to fast training on some benchmarks.

However, it has been observed that despite reaching a low value of the loss function, the solutions found by Adam do not generalize as well as SGD solutions do.

An assumption, widely spread and adopted in the community, has been that SGD's randomness helps escaping local critical points [WRS + 17] .

While the problem has been thoroughly studied theoretically [MR14, HSS15, HSSS16, BAGJ18], our contribution is to propose experimenting with this simple model to challenge claims such as those on randomized gradient algorithms in this very simple setup.

It is noteworthy that the landscape of non-global critical points of these toy datasets are studied BID0 BID2 BID1 and formally linked to the neural nets empirical loss functions BID2 BID9 .

For this problem, the statistical properties of the optimizers are well understood, and in the more challenging tensor situation, also the impact of (spectral) warm start has been discussed BID10 .

We will examine the solutions found by SGD and Adam and compare them with spectral and power methods.

This allows to empirically elucidate the existence of multiple regimes: (1) the strong signal regime where all first order methods seem to find good solutions (2) when polynomially many critical points appear, in the matrix case, SGD converges while Adam gets trapped, unless if initialized in the basin of attraction (3) in the presence of exponentially many critical points (the tensor case), all algorithms fail, unless if d is moderately small and the SNR large enough to allow for proper initialization.2 Single spiked models, and stochastic gradients

Even though proving strong results about non-convex loss functions is in general challenging, a class of nonconvex statistical problems is very well studied and relatively well understood.

Principal component analysis (PCA) or finding the leading eigenvector of a covariance matrix is a problem of interest in statistics and machine learning.

The proof of convergence of power method to the leading principal component and the geometry of critical points of the maximum likelihood problem maximize u, Au s.t.

u 2 = 1 , Rayleigh quotient for matrix PCA are well established using eigenvalue decomposition.

In addition, more recently, a class of extremely simplified models have shed light on the phase transitions of the problem difficulty as a function of the signal-to-noise ratio in the model.

The so-called single spiked models consist of considering a symmetric normalized noise matrix to which a rank one signal is added.

DISPLAYFORM0 It is known BID3 that the spectrum of the noise matrix asymptotically forms a semi-circle situated between −2 and 2.

When the signal to noise ratio is weak λ ∈ [0, 1] then the signal dilutes in noise, while the leading principal component pops out of the semi-circle as soon as the signal to noise ratio λ is above the critical value λ > λ c = 1, in which case the solution of the problem forms asymptotically a cosine value of 1 − λ −2 1/2 with the signal and the optimal value of the Rayleigh quotient is λ + λ −1 BID12 BID3 BID5 .

It is proven that the power method allows to obtain the solution after logarithmically many steps, as a function of the problem dimension d.

We will minimize the unconstrained objective function ( DISPLAYFORM1 We set the value of γ to the theoretical asymptotic value of the leading eigenvalue, γ = 2 for λ < 1 or λ + λ −1 for larger λ, and will add random normal noise to the gradient for stochasticity: ∇ σ (u) = −Au + γu + σz where z i ∼ N (0, 1/d).

This function has a constant Hessian H (u) = −λu 0 u 0 T − Z + γI d which is positive semi-definite as soon as γ is equal or larger than the value of the leading eigenvalue of A.

The tensor version of the problem (see BID10 for notations and more discussion on problem setting) DISPLAYFORM0 u 2 = 1 Rayleigh quotient tensor PCA under the tensor single spiked model defined for a symmetric (π is a permutation of 3 elements) How does data abundance explain the success of first order methods?

On the positive side, we discuss that in this model and considering a large dataset of i.id.

samples, weak signals in individual observations accumulate and allow to solve the problem if n √ d. The counter part to the strong requirement λ d 1/4 (conjectured in BID10 and proven in BID6 BID7 ) is that accumulation of observations compensate low signal to noise ratio in each individual sample.

Formally, Remark 2.1.

Assume n sample of data according to model "Single spiked tensor", with the same signal u 0 and different i.i.d.

noises Z q are observed: DISPLAYFORM1 DISPLAYFORM2 There exists constants c 0 , c 1 such that if n ≥ c 0 √ d, then, warm started power iteration produces a vector u, such that with high probability u 0 , u > 1 − c 1 /λ.

This result is established using Theorems 5 in BID10 and 6.3 in BID7 and considering the average tensorĀ DISPLAYFORM3 SinceZ is symmetric andZ i,j,k ∼ N (0, 1/d), the tensorĀ is sampled from a similar distribution as Single spiked tensor with a SNR λ n = √ nλ.

This means that the requirement λ d 1/4 BID7 , in the average tensor case, relaxes to n √ d. In words, this means that if we are solving a problem with a tensor PCA complexity, and if the number of i.i.d.

observations grows quadratically as a function of the problem dimension, we can compute the solution reliably using spectral warm start, even though the original problem looks intractable.

Our numerical results report performance of different algorithms at solving simulations of matrix and tensor PCA problems.

Under various problem generation parameter choices, we report values of• cosine or u, u 0 .

This is measures the quality of planted (hidden) signal recovery from the noisy observation.

Higher values are preferred, and it cannot exceed 1, since both u 0 and u are normalized.

This quantity is to be qualitatively compared with the test error in standard learning problems where the true value of the parameter is not available.• Rayleigh is the value of the log-likelihood objective function that we are maximizing.

Higher values are preferred.

The theoretical maximum value of this objective is the operator norm of the observed tensor A. This is comparable with (minus) the training loss.

The signal to noise ratio λ is to be compared with the number of observations in a supervised learning problem.

The stochasticity of the gradients σ is to be compared with the number of sample points in each minibatch of data: large stochasticity mimics small minibatch situations.

In Figure 1 we plot the values of the objective function or Rayleigh quotient and the cosine of the ground truth with the solution as a function of the iterates.

We replicated these plots at values of the SNR parameter λ < λ c = 1, at the critical value λ = 1 and above it for λ = 2 where the problems is considered to be easy.

The learning rate and stochasticity σ were set by generating instances of the problem with different noise matrices.

These plots allow to compare the optimization power of different algorithms and also keep track of the quality of the solution found.

We can see in these plots that Adam gets stuck around the wrong region very fast.

Note that with the value (or larger) of γ = 2 for λ < 1 and γ = λ + λ −1 for λ ≥ 1 that we can set given the true value of λ, and knowing the concentration around the asymptotics BID12 BID3 BID5 , the objective function is strongly convex so we expect first order methods to show the same convergence rates as the power method.

One can also observe that gradient descent corresponds to performing power iteration on a shifted matrix A + αI d .

Figure 2 shows the value of the objective and the correlation with the ground truth as a function of SNR λ.

We can see that SGD is superior to Adam, uniformly along λ, while power method (also a first order method) rivals with SGD.

These plots exhibit the instability around and below the critical value λ c = 1 while above λ c the behavior is more stable.

In the tensor setting we experimented with spectral initialization.

Spectral initialization consists of flattening the tensor to a d×d 2 matrix and initializing tensor algorithms with the left singular vector of the flattened tensor.

We observe benefits of spectral initialization at locating the initial point in a basin of attraction that leads to better solutions when λ is large enough.

For λ = 1.0 spectral initialization does not result in better estimates in Figure 3 , while for larger values of λ we can see the benefit of warm start.

In FIG2 we also plot values of the gradient and the number of positive eigenvalues of the Hessian along the optimization iterates for d = 100, λ = 2, γ = 2.

We observe that spectral initialization located the initial point of the iterations in the basin of attraction where the problem is convex (all eigenvalues of the Hessian are positive).

Adam, while starting in this region, fails at finding a solution as good as SGD's.

We experimented with the amount of noise added to each gradient evaluation and mapped median values of the estimate and optimization problems for values of λ and σ over 100 instances of the problem generated with different noise matrices and stochastic gradients.

Stochasticity does not seem to remedy to the problem difficulty.

Numerical experiments suggest that irrespective of the magnitude of the stochastic component added to the gradient, the first order methods, initialized at random, fail at finding the best basin of attraction.

In the same setup, spectral initialized first order methods successfully find the solutions.

We propose to study algorithms used for minimizing deep learning loss functions, at optimizing a non-convex objective on simple synthetic datasets.

Studying simplified problems has the advantage that the problem's properties, and the behavior of the optimizer and the solution, can be studied rigorously.

The use of such datasets can help to perform sanity checks on improvement ideas to the algorithms, or to mathematically prove or disprove intuitions.

The properties of the toy data sets align with some properties of deep learning loss functions.

From the optimization standpoint, the resulting tensor problems may appear to be even harder than deep learning problems.

We observe that finding good solutions is hard unless if proper initialization is performed, while the value of stochasticity in gradient estimates seems too narrow and does not appear to compensate for poor initialization heuristics.

Each column represents the values of those quantities along iterations of the algorithm.

The prefix sp.

refers to spectral initialization and l. refers to a decreasing learning weight scheduled in 1/ √ t. We observe the value of warm starting as soon as λ is large enough.

Even at high SNR λ = 6, randomly initialized SGD fails while spectrally initialized SGD succeeds.

Adam drifts to a non optimal critical point in that regime, even with spectral warm start.

@highlight

SGD and Adam under single spiked model for tensor PCA
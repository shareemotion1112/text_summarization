We give a simple, fast algorithm for hyperparameter optimization inspired by techniques from the analysis of Boolean functions.

We focus on the high-dimensional regime where the canonical example is training a neural network with a large number of hyperparameters.

The algorithm --- an iterative application of compressed sensing techniques for orthogonal polynomials --- requires only uniform sampling of the hyperparameters and is thus easily parallelizable.

Experiments for training deep neural networks on Cifar-10 show that compared to state-of-the-art tools (e.g., Hyperband and Spearmint), our algorithm finds significantly improved solutions, in some cases better than what is attainable by hand-tuning.

In terms of overall running time (i.e., time required to sample various settings of hyperparameters plus additional computation time), we are at least an order of magnitude faster than Hyperband and Bayesian Optimization.

We also outperform Random Search $8\times$.     Our method is inspired by provably-efficient algorithms for learning decision trees using the discrete Fourier transform.

We obtain improved sample-complexty bounds for learning decision trees while matching state-of-the-art bounds on running time (polynomial and quasipolynomial, respectively).

Large scale machine learning and optimization systems usually involve a large number of free parameters for the user to fix according to their application.

A timely example is the training of deep neural networks for a signal processing application: the ML specialist needs to decide on an architecture, depth of the network, choice of connectivity per layer (convolutional, fully-connected, etc.), choice of optimization algorithm and recursively choice of parameters inside the optimization library itself (learning rate, momentum, etc.).Given a set of hyperparameters and their potential assignments, the naive practice is to search through the entire grid of parameter assignments and pick the one that performed the best, a.k.a.

"grid search".

As the number of hyperparameters increases, the number of possible assignments increases exponentially and a grid search becomes quickly infeasible.

It is thus crucial to find a method for automatic tuning of these parameters.

This auto-tuning, or finding a good setting of these parameters, is now referred to as hyperparameter optimization (HPO), or simply automatic machine learning (auto-ML).

For continuous hyperparameters, gradient descent is usually the method of choice BID25 BID24 BID9 .

Discrete parameters, however, such as choice of architecture, number of layers, connectivity and so forth are significantly more challenging.

More formally, let DISPLAYFORM0 be a function mapping hyperparameter choices to test error of our model.

That is, each dimension corresponds to a certain hyperparameter (number of layers, connectivity, etc.), and for simplicity of illustration we encode the choices for each parameter as binary numbers {???1, 1}. The goal of HPO is to approximate the minimizer x * = arg min x???{0,1} n f (x) in the following setting:1.

Oracle model: evaluation of f for a given choice of hyperparameters is assumed to be very expensive.

Such is the case of training a given architecture of a huge dataset.2.

Parallelism is crucial: testing several model hyperparameters in parallel is entirely possible in cloud architecture, and dramatically reduces overall optimization time.3.

f is structured.

The third point is very important since clearly HPO is information-theoretically hard and 2 n evaluations of the function are necessary in the worst case.

Different works have considered exploiting one or more of the properties above.

The approach of Bayesian optimization BID32 addresses the structure of f , and assumes that a useful prior distribution over the structure of f is known in advance.

Multi-armed bandit algorithms BID22 , and Random Search BID2 , exploit computational parallelism very well, but do not exploit any particular structure of f 1 .

These approaches are surveyed in more detail later.

In this paper we introduce a new spectral approach to hyperparameter optimization.

Our main idea is to make assumptions on the structure of f in the Fourier domain.

Specifically we assume that f can be approximated by a sparse and low degree polynomial in the Fourier basis.

This means intuitively that it can be approximated well by a decision tree.

The implication of this assumption is that we can obtain a rigorous theoretical guarantee: approximate minimization of f over the boolean hypercube with function evaluations only linear in sparsity that can be carried out in parallel.

We further give improved heuristics on this basic construction and show experiments showing our assumptions are validated in practice for HPO as applied to deep learning over image datasets.

Thus our contributions can be listed as:??? A new spectral method called Harmonica that has provable guarantees: sample-efficient recovery if the underlying objective is a sparse (noisy) polynomial and easy to implement on parallel architectures.???

We demonstrate significant improvements in accuracy, sample complexity, and running time for deep neural net training experiments.

We compare ourselves to state-of-the-art solvers from Bayesian optimization, Multi-armed bandit techniques, and Random Search.

Projecting to even higher numbers of hyperparameters, we perform simulations that show several orders-of-magnitude of speedup versus Bayesian optimization techniques.??? Improved bounds on the sample complexity of learning noisy, size s decision trees over n variables under the uniform distribution.

We observe that the classical sample complexity bound of n O(log(s/??)) due to BID23 can be improved to quadratic in the size of the tree??(s 2 /?? ?? log n) while matching the best known quasipolynomial bound in running time.

The literature on discrete-domain HPO can be roughly divided into two: probabilistic approaches and decision-theoretic methods.

In critical applications, researchers usually use a grid search over all parameter space, but that becomes quickly prohibitive as the number of hyperparameter grows.

Gradient-based methods such as BID25 BID24 BID9 BID1 are applicable only to continuous hyperparameters which we do not consider.

Neural network structural search based on reinforcement learning is an active direction BID0 BID44 BID43 , which usually needs many samples of network architectures.

Probabilistic methods and Bayesian optimization.

Bayesian optimization (BO) algorithms BID3 BID32 BID37 BID33 BID10 BID40 BID17 tune hyperparameters by assuming a prior distribution of the loss function, and then keep updating this prior distribution based on the new observations.

Each new observation is selected according to an acquisition function, which balances exploration and exploitation such that the new observation gives us a better result, or helps gain more information.

The BO approach is inherently serial and difficult to parallelize, and its theoretical guarantees have thus far been limited to statistical consistency (convergence in the limit).Decision-theoretic methods.

Perhaps the simplest approach to HPO is random sampling of different choices of parameters and picking the best amongst the chosen evaluations BID2 .

It is naturally very easy to implement and parallelize.

Upon this simple technique, researchers have tried to allocate different budgets to the different evaluations, depending on their early performance.

Using adaptive resource allocation techniques found in the multi-armed bandit literature, Successive Halving (SH) algorithm was introduced .

Hyperband further improves SH by automatically tuning the hyperparameters in SH BID22 .Learning decision trees.

Prior work for learning decision trees (more generally Boolean functions that are approximated by low-degree polynomials) used the celebrated "low-degree" algorithm of BID23 .

Their algorithm uses random sampling to estimate each low-degree Fourier coefficient to high accuracy.

We make use of the approach of BID35 , who showed how to learn low-degree, sparse Boolean functions using tools from compressed sensing (similar approaches were taken by BID21 and BID27 ).

We observe that their approach can be extended to learn functions that are both "approximately sparse" (in the sense that the L 1 norm of the coefficients is bounded) and "approximately low-degree" (in the sense that most of the L 2 mass of the Fourier spectrum resides on monomials of low-degree).

This implies the first decision tree learning algorithm with polynomial sample complexity that handles adversarial noise.

In addition, we obtain the optimal dependence on the error parameter ??.

For the problem of learning exactly k-sparse Boolean functions over n variables, BID12 have recently shown that O(nk log n) uniformly random samples suffice.

Their result is not algorithmic but does provide an upper bound on the information-theoretic problem of how many samples are required to learn.

The best algorithm in terms of running time for learning k-sparse Boolean functions is due to BID8 , and requires time 2 ???(n/ log n) .

It is based on the BID4 algorithm for learning parities with noise.

Techniques.

Our methods are heavily based on known results from the analysis of boolean functions as well as compressed sensing.

The problem of hyperparameter optimization is that of minimizing a discrete, real-valued function, which we denote by f : {???1, 1} n ??? [???1, 1] (we can handle arbitrary inputs, binary is chosen for simplicity of presentation).In the context of hyperparameter optimization, function evaluation is very expensive, although parallelizable, as it corresponds to training a deep neural net.

In contrast, any computation that does not involve function evaluation is considered less expensive, such as computations that require time ???(n d ) for "somewhat large" d or are subexponential (we still consider runtimes that are exponential in n to be costly).

The reader is referred to BID28 for an in depth treatment of Fourier analysis of Boolean functions.

Let f : X ??? [???1, 1] be a function over domain X ??? R n .

Let D a probability distribution on X .

We write g ??? ?? f and say that f, g are ??- DISPLAYFORM0 Definition 1.

BID29 We say a family of functions ?? 1 , . . .

, ?? N (?? i maps X to R) is a Random Orthonormal Family with respect to D if DISPLAYFORM1 The expectation is taken with respect to probability distribution D. We say that the family is Kbounded if sup x???X |?? i (x)| ??? K for every i. Henceforth we assume K = 1.An important example of a random orthonormal family is the class of parity functions with respect to the uniform distribution on {???1, 1} n :Definition 2.

A parity function on some subset of variables DISPLAYFORM2 It is easy to see that the set of all 2 n parity functions {?? S }, one for each S ???

[n], form a random orthonormal family with respect to the uniform distribution on {???1, 1}n .This random orthonormal family is often referred to as the Fourier basis, as it is a complete orthonormal basis for the class of Boolean functions with respect to the uniform distribution on {???1, 1} n .

More generally, for any f : {???1, 1} n ??? R, f can be uniquely represented in this basis as DISPLAYFORM3 is the Fourier coefficient corresponding to S where x is drawn uniformly from {???1, 1} n .

We also have Parseval's identity: DISPLAYFORM4 S .

In this paper, we will work exclusively with the above parity basis.

Our results apply more generally, however, to any orthogonal family of polynomials (and corresponding product measure on R n ).

For example, if we wished to work with continuous hyperparameters, we could work with families of Hermite orthogonal polynomials with respect to multivariate spherical Gaussian distributions.

We conclude with a definition of low-degree, approximately sparse (bounded L 1 norm) functions: Definition 3 (Approximately sparse function).

Let {?? S } be the parity basis, and let C be a class of functions mapping {???1, 1} n to R. Thus for f ??? C, f = Sf (S)?? S .

We say a function f ??? C is s-sparse if L 0 (f ) ??? s, ie., f has at most s nonzero entries in its polynomial expansion.

f is DISPLAYFORM5 It is easy to see that the class of functions with bounded L 1 norm is more general than sparse functions.

For example, the Boolean AND function has L 1 norm bounded by 1 but is not sparse.

We also have the following simple fact: Fact 4.

BID26 Let f be such that DISPLAYFORM6 The function g is constructed by taking all coefficients of magnitude ??/s or larger in f 's expansion as a polynomial.

In the problem of sparse recovery, a learner attempts to recover a sparse vector x ??? R n which is s sparse, i.e. x 0 ??? s, from an observation vector y ??? R m that is assumed to equal y = Ax + e, where e is assumed to be zero-mean, usually Gaussian, noise.

The seminal work of BID6 ; BID7 shows how x can be recovered exactly under various conditions on the observation matrix A ??? R m??n and the noise.

The usual method for recovering the signal proceeds by solving a convex optimization problem consisting of 1 minimization as follows (for some parameter ?? > 0): DISPLAYFORM0 (1)The above formulation comes in many equivalent forms (e.g., Lasso), where one of the objective parts may appear as a hard constraint.

DISPLAYFORM1 Figure 1: Compressed sensing over the Fourier domain: Harmonica recovers the Fourier coefficients of a sparse low degree polynomial S ??S??S(xi) from observations f (xi) of randomly chosen points xi ??? {???1, 1} n .For our work, the most relevant extension of traditional sparse recovery is due to BID29 , who considers the problem of sparse recovery when the measurements are evaluated according to a random orthonormal family.

More concretely, fix x ??? R n with s non-zero entries.

For Kbounded random orthonormal family F = {?? 1 , . . .

, ?? N }, and m independent draws z 1 , . . . , z DISPLAYFORM2 Rauhut gives the following result for recovering sparse vectors x: Theorem 5 (Sparse Recovery for Random Orthonormal Families, BID29 Theorem 4.4).

Given as input matrix A ??? R m??N and vector y with y i = Ax + e i for some vector e with e 2 ??? ?? ??? m, mathematical program (1) finds a vector x * such that for constants c 1 and c 2 , DISPLAYFORM3 + c 2 ?? with probability 1 ??? ?? as long as for sufficiently large constant C, DISPLAYFORM4 The term ?? s (x) 1 is equal to min{ x ??? z 1 , z is s sparse}. Recent work BID5 BID13 has improved the dependence on the polylog factors in the lower bound for m.

The main component of our spectral algorithm for hyperparameter optimization is given in Algorithm 1 2 .

It is essentially an extension of sparse recovery (basis pursuit or Lasso) to the orthogonal basis of polynomials in addition to an optimization step.

See Figure 1 for an illustration.

We prove Harmonica's theoretical guarantee, and show how it gives rise to new theoretical results in learning from the uniform distribution.

In the next section we describe extensions of this basic algorithm to a more practical algorithm with various heuristics to improve its performance.

Algorithm 1 Harmonica-1 1: Input: oracle for f , number of samples T , sparsity s, degree d, parameter ??.

2: Invoke PSR(f, T, s, d, ??) (Procedure 2) to obtain (g, J), where g is a function defined on variables specified by index set J ??? [n].

3: Set the variables in [n] \ J to arbitrary values, compute a minimizer x ??? arg min g(x).

4: return x Theorem 6 (Noiseless recovery).

Let {?? S } be a 1-bounded orthonormal polynomial basis for distribution D. Let f : R n ??? R be a (0, d, s)-bounded function as per definition 3 with respect to the basis ?? S .

Then Algorithm 1, in time n O(d) and sample complexity T =??(s ?? d log n), returns x such that x ??? arg min f (x).This theorem, and indeed most of the results in this paper, follows from the main recovery properties of Procedure 2.

This recovery procedure satisfies the following main lemma.

See its proof in Section A.1.

Lemma 7 (Noisy recovery).

Let {?? S } be a 1-bounded orthonormal polynomial basis for distribution DISPLAYFORM0 DISPLAYFORM1 4: Let S 1 , ..., S s be the indices of the largest coefficients of ??.

DISPLAYFORM2 Remark: Note that the above Lemma also holds in the adversarial or agnostic noise setting.

That is, an adversary could add a noise vector v to the labels received by the learner.

In this case, the learner will see label vector y = Ax + e + v. If v 2 ??? ??? ??m, then we will recover a polynomial with squared-error at most ?? + O(??) via re-scaling ?? by a constant factor and applying the triangle inequality to e + v 2 .While this noisy recovery lemma is the basis for our enhanced algorithm in the next section as well as the learning-theoretic result on learning of decision trees detailed in the next subsection, it does not imply recovery of the global optimum.

The reason is that noisy recovery guarantees that we output a hypothesis close to the underlying function, but even a single noisy point can completely change the optimum.

Nevertheless, we can use our techniques to prove recovery of optimality for functions that are computed exactly by a sparse, low-degree polynomial (Theorem 6).

See the proof in Section A.2.

Lemma 7 has important applications for learning (in the PAC model BID39 ) well-studied function classes with respect to the uniform distribution on {???1, 1} n3 .

For example, we obtain the first quasi-polynomial time algorithm for learning decision trees with respect to the uniform distribution on {???1, 1} n with polynomial sample complexity:Corollary 8.

Let X = {???1, 1} n and let C be the class of all decision trees of size s on n variables.

Then C is learnable with respect to the uniform distribution in time n O(log(s/??)) and sample complexity m =??(s 2 /?? ?? log n).

Further, if the labels are corrupted by arbitrary noise vector v such that v 2 ??? ??? ??m, then the output classifier will have squared-error at most ?? + O(??).See the proof of Corollary 8 in Section A.3.Comparison with the "Low-Degree" Algorithm.

Prior work for learning decision trees (more generally Boolean functions that are approximated by low-degree polynomials) used the celebrated "low-degree" algorithm of BID23 .

Their algorithm uses random sampling to estimate each low-degree Fourier coefficient to high accuracy.

In contrast, our approach is to use algorithms for compressed sensing to estimate the coefficients.

Tools for compressed sensing take advantage of the incoherence of the design matrix and give improved results that seem unattainable from the "low-degree" algorithm.

For learning noiseless, Boolean decision trees, the low-degree algorithm uses quasipolynomial time and sample complexity??(s 2 /?? 2 ??log n) to learn to accuracy ??.

It is not clear, however, how to obtain any noise tolerance from their approach.

For general real-valued decision trees where B is an upper bound on the maximum value at any leaf of a size s tree, our algorithm will succeed with sample complexity??(B 2 s 2 /?? ?? log n) and be tolerant to noise while the low-degree algorithm will use??(B 4 s 2 /?? 2 ?? log n) (and will have no noise tolerance properties).

Note our improvement in the dependence on ?? (even in the noiseless setting), which is a consequence of the RIP property of the random orthonormal family.4 HARMONICA: THE FULL ALGORITHM Rather than applying Algorithm 1 directly, we found that performance is greatly enhanced by iteratively using Procedure 2 to estimate the most influential hyperparameters and their optimal values.

In the rest of this section we describe this iterative heuristic, which essentially runs Algorithm 1 for multiple stages.

More concretely, we continue to invoke the PSR subroutine until the search space becomes small enough for us to use a "base" hyperparameter optimizer (in our case either SH or Random Search).The space of minimizing assignments to a multivariate polynomial is a highly non-convex set that may contain many distinct points.

As such, we take an average of several of the best minimizers (of subsets of hyperparameters) during each stage.

In order to describe this formally we need the following definition of a restriction of function: DISPLAYFORM0 , and z ??? {???1, 1} J be given.

We call (J, z) a restriction pair of function f .

We denote f J,z the function over n ??? |J| variables given by setting the variables of J to z.

We can now describe our main algorithm (Algorithm 3).

Here q is the number of stages for which we apply the PSR subroutine, and the restriction size t serves as a tie-breaking rule for the best minimizers (which can be set to 1).Algorithm 3 Harmonica-q 1: Input: oracle for f , number of samples T , sparsity s, degree d, regularization parameter ??, number of stages q, restriction size t, base hyperparameter optimizer ALG.

2: for stage i = 1 to q do 3:Invoke PSR(f, T, s, d, ??) (Procedure 2) to obtain (g i , J i ), where g i is a function defined on variables specified by index set J i ???

[n].

Let M i = {x 1 , ..., x t } = arg min g i (x) be the best t minimizers of g i .

5: DISPLAYFORM0

We compare Harmonica 5 with Spearmint 6 BID32 , Hyperband, SH 7 and Random Search.

Both Spearmint and Hyperband are state-of-the-art algorithms, and it is observed that Random Search 2x (Random Search with doubled function evaluation resources) is a very competitive benchmark that beats many algorithms 8 .Our first experiment is over training residual network on Cifar-10 dataset 9 .

We included 39 binary hyperparameters, including initialization, optimization method, learning rate schedule, momentum 5 A python implementation of Harmonica can be found at https://github.com/callowbird/ Harmonica 6 https://github.com/HIPS/Spearmint.git 7 We implemented a parallel version of Hyperband and SH in Lua.

8 E.g., see BID30 b) .

9 https://github.com/facebook/fb.resnet.torch Table 1 (Section C.1) details the hyperparameters considered.

We also include 21 dummy variables to make the task more challenging.

Notice that Hyperband, SH, and Random Search are agnostic to the dummy variables in the sense that they just set the value of dummy variables randomly, therefore select essentially the same set of configurations with or without the dummy variables.

Only Harmonica and Spearmint are sensitive to the dummy variables as they try to learn the high dimensional function space.

To make a fair comparison, we run Spearmint without any dummy variables.

As most hyperparameters have a consistent effect as the network becomes deeper, a common handtuning strategy is "tune on small network, then apply the knowledge to big network" (See discussion in Section C.3).

Harmonica can also exploit this strategy as it selects important features stageby-stage.

More specifically, during the feature selection stages, we run Harmonica for tuning an 8 layer neural network with 30 training epochs.

At each stage, we take 300 samples to extract 5 important features, and set restriction size t = 4 (see Procedure 2).

After that, we fix all the important features, and run the SH or Random Search as our base algorithm on the big 56 layer neural network for training the whole 160 epochs 10 .

To clarify, "stage" means the stages of the hyperparameter algorithms, while "epoch" means the epochs for training the neural network.

We tried three versions of Harmonica for this experiment, Harmonica with 1 stage (Harmonica-1), 2 stages (Harmonica-2) and 3 stages (Harmonica-3).

All of them use SH as the base algorithm.

The top 10 test error results and running times of the different algorithms are depicted in Figure 2 .

SH based algorithms may return fewer than 10 results.

For more runs of variants of Harmonica and its resulting test error, see Figure 3 (the results are similar to Figure 2 ).Test error and scalability: Harmonica-1 uses less than 1/5 time of Spearmint, 1/7 time of Hyperband and 1/8 time compared with Random Search, but gets better results than the competing algorithms.

It beats the Random Search 8x benchmark (stronger than Random Search 2x benchmark of BID22 ).

Harmonica-2 uses slightly more time, but is able to find better results.

10 Other algorithms like Spearmint, Hyperband, etc. can be used as the base algorithms as well.

Improving upon human-tuned parameters: Harmonica-3 obtains a better test error (6.58%) as compared to the best hand-tuning rate 6.97% reported in BID15 11 .

Harmonica-3 uses only 6.1 GPU days, which is less than half day in our environment, as we have 20 GPUs running in parallel.

Notice that we did not cherry pick the results for Harmonica-3.

In Section 5.3 we show by running Harmonica-3 for longer time, one can obtain a few other solutions better than hand tuning.

Performance of provable methods: Harmonica-1 has noiseless and noisy recovery guarantees (Lemma 7), which are validated experimentally.

We computed the average test error among 300 random samples for an 8 layer network with 30 epochs after each stage.

See Figure 4 in Appendix.

After selecting 5 features in stage 1, the average test error drops from 60.16 to 33.3, which indicates the top 5 features are very important.

As we proceed to stage 3, the improvement on test error becomes less significant as the selected features at stage 3 have mild contributions.

To be clear, Harmonica itself has six hyperparameters that one needs to set including the number of stages, 1 regularizer for Lasso, the number of features selected per stage, base algorithm, small network configuration, and the number of samples per stage.

Note, however, that we have reduced the search space of general hyperparameter optimization down to a set of only six hyperparameters.

Empirically, our algorithm is robust to different settings of these parameters, and we did not even attempt to tune some of them (e.g., small network configuration).Base algorithm and #stages.

We tried different versions of Harmonica, including Harmonica with 1 stage, 2 stages and 3 stages using SH as the base algorithm (Harmonica-1, Harmonica-2, Harmonica-3), with 1 stage and 2 stages using Random Search as the base algorithm (Harmonica-1-Random-Search, Harmonica-2-Random-Search), and with 2 stages and 3 stages running SH as the base for longer time (Harmonica-2-Long, Harmonica-3-Long).

As can be seen in Figure 3 , most variants produce better results than SH and use less running time.

Moreover, if we run SH for longer time, we will obtain more stable solutions with less variance in test error.

Lasso parameters are stable.

See TAB3 in Appendix for stable range for regularization term ?? and the number of samples.

Here stable range means as long as the parameters are set in this range, the top 5 features and the signs of their weights (which are what we need for computing g(x) in Procedure 2) do not change.

In other words, the feature selection outcome is not affected.

When parameters are outside the stable ranges, usually the top features are still unchanged, and we miss only one or two out of the five features.

On the degree of features.

We set degree to be three because it does not find any important features with degree larger than this.

Since Lasso can be solved efficiently (less than 5 minutes in our experiments), the choice of degree can be decided automatically.

Our second experiment considers a synthetic hierarchically bounded function h(x).

In this experiment, we showed that the optimization time of Harmonica is significantly faster than Spearmint, and the estimation error of Harmonica is linear in the noise level of the function.

See Section C.4 for details.

A MISSING PROOFS A.1 PROOF OF LEMMA 7Recall the Chebyshev inequality: Fact 10 (Multidimensional Chebyshev inequality).

Let X be an m dimensional random vector, with expected value ?? = E[X], and covariance matrix DISPLAYFORM0 If V is a positive definite matrix, for any real number ?? > 0: DISPLAYFORM1 For ease of notation we assume K = 1.

Let f be an (??/4, s, d)-bounded function written in the orthonormal basis as Sf (S)?? S .

We can equivalently write f as f = h + g, where h is a degree d polynomial that only includes coefficients of magnitude at least ??/4s and the constant term of the polynomial expansion of f .Since L 1 (f ) = S |f S | ??? s, by Fact 4 we have that h is 4s 2 /?? + 1 sparse.

The function g is thus the sum of the remainingf (S)?? S terms not included in h.

Draw m (to be chosen later) random labeled examples {(z 1 , y 1 ), . . .

, (z m , y m )} and enumerate all N = n d basis functions ?? S for |S| ??? d as {?? 1 , . . .

, ?? N }.

Form matrix A such that A ij = ?? j (z i ) and consider the problem of recovering 4s 2 /?? + 1 sparse x given Ax + e = y where x is the vector of coefficients of h, the ith entry of y equals y i , and e i = g(z i ).We will prove that with constant probability over the choice m random examples, e 2 ??? ??? ??m.

Applying Theorem 5 by setting ?? = ??? ?? and observing that ?? 4s 2 /??+1 (x) 1 = 0, we will recover x such that x ??? x 2 2 ??? c 2 2 ?? for some constant c 2 .

As such, for the functionf DISPLAYFORM2 ?? by Parseval's identity.

Note, however, that we may rescale ?? by constant factor 1/(2c 2 2 ) to obtain error ??/2 and only incur an additional constant (multiplicative) factor in the sample complexity bound.

By the definition of g, we have DISPLAYFORM3 where eachf (R) is of magnitude at most ??/4s.

By Fact 4 and Parseval's identity we have DISPLAYFORM4 It remains to bound e 2 .

Note that since the examples are chosen independently, the entries e i = g(z i ) are independent random variables.

Since g is a linear combination of orthonormal monomials (not including the constant term), we have E z???D [g(z)] = 0.

Here we can apply linearity of variance (the covariance of ?? i and ?? j is zero for all i = j) and calculate the variance DISPLAYFORM5 With the same calculation as (3), we know Var(g(z i )) is at most ??/2.

Now consider the covariance matrix V of the vector e which equals E[ee ] (recall every entry of e has mean 0).

Then V is a diagonal matrix (covariance between two independent samples is zero), and every diagonal entry is at most ??/2.

Applying Fact 10 we have DISPLAYFORM6 Hence with probability at least 1/2, we have that e 2 ??? ??? ??m.

From Theorem 5, we may choose m =??(s 2 /?? ?? log n d ).

This completes the proof.

Note that the probability 1/2 above can be boosted to any constant probability with a constant factor loss in sample complexity.

There are at most N = n d polynomials ?? S with |S| ??? d. Let the enumeration of these polynomials be ?? 1 , . . .

, ?? N .

Draw m labeled examples {(z 1 , y 1 ), . . . , (z m , y m )} independently from D and construct an m ?? N matrix A with A ij = ?? j (z i ).

Since f can be written as an s sparse linear combination of ?? 1 , . . .

, ?? N , there exists an s-sparse vector x such that Ax = y where the ith entry of y is y i .

Hence we can apply Theorem 5 to recover x exactly.

These are the s non-zero coefficients of f 's expansion in terms of {?? S }.

Since f is recovered exactly, its minimizer is found in the optimization step.

A.3 PROOF OF COROLLARY 8As mentioned earlier, the orthonormal polynomial basis for the class of Boolean functions with respect to the uniform distribution on {???1, 1} n is the class of parity functions {?? S } for S ??? {???1, 1} n .

Further, it is easy to show that for Boolean function DISPLAYFORM0 The corollary now follows by applying Lemma 7 and two known structural facts about decision trees: 1) a tree of size s is (??, log(s/??))-concentrated and has L 1 norm bounded by s (see e.g., BID26 ) and 2) by Fact 4, for any function f with L 1 norm bounded by s (i.e., a decision tree of size s), there exists an s 2 /?? sparse function g such that E[(f ??? g) 2 ] ??? ??.

The noise tolerance property follows immediately from the remark after the proof of Lemma 7.

Scalability.

If the hidden function if s-sparse, Harmonica can find such a sparse function using O(s log s) samples.

If at every stage of Harmonica, the target function can be approximated by an s sparse function, we only need??(qs log s) samples where q is the number of stages.

For real world applications such as deep neural network hyperparameter tuning, it seems (empirically) reasonable to assume that the hidden function is indeed sparse at every stage (see Section 5).For Hyperband BID22 , SH or Random Search, even if the function is s-sparse, in order to cover the optimal configuration by random sampling, we need ???(2 s ) samples.

Optimization time.

Harmonica runs the Lasso BID38 algorithm after each stage to solve (2), which is a well studied convex optimization problem and has very fast implementations.

Hyperband and SH are also efficient in terms of running time as a function of the number of function evaluations, and require sorting or other simple computations.

The running time of Bayesian optimization is cubic in number of function evaluations, which limits applicability for large number of evaluations / high dimensionality, as we shall see in Section C.4.Parallelizability.

Harmonica, similar to Hyperband, SH, and Random Search, has straightforward parallel implementations.

In every stage of those algorithms, we could simply evaluate the objective functions over randomly chosen points in parallel.

It is hard to run Bayesian optimization algorithm in parallel due to its inherent serial nature.

Previous works explored variants in which multiple points are evaluated at the same time in parallel BID41 , though speed ups do not grow linearly in the number of machines, and the batch size is usually limited to a small number.

Feature Extraction.

Harmonica is able to extract important features with weights in each stages, which automatically sorts all the features according to their importance.

See Section C.2.11 In order to evaluate fi, we first sample k ??? [t] to obtain f J i ,x * k , and then evaluate DISPLAYFORM0 38.

nthreads (Detail 1) 8, 4, 2, or 1?

39-60.

Dummy variables Just dummy variables, no effect at all.

See Table 1 for the specific hyperparameter options that we use in Section 5.

For those variables with k options (k > 2), we use log k binary variables under the same name to represent them.

For example, we have two variables (01, 02) and their binary representation to denote four kinds of possible initializations: Xavier Glorot , Kaiming , 1/n, or 1/n 2 .

We show the selected important features and their weights during the first 3 stages in TAB1 , where each feature is a monomial of variables with degree at most 3.

We do not include the 4th stage because in that stage there are no features with nonzero weights.

Smart choices on important options.

Based on BID18 .No dummy/irrelevant variables selected.

Although there are 21/60 dummy variables, we never select any of them.

Moreover, the irrelevant variables like cudnn, backend, nthreads, which do not affect the test error, were not selected.

Harmonica, n=100 Harmonica, n=200Figure 5: Optimization time comparison

In our experiments, Harmonica first runs on a small network to extract important features and then uses these features to do fine tuning on a big network.

Since Harmonica finds significantly better solutions, it is natural to ask whether other algorithms can also exploit this strategy to improve performance.

Unfortunately, it seems that all the other algorithms do not naturally support feature extraction from a small network.

For Bayesian Optimization techniques, small networks and large networks have different optimization spaces.

Therefore without some modification, Spearmint cannot use information from the small network to update the prior distribution for the large network.

Random-search-based techniques are able to find configurations with low test error on the small network, which might be good candidates for the large network.

However, based on our simulation, good configurations of hyperparameters from random search do not generalize from small networks to large networks.

This is in contrast to important features in our (Fourier) space, which do seem to generalize.

To test the latter observation using Cifar-10 dataset, we first spent 7 GPU days on 8 layer network to find top 10 configurations among 300 random selected configurations.

Then we apply these 10 configurations, as well as 90 locally perturbed configurations (each of them is obtained by switching one random option from one top-10 configuration), so in total 100 "promising" configurations, to the large 56 layer network.

This simulation takes 27 GPU days, but the best test error we obtained is only 11.1%, even worse than purely random search.

Since Hyperband is essentially a fast version of Random Search, it also does not support feature extraction.

Hence, being able to extract important features from small networks seems empirically to be a unique feature of Harmonica.

Our second experiment considers a synthetic hierarchically bounded function h(x).

We run Harmonica with 100 samples, 5 features selected per stage, for 3 stages, using degree 3 features.

See Figure 5 for optimization time comparison.

We only plot the optimization time for Spearmint when n = 60, which takes more than one day for 500 samples.

Harmonica is several magnitudes faster than Spearmint.

In FIG3 , we show that Harmonica is able to estimate the hidden function with error proportional to the noise level.

The synthetic function h(x) ??? {???1, +1} n ??? R is defined as follows.

h(x) has three stages, and in i-th stage (i = 0, 1, 2), it has 32 i sparse vectors s i,j for j = 0, ?? ?? ?? , 32 i ??? 1.

Each s i,j contains 5 pairs of weight w contains 5 binaries and represents a integer in [0, 31] , denoted as c i,j (x).

Let h(x) = s 1,1 (x) + s 2,c1,1(x) (x)+s 3,c1,1(x) * 32+c 2,c 1,1 (x) (x) (x)+??, where ?? is the noise uniformly sampled from [???A, A] (A is the noise level).

In other words, in every stage i we will get a sparse vector s i,j .

Based on s i,j (x), we pick a the next sparse function and proceed to the next stage.

<|TLDR|>

@highlight

A hyperparameter tuning algorithm using discrete Fourier analysis and compressed sensing

@highlight

Investigates problem of optimizing hyperparameters under the assumption that the unknown function can be approximated, showing that the approximate minimization can be performed over the boolean hypercube.

@highlight

The paper explores hyperparameter optimization by assuming structure in the unknown function mapping hyperparameters to classification accuracy
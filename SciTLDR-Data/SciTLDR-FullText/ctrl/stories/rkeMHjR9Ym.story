We study discrete time dynamical systems governed by the state equation $h_{t+1}=ϕ(Ah_t+Bu_t)$. Here A,B are weight matrices, ϕ is an activation function, and $u_t$ is the input data.

This relation is the backbone of recurrent neural networks (e.g. LSTMs) which have broad applications in sequential learning tasks.

We utilize stochastic gradient descent to learn the weight matrices from a finite input/state trajectory $(u_t,h_t)_{t=0}^N$. We prove that SGD estimate linearly converges to the ground truth weights while using near-optimal sample size.

Our results apply to increasing activations whose derivatives are bounded away from zero.

The analysis is based on i) an SGD convergence result with nonlinear activations and ii) careful statistical characterization of the state vector.

Numerical experiments verify the fast convergence of SGD on ReLU and leaky ReLU in consistence with our theory.

A wide range of problems involve sequential data with a natural temporal ordering.

Examples include natural language processing, time series prediction, system identification, and control design, among others.

State-of-the-art algorithms for sequential problems often stem from dynamical systems theory and are tailored to learn from temporally dependent data.

Linear models and algorithms; such as Kalman filter, PID controller, and linear dynamical systems, have a long history and are utilized in control theory since 1960's with great success (Brown et al. (1992) ; Ho & Kalman (1966) ; Åström & Hägglund (1995) ).

More recently, nonlinear models such as recurrent neural networks (RNN) found applications in complex tasks such as machine translation and speech recognition (Bahdanau et al. (2014) ; Graves et al. (2013) ; Hochreiter & Schmidhuber (1997) ).

Unlike feedforward neural networks, RNNs are dynamical systems that use their internal state to process inputs.

The goal of this work is to shed light on the inner workings of RNNs from a theoretical point of view.

In particular, we focus on the RNN state equation which is characterized by a nonlinear activation function φ, state weight matrix A, and input weight matrix B as follows h t+1 = φ(Ah t + Bu t ),(1.1)Here h t is the state vector and u t is the input data at timestamp t.

This equation is the source of dynamic behavior of RNNs and distinguishes RNN from feedforward networks.

The weight matrices A and B govern the dynamics of the state equation and are inferred from data.

We will explore the statistical and computational efficiency of stochastic gradient descent (SGD) for learning these weight matrices.

Contributions: Suppose we are given a finite trajectory of input/state pairs (u t , h t ) N t=0 generated from the state equation (1.1).

We consider a least-squares regression obtained from N equations; with inputs (u t , h t ) N t=1 and outputs (h t+1 ) N t=1 .

For a class of activation functions including leaky ReLU and for stable systems 1 , we show that SGD linearly converges to the ground truth weight matrices while requiring near-optimal trajectory length N .

In particular, the required sample size is O(n + p) where n and p are the dimensions of the state and input vectors respectively.

The results are extended to unstable systems when the samples are collected from multiple independent RNN trajectories rather than a single trajectory.

Our theory applies to increasing activation functions whose derivatives are bounded away from zero, which includes leaky ReLU, and Gaussian input data.

Numerical experiments on ReLU and leaky ReLU corroborate our theory and demonstrate that SGD converges faster as the activation slope increases.

To obtain our results, we i) characterize the statistical properties of the state vector (e.g. well-conditioned covariance) and ii) derive a novel SGD convergence result with nonlinear activations; which may be of independent interest.

As a whole, this paper provides a step towards foundational understanding of RNN training via SGD.

Our work is related to the recent optimization and statistics literature on linear dynamical systems (LDS) and neural networks.

The state-equation (1.1) reduces to a LDS when φ is the linear activation (φ(x) = x).

Identifying the weight matrices is a core problem in linear system identification and is related to the optimal control problem (e.g. linear quadratic regulator) with unknown system dynamics.

While these problems are studied since 1950's BID3 BID2 ; Åström & Eykhoff (1971) ), our work is closer to the recent literature that provides data dependent bounds and characterize the non-asymptotic learning performance.

Recht and coauthors have a series of papers exploring optimal control problem BID14 ; BID14 BID15 ).

In particular, Hardt et al. (2016) shows gradient descent learns single-input-single-output (SISO) LDS with polynomial guarantees.

BID7 and Faradonbeh et al. (2018) provide sample complexity bounds for learning LDS.

BID12 a) ; BID10 study the identification of sparse systems.

BID6 showed that stable RNNs can be approximated by feed-forward networks.

We first introduce the notation.

· returns the spectral norm of a matrix and s min (·) returns the minimum singular value.

The activation φ : R → R applies entry-wise if its input is a vector.

Throughout, φ is assumed to be a 1-Lipschitz function.

With proper scaling of its parameters, the system (1.1) with a Lipschitz activation can be transformed into a system with 1-Lipschitz activation.

The functions Σ[·] and var [·] return the covariance of a random vector and variance of a random variable respectively.

I n is the identity matrix of size n × n. Normal distribution with mean µ and covariance Σ is denoted by N (µ, Σ).

Throughout, c, C, c 0 , c 1 , . . .

denote positive absolute constants.

We consider the dynamical system parametrized by an activation function φ(·) and weight matrices A ∈ R n×n , B ∈ R n×p as described in (1.1).

Here, h t is the n dimensional state-vector and u t is the p dimensional input to the system at time t. As mentioned previously, (1.1) corresponds to the state equation of a recurrent neural network.

For most RNNs of interest, the state h t is hidden and we only get to interact with h t via an additional output equation.

For Elman networks Elman (1990) , this equation is characterized by some output activation φ y and output weights C, D as follows DISPLAYFORM0 DISPLAYFORM1 Pick γ τ from {1, 2, . . .

, N } uniformly at random.

7: DISPLAYFORM2 (A, B).

Our goal is learning the unknown weights A and B in a data and computationally efficient way.

In essence, we will show that, if the trajectory length satisfies N n + p, SGD can quickly and provably accomplish this goal using a constant step size.

Appoach: Our approach is described in Algorithm 1.

It takes two hyperparameters; the scaling factor µ and learning rate η.

Using the RNN trajectory, we construct N triples of the form {u t , h t , h t+1 } N t=1 .

We formulate a regression problem by defining the output vector y t , input vector x t , and the target parameter C as follows DISPLAYFORM3 With this reparameterization, we find the input/output identity y t = φ(Cx t ).

We will consider the least-squares regression given by DISPLAYFORM4 For learning the ground truth parameter C, we utilize SGD on the loss function (2.3) with a constant learning rate η.

Starting from an initial point Θ 0 , after END SGD iterations, Algrorithm 1 returns an estimateĈ = Θ END .

Estimates of A and B are decoded from the left and right submatrices ofĈ respectively.

The analysis of the state equation naturally depends on the choice of the activation function; which is the source of nonlinearity.

We first define a class of Lipschitz and increasing activation functions.

Definition 3.1 (β-increasing activation).

Given 1 ≥ β ≥ 0, the activation function φ satisfies φ(0) = 0 and 1 ≥ φ (x) ≥ β for all x ∈ R.Our results will apply to strictly increasing activations where φ is β-increasing for some β > 0.Observe that, this excludes ReLU activation which has zero derivative for negative values.

However, it includes Leaky ReLU which is a generalization of ReLU.

Parameterized by 1 ≥ β ≥ 0, Leaky ReLU is a β-increasing function given by DISPLAYFORM0 In general, given an increasing and 1-Lipschitz activation φ, a β-increasing function φ β can be obtained by blending φ with the linear activation, i.e. φ β (x) = (1 − β)φ(x) + βx.

A critical property that enables SGD is that the state-vector covariance Σ[h t ] is well-conditioned under proper assumptions.

The lemma below provides upper and lower bounds on this covariance matrix in terms of problem variables.

DISPLAYFORM1 • Suppose φ is 1-Lipschitz and φ(0) = 0.

Then, for all t ≥ 0, DISPLAYFORM2 As a natural extension from linear dynamical systems, we will say the system is stable if A < 1 and unstable otherwise.

For activations we consider, stability implies that if the input is set to 0, state vector h t will exponentially converge to 0 i.e. the system forgets the past states quickly.

This is also the reason (B t ) t≥0 sequence converges for stable systems and diverges otherwise.

The condition number of the covariance will play a critical role in our analysis.

Using Lemma 3.2, this number can be upper bounded by ρ defined as DISPLAYFORM3 Observe that, the condition number of B appears inside the ρ term.

Our main result applies to stable systems ( A < 1) and provides a non-asymptotic convergence guarantee for SGD in terms of the upper bound on the state vector covariance.

This result characterizes the sample complexity and the rate of convergence of SGD; and also provides insights into the role of activation function and the spectral norm of A. DISPLAYFORM0 be a finite trajectory generated from the state equation DISPLAYFORM1 DISPLAYFORM2 Here the expectation is over the randomness of the SGD updates.

Sample complexity: Theorem 3.3 essentially requires N (n + p)/β 4 samples for learning.

This can be seen by unpacking (3.3) and ignoring the logarithmic L term and the condition number of B. Observe that O(n + p) growth achieves near-optimal sample size for our problem.

Each state equation (1.1) consists of n sub-equations (one for each entry of h t+1 ).

We collect N state equations to obtain a system of N n equations.

On the other hand, the total number of unknown parameters in A and B are n(n + p).

This implies Theorem 3.3 is applicable as soon as the problem is mildly overdetermined i.e. N n n(n + p).Computational complexity: Theorem 3.3 requires O(n(n + p) log 1 ε ) iterations to reach ε-neighborhood of the ground truth.

Our analysis reveals that, this rate can be accelerated if the state vector is zero-mean.

This happens for odd activation functions satisfying φ(−x) = −φ(x) (e.g. linear activation).

The result below is a corollary and requires ×n less iterations.

Theorem 3.4 (Faster learning for odd activations).

Consider the same setup provided in Theorem 3.3.

Additionally, assume that φ is an odd function.

Pick scaling µ = 1/B ∞ , learning rate η = c 0 β 2 ρ(n+p) , and consider the loss function (2.3).

With probability 1 − 4N exp(−100n) − 8L exp(−O( N Lρ 2 )), starting from an initial point Θ 0 , for all τ ≥ 0, the SGD iterations described in Algorithm 1 satisfies DISPLAYFORM3 where the expectation is over the randomness of the SGD updates.

Another aspect of the convergence rate is the dependence on β.

In terms of β, the SGD error (3.4) decays as (1 − O(β 8 )) τ .

While it is not clear how optimal is the exponent 8, numerical experiments in Section 6 demonstrate that larger β indeed results in drastically faster convergence.

We first outline our high-level proof strategy for Theorem 3.3; which brings together ideas from statistics and optimization.1.

We first show that input data is well-behaved by proving that state-vector h t has a wellconditioned covariance as discussed in Lemma 3.2 and shown in Appendix B.

The key idea is if φ is β-increasing, then the random input data u t provides sufficient excitation for the output state h t+1 .2.

Even if individual samples are well-behaved, analyzing (2.3) is still challenging due to temporal dependencies between the samples.

These dependencies prevent us from directly using statistical learning results that typically assume i.i.d.

samples.

We show that the dependency between samples at time t and t + T decay exponentially fast in separation T (for stable systems).

This is outlined in Appendix C.3.

This observation allows us to obtain nearly independent data by subsampling the original trajectory to get (h iT , u iT ) i≥0 .

Thanks to exponential decay, a logarithmically small T can be chosen to generate large subtrajectories of size N/T .

Appendix D uses additional perturbation arguments to establish the well-behavedness of the overall data matrix.4.

To conclude, we obtain a deterministic result which establishes fast convergence result for β-increasing activations and well-behaved dataset.

This is provided in Theorem 4.1 and proved in Appendix A.The first three steps are related to the statistical nature of the problem which can be decoupled from the last step.

Specifically, the last step derives a deterministic result that establishes the linear convergence of SGD for β-increasing functions.

For linear convergence proofs, a typical strategy is showing the strong convexity of the loss function i.e. showing that, for some α > 0 and all points v, u, the gradient satisfies DISPLAYFORM0 The core idea of our convergence result is that the strong convexity parameter of the loss function with β-increasing activations can be connected to the loss function with linear activations.

In particular, recalling (2.3), set y lin t = Cx t and define the linear loss to be DISPLAYFORM1 Denoting the strong convexity parameter of the original loss by α φ and that of linear loss by α lin , we argue that α φ ≥ β 2 α lin ; which allows us to establish a convergence result as soon as α lin is strictly positive.

Next result is our SGD convergence theorem which follows from this discussion.

DISPLAYFORM2 is given; where output y i is related to input x i via y i = φ( x i , θ ) for some θ ∈ R n .

Suppose β > 0 and φ is a β-increasing.

Let γ + ≥ γ − > 0 be scalars.

Assume that input samples satisfy the bounds DISPLAYFORM3 Let {r τ } ∞ τ =0 be a sequence of i.i.d.

integers uniformly distributed between 1 to N .

Then, starting from an arbitrary point θ 0 , setting learning rate η = β 2 γ− γ+B , for all τ ≥ 0, the SGD iterations for quadratic loss DISPLAYFORM4 satisfies the error bound DISPLAYFORM5 where the expectation is over the random selection of the SGD iterations DISPLAYFORM6 This theorem provides a clean convergence rate for SGD for β-increasing activations and naturally generalizes standard results on linear regression which corresponds to β = 1.

We remark that related results appear in the literature on generalized linear models.

Kakade et al. (2011); Foster et al. (2018) ; BID4 provide learning theoretic loss/gradient/hessian convergence results for isotonic regression, robust regression, and β-increasing activations.

Goel et al. FORMULA0 establishes a similar result for leaky ReLU activations under the assumption of symmetric input distribution and infinitely many samples (i.e. in population limit).

Compared to these, we establish a deterministic linear convergence guarantee for SGD that works whenever the data matrix is full rank.

We believe extensions to proximal gradient methods might be beneficial for high-dimensional nonlinear problems (e.g. sparse/low-rank approximation, manifold constraints Cai et al. FORMULA0 FORMULA0 ) and is left as a future work.

To derive our main results in Section 3, we need to address the first three steps outlined earlier and determine the conditions under which Theorem 4.1 is applicable to the data obtained from RNN state equation with high probability.

Below we provide desirable characteristics of the state vector; which enables our statistical results.

Assumption 1 (Well-behaved state vector).

Let L > 1 be an integer.

There exists positive scalars γ + , γ − , θ and an absolute constant C > 0 such that θ ≤ 3 √ n and the following holds DISPLAYFORM7 • Upper bound: for all t, the state vector satisfies DISPLAYFORM8 Here · ψ2 returns the subgaussian norm of a vector (see Def.

5.22 of Vershynin FORMULA0 ).Assumption 1 ensures that covariance is well-conditioned, state vector is well-concentrated, and it has a reasonably small expectation.

Our next theorem establishes statistical guarantees for learning the RNN state equation based on this assumption.

DISPLAYFORM9 be a length N trajectory of the state equation (1.1).

Suppose A < 1, φ is β-increasing, h 0 = 0, and u t DISPLAYFORM10 Suppose Assumption 1 holds with L, γ + , γ − , θ.

Pick scaling to be µ = 1/ √ γ + and learning rate to DISPLAYFORM11 .

With probability DISPLAYFORM12 , starting from Θ 0 , for all τ ≥ 0, the SGD iterations on loss (2.3) as described in Algorithm 1 satisfies DISPLAYFORM13 where the expectation is over the randomness of SGD updates.

The advantage of this theorem is that, it isolates the optimization problem from the statistical properties of state vector.

If one can prove tighter bounds on achievable (γ + , γ − , θ), it will immediately imply improved performance for SGD.

In particular, Theorems 3.3 and 3.4 are simple corollaries of Theorem 4.2 with proper choices.• Theorem 3.3 follows by setting DISPLAYFORM14 2 , and θ = √ n.• Theorem 3.4 follows by setting DISPLAYFORM15 2 , and θ = 0.

So far, we considered learning from a single RNN trajectory for stable systems ( A < 1).

For such systems, as the time goes on, the impact of the earlier states disappear.

In our analysis, this allows us to split a single trajectory into multiple nearly-independent trajectories.

This approach will not work for unstable systems (A is arbitrary) where the impact of older states may be amplified over time.

To address this, we consider a model where the data is sampled from multiple independent trajectories.

Suppose N independent trajectories of the state-equation (1.1) are available.

Pick some integer T 0 ≥ 1.

Denoting the ith trajectory by the triple (h DISPLAYFORM0 t ) t≥0 , we collect a single sample from each trajectory at time T 0 to obtain the triple (h DISPLAYFORM1 To utilize the existing optimization framework (2.3); for 1 ≤ i ≤ N , we set, DISPLAYFORM2 With this setup, we can again use the SGD Algorithm 1 to learn the weights A and B. The crucial difference compared to Section 3 is that, the samples DISPLAYFORM3 are now independent of each other; hence, the analysis is simplified.

As previously, having an upper bound on the condition number of the state-vector covariance is critical.

This upper bound can be shown to be ρ defined as DISPLAYFORM4 Theρ term is similar to the earlier definition (3.3); however it involves B T0 rather than B ∞ .

This modification is indeed necessary since B ∞ = ∞ when A > 1.

On the other hand, note that, B 2 T0 grows proportional to A 2T0 ; which results in exponentially bad condition number in T 0 .

Our ρ definition remedies this issue for single-output systems; where n = 1 and A is a scalar.

In particular, when β = 1 (e.g. φ is linear) ρ becomes equal to the correct value 1 2 .

The next theorem provides our result on unstable systems in terms of this condition number and other model parameters.

Theorem 5.1 (Unstable systems).

Suppose we are given N independent trajectories (h DISPLAYFORM5 where the ith sample is given by (5.1).

Suppose the sample size satisfies DISPLAYFORM6 where ρ is given by (5.2).

Assume the initial states are 0, φ is β-increasing, p ≥ n, and u t DISPLAYFORM7 ρn(n+p) , and run SGD over the equations described in (2.2) and (2.3).

Starting from Θ 0 , with probability 1 − 2N exp(−100(n + p)) DISPLAYFORM8 where the expectation is over the randomness of the SGD updates.

We conducted experiments on ReLU and Leaky ReLU activations.

Let us first describe the experimental setup.

We pick the state dimension n = 50 and the input dimension p = 100.

We choose the ground truth matrix A to be a scaled random unitary matrix; which ensures that all singular values of A are equal.

B is generated with i.i.d.

N (0, 1) entries.

Instead of using the theoretical scaling choice, we determine the scaling µ from empirical covariance matrices outlined in Algorithm 2.

Similar to our proof strategy, this algorithm equalizes the spectral norms of the input and state covariances to speed up convergence.

We also empirically determined the learning rate and used η = 1/100 in all experiments.

Algorithm 2 Empirical hyperparameter selection.

repeat the same experiments.

The difference is the spectral norm of the ground truth state matrix A.Evaluation: We consider two performance measures in the experiments.

LetĈ be an estimate of the ground truth parameter DISPLAYFORM0 .

The first measure is the normalized error defined as DISPLAYFORM1 The second measure is the normalized loss defined as DISPLAYFORM2 In all experiments, we run Algorithm 1 for 50000 SGD iterations and plot these measures as a function of τ ; by using the estimate available at the end of the τ th SGD iteration for 0 ≤ τ ≤ 50000.

Each curve is obtained by averaging the outcomes of 20 independent realizations.

Our first experiments use N = 500; which is mildly larger than the total dimension n + p = 150.

In FIG4 , we plot the Leaky ReLU errors with varying slopes as described in (3.1).

Here β = 0 corresponds to ReLU and β = 1 is the linear model.

In consistence with our theory, SGD achieves linear convergence and as β increases, the rate of convergence drastically improves 3 .

The improvement is more visible for less stable systems driven by A with a larger spectral norm.

In particular, while ReLU converges for small A , SGD gets stuck before reaching the ground truth when A = 0.8.To understand, how well SGD fits the training data, in FIG1 , we plotted the normalized loss for ReLU activation.

For more unstable system ( A = 0.9), training loss stagnates in a similar fashion to the parameter error.

We also verified that the norm of the overall gradient ∇L(Θ τ ) F continues to decay (where Θ τ is the τ th SGD iterate); which implies that SGD converges before reaching a global minima.

As A becomes more stable, rate of convergence improves and linear rate is visible.

Finally, to better understand the population landscape of the quadratic loss with ReLU activations, FIG1 repeats the same ReLU experiments while increasing the sample size five times to N = 2500.

For this more overdetermined problem, SGD converges even for A = 0.9; indicating that• population landscape of loss with ReLU activation is well-behaved,• however ReLU problem requires more data compared to the Leaky ReLU for finding global minima.

Overall, as predicted by our theory, experiments verify that SGD indeed quickly finds the optimal weight matrices of the state equation (1.1) and as the activation slope β increases, the convergence rate improves.

The difference is that a) uses N = 500 trajectory length whereas b) uses N = 2500 (i.e. ×5 more data).

Shaded regions highlight the one standard deviation around the mean.

This work showed that SGD can learn the nonlinear dynamical system (1.1); which is characterized by weight matrices and an activation function.

This problem is of interest for recurrent neural networks as well as nonlinear system identification.

We showed that efficient learning is possible with optimal sample complexity and good computational performance.

Our results apply to strictly increasing activations such as Leaky ReLU.

We empirically showed that Leaky ReLU converges faster than ReLU and requires less samples; in consistence with our theory.

We list a few unanswered problems that would provide further insights into recurrent neural networks.• Covariance of the state-vector: Our results depend on the covariance of the state-vector and requires it to be positive definite.

One might be able to improve the current bounds on the condition number and relax the assumptions on the activation function.

Deriving similar performance bounds for ReLU is particularly interesting.• Hidden state: For RNNs, the state vector is hidden and is observed through an additional equation (2.1); which further complicates the optimization landscape.

Even for linear dynamical systems, learning the (A, B, C, D) system ((1.1), (2.1)) is a non-trivial task Ho & Kalman (1966); Hardt et al. (2016) .

What can be said when we add the nonlinear activations?• Classification task: In this work, we used normally distributed input and least-squares regression for our theoretical guarantees.

More realistic input distributions might provide better insight into contemporary problems, such as natural language processing; where the goal is closer to classification (e.g. finding the best translation from another language).

Proof of Theorem 4.1.

Given two distinct scalars a, b; DISPLAYFORM0 .

φ (a, b) ≥ β since φ is β-increasing.

Define wτ to be the residual wτ = θτ − θ.

Observing DISPLAYFORM1 Since φ is 1-Lipschitz and β-increasing, Gr τ is a positivesemidefinite matrix satisfying DISPLAYFORM2 Bxr τ x T rτ .

Consequently, we find the following bounds in expectation DISPLAYFORM3 Observe that (A.1) essentially lower bounds the strong convexity parameter of the problem with β 2 γ−; which is the strong convexity of the identical problem with the linear activation (i.e. β = 1).

However, we only consider strong convexity around the ground truth parameter θ i.e. we restricted our attention to (θ, θτ ) pairs.

With this, wτ+1 can be controlled as, DISPLAYFORM4 , we find the advertised bound DISPLAYFORM5 Applying induction over the iterations τ , we find the advertised bound (4.2) DISPLAYFORM6 Ni and form the concatenated matrix DISPLAYFORM7 . . .

DISPLAYFORM8 .

Denote ith row of X by xi.

Then, for DISPLAYFORM9 Proof.

The bound on the rows xi 2 directly follows by assumption.

For the remaining result, first observe that DISPLAYFORM10 Combining these two yields the desired upper/lower bounds on X T X/N .

This section characterizes the properties of the state vector ht when input sequence is normally distributed.

These bounds will be crucial for obtaining upper and lower bounds for the singular values of the data matrix X = [x1 . . .

xN ] T described in (2.2).

For probabilistic arguments, we will use the properties of subgaussian random variables.

Orlicz norm provides a general framework that subsumes subgaussianity.

Definition B.1 (Orlicz norms).

For a scalar random variable Orlicz-a norm is defined as DISPLAYFORM0 Orlicz-a norm of a vector x ∈ R p is defined as x ψa = sup v∈B p v T x ψa where B p is the unit 2 ball.

The subexponential norm is the Orlicz-1 norm · ψ 1 and the subgaussian norm is the Orlicz-2 norm · ψ 2 .

Consider the state equation (1.1).

Suppose activation φ is 1-Lipschitz.

Observe that ht+1 is a deterministic function of the input sequence {uτ } t τ =0 .

Fixing all vectors {ui} i =τ (i.e. all except uτ ), ht+1 is A t−τ B Lipschitz function of uτ for 0 ≤ τ ≤ t.

Proof.

Fixing {ui} i =τ , denote ht+1 as a function of uτ by ht+1(uτ ).

Given a pair of vectors uτ , u τ using 1-Lipschitzness of φ, for any t > τ , we have DISPLAYFORM0 Proceeding with this recursion until t = τ , we find DISPLAYFORM1 This bound implies ht+1(uτ )

is A t−τ B Lipschitz function of uτ .

T ∈ R tp .•

There exists an absolute constant c > 0 such that ht − E[ht] ψ 2 ≤ cBt and Σ[ht] B 2 t In.• ht satisfies DISPLAYFORM2 Also, there exists an absolute constant c > 0 such that for any m ≥ n, with probability 1 − 2 exp(−100m), ht 2 ≤ c √ mBt.

Observe that ht is a deterministic function of qt i.e. ht = f (qt) for some function f .

To bound Lipschitz constant of f , for all (deterministic) vector pairs qt andqt, we find a scalar L f satisfying, DISPLAYFORM0 Define the vectors, {ai} t i=0 , as follows DISPLAYFORM1 Observing that a0 = qt, at =qt, we write the telescopic sum, DISPLAYFORM2 Focusing on the individual terms f (ai+1) − f (ai), observe that the only difference is the ui,ûi terms.

Viewing ht as a function of ui and applying Lemma B.2, DISPLAYFORM3 To bound the sum, we apply the Cauchy-Schwarz inequality; which yields DISPLAYFORM4 2)The final line achieves the inequality (B.1) with L f = Bt hence ht is Bt Lipschitz function of qt.ii) Bounding subgaussian norm: When ut DISPLAYFORM5 ∼ N (0, Ip), the vector qt is distributed as N (0, Itp).

Since ht a Bt Lipschitz function of qt, for any fixed unit length vector v, αv := v T ht = v T f (qt) is still Bt-Lipschitz function of qt.

Hence, using Gaussian concentration of Lipschitz functions, αv satisfies DISPLAYFORM6 ). ].

Since φ is 1-Lipschitz and φ(0) = 0, we have the deterministic relation ht+1 2 ≤ Aht + But 2 .

Taking squares of both sides, expanding the right hand side, and using the independence of ht, ut and the covariance information of ut, we obtain DISPLAYFORM0 Now that the recursion is established, expanding ht on the right hand side until h0 = 0, we obtain DISPLAYFORM1

.

Finally, using the fact that ht is Bt-Lipschitz function and utilizing Gaussian concentration of qt ∼ N (0, Itp), we find DISPLAYFORM0 ).Setting t = (c − 1) √ mBt for sufficiently large c > 0, we find P( Proof.

We will inductively show that {ht} t≥0 has a symmetric distribution around 0.

Suppose the vector ht satisfies this assumption.

Let S ⊂ R n be a set.

We will argue that P(ht+1 ⊂ S) = P(ht+1 ⊂ −S).

Since φ is strictly increasing, it is bijective on vectors, and we can define the unique inverse set S = φ −1 (S).

Also since φ is odd, φ(−S ) = −S. Since ht, ut are independent and symmetric, we reach the desired conclusion as follows DISPLAYFORM1 DISPLAYFORM2 Theorem B.5 (State-vector lower bound).

Consider the nonlinear state equation (1.1) with {ut} t≥0 DISPLAYFORM3 Suppose φ is a β-increasing function for some constant β > 0.

For any t ≥ 1, the state vector obeys DISPLAYFORM4 Proof.

The proof is an application of Lemma B.7.

The main idea is to write ht as sum of two independent vectors, one of which has independent entries.

Consider a multivariate Gaussian vector g ∼ N (0, Σ).

g is statistically identical to g1 + g2 where g1 ∼ N (0, smin(Σ)I d ) and g2 ∼ N (0, Σ − smin(Σ)I d ) are independent multivariate Gaussians.

Since But ∼ N (0, BB T ), setting Σ = BB T and smin = smin(Σ), we have that But ∼ g1 + g2 where g1, g2 are independent and g1 ∼ N (0, sminIn) and g2 ∼ N (0, Σ − sminIn).

Consequently, we may write DISPLAYFORM5 For lower bound, the crucial component will be the g1 term; which has i.i.d.

entries.

Applying Lemma B.7 by setting x = g1 and y = g2 + Aht, and using the fact that ht, g1, g2 are all independent of each other, we find the advertised bound, for all t ≥ 0, via DISPLAYFORM6 The next theorem applies to multiple-input-single-output (MISO) systems where A is a scalar and B is a row vector.

The goal is refining the lower bound of Theorem B.5.

Theorem B.6 (MISO lower bound).

Consider the setup of Theorem B.5 with single output i.e. n = 1.

For any t ≥ 1, the state vector obeys DISPLAYFORM7 Proof.

For any random variable X, applying Lemma B.7, we have var[φ(X)]

≥ β 2 var [X] .

Recursively, this yields DISPLAYFORM8 Expanding these inequalities till h0, we obtain the desired bound DISPLAYFORM9 Lemma B.7 (Vector lower bound).

Suppose φ is a β-increasing function.

Let x = [x1 . . .

xn] T be a vector with i.i.d.

entries distributed as xi ∼ X. Let y be a random vector independent of x. Then, DISPLAYFORM10 Proof.

We first apply law of total covariance (e.g. Lemma B.8) to simplify the problem using the following lower bound based on the independence of x and y, DISPLAYFORM11 Now, focusing on the covariance Σx[φ(x + y)], fixing a realization of y, and using the fact that x has i.i.d.

entries; φ(x + y) has independent entries as φ applies entry-wise.

This implies that Σx[φ(x + y)] is a diagonal matrix.

Consequently, its lowest eigenvalue is the minimum variance over all entries, DISPLAYFORM12 Fortunately, Lemma B.9 provides the lower bound var[φ(xi + yi)]

≥ β 2 var [X] .

Since this lower bound holds for any fixed realization of y, it still holds after taking expectation over y; which concludes the proof.

The next two lemmas are helper results for Lemma B.7 and are provided for the sake of completeness.

Lemma B.8 (Law of total covariance).

Let x, y be two random vectors and assume y has finite covariance.

Then DISPLAYFORM13 Then, applying the law of total expectation to each term, DISPLAYFORM14 Next, we can write the conditional expectation as E[E[yy DISPLAYFORM15 T .

To conclude, we obtain the covariance of E[y x] via the difference, DISPLAYFORM16 , which yields the desired bound.

Lemma B.9 (Scalar lower bound).

Suppose φ is a β-increasing function with β > 0 as defined in Definition 3.1.

Given a random variable X and a scalar y, we have DISPLAYFORM17 Proof.

Since φ is β-increasing, it is invertible and φ −1 is strictly increasing.

Additionally, DISPLAYFORM18 Using this observation and the fact that E[X] minimizes E(X − α) 2 over α, var[φ(X + y)] can be lower bounded as follows DISPLAYFORM19 Note that, the final line is the desired conclusion.

One of the challenges in analyzing dynamical systems is the fact that samples from the same trajectory have temporal dependence.

This section shows that, for stable systems, the impact of the past states decay exponentially fast and the system can be approximated by using the recent inputs only.

We first define the truncation of the state vector.

Definition C.1 (Truncated state vector).

Suppose φ(0) = 0, initial condition h0 = 0, and consider the state equation (1.1).

Given a timestamp t, L-truncation of the state vector ht is denoted byht,L and is equal to qt where qτ+1 = φ(Aqτ + Bu τ ) , q0 = 0 (C.1) is the state vector generated by the inputs u τ satisfying DISPLAYFORM0 In words, L truncated state vectorht,L is obtained by unrolling ht until time t − L and setting the contribution of the state vector ht−L to 0.

This way,ht,L depends only on the variables {uτ } t−1 τ =t−L .

The following lemma states that impact of truncation can be made fairly small for stable systems ( A < 1).

Lemma C.2 (Truncation impact -deterministic).

Consider the state vector ht and its L-truncationht,L from Definition C.1.

Suppose φ is 1-Lipschitz.

We have that DISPLAYFORM1 When t > L, we again use Definition C.1 and recall that u τ = 0 until time τ = t − L − 1.

For all t − L < τ ≤ t, using 1-Lipschitzness of φ, we have that hτ − qτ 2 = φ(Ahτ−1 + Buτ−1) − φ(Aqτ−1 + Buτ−1) 2 ≤ (Ahτ−1 + Buτ−1) − (Aqτ−1 + Buτ−1) 2 ≤ A(hτ−1 − qτ−1) 2 ≤ A hτ−1 − qτ−1 2 .

Applying this recursion between t − L < τ ≤ t and using the fact that qt−L = 0 implies the advertised result DISPLAYFORM2

We will now argue that, for stable systems, a single trajectory can be split into multiple nearly independent trajectories.

First, we describe how the sub-trajectories are constructed.

Definition C.3 (Sub-trajectory).

Let sampling rate L ≥ 1 and offset 1 ≤τ ≤ L be two integers.

LetN =Nτ be the largest integer obeying (N − 1)L +τ ≤ N .

We sample the trajectory {ht, ut} N t=0 at the points τ ,τ + L, . . .

,τ + (N − 1)L +τ and define theτ th sub-trajectory as DISPLAYFORM0 Definition C.4 (Truncated sub-trajectory).

Consider the state equation (1.1) and recall Definition C.1.

Given offsetτ and sampling rate L, for 1 ≤ i ≤N , the ith truncated sub-trajectory states are {h DISPLAYFORM1 The truncated samples are independent of each other as shown in the next lemma.

Lemma C.5.

Consider the truncated states of Definition C.4.

If (1.1) is generated by independent vectors {ut} t≥0 , for any offsetτ and sampling rate L, the vectors {h DISPLAYFORM2 are all independent of each other.

(i) only depends on the vectors {uτ } DISPLAYFORM0 which is not covered by the dependence range of (h DISPLAYFORM1 If the input is randomly generated, Lemma C.2 can be combined with a probabilistic bound on ht, to show that truncated statesh (i) are fairly close to the actual states h (i) .Lemma C.6 (Truncation impact -random).

Given offsetτ and sampling rate L, consider the state vectors of the sub-trajectory {h DISPLAYFORM2 ∼ N (0, Ip), A < 1, h0 = 0, φ is 1-Lipschitz, and φ(0) = 0.

Also suppose upper bound (4.3) of Assumption 1 holds for some θ ≤ √ n, γ+ > 0.

There exists an absolute constant c > 0 such that with probability at least 1 − 2N exp(−100n), for all 1 ≤ i ≤N , the following bound holds DISPLAYFORM3 In particular, we can always pick γ+ = B 2 ∞ (via Lemma B.3).Proof.

Using Assumption 1, we can apply Lemma F.3 on vectors {h (i−2)L+τ +1 }N i=1 .

Using a union bound, with desired probability, all vectors obey DISPLAYFORM4 for sufficiently large c. Since θ ≤ √ n, triangle inequality implies h (i−2)L+τ +1 2 ≤ c √ nγ+.

Now, applying Lemma C.2, for all 1 ≤ i ≤N , we find DISPLAYFORM5

This section utilizes the probabilistic estimates from Section B to provide bounds on the condition number of data matrices obtained from the RNN trajectory (1.1).

Following (2.2), these matrices H, U and X are defined as DISPLAYFORM0 The challenge is that, the state matrix H has dependent rows; which will be addressed by carefully splitting the trajectory {ut, ht} N t=0 into multiple sub-trajectories which are internally weakly dependent as discussed in Section C. We first define the matrices obtained from these sub-trajectories.

Definition D.1.

Given sampling rate L and offsetτ , consider the L-subsampled trajectory {h (i) , u (i) }N i=1 as described in Definitions C.3 and C.4.

Define the matricesH , the perturbed Q matrices given by, DISPLAYFORM1 DISPLAYFORM2 Proof.

This result is a direct application of Theorem F.1 after determining minimum/maximum eigenvalues of population covariance.

The cross covariance obeys E[H TŨ ] = 0 due to independence.

Also, for i > 1, the DISPLAYFORM3 In for all i and DISPLAYFORM4 , for all i > 1 DISPLAYFORM5 T .

Applying Theorem F.1 onQ and Corollary F.2 on Q, we find that, with the desired probability, DISPLAYFORM6 , the impact of the perturbation E can be bounded DISPLAYFORM7 Using the assumed bound on E , this yields DISPLAYFORM8 This final inequality is identical to the desired bound (D.3).

DISPLAYFORM9 and pick scaling µ = DISPLAYFORM10 ∼ N (0, Ip), and Assumption 1 holds with γ+, γ−, θ, L. Matrix X = [x1 . . .

xN ] T of (D.1) satisfies the following with probability 1 − 4N exp(−100n) − 8L exp(−O(N0/ρ 2 )).•

Each row of X has 2 norm at most c0 √ p + n where c0 is an absolute constant.• X T X obeys the bound DISPLAYFORM11 Proof.

The first statement on 2-norm bound can be concluded from Lemma D.4 and holds with probability 1 − 2N exp(−100(n + p)).

To show the second statement, for a fixed offset 1 ≤τ ≤ L, consider Definition D.1 and the matricesH (τ ) ,Ũ (τ ) ,X (τ ) .

Observe that X is obtained by merging multiple sub-trajectory matrices DISPLAYFORM12 .

We will first show the advertised bound for an individualX (τ ) by applying Lemma D.2 and then apply Lemma A.1 to obtain the bound on the combined matrix X.Recall thatNτ is the length of theτ th sub-trajectory i.e. number of rows ofX (τ ) .

DISPLAYFORM13 Since N0 is chosen to be large enough, applying Theorem D.2 with µ = 1/ √ γ+ choice, and noting ρ = γ+/γ−, we find that, with probability 1 − 4 exp(−c1N0/ρ 2 ), all matrices M satisfying M −H (τ ) ≤ √ γ−N0/10 and Q as in (D.2) obeys DISPLAYFORM14 Let us call this Event 1.

To proceed, we will argue that with high probability H (τ ) −H (τ ) is small so that the bound above is applicable with M =H (τ ) choice; which setsQ =X (τ ) in (D.5).

Applying Lemma C.6, we find that, with probability 1 − 2Nτ exp(−100n), DISPLAYFORM15 Let us call this Event 2.

We will show that our choice of L ensures right hand side is small enough and guarantees DISPLAYFORM16 2 0 , 1}. Desired claim follows by taking logarithms of upper/lower bounds and cancelling out √ N0 terms as follows DISPLAYFORM17 Here we use the fact that log A < 0 since A < 1 and cnρ ≥ 0.

Consequently, both Event 1 and Event 2 hold with probability 1−4 exp(−c1N0/ρ 2 )−2Nτ exp(−100n), implying (D.5) holds withQ =X (τ ) .

Union bounding this over 1 ≤τ ≤ L, (D.5) uniformly holds withQ =X (τ ) and all rows of X are 2-bounded with probability 1 − 4N exp(−100n) − 8L exp(−c1N0/ρ 2 ).

Applying Lemma A.1 on (X (τ ) ) L τ =1 , we conclude with the bound (D.4) on the merged matrix X.Lemma D.4 ( 2-bound on rows).

Consider the setup of Theorem D.3.

With probability 1 − 2N exp(−100(n + p)), each row of X has 2-norm at most c √ p + n for some constant c > 0.Proof.

The tth row of X is equal to xt = [ DISPLAYFORM18 .

Now, applying Lemma F.3 on all rows {xt} N t=1 , and using a union bound, with probability at least 1 − 2N exp(−100(n + p)), we have that Proof.

To prove this theorem, we combine Theorem D.3 with deterministic SGD convergence result of Theorem 4.1.

Applying Theorem D.3, with the desired probability, inequality (D.4) holds and for all i, input data satisfies the bound xi 2 ≤ (n + p)/(2c0) for a sufficiently small constant c0 > 0.

As the next step, we will argue that these two events imply the convergence of SGD.

DISPLAYFORM19 Let θ (i) , c (i) ∈ R n+p denote the ith rows of Θ, C respectively.

Observe that the square-loss is separable along the rows of DISPLAYFORM20 .

Hence, SGD updates each row c (i) via its own state equation DISPLAYFORM21 where yt,i is the ith entry of yt.

Consequently, we can establish the convergence result for an individual row of C. Convergence of all individual rows will imply the convergence of the overall matrix Θτ to the ground truth C. Pick a row index i (1 ≤ i ≤ n), set c = c (i) and denote ith row of Θτ by θτ .

Also denote the label corresponding to ith row by yt = yt,i.

With this notation, SGD over (2.3) runs SGD over the ith row with equations yt = φ( c, xt ) and with loss functions DISPLAYFORM22 Substituting our high-probability bounds on xt (e.g. FIG7 ) into Theorem 4.1, we can set B = (n + p)/(2c0), γ+ = (θ + √ 2) 2 , and γ− = ρ −1 /2.

Consequently, using the learning rate η = c0 DISPLAYFORM23 , for all τ ≥ 0, the τ th SGD iteration θτ obeys DISPLAYFORM24 where the expectation is over the random selection of SGD updates.

This establishes the convergence for a particular row of C. Summing up these inequalities (E.1) over all rows θ DISPLAYFORM25 (which converge to c(1) , . . .

, c (n) respectively) yields the targeted bound (4.4).

E.3.1 PROOF OF THEOREM 3.3Proof.

Applying Lemmas B.3 and 3.2, independent of L, Assumption 1 holds with parameters DISPLAYFORM0 This yields (θ + √ 2) 2 = 6n.

Hence, we can apply Theorem 4.2 with the learning rate η = c0 DISPLAYFORM1 and convergence rate 1 − DISPLAYFORM2 .

To conclude with the stated result, we use the change of variable c0/6 → c0.

where ρ is given by (E.2).

Use the change of variable c0/2 → c0 to conclude with the stated result.

In a similar fashion to Section 4, we provide a more general result on unstable systems that makes a parametric assumption on the statistical properties of the state vector.

Assumption 2 (Well-behaved state vector -single timestamp).

Given timestamp T0 > 0, there exists positive scalars γ+, γ−, θ and an absolute constant C > 0 such that θ ≤ 3 √ n and the following holds γ+In Σ[hT 0 ] γ−In , DISPLAYFORM0 The next theorem provides the parametrized result on unstable systems based on this assumption.

Theorem E.1 (Unstable system -general).

Suppose we are given N independent trajectories (h DISPLAYFORM1 t ) t≥0 for 1 ≤ i ≤ N .

Sample each trajectory at time T0 to obtain N samples (yi, hi, ui) N i=1 where ith sample is (yi, hi, ui) = (h (i) DISPLAYFORM2 ).

Let C, c0 > 0 be absolute constants.

Suppose Assumption 1 holds with T0 and sample size satisfies N ≥ Cρ 2 (n + p) where ρ = γ+/γ−. Assume φ is β-increasing, zero initial state conditions, and ut DISPLAYFORM3 ∼ N (0, Ip).

Set scaling to be µ = 1/ √ γ+ and learning rate to be η = c0 .

Starting from Θ0, we run SGD over the equations described in (2.2) and (2.3).

With probability 1 − 2N exp(−100(n + p)) − 4 exp(−O( N ρ 2 )), all iterates satisfy DISPLAYFORM4 where the expectation is over the randomness of the SGD updates.

√ n, hence all rows of X obeys xi 2 ≤ (n + p)/(2c0), DISPLAYFORM5 To proceed, using γ− = ρ −1 /2, B = (n + p)/(2c0), and γ+ = (θ + √ 2) 2 , we apply Theorem 4.1 on the loss function (2.3); which yields the desired result.

E.5 PROOF OF THEOREM 5.1Proof.

The proof is a corollary of Theorem E.1.

We need to substitute the proper values in Assumption 2.

Applying Lemma B.3, we can substitute γ+ = B 2 T 0 and θ = √ 6n − √ 2 ≥ √ n. Next, we need to find a lower bound.

Applying Lemma 3.2 for n > 1 and Lemma B.6 for n = 1, we can substitute γ− = γ+/ρ with the ρ definition of (5.2).

With these, the result follows as an immediate corollary of Theorem E.1.

The following theorem bounds the empirical covariance of matrices with independent subgaussian rows.

Given a random vector x, define the de-biasing operation as zm(x) = x − E[x].Theorem F.1.

Let A ∈ R n×d be a matrix with independent subgaussian rows {ai} .

Then, each of the following happens with probability at least 1 − 2 exp(−cK −4 λ 2 n), DISPLAYFORM0 • Suppose all rows of A have equal expectations.

Then Centering this subexponential variable around zero introduces a factor of 2 when bounding subexponential norm and yields (x .

Now, using the fact that Yu − Yv is sum of n independent zero-mean subexponential random variables, we have the tail bound P(n −1 |Yu − Yv| ≥ t) ≤ 2 exp(−c n min{ t 2 K 4 y 2 2 , t K 2 y 2 }).Applying Talagrand's chaining bound for mixed tail processes with distance metrics ρ2 =

<|TLDR|>

@highlight

We study the state equation of a recurrent neural network. We show that SGD can efficiently learn the unknown dynamics from few input/output observations under proper assumptions.

@highlight

The paper studies discrete-time dynamical systems with a non-linear state equation, proving that running SGD on a fixed-length trajectory gives logarithmic convergence.

@highlight

This work considers the problem of learning a non-linear dynamical system in which the output equals the state. 

@highlight


This paper studies the ability of SGD to learn dynamics of a linear system and non-linear activation.
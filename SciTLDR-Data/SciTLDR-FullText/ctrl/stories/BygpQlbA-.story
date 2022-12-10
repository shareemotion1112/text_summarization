We study the control of symmetric linear dynamical systems with unknown dynamics and a hidden state.

Using a recent spectral filtering technique for concisely representing such systems in a linear basis, we formulate optimal control in this setting as a convex program.

This approach eliminates the need to solve the non-convex problem of explicit identification of the system and its latent state, and allows for provable optimality guarantees for the control signal.

We give the first efficient algorithm for finding the optimal control signal with an arbitrary time horizon T, with sample complexity (number of training rollouts) polynomial only in log(T) and other relevant parameters.

Recent empirical successes of reinforcement learning involve using deep nets to represent the underlying MDP and policy.

However, we lack any supporting theory, and are far from developing algorithms with provable guarantees for such settings.

We can make progress by addressing simpler setups, such as those provided by control theory.

Control theory concerns the control of dynamical systems, a non-trivial task even if the system is fully specified and provable guarantees are not required.

This is true even in the simplest setting of a linear dynamical system (LDS) with quadratic costs, since the resulting optimization problems are high-dimensional and sensitive to noise.

The task of controlling an unknown linear system is significantly more complex, often giving rise to non-convex and high-dimensional optimization problems.

The standard practice in the literature is to first solve the non-convex problem of system identification-that is, recover a model that accurately describes the system-and then apply standard robust control methods.

The non-convex problem of system identification is the main reason that we have essentially no provable algorithms for controlling even the simplest linear dynamical systems with unknown latent states.

In this paper, we take the first step towards a provably efficient control algorithm for linear dynamical systems.

Despite the highly non-convex and high-dimensional formulation of the problem, we can efficiently find the optimal control signal in polynomial time with optimal sample complexity.

Our method is based on wave-filtering, a recent spectral representation technique for symmetric LDSs BID7 ).

A dynamical system converts input signals { 1 , . . .

, } ∈ R into output signals { 1 , . . . , } ∈ R , incurring a sequence of costs 1 , . . .

, ∈ R. We are interested in controlling unknown dynamical systems with hidden states (which can be thought of as being partially "observed"via the output signals).

A vast body of work focuses on linear dynamical systems with quadratic costs, in which the { } and { } are governed by the following dynamics: DISPLAYFORM0 where ℎ 1 , . . .

, ℎ ∈ R is a sequence of hidden states starting with a fixed ℎ 1 .

Matrices , , , , , of appropriate dimension describe the system and cost objective; the { } are Gaussian noise vectors.

All of these matrices, as well as the parameters of the Gaussian, can be unknown.

The most fundamental control problem involves controlling the system for some time horizon : find a signal 1 , . . . , that minimizes the sum of these quadratic output costs .

Clearly, any algorithm for doing so must first learn the system parameters in some form, and this is often the source of computational intractability (meaning algorithms that take time exponential in the number of system parameters).Previously known algorithms are of two types.

The first type tries to solve the non-convex problem, with algorithms that lack provable guarantees and may take exponential time in the worst case: e.g., expectation-maximization (EM) or gradient-based methods (back-propagation through time, like the training of RNNs) which identify both the hidden states and system parameters.

Algorithms of the second type rely upon regression, often used in time-series analysis.

Since -step dynamics of the system involve the first powers of , the algorithm represents these powers as new variables and learns the -step dynamics via regression (e.g., the so-called VARX( , ) model) assuming that the system is well-conditioned; see Appendix A. This has moderate computational complexity but high sample complexity since the number of parameters in the regression scales with , the number of time steps.

Our new method obtains the best of both results: few parameters to train resulting in low sample complexity, as well as polynomial computational complexity.

In Section 2 we state the precise algorithm.

The informal result is as follows.

DISPLAYFORM1 Theorem 1.1 (Controlling an unknown LDS; informal).

Let D be a linear dynamical system with a symmetric transition matrix and with size |D|.

Then, for every > 0, Algorithm 1 produces a sequence of controls (ˆ1, . .

.ˆ), ‖ˆ‖ 2 ≤ 1 with ‖ˆ1 : ‖ 2 ≤ , such that DISPLAYFORM2 Assuming i.i.d.

Gaussian noise ∼ (0, Σ), the algorithm samples˜(poly (|D|, , Tr Σ, 1/ )) trajectories from D, and runs in time polynomial in the same parameters.

The field of optimal control for dynamical systems is extremely broad and brings together literature from machine learning, statistical time-series analysis, dynamical system tracking and Kalman filtering, system identification, and optimal control.

For an extensive survey of the field, see e.g. BID12 BID1 .Tracking a known system.

A less ambitions goal than control is tracking of a dynamical system, or prediction of the output given a known input.

For the special case of LDS, the well-known Kalman filter BID9 is an optimal recursive least-squares solution for maximum likelihood estimation (MLE) under Gaussian perturbations to a linear dynamical system.

System identification.

When the underlying dynamical system is unknown, there are essentially no provably efficient methods for recovering it.

For various techniques used in practice, see the classic survey BID10 .

BID11 suggest using the EM algorithm to learn the parameters of an LDS, nowadays widespread, but it is well-known that optimality is not guaranteed.

The recent result of BID6 gives a polynomial time algorithm for system recovery, although it applies only to the single-input-single-output case and makes various statistical assumptions on the inputs.

Model-free tracking.

Our methods depend crucially on a new algorithm for LDS sequence prediction, at the heart of which is a new convex relaxation for the tracking formulation BID7 .

In particular, this method circumvent the obstacle of explicit system identification.

We detail our usage of this result in Definition 2.3.We note an intriguing connection to the recently widespread use of deep neural networks to represent an unknown MDP in reinforcement learning: the main algorithm queries the unknown dynamical system with exploration signals, and uses its responses to build a compact representation (denoted byˆin Algorithm 1) which estimates the behavior of the system.

Time-series analysis.

One of the most common approaches to modeling dynamical systems is the autoregressive-moving average (ARMA) model and its derivatives in the time-series analysis literature BID5 BID2 BID3 .

At the heart of this method is the autoregressive form of a time series, namely, DISPLAYFORM0 Using online learning techniques, it is possible to completely identify an autoregressive model, even in the presence of adversarial noise BID0 .

This technique lies at the heart of a folklore regression method for optimal control, given in the second row of table 1.Optimal control.

The most relevant and fundamental primitive from control theory, as applied to the control of linear dynamical systems, is the linear-quadratic-Gaussian (LQG) problem.

In this setting, the system dynamics are assumed to be known, and the task is to find a sequence of inputs which minimize a given quadratic cost.

A common solution, the LQG controller, is to combine Kalman filtering with a linear-quadratic regulator, a controller selected by solving the Bellman equation for the problem.

Such an approach admits theoretical guarantees under varied assumptions on the system; see, for example, BID4 .Our setting also involves a linear dynamical system with quadratic costs, and thus can be seen as a special case of the LQG setup, in which the process noise is zero, and the transition matrix is assumed to be symmetric.

However, our results are not analogous: our task also includes learning the system's dynamics.

As such, our main algorithm for control takes a very different approach than that of the standard LQR: rather than solving a recursive system of equations, we provide a formulation of control as a one-shot convex program.

First, we state the formal definitions of the key objects of interest.

Definition 2.1 (Dynamical system).

A dynamical system D is a mapping that takes a sequence of input vectors 1 , . . .

, ∈ B 2 = { ∈ R : ‖ ‖ 2 ≤ 1} to a sequence of output vectors 1 , . . . , ∈ R and costs 1 , . . . , ∈ R. Denote : = [ ; . . . ; ] as the concatenation of all input vectors from time to , and write DISPLAYFORM0 Definition 2.2 (Linear dynamical system).

A linear dynamical system (LDS) is a dynamical system whose outputs and costs are defined by DISPLAYFORM1 where ℎ 1 , . . .

, ℎ ∈ R is a sequence of hidden states starting with fixed ℎ 1 , and , , , , , are matrices (or vectors) of appropriate dimension.

We assume ‖ ‖ op ≤ 1, i.e., all singular values of are at most one, and that 0, 0.Our algorithm and its guarantees depend on the construction of a family of orthonormal vectors in R , which are interpreted as convolution filters on the input time series.

We define the wave-filtering matrix below; for more details, see Section 3 of BID7 .

Definition 2.3 (Wave-filtering matrix).

Fix any , , and 1 ≤ ≤ .

Let be the eigenvector corresponding to the -th largest eigenvalue of the Hankel matrix ∈ R × , with entries := 2 ( + ) 3 −( + ) .

The wave-filtering matrix Φ ∈ R × is defined by vertically stacked block matrices {Φ ( ) ∈ R × }, defined by horizontally stacked multiples of the identity matrix: DISPLAYFORM2 Then, letting range from 1 to , Φ : − then gives a dimension-wise convolution of the input time series by the filters { } of length .

Theorem 3.3 uses a structural result from BID7 , which guarantees the existence of a concise representation of D in the basis of these filters.

The main theorem we prove is the following.

Tr(Σ) ln( ) 2 ã , produces a sequence of controls (ˆ1, . .

.ˆ)

∈ B 2 , such that with probability at least 1 − , DISPLAYFORM3 assuming that DISPLAYFORM4 Further, the algorithm samples poly 1 , log 1 , log , log , 1 , , , , Tr(Σ) trajectories from the dynamical system, and runs in time polynomial in the same parameters.

We remark on some of the conditions.

is bounded away from 0 when we suffer loss in all directions of .

In condition (3), Tr( Σ) is inevitable loss due to background noise, so (3) is an assumption on the system's controllability.

We set up some notation for the algorithm.

Let Φ ∈ R × be the wave-filtering matrix from Definition 2.3.

Let = 0 for ≤ 0, and let = : − +1 .

Let = max(‖ ‖ , ‖ ‖ , ‖ ‖ , ‖ ‖ ) be an upper bound for the matrices involved in generating the outputs and costs.

To prove Theorem 2.4, we invoke Lemma 3.1 and Lemma 3.2, proved in Subsection 3.1 and Subsection 3.2, respectively.

DISPLAYFORM0 Lemma 3.2 (Robustness of control to uncertainty in dynamics).

Let 1: = arg min 1: DISPLAYFORM1 whereˆ∈ R × is such that for every sequence of input signals ( 1 , . . .

, ) ∈ B 2 with ‖ 1: ‖ 2 ≤ and ≤ , Equation (5) holds withˆ=D ( 1: ).

Assume (3).

Then DISPLAYFORM2 Moreover, the minimization problem posed above is convex (for , 0), and can be solved to within opt acccuracy in poly( , , 1/ opt ) time using the ellipsoid method.

Proof of Theorem 2.4.

Use Lemma 3.1 with ← » .

Note that 1: = is a valid input to the LDS because ‖ ‖ 2 = 1.

Now use Lemma 3.2 on the conclusion of Lemma 3.1.

To prove Lemma 3.1, we will use the following structural result from BID7 restated to match the setting in consideration.

DISPLAYFORM0 Proof.

This follows from Theorem 3b in BID7 after noting four things.1.

E( − ) are the outputs when the system is started at ℎ 1 = 0 with inputs 1: and no noise.

A linear relationship − = ′ holds by unfolding the recurrence relation.2.

Examining the construction of in the proof of Theorem 3b, the is exactly the projection of ′ onto the subspace defined by Φ. (Note we are using the fact that the rows of Φ are orthogonal.)

3.

Theorem 3b is stated in terms of the quantity , which is bounded by 2 by Lemma F.5 in BID7 .4.

We can modify the system so that = by replacing ( , , , ) with (( ) , ( ) , ( ), ).

This replaces by (max( , √ )).

Theorem 3b originally had dependence of on both Φ and , but if = then the dependence is only on Φ .Proof of Lemma 3.1.

DISPLAYFORM1 .

Letting ′ be the matrix such that − = ′ , and = ′ Φ ⊤ as in Theorem 3.3, we have that DISPLAYFORM2 Let 1 ≤ ≤ .

We bound the error under controls 1: ∈ B 2 , ‖ 1: ‖ 2 ≤ using the triangle inequality.

Letting 1: be the output under 1: , DISPLAYFORM3 By Theorem 3.3, for = Ω(log 2 log( / )), choosing constants appropriately, the first term is ≤ 4 .To bound the second term in (10), we show concentrates around E .

We have DISPLAYFORM4 By concentration of sums of 2 random variables (see BID8 , for example), DISPLAYFORM5 Take ′ = 2 and ′ = 4 √ and note was chosen to satisfy (12).

Use the union bound to get that DISPLAYFORM6 To bound the third term in (10), we first show thatˆconcentrates around .

We havê DISPLAYFORM7 By 2 concentration, DISPLAYFORM8 We also have (E − )1 DISPLAYFORM9 With ≥ 1 − probability, we avoid both the bad events in FORMULA2 and FORMULA2 DISPLAYFORM10 and for all ‖ 1: ‖ ≤ , the third term of FORMULA2 is bounded by (note ‖Φ‖ op = 1 because it has orthogonal rows) DISPLAYFORM11 Thus by (10), E[ ] − −ˆΦ 2 ≤ with probability ≥ 1 − .

To prove Lemma 3.2, we need the following helpful lemma.

Lemma 3.4.

For a symmetric LDS D with = max(‖ ‖ , ‖ ‖ , ‖ ‖ , ‖ ‖ op ) and , , and an approximationD whose predictionsˆsatisfy the conclusion of Lemma 3.1, we have that for every sequence of controls ( 1 , . . .

, ) ∈ B 2 , and for every 1 ≤ ≤ , DISPLAYFORM0 where costD ( DISPLAYFORM1 DISPLAYFORM2 using Cauchy-Schwarz.

Proof of Lemma 3.2.

Define DISPLAYFORM3 1: = arg min 1: DISPLAYFORM4 DISPLAYFORM5 .

By assumption (3), ≤ .

We have DISPLAYFORM6 By Lemma 3.4 for inputs DISPLAYFORM7 Lettingˆ1 : be the outputs underD under the controlˆ1 : , note that similar to (32), DISPLAYFORM8 becauseˆ1 : is optimal forD. By Lemma 3.4 for inputsˆ1 : , DISPLAYFORM9 Now by (34), (35), and (36), DISPLAYFORM10

We have presented an algorithm for finding the optimal control inputs for an unknown symmetric linear dynamical system, which requires querying the system only a polylogarithmic number of times in the number of such inputs , while running in polynomial time.

Deviating significantly from previous approaches, we circumvent the non-convex optimization problem of system identification by a new learned representation of the system.

We see this as a first step towards provable, efficient methods for the traditionally non-convex realm of control and reinforcement learning.

In this section, we verify the statement made in Section 1 on the time and sample complexity of approximating a linear dynamical system with an autoregressive model.

Although this is wellknown (see, for example, Section 6 of BID6 ), we present a self-contained presentation for convenience and to unify notation.

The vector autoregressive model with exogenous variables, or VARX( , ), is a touchstone in timeseries analysis.

Given a time series of inputs (sometimes known as biases) { }, it generates the time series of responses { } by the following recurrence: DISPLAYFORM0 Here, and are memory parameters, the { } and { } are matrices of appropriate dimension, and the { } are noise vectors.

In the special case of = 0, the problem can be solved efficiently with linear regression: in this case, is a linear function of the concatenated inputs [ ; −1 ; . . .

, − +1 ].A VARX(0, ) model is specified by = [ (0) , . . .

, ( −1) ]

∈ R × and predicts = : − +1 .

We quantify the relationship between VARX(0, ) and linear dynamical systems, with a statement analogous to Theorem 3.1: Theorem A.1.

Let D be an LDS with size ℒ, fixed ℎ 1 , and noise = 0, producing outputs { 1 , . . . , } from inputs { 1 , . . . , }.

Suppose that the transition matrix of D has operator norm at most < 1.

Then, for each > 0, there is a VARX(0, ) model with = ( 1 1− log(ℒ/ )), specified by a matrix , such that DISPLAYFORM1 Proof.

By the modification of D given in the proof of Theorem 3.3, we may assume without loss of generality that = .

Also, as in the discussion of Theorem 3.1, it can be assumed that the initial hidden state ℎ 1 is zero.

Then, we construct the block of corresponding to lag as DISPLAYFORM2 This is well-defined for all 1 ≤ ≤ .

Note that when ≥ , the autoregressive model completely specifies the system D, which is determined by its (infinite-horizon) impulse response function.

Furthermore, by definition of , we have DISPLAYFORM3 Noting that DISPLAYFORM4 we conclude that DISPLAYFORM5 implying the claim by the stated choice of .VARX(0, ) only serves as a good approximation of an LDS whose hidden state decays on a time scale shorter than ; when the system is ill-conditioned ( is close to 1), this can get arbitrarily large, requiring the full time horizon = .On the other hand, it is clear that both the time and sample complexity of learning a VARX(0, ) model grows linearly in .

This verifies the claim in the introduction.

<|TLDR|>

@highlight

Using a novel representation of symmetric linear dynamical systems with a latent state, we formulate optimal control as a convex program, giving the first polynomial-time algorithm that solves optimal control with sample complexity only polylogarithmic in the time horizon.
We derive reverse-mode (or adjoint) automatic differentiation for solutions of stochastic differential equations (SDEs), allowing time-efficient and constant-memory computation of pathwise gradients, a continuous-time analogue of the reparameterization trick.

Specifically, we construct a backward SDE whose solution is the gradient and provide conditions under which numerical solutions converge.

We also combine our stochastic adjoint approach with a stochastic variational inference scheme for continuous-time SDE models, allowing us to learn distributions over functions using stochastic gradient descent.

Our latent SDE model achieves competitive performance compared to existing approaches on time series modeling.

Deterministic dynamical systems can often be modeled by ordinary differential equations (ODEs).

For training, a memory-efficient implementation of the adjoint sensitivity method (Chen et al., 2018) effectively computes gradients through ODE solutions with constant memory cost.

Stochastic differential equations (SDEs) are a generalization of ODEs which incorporate instantaneous noise into their dynamics (Arnold, 1974; Øksendal, 2003) .

They are a natural fit for modeling phenomena governed by small and unobserved interactions.

In this paper, we generalize the adjoint method to dynamics defined by SDEs resulting in an approach which we call the stochastic adjoint sensitivity method.

Building on theoretical advances by Kunita (2019), we derive a memory-efficient adjoint method whereby we simultaneously reconstruct the original trajectory and evaluate the gradients by solving a backward SDE (in the sense of Kunita (2019)) whose formulation we detail in Section 3.

Computationally, in order to retrace the original trajectory during the backward pass, we need to reuse noise samples generated in the forward pass.

In Section 4, we give an algorithm that allows arbitrarily-precise querying of a Brownian motion realization at any time point, while only storing a single random seed.

Overall, this results in a constant-memory algorithm that approximates the gradient arbitrarily well as step size reduces by computing vector-Jacobian products a constant number of times per-iteration.

See Table 2 for a comparison of our method against previous approaches in terms of asymptotic time and memory complexity.

We incorporate SDEs into a stochastic variational inference framework, whereby we efficiently compute likelihood ratios and backpropagate through the evidence lower bound using our adjoint approach.

This effectively generalizes existing model families such as latent ODEs (Rubanova et al., 2019) and deep Kalman filters (Krishnan et al., 2017) .

We review works on learning ODEs and SDEs.

We refer the reader to Appendix B on background for stochastic flows (Kunita, 2019) and (backward) Stratonovich integrals.

The adjoint sensitivity method is an efficient approach to solve optimization problems by considering the dual form (Pontryagin, 2018) .

Chen et al. (2018) recently applied this idea to obtain gradients with respect to parameters of a neural network defining an ODE.

The method is scalable due to its memory-efficiency, as intermediate computations need not be cached as in regular backpropagation (Rumelhart et al., 1988) .

Recent works have considered SDEs whose drift and diffusion functions are defined by neural networks (Tzen and Raginsky, 2019a,b; Liu et al., 2019; Jia and Benson, 2019) .

Consider a filtered probability space (Ω, F, {F t } t∈T , P ) on which an m-dimensional adapted Wiener process {W t } t∈T is defined.

An Itô SDE defines a stochastic process {Z t } t∈T by

where z 0 ∈ R d is a deterministic starting value, and b :

are the drift and diffusion functions, respectively.

Here, the second integral on the right hand side of (1) is the Itô stochastic integral (Øksendal, 2003) .

When the coefficients are globally Lipschitz in both the state and time components, there exists a unique strong solution to the SDE (Øksendal, 2003) .

Therefore, one can consider coefficients defined by neural networks that have smooth activation functions (e.g. tanh) of the form b(z, t, θ) and σ (z, t, θ) .

This results in a model known as the neural SDE.

We derive a backward Stratonovich SDE for what we call the stochastic adjoint process.

A direct implication of this is a gradient computation algorithm that works by solving a set of dynamics in reverse time and relies on vector-Jacobian products without storing intermediate computation.

Recall from Appendix B.3, Φ s,t (z) := Z s,z t is the solution at time t when the process is started at z at time s, and its inverse is defined asΨ s,t (z) := Φ −1 s,t (z).

Consider A s,t (z) = ∇(L(Φ s,t (z))), where L is a scalar loss function.

The chain rule gives A s,t (z) = ∇L(Φ s,t (z))∇Φ s,t (z).

LetÃ s,t (z) := A s,t (Ψ s,t (z)) = ∇L(z)∇Φ s,t (Ψ s,t (z)) = ∇L(z)K s,t (z).

Note that A s,t (z) =Ã s,t (Φ s,t (z)).

Since ∇L(z) is constant, we see that (Ã s,t (z),Ψ s,t (z)) satisfies the following backward SDE system by Lemma C.1 (cf.

Appendix C)

Since (11) can be viewed as a single SDE (with smooth coefficients) for an augmented state,Ã s,T (z) also has a unique strong solution.

Therefore, for t = 0, we may writẽ

where W · = {W t } 0≤t≤T denotes the path of the Brownian motion and F :

is a deterministic measurable function (the Itô map) (Rogers and Williams, 2000, Definition 10.9).

The next theorem follows immediately from (3) and the definition of F.

Theorem 3.1:

The theorem is a consequence of

T ) and (3).

This implies we may solve the dynamics (11) starting from the end state of the forward solve Z 0,z T to obtain the gradient of the loss with respect to the starting value z. To obtain the gradient with respect to the parameters, we augment the original state with parameters.

Algorithm 1 summarizes this assuming access to a black-box solver SDESolve.

See details in Appendix C.

Input: parameters θ, start time t 0 , stop time t 1 , final state z t 1 , loss gradient ∂L/z t 1 .

Input: drift b(z, t, θ), diffusion σ(z, t, θ), Wiener process sample w(t).

We present a data structure that allows arbitrarilyprecise query of the sample path of the Wiener process given a global random seed based on the Brownian tree construction.

The data structure facilitates the adjoint method such that we can ensure the noise sample in the backward solve is the same as in the forward solve using a split pseudorandom random number generator (PRNG).

We present the procedure in Algorithm D.2 and details in Appendix D.

Consider the SDEs

where

are Lipschitz in both arguments.

Suppose (4) and (5) define the prior and posterior processes, respectively.

Additionally, assume there is a function u :

for all x ∈ R d and t ∈ R.

Then, the variational free energy (Opper, 2019) can be written as

where the expectation is taken over the distribution of the posterior process defined by (5), and y 1 , . . .

, y N are observations at times t 1 , . . .

, t N , respectively.

To compute the gradient with respect to parameters, we need only augment the forward equation with an extra variable whose drift function returns 1 2 |u(Z t , t)| 2 and diffusion function is 0.

In this case, the backward adjoint dynamics can be derived analogously using (11).

Appendix E includes details.

We verify our theory by comparing the gradients obtained by our stochastic adjoint framework against analytically derived gradients for chosen test problems with closed-form solutions.

We then fit latent SDE models with our framework on two synthetic datasets and a real dataset, verifying that the variational inference framework promotes learning a generative model of time series.

Due to space constraint, we refer the reader to Appendix F for results on numerical studies and Appendix N for results on synthetic data.

We present only the results on the motion capture dataset here.

We experiment on a dataset extracted from the CMU motion capture library.

We use the dataset adopted by Gan et al. (2015) which consists of 23 walking sequences of subject number 35 that is partitioned into 16 training, 3 validation, and 4 test sequences.

We include the settings in Appendix O and report the test MSE here following Yıldız et al. (2019) .

Appendix B. Additional Background t .

For the diffusion functions σ 1 , . . .

, σ m , we will also write σ : R d → R d×m as the matrix-valued function obtained by stacking the component functions σ i in a columnwise fashion, and index its jth row and ith column by σ j,i .

Among recent work on neural SDEs, none has enabled an efficient training framework.

In particular, Tzen and Raginsky (2019a); Liu et al. (2019) considered computing the gradient by simulating the forward dynamics of an explicit Jacobian matrix the size either the squared number of parameters or the number of parameters times the number of states, building on the pathwise approach (Gobet and Munos, 2005; Yang and Kushner, 1991) .

By contrast, the approach we present only requires evaluating vector-Jacobian products a constant number of times with respect to the number of parameters and states, which has the same asymptotic time cost as evaluating the drift and diffusion functions, and can be done automatically by modern machine learning libraries (Maclaurin et al., 2015; Paszke et al., 2017; Abadi et al., 2016; Frostig et al., 2018) .

Our stochastic adjoint sensitivity method involves stochastic processes running forward and backward in time.

The Stratonovich stochastic integral, due to its symmetry, gives nice expressions for the backward dynamics and so is more convenient for our purpose.

Our results can be applied straightforwardly to Itô SDEs as well using a conversion result (see e.g. (Platen, 1999, Sec. 2

Following the treatment of Kunita (Kunita, 2019), we introduce the forward and backward Stratonovich integrals.

Let {F s,t } s≤t;s,t∈T be a two-sided filtration, where F s,t is the σ-algebra generated by

For a continuous semimartingale {Y t } t∈T adapted to the forward filtration {F 0,t } t∈T , the Stratonovich stochastic integral is defined as

where

denotes the size of largest segment of the partition, and the limit is to be interpreted in the L 2 sense.

The Itô integral uses instead the left endpoint Y t k rather than the average.

In general, the Itô and Stratonovich integrals differ by a term of finite variation.

To define the backward Stratonovich integral, we consider the backward Wiener process {W t } t∈T defined asW t = W t − W T for all t ∈ T that is adapted to the backward filtration {F t,T } t∈T .

For a continuous semimartingaleY t adapted to the backward filtration, we define

where Π = {0 = t N < · · · < t 0 = T } is a partition, and the limit is again in the L 2 sense.

It is well known that an ODE defines a flow of diffeomorphisms (Arnold, 1978) .

Here we consider the stochastic analogue for the Stratonovich SDE

Throughout the paper, we assume b, σ are of class C ∞,1 b

, so that the SDE has a unique strong solution.

Let Φ s,t (z) := Z s,z t be the solution at time t when the process is started at z at time s.

Given a realization of the Wiener process, this defines a collection S = {Φ s,t } s≤t;s,t∈T of continuous maps from R d to itself.

The following theorem shows that these maps are diffeomorphisms and that they satisfy backward SDEs.

Theorem B.1 (Thm.

3.7.1 (Kunita, 2019)):

(i) With probability 1, the collection S = {Φ s,t } s≤t;s,t∈T satisfies the flow property

Moreover, each Φ s,t is a smooth diffeomorphism from R d to itself.

We thus call S the stochastic flow of diffeomorphisms generated by the SDE (7).

(ii) The backward flowΨ s,t := Φ −1 s,t satisfies the backward SDE:

for all z ∈ R d and s, t ∈ T such that s ≤ t a.s.

Note that the coefficients in (7) and (8) differ by only a negative sign.

This symmetry is due to our use of the Stratonovich integral (see Figure 2) .

Figure 2: Negating the drift and diffusion functions for an Itô SDE and simulating backwards from the end state gives the wrong solution.

Negating the drift and diffusion functions for the converted Stratonovich SDE, however, gives the correct path when simulated in reverse time.

We present our main contribution, i.e. the stochastic analog of the adjoint sensitivity method for SDEs.

We use (8) to derive another backward Stratonovich SDE for what we call the stochastic adjoint process.

The direct implication of this is a gradient computation algorithm that works by solving a set of dynamics in reverse time and relies on vector-Jacobian products without storing intermediate computation produced in the forward pass.

The goal is to derive the stochastic adjoint process {∂L/∂Z t } t∈T that can be simulated by evaluating only vector-Jacobian products, where L = L(Z T ) is a scalar loss of the terminal state Z T .

The main theoretical result is Theorem 3.1.

We first derive a backward SDE for the process {∂Z T /∂Z t } t∈T , assuming that Z t =Ψ t,T (Z T ) for a deterministic Z T ∈ R d that does not depend on the realized Wiener process.

We then extend to the case where

In the latter case, the resulting value cannot be interpreted as the solution to a backward SDE anymore due to loss of adaptiveness; instead we will formulate the result using the Itô map (Rogers and Williams, 2000) .

Finally, we extend the state of Z to include parameters and obtain the gradient with respect to them.

We first derive the SDE for the Jacobian matrix of the backward flow.

Consider the stochastic flow generated by the backward SDE (8) as in Theorem B.1(ii).

Let J s (z) := ∇Ψ s,T (z), then it satisfies the backward SDE

for all s ≤ t and x ∈ R d a.s.

Furthermore, let K s,t (z) = J s,t (z) −1 , we have

for all s ≤ t and x ∈ R d a.s.

The proof included in Appendix I relies on Itô's lemma in the Stratonovich form (Kunita, 2019, Theorem 2.4.1).

This lemma considers only the case where the endpoint z is fixed and deterministic.

Now we compose the state process (represented by the flow) and the loss function L.

WritingX s = (Ã s,T (z) ,Ψ s,T (z) ) as the augmented process, the system (11) is a backward Stratonovich SDE of the form

.

As a result (11) has a unique strong solution.

Without loss of generality, assume t = 0.

Since (11) admits a strong solution, we may writeÃ

where W · = {W t } 0≤t≤T denotes the path of the Brownian motion and

is a deterministic measurable function (the Itô map) (Rogers and Williams, 2000, Definition 10.9).

Intuitively, F can be thought as an algorithm that computes the solution to the backward SDE (11) given the position z at time T and the realized Brownian path.

Similarly, we let G be the solution map for the forward flow (7).

Immediately, we arrive at Theorem 3.1.

In practice, we compute solutions to SDEs with numerical solvers F h and G h , where h = T /N denotes the mesh size of a fixed grid 1 .

The approximate algorithm thus outputs

The following theorem provides sufficient conditions for convergence.

Suppose the schemes F h and G h satisfy the following conditions:

converge to 0 in probability as h → 0, and (ii) for any M > 0, we have sup |z|≤M |F h (z, W · ) − F(z, W · )| → 0 in probability as h → 0.

Then, for any starting point z of the forward flow, we have

in probability as h → 0.

For details and the proof see Appendix J. Usual schemes such as the Euler-Maruyama and Milstein method satisfy condition (i).

Indeed, they converge pathwise (i.e. almost surely) with explicit rates for any fixed starting point (Kloeden and Neuenkirch, 2007) .

While condition (ii) is rather strong, we note that the SDEs considered here have smooth coefficients and thus the solutions enjoy nice regularity properties in the starting position.

Therefore, it is reasonable to expect that the corresponding numerical schemes to also behave nicely as a function of both the mesh size and the starting position.

To the best of our knowledge this property is not considered at all in the literature on numerical methods for SDEs (where the initial position is fixed), but is crucial in the proof of Theorem C.2.

Detailed analysis for specific schemes is beyond the scope of this paper and is left for future research.

So far we have derived the gradient of the loss with respect to the initial state.

We can extend these results to give gradients with respect to parameters of the drift and diffusion functions by treating them as an additional part of the state whose dynamics has zero drift and diffusion.

We summarize this in Algorithm 1 2 , assuming access to a numerical solver SDESolve.

Note for the Euler-Maruyama scheme, the most costly terms to compute a t ∂b/∂θ and a t ∂σ i /∂θ can be evaluated by calling vjp(a t , b, θ) and vjp(a t , σ i , θ), respectively.

In principle, we can simulate the forward and backward adjoint dynamics with any high-order solver of choice.

However, in practice, to obtain a strong numerical solution 3 with order beyond 1/2, we need to simulate multiple integrals of the Wiener process such as

These random variables are difficult to simulate exactly and costly to approximate using truncated infinite series (Wiktorsson et al., 2001) .

Note that even though the backward SDE for the stochastic adjoint does not have diagonal noise, it satisfies a commutativity property (Rößler, 2004) when the SDE of the original 1.

We may also use adaptive solvers (Ilie et al., 2015) .

2.

We use row vector notation here.

3.

A numerical scheme is of strong order p if E [|XT − XNη|]

≤ Cη p for all T > 0, where Xt and XNη are respectively the coupled true solution and numerical solution, N and η are respectively the iteration index and step size such that N η = T , and C is independent of η.

dynamics has diagonal noise.

In this case, we can safely adopt certain numerical schemes of strong order 1.0 (e.g. Milstein (Milstein, 1994) and stochastic Runge-Kutta (Rößler, 2010)) without approximating multiple integrals or the Lévy area during simulation.

We verify this formally in Appendix K.

We have implemented several SDE solvers in PyTorch (Paszke et al., 2017) which include Euler-Maruyama, Milstein, and stochastic Runge-Kutta schemes with adaptive time-stepping using a PI controller (Ilie et al., 2015) .

In addition, following torchdiffeq (Chen et al., 2018), we have created a user-friendly subclass of torch.autograd.Function that facilitates gradient computation using our stochastic adjoint framework when the neural SDE is implemented as a subclass of torch.nn.Module.

We include a short code snippet covering the main idea of the stochastic adjoint in Appendix L and plan to release all code after the double-blind reviewing process.

The formulation of the adjoint ensures it can be numerically integrated by merely evaluating dynamics cheaply defined by vector-Jacobian products, as opposed to whole Jacobians.

However, the backward-in-time nature also introduces the additional difficulty that the same Wiener process sample path in the forward pass has to be queried again during the backward pass.

Naïvely storing Brownian motion increments and related quantities (e.g. Lévy area approximations) not only implies a large memory consumption but also disables using adaptive time-stepping numerical integrators, where the evaluation timestamps in the backward pass may be different from those in the forward pass.

To overcome this issue, we combine Brownian trees with splittable Pseudorandom number generators (PRNGs) and obtain a data structure that allows querying values of the Wiener process path at arbitrary times with logarithmic time cost with respect to some error tolerance.

Lévy's Brownian bridge (Revuz and Yor, 2013) states that given a start time t s and end time t e along with their respective Wiener process values w s and w e , the marginal of the process at time t ∈ (t s , t e ) is a normal distribution:

We can recursively apply this formula to evaluate the process at the midpoint of any two distinct timestamps where the values are already known.

Constructing the whole sample path of a Wiener process in this manner results in what is known as the Brownian tree (Gaines and Lyons, 1997).

We assume access to a splittable PRNG (Claessen and Pałka, 2013) , which has an operation split that deterministically generates two (or more) keys 4 using an existing key.

In addition, we assume access to an operation BrownianBridge which samples from (12) given a key.

To obtain the Wiener process value at a specific time, the seeded Brownian tree works by recursively sampling according to the Brownian tree with keys split from those of parent nodes, assuming the values at some initial and terminal times are known.

The algorithm terminates when the current time under consideration is within a certain error tolerance of the desired time.

We outline the full procedure in Algorithm D.2.

This algorithm has constant memory cost.

For fixed-step-size solvers, the tolerance that the tree will be queried at will scale as 1/L, where L is the number of steps in the solver.

Thus the complexity per-step will scale as log L.

Note that the variational free energy (6) can be derived from Girsanov's change of measure theorem (Opper, 2019) .

To efficiently Monte Carlo estimate this quantity and its gradient, we simplify the equation by noting that for a one-dimensional process {V t } t∈T adapted to the filtration generated by a one-dimensional Wiener process {W t } t∈T , if Novikov's condition (Øksendal, 2003) is satisfied, then the process defined by the Itô integral t 0 V s dW s is a Martingale (Øksendal, 2003) .

Hence, E T 0 u(Z t , t) dW t = 0, and

To Monte Carlo simulate the quantity in the forward pass along with the original dynamics, we need only extend the original augmented state with an extra variable L t such that the new drift and diffusion functions for the new augmented state

By (11), the backward SDEs of the adjoint processes become

(13) In this case, neither do we need to actually simulate the backward SDE of the extra variable nor do we need to simulate its adjoint.

Moreover, when considered as a single system for the augmented adjoint state, the diffusion function of the backward SDE (13) satisfies the commutativity property (17).

We consider three carefully designed test problems (examples 1-3 (Rackauckas and Nie, 2017); details in Appendix M) all of which have closed-form solutions.

We compare the gradient computed from simulating our stochastic adjoint process using the Milstein scheme against the gradient evaluated by analytically solving the equations.

Figure F (a) shows that for test example 1, the error between the adjoint gradient and analytical gradient decreases as the fixed step size decreases.

One phenomenon not covered by our theory is that the error can be indeed be controlled by the adaptive solver.

This is shown by the fact that for all three test problems, the mean-square error across dimensions tends to be smaller as the absolute tolerance is reduced (see Figure F (c, f, j) ).

However, we note that the Number of Function Evaluations (NFEs) tends to be much larger than that in the ODE case (Chen et al., 2018) , which is expected given the inherent roughness of Brownian motion paths.

Sensitivity Analysis for SDEs.

Gradient computation is closely related to sensitivity analysis.

Computing gradients with respect to parameters of vector fields of an SDE has been extensively studied in the stochastic control literature (Kushner and Dupuis, 2013) .

In particular, for low dimensional problems, this is done effectively using dynamic programming (Baxter and Bartlett, 2001 ) and finite differences (Glasserman and Yao, 1992; L'Ecuyer and Perron, 1994) .

However, both approaches scale poorly with the dimensionality of the parameter vector.

Analogous to REINFORCE (or score-function estimator) (Williams, 1992; Kleijnen and Rubinstein, 1996; Glynn, 1990) , Yang and Kushner (1991) as

for some random variable H 5 .

However, H usually depends on the density of Z T with respect to the Lebesgue measure which can be difficulty to compute.

Gobet and Munos (2005) extended this approach by weakening a non-degeneracy condition using Mallianvin calculus.

Closely related to the current submission is the pathwise method (Yang and Kushner, 1991), which is the continuous-time analog of the reparameterization trick (Kingma and Welling, 2013; Rezende et al., 2014) .

Existing methods in this regime (Tzen and Raginsky, 2019a; Gobet and Munos, 2005; Liu et al., 2019) all require simulating a forward SDE where each step requires computing entire Jacobian matrices.

This computational cost is prohibitive for high-dimensional systems with a large number of parameters.

Based on the Euler discretization, Giles and Glasserman (2006) considered storing the intermediate values and performing reverse-mode automatic differentiation.

They named this method the adjoint approach, which, by modern standards, is a form of "backpropagation 5.

The random variable H is not unique.

through the operations of a numerical solver".

We comment that this approach, despite widely adopted in the field of finance for calibrating market models (Giles and Glasserman, 2006) , has high memory cost, and relies on a fixed step size Euler-Maruyama discretization.

This approach was used by (Hegde et al., 2019) to parameterize the drift and diffusion of an SDE using Gaussian processes.

Backward SDEs.

Our backward SDE for the stochastic adjoint process relies on the notion of backward SDEs by Kunita (2019) which is based on two-sided filtrations.

This is different from the more traditional notion of backward SDEs where only a single filtration is defined (Peng, 1990; Pardoux and Peng, 1992) .

Based on the latter notion, forward-backward SDEs (FBSDEs) have been proposed to solve the stochastic optimal control problem (Peng and Wu, 1999) .

However, simulating FBSDEs is costly due to the need to estimate conditional expectations in the backward pass.

Estimating conditional expectations, however, is a direct consequence of the appearance of an auxiliary process from the Martingale representation theorem (Pardoux and Peng, 1992) .

For notational convenience we suppress z and W · .

Bounding I 1 .

Let > 0 be given.

Since G h → G in probability, there exist M 1 > 0 and h 0 > 0 such that

Since the SDE defines a stochastic flow of diffeomorphisms, there exists a finite random variable C 2 such that sup |z|≤2M 1 |∇ z F| ≤ C 2 , and there exists M 2 > 0 such that P(|C 2 | > M 2 ) < .

Given M 2 , there exists h 1 > 0 such that

Now suppose h ≤ min{h 0 , h 1 }.

Then, by the union bound, with probability at least 1 − 4 , we have

On this event, we have

Thus, we have shown that I 1 converges to 0 in probability as h → 0.

Bounding I 2 .

The idea is similar.

By condition (ii), we have

in probability.

Using this and condition (i), for given > 0, there exist M > 0 and h 0 > 0 such that for h ≥ h 0 , we have

with probability at least 1 − .

On this event, we have

Thus I 2 also converges to 0 in probability.

Recall the Stratonovich SDE (7) with drift and diffusion functions b, σ 1 , . . .

, σ m ∈ R d × R → R d governed by a set of parameters θ ∈ R p .

Consider the augmented state composed of the original state and parameters Y t = (Z t , θ ) .

The augmented state satisfies a Stratonovich SDE with the drift function f (y, t) = (b(z, t) , 0 p ) and diffusion functions

.

By (10) and (3), the dynamics for the adjoint process of the augmented state is characterized by the backward SDE:

By definitions of f and g i , the Jacobian matrices ∇f (x, s) and ∇g i (x, s) can be written as:

Thus, we can write out the backward SDEs for the adjoint processes of the state and parameters separately:

Now assume the original SDE has diagonal noise.

Then, m = d and Jacobian matrix ∇σ i (z) can be written as:

Consider the adjoint process for the augmented state along with the backward flow of the backward SDE (8).

We write the overall state as X t = (Z t , (A z t ) , (A θ t ) ) , where we abuse notation slightly to let {Z t } t∈T denote the backward flow process.

Then, by (14) and (15), {X t } t∈T satisfies a backward SDE with a diffusion function that can be written as:

Recall, for an SDE with diffusion function Σ(x) ∈ R d×m , it is said to satisfy the commutativity property (Rößler, 2004)

When an SDE has commutative noise, the computationally intensive double Itô integrals (and the Lévy areas) need not be simulated by having the numerical scheme take advantage of the following property of iterated integrals (Ilie et al., 2015) :

where the Brownian motion increment

To see that the diffusion function (16) indeed satisfies the commutativity condition (17), we consider several cases:

• k = 1, . . . , d: Both LHS and RHS are zero unless j 1 = j 2 = k, since for Σ i,j 2 (x)

to be non-zero, i = j 1 = j 2 = k.

• k = d + 1 . . .

, 2d: Similar to the case above.

• k = 2d + 1 . . .

, 2d + p:

.

Both LHS and RHS are zero unless

Since in all scenarios, LHS = RHS, we conclude that the commutativity condition holds.

Finally, we comment that the Milstein scheme for the stochastic adjoint of diagonal noise SDEs can be implemented such that during each iteration of the backward solve, vjp is only called a number of times constant with respect to the dimensionality of the original SDE.

r e t u r n ans @staticmethod d e f backward ( ctx , * grad_outputs ) : t s , flat_params_f , flat_params_g , * ans = c t x .

s a v e d _ t e n s o r s f , g , dt , bm = c t x .

f , c t x .

g , c t x .

dt , c t x .

bm f_params , g_params = t u p l e ( f .

p a r a m e t e r s ( ) ) , t u p l e ( g .

p a r a m e t e r s ( ) ) n _ t e n s o r s = l e n ( ans ) # Accumulate g r a d i e n t s a t i n t e r m e d i a t e p o i n t s .

adj_y = _sequence_add ( adj_y , t u p l e ( grad_outputs_ [ i − 1 ] f o r grad_outputs_ i n grad_outputs ) ) r e t u r n ( * adj_y , None , None , None , adj_params_f , adj_params_g , None , None )

In the following, α, β, and p are parameters of SDEs, and x 0 is a fixed initial value.

Example 1.

Analytical solution:

Example 2.

Analytical solution:

Example 3.

Analytical solution:

In each numerical experiment, we duplicate the equation 10 times to obtain a system of SDEs where each dimension had their own parameter values sampled from the standard Gaussian distribution and then passed through a sigmoid to ensure positivity.

Moreover, we also sample the initial value for each dimension from a Gaussian distribution.

We consider training latent SDE models with our adjoint framework to recover (1) a 1D Geometric Brownian motion, and (2) a 3D stochastic Lorenz attractor process.

The main objective is to verify that the learned posterior is able to reconstruct the training data, and the learned prior exhibit stochastic behavior.

We jointly optimize the variational free energy (6) with respect to parameters of the prior and posterior distributions at the initial latent state z 0 , the prior and posterior drift, the diffusion function, the encoder, and the decoder.

We include the details of dataset and architecture in Appendix N.1.

For the stochastic Lorenz attractor, not only is the model able to reconstruct the data well, but also the learned prior process can produce bimodal samples in both data and latent space.

This is showcased in the last row of Figure 4 , where once the initial position sampled from the learned prior distribution is fixed, the latent and data space samples cluster around two modes.

Note that this cannot be achieved by a latent ODE, where trajectories are determined once their initial latent state is determined.

See Figure 4 for additional visualization on the synthetic Lorenz attractor dataset.

See Figure 5 for visualization on the synthetic geometric Brownian motion dataset.

We comment that for the second example, the posterior reconstructs the data well, and the prior process exhibit behavior of the data.

However, from the third row, we can observe that the prior process is learned such that most of the uncertainty is account for in the initial latent state.

We leave the investigation of more interpretable prior process for future work.

Consider a geometric Brownian motion SDE: We use µ = 1, σ = 0.5, and x 0 = 0.1 + as the ground-truth model, where ∼ N(0, 0.03 2 ).

We sample 1024 time series, each of which is observed at intervals of 0.02 from time 0 to time 1.

We corrupt this data using Gaussian noise with mean zero and standard deviation 0.01.

To recover the dynamics, we use a GRU-based Cho et al. (2014) latent SDE model where the GRU has 1 layer and 100 hidden units, the prior and posterior drift functions are MLPs with 1 hidden layer of 100 units, and the diffusion function is an MLP with 1 hidden layer of 100 hidden units and the sigmoid activation applied at the end.

The drift function in the posterior is time-inhomogenous in the sense that it takes in a context vector of size 1 at each observation that is output by the GRU from running backwards after processing all future observations.

The decoder is a linear mapping from a 4 dimensional latent space to observation space.

For all nonlinearities, we use the softplus function.

We fix the observation model to be Gaussian with noise standard deviation 0.01.

We optimize the model jointly with respect to the parameters of a Gaussian distribution for initial latent state distribution, the prior and posterior drift functions, the diffusion function, the GRU encoder, and the decoder.

We use a fixed discretization with step size of 0.01 in both the forward and backward pass.

We use the Adam optimizer Kingma and Ba (2014) with an initial learning rate of 0.01 that is decay by a factor of 0.999 after each iteration.

We use a linear KL annealing schedule over the first 50 iterations.

Consider a stochastic Lorenz attractor SDE with diagonal noise: dX t =σ (Y t − X t ) dt + α x dW t , X 0 = x 0 , dY t = (X t (ρ − Z t ) − Y t ) dt + α y dW t , Y 0 = y 0 , dZ t = (X t Y t − βZ t ) dt + α z dW t , Z 0 = z 0 .

We use σ = 10, ρ = 28, β = 8/3, (α x , α y , α z ) = (.1, .28., .3), and (x 0 , y 0 , z 0 ) sampled from the standard Gaussian distribution as the ground-truth model.

We sample 1024 time series, each of which is observed at intervals of 0.025 from time 0 to time 1.

We normalize these samples by their mean and standard deviation across each dimension and corrupt this data by Gaussian noise with mean zero and standard deviation 0.01.

We use the same architecture and training procedure for the latent SDE model as in the geometric Brownian motion section, except that the diffusion function consists of four small neural networks, each for a single dimension of the latent SDE.

We follow the preprocessing used by Wang et al. (2007) .

Following Yıldız et al. (2019) , we use a fully connected network to encode the first three observations of each sequence and thereafter predicted the remaining sequence.

Note the current choice of encoder is for comparing fairly to models in the existing literature, and it may be extended to be a recurrent or attention model Vaswani et al. (2017) to enhance performance.

The overall architecture is described in Appendix O and is similar to that of ODE 2 VAE Yıldız et al. (2019) with a similar number of parameters.

We also use a fixed step size that is 1/5 of smallest interval between any two observations Yıldız et al. (2019) .

We train latent ODE and latent SDE models with the Adam optimizer Kingma and Ba (2014) and its default hyperparameter settings, with an initial learning rate of 0.01 that is exponentially decayed with rate 0.999 during each iteration.

We perform validation over the number of training iterations, KL penalty Higgins et al. (2017) , and KL annealing schedule.

All models were trained for at most 400 iterations, where we start to observe severe overfitting for most model instances.

We use a latent SDE model with an MLP encoder which takes in the first three frames and outputs the mean and log-variance of the variational distribution of the initial latent state and a context vector.

The decoder has a similar architecture as that for the ODE 2 VAE model Yıldız et al. (2019) and projects the 6-dimensional latent state into the 50-dimensional observation space.

The posterior drift function takes in a 3-dimensional context vector output by the encoder and the current state and time, whereas the prior drift only takes in the current state and time.

The diffusion function is composed of multiple small neural nets, each producing a scalar for the corresponding dimension such that the posterior SDE has diagonal noise.

We comment that the overall parameter count of our model (11605) is smaller than that of ODE 2 VAE for the same task (12157).

The latent ODE baseline was implemented with a similar architecture, except is does not have the diffusion and prior drift components, and its vector field defining the ODE does not First row from left to right are the encoder and decoder.

Second row from left to right are the prior drift, posterior drift, and diffusion functions.

take in a context vector.

Therefore, the model has slightly fewer parameters (10573) than the latent SDE model.

See Figure 6 for overall details of the architecture.

The main hyperparameter we tuned was the coefficient for reweighting the KL.

For both the latent ODE and SDE, we considered training the model with a reweighting coefficient in {1, 0.1, 0.01, 0.001}, either with or without a linear KL annealing schedule that increased from 0 to the prescribed value over the first 200 iterations of training.

<|TLDR|>

@highlight

We present a constant memory gradient computation procedure through solutions of stochastic differential equations (SDEs) and apply the method for learning latent SDE models.
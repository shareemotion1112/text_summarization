This paper proposes the use of spectral element methods \citep{canuto_spectral_1988} for fast and accurate training of Neural Ordinary Differential Equations (ODE-Nets; \citealp{Chen2018NeuralOD}) for system identification.

This is achieved by expressing their dynamics as a truncated series of Legendre polynomials.

The series coefficients, as well as the network weights, are computed by minimizing the weighted sum of the loss function and the violation of the ODE-Net dynamics.

The problem is solved by coordinate descent that alternately minimizes, with respect to the coefficients and the weights, two unconstrained sub-problems using standard backpropagation and gradient methods.

The resulting optimization scheme is fully time-parallel and results in a low memory footprint.

Experimental comparison to standard methods, such as backpropagation through explicit solvers and the adjoint technique \citep{Chen2018NeuralOD}, on training surrogate models of small and medium-scale dynamical systems shows that it is at least one order of magnitude faster at reaching a comparable value of the loss function.

The corresponding testing MSE is one order of magnitude smaller as well, suggesting generalization capabilities increase.

Neural Ordinary Differential Equations (ODE-Nets; Chen et al., 2018) can learn latent models from observations that are sparse in time.

This property has the potential to enhance the performance of neural network predictive models in applications where information is sparse in time and it is important to account for exact arrival times and delays.

In complex control systems and model-based reinforcement learning, planning over a long horizon is often needed, while high frequency feedback is necessary for maintaining stability (Franklin et al., 2014) .

Discrete-time models, including RNNs (Jain & Medsker, 1999) , often struggle to fully meet the needs of such applications due to the fixed time resolution.

ODE-Nets have been shown to provide superior performance with respect to classic RNNs on time series forecasting with sparse training data.

However, learning their parameters can be computationally intensive.

In particular, ODE-Nets are memory efficient but time inefficient.

In this paper, we address this bottleneck and propose a novel alternative strategy for system identification.

We propose SNODE, a compact representation of ODE-Nets for system identification with full state information that makes use of a higher-order approximation of its states by means of Legendre polynomials.

This is outlined in Section 4.

In order to find the optimal polynomial coefficients and network parameters, we develop a novel optimization scheme, which does not require to solve an ODE at each iteration.

The resulting algorithm is detailed in Section 3 and is based on backpropagation (Linnainmaa, 1970; Werbos, 1981; Lecun, 1988) and automatic differentiation (Paszke et al., 2017) .

The proposed method is fully parallel with respect to time and its approximation error reduces exponentially with the Legendre polynomial order (Canuto et al., 1988) .

Summary of numerical experiments.

In Section 5, our method is tested on a 6-state vehicle problem, where it is at least one order or magnitude faster in each optimizer iteration than explicit and adjoint methods, while convergence is achieved in a third of the iterations.

At test time, the MSE is reduced by one order of magnitude.

In Section 6, the method is used for a 30-state system consisting of identical vehicles, coupled via a known collision avoidance policy.

Again, our method converges in a third of the iterations required by backpropagation thourgh a solver and each iteration is 50x faster than the fastest explicit scheme.

The minimization of a scalar-valued loss function that depends on the output of an ODE-Net can be formulated as a general constrained optimization problem:

where x(t) ∈ X is the state, u(t) ∈ U is the input, the loss and ODE functions L and f are given, and the parameters θ have to be learned.

The spaces X and U are typically Sobolev (e.g. Hilbert) spaces expressing the smoothness of x(t) and u(t) (see Section 8).

Equation (1) can be used to represent several inverse problems, for instance in machine learning, estimation, and optimal control (Stengel, 1994; Law et al., 2015; Ross, 2009) .

Problem (1) can be solved using gradient-based optimization through several time-stepping schemes for solving the ODE. (Chen et al., 2018; Gholami et al., 2019) have proposed to use the adjoint method when f is a neural network.

These methods are typically relying on explicit time-stepping schemes (Butcher & Wanner, 1996) .

Limitations of these approaches are briefly summarized:

Limitations of backpropagation through an ODE solver.

The standard approach for solving this problem is to compute the gradients ∂L/∂θ using backpropagation through a discrete approximation of the constraints, such as Runge-Kutta methods (Runge, 1895; Butcher & Wanner, 1996) or multistep solvers .

This ensures that the solution remains feasible (within a numerical tolerance) at each iteration of a gradient descent method.

However, it has several drawbacks: 1) the memory cost of storing intermediate quantities during backpropagation can be significant, 2) the application of implicit methods would require solving a nonlinear equation at each step, 3) the numerical error can significantly affect the solution, and 4) the problem topology can be unsuitable for optimization (Petersen et al., 2018) .

Limitations of adjoint methods.

ODE-Nets (Chen et al., 2018) solve (1) using the adjoint method, which consists of simulating a dynamical system defined by an appropriate augmented Hamiltonian (Ross, 2009) , with an additional state referred to as the adjoint.

In the backward pass the adjoint ODE is solved numerically to provide the gradients of the loss function.

This means that intermediate states of the forward pass do not need to be stored.

An additional step of the ODE solver is needed for the backward pass.

This suffers from a few drawbacks: 1) the dynamics of either the hidden state or the adjoint might be unstable, due to the symplectic structure of the underlying Hamiltonian system, referred to as the curse of sensitivity in (Ross, 2009) ; 2) the procedure requires solving a differential algebraic equation and a boundary value problem which is complex, time consuming, and might not have a solution (Ross & Karpenko, 2012) .

Limitations of hybrid methods.

ANODE (Gholami et al., 2019) splits the problem into time batches, where the adjoint is used, storing in memory only few intermediate states from the forward pass.

This allows to improve the robustness and generalization of the adjoint method.

A similar improvement could be obtained using reversible integrators.

However, its computational cost is of the same order of the adjoint method and it does not offer further opportunities for parallelization.

Our algorithm is based on two ingredients: i) the discretization of the problem using spectral elements leading to SNODE, detailed in Section 4, and ii) the relaxation of the ODE constraint from (1), enabling efficient training through backpropagation.

The latter can be applied directly at the continuous level and significantly reduces the difficulty of the optimization, as shown in our examples.

The problem in (1) is split into two smaller subproblems: one finds the trajectory x(t) that minimizes an unconstrained relaxation of (1).

The other trains the network weights θ such that the trajectory becomes a solution of the ODE.

Both are addressed using standard gradient descent and backpropagation.

In particular, a fixed number of ADAM or SGD steps is performed for each problem in an alternate fashion, until convergence.

In the following, the details of each subproblem are discussed.

Step 0: Initial trajectory.

The initial trajectory x(t) is chosen by solving the problem

If this problem does not have a unique solution, a regularization term is added.

For a quadratic loss, a closed-form solution is readily available.

Otherwise, a prescribed number of SGD iterations is used.

Step 1: Coordinate descent on residual.

Once the initial trajectory x (t) is found, θ is computed by solving the unconstrained problem:

If the value of the residual at the optimum θ * is smaller than a prescribed tolerance, then the algorithms stops.

Otherwise, steps 1 and 2 are iterated until convergence.

Step 2: Coordinate descent on relaxation.

Once the candidate parameters θ are found, the trajectory is updated by minimizing the relaxed objective:

Discussion.

The proposed algorithm can be seen as an alternating coordinate gradient descent on the relaxed functional used in problem (4), i.e., by alternating a minimization with respect to x(t) and θ.

If γ = 0, multiple minima can exist, since each choice of the parameters θ would induce a different dynamics x(t), solution of the original constraint.

For γ = 0, the loss function in (4) trades-off the ODE solution residual for the data fitting, providing a unique solution.

The choice of γ implicitly introduces a satisfaction tolerance (γ), i.e., similar to regularized regression (Hastie et al., 2001) , implying that ẋ(t) − f (t, x(t); θ) ≤ (γ).

Concurrently, problem (3) reduces the residual.

In order to numerically solve the problems presented in the previous section, a discretization of x(t) is needed.

Rather than updating the values at time points t i from the past to the future, we introduce a compact representation of the complete discrete trajectory by means of the spectral element method.

Spectral approximation.

We start by representing the scalar unknown trajectory, x(t), and the known input, u(t), as truncated series:

where x i , u i ∈ R and ψ i (t), ζ i (t) are sets of given basis functions that span the spaces X h ⊂ X and U h ⊂ U. In this work, we use orthogonal Legendre polynomials of order p (Canuto et al., 1988) for

, where p is a hyperparameter, and the cosine Fourier basis for ζ i (t), where z is fixed.

Collocation and quadrature.

In order to compute the coefficients x i of (5), we enforce the equation at a discrete set Q of collocation points t q .

Here, we choose p + 1 Gauss-Lobatto nodes, which include t = t 0 .

This directly enforces the initial condition.

Other choices are also possible (Canuto et al., 1988) .

Introducing the vectors of series coefficients

and of evaluations at quadrature points x(t Q ) = {x(t q )} q∈Q , the collocation problem can be solved in matrix form as

We approximate the integral (3) as a sum of residual evaluations over Q. Assuming that x(0) = x 0 , the integrand at all quadrature points t Q can be computed as a component-wise norm

Fitting the input data.

For the case when problem (2) admits a unique a solution, we propose a new direct training scheme, δ-SNODE, which is summarized in Algorithm 1.

In general, a least-squares approach must be used instead.

This entails computing the integral in (2), which can be done by evaluating the loss function L at quadrature points t q .

If the input data is not available at t q , we approximate the integral by evaluating L at the available time points.

The corresponding alternating coordinate descent scheme α-SNODE is presented in Algorithm 2.

In the next sections, we study the consequences of a low-data scenario on this approach.

We use fixed numbers N t and N x of updates for, respectively, θ and x(t).

Both are performed with standard routines, such as SGD.

In our experiments, we use ADAM to optimize the parameters and an interpolation order p = 14, but any other orders and solvers are possible.

Input: M, D from (6)- (7).

] end for end while Output: θ * Ease of time parallelization.

If R(t q ) = 0 is enforced explicitly at q ∈ Q, then the resulting discrete system can be seen as an implicit time-stepping method of order p.

However, while ODE integrators can only be made parallel across the different components of x(t), the assembly of the residual can be done in parallel also across time.

This massively increases the parallelization capabilities of the proposed schemes compared to standard training routines.

Memory cost.

If an ODE admits a regular solution, with regularity r > p, in the sense of Hilbert spaces, i.e., of the number of square-integrable derivatives, then the approximation error of the SNODE converges exponentially with p (Canuto et al., 1988) .

Hence, it produces a very compact representation of an ODE-Net.

Thanks to this property, p is typically much lower than the equivalent number of time steps of explicit or implicit schemes with a fixed order.

This greatly reduces the complexity and the memory requirements of the proposed method, which can be evaluated at any t via (5) by only storing few x i coefficients.

Stability and vanishing gradients.

The forward Euler method is known to have a small region of convergence.

In other words, integrating very fast dynamics requires a very small time step, dt, in order to provide accurate results.

In particular, for the solver error to be bounded, the eigenvalues of the state Jacobian of the ODE need to lie into the circle of the complex plane centered at (−1, 0) with radius 1/dt (Ciccone et al., 2018; Isermann, 1989) .

Higher-order explicit methods, such as Runge-Kutta (Runge, 1895) , have larger but still limited convergence regions.

Our algorithms on the other hand are implicit methods, which have a larger region of convergence than recursive (explicit) methods (Hairer et al., 1993) .

We claim that this results in a more stable and robust training.

This claim is supported by our experiments.

Reducing the time step can improve the Euler accuracy but it can still lead to vanishing or exploding gradients (Zilly et al., 2016; Goodfellow et al., 2016) .

In Appendix C, we show that our methods do not suffer from this problem.

Experiments setup and hyperparameters.

For all experiments, a common setup was employed and no optimization of hyperparameters was performed.

Time horizon T = 10s and batch size of 100 were used.

Learning rates were set to 10 −2 for ADAM (for all methods) and 10 −3 for SGD (for α-SNODE).

For the α-SNODE method, γ = 3 and 10 iterations were used for the SGD and ADAM algorithms at each epoch, as outlined in Algorithm 2.

The initial trajectory was perturbed as x 0 = x 0 + ξ, ξ ∼ U (−0.1, 0.1).

This perturbation prevents the exact convergence of Algorithm 1 during initialization, allowing to perform the alternating coordinate descent algorithm.

Let us consider the systemη

where η, v ∈ R 3 are the states, u = (F x , 0, τ xy ) is the control, C(v) is the Coriolis matrix, d(v) is the (linear) damping force, and J(η) encodes the coordinate transformation from the body to the world frame (Fossen, 2011) .

A gray-box model is built using a neural network for each matrix

Each network consists of two layers, the first with a tanh activation.

Bias is excluded for f C and f d .

For f J , sin(φ) and cos(φ) are used as input features, where φ is the vehicle orientation.

When inserted in (8), these discrete networks produce an ODE-Net that is a surrogate model of the physical system.

The trajectories of the system and the learning curves are shown in Appendix A. Comparison of methods in the high-data regime.

In the case of δ-SNODE and α-SNODE, only p + 1 points are needed for the accurate integration of the loss function, if such points coincide with the Gauss-Lobatto quadrature points.

We found that 100 equally-spaced points produce a comparable result.

Therefore, the training performance of the novel and traditional training methods were compared by sampling the trajectories at 100 equally-spaced time points.

Table 1a shows that δ-SNODE outperforms BKPR-DoPr5 by a factor of 50, while producing a significantly improved generalization.

The speedup reduces to 20 for α-SNODE, which however yields a further reduction of the testing MSE by a factor of 10, as can be seen in Figure 1 .

Comparison in the low-data regime.

The performance of the methods was compared using fewer time points, randomly sampled from a uniform distribution.

For the baselines, evenly-spaced points were used.

Table 1b shows that α-SNODE preserves a good testing MSE, at the price of an increased number of iterations.

With only 25% of data, α-SNODE is 10x faster than BKPR-DoPr5.

Moreover, its test MSE is 1/7 than BKPR-DoPr5 and up to 1/70 than BKPR-Euler, showing that the adaptive time step of DoPr5 improves significantly the baseline but it is unable to match the accuracty of the proposed methods.

The adjoint method produced the same results as the backprop (±2%).

Consider the multi-agent system consisting of N a kinematic vehicles:

where η ∈ R 3Na are the states (planar position and orientation), J(η i ) is the coordinate transform from the body to world frame, common to all agents.

The agents velocities are determined by known arbitrary control and collision avoidance policies, respectively, K c and K o plus some additional high frequency measurable signal w = w(t), shared by all vehicles.

The control laws are non-linear and are described in detail in Appendix B. We wish to learn their kinematics matrix by means of a neural network as in Section 5.

The task is simpler here, but the resulting ODE has 3N a states, coupled by K 0 .

We simulate N a = 10 agents in series.

Comparison of methods with full and sparse data.

The learning curves for high-data regime are in Figure 3 .

For method α-SNODE, training was terminated when the loss in (4) is less than γL +R, withL = 0.11 andR = 0.01.

For the case of 20% data, we setL = 0.01.

Table 2 summarizes results.

δ-SNODE is the fastest method, followed by α-SNODE which is the best performing.

Iteration time of BKPR-Euler is 50x slower, with 14x worse test MSE.

ADJ-Euler is the slowest but its test MSE is in between BKPR-Euler and our methods.

Random down-sampling of the data by 50% and 20% (evenly-spaced for the baselines) makes ADJ-Euler fall back the most.

BKPR-DoPr5 failed to find a time step meeting the tolerances, therefore they were increased to rtol= 10 −5 , atol= 10 −7 .

Since the loss continued to increase, training was terminated at 200 epochs.

ADJ-DoPr5 failed to compute gradients.

Test trajectories are in Figure 2 .

Additional details are in Appendix B. Robustness of the methods.

The use of a high order variable-step method (DoPr5), providing an accurate ODE solution, does not however lead to good training results.

In particular, the loss function continued to increase over the iterations.

On the other hand, despite being nearly 50 times slower than our methods, the fixed-step forward Euler solver was successfully used for learning the dynamics of a 30-state system in the training configuration described in Appendix B. One should however note that, in this configuration, the gains for the collision avoidance policy K o (which couples the ODE) were set to small values.

This makes the system simpler and more stable than having a larger gain.

As a result, if one attempts to train with the test configuration from Appendix B, where the gains are increased and the system is more unstable, then backpropagating trough Euler simply fails.

Comparing Figures 3 and 4 , it can be seen that the learning curves of our methods are unaffected by the change in the gains, while BKPR-Euler and ADJ-Euler fail to decrease the loss.

RNN training pathologies.

One of the first RNNs to be trained successfully were LSTMs (Greff et al., 2017) , due to their particular architecture.

Training an arbitrary RNN effectively is generally difficult as standard RNN dynamics can become unstable or chaotic during training and this can cause the gradients to explode and SGD to fail (Pascanu et al., 2012) .

When RNNs consist of discretised ODEs, then stability of SGD is intrinsically related to the size of the convergence region of the solver (Ciccone et al., 2018) .

Since higher-order and implicit solvers have larger convergence region (Hairer et al., 1993) , following (Pascanu et al., 2012) it can be argued that our method has the potential to mitigate instabilities and hence to make the learning more efficient.

This is supported by our results.

Unrolled architectures.

In (Graves, 2016) , an RNN has been used with a stopping criterion, for iterative estimation with adaptive computation time.

Highway (Srivastava et al., 2015) and residual networks (He et al., 2015) have been studied in (Greff et al., 2016) as unrolled estimators.

In this context, (Haber & Ruthotto, 2017) treated residual networks as autonomous discrete-ODEs and investigated their stability.

Finally, in (Ciccone et al., 2018) a discrete-time non-autonomous ODE based on residual networks has been made explicitly stable and convergent to an input-dependant equilibrium, then used for adaptive computation.

Training stable ODEs.

In (Haber & Ruthotto, 2017; Ciccone et al., 2018) , ODE stability conditions where used to train unrolled recurrent residual networks.

Similarly, when using our method on (Ciccone et al., 2018) ODE stability can be enforced by projecting the state weight matrices, A, into the Hurwitz stable space: i.e. A ≺ 0.

At test time, overall stability will also depend on the solver (Durran, 2010; Isermann, 1989) .

Therefore, a high order variable step method (e.g. DoPr5) should be used at test time in order to minimize the approximation error.

Dynamics and machine learning.

A physics prior on a neural network was used by (Jia et al., 2018) in the form of a consistency loss with data from a simulation.

In (De Avila Belbute-Peres et al., 2018) , a differentiable physics framework was introduced for point mass planar models with contact dynamics. (Ruthotto & Haber, 2018) looked at Partial Differential Equations (PDEs) to analyze neural networks, while Raissi et al., 2017) used Gaussian Processes (GP) to model PDEs.

The solution of a linear ODE was used in (Soleimani et al., 2017) in conjunction with a structured multi-output GP to model patients outcome of continuous treatment observed at random times. (Pathak et al., 2017) predicted the divergence rate of a chaotic system with RNNs.

Test time and cross-validation At test time, since the future outputs are unknown, an explicit integrator is needed.

For cross-validation, the loss needs instead to be evaluated on a different dataset.

In order to do so, one needs to solve the ODE forward in time.

However, since the output data is available during cross-validation, a corresponding polynomial representation of the form (5) can be found and the relaxed loss (4) can be evaluated efficiently.

Nonsmooth dynamics.

We have assumed that the ODE-Net dynamics has a regularity r > p in order to take advantage of the exponential convergence of spectral methods, i.e., that their approximation error reduces as O(h p ), where is h is the size of the window used to discretize the interval.

However, this might not be true in general.

In these cases, the optimal choice would be to use a hp-spectral approach (Canuto et al., 1988) , where h is reduced locally only near the discontinuities.

This is very closely related to adaptive time-stepping for ODE solvers.

Topological properties, convergence, and better generalization.

There are few theoretical open questions stemming from this work.

We argue that one reason for the performance improvement shown by our algorithms is the fact that the set of functions generated by a fixed neural network topology does not posses favorable topological properties for optimization, as discussed in (Petersen et al., 2018) .

Therefore, the constraint relaxation proposed in this work may improve the properties of the optimization space.

This is similar to interior point methods and can help with accelerating the convergence but also with preventing local minima.

One further explanation is the fact that the proposed method does not suffer from vanishing nor exploding gradients, as shown in Appendinx C. Moreover, our approach very closely resembles the MAC scheme, for which theoretical convergence results are available (Carreira-Perpinan & Wang, 2014) .

Multiple ODEs: Synchronous vs Asynchronous.

The proposed method can be used for an arbitrary cascade of dynamical systems as they can be expressed as a single ODE.

When only the final state of one ODE (or its trajectory) is fed into the next block, e.g. as in (Ciccone et al., 2018) , the method could be extended by means of 2M smaller optimizations, where M is the number of ODEs.

Hidden states.

Latent states do not appear in the loss, so training and particularly initializing the polynomial coefficients is more difficult.

A hybrid approach is to warm-start the optimizer using few iterations of backpropagation.

We plan to investigate a full spectral approach in the future.

The model is formulated in a concentrated parameter form (Siciliano et al., 2008) .

We follow the notation of (Fossen, 2011) .

Recall the system definition:

where η, v ∈ R 3 are the states, namely, the x, and y coordinates in a fixed (world) frame, the vehicle orientation with respect this this frame, φ, and the body-frame velocities, v x , v y , and angular rate, ω.

The input is a set of torques in the body-frame, u = (F x , 0, τ xy ).

The Kinematic matrix is (Butcher & Wanner, 1996) with backpropagation.

First three columns: comparison of true trajectories (red) with the prediction from the surrogate (black) at different iterations of the optimization.

Last column: loss function at each iteration.

δ-SNODE has faster convergence than Euler.

The multi-agent simulation consists of N a kinematic vehicles:

where η ∈ R 3Na are the states for each vehicle, namely, the x i , y i positions and the orientation φ i of vehicle i in the world frame, while v i ∈ R 2Na are the controls signals, in the form of linear and angular velocities, ν i , ω i .

The kinematics matrix is J(η i ) = cos(φ i ) 0 sin(φ i ) 0 0 1 .

@highlight

This paper proposes the use of spectral element methods for fast and accurate training of Neural Ordinary Differential Equations for system identification.
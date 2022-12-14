Direct policy gradient methods for reinforcement learning and continuous control problems are a popular approach for a variety of reasons:  1) they are easy to implement without explicit knowledge of the underlying model; 2) they are an "end-to-end" approach, directly optimizing the performance metric of interest; 3) they inherently allow for richly parameterized policies.

A notable drawback is that even in the most basic continuous control problem (that of linear quadratic regulators), these methods must solve a non-convex optimization problem, where little is understood about their efficiency from both computational and statistical perspectives.

In contrast, system identification and model based planning in optimal control theory have a much more solid theoretical footing, where much is known with regards to their computational and statistical properties.

This work bridges this gap showing that (model free) policy gradient methods globally converge to the optimal solution and are efficient (polynomially so in relevant problem dependent quantities) with regards to their sample and computational complexities.

Recent years have seen major advances in the control of uncertain dynamical systems using reinforcement learning and data-driven approaches; examples range from allowing robots to perform more sophisticated controls tasks such as robotic hand manipulation (Tassa et al., 2012; BID1 Kumar et al., 2016; Levine et al., 2016; Tobin et al., 2017; Rajeswaran et al., 2017a) , to sequential decision making in game domains, e.g. AlphaGo (Silver et al., 2016) and Atari game playing (Mnih et al., 2015) .

Deep reinforcement learning (DeepRL) are becoming increasingly popular for tackling such challenging sequential decision making problems.

Many of these successes have relied on sampling based reinforcement learning algorithms such as policy gradient methods, including the DeepRL approaches; here, there is little theoretical understanding of their efficiency, either from a statistical or a computational perspective.

In contrast, control theory (optimal and adaptive control) has a rich body of tools, with provable guarantees, for related sequential decision making problems, particularly those that involve continuous control.

These latter techniques are often model-based -they estimate an explicit dynamical model first (e.g. system identification) and then design optimal controllers.

This work builds bridges between these two lines of work, namely, between optimal control theory and sample based reinforcement learning methods, using ideas from mathematical optimization.

In the standard optimal control problem, the dynamics model f t , where f t is specified as x t+1 = f t (x t , u t , w t ) , maps a state x t ??? R d , a control (the action) u t ??? R k , and a disturbance w t , to the next state x t+1 ??? R d .

The objective is to find the control input u t which minimizes the long term cost, minimize T t=1 c t (x t , u t )such that x t+1 = f t (x t , u t , w t ) .Here the u t are allowed to depend on the history of observed states.

In practice, this is often solved by considering the linearized control (sub-)problem where the dynamics are approximated by x t+1 = A t x t + B t u t + w t , and the costs are approximated by a quadratic function in x t and u t , e.g. (Todorov & Li, 2004) .

This work considers an important special case: the time homogenous, infinite horizon problem referred to as the linear quadratic regulator (LQR) problem.

The results herein can also be extended to the finite horizon, time in-homogenous setting, discussed in Section 5.In the LQR problem, the objective is minimize E ??? t=0 (x t Qx t + u t Ru t )such that x t+1 = Ax t + Bu t , x 0 ??? D .where initial state x 0 ??? D is assumed to be randomly distributed according to distribution D; the matrices A ??? R d??d and B ??? R d??k are referred to as system (or transition) matrices; Q ??? R d??d and R ??? R k??k are both positive definite matrices that parameterize the quadratic costs.

For clarity, this work does not consider a noise disturbance but only a random initial state.

The importance of (some) randomization for analyzing direct methods is discussed in Section 3.Throughout, assume that A and B are such that the optimal cost is finite (for example, the controllability of the pair (A, B) would ensure this).

Optimal control theory BID2 BID13 BID5 BID6 shows that the optimal control input can be written as a linear function in the state, DISPLAYFORM0 Planning with a known model.

Planning can be achieved by solving the algebraic Riccati equation, DISPLAYFORM1 for a positive definite matrix P which parameterizes the "cost-to-go" (the optimal cost from a state going forward).

The optimal control gain is then given as: DISPLAYFORM2 There are both algebraic solution methods to find P and (convex) SDP formulations to solve for P .

More broadly, even though there are convex formulations for planning, these formulations: 1) do not directly parameterize the policy 2)

they are not "end-to-end" approaches in that they are not directly optimizing the cost function of interest and 3) it is not immediately clear how to utilize these approaches in the model-free setting, where the agent only has simulation access.

These formulations are discussed in Section A, where there is a discussion of how the standard SDP formulation is not a direct method that minizes the cost over the set of feasible policies.

Even in the most basic case of the standard linear quadratic regulator model, little is understood as to how direct (model-free) policy gradient methods fare.

This work provides rigorous guarantees, showing that, while in fact the approach is a non-convex one, directly using (model free) local search methods leads to finding the globally optimal policy.

The main contributions are as follows:??? (Exact case) Even with access to exact gradient evaluation, little is understood about whether or not convergence to the optimal policy occurs, even in the limit, due to the non-convexity in the problem.

This work shows that global convergence does indeed occur (and does so efficiently) for local search based methods.??? (Model free case) Without a model, this work shows how one can use simulated trajectories (as opposed to having knowledge of the model) in a stochastic policy gradient method where provable convergence to a globally optimal policy is guaranteed, with (polynomially) efficient computational and sample complexities.??? (The natural policy gradient) Natural policy gradient methods BID18 )

-and related algorithms such as Trust Region Policy Optimization (Schulman et al., 2015) and the natural actor critic (Peters & Schaal, 2007)

-are some of the most widely used and effective policy gradient methods (see BID12 ).

While many results argue in favor of this method based on either information geometry BID18 BID3 or based on connections to actor-critic methods BID11 , these results do not provably show an improved convergence rate.

This work is the first to provide a guarantee that the natural gradient method enjoys a considerably improved convergence rate over its naive gradient counterpart.

More broadly, the techniques in this work merge ideas from optimal control theory, mathematical (and zeroth order) optimization, and sample based reinforcement learning methods.

These techniques may ultimately help in improving upon the existing set of algorithms, addressing issues such as variance reduction or improving upon the natural policy gradient method (with, say, a GaussNewton method).

The Discussion touches upon some of these issues.

In the reinforcement learning setting, the model is unknown, and the agent must learn to act through its interactions with the environment.

Here, solution concepts are typically divided into: modelbased approaches, where the agent attempts to learn a model of the world, and model-free approaches, where the agent directly learns to act and does not explicitly learn a model of the world.

The related work on provably learning LQRs is reviewed from this perspective.

Model-based learning approaches.

In the context of LQRs, the agent attempts to learn the dynamics of "the plant" (i.e. the model) and then plans, using this model, for control synthesis.

Here, the classical approach is to learn the model with subspace identification (Ljung, 1999) .

BID14 provides a provable learning (and non-asymptotic) result, where the quality of the policy obtained is shown to be near optimal (efficiency is in terms of the persistence of the training data and the controllability Gramian).

BID0 also provides provable, nonasymptotic learning results (in a regret context), using a bandit algorithm that achieves lower sample complexity (by balancing exploration-exploitation more effectively); the computational efficiency of this approach is less clear.

More recently, BID10 expands on an explicit system identification process, where a robust control synthesis procedure is adopted that relies on a coarse model of the plant matrices (A and B are estimated up to some accuracy level, naturally leading to a "robust control" setup).

Arguably, this is the most general (and non-asymptotic) result, that is efficient from both a statistical perspective (computationally, the method works with a finite horizon to approximate the infinite horizon).

This result only needs the plant to be controllable; the work herein needs the stronger assumption that the initial policy in the local search procedure is a stable controller (an assumption which may be inherent to local search procedures, discussed in Section 5).Model-free learning approaches.

Model-free approaches that do not rely on an explicit system identification step typically either: 1) estimate value functions (or state-action values) through Monte Carlo simulation which are then used in some approximate dynamic programming variant (Bertsekas, 2011) or 2) directly optimize a (parameterized) policy, also through Monte Carlo simulation.

Model-free approaches for learning optimal controllers is not well understood, from a theoretical perspective.

Here, BID7 provides an asymptotic learnability result using a value function approach, namely Q-learning.

This work seeks to characterize the behavior of (direct) policy gradient methods, where the policy is linearly parameterized, as specified by a matrix K ??? R k??d which generates the controls: DISPLAYFORM0 The cost of this K is denoted as: DISPLAYFORM1 where {x t , u t } is the trajectory induced by following K, starting with x 0 ??? D. The importance of (some) randomization, either in x 0 or noise through having a disturbance, for analyzing gradient methods is discussed in Section 3.

Here, K * is a minimizer of C(??).Gradient descent on C(K), with a fixed stepsize ??, follows the update rule: DISPLAYFORM2 It is helpful to explicitly write out the functional form of the gradient.

Define P K as the solution to: DISPLAYFORM3 and, under this definition, it follows that C(K) can be written as: DISPLAYFORM4 Also, define ?? K as the (un-normalized) state correlation matrix, i.e. DISPLAYFORM5 Lemma 1. (Policy Gradient Expression) The policy gradient is: DISPLAYFORM6 Observe: DISPLAYFORM7 This implies: DISPLAYFORM8 x t x t using recursion and that x 1 = (A ??? BK)x 0 .

Taking expectations completes the proof.

Sample based policy gradient methods introduce some randomization for estimating the gradient.

REINFORCE.

Let ?? ?? (u|x) be a parametric stochastic policy, where u ??? ?? ?? (??|x).

The policy gradient of the cost, C(??), is: DISPLAYFORM0 where the expectation is with respect to the trajectory {x t , u t } induced under the policy ?? ?? and where Q ?? ?? (x, u) is referred to as the state-action value.

The REINFORCE algorithm uses Monte Carlo estimates of the gradient obtained by simulating ?? ?? .The natural policy gradient.

The natural policy gradient BID18 ) follows the update: DISPLAYFORM1 where G ?? is the Fisher information matrix.

There are numerous succesful related approaches (Peters & Schaal, 2007; Schulman et al., 2015; BID12 ).

An important special case is using a linear policy with additive Gaussian noise (Rajeswaran et al., 2017b) DISPLAYFORM2 where K ??? R k??d and ?? 2 is the noise variance.

Here, the natural policy gradient of K (when ?? is considered fixed) takes the form: DISPLAYFORM3 To see this, one can verify that the Fisher matrix of size kd ?? kd, which is indexed as DISPLAYFORM4 where i, i ??? {1, . . .

k} and j, j ??? {1, . . .

d}, has a block diagonal form where the only non-zeros blocks are [G K ] (i,??),(i,??) = ?? K (this is the block corresponding to the i-th coordinate of the action, as i ranges from 1 to k).

This form holds more generally, for any diagonal noise.

Zeroth order optimization.

Zeroth order optimization is a generic procedure BID9 Nesterov & Spokoiny, 2015) for optimizing a function f (x), using only query access to the function values of f (??) at input points x (and without explicit query access to the gradients of f ).

This is also the approach in using "evolutionary strategies" (Salimans et al., 2017) .

The generic approach can be described as follows: define the perturbed function as DISPLAYFORM5 For small ??, the smooth function is a good approximation to the original function.

Due to the Gaussian smoothing, the gradient has the particularly simple functional form (see BID9 Nesterov & Spokoiny (2015) ): DISPLAYFORM6 .

This expression implies a straightforward method to obtain an unbiased estimate of the ???f ?? 2 (x), through obtaining only the function values f (x + ??) for random ??.

This section provides a brief characterization of the optimization landscape, in order to help provide intuition as to why global convergence is possible and as to where the analysis difficulties lie.

Lemma 2. (Non-convexity) If d ??? 3, there exists an LQR optimization problem, min K C(K), which is not convex, quasi-convex, and star-convex.

Section B provides a specific example.

In general, for a non-convex optimization problem, gradient descent may not even converge to the global optima in the limit.

For the case of LQRs, the following corollary (of Lemma 8) provides a characterization of the stationary points.

Corollary 3. (Stationary point characterization) If ???C(K) = 0, then either K is an optimal policy or ?? K is rank deficient.

This lemma is the motivation for using a distribution over x 0 (as opposed to a deterministic starting point): E x0???D x 0 x 0 being full rank guarantees that ?? K is full rank, which implies all stationary points are a global optima.

An additive disturbance in the dynamics model also suffices.

The concept of gradient domination is important in the non-convex optimization literature (Polyak, 1963; Nesterov & Polyak, 2006; Karimi et al., 2016) .

A function f : R d ??? R is said to be gradient dominated if there exists some constant ??, such that for all x, DISPLAYFORM0 If a function is gradient dominated, this implies that if the magnitude of the gradient is small at some x, then the function value at x will be close to that of the optimal function value.

The following corollary of Lemma 8 shows that C(K) is gradient dominated.

DISPLAYFORM1 where ?? is a problem dependent constant (and ??, ?? denotes the trace inner product).With gradient domination and no (spurious) local optima, one may hope that recent results on escaping saddle points (Nesterov & Polyak, 2006; BID16 BID17 immediately imply that gradient descent converges quickly.

This is not the case due to that it is not straightforward to characterize the (local) smoothness properties of C(K); this is a difficulty well studied in the optimal control theory literature, related to robustness and stability.

In fact, if it were the case that C(K) is a smooth function 1 (in addition to being gradient dominated), then classical mathematical optimization results (Polyak, 1963) would not only immediately imply global convergence, these results would also imply convergence at a linear rate.

First, results on exact gradient methods are provided.

From an analysis perspective, this is the natural starting point; once global convergence is established for exact methods, the question of using simulation-based, model-free methods can be approached with zeroth-order optimization methods.

Notation.

Z denotes the spectral norm of a matrix Z; Tr(Z) denotes the trace of a square matrix; ?? min (Z) denotes the minimal singular value of a square matrix Z. Also, it is helpful to define DISPLAYFORM0

The following three exact update rules are considered: DISPLAYFORM0 Natural policy gradient descent: DISPLAYFORM1 Gauss-Newton: DISPLAYFORM2 Kn .

(7) The natural policy gradient descent direction is defined so that it is consistent with the stochastic case, as per Equation 4.

It is straightforward to verify that the policy iteration algorithm is a special case of the Gauss-Newton method when ?? = 1 (for the case of policy iteration, convergence in the limit is provided in Todorov & Li ( The Gauss-Newton method requires the most complex oracle to implement: it requires access to ???C(K), ?? K , and R+B P K B; it also enjoys the strongest convergence rate guarantee.

At the other extreme, gradient descent requires oracle access to only ???C(K) and has the slowest convergence rate.

The natural policy gradient sits in between, requiring oracle access to ???C(K) and ?? K , and having a convergence rate between the other two methods.

Theorem 5. (Global Convergence of Gradient Methods) Suppose C(K 0 ) is finite and and ?? > 0.??? Gauss-Newton case: For a stepsize ?? = 1 and for DISPLAYFORM3 the Gauss-Newton algorithm (Equation 7) enjoys the following performance bound: DISPLAYFORM4 ??? Natural policy gradient case: For a stepsize DISPLAYFORM5 and for DISPLAYFORM6 natural policy gradient descent (Equation 6) enjoys the following performance bound: DISPLAYFORM7 Algorithm 1 Model-Free Policy Gradient (and Natural Policy Gradient) Estimation 1: Input: K, number of trajectories m, roll out length , smoothing parameter r, dimension d 2: DISPLAYFORM8 Sample a policy K i = K + U i , where U i is drawn uniformly at random over matrices whose (Frobenius) norm is r.

Simulate K i for steps starting from x 0 ??? D. Let C i and ?? i be the empirical estimates: DISPLAYFORM0 where c t and x t are the costs and states on this trajectory.

5: end for 6: Return the (biased) estimates: DISPLAYFORM1 ??? Gradient descent case: For an appropriate (constant) setting of the stepsize ??, DISPLAYFORM2 and for DISPLAYFORM3 , gradient descent (Equation 5) enjoys the following performance bound: DISPLAYFORM4 In comparison to model-based approaches, these results require the (possibly) stronger assumption that the initial policy is a stable controller, i.e. C(K 0 ) is finite (an assumption which may be inherent to local search procedures).

The Discussion mentions this as direction of future work.

In the model free setting, the controller has only simulation access to the model; the model parameters, A, B, Q and R, are unknown.

The standard optimal control theory approach is to use system identification to learn the model, and then plan with this learned model This section proves that model-free, policy gradient methods also lead to globally optimal policies, with both polynomial computational and sample complexities (in the relevant quantities).Using a zeroth-order optimization approach (see Section 2.2), Algorithm 1 provides a procedure to find (controllably biased) estimates, ???C(K) and ?? K , of both ???C(K) and ?? K .

These can then be used in the policy gradient and natural policy gradient updates as follows: DISPLAYFORM0 Natural policy gradient descent: DISPLAYFORM1 where Algorithm 1 is called at every iteration to provide the estimates of ???C(K n ) and ?? Kn .The choice of using zeroth order optimization vs using REINFORCE (with Gaussian additive noise, as in Equation 3) is primarily for technical reasons 2 .

It is plausible that the REINFORCE estimation procedure has lower variance.

One additional minor difference, again for technical reasons, is that Algorithm 1 uses a perturbation from the surface of a sphere (as opposed to a Gaussian perturbation).Theorem 6. (Global Convergence in the Model Free Setting) Suppose C(K 0 ) is finite, ?? > 0, and that x 0 ??? D has norm bounded by L almost surely.

Also, for both the policy gradient method and the natural policy gradient method, suppose Algorithm 1 is called with parameters: DISPLAYFORM2 ??? Natural policy gradient case:

For a stepsize DISPLAYFORM3 and for DISPLAYFORM4 then, with high probability, i.e. with probability greater than 1 ??? exp(???d), the natural policy gradient descent update (Equation 9) enjoys the following performance bound: DISPLAYFORM5 ??? Gradient descent case: For an appropriate (constant) setting of the stepsize ??, DISPLAYFORM6 and if N satisfies DISPLAYFORM7 , then, with high probability, gradient descent (Equation 8) enjoys the following performance bound: DISPLAYFORM8

This work has provided provable guarantees that model-based gradient methods and model-free (sample based) policy gradient methods convergence to the globally optimal solution, with finite polynomial computational and sample complexities.

Taken together, the results herein place these popular and practical policy gradient approaches on a firm theoretical footing, making them comparable to other principled approaches (e.g. subspace ID methods and algebraic iterative approaches).Finite C(K 0 ) assumption, noisy case, and finite horizon case.

These methods allow for extensions to the noisy case and the finite horizon case.

This work also made the assumption that C(K 0 ) is finite, which may not be easy to achieve in some infinite horizon problems.

The simplest way to address this is to model the infinite horizon problem with a finite horizon one; the techniques developed in Section D.1 shows this is possible.

This is an important direction for future work.

Open Problems.??? Variance reduction: This work only proved efficiency from a polynomial sample size perspective.

An interesting future direction would be in how to rigorously combine variance reduction methods and model-based methods to further decrease the sample size.??? A sample based Gauss-Newton approach: This work showed how the Gauss-Newton algorithm improves over even the natural policy gradient method, in the exact case.

A practically relevant question for the Gauss-Newton method would be how to both: a) construct a sample based estimator b) extend this scheme to deal with (non-linear) parametric policies.??? Robust control: In model based approaches, optimal control theory provides efficient procedures to deal with (bounded) model mis-specification.

An important question is how to provably understand robustness in a model free setting.

This section briefly reviews some parameterizations and solution methods for the classic LQR and related problems from control theory.

Finite horizon LQR.

First, consider the finite horizon case.

The basic approach is to view it as a dynamic program with the value function x T t P t x t , where DISPLAYFORM0 which in turn gives optimal control DISPLAYFORM1 Another approach is to view the LQR problem as a linearly-constrained Quadratic Program in all x t and u t (where the constraints are given by the dynamics, and the problem size equals the horizon).The QP is clearly a convex problem, but this observation is not useful by itself as the problem size grows with the horizon, and naive use of quadratic programming scales badly.

However, the special structure due to linear dynamics allows for simplifications and control-theoretic interpretation as follows: the Lagrange multipliers can be interpreted as "co-state" variables, and they follow a recursion that runs backwards in time known as the "adjoint system".

Using Lagrange duality, one can show that this approach is equivalent to solving the Riccati recursion mentioned above.

Popular use of the LQR in control practice is often in the receding horizon LQR, BID8 ; Rawlings & Mayne (2009): at time t, an input sequence is found that minimizes the T -step ahead LQR cost starting at the current time, then only the first input in the sequence is used.

The resulting static feedback gain converges to the infinite horizon optimal solution as horizon T becomes longer.

Infinite horizon LQR.

Here, the constrained optimization view (QP) is not informative as the problem is infinite dimensional; the dynamic programming viewpoint readily extends.

Suppose the system A, B is controllable (which guarantees optimal cost is finite).

It turns out that the value function and the optimal controller are static (do not depend on t) and can be found by solving the Algebraic Riccati Equation (ARE) given in (1).

The optimal K can then be found from equation (2).The main computational step is solving the ARE, which is extensively studied (e.g. (Lancaster & Rodman, 1995) ).

One approach due to Kleinman (1968) is to simply run the recursion DISPLAYFORM2 T P k A with P 1 = Q, which converges to the unique positive semidefinite solution of the ARE (since the fixed-point iteration is contractive).

Other approaches are direct and based on linear algebra, which carry out an eigenvalue decomposition on a certain block matrix followed by a matrix inversion (Lancaster & Rodman, 1995) .Direct computation of the control input has also been considered in the optimal control literature, e.g., gradient updates in function spaces (Polak, 1973) .

For the linear quadratic setup, direct iterative computation of the feedback gain has been examined in (M??rtensson & Rantzer, 2009) , and explored further in (M??rtensson, 2012) with a view towards distributed implementations.

There methods are presented as local search heuristics without provable guarantees of reaching the optimal policy.

SDP formulation.

The LQR problem can also be expressed as a semidefinite program (SDP) with variable P , as given in (Balakrishnan & Vandenberghe, 2003) (section 5, equation (34) , this is for a continuous-time system but there are similar discrete-time versions).

This SDP can be derived by relaxing the equality in the Riccati equation to an inequality, then using the Schur complement formula to rewrite the resulting Riccati inequality as linear matrix inequality; the objective in the case of LQR is the trace of the positive definite matrix variable.

This formulation and its dual has been explored in BID4 .It is important to note that while the optimal solution of this SDP is the unique positive semidefinite solution to the Riccati equation, which in turn gives the optimal policy K * , other feasible P (not equal to P * ) do not necessarily correspond to a feasible, stabilizing policy K. This means that the feasible set of this SDP is not a convex characterization of all P that correspond to stabilizing K. Thus it also implies that if one uses any optimization algorithm that maintains iterates in the feasible set (e.g. interior point methods), no useful policy can be extracted from the iterates before convergence to P * .

For this reason, this convex formulation is not helpful for parametrizing the space of policies K in manner that supports the use of local search methods (those that directly lower the cost function of interest), which is the focus of this work.

Let K(A, B) denote the set of state feedback gains K such that A ??? BK is stable, i.e., its eigenvalues are inside the unit circle in the complex plane.

This set is generally nonconvex.

A concise counterexample to convexity is provided here.

Let A and B be 3 ?? 3 identity matrices and

This section provides the analysis of the convergence rates of the (exact) gradient based methods.

First, some helpful lemmas for the analysis are provided.

Throughout, it is convenient to use the following definition: DISPLAYFORM0 The policy gradient can then be written as: DISPLAYFORM1

Define the value V K (x), the state-action value Q K (x, u), and the advantage A K (x, u).

V K (x, t) is the cost of the policy starting with x 0 = x and proceeding with K onwards: DISPLAYFORM0 is the cost of the policy starting with x 0 = x, taking action u 0 = u and then proceeding with K onwards: DISPLAYFORM1 The advantage can be viewed as the change in cost starting at state x and taking a one step deviation from the policy K.The next lemma is identical to that in BID19 BID20 for Markov decision processes.

Lemma 7. (Cost difference lemma) Suppose K and K have finite costs.

Let {x t } and {u t } be state and action sequences generated by K , i.e. starting with x 0 = x and using u t = ???K x t .

It holds that: DISPLAYFORM2 Also, for any x, the advantage is: DISPLAYFORM3 Proof.

Let c t be the cost sequence generated by K .

Telescoping the sum appropriately: DISPLAYFORM4 which completes the first claim.

For the second claim, observe that: DISPLAYFORM5 And, for u = K x, DISPLAYFORM6 which completes the proof.

This lemma is helpful in proving that C(K) is gradient dominated.

Lemma 8. (Gradient domination) Let K * be an optimal policy.

Suppose K has finite cost and ?? > 0.

It holds that: DISPLAYFORM7 For a lower bound, it holds that: DISPLAYFORM8 Proof.

From Equation 10 and by completing the square, DISPLAYFORM9 with equality when DISPLAYFORM10 Let x * t and u * t be the sequence generated under K * .

Using this and Lemma 7, DISPLAYFORM11 which completes the proof of the upper bound.

For the lower bound, consider DISPLAYFORM12 ???1 E K where equality holds in Equation 11.

Let x t and u t be the sequence generated under K .

Using that DISPLAYFORM13 which completes the proof.

Recall that a function f is said to be smooth (or C 1 -smooth) if it satisfies for some finite ??, it satisfies: DISPLAYFORM14 for all x, y (equivalently, it is smooth if the gradients of f are continuous).

Lemma 9. ("Almost" smoothness) C(K) satisfies: DISPLAYFORM15 To see why this is related to smoothness (e.g. compare to Equation 12), suppose K is sufficiently close to K so that: DISPLAYFORM16 and the leading order term 2Tr(?? K (K ??? K) E K ) would then behave as Tr((K ??? K) ???C(K)).

The challenge in the proof (for gradient descent) is quantifying the lower order terms in this argument.

Proof.

The claim immediately results from Lemma 7, by using Equation 10 and taking an expectation.

The next lemma spectral norm bounds on P K and ?? K are helpful: Lemma 10.

It holds that: DISPLAYFORM17 Proof.

For the first claim, C(K) is lower bounded as: DISPLAYFORM18 Alternatively, C(K) can be lower bounded as: DISPLAYFORM19 which proves the second claim.

The next lemma bounds the one step progress of Gauss-Newton.

Lemma 11.

Suppose that: DISPLAYFORM0 ???1 E K .

Using Lemma 9 and the condition on ??, DISPLAYFORM1 where the last step uses Lemma 8.With this lemma, the proof of the convergence rate of the Gauss Newton algorithm is immediate.

Proof. (of Theorem 5, Gauss-Newton case) The theorem is due to that ?? = 1 leads to a contraction of 1 ??? ???? ?? K * at every step.

The next lemma bounds the one step progress of the natural policy gradient.

Lemma 12.

Suppose: DISPLAYFORM0 .

It holds that: DISPLAYFORM1 Proof.

Since K = K ??? ??E K , Lemma 9 implies: DISPLAYFORM2 The last term can be bounded as: DISPLAYFORM3 Continuing and using the condition on ??, DISPLAYFORM4 using Lemma 8.With this lemma, the proof of the natural policy gradient convergence rate can be completed.

Proof. (of Theorem 5, natural policy gradient case) Using Lemma 10, DISPLAYFORM5 The proof is completed by induction: DISPLAYFORM6 , since Lemma 12 can be applied.

The proof proceeds by arguing that Lemma 12 can be applied at every step.

If it were the case that DISPLAYFORM7 and by Lemma 12: DISPLAYFORM8 which completes the proof.

As informally argued by Equation 13, the proof seeks to quantify how ?? K changes with ??.

Then the proof bounds the one step progress of gradient descent.

DISPLAYFORM0 This subsections aims to prove the following: Lemma 13. (?? K perturbation) Suppose K is such that: DISPLAYFORM1 It holds that: DISPLAYFORM2 The proof proceeds by starting with a few technical lemmas.

First, define a linear operator on symmetric matrices, T K (??), which can be viewed as a matrix on d+1 2dimensions.

Define this operator on a symmetric matrix X as follows: DISPLAYFORM3 Also define the induced norm of T as follows: DISPLAYFORM4 where the supremum is over all symmetric matrices X (whose spectral norm is non-zero).Also, define DISPLAYFORM5 Proof.

For a unit norm vector v ??? R d and unit spectral norm matrix X, DISPLAYFORM6 The proof is completed using the upper bound on ?? K in Lemma 10.Also, with respect to K, define another linear operator on symmetric matrices: DISPLAYFORM7 Let I to denote the identity operator on the same space.

Define the induced norm ?? of these operators as in Equation 14.

Note these operators are related to the operator T K as follows:Lemma 15.

When (A ??? BK) has spectral radius smaller than 1, DISPLAYFORM8 Proof.

When (A ??? BK) has spectral radius smaller than 1, T K is well defined and is the solution of DISPLAYFORM9 Since, DISPLAYFORM10 The proof of Lemma 13 seeks to bound: DISPLAYFORM11 The following two perturbation bounds are helpful in this.

Lemma 16.

It holds that: DISPLAYFORM12 Proof.

Let ??? = K ??? K .

For every matrix X, DISPLAYFORM13 The operator norm of F K ??? F K is the maximum possible ratio in spectral norm of (F K ??? F K )(X) and X. Then the claim follows because AX ??? A X .

DISPLAYFORM14 Proof.

Define A = I ??? F K , and DISPLAYFORM15 Observe: DISPLAYFORM16 .

DISPLAYFORM17 and so DISPLAYFORM18 This proves the main inequality.

The last step of the inequality is just applying definition of the norm of DISPLAYFORM19 With these Lemmas, the proof is completed as follows:

Proof. (of Lemma 13) First, the proof shows T K F K ??? F K ??? 1/2, which is the desired condition in Lemma 17.

First, observe that under the assumed condition on K ??? K , implies that DISPLAYFORM20 using that DISPLAYFORM21 ??? 1 due to Lemma 10.

Using Lemma 16, DISPLAYFORM22 Using this and Lemma 14, DISPLAYFORM23 where the last step uses the condition on K ??? K .Thus, DISPLAYFORM24 using Lemmas 10 and 16.

Equipped with these lemmas, the one step progress of gradient descent can be bounded.

Lemma 18.

Suppose that DISPLAYFORM0 (16) It holds that: DISPLAYFORM1 Proof.

By Lemma 9, DISPLAYFORM2 where the last step uses Lemma 8.By Lemma 13, DISPLAYFORM3 using the assumed condition on ??.

Using this last claim and Lemma 10, DISPLAYFORM4 using the condition on ??.

In order to prove a gradient descent convergence rate, the following bounds are helpful: Lemma 19.

It holds that DISPLAYFORM5 and that: DISPLAYFORM6 Proof.

Using Lemma 10, DISPLAYFORM7 By Lemma 8, DISPLAYFORM8 ?? which proves the first claim.

Again using Lemma 8, DISPLAYFORM9 which proves the second claim.

With these lemmas, the proof of the gradient descent convergence rate follows:Proof. (of Theorem 5, gradient descent case) First, the following argues that progress is made at t = 1.

Based on Lemma 10 and Lemma 19, by choosing ?? to be an appropriate polynomial in DISPLAYFORM10 , ?? min (Q) and ??, the stepsize condition in Equation 16 is satisfied.

Hence, by Lemma 18, DISPLAYFORM11 which implies that the cost decreases at t = 1.

Proceeding inductively, now suppose that C(K t ) ??? C(K 0 ), then the stepsize condition in Equation 16 is still satisfied (due to the use of C(K 0 ) in bounding the quantities in Lemma 19).

Thus, Lemma 18 can again be applied for the update at time t + 1 to obtain: DISPLAYFORM12 ??? ??, and the result follows.

This section shows how techniques from zeroth order optimization allow the algorithm to run in the model-free setting with only black-box access to a simulator.

The dependencies on various parameters are not optimized, and the notation h is used to represent different polynomial factors in the relevant factors ( DISPLAYFORM0 ????min(Q) , A , B , R , 1/?? min (R)).

When the polynomial also depend on dimension d or accuracy 1/ , this is specified as parameters (h FIG2 ).The section starts by showing how the infinite horizon can be approximated with a finite horizon.

This section shows that as long as there is an upper bound on C(K), it is possible to approximate both C(K) and ??(K) with any desired accuracy.

Lemma 20.

For any K with finite C(K), let ?? DISPLAYFORM0 Proof.

First, the bound on ?? K is proved.

Define the operators T K and F K as in Section C.4, observe DISPLAYFORM1 , this follows immediately from the form of F K (X) = (A + BK)X(A + BK) .

If X is PSD then W XW is also PSD for any W .

Now, since tr( DISPLAYFORM2 (Here the last step is by Lemma 10), and all traces are nonnegative, then there must exists j < such that tr( DISPLAYFORM3 Therefore as long as DISPLAYFORM4 , it follows that: DISPLAYFORM5 Here the first step is again because of all the terms are PSD, so using more terms is always better.

The last step follows because F j (?? K ) is also a PSD matrix so the spectral norm is bounded by trace.

In fact, it holds that tr( DISPLAYFORM6 Therefore if DISPLAYFORM7

The next lemma show that the function value and its gradient are approximate preserved if a small perturbation to the policy K is applied.

Lemma 21. (C K perturbation) Suppose K is such that: DISPLAYFORM0 , K then: DISPLAYFORM1 As in the proof of Lemma 16, the assumption implies that T K F K ??? F K ??? 1/2, and, from Equation 15, that DISPLAYFORM2 Hence, DISPLAYFORM3 For the first term, DISPLAYFORM4 Combining the two terms completes the proof.

The next lemma shows the gradient is also stable after perturbation.

Lemma 22. (???C K perturbation) Suppose K is such that: DISPLAYFORM5 Let's first look at the second term.

By Lemma 8, DISPLAYFORM6 then by Lemma 13 DISPLAYFORM7 Therefore the second term is bounded by DISPLAYFORM8 By the previous lemma, DISPLAYFORM9 Since K ??? 2 K , and K can be bounded by C(K) (Lemma 19), all the terms can be bounded by polynomials of related parameters multiplied by K ??? K .

This section analyzes the smoothing procedure and completes the proof of gradient descent.

Although Gaussian smoothing is more standard, the objective C(K) is not finite for every K, therefore technically E u???N (0,?? 2 I) [C(K + u)] is not well defined.

This is avoidable by smoothing in a ball.

Let S r represent the uniform distribution over the points with norm r (boundary of a sphere), and B r represent the uniform distribution over all points with norm at most r (the entire sphere).

When applying these sets to matrix a U , the Frobenius norm ball is used.

The algorithm performs gradient descent on the following function DISPLAYFORM0 The next lemma uses the standard technique (e.g. in BID15 ) to show that the gradient of C r (K) can be estimated just with an oracle for function value.

DISPLAYFORM1 This is the same as Lemma 2.1 in BID15 , for completeness the proof is provided below.

Proof.

By Stokes formula, DISPLAYFORM2 By definition, DISPLAYFORM3 Under review as a conference paper at ICLR 2018Also, DISPLAYFORM4 The Lemma follows from combining these equations, and use the fact that DISPLAYFORM5 From the lemma above and standard concentration inequalities, it is immediate that it suffices to use a polynomial number of samples to approximate the gradient.

Lemma 24.

Given an , there are fixed polynomials h r (1/ ), h sample (d, 1/ ) such that when r ??? 1/h r (1/ ), with m ??? h sample (d, 1/ ) samples of U 1 , ..., U n ??? S r , with high probability (at least DISPLAYFORM6 is also close to ???C(K) with high probability.

Proof.

For the first part, the difference is broken into two terms: DISPLAYFORM7 For the first term, choose h r (1/ ) = min{1/r 0 , 2h grad / } (r 0 is chosen later).

By Lemma 22 when r is smaller than 1/h r (1/ ) = /2h grad , every point u on the sphere have ???C(K + U ) ??? ???C(K) ??? /4.

Since ???C r (K) is the expectation of ???C(K + U ), by triangle inequality ???C r (K) ??? ???C(K) ??? /2.The proof also makes sure that r ??? r 0 such that for any U ??? S r , it holds that C(K + U ) ??? 2C(K).

By Lemma 21, 1/r 0 is a polynomial in the relevant factors.

Adding these two terms and apply triangle inequality gives the result.

For the second part, the proof breaks it into more terms.

Let ??? be equal to The third term is just what was bounded earlier, using h r,trunc (1/ ) = h r (2/ ) and making sure h sample,trunc (d, 1/ ) ??? h sample (d, 2/ ), this guarantees that it is smaller than /2.For the second term, choose ??? Therefore,??? ??? ??? is again a sum of independent vectors with bounded norm, so by Vector Bernstein's inequality, when h sample,trunc (d, 1/ , L 2 /??) is a large enough polynomial, ??? ??? ??? ??? /4 with high probability.

Adding all the terms finishes the proof.

Theorem 25.

There are fixed polynomials h GD,r (1/ ), h GD,sample (d, 1/ , L 2 /??), h GD, (d, 1/ ) such that if every step the gradient is computed as Lemma 24 (truncated at step ), pick step size ?? and T the same as the gradient descent case of Theorem 5, it holds that C(K T ) ??? C(K ) ??? with high probability (at least 1 ??? exp(???d)).

Before the Theorem for natural gradient is proven, the following lemma shows the variance can be estimated accurately.

satisfies ?? ??? ?? K ??? .

Further, when ??? ??/2, it holds that ?? min (?? K ) ??? ??/2.Proof.

This is broken into three terms: let ?? Therefore, standard concentration bounds show that when h varsample,trunc is a large enough polynomial, ?? ????? ( ) ??? /2 holds with high probability.

Adding these three terms gives the result.

Finally, the bound on ?? min (?? K ) follows simply from Weyl's Theorem.

Theorem 27.

Suppose C(K 0 ) is finite and and ?? > 0.

The natural gradient follows the update rule: DISPLAYFORM0 Suppose the stepsize is set to be: DISPLAYFORM1 If the gradient and variance are estimated as in Lemma 24, Lemma 26 with r = 1/h N GD,r (1/ ), with m ??? h N GD,sample (d, 1/ , L 2 /??) samples, both are truncated to h N GD, (d, 1/ ) iterations, then with high probability (at least 1 ??? exp(???d)) in T iterations where DISPLAYFORM2 ???? min (R) log 2(C(K 0 ) ??? C(K * ))

?? then the natural gradient satisfies the following performance bound: DISPLAYFORM3

<|TLDR|>

@highlight

This paper shows that model-free policy gradient methods can converge to the global optimal solution for non-convex linearized control problems.
Su-Boyd-Candes (2014) made a connection between Nesterov's method and an ordinary differential equation (ODE).

We show if a Hessian damping term is added to the ODE from Su-Boyd-Candes (2014), then Nesterov's method arises as a straightforward discretization of the modified ODE.

Analogously,  in the strongly convex case, a Hessian damping term is added to Polyak's ODE, which is then discretized to yield Nesterov's method for strongly convex functions.

Despite the Hessian term, both second order ODEs can be represented as first order systems.



Established Liapunov analysis is used to recover the accelerated rates of convergence in both continuous and discrete time.

Moreover, the Liapunov analysis can be extended to the case of stochastic gradients which allows the full gradient case to be considered as a special case of the stochastic case.

The result is a unified approach to convex acceleration in both continuous and discrete time and in  both the stochastic and full gradient cases.

Su et al. (2014) made a connection between Nesterov's method for a convex, L-smooth function, f , and the second order, ordinary differential equation (ODE) x + 3 tẋ + ∇f (x) = 0 (A-ODE)However Su et al. (2014) did not show that Nesterov's method arises as a discretization of (A-ODE).

In order to obtain such a discretization, we consider the following ODE, which has an additional Hessian damping term with coefficient 1/ √ L. DISPLAYFORM0 Notice that (H-ODE) is a perturbation of (A-ODE), and the perturbation goes to zero as L → ∞. Similar ODEs have been studied by BID1 , they have been shown to accelerate gradient descent in continuous time in .Next, we consider the case where f is also µ-strongly convex, and write C f := L/µ for the condition number of f .

Then Nesterov's method in the strongly convex case arises as discretization of the following second order ODË DISPLAYFORM1 (H-ODE-SC) is a perturbation of Polyak's ODE (Polyak, 1964) x + 2 √ µẋ + ∇f (x) = 0 which is accelerates gradient when f is quadratic see (Scieur et al., 2017) .In each case, both continuous and discrete, as well and convex and strongly convex, it is possible to provide a proof of the rate using a Liapunov function.

These proofs are already established in the literature: we give citations below, and also provide proof in the Appendix.

Moreover, the analysis for Nesterov's method in the full gradient can be extended to prove acceleration in the case of stochastic gradients.

Acceleration of stochastic gradient descent has been established by Lin et al. (2015) and BID7 , see also BID8 .

A direct acceleration method with a connection to Nestero'v method was done by BID0 .

Our analysis unifies the continuous time ODE with the algorithm, and includes full gradient acceleration as a special case.

The analysis proceeds by first rewriting (H-ODE) (and (H-ODE-SC)) as first order systems involving ∇f , and then replacing the ∇f with g = ∇f + e. Both the continuous and discrete time methods achieve the accelerated rate of convergence, provided |e| goes to zero quickly enough.

The condition on |e|, is given below in (12) and (13) -it is faster than the corresponding rate for stochastic gradient descent.

When e = 0 we recover the full gradient case.

The renewed interested in the continuous time approach began with the work of Su et al. (2014) and was followed Wibisono et al. (2016); Wilson et al. (2016) .

Continuous time analysis also appears in BID6 , BID11 , and BID10 .

However, continuous time approaches to optimization have been around for a long time.

Polyak's method Polyak (1964) is related to successive over relaxation for linear equations (Varga, 1957) which were initially used to accelerate solutions of linear partial differential equations (Young, 1954) .

A continuous time interpretation of Newton's method can be found in (Polyak, 1987) or BID1 .

The mirror descent algorithm of Nemirovskii et al. (1983) has a continuous time interpretation BID5 .

The Liapunov approach for acceleration had already appeared in BID4 for FISTA.The question of when discretizations of dynamical systems also satisfy a Liapunov function has been studied in the context of stabilization in optimal control BID12 .

More generally, Stuart & Humphries (1996) studies when a discretization of a dynamical system preserves a property such as energy dissipation.

Despite the Hessian term, (H-ODE-SC) can be represented as the following first order system.

Lemma 2.1.

The second order ODE (H-ODE) is equivalent to the first order system DISPLAYFORM0 Proof.

Solve for v in the first line of (1st-ODE) DISPLAYFORM1 Insert into the second line of (1st-ODE) DISPLAYFORM2 Simplify to obtain (H-ODE).The system (1st-ODE) can be discretized using the forward Euler method with a constant time step, h, to obtain Nesterov's method.

Definition 2.2.

Define y k as the following convex combination of x k and v k .

DISPLAYFORM3 Let h > 0 be a given small time step/learning rate and let t k = h(k + 2).

The forward Euler method for (1st-ODE) with gradients evaluated at y k is given by DISPLAYFORM4 Remark 2.3.

The forward Euler method simply comes from replacingẋ with (x k+1 − x k )/h and similarly for v. Normally the velocity field is simply evaluated at x k , v k .

The only thing different about (FE-C) from the standard forward Euler method is that ∇f is evaluated at y k instead of x k .

However, this is still an explicit method.

More general multistep methods and one leg methods in this context are discussed in Scieur et al. (2017) .Recall the standard Nesterov's method from Nesterov (2013, Section 2.2) DISPLAYFORM5 Theorem 2.4.

The discretization of (H-ODE) given by (FE-C)(1) with h = 1/ √ L and t k = h(k+2) is equivalent to the standard Nesterov's method (Nest).

Eliminate the variable v using (1) to obtain (Nest).

Now we consider µ-strongly convex, and L-smooth functions, f , and write C f := L µ for the condition number.

We first show that (H-ODE-SC) can be represented as a first order system.

Lemma 2.5.

The second order ODE (H-ODE-SC) is equivalent to the first order system DISPLAYFORM0 Proof.

Solve for v in the first line of (1st-ODE-SC) DISPLAYFORM1 Insert into the second line of (1st-ODE-SC) DISPLAYFORM2 Simplify to obtain (H-ODE-SC).System (1st-ODE-SC) can be discretized using a forward Euler method with a constant time step h to obtain Nesterov's method.

Let h > 0 be a small time step, and apply the forward Euler method for (1st-ODE-SC) evaluated at y k : DISPLAYFORM3 where, DISPLAYFORM4 Now we recall the usual Nesterov's method for strongly convex functions from Nesterov (2013, Section 2.2) DISPLAYFORM5 Theorem 2.6.

The discretization of (H-ODE-SC) given by (FE-SC) with h = 1/ √ L is equivalent to the standard Nesterov's method (SC-Nest).

Eliminate the variable v k using the definition of y k to obtain (SC-Nest).3 LIAPUNOV ANALYSIS 3.1 CONVEX CASE: CONTINUOUS AND DISCRETE TIME Definition 3.1.

Define the continuous time Liapunov function DISPLAYFORM0 where E(t, x, v) in given by (3).

In particular, for all t > 0, DISPLAYFORM1 Furthermore, let x k , v k be given by (FE-C).

Then for all k ≥ 0, DISPLAYFORM2 In particular, if DISPLAYFORM3 then E k is decreasing.

When equality holds in (5), DISPLAYFORM4 Most of the results stated above are already known, but for completeness we refer the proofs in Appendix A. Since (FE-C) is equivalent to Nesterov's method, the rate is known.

The proof of the rate using a Liapunov function can be found in BID4 .

Refer to ?

which shows that we can use the constant time step.

The discrete Liapunov function (4) was used in Su et al. FORMULA5 ; to prove a rate.3.2 STRONGLY CONVEX CASE: CONTINUOUS AND DISCRETE TIME Definition 3.3.

Define the continuous time Liapunov function E(x, v) DISPLAYFORM5 Define the discrete time Liapunov function by DISPLAYFORM6 Proposition 3.4.

Let (x, v) be the solution of (1st-ODE-SC), then DISPLAYFORM7 In particular, for all t > 0, DISPLAYFORM8 , we have DISPLAYFORM9 In particular, for h = DISPLAYFORM10 The discrete Liapunov function E k was used to prove a rate in the strongly convex case by Wilson et al. (2016) .

The proof of (10) can be found in Wilson et al. (2016, Theorem 6) .

For completeness we also provide the proof in Appendix E.

In the appendix we present results in continuous and discrete time for (non-accelerated) stochastic gradient descent.

We also present results in continuous time for the stochastic accelerated case in the Appendix.

We present the results in discrete time here.

In this section we consider stochastic gradients, which we write as a gradient plus an error term DISPLAYFORM0 The stochastic gradient can be abstract, or it can error be a mini-batch gradient when f is a sum.

Moreover, we can include the case where DISPLAYFORM1 corresponding to a correction by a snapshot of the full gradient at a snapshot location, which is updated every m iterations, as inJohnson & Zhang (2013).

The combination of gradient reduction and momentum was discussed in Allen-Zhu (2017).In order to obtain the accelerated rate, our Liapuonov analysis requires that the |e i | be decreasing fast enough.

This can also be accomplished in the minibatch setting by using larger minibatches.

In this case, the rate of decrease of e i required gives a schedule for minibatch sizes.

A similar result was obtained in .When we replace gradients with (11) the Forward Euler scheme (FE-C) becomes DISPLAYFORM2 where y k is given by (1), h is a constant time step, and t k := h(k + 2).

In Appendix C, we study the continuous version of (Sto-FE-C) and obtain a rate of convergence using a Liapunov function.

Definition 4.1.

Define the discrete stochastic Liapunov functionẼ k := E k + I k , for k ≥ 0, where E k is given by (4) and and, e −1 := 0 and for k ≥ 0, DISPLAYFORM3 Theorem 4.2.

Assume that the sequence e k satisfies DISPLAYFORM4 We immediately have the following result.

DISPLAYFORM5 Remark 4.4.

The assumption on e k is satisfied, for example, by a sequence of the form |e k | = 1/k α for any α > 2.

By comparison for SGD, the corresponding condition is satisfied by such sequences with α > 1.

Thus the norm of the noise needs to go to zero faster for accelerated SGD compared to regular SGD (see Appendix B) in order to obtain the rate.

Remark 4.5.

In Theorem 4.2, we focus on the maximum possible time step h = 1/ √ L. The result is still true if we shrink the time step.

In this case, I k can be defined using the tails h DISPLAYFORM6 * , e i−1 , see .

In this section, we consider that stochastic gradient, which we write as a gradient plus an error, as in section 4.1.

In Appendix B.2, we study the Stochastic gradient descent and Appendix C.2 is devoted to the analysis of the continuous framework of Stochastic Accelerated method.

The Forward Euler scheme (FE-SC) becomes DISPLAYFORM0 where e k is a given error and DISPLAYFORM1 Inspired by the continuous framework (Appendix C.2), we define a discrete Lyapunov function.

Definition 4.6.

DefineẼ k := E k + I k , where E k is given by (7) and DISPLAYFORM2 with the convention e −1 = 0.Then we obtain the following convergence result for sequences generated by (Sto-FE-SC).

Theorem 4.7.

Let x k , v k be two sequences generated by the scheme (Sto-FE-SC) with initial con- DISPLAYFORM3 Then,Ẽ DISPLAYFORM4 In addition, sup i≥0 |v i − x * | ≤ M for a positive constant M and DISPLAYFORM5 We include the proof of Theorem 4.7 since this result is new.

Proof of Theorem 4.7.

First we prove that DISPLAYFORM6 For the term I k , we obtain DISPLAYFORM7 Putting all together, we obtaiñ DISPLAYFORM8 And by definition of v k+1 − v k , we havẽ DISPLAYFORM9 We conclude, as in the convex case, applying discrete Gronwall Lemma and (13).

DISPLAYFORM10 The proof is concluded by convexity, DISPLAYFORM11 Proof of Proposition 3.4.

Using (1st-ODE-SC), we obtain DISPLAYFORM12 By strong convexity, we have DISPLAYFORM13

Let e : [0, +∞) → R d be a integrable function.

Consider the gradient descenṫ DISPLAYFORM0 Then define the Lyapunov function,Ẽ, bỹ DISPLAYFORM1 where, DISPLAYFORM2 and, DISPLAYFORM3 Then the following result holds.

Proposition B.1.

Let x be a solution of (14) with initial condition x 0 .

Then, DISPLAYFORM4 Proof.

For all t > 0, we have DISPLAYFORM5 Then, since f is convex, we obtain the first result.

We deduce thatẼ is decreasing.

Arguing as along with the co-coercivity inequality, we prove that sup s≥0 |x(s) − x * | < +∞, sup s≥0 s|∇f (x(s))| < +∞ which concludes the proof.

The discretization of FORMULA5 is DISPLAYFORM6 where e k = e(hk).

DISPLAYFORM7 and, DISPLAYFORM8 Proposition B.2.

Let x k be the sequence generated by (15) with initial condition x 0 .

Assume that h satisfies, for all k ≥ 0, DISPLAYFORM9 Proof.

By L-smoothness and convexity of f , we have DISPLAYFORM10 In addition, DISPLAYFORM11 when h satisfies (16).

We conclude the proof with the same argument as Proposition B.1.

Let us study the equationẋ DISPLAYFORM0 for an error function, e satisfying +∞ 0 e µs |e(s)| ds < +∞.This condition on the error function is classical Robbins & Monro (1951) .

The case e = 0 is satisfied trivially and corresponds to the gradient descent ODE.We define the function DISPLAYFORM1 where, DISPLAYFORM2 Then we have the following result.

Proposition B.3.

Let x be a solution of (17) with initial data x 0 and suppose that e satisfies (18).

DISPLAYFORM3 In addition, sup t≥0 |x − x * | < +∞ and DISPLAYFORM4 Therefore E(t, x(t)) is decreasing and then for all t > 0, DISPLAYFORM5 By Gronwall Lemma and FORMULA5 , we deduce that sup t≥0 |x − x * | < +∞ and the proof is concluded.

The discretization of (17) is DISPLAYFORM6 where e k = e(hk).

We define E k , for k ≥ 1, by DISPLAYFORM7 where, DISPLAYFORM8 with the notation e −1 = 0.

DISPLAYFORM9 In addition, if the sequence (1 − hµ) −i |e i | is summable, sup i≥1 |x i − x * | < +∞ and we deduce, DISPLAYFORM10 Proof.

First, as usual, we have DISPLAYFORM11 In addition, DISPLAYFORM12 Combining these two inequalities, DISPLAYFORM13 In order to conclude, we also need to establish that E k is bounded below.

That follows from discrete Gronwall's inequality, as was already done in the continuous case in Proposition B.3.

In this section, we consider that an error e(t) is made in the calculation of the gradient.

We study the following perturbation of system (1st-ODE), DISPLAYFORM0 where e is a function satisfying DISPLAYFORM1 The corresponding ODE is DISPLAYFORM2 We follow the argument from Attouch et al. (2016, section 5) to define a Lyapunov function for this system.

LetẼ be defined byẼ DISPLAYFORM3 where, DISPLAYFORM4 and DISPLAYFORM5 Lemma C.1.

Let (x, v) be a solution of (Sto-1st-ODE) with initial condition (x(0), v(0)) = (x 0 , v 0 ) and suppose that e satisfies (19).

Then DISPLAYFORM6 In addition, sup t≥0 |v(t) − x * | < +∞ and sup t≥0 |t∇f (x)| < +∞.Proof.

Following the proof of Proposition 3.2, we have DISPLAYFORM7 In particular,Ẽ is decreasing and DISPLAYFORM8 Using the inequality of co-coercitivity, we obtain 1 2L DISPLAYFORM9 Using FORMULA5 , we conclude applying Gronwall Lemma.

Then we deduce Proposition C.2.

Let (x, v) be a solution of (Sto-1st-ODE) with initial condition (x(0), v(0)) = (x 0 , v 0 ) and suppose that e satisfies (19).

Then, DISPLAYFORM10

Define the perturbed system of (1st-ODE-SC) by DISPLAYFORM0 where e is a locally integrable function.

Definition C.3.

Define the continuous time Liapunov function E(x, v) DISPLAYFORM1 Define the perturbed Liapunov functionẼ, bỹ E(t, x, v) := E(x, v) + I(t, x, v), DISPLAYFORM2 Lemma C.5.

Suppose f is bounded from below and s → e √ µs e(s) ∈ L 1 .

Let (x, v) be a solution of (Sto-1st-ODE-SC), then sup t≥0 |v(t) − x * | < +∞ and sup t≥0 |∇f (x)| < +∞.Proof.

Same as Attouch et al. (2016, Lemma 5.2) , using the fact that DISPLAYFORM3 ) is decreasing and Gronwall's inequality.

Then, combining the two previous result, we obtain: Corollary C.6.

Suppose that s → e λs e(s) is a L 1 (0, +∞) function.

Let (x, v) be a solution of (Sto-1st-ODE-SC) with initial condition (x(0), v(0)) = (x 0 , v 0 ).

Then, DISPLAYFORM4 where, DISPLAYFORM5 Proof.

By Proposition C.4 and Gronwall's Lemma, we havẽ DISPLAYFORM6 This is equivalent to DISPLAYFORM7 which concludes the proof with Lemma C.5.

First, using the convexity and the L-smoothness of f , we obtain the following classical inequality (see or Su et al. (2014) in the case e k = 0), DISPLAYFORM0 Then, we have DISPLAYFORM1 By defintion of v k+1 , we have DISPLAYFORM2 In addition, DISPLAYFORM3 Combining these three previous inequalities, we obtaiñ DISPLAYFORM4 , we deduce thatẼ k is decreasing.

In particular, DISPLAYFORM5 and the discrete version of Gronwall Lemma gives the result since (i+3)|e i | is a summable sequence due to (12).E PROOF OF PROPOSITION 3.4To simplify, we denote λ h = h √ µ 1+h √ µ .

Note, however, since the gradients are evaluated at y k , not x k , the first step is to use strong convexity and L-smoothness to estimate the differences of E in terms of gradients evaluated at y k .

Lemma E.1.

Suppose that f is a µ-stgrongly convex and L-smooth function, then DISPLAYFORM6 Proof.

First, we remark that DISPLAYFORM7 Since the first line of (1st-ODE-SC) can be rewritten as DISPLAYFORM8 we obtain (21).Proof of Proposition 3.4.

Once FORMULA25 is established, since the expression on the right hand side is monotone in h, the largest choice of h is given by h = 1 √ L , which leads immediately to (10).In the proof we will estimate the linear term y k − x k , ∇f (y k ) in terms of y k − x * , ∇f (y k ) plus a correction which is controlled by the gap (the negative quadratic) in (21) and the quadratic term in E.The second term in the Liapunov function gives, using 1-smoothness of the quadratic term in E. µ DISPLAYFORM9

@highlight

We derive Nesterov's method arises as a straightforward discretization of an ODE different from the one in Su-Boyd-Candes and prove acceleration the stochastic case
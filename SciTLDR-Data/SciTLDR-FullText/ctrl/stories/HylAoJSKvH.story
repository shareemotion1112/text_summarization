We consider the problem of unconstrained minimization of a smooth objective function in $\mathbb{R}^d$ in setting where only function evaluations are possible.

We propose and analyze stochastic zeroth-order method with heavy ball momentum.

In particular, we propose, SMTP, a momentum version of the stochastic three-point method (STP) Bergou et al. (2019).

We show new complexity results for non-convex, convex and strongly convex functions.

We test our method on a collection of learning to continuous control tasks on several MuJoCo Todorov et al. (2012) environments with varying difficulty and compare against STP, other state-of-the-art derivative-free optimization algorithms and against policy gradient methods.

SMTP significantly outperforms STP and all other methods that we considered in our numerical experiments.

Our second contribution is SMTP with importance sampling which we call SMTP_IS.

We provide convergence analysis of this method for non-convex, convex and strongly convex objectives.

In this paper, we consider the following minimization problem

where f : R d → R is "smooth" but not necessarily a convex function in a Derivative-Free Optimization (DFO) setting where only function evaluations are possible.

The function f is bounded from below by f (x * ) where x * is a minimizer.

Lastly and throughout the paper, we assume that f is L-smooth.

DFO.

In DFO setting Conn et al. (2009); Kolda et al. (2003) , the derivatives of the objective function f are not accessible.

That is they are either impractical to evaluate, noisy (function f is noisy) (Chen, 2015) or they are simply not available at all.

In standard applications of DFO, evaluations of f are only accessible through simulations of black-box engine or software as in reinforcement learning and continuous control environments Todorov et al. (2012) .

This setting of optimization problems appears also in applications from computational medicine Marsden et al. (2008) and fluid dynamics Allaire (2001) ; Haslinger & Mäckinen (2003) ; Mohammadi & Pironneau (2001) to localization Marsden et al. (2004; 2007) and continuous control Mania et al. (2018) ; Salimans et al. (2017) to name a few.

The literature on DFO for solving (1) is long and rich.

The first approaches were based on deterministic direct search (DDS) and they span half a century of work Hooke & Jeeves (1961) ; Su (1979); Torczon (1997) .

However, for DDS methods complexity bounds have only been established recently by the work of Vicente and coauthors Vicente (2013); Dodangeh & Vicente (2016) .

In particular, the work of Vicente Vicente (2013) showed the first complexity results on non-convex f and the results were extended to better complexities when f is convex Dodangeh & Vicente (2016) .

However, there have been several variants of DDS, including randomized approaches Matyas (1965) ; Karmanov (1974a; b) ; Baba (1981) ; Dorea (1983) ; Sarma (1990) .

Only very recently, complexity bounds have also been derived for randomized methods Diniz-Ehrhardt et al. (2008) ; Stich et al. (2011); Ghadimi & Lan (2013) ; Ghadimi et al. (2016) ; Gratton et al. (2015) .

For instance, the work of Diniz-Ehrhardt et al. (2008) ; Gratton et al. (2015) imposes a decrease condition on whether to accept or reject a step of a set of random directions.

Moreover, Nesterov & Spokoiny (2017) derived new complexity bounds when the random directions are normally distributed vectors for both smooth and non-smooth f .

They proposed both accelerated and non-accelerated zero-order (ZO) methods.

Accelerated derivative-free methods in the case of inexact oracle information was proposed in Dvurechensky et al. (2017) .

An extension of Nesterov & Spokoiny (2017) for non-Euclidean proximal setup was proposed by Gorbunov et al. (2018) for the smooth stochastic convex optimization with inexact oracle.

More recently and closely related to our work, Bergou et al. (2019) proposed a new randomized direct search method called Stochastic Three Points (STP).

At each iteration k STP generates a random search direction s k according to a certain probability law and compares the objective function at three points: current iterate x k , a point in the direction of s k and a point in the direction of −s k with a certain step size α k .

The method then chooses the best of these three points as the new iterate:

The key properties of STP are its simplicity, generality and practicality.

Indeed, the update rule for STP makes it extremely simple to implement, the proofs of convergence results for STP are short and clear and assumptions on random search directions cover a lot of strategies of choosing decent direction and even some of first-order methods fit the STP scheme which makes it a very flexible in comparison with other zeroth-order methods (e.g. two-point evaluations methods like in Nesterov & Spokoiny (2017) , Ghadimi & Lan (2013) , Ghadimi et al. (2016) , Gorbunov et al. (2018) that try to approximate directional derivatives along random direction at each iteration).

Motivated by these properties of STP we focus on further developing of this method.

1 is a special technique introduced by Polyak in 1964 Polyak (1964 to get faster convergence to the optimum for the first-order methods.

In the original paper, Polyak proved that his method converges locally with O L /µ log 1 /ε rate for twice continuously differentiable µ-strongly convex and L-smooth functions.

Despite the long history of this approach, there is still an open question whether heavy ball method converges to the optimum globally with accelerated rate when the objective function is twice continuous differentiable, L-smooth and µ-strongly convex.

For this class of functions, only non-accelerated global convergence was proved Ghadimi et al. (2015) and for the special case of quadratic strongly convex and L-smooth functions Lessard et.

al. Lessard et al. (2016) recently proved asymptotic accelerated global convergence.

However, heavy ball method performs well in practice and, therefore, is widely used.

One can find more detailed survey of the literature about heavy ball momentum in Loizou & Richtárik (2017) .

Importance Sampling.

Importance sampling has been celebrated and extensively studied in stochastic gradient based methods Zhao & Zhang (2015) or in coordinate based methods Richtárik & Takáč (2016) .

Only very recently, Bibi et al. (2019) proposed, STP_IS, the first DFO algorithm with importance sampling.

In particular, under coordinate-wise smooth function, they show that sampling coordinate directions, can be generalized to arbitrary directions, with probabilities proportional to the function coordinate smoothness constants, improves the leading constant by the same factor typically gained in gradient based methods.

Contributions.

Our contributions can be summarized into three folds.

• First ZO method with heavy ball momentum.

Motivated by practical effectiveness of first-order momentum heavy ball method, we introduce momentum into STP method and propose new DFO algorithm with heavy ball momentum (SMTP).

We summarized the method in Algorithm 1, with theoretical guarantees for non-convex, convex and strongly convex functions under generic sampling directions D. We emphasize that the SMTP with momentum is not a straightforward generalization of STP and Polyak's method and requires insights from virtual iterates analysis from Yang et al. (2016) .

To the best of our knowledge it is the first analysis of derivative-free method with heavy ball momentum, i.e. we show that the same momentum trick that works for the first order method could be applied for zeroth-order methods as well.

• First ZO method with both heavy ball momentum and importance sampling.

In order to get more gain from momentum in the case when the sampling directions are coordinate directions and the objective function is coordinate-wise L-smooth (see Assumption 4.1), we consider importance sampling to the above method.

In fact, we propose the first zeroth-order Algorithm 1 SMTP: Stochastic Momentum Three Points Require: learning rates {γ (2019) where we propose an importance sampling that improves the leading constant marked in red.

Note that r 0 = f (x 0 ) − f (x * ) and that all assumptions listed are in addition to Assumption 2.1.

Complexity means number of iterations in order to guarantee E ∇f (z K ) D ≤ ε for the non-convex case, E f (z K ) − f (x * ) ≤ ε for convex and strongly convex cases.

R 0 < ∞ is the radius in · * D -norm of a bounded level set where the exact definition is given in Assumption 3.2.

We notice that for STP_IS · D = · 1 and · * D = · ∞ in non-convex and convex cases and · D = · 2 in the strongly convex case.

momentum method with importance sampling (SMTP_IS) summarized in Algorithm 2 with theoretical guarantees for non-convex, convex and strongly convex functions.

The details and proofs are left for Section 4 and Appendix E.

• Practicality.

We conduct extensive experiments on continuous control tasks from the MuJoCo suite Todorov et al. (2012) following recent success of DFO compared to modelfree reinforcement learning Mania et al. (2018); Salimans et al. (2017) .

We achieve with SMTP_IS the state-of-the-art results on across all tested environments on the continuous control outperforming DFO Mania et al. (2018) and policy gradient methods Schulman et al. (2015) ; Rajeswaran et al. (2017) .

We provide more detailed comparison of SMTP and SMTP_IS in Section E.4 of the Appendix.

We use · p to define p -norm of the vector

for p ≥ 1 and

As we mention in the introduction we assume throughout the paper 2 that the objective function f is L-smooth.

Assumption 2.1. (L-smoothness) We say that f is L-smooth if

From this definition one can obtain

and if additionally f is convex, i.e. f (y) ≥ f (x) + ∇f (x), y − x , we have

Our analysis of SMTP is based on the following key assumption.

Assumption 3.1.

The probability distribution D on R d satisfies the following properties:

Some examples of distributions that meet above assumption are described in Lemma 3.4 from Bergou et al. (2019) .

For convenience we provide the statement of the lemma in the Appendix (see Lemma F.1).

Recall that one possible view on STP Bergou et al. (2019) is as following.

If we substitute gradient ∇f (x k ) in the update rule for the gradient descent

by ±s k where s k is sampled from distribution D satisfied Assumption 3.1 and then select x k+1 as the best point in terms of functional value among

we will get exactly STP method.

However, gradient descent is not the best algorithm to solve unconstrained smooth minimization problems and the natural idea is to try to perform the same substitution-trick with more efficient first-order methods than gradient descent.

We put our attention on Polyak's heavy ball method where the update rule could be written in the following form: By definition of z k+1 , we get that the sequence {f (z k )} k≥0 is monotone:

Now, we establish the key result which will be used to prove the main complexity results and remaining theorems in this section.

Lemma 3.1.

Assume that f is L-smooth and D satisfies Assumption 3.1.

Then for the iterates of SMTP the following inequalities hold:

and

3.1 NON-CONVEX CASE

In this section, we show our complexity results for Algorithm 1 in the case when f is allowed to be non-convex.

In particular, we show that SMTP in Algorithm 1 guarantees complexity bounds with the same order as classical bounds, i.e. 1/ √ K where K is the number of iterations, in the literature.

We notice that query complexity (i.e. number of oracle calls) of SMTP coincides with its iteration complexity up to numerical constant factor.

For clarity and completeness, proofs are left for the appendix.

Theorem 3.1.

Let Assumptions 2.1 and 3.1 be satisfied.

Let SMTP with γ k ≡ γ > 0 produce points {z 0 , z 1 , . . .

, z K−1 } and z K is chosen uniformly at random among them.

Then

Moreover, if we choose γ = γ0 √ K the complexity (10) reduces to

minimizes the right-hand side of (11) and for this choice we have

In other words, the above theorem states that SMTP converges no worse than STP for non-convex problems to the stationary point.

In the next sections we also show that theoretical convergence guarantees for SMTP are not worse than for STP for convex and strongly convex problems.

However, in practice SMTP significantly outperforms STP.

So, the relationship between SMTP and STP correlates with the known in the literature relationship between Polyak's heavy ball method and gradient descent.

In this section, we present our complexity results for Algorithm 1 when f is convex.

In particular, we show that this method guarantees complexity bounds with the same order as classical bounds, i.e. 1/K, in the literature.

We will need the following additional assumption in the sequel.

Assumption 3.2.

We assume that f is convex, has a minimizer x * and has bounded level set at x 0 :

where

From the above assumption and Cauchy-Schwartz inequality we get the following implication:

Theorem 3.2 (Constant stepsize).

Let Assumptions 2.1, 3.1 and 3.2 be satisfied.

If we set γ k ≡ γ <

, then for the iterates of SMTP method the following inequality holds:

If we choose γ =

and run SMTP for k = K iterations where

In order to get rid of factor ln

in the complexity we consider decreasing stepsizes.

(1−β)R0 and θ ≥ 2 α , then for the iterates of SMTP method the following inequality holds:

where

(1−β)R0 and run SMTP for k = K iterations where

We notice that if we choose β sufficiently close to 1, we will obtain from the formula (18) that

In this section we present our complexity results for Algorithm 1 when f is µ-strongly convex.

Assumption 3.3.

We assume that f is µ-strongly convex with respect to the norm · * D :

It is well known that strong convexity implies

Theorem 3.4 (Solution-dependent stepsizes).

Let Assumptions 2.1, 3.1 and 3.3 be satisfied.

If we set

, then for the iterates of SMTP, the following inequality holds:

Then, If we run SMTP for k = K iterations where

where

is the condition number of the objective, we will get E f (z

Note that the previous result uses stepsizes that depends on the optimal solution f (x * ) which is often not known in practice.

The next theorem removes this drawback without spoiling the convergence rate.

However, we need an additional assumption on the distribution D and one extra function evaluation.

Assumption 3.4.

We assume that for all s ∼ D we have s 2 = 1.

Theorem 3.5 (Solution-free stepsizes).

Let Assumptions 2.1, 3.1, 3.3 and 3.4 be satisfied.

If additionally we compute

, then for the iterates of SMTP the following inequality holds:

Moreover, for any ε > 0 if we set t such that

and run SMTP for k = K iterations where

where

In this section we consider another assumption, in a similar spirit to Bibi et al. (2019) , on the objective.

Assumption 4.1 (Coordinate-wise L-smoothness).

We assume that the objective f has coordinatewise Lipschitz gradient, with Lipschitz constants

where ∇ i f (x) is i-th partial derivative of f at the point x.

For this kind of problems we modify SMTP and present STMP_IS method in Algorithm 2.

In general, the idea behind methods with importance sampling and, in particular, behind SMTP_IS is to adjust probabilities of sampling in such a way that gives better convergence guarantees.

In the case when f satisfies coordinate-wise L-smoothness and Lipschitz constants L i are known it is natural to sample direction s k = e i with probability depending on L i (e.g. proportional to L i ).

One can find more detailed discussion of the importance sampling in Zhao & Zhang (2015) and Richtárik & Takáč (2016) .

Now, we establish the key result which will be used to prove the main complexity results of STMP_IS.

Lemma 4.1.

Assume that f satisfies Assumption 4.1.

Then for the iterates of SMTP_IS the following inequalities hold:

and

Due to the page limitation, we provide the complexity results of SMTP_IS in the Appendix.

Require: stepsize parameters w 1 , . . . , w n > 0, probabilities p 1 , . . .

, p n > 0 summing to 1, starting point

Select i k = i with probability p i > 0

Choose stepsize γ k i proportional to

Experimental Setup.

We conduct extensive experiments 3 on challenging non-convex problems on the continuous control task from the MuJoCO suit Todorov et al. (2012) .

In particular, we address the problem of model-free control of a dynamical system.

Policy gradient methods for model-free reinforcement learning algorithms provide an off-the-shelf model-free approach to learn how to control a dynamical system and are often benchmarked in a simulator.

We compare our proposed momentum stochastic three points method SMTP and the momentum with importance sampling version SMTP_IS against state-of-art DFO based methods as STP_IS Bibi et al. (2019) and ARS Mania et al. (2018) .

Moreover, we also compare against classical policy gradient methods as TRPO Schulman et al. (2015) and NG Rajeswaran et al. (2017) .

We conduct experiments on several environments with varying difficulty Swimmer-v1, Hopper-v1, HalfCheetah-v1, Ant-v1, and Humanoid-v1.

Note that due to the stochastic nature of problem where f is stochastic, we use the mean of the function values of f (

, see Algorithm 1, over K observations.

Similar to the work in Bibi et al. (2019), we use K = 2 for Swimmer-v1, K = 4 for both Hopper-v1 and HalfCheetah-v1, K = 40 for Ant-v1 and Humanoid-v1.

Similar to Bibi et al. (2019) , these values were chosen based on the validation performance over the grid that is K ∈ {1, 2, 4, 8, 16} for the smaller dimensional problems Swimmer-v1, Hopper-v1, HalfCheetah-v1 and K ∈ {20, 40, 80, 120} for larger dimensional problems Ant-v1, and Humanoid-v1.

As for the momentum term, for SMTP we set β = 0.5.

For SMTP_IS, as the smoothness constants are not available for continuous control, we use the coordinate smoothness constants of a θ parameterized smooth functionf θ (multi-layer perceptron) that estimates f .

In particular, consider running any DFO for n steps; with the queried sampled

2 .

See Bibi et al. (2019) for further implementation details as we follow the same experimental procedure.

In contrast to STP_IS, our method (SMTP) does not required sampling from directions in the canonical basis; hence, we use directions from standard Normal distribution in each iteration.

For SMTP_IS, we follow a similar procedure as Bibi et al. (2019) and sample from columns of a random matrix B.

Similar to the standard practice, we perform all experiments with 5 different initialization and measure the average reward, in continuous control we are maximizing the reward function f , and best and worst run per iteration.

We compare algorithms in terms of reward vs. sample complexity.

Comparison Against STP.

Our method improves sample complexity of STP and STP_IS significantly.

Especially for high dimensional problems like Ant-v1 and Humanoid-v1, sample efficiency of SMTP is at least as twice as the STP.

Moreover, SMTP_IS helps in some experiments by Table  2 to demonstrate complexity of each method.

improving over SMTP.

However, this is not consistent in all environments.

We believe this is largely due to the fact that SMTP_IS can only handle sampling from canonical basis similar to STP_IS.

Comparison Against State-of-The-Art.

We compare our method with state-of-the-art DFO and policy gradient algorithms.

For the environments, Swimmer-v1, Hopper-v1, HalfCheetah-v1 and Ant-v1, our method outperforms the state-of-the-art results.

Whereas for Humanoid-v1, our methods results in a comparable sample complexity.

We have proposed, SMTP, the first heavy ball momentum DFO based algorithm with convergence rates for non-convex, convex and strongly convex functions under generic sampling direction.

We specialize the sampling to the set of coordinate bases and further improve rates by proposing a momentum and importance sampling version SMPT_IS with new convergence rates for non-convex, convex and strongly convex functions too.

We conduct large number of experiments on the task of controlling dynamical systems.

We outperform two different policy gradient methods and achieve comparable or better performance to the best DFO algorithm (ARS) on the respective environments.

Assumption A.2.

The probability distribution D on R d satisfies the following properties:

2 is positive and finite.

2.

There is a constant µ D > 0 and norm

We establish the key lemma which will be used to prove the theorems stated in the paper.

Lemma A.1.

Assume that f is L-smooth and D satisfies Assumption A.2.

Then for the iterates of SMTP the following inequalities hold:

and

Proof.

By induction one can show that

That is, for k = 0 this recurrence holds and update rules for z k , x k and v k−1 do not brake it.

From this we get

Similarly,

Unifying these two inequalities we get

which proves (31).

Finally, taking the expectation E s k ∼D of both sides of the previous inequality and invoking Assumption A.2, we obtain

Theorem B.1.

Let Assumptions A.1 and A.2 be satisfied.

Let SMTP with γ k ≡ γ > 0 produce points {z 0 , z 1 , . . .

, z K−1 } and z K is chosen uniformly at random among them.

Then

Moreover, if we choose γ = γ0 √ K the complexity (34) reduces to

minimizes the right-hand side of (35) and for this choice we have

Proof.

Taking full expectation from both sides of inequality (32) we get

Further, summing up the results for k = 0, 1, . . .

, K −1, dividing both sides of the obtained inequality by K and using tower property of the mathematical expectation we get

The last part where γ = γ0 √ K is straightforward.

Assumption C.1.

We assume that f is convex, has a minimizer x * and has bounded level set at x 0 :

where

Theorem C.1 (Constant stepsize).

Let Assumptions A.1, A.2 and C.1 be satisfied.

If we set γ k ≡ γ <

, then for the iterates of SMTP method the following inequality holds:

If we choose γ =

and run SMTP for k = K iterations where

then we will get E f (z

Proof.

From the (32) and monotonicity of {f (z k )} k≥0 we have

Taking full expectation, subtracting f (x * ) from the both sides of the previous inequality and using the tower property of mathematical expectation we get

Since γ <

(1−β)R0 is positive and we can unroll the recurrence (40):

Lastly, putting γ = (39) in (38) we have

Next we use technical lemma from Mishchenko et al. (2019) .

We provide the original proof for completeness.

Lemma C.1 (Lemma 6 from Mishchenko et al. (2019)).

Let a sequence {a k } k≥0 satisfy inequality

and take C such that N ≤ αθ 4 C and a 0 ≤ C. Then, it holds

Proof.

We will show the inequality for a k by induction.

Since inequality a 0 ≤ C is one of our assumptions, we have the initial step of the induction.

To prove the inductive step, consider

To show that the right-hand side is upper bounded by θC α(k+1)+θ , one needs to have, after multiplying both sides by (αk + θ)(αk + α + θ)(θC)

which is equivalent to

The last inequality is trivially satisfied for all k ≥ 0.

(1−β)R0 and θ ≥ 2 α , then for the iterates of SMTP method the following inequality holds:

where

(1−β)R0 and run SMTP for k = K iterations where

Proof.

In (40) we proved that

Having that, we can apply Lemma C.1 to the sequence E f (z k ) − f (x * ) .

The constants for the lemma are:

α 2 k+2 is equivalent to the choice θ = 2 α .

In this case, we have αθ = 2,

.

Putting these parameters and K from (42) in the (41) we get the result.

Assumption D.1.

We assume that f is µ-strongly convex with respect to the norm · * D :

It is well known that strong convexity implies

Theorem D.1 (Solution-dependent stepsizes).

Let Assumptions A.1, A.2 and D.1 be satisfied.

If we set

, then for the iterates of SMTP the following inequality holds:

If we run SMTP for k = K iterations where

where κ def = L µ is the condition number of the objective, we will get E f (z

Proof.

From (32) and

Using θ = inf

and taking the full expectation from the previous inequality we get

Lastly, from (45) we have

≤ ε.

, then for the iterates of SMTP the following inequality holds:

Moreover, for any ε > 0 if we set t such that

and run SMTP for k = K iterations where

where

Proof.

Recall that from (31) we have

If we minimize the right hand side of the previous inequality as a function of γ k , we will get that the optimal choice in this sense is γ

.

However, this stepsize is impractical for derivative-free optimization, since it requires to know ∇f (z k ).

The natural way to handle this is to

and that is what we do.

We choose

Next we estimate |δ k |:

It implies that

+ Lt 2 8 and after taking full expectation from the both sides of the obtained inequality we get

Note that from the tower property of mathematical expectation and Jensen's inequality we have

Putting all together we get

Lastly, from (47) we have

Again by definition of z k+1 we get that the sequence {f (z k )} k≥0 is monotone:

Lemma E.1.

Assume that f satisfies Assumption 4.1.

Then for the iterates of SMTP_IS the following inequalities hold:

and

Proof.

In the similar way as in Lemma A.1 one can show that

and

It implies that

Unifying these two inequalities we get

which proves (51).

Finally, taking the expectation E[· | z k ] conditioned on z k from the both sides of the previous inequality we obtain

Theorem E.1.

Assume that f satisfies Assumption 4.1.

Let SMTP_IS with γ k i = γ wi k for some γ > 0 produce points {z 0 , z 1 , . . .

, z K−1 } and z K is chosen uniformly at random among them.

Then

Moreover, if we choose γ =

in order to minimize right-hand side of (55), we will get

Note that for

Li with w i = L i we have that the rates improves to

Proof.

Recall that from (52) we have

Putting it in (58) and taking full expectation from the both sides of obtained inequality we get

Summing up previous inequality for k = 0, 1, . . .

, K − 1 and dividing both sides of the result by K, we get

It remains to notice that

As for SMTP to tackle convex problems by SMTP_IS we use Assumption 3.2 with

Theorem E.2 (Constant stepsize).

Let Assumptions 3.2 and 4.1 be satisfied.

If we set γ

, then for the iterates of SMTP_IS method the following inequality holds:

and run SMTP_IS for k = K iterations where

we will get E f (z

Li with w i = L i , the rate improves to

Proof.

Recall that from (52) we have

and

Putting it in (62) and taking full expectation from the both sides of obtained inequality we get

pi wi

we have that the factor 1 −

pi wi is nonnegative and, therefore,

pi wi

and k = K from (60) in (59)

we have (1−β)R0 and θ ≥ 2 α , then for the iterates of SMTP_IS method the following inequality holds:

where

(1−β)R0 and run SMTP_IS for k = K iterations where

Proof.

In (63) we proved that

Under review as a conference paper at ICLR 2020

Having that, we can apply Lemma C.1 to the sequence E f (z k ) − f (x * ) .

The constants for the lemma are:

α .

In this case we have αθ = 2 and C = max f (x 0 ) − f (x * ),

.

Putting these parameters and K from (65) in the (64) we get the result.

Theorem E.4 (Solution-dependent stepsizes).

Let Assumptions 3.3 (with · D = · 1 ) and 4.1 be satisfied.

If we set γ

, then for the iterates of SMTP_IS method the following inequality holds:

If we run SMTP_IS for k = K iterations where

we will get E f (z

Proof.

Recall that from (52) we have

≤ ε.

The previous result based on the choice of γ k which depends on the f (z k ) − f (x * ) which is often unknown in practice.

The next theorem does not have this drawback and makes it possible to obtain the same rate of convergence as in the previous theorem using one extra function evaluation.

Theorem E.5 (Solution-free stepsizes).

Let Assumptions 3.3 (with · D = · 2 ) and 4.1 be satisfied.

Li k t for t > 0, then for the iterates of SMTP_IS method the following inequality holds:

Moreover, for any ε > 0 if we set t such that

and run SMTP_IS for k = K iterations where

we will get E f (z

Proof.

Recall that from (51) we have

2(1 − β) 2 .

If we minimize the right hand side of the previous inequality as a function of γ k i , we will get that the optimal choice in this sense is γ

.

However, this stepsize is impractical for derivative-free optimization, since it requires to know ∇ i k f (z k ).

The natural way to handle this is to

and that is what we do.

We choose γ

.

From this we get

Next we estimate |δ k i |:

It implies that

8 and after taking expectation E · | z k conditioned on z k from the both sides of the obtained inequality we get

Note that

, and

Putting all together we get

Taking full expectation from the previous inequality we get

Since µ ≤ L i for all i = 1, . . .

, d we have

Lastly, from (69) we have

≤ ε 2 + ε 2 = ε.

Here we compare SMTP when D is normal distribution with zero mean and Table 3 summarizes complexities in this case.

We notice that for SMTP we have · D = · 2 .

That is why one needs to compare SMTP with SMTP_IS accurately.

At the first glance, Table 3 says that for non-convex and convex cases we get an extra d factor in the complexity of SMTP_IS when L 1 = . . .

= L d = L. However, it is natural since we use different norms for SMTP and SMTP_IS.

In the non-convex case for SMTP we give number of iterations in order to guarantee E ∇f (z K ) 2 ≤ ε while for SMTP_IS we provide number of iterations in order to guarantee E ∇f (z K ) 1 ≤ ε.

From Holder's inequality · 1 ≤ √ d · 2 and, therefore, in order to have E ∇f (z K ) 1 ≤ ε for SMTP we need to ensure

.

That is, to guarantee E ∇f (z K ) 1 ≤ ε SMTP for aforementioned distribution needs to perform

Analogously, in the convex case using Cauchy-Schwartz inequality · 2 ≤ √ d · ∞ we have that R 0, 2 ≤ √ dR 0, ∞ .

Typically this inequality is tight and if we assume that R 0, ∞ ≥ C

<|TLDR|>

@highlight

We develop and analyze a new derivative free optimization algorithm with momentum and importance sampling with applications to continuous control.
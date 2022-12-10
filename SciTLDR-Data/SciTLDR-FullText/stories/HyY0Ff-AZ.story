Two main families of reinforcement learning algorithms, Q-learning and policy gradients, have recently been proven to be equivalent when using a softmax relaxation on one part, and an entropic regularization on the other.

We relate this result to the well-known convex duality of Shannon entropy and the softmax function.

Such a result is also known as the Donsker-Varadhan formula.

This provides a short proof of the equivalence.

We then interpret this duality further, and use ideas of convex analysis to prove a new policy inequality relative to soft Q-learning.

• Policy gradients (V. ), looks to maximize the expected reward by improving policies to favor high-reward actions.

In general, the target loss function is regularized by the addition of an entropic functional for the policy.

This makes policies more diffuse and less likely to yield degenerate results.

A critical step in the theoretical understanding of the field has been a smooth relaxation of the greedy max operation involved in selecting actions, turned into a Boltzmann softmax O. BID8 .

This new context has lead to a breakthrough this year J. BID7 with the proof of the equivalence of both methods of Q-learning and policy gradients.

While that result is extremely impressive in its unification, we argue that it is critical to look additionally at the fundamental reasons as to why it occurs.

We believe that the convexity of the entropy functional used for policy regularization is at the root of the phenomenon, and that (Lagrangian) duality can be exploited as well, either yielding faster proofs, or further understanding.

The contributions of our paper are as follows:1.

We show how convex duality expedites the proof of the equivalence between soft Qlearning and softmax entropic policy gradients -heuristically in the general case, rigorously in the bandit case.2.

We introduce a transportation inequality that relates the expected optimality gap of any policy with its Kullback-Leibler divergence to the optimal policy.

We describe our notations here.

Abusing notation heavily by identifying measures with their densities as in dπ(a|s) = π(a|s)da, if we note as either r(s, a) or r(a, s) the reward obtained by taking action a in state s, the expected reward expands as: DISPLAYFORM0 K r is a linear functional of π.

Adding Shannon entropic regularization 1 improves numerical stability of the algorithm, and prevents early convergence to degenerate solutions.

Noting regularization strength β, the objective becomes a free energy functional, named by analogy with a similar quantity in statistical mechanics: DISPLAYFORM1 Crucially, viewed as a functional of π, J is convex and is the sum of two parts DISPLAYFORM2 2 THE GIBBS VARIATIONAL PRINCIPLE FOR POLICY EVALUATION

Here we are interested in the optimal value of the policy functional J, achieved for an optimal policy π * .

We hence look for J * = J(π * ) = sup π∈P J(π).

In the one step-one state bandit setting we are in, this is in fact almost the same as deriving the state-value function.

The principles of convex duality BID3 BID16 DISPLAYFORM0 with H the entropy functional defined above, we recover exactly the definition of the LegendreFenchel transformation, or convex conjugate, of β · H. The word convex applies to the entropy functional, and doesn't make any assumptions on the rewards r(s, a), other that they be well-behaved enough to be integrable in a.

The Legendre transform inverts derivatives.

A simple calculation shows that the formal convex conjugate of f : t → t log t is f * : p → e (p−1) -this because their respective derivatives log and exp are reciprocal.

We can apply this to f (π(a|s)) = π(a|s) log π(a|s), and then this relationship can also be integrated in a. Hence the dual Legendre representation of the entropy functional H is known.

The Gibbs variational principle states that, taking β = 1/λ as the inverse temperature parameter, and for each Borelian (measurable) test function Φ ∈ C b (A): DISPLAYFORM1 or in shorter notation, for each real random variable X with exponential moments, DISPLAYFORM2 We can prove a stronger result.

If µ is a reference measure (or policy), and we now consider the relative entropy (or Kullback-Leibler divergence) with respect to µ, H µ (·), instead of the entropy H(·), then the Gibbs variational principle still holds BID15 , chapter 22).

This result regarding dual representation formulas for entropy is important and in fact found in several areas of science:• as above, in thermodynamics, where it is named the Gibbs variational principle;• in large deviations, this also known as the Donsker-Varadhan variational formula BID4 ;• in statistics, it is the well-known duality between maximum entropy and maximum likelihood estimation BID0 ; • finally, the theory of information geometry BID1 groups all three views and posits that there exists a general, dually flat Riemannian information manifold.

The general form of the result is as follows.

For each Φ representing a rewards function r(s, a) or an estimator of it: DISPLAYFORM3 and the supremum is reached for the measure π * ∈ P defined by its Radon-Nikodym derivative equal to the Gibbs-Boltzmann measure yielding an energy policy: DISPLAYFORM4 In the special case where µ is the Lebesgue measure on a bounded domain (that is, the uniform policy), we find back the result 5 above, up to a constant irrelevant for maximization.

In the general case, the mathematically inclined reader will also see this as a rephrasing of the fact the Bregman divergence associated with Shannon entropy is the Kullback-Leibler divergence.

For completeness' sake, we provide here its full proof : Proposition 1.

Donsker-Varadhan variational formula.

Let G be a bounded measurable function on A and π,π be probability measures on A, with π absolutely continuous w.r.t.π.

Then DISPLAYFORM5 where π * is a probability measure defined by the Radon-Nikodym derivative: DISPLAYFORM6 Proposition 2.

Corollary : DISPLAYFORM7 and the maximum is attained uniquely by π * .Proof.

DISPLAYFORM8 Under review as a conference paper at ICLR 2018The link with reinforcement learning is made by picking Φ = r(s, a), π = π(a|s), λ = 1/β, and by recalling the implicit dependency of the right member on s but not on π at optimality, so that we can write DISPLAYFORM9 which is the definition of the one-step soft Bellman operator at optimum R. Fox & Tishby.

FORMULA0 ; O. Nachum & Schuurmans.

(2017b.); T. BID12 .

Note that here V * (s) depends on the reference measure µ which is used to pick actions frequency -we can be off-policy, in which case V * is only a pseudo state-value function.

In this simplified one-step setting, this provides a short and direct proof that in expectation, and trained to optimality, soft Q-learning and policy gradients ascent yield the same result J. BID7 .

Standard Q-learning is the special case β → 0, λ → ∞ where by the Laplace principle we recover V (s)

→ max A r(s, a) ; that is, the zero-temperature limit, with no entropy regularization.

For simplicity of exposition, we have restricted so far to the proof in the bandit setting; now we extend it to the general case.

First by inserting V * (s) = sup π V π (s) in the representation formulas above applied to r(s, a) + γV * (s ), so that DISPLAYFORM0 The proof in the general case will then be finished if we assume that we could apply the Bellman optimality principle not to the hard-max, but to the soft-max operator.

This requires proving that the soft-Bellman operator admits a unique fixed point, which is the above.

By the Brouwer fixed point theorem, it is enough to prove that it is a contraction, or at least non-expansive (we assume that the discount factor γ < 1 to that end).

We do so below, noting that this result has been shown many times in the literature, for instance in O. BID8 .

Refining the soft-Bellman operator just like above, but in the multi-step case, by the expression DISPLAYFORM1 we get the:Proposition 3.

Nonexpansiveness of the soft-Bellman operator for the supremum norm f ∞ .

DISPLAYFORM2 Proof.

Let us consider two state-value functions V (1) (s) and V (2) (s) along with the associated action-value functions Q(1) (s, a) and Q (2) (s, a).

Besides, denote MDP transition probability by p(s |s, a).

Then : DISPLAYFORM3 ∞ by Hölder's inequality DISPLAYFORM4

In summary, the program of the proof was as below :1.

Write down the entropy-regularised policy gradient functional, and apply the DonskerVaradhan formula to it.

2.

Write down the resulting softmax Bellman operator as a solution to the sup maximization -this obviously also proves existence.

3.

Show that the softmax operator, just like the hard max, is still a contraction for the max norm, hence prove uniqueness of the solution by fixed point theorem.

The above also shows formally that, should we discretize the action space A to replace integration over actions by finite sums, any strong estimatorr(s, a) of r(s, a), applied to the partition function of rewards 1 λ log a e λr(s,a) , could be used for Q-learning-like iterations.

This is because strong convergence would imply weak convergence (especially convergence of the characteristic function, via Levy's continuity theorem), and hence convergence towards the log-sum-exp cumulant generative function above.

Different estimatorsr(s, a) lead to different algorithms.

When the MDP and the rewards function r are not known, the parameterised critic choicer(s, a) ≈ Q w (s, a) recovers Nachum's Path Consistency Learning O. BID8 BID15 .

O'Donoghue's PGQ method B. O'Donoghue & Mnih.

FORMULA0 can be seen as a control variate balancing of the two terms in 7.

In theory, the rewards distribution could be also recovered simply by varying λ (or β), for instance by inverse Laplace transform.

In this section, we propose an inequality that relates the optimality gap of a policy -by how much that policy is sub-optimal on average -to the Kullback-Leibler divergence between the current policy and the optimum.

The proof draws on ideas of convex analysis and Legendre transormation exposed earlier in the context of soft Q-learning.

Let us assume that X is a real-valued bounded random variable.

We denote sup |X| ≤ M with M constant.

Furthermore we assume that X is centered, that is, E[X] = 0.

This can always be achieved just by picking DISPLAYFORM0 Then, by the Hoeffding inequality : DISPLAYFORM1 with K a positive real constant, i.e., the variable X is sub-Gaussian, so that its cumulant generating function grows less than quadratically.

By taking a Legendre transformation and inverting it, we get that for any pair of measures P and Q that are mutually absolutely continuous, one has DISPLAYFORM2 which by specializing Q to be the measure associated to P * the optimal policy, P θ the current parameterized policy, and X an advantage return r : DISPLAYFORM3 By the same logic, any upper bound on log E e βX can give us information about E Q X − E P X .

This enables us to relate the size of Kullback-Leibler trust regions to the amount by which our policy could be improved.

In fact by combining the entropy duality formula with the Legendre transformation, one easily proves the below : Proposition 4.

Let X a real-valued integrable random variable, and f a convex and differentiable function such that f (0) = f (0) = 0.

Then with f * : x → f * (x) = sup(βx − f (β)) the Legendre transformation of f , f * −1 its reciprocal, and P and Q any two mutually absolutely continuous measures, one has the equivalence: DISPLAYFORM4 Proof.

By Donsker-Varadhan formula, one has that the equivalence is proven if and only if DISPLAYFORM5 but this right term is easily proven to be nothing but DISPLAYFORM6 the inverse of the Legendre transformation of f applied to D KL (Q||P).This also opens up the possibility of using various softmax temperatures β i in practical algorithms in order to estimate f .

Finally, note that if P θ is a parameterized softmax policy associated with action-value functions Q θ (a, s) and temperature β, then because P * is proportional to e −r(a,s)/β , one readily has DISPLAYFORM7 which can easily be inserted in the inequality above for the special case Q = P * .

Entropic reinforcement learning has appeared early in the literature with two different motivations.

The view of exploration with a self-information intrinsic reward was pioneered by Tishby, and developed in Ziebart's PhD. thesis BID16 .

It was rediscovered recently that within the asynchronous actor-critic framework, entropic regularization is crucial to ensure convergence in practice V. .

Furthermore, the idea of taking steepest KL divergence steps as a practical reinforcement learning method per se was adopted by Schulman J. Schulman & Abbeel.

The key common development in these works has been to make entropic regularization recursively follow the Bellman equation, rather than naively regularizing one-step policies G. BID5 .

Schulman thereafter proposed a general proof of the equivalence, in the limit, of policy gradient and soft Q-learning methods J. BID7 , but the proof does not explicitly make the connection with convex duality and the expeditive justification it yields in the one-step case.

Applying the Gibbs/Donsker-Varadhan variational formula to entropy in a machine learning context is, however, not new; see for instance BID0 .

Some of the convex optimization results they invoke, including proximal stepping, can be found in the complete treatment by BID3 .

In the context of neural networks, convex analysis and partial differential equation methods are covered by BID10 .

Using dual formulas for the entropy functional in reinforcement learning has vast potential ramifications.

One avenue of research will be to interpret our findings in a large deviations framework -the log-sum-exp cumulant generative function being an example of rate function governing fluctuations of the tail of empirical n-step returns.

Smart drift change techniques could lead to significant variance reduction for Monte-Carlo rollout estimators.

We also hope to exploit further concentration inequalities in order to provide more bounds for the state value function.

Finally, a complete theory of the one-to-one correspondence between convex approximation algorithms and reinforcement learning methods is still lacking to date.

We hope to be able to contribute in this direction through further work.

@highlight

A short proof of the equivalence of soft Q-learning and policy gradients.
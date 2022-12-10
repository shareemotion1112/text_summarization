Multiagent systems where the agents interact among themselves and with an stochastic environment can be formalized as stochastic games.

We study a subclass of these games, named Markov potential games (MPGs), that appear often in economic and engineering applications when the agents share some common resource.

We consider MPGs with continuous state-action variables, coupled constraints and nonconvex rewards.

Previous analysis followed a variational approach that is only valid for very simple cases (convex rewards, invertible dynamics, and no coupled constraints); or considered deterministic dynamics and provided open-loop (OL) analysis, studying strategies that consist in predefined action sequences, which are not optimal for stochastic environments.

We present a closed-loop (CL) analysis for MPGs and consider parametric policies that depend on the current state and where agents adapt to stochastic transitions.

We provide easily verifiable, sufficient and necessary conditions for a stochastic game to be an MPG, even for complex parametric functions (e.g., deep neural networks); and show that a closed-loop Nash equilibrium (NE) can be found (or at least approximated) by solving a related optimal control problem (OCP).

This is useful since solving an OCP---which is a single-objective problem---is usually much simpler than solving the original set of coupled OCPs that form the game---which is a multiobjective control problem.

This is a considerable improvement over the previously standard approach for the CL analysis of MPGs, which gives no approximate solution if no NE belongs to the chosen parametric family, and which is practical only for simple parametric forms.

We illustrate the theoretical contributions with an example by applying our approach to a noncooperative communications engineering game.

We then solve the game with a deep reinforcement learning algorithm that learns policies that closely approximates an exact variational NE of the game.

In a noncooperative stochastic dynamic game, the agents compete in a time-varying environment, which is characterized by a discrete-time dynamical system equipped with a set of states and a state-transition probability distribution.

Each agent has an instantaneous reward function, which can be stochastic and depends on agents' actions and current system state.

We consider that both the state and action sets are subsets of real vector spaces and subject to coupled constraints, as usually required by engineering applications.

A dynamic game starts at some initial state.

Then, the agents take some action and the game moves to another state and gives some reward values to the agents.

This process is repeated at every time step over a (possibly) infinite time horizon.

The aim of each agent is to find the policy that maximizes its expected long term return given other agents' policies.

Thus, a game can be represented as a set of coupled optimal-control-problems (OCPs), which are difficult to solve in general.

OCPs are usually analyzed for two cases namely open-loop (OL) or closed-loop (CL), depending on the information that is available to the agents when making their decisions.

In the OL analysis, the action is a function of time, so that we find an optimal sequence of actions that will be executed in order, without feedback after any action.

In the CL setting, the action is a mapping from the state, usually referred as feedback policy or simply policy, so the agent can adapt its actions based on feedback from the environment (the state transition) at every time step.

For deterministic systems, both OL and CL solutions can be optimal and coincide in value.

But for stochastic system, an OL strategy consisting in a precomputed sequence of actions cannot adapt to the stochastic dynamics so that it is unlikely to be optimal.

Thus, CL are usually preferred over OL solutions.

For dynamic games, the situation is more involved than for OCPs, see, e.g., BID1 .

In an OL dynamic game, agents' actions are functions of time, so that an OL equilibrium can be visualized as a set of state-action trajectories.

In a CL dynamic game, agents' actions depend on the current state variable, so that, at every time step, they have to consider how their opponents would react to deviations from the equilibrium trajectory that they have followed so far, i.e., a CL equilibrium might be visualized as a set of trees of state-action trajectories.

The sets of OL and CL equilibria are generally different even for deterministic dynamic games BID10 BID5 .The CL analysis of dynamic games with continuous variables is challenging and has only be addressed for simple cases.

The situation is even more complicated when we consider coupled constraints, since each agent's actions must belong to a set that depends on the other agents' actions.

These games, where the agents interact strategically not only with their rewards but also at the level of the feasible sets, are known as generalized Nash equilibrium problems BID3 .There is a class of games, named Markov potential games (MPGs), for which the OL analysis shows that NE can be found by solving a single OCP; see BID6 BID25 for recent surveys on MPGs.

Thus, the benefit of MPGs is that solving a single OCP is generally simpler than solving a set of coupled OCPs.

MPGs appear often in economics and engineering applications, where multiple agents share a common resource (a raw material, a communication link, a transportation link, an electrical transmission line) or limitations (a common limit on the total pollution in some area).

Nevertheless, to our knowledge, none previous study has provided a practical method for finding CL Nash equilibrium (CL-NE) for continuous MPGs.

Indeed, to our knowledge, no previous work has proposed a practical method for finding or approximating CL-NE for any class of Markov games with continuous variables and coupled constraints.

State-of-the-art works on learning CL-NE for general-sum Markov games did not consider coupled constraints and assumed finite state-action sets BID18 BID16 .In this work, we extend previous OL analysis due to BID26 BID23 and tackle the CL analysis of MPGs with coupled constraints.

We assume that the agents' policies lie in a parametric set.

This assumption makes derivations simpler, allowing us to prove that, under some potentiality conditions on the reward functions, a game is an MPG.

We also show that, similar to the OL case, the Nash equilibrium (NE) for the approximate game can be found as an optimal policy of a related OCP.

This is a practical approach for finding or at least approximating NE, since if the parametric family is expressive enough to represent the complexities of the problem under study, we can expect that the parametric solution will approximate an equilibrium of the original MPG well (under mild continuity assumptions, small deviations in the parametric policies should translate to small perturbations in the value functions).

We remark that this parametric policy assumption has been widely used for learning the solution of single-agent OCPs with continuous state-action sets; see, e.g., BID9 Melo and Lopes, 2008; BID17 BID24 BID20 .

Here, we show that the same idea can be extended to MPGs in a principled manner.

Moreover, once we have formulated the related OCP, we can apply reinforcement learning techniques to find an optimal solution.

Some recent works have applied deep reinforcement learning (DRL) to cooperative Markov games BID4 BID22 , which are a particular case of MPGs.

Our results show that similar approaches can be used for more general MPGs.

We provide sufficient and necessary conditions on the agents' reward function for a stochastic game to be an MPG.

Then, we show that a closed-loop Nash equilibrium can be found (or at least approximated) by solving a related optimal control problem (OCP) that is similar to the MPG but with a single-objective reward function.

We provide two ways to obtain the reward function of this OCP: i) computing the line integral of a vector field composed of the partial derivatives of the agents' reward, which is theoretically appealing since it has the form of a potential function but difficult to obtain for complex parametric policies; ii) and as a separable term in the agents' reward function, which can be obtained easily by inspection for any arbitrary parametric policy.

We illustrate the proposed approach by applying DRL to a noncoooperative Markov game that models a communications engineering application (in addition, we illustrate the differences with the previous standard approach by solving a classic resource sharing game analytically in the appendix).

Let N {1, . . .

, N } denote the set of agents.

Let a k,i be the real vector of length A k that represents the action taken by agent k ∈ N at time i, where A k ⊆ R A k is the set of actions of agent k ∈ N .

Let A k∈N A k denote the set of actions of all agents that is the Cartesian product of every agent's action space, such that A ⊆ R A , where A = k∈N A k .

The vector that contains the actions of all agents at time i is denoted a i ∈ A. Let X ⊆ R S denote the set of states of the game, such that x i is a real vector of length S that represents the state of the game at time i, with components x i (s): DISPLAYFORM0 (1)Note that the dimensionality of the state set can be different from the number of agents (i.e., S = N ).

State transitions are determined by a probability distribution over the future state, conditioned on the current state-action pair: DISPLAYFORM1 where we use boldface notation for denoting random variables.

State transitions can be equivalently expressed as a function, f : X × A × Θ → X, that depends on some random variable θ i ∈ Θ, with distribution p θ (·|x i , a i ), such that DISPLAYFORM2 We include a vector of C constraint functions, DISPLAYFORM3 , where g c : X × A → R; and define the constraint sets for i = 0: C 0 A ∩ {a 0 : g(x 0 , a 0 ) ≤ 0}; and for i = 0, . . .

, ∞: C i {X ∩ {x i : DISPLAYFORM4 , which determine the feasible states and actions.

The instantaneous reward of each agent, r k,i , is also a random variable conditioned on the current state-action pair: DISPLAYFORM5 We assume that θ i and σ k,i are independent of each other and of any other θ j and σ k,j , at every time step j = i, given x i and a i .Let π k : X → A k and π : X → A denote the policy for agent k and all agents, respectively, such that: DISPLAYFORM6 Let Ω k and Ω = k∈N Ω k denote the policy spaces for agent k and for all agents, respectively, such that π k ∈ Ω k and π ∈ Ω. Note that Ω(X) = A. Introduce also π −k : X → A −k as the policy of all agents except that of agent k. Then, by slightly abusing notation, we write: DISPLAYFORM7 The general (i.e., nonparametric) stochastic game with Markov dynamics consists in a multiobjective variational problem with design space Ω and objective space R N , where each agent aims to find a stationary policy that maximizes its expected discounted cumulative reward, for which the vector of constraints, g, is satisfied almost surely: DISPLAYFORM8 Similar to static games, since there might not exist a policy that maximizes every agent's objective, we will rely on Nash equilibrium (NE) as solution concept.

But rather than trying to find a variational NE solution for (5), we propose a more tractable approximate game by constraining the policies to belong to some finite-dimensional parametric family.

Introduce the set of parametric policies, Ω w , as a finite-dimensional function space with parameter w ∈ W ⊆ R W : Ω w {π(·, w) : w ∈ W}. Note that for a given w, the parametric policy is still a mapping from states to actions: π(·, w) : X → A. Let w k ∈ W k ⊆ R W k denote the parameter vector of length W k for the parametrized policy π k , so that it lies in the finite-dimensional space Ω w k DISPLAYFORM9 Let w −k denote the parameters of all agents except that of agent k, so that we can also write: DISPLAYFORM10 In addition, we use w k ( ) to denote the -th component of DISPLAYFORM11 .

By constraining the policy of G 1 to lie in Ω w k , we obtain a multiobjective optimization problem with design space W: DISPLAYFORM12 The solution concept in which we are interested is the parametric closed-loop Nash equilibrium (PCL-NE), which consists in a parametric policy for which no agent has incentive to deviate unilaterally.

DISPLAYFORM13 Since G 2 is similar to G 1 but with an extra constraint on the policy set, loosely speaking, we can see a PCL-NE as a projection of some NE of G 1 onto the manifold spanned by parametric family of choice.

Hence, if the parametric family has arbitrary expressive capacity (e.g., a neural network with enough neurons in the hidden layers), we can expect that the resulting PCL-NE evaluated on G 1 will approximate arbitrarily close the performance of an exact variational equilibrium.

We consider the following general assumptions.

The state and parameter sets, X and W, are nonempty and convex.

Assumption 2 The reward functions r k are twice continuously differentiable in X × W, ∀k ∈ N .

The state-transition function, f , and constraints, g, are continuously differentiable in X × W, and satisfy some regularity conditions (e.g., Mangasarian-Fromovitz).Assumption 4 The reward functions r k are proper, and there exists a scalar B such that the level DISPLAYFORM0 are nonempty and bounded ∀k ∈ N .Assumptions 1-2 usually hold in engineering applications.

Assumption 3 ensures the existence of feasible dual variables, which is required for establishing the optimality conditions.

Assumption 4 will allow us to ensure the existence of PCL-NE.

We say that r k is proper if: DISPLAYFORM1

In this section, we review the standard approach for tackling CL dynamic games (González-Sánchez and Hernández-Lerma, 2013).

For simplicity, we consider deterministic game and no constraints: DISPLAYFORM0 First, it inverts f to express the policy in reduced form, i.e., as a function of current and future states: DISPLAYFORM1 (11) This implicitly assumes that such function h : X × X → A exists, which might not be the case if f is not invertible.

Next, π k is replaced with (11) in each r k : DISPLAYFORM2 where r k : X×X → R is the reward in reduced-form.

Then, the Euler equation (EE) and transversality condition (TC) are obtained from r k for all k ∈ N and used as necessary optimality conditions: DISPLAYFORM3 When r k are concave for all agents, and X ⊆ R + (i.e., X = {x i : DISPLAYFORM4 , these optimality conditions become sufficient for Nash equilibrium (González-Sánchez and Hernández-Lerma, 2013, Theorem 4.1).

Thus, the standard approach consists in guessing parametric policies from the space of functions Ω, and check whether any of these functions satisfies the optimality conditions.

We illustrate this procedure with a well known resource-sharing game named "the great fish war" due to BID11 , with Example 1 in Appendix A.Although the standard approach sketched above (see also Appendix A) has been the state-of-the-art for the analysis of CL dynamic games, it has some drawbacks: i) The reduced form might not exist; ii) constraints are not handled easily and we have to rely in ad hoc arguments for ensuring feasibility; iii) finding a specific parametric form that satisfies the optimality conditions can be extremely difficult since the space of functions is too large; and iv) the rewards have to be concave for all agents in order to guarantee that any policy that satisfies the conditions is an equilibrium.

In order to overcome these issues, we propose to first constrain the set of policies to some parametric family, and then derive the optimality conditions for this parametric problem; as opposed to the standard approach that first derives the optimality conditions of G 1 , and then guesses a parametric form that satisfies them.

Based on this insight, we will introduce MPG with parametric policies as a class of games that can be solved with standard DRL techniques by finding the solution of a related (single-objective) OCP.

We explain the details in the following section.

In this section, we extend the OL analysis of BID25 to the CL case.

We define MPGs with CL information structure; introduce a parametric OCP; provide verifiable conditions for a parametric approximate game to be an MPG in the CL setting; show that when the game is an MPG, we can find a PCL-NE by solving the parametric OCP with a specific objective function; and provide a practical method for obtaining such objective function.

First, we define MPGs with CL information structure and parametric policies as follows.

Definition 2 Given a policy family π(·, w) ∈ Ω w , game (8) is an MPG if and only if there is a function J : X × W × Σ → R, named the potential, that satisfies the following condition ∀k ∈ N : DISPLAYFORM0 Definition 2 means that there exists some potential function, J, shared by all agents, such that if some agent k changes its policy unilaterally, the change in its reward, r k , equals the change in J.The main contribution of this paper is to show that when (8) is a MPG, we can find one PCL-NE by solving a related parametric OCP.

The generic form of such parametric OCP is as follows: DISPLAYFORM1 where we replaced the multiple objectives (one per agent) with the potential J as single objective.

This is convenient since solving a single objective OCP is generally much easier than solving the Markov game.

However, we still have to find out how to obtain J. The following Theorem formalizes the relationship between G 2 and P 1 and shows one way to obtain J (proof in Appendix C).Theorem 1 Let Assumptions 1-4 hold.

Let the reward functions satisfy the following ∀k, j ∈ N : DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 where the expected value is taken component-wise.

Then, game (8) is an MPG that has a PCL-NE equal to the solution of OCP (16).

The potential J that is the instantaneous reward for the OCP is given by line integral: DISPLAYFORM5 where η(z) (η k (z)) S m=1 and ξ(z) (ξ k (z)) k∈N are piecewise smooth paths in X and W, respectively, with components ξ k (z) (ξ k, (z)) W k =1 , such that the initial and final state-action conditions are given by (η(0), ξ(0)) and (η(1) = x i , ξ(1) = w).From FORMULA2 , we can see that J is obtained through the line integral of a vector field with components the partial derivatives of the agents' rewards (see Appendix C), and so the name potential function.

Note also that Theorem 1 proves that any solution to P 1 is also a PCL-NE of G 2 , but we remark that there may be more equilibria of the game that are not solutions to P 1 (see Appendix C).The usefulness of Theorem 1 is that, once we have the potential function, we can formulate and solve the related OCP for any specific parametric policy family.

This is a considerable improvement over the standard approach.

On one hand, if the chosen parametric policy contains the optimal solution, then we will obtain the same equilibrium as the standard approach.

On the other hand, if the chosen parametric family does not have the optimal solution, the standard approach will fail, while our approach will always provide a solution that is an approximation (a projection over Ω w ) of an exact variational equilibrium.

Moreover, as mentioned above, we can expect that the more expressive the parametric family, the more accurate the approximation to the variational equilibrium.

In Appendix B, we show how to to solve "the great fish war" game with the proposed framework, yielding the same solution as with the standard approach, with no loss of accuracy.

Although expressing J as a line integral of a field is theoretically appealing, if the parametric family is involved-as it is usually the case for expressive policies like deep neural-networks-then (20) might be difficult to evaluate.

The following results show how to obtain J easily by visual inspection.

First, the following corollary follows trivially from FORMULA10 - FORMULA25 and shows that cooperative games, where all agents have the same reward, are MPGs, and the potential equals the reward:Corollary 1 Cooperative games, where all agents have a common reward, such that DISPLAYFORM6 are MPGs; and the potential function (20) equals the common reward function in (21).Second, we address noncooperative games, and show that the potential can be found by inspection as a separable term that is common to all agents' reward functions.

Interestingly, we will also show that a game is an MPG in the CL setting if and only if all agents' policies depend on disjoint subsets of components of the state vector.

More formally, introduce X π k as the set of state vector components that influence the policy of agent k and introduce a new state vector, x π k , and let x π −k,i be the vector of components that do not influence the policy of agent k: DISPLAYFORM7 In addition, introduce X r k as the set of components of the state vector that influence the reward of agent k directly (not indirectly through any other agent's policy), and define the state vectors: DISPLAYFORM8 Introduce also the union of these two subsets, DISPLAYFORM9 , and its corresponding vectors: DISPLAYFORM10 Then, the following theorem allows us to obtain the potential function (proof in Appendix D).Theorem 2 Let Assumptions 1-4 hold.

Then, game (8) is an MPG if and only if: i) the reward function of every agent can be expressed as the sum of a term common to all agents plus another term that depends neither on its own state-component vector, nor on its policy parameter: DISPLAYFORM11 25) and ii) the following condition on the non-common term holds: DISPLAYFORM12 Moreover, if (26) holds, then the common term in (25), J, equals the potential function (20).Note that (26) holds in the following cases: i) when Θ k = 0, as the cooperative case described in Corollary 1; ii) when Θ k does not depend on the state but only on the parameter vector, i.e., Θ k : j∈N ,j =k W j → R, as in "the great fish war" example described in Appendix B; or iii) when all agents have disjoint state-component subsets, i.e., X Θ k ∩ X Θ j = ∅, ∀(k, j) ∈ {N × N : k = j}. An interesting insight from Theorem 2 is that a dynamic game that is potential when it is analyzed in the OL case (i.e., the policy is a predefined sequence of actions), might not be potential when analyzed in the CL parametric setting.

This conclusion is straightforward since the potentiality condition in the OL case provided by (Valcarcel Macua et al., 2016, Cor.

1

In order to apply Theorems 1 and 2, we are implicitly assuming that there exists solution to the OCP.

We finish this section, by showing that this is actually the case in our setting (proof in Appendix E).

In other words, Prop.

1 shows that there exists a deterministic policy that achieves the optimal value of P 1 , which is also an NE of G 2 if conditions (17)-(19) or equivalently (25)-(26) hold.

We remark that there might be many other-possibly stochastic-policies that are also NE of the game.

In this section, we show how to use the proposed MPGs framework to learn an equilibrium of a communications engineering application.

We extend the Medium Access Control (MAC) game presented in BID25 to stochastic dynamics and rewards (where previous OL solutions would fail), and use the Trust Region Policy Optimization (TRPO) algorithm BID20 , which is a reliable reinforcement learning method policy search method that approximates the policy with a deep-neural network, to learn a policy that is a PCL-NE of the game.

We consider a MAC uplink scenario with N = 4 agents, where each agent is a user that sets its transmitter power aiming to maximize its data rate and battery lifespan.

If multiple users transmit at the same time, they will interfere with each other and decrease their rate, using their batteries inefficiently, so that they have to find an equilibrium.

Let x k,i ∈ [0, B k,max ]

X k denote the battery level for each agent k ∈ N , which is discharged proportionally to the transmitted power, Let a k,i ∈ [0, P k,max ]

A k be the transmitted power for the k-th user, where constants P k,max and B k,max stand for the maximum allowed transmitter power and battery level, respectively.

The system state is the vector with all user's battery levels: x i = (x k,i ) k∈N ∈ X; such that S = N and all state vector components are unshared, i.e., X = k∈N X k ⊂ R N , and X k = {k}. We remark that although each agent's battery depletion level depends directly on its action and its previous battery level only, it also depends indirectly on the strategies and battery levels of the rest of agents.

The game can be formalized as follows: DISPLAYFORM0 where h k is the random fading channel coefficient for user k, α is the weight for the battery reward term, and δ is the discharging factor.

First of all, note that each agent's policy and reward depend only on its own battery level, x k,i .

Therefore, we can apply Theorem 2 and establish that the game is a MPG, with potential function: DISPLAYFORM1 Thus, we can formulate OCP (16) with single objective given by (28).Since the battery level is a positive term in the reward, the optimal policy will make the battery deplete in finite time (formal argument can be derived from transversality condition (54)).

Moreover, since δ k,i ≥ 0, the episode gets into a stationary (i.e., terminal) state once the battery has been depleted.

We have chosen the reward to be convex.

The reason is that in order to compute a benchmark solution, we can solve the finite time-horizon convex OCP exactly with a convex optimization solver, e.g., CVX BID7 , and use the result as a baseline for comparing with the solution learned by a DRL algorithm.

Nevertheless, standard solvers do not allow to include random variables.

To surmount this issue, we generated 100 independent sequences of samples of h k,i and δ k,i for all k ∈ N and length T = 100 time steps each, and obtain two solutions with them.

We set |h DISPLAYFORM2 where v k,i is uniform in [0.5, 1], |h 1 | 2 = 2.019, |h 2 | 2 = 1.002, |h 3 | 2 = 0.514 and |h 4 | 2 = 0.308; and δ k,i is uniform in [0.7, 1.3].

The first solution is obtained by averaging the sequences, and building a deterministic convex problem with the average sequence, which yielded an optimal value V cvx = 33.19.

We consider V cvx to be an estimator of the optimal value of the stochastic OCP.

The second solution is obtained by building 100 deterministic problems, solving them, and averaging their optimal values, which yielded an optimal value V avg,cvx = 34.90.

We consider V avg,cvx to be an upper bound estimate of the optimal value of the stochastic OCP (Jensen's inequality).

The batteries depleted at a level x T < 10 −6 in all cases, concluding that time horizon of T = 100 steps is valid.

We remark that these benchmark solutions required complete knowledge of the game.

When we have no prior knowledge of the dynamics and rewards, the proposed approach allows as to learn a PCL-NE of (27) by using any DRL method that is suitable for continuous state and actions, like TRPO BID20 , DDPG or A3C BID15 .

DRL methods learn by interacting with a black-box simulator, such that at every time step i, agents observe state x i , take action a i = π w (x i ) and observe the new stochastic battery levels and reward values, with no prior knowledge of the reward or state-dynamic functions.

As a proof of concept, we perform simulations with TRPO, approximating the policy with a neural network with 3 hidden layers of size 32 neurons per layer and RELU activation function, and an output layer that is the mean of a Gaussian distribution.

Each iteration of TRPO uses a batch of size 4000 simulation steps (i.e., tuples of state transition, action and rewards).

The step-size is 0.01.

FIG0 shows the results.

After 400 iterations, TRPO achieves an optimal value V trpo = 32.34, which is 97.44% of V cvx , and 92.7% of the upper bound V avg,cvx .

We have extended previous results on MPGs with constrained continuous state-action spaces providing practical conditions and a detailed analysis of Nash equilibrium with parametric policies, showing that a PCL-NE can be found by solving a related OCP.

Having established a relationship between a MPG and an OCP is a significant step for finding an NE, since we can apply standard optimal control and reinforcement learning techniques.

We illustrated the theoretical results by applying TRPO (a well known DRL method) to an example engineering application, obtaining a PCL-NE that yields near optimal results, very close to an exact variational equilibrium.

A EXAMPLE: THE "GREAT FISH WAR" GAME -STANDARD APPROACH Let us illustrate the standard approach described in Section 3 with a well known resource-sharing game named "the great fish war" due to BID11 .

We follow (González-Sánchez and Hernández-Lerma, 2013, Sec. 4.2).

Example 1.

Let x i be the stock of fish at time i, in some fishing area.

Suppose there are N countries obtaining reward from fish consumption, so that they aim to solve the following game: DISPLAYFORM0 where x 0 ≥ 0 and 0 < α < 1 are given.

In order to solve G fish , let us express each agent's action as: DISPLAYFORM1 so that the rewards can be also expressed in reduced form, as required by the standard-approach: DISPLAYFORM2 Thus, the Euler equations for every agent k ∈ N and all t = 0, . . .

, ∞ become: DISPLAYFORM3 Now, the standard method consists in guessing a family of parametric functions that replaces the policy, and checking whether such parametric policy satisfies (32) for some parameter vector.

Let us try with policies that are linear mappings of the state: DISPLAYFORM4 By replacing (33) in (32), we obtain the following set of equations: DISPLAYFORM5 Fortunately, it turns out that (34) has solution (which might not be the case for other policy parametrization), with parameters given by: DISPLAYFORM6 Since 0 < α < 1 and 0 ≤ γ < 1, it is apparent that w k > 0 and the constraint π k (x i ) ≥ 0 holds for all x i ≥ 0.

Moreover, since k∈N w k < 1, we have that x i+1 ≥ 0 for any x 0 ≥ 0.

In addition, since x i is a resource and the actions must be nonnegative, it follows that lim i→∞ x i = 0 (there is no reason to save some resource).

Therefore, the transversality condition holds.

Since the rewards are concave, the states are non-negative and the linear policies with these coefficients satisfy the Euler and transversality equations, we conclude that they constitute an equilibrium (González-Sánchez and Hernández-Lerma, 2013, Theorem 4.1).B EXAMPLE: "GREAT FISH WAR" GAME -PROPOSED APPROACHIn this section, we illustrate how to apply the proposed approach with the same "the great fish war" example, obtaining the same results as with the standard approach.

Example 2.

Consider "the great fish war" game described in Example 1.

In order to use our approach, we replace the generic policy with the specific policy mapping of our preference.

We choose the linear mapping, π k (x i ) = w k x i , to be able to compare the results with those obtained with the standard approach.

Thus, we have the following game: DISPLAYFORM7 Let us verify conditions FORMULA9 - FORMULA9 .

For all k, j ∈ N we have: DISPLAYFORM8 DISPLAYFORM9 Since conditions FORMULA9 - FORMULA9 hold, we conclude that FORMULA5 is an MPG.

By applying the line integral FORMULA2 , we obtain: DISPLAYFORM10 Now, we can solve OCP (16) with potential function (43).

For this particular problem, it is easy to solve the KKT system in closed form.

Introduce a shorthand: DISPLAYFORM11 The Euler-Lagrange equation (62) for this problem becomes: DISPLAYFORM12 The optimality condition (64) with respect to the policy parameter becomes: DISPLAYFORM13 Let us solve for β i in (46): DISPLAYFORM14 Replacing FORMULA6 and the state-transition dynamics in FORMULA6 , we obtain the following set of equations: DISPLAYFORM15 Hence, the parameters can be obtained as: DISPLAYFORM16 This is exactly the same solution that we obtained in Example 1 with the standard approach.

We remark that for the standard approach, we were able to obtain the policy parameters since we put the correct parametric form of the policy in the Euler equation.

If we had used another parametric family without a linear term, the Euler equations (32) might have no solution and we would have got stuck.

In contrast, with our approach, we could freely choose any other form of the parametric policy, and always solve the KKT system of the approximate game.

Broadly speaking, we can say that the more expressive the parametric family, the more likely that the optimal policy of the original game will be accurately approximated by the optimal solution of the approximate game.

Proof: The proof mimics the OL analysis from BID25 .

Let us build the KKT systems for the game and the OCP with parametric policies.

For game (8), each agent's Lagrangian is given ∀k ∈ N by DISPLAYFORM0 where DISPLAYFORM1 ∈ R C are the vectors of multipliers at time i (which are random since they depend on θ i and x i ), and we introduced: DISPLAYFORM2 Introduce a shorthand for the instantaneous Lagrangian of agent k: DISPLAYFORM3 The discrete time stochastic Euler-Lagrange equations applied to each agent's Lagrangian are different from the OL case studied in BID25 (see also (Sage and White, 1977, Sec. 6 .1)), since we only take into account the variation with respect to the state: DISPLAYFORM4 where 0 S denotes the vector of length S. The transversality condition is given by DISPLAYFORM5 In addition, we have an optimality condition for the policy parameter w k : DISPLAYFORM6 From these first-order optimality conditions, we obtain the KKT system for every agent k ∈ N and all time steps i = 1, . . .

, ∞: DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9 DISPLAYFORM10 where λ k,i−1 is considered deterministic since it is known at time i. Now, we derive the KKT system of optimality conditions for the OCP (16).

The Lagrangian for (16) is given by: DISPLAYFORM11 where DISPLAYFORM12 ∈ R C are the corresponding multipliers, which are random variables since they depend on θ i and x i .

By taking the discrete time stochastic EulerLagrange equations and the optimality condition with respect to the policy parameter for the OCP, we obtain are a KKT system for the OCP: i = 1, . . .

, ∞: DISPLAYFORM13 DISPLAYFORM14 DISPLAYFORM15 DISPLAYFORM16 DISPLAYFORM17 where β i−1 is known at time i and includes the multipliers related to x i−1 .By comparing FORMULA8 - FORMULA9 and FORMULA2 - FORMULA9 , we conclude that both KKT systems are equal if the following holds ∀k ∈ N and i = 1, . . .

, ∞: DISPLAYFORM18 DISPLAYFORM19 DISPLAYFORM20 Since Assumption 4 ensures existence of primal variable for the OCP, Assumption 3 guarantee the existence of dual variables that satisfy its KKT system.

By applying (69) and replacing the dual variables of the KKT of the game with the OCP dual variables for every agent, we obtain a system of equations where the only unknowns are the user strategies.

This system is similar to the OCP in the primal variables.

Therefore, the OCP primal solution also satisfies the KKT necessary conditions of the game.

Moreover, from the potentiality condition, it is straightforward to show that this primal solution of the OCP is also a PCL-NE of the MPG (see also BID25 , Theorem 1)).Introduce the following vector field:F (x i , w, σ i ) ∇ (xi,w) J (x i , π(x i , w), σ i ) .Since F is conservative by construction (Apostol, 1969, Theorems 10.4, 10.5 and 10.9) , conditions (67)-(68) are equivalent to FORMULA10 - FORMULA25 and we can calculate a potential J through line integral (20).

Proof: We can rewrite game (8) by making explicit that the actions result from the policy mapping, which yields an expression that reminds the OL problem but with extra constraints: DISPLAYFORM0 where it is clear that: a i (a k,i , a −k,i ) = π(x i , w) Rewrite also OCP (16) with explicit dependence on the actions: DISPLAYFORM1 By following the Euler-Lagrange approach described in Theorem 1, we have that the KKT systems for game and OCP are equal if the dual variables are equal (including new extra dual variables for the equality constraints that relate the action and the policy) and the following first-order conditions hold ∀k ∈ N and i = 1, . . .

, ∞: DISPLAYFORM2 E ∇ a k,i r k x r k,i , a k,i , a −k,i , σ k,i = E ∇ a k,i J (x i , a i , σ i ) .The benefit of this reformulation is that the gradient in (73) is taken with respect to the components in X r k only (instead of the whole set X ), at the cost of replacing (68) with the sequence of conditions (74).

We have to realize that a k,i is indeed a function of variables x π k,i and w k .

In order to understand the influence of this variable change, we use the identity a k,i = π w k (x π k,i ) and apply the chain rule to both sides of (74), obtaining: DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 In addition, since J is proper, it must be upper bounded, i.e., ∃U ∈ R, such that J ≤ U .

Then, we have: DISPLAYFORM7 Since B ≤ U , we have that DISPLAYFORM8 Therefore, the level sets (86) are bounded.

From Assumption 2 the fact that J can be obtained from line integral FORMULA2 , and fundamental theorem of calculus, we deduce that J is continuous.

Therefore, we conclude that these level sets are also compact.

Thus, we can use (Bertsekas, 2007, Prop.

3.1.7 , see also Sections 1.2 and 3.6) to ensure existence of an optimal policy.

<|TLDR|>

@highlight

We present general closed loop analysis for Markov potential games and show that deep reinforcement learning can be used for learning approximate closed-loop Nash equilibrium.
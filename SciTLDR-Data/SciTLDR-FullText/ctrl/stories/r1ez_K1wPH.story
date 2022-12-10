Despite an ever growing literature on reinforcement learning algorithms and applications, much less is known about their statistical inference.

In this paper, we investigate the large-sample behaviors of the Q-value estimates with closed-form characterizations of the asymptotic variances.

This allows us to efficiently construct confidence regions for Q-value and optimal value functions, and to develop policies to minimize their estimation errors.

This also leads to a policy exploration strategy that relies on estimating the relative discrepancies among the Q estimates.

Numerical experiments show superior performances of our exploration strategy than other benchmark approaches.

We consider the classical reinforcement learning (RL) problem where the agent interacts with a random environment and aims to maximize the accumulated discounted reward over time.

The environment is formulated as a Markov decision process (MDP) and the agent is uncertain about the true dynamics to start with.

As the agent interacts with the environment, data about the system dynamics are collected and the agent becomes increasingly confident about her decision.

With finite data, however, the potential reward from each decision is estimated with errors and the agent may be led to a suboptimal decision.

Our focus in this paper is on statistically efficient methodologies to quantify these errors and uncertainties, and to demonstrate their use in obtaining better policies.

More precisely, we investigate the large-sample behaviors of estimated Q-value, optimal value function, and their associated policies.

Our results are in the form of asymptotic convergence to an explicitly identified and computable Gaussian (or other) distribution, as the collected data sizes increase.

The motivation of our investigation is three-fold.

First, these precise asymptotic statements allow us to construct accurate confidence regions for quantities related to the optimal policy, and, like classical statistical inference, they can assess the reliability of the current estimates with respect to the data noises.

Second, our results complement some finite-sample error bounds developed in the literature (Kearns & Singh, 1998; Kakade, 2003; Munos & Szepesvári, 2008) , by supplementing a closed-form asymptotic variance that often shows up in the first-order terms in these bounds.

Our third and most important motivation is to design good exploration policies by directly using our tight error estimates.

Motivated by recent autonomous-driving and other applications (e.g., Kalashnikov et al. (2018) ), we consider the pure exploration setting where an agent is first assigned an initial period to collect as much experience as possible, and then, with the optimal policy trained offline, starts deployment to gain reward.

We propose an efficient strategy to explore by optimizing the worst-case estimated relative discrepancy among the Q-values (ratio of mean squared difference to variance), which provides a proxy for the probability of selecting the best policy.

Similar criteria have appeared in the so-called optimal computing budget allocation (OCBA) procedure in simulation-based optimization (Chen & Lee, 2011 ) (a problem closely related to best-arm identification (Audibert & Bubeck, 2010) in online learning).

In this approach, one divides computation (or observation) budget into stages in which one sequentially updates mean and variance estimates, and optimizes next-stage budget allocations according to the worst-case relative discrepancy criterion.

Our proposed procedure, which we term Q-OCBA, follows this idea with a crucial use of our Q-value estimates and randomized policies to achieve the optimal allocation.

We demonstrate how this idea consistently outperforms other benchmark exploration policies, both in terms of the probability in selecting the best policy and generating the tightest confidence bounds for value estimates at the end of the exploration period.

Regarding the problem of constructing tight error estimates in RL, the closest work to ours is Mannor et al. (2004; 2007) , which studies the bias and variance in value function estimates with a fixed policy.

Our technique resolves a main technical challenge in Mannor et al. (2004; 2007) , which allows us to substantially generalize their variance results to Q-values, optimal value functions and asymptotic distributional statements.

The derivation in Mannor et al. (2004; 2007) hinges on an expansion of the value function in terms of the perturbation of the transition matrix, which (as pointed out by the authors) is not easily extendable from a fixed-policy to the optimal value function.

In contrast, our results utilize an implicit function theorem applied to the Bellman equation that can be verified to be sufficiently smooth.

This idea turns out to allow us to obtain gradients for Q-values, translate to the optimal value function, and furthermore generalize to similar results for constrained MDP and approximate value iterations.

We also relate our work to the line of studies on dynamic treatment regimes (DTR) (Laber et al., 2014) applied commonly in medical decision-making, which focuses on the statistical properties of polices on finite horizon (such as two-period).

Our infinite-horizon results on the optimal value and Q-value distinguishes our developments from the DTR literature.

Moreover, our result on the non-unique policy case can be demonstrated to correspond to the "non-regularity" concept in DTR, where the true parameters are very close to the decision "boundaries" that switch the optimal policy (motivated by situations of small treatment effects), thus making the obtained policy highly sensitive to estimation noises.

In the rest of this paper, we first describe our MDP setup and notations (Section 2).

Then we present our results on large-sample behaviors (Section 3), demonstrate their use in exploration strategies (Section 4), and finally substantiate our findings with experimental results (Section 5).

In the Appendix, we first present generalizations of our theoretical results to constrained MDP (A.1) and problems using approximate value iteration (A.2).

Then we include more numerical experiments (B), followed by all the proofs (C).

Consider an infinite horizon discounted reward MDP, M = (S, A, R, P, γ, ρ), where S is the state space, A is the action space, R(s, a) denotes the random reward when the agent is in state s ∈ S and selects action a ∈ A, P (s |s, a) is the probability of transitioning to state s in the next epoch given current state s and taken action a, γ is the discount factor, and ρ is the initial state distribution.

The distribution of the reward R and the transition probability P are unknown to the agent.

We assume both S and A are finite sets.

Without loss of generality, we denote S = {1, 2, . . .

, m s } and A = {1, 2, . . .

, m a }.

Finally, we make the following stochasticity assumption: Assumption 1.

R(s, a) has finite mean µ R (s, a) and finite variance σ 2 R (s, a) ∀ s ∈ S, a ∈ A. For any given s ∈ S and a ∈ A, R(s, a) and S ∼ P (·|s, a) are all independent random variables.

A policy π is a mapping from each state s ∈ S to a probability measure over actions a ∈ A. Specifically, we write π(a|s) as the probability of taking action a when the agent is in state s and π(·|s) as the m adimensional vector of action probabilities at state s.

For convenience, we sometimes write π(s) as the realized action given the current state is s.

The value function associated with a policy π is defined as

For convenience, we denote V * = V π * and χ * = s ρ(s)V * (s).

The Q-value, denoted by Q(s, a), is defined as Q(s, a) = µ R (s, a) + γE[V * (S )|s, a].

Correspondingly, V * (s) = max a Q(s, a) and the Bellman equation for Q takes the form

for any (s, a) ∈ S × A. Denoting the Bellman operator as T µ R ,P (·), Q is a fixed point associated with T µ R ,P , i.e. Q = T µ R ,P (Q).

For the most part of this paper we make the following assumption about Q: Assumption 2.

For any state s ∈ S, arg max a∈A Q(s, a) is unique.

Under Assumption 2, the optimal policy π * is unique and deterministic.

Let a * (s) = arg max a∈A Q(s, a).

Then π * (a|s) = 1 (a = a * (s)), where 1(·) denotes the indicator function.

We next introduce some statistical quantities arising from data.

Suppose we have n observations (whose collection mechanism will be made precise later), which we denote as {(s t , a t , r t (s t , a t ), s t (s t , a t )) : 1 ≤ t ≤ n}, where r t (s t , a t ) is the realized reward at time t and s t (s t , a t ) = s t+1 .

We define the sample meanμ R,n and the sample varianceσ 2 R,n of the reward aŝ

Similarly, we define the empirical transition matrixP n aŝ

and its m s × m s sampling covariance matrix Σ Ps,a (with one sample point of 1(s t = s, a t = a)) as

With the data, we construct our estimate of Q, calledQ n , which is the empirical fixed point of Tμ R,n ,Pn , i.e.

Q n = Tμ R,n ,Pn (Q n ).

Correspondingly, we also writeV * n (s) = max a∈AQn (s, a) and χ * n = s∈S ρ(s)V * n (s).

We shall focus on the empirical errors due to noises of the collected data, and assume the MDP or Q-value evaluation can be done off-line so that the fixed point equation forQ n can be solved exactly. .

We present an array of results regarding the asymptotic behaviors ofQ n andV * n .

To prepare, we first make an assumption on our exploration policy π to gather data.

Define the extended transition probabilityP π asP π (s , a |s, a) = P (s |s, a)π(a |s ).

We make the assumption:

Assumption 3.

The Markov chain with transition probabilityP π is positive recurrent.

Under Assumption 3,P π has a unique stationary distribution, denoted w, equal to the long run frequency in visiting each state-action pair, i.e. w(s, a) = lim n→∞ 1 n 1≤t≤n 1(s t = i, a t = j), where all w(s, a)'s are positive.

Note that Assumption 3 is satisfied if for any two states s, s , there exists a sequence of actions such that s is attainable from s under P , and, moreover, if π is sufficiently mixed, e.g., π satisfies π(a |s ) > 0 for all s , a .

Our results in the sequel uses the following further notations.

We denote "⇒" as "convergence in distribution", and N (µ, Σ) as a multivariate Gaussian distribution with mean vector µ and covariance matrix Σ. We write I as the identity matrix, and e i as the i-th unit vector.

The dimension of N (µ, Σ), I and e i should be clear from the context.

When not specified, all the vectors are column vectors.

Let N = m s m a .

In our algebraic derivations, we need to re-arrange µ R , Q and w as N -dimensional vectors.

We thus define the following indexing rule: (s = i, a = j) is re-indexed as

We also need to re-arrangeP π as an N × N matrix following the same indexing rule, i.e.

We first establish the asymptotic normality ofQ n under exploration policy π: Theorem 1.

Under Assumptions 1 and 2, if the data is collected according to π satisfying Assumption 3, thenQ n is a strongly consistent estimator of Q, i.e.

Q n → Q almost surely as n → ∞. Moreover,

In addition to the asymptotic Gaussian behavior, a key element of Theorem 1 is the explicit form of the asymptotic variance Σ. This is derived from the delta method (Serfling, 2009) and, intuitively, is the product of the sensitivities (i.e., gradient) of Q with respect to its parameters and the variances of the parameter estimates.

Here the parameters are µ R and P , with corresponding gradients (I − γP

The variances of these parameter estimates (i.e., (2) and (4)) involve σ 2 R (i, j) and Σ Pi,j , and the sample size allocated to estimate each parameter, which is proportional to w(i, j).

Using the relations that V * n (s) = max a∈A Q(s, a) andV * n (s) = max a∈AQn (s, a), we can leverage Theorem 1 to further establish the asymptotic normality ofV * n andχ * n : Corollary 1.

Under Assumptions 1, 2 and 3,

In the Appendix we also prove, using the same technique as above, a result on the large-sample behavior of the value function for a fixed policy (Corollary 2), which essentially recovers Corollary 4.1 in Mannor et al. (2007) .

Different from Mannor et al. (2007) , we derive our results by using an implicit function theorem on the corresponding Bellman equation to obtain the gradient of Q, viewing the latter as the solution to the equation and as a function of µ R , P .

This approach is able to generalize the results for fixed policies in Mannor et al. (2007) to the optimal value functions, and also provide distributional statements as Theorem 1 and Corollary 1 above.

We also note that another potential route to obtain our results is to conduct perturbation analysis on the linear program (LP) representation of the MDP, which would also give gradient information of V * (and hence also Q), but using the implicit function theorem here seems sufficient.

Theorem 1 and Corollary 1 can be used immediately for statistical inference.

In particular, we can construct confidence regions for subsets of the Q-value jointly, or for linear combinations of the Q-values.

A quantity of interest that we will later utilize in designing good exploration policies is Q(s, a 1 ) − Q(s, a 2 ), i.e. the difference between action a 1 and a 2 when the agent is in state s. Define σ

and its estimatorσ 2 ∆Q,n by replacing Q, V * , σ 2 R,n , w, P withQ n ,V * nσ 2 R,n ,ŵ n ,P n in Σ, wherê w n is the empirical frequency of visiting each state-action pair, i.e.ŵ n (i, j) =

Suppose the optimal policy for the MDP M is not unique, i.e., Assumption 2 does not hold.

In this situation, the estimatedQ n andV * n may "jump" around different optimal actions, leading to a more complicated large-sample behavior as described below:

Theorem 2.

Suppose Assumptions 1 and 3 hold but there is no unique optimal policy.

Then there exists K ≥ 1 distinct m s ×(N m s +N ) matrices {G k } 1≤k≤K and a deterministic partition of U = {u ∈ R msN +ms :

In the case that K > 1 in Theorem 2, the limit distribution becomes non-Gaussian.

This arises because the sensitivity to P or µ R can be very different depending on the perturbation direction, which is a consequence of solution non-uniqueness that can be formalized as a non-degeneracy in the LP representation of the MDP.

We note that this phenomenon is analogous to the "non-regularity" concept in DTR that arises because the "true" parameters in these problems are very close to the decision "boundaries", which makes the obtained policy highly sensitive to estimation noises and incurs a 1/ √ n-order bias behavior.

Our case of non-unique optimal policy here captures precisely this same behavior, where we see in Theorem 2 that when K > 1 the asymptotic limit no longer has mean zero and consequently a 1/ √ n-order bias arises.

We also develop two other generalizations of large-sample results, for constrained MDP and approximate value iteration respectively (see Appendices A.1 and A.2).

We utilize our results in Section 3 to design exploration policies.

We focus on the setting where an agent is assigned a period to collect data by running the state transition with an exploration policy.

The goal is to obtain the best policy at the end of the period in a probabilistic sense, i.e., minimize the probability of selecting a suboptimal policy for the accumulated reward.

We propose a strategy that maximizes the worst-case relative discrepancy among all Q-value estimates.

More precisely, we define, for i ∈ S, j ∈ A and j = a * (i), the relative discrepancy as

is defined in (6).

Our procedure attempts to maximize the minimum of h ij 's,

where w denotes the proportions of visits on the state-action pairs, within some allocation set W η (which we will explain).

Intuitively, h ij captures the relative "difficulty" in obtaining the optimal policy given the estimation errors of Q's.

If the Q-values are far apart, or if the estimation variance is small, then h ij is large which signifies an "easy" problem, and vice versa.

Criterion (7) thus aims to make the problem the "easiest".

Alternatively, one can also interpret (7) from a large deviations view (Glynn & Juneja, 2004; Dong & Zhu, 2016) .

Suppose the Q-values for state i between two different actions a * (i) and j are very close.

Then, one can show that the probability of suboptimal selection between the two has roughly an exponential decay rate controlled by h ij .

Obviously, there can be many more comparisons to consider, but the exponential form dictates that the smallest decay rate dominates the calculation, thus leading to the inner min's in (7).

Criterion like (7) is motivated from the OCBA procedure in simulation optimization (which historically has considered simple mean-value alternatives (Chen & Lee, 2011) ).

Here, we consider the Q-values.

For convenience, we call our procedure Q-OCBA.

Implementing criterion (7) requires two additional considerations.

First, solving (7) needs the model primitives Q, P and σ 2 R that appear in the expression of h ij .

These quantities are unknown a priori, but as we collect data they can be sequentially estimated.

This leads to a multi-stage optimization plus parameter update scheme.

Second, since data are collected through running a Markov chain on the exploration actions, not all allocation w is admissible, i.e., realizable as the stationary distribution of the MDP.

To resolve this latter issue, we will derive a convenient characterization for admissibility.

Call π(·|s) admissible if the Markov Chain with transition probabilityP π , defined for Assumption 3, is positive recurrent, and denote w π as its stationary distribution.

Define the set

The following provides a characterization of the set of admissible π: Lemma 1.

For any admission policy π, w π ∈ W.

For any w ∈ W, π w with π w (a = j|s

In other words, optimizing over the set of admissible policies is equivalent to optimizing over the set of stationary distributions.

The latter is much more tractable thanks to the linear structure of W. In practice, we will use W η = W ∩ {w ≥ η} for some small η > 0 to ensure closedness of the set (our experiments use η = 10 −6 ).

Algorithm 1 describes Q-OCBA.

In our experiments shown next, we simply use two stages, i.e., K = 2.

Finally, we also note that criterion like (7) can be modified according to the decision goal.

For example, if one is interested in obtaining the best estimate of χ * , then it would be more beneficial to consider min w∈Wη σ 2 χ .

We showcase this with additional experiments in the Appendix.

Input: Number of iterations K, length of each batch {B k } 1≤k≤K , initial exploration policy π 0 ; Initialization: k = 0; while k ≤ K do Run π k for B k steps and set k = k + 1;

Algorithm 1: Q-OCBA sequential updating rule for exploration

Note that (7) is equivalent to min w max i∈S max j∈A,j =a * (i) s,a c ij (s, a)/w s,a subject to w ∈ W η , where c ij (s, a)'s are non-negative coefficients.

Based on the closed-form characterization of Σ in Theorem 1, c ij (s, a)'s can be estimated with plug-in estimators using data collected in earlier stages.

We conduct several numerical experiments to support our large-sample results in Sections 3 and demonstrate the performance of Q-OCBA against some benchmark methods.

We use the RiverSwim problem in (Osband et al., 2013) with m s states and two actions at each state: swim left (0) or swim right (1) (see Figure 1 ).

The triplet above each arc represents i) the action, 0 or 1, ii) the transition probability to the next state given the current state and action, iii) the reward under the current state and action.

Note that, in this problem, rewards are given only at the left and right boundary states (where the value of r L will be varied).

We consider the infinite horizon setting with γ = 0.95 and ρ = [1/m s , . . .

, 1/m s ] T .

We first demonstrate the validity of our large-sample results.

We use a policy that swims right with probability 0.8 at each state, i.e. π(1|s) = 0.8.

Tables 1 and 2 show the coverage rates of the constructed 95% CIs, for a small m s = 6 (using Theorem 1 and Corollary 1) and a large m s = 31 (using Theorem 4 in the Appendix) respectively.

The latter case uses a linear interpolation with S 0 = {1, 4, . . .

, 28, 31}. All coverage rates are estimated using 10 3 independent experimental repetitions (the bracketed numbers in the tables show the half-widths of 95% CI for the coverage estimates).

For the Q-values, we report the average coverage rate over all (s, a) pairs.

When the number of observations n is large enough (≥ 3 × 10 4 for exact update and ≥ 10 5 for interpolation), we see highly accurate CI coverages, i.e., close to 95%.

Next we investigate the efficiency of our exploration policy.

We compare Q-OCBA with K = 2 to four benchmark policies: i) -greedy with different values of , ii) random exploration (RE) with different values of π(1|s), iii) UCRL2 (a variant of UCRL) with δ = 0.05 (Jaksch et al., 2010) , iv) PSRL with different posterior updating frequencies (Osband et al., 2013) , i.e., PSRL(x) means PSRL is implemented with x episodes.

We use m s = 6 and vary r L from 1 to 3.

To ensure fairness, we use a two-stage implementation for all policies, with 30% of iterations first dedicated to RE (with π(1|s) = 0.6) as a warm start, i.e., the data are used to estimate the parameters needed for the second stage.

To give enough benefit of the doubt, we notice the probabilities of correct selection for both UCRL2 and PSRL are much worse without the warm start.

Tables 3 and 4 compare the probabilities of obtaining the optimal policy (based on the estimated Q n 's).

For -greedy, RE, and PSRL, we report the results with the parameters that give the best performances in our numerical experiments.

The probability of correct selection is estimated using 10 3 replications of the procedure.

We observe that Q-OCBA substantially outperforms the other methods, both with a small data size (n = 10 3 in Table 3 ) and a larger one (n = 10 4 in Table  4 ).

Generally, these benchmark policies perform worse for larger values of r L .

This is because for small r L , the (s, a) pairs that need to be explored more also tend to have larger Q-values.

However, as r L increase, there is a misalignment between the Q-values and the (s, a) pairs that need more exploration.

The superiority of our Q-OCBA in these experiments come as no surprise to us.

The benchmark methods like UCRL2 and PSRL are designed to minimize regret which involves balancing the exploration-exploitation trade-off.

On the other hand, Q-OCBA focuses on efficient exploration only, i.e., our goal is to minimize the probability of incorrect policy selection, and this is achieved by carefully utilizing the variance information gathered from the first stage that is made possible by our derived asymptotic formulas.

We provide additional numerical results in Appendix B.

In this section, we present additional results on large-sample behaviors for constrained MDPs and also estimations based on approximation value iteration.

We consider the constrained MDP setting for budgeted decision-making (Boutilier & Lu, 2016 ) and more recently safety-critical applications (Achiam et al., 2017; Chow et al., 2017) .

Suppose now we aim to maximize the long-run accumulated discounted reward,

, while at the same time want to ensure that a long-run accumulated discounted cost, denoted as

γ t C(s t , π(s t ))|s 0 = s] which we call the loss function, is constrained by some given value η, i.e., max

We assume data coming in like before and, in addition, that we have observations on the incurred cost at each sample of (s, a).

Call the empirical estimate of the costμ C,n .

We follow our paradigm to solve the empirical counterpart of the problem, namely to find a policyπ *

s are the value functions and loss functions evaluated using the empirical estimatesμ R,n ,μ C,n ,P n .

We focus on the estimation error of the optimal value (instead of the feasibility, which could also be important but not pursued here).

To understand the error, we first utilize an optimality characterization of constrained MDPs.

In general, an optimal policy for (8) is a "split" policy (Feinberg & Rothblum, 2012) , namely, a policy that is deterministic except that at one particular state a randomization between two different actions is allowed.

This characterization can be deduced from the associated LP using occupancy measures (Altman, 1999) .

We call the randomization probability the mixing parameter α * , i.e., whenever this particular state, say s r , is visited, action a * 1 (s r ) is chosen with probability α * and action a * 2 (s r ) is chosen with probability 1 − α * .

We then have the following result:

Theorem 3.

Suppose Assumptions 1 and 3 hold and there is a unique optimal policy.

Moreover, assume that there is no deterministic policy π that satisfies

as n → ∞, where one of the following cases hold:

1.

The optimal policy is deterministic.

We then have Σ = Σ V where Σ V is defined in Theorem 1.

The optimal policy is deterministic, except at one state where a randomization between two actions occurs, with the mixing parameter α * .

Denote the state where the randomization occurs by s r and two possible actions for s r by a * 1 (s r ) and a * 2 (s r ), We have Case 1 in Theorem 3 corresponds to the case where the constraint in (8) is non-binding.

This effectively reduces to the unconstrained scenario in Theorem 1 since a small perturbation of µ R , µ C , P does not affect feasibility.

Case 2 is when the constraint is binding.

In this case, α * must be chosen such that the split policy ensures equality in the constraint, and when µ R , µ C , P is perturbed the estimatedα * n would adjust accordingly.

Thus, in this case the estimation of V * incurs two sources of noises, one from the uncertainty of R, P that appears also in unconstrained problems, and one from the uncertainty in calibratingα * n that is in turn affected by C, P , thus leading to the extra terms in the variance expression.

When the state space S is large, updating an m s × m a look-up table via T µ R ,P (.) can be computationally infeasible.

Approximate value iteration operates by applying a mapping M over T µ R ,P .

In many

where M S0 I is a dimension-reducing "inherit" mapping R msma → R ms 0 ma , and M g is the "generalization" mapping R ms 0 ma → R msma that lifts back to the full dimension.

By selecting a "representative" subset S 0 ⊂ S with cardinality

where [x i ] i∈I denotes the set of entries of x whose index i ∈ I. In this setup, we define Q M as a fixed point of the operator M • T µ R ,P (·), and

We derive large-sample error estimates in this case.

For this, we first assume there is a welldefined metric on S. To guarantee the existence of Q M , we make the following assumption on the generalization map M g : Assumption 4.

M g is a max-norm non-expansion mapping in S, i.e., ||M g (x) − M g (y)|| ∞ < ||x − y|| ∞ ∀x, y ∈ S.

We also need the following analogs of Assumptions 2 and 3 to Q M and S 0 :

Assumption 5.

For any state s ∈ S, arg max a∈A Q M (s, a) is unique.

Assumption 6.

For the Markov Chain with transition probabilityP π , the set of states {(s, a) : s ∈ S 0 , a ∈ A} is in the same communication class and this class is positive recurrent.

Let N 0 = m s0 m a and I S0 = {(i − 1)m a + j : i ∈ S 0 , a ∈ A}. With Assumption 6, we denoteP M S0 as a sub-matrix of the matrixP π that only contains rows with indexes in I S0 .

We also denote S 0 (i) as the i-th element (state) in S 0 .

We defineQ M n as the empirical estimator of Q M built on n observations.

Then we have: Theorem 4.

Under Assumption 4, 5 and 6, if M g is continuously differentiable, then

Assumption 4 is generally satisfied by "local" approximation methods such as linear interpolation, k-nearest neighbors and local weighted average (Gordon, 1995) .

In all these cases, ∇M g in Theorem 4 is actually a constant matrix.

This section reports additional numerical experiments.

Sections B.1 present further results on the estimation quality of Q-values, V * and χ * .

Section B.2 provides additional results to demonstrate the efficiency of our proposed exploration strategy.

In this section, we provide additional numerical results about Tables 1 and 2 in the main paper.

For Q-values in Table 1 of main paper, we only report the average coverage rate over all (s, a) pairs.

T .

We use RE with π(1|s) = 0.5 as our exploration policy.

We see that the behaviors of these individual estimates are largely consistent.

The coverage rates all converge to the nominal 95% as the number of observations n increases.

Moreover, the coverages for the individual Q's, V * 's, and the averages of these quantities are similar at any given n. Specifically, when n = 10 4 , the coverages are all around 77% − 78%, when n = 3 × 10 4 they are all around 93%, and when n = 5 × 10 4 they are all very close to 95%.

These suggest a sample size of 5 × 10 4 (or lower) is enough to elicit our asymptotic results in Theorem 1 and Corollary 1 in this problem.

Tables 6, 7 and 8 compare the CI coverage rates when the state space is large, i.e. m s = 31, using RE with different values of π(1|s), i.e., π(1|s) = 0.8, 0.85, and 0.9.

Compared to exact update, the coverage convergence for approximate update appears generally slower.

Specifically, comparing Tables 5 and 6 that use the same RE with π(1|s) = 0.8, we see that the coverages on the averages of Q's and V * 's for approximate update are only around 23% − 25% when n = 10 4 , whereas they are 77% − 78% for exact update.

Also, while the nominal coverage, i.e., 95%, is obtained when n = 5 × 10 4 in the exact update for all studied quantities, this sample size is not enough for approximate update, where it appears we need n to be of order 10 7 to obtain the nominal coverage.

Furthermore, Tables 6, 7 and 8 show that the rates of convergence to the nominal coverage are quite different for different values of π(1|s)'s.

The convergence rate when π(1|s) = 0.85 seems to be the fastest, with the coverage close to 95% already when n = 10 5 .

On the other hand, when π(1|s) = 0.8, the coverage is close to 95% only when n is as large as 10 7 , and when π(1|s) = 0.9, even n = 10 7 is not enough for convergence to kick in.

We also see that, when the coverage is very far from the nominal rate, discrepancies can show up among the estimates of Q, V * and χ * .

For example, when π(1|s) = 0.8 and n = 10 4 , the coverages of Q and V * are around 23% − 25% but the coverage of χ * is as low as 1%, and when π(1|s) = 0.9 and n = 10 7 , the coverages of Q and V * are around 75% − 77% but that of χ * is only 29%.

However, in settings where the coverage is close to 95%, all these quantities appear to attain this accuracy simultaneously in all considered cases.

These caution that coverage accuracy can be quite sensitive to the specifications of the exploration policy.

Nonetheless, the convergence behaviors predicted by Theorem 1, Corollary 1 and Theorem 4 are all observed to hold.

In Q-OCBA, our second-stage exploration policy is derived by maximizing the worst-case relative discrepancy among all Q-value estimates.

If one is interested in obtaining the best estimate of χ * (i.e., the optimal value function initialized at a distribution ρ), then it would be more beneficial to consider solving min

to derive the optimal second-stage exploration policy π w (recall Lemma 1).

The motivation is that by doing so we would obtain a CI for χ * as short as possible.

Table 9 compares the 95% CI lengths and coverages for this exploration policy with other benchmark strategies, for r L ranging from 1 to 3.

For each r L , we show the averages of the coverages and CI lengths of Q estimates among all (s, a) pairs, and also the coverage and CI length of χ * estimates.

Note that our strategy intends to shorten the CI lengths of χ * estimates.

Like our experiment in Section 5 in the main paper, we use a total observation budget n = 10 4 , and devote 30% to the initial stage where RE with π(1|s) = 0.8 is used to estimate the parameters used to plug in the criterion to be optimized in the second stage.

For convenience and the consistency of terminology, we continue to call our procedure to attain criterion (9) Q-OCBA.

We compare this with pure RE and -greedy, with ranging from 0.01 to 0.2.

Table 9 shows that our budget is enough to achieve the nominal 95% coverages for both the Q-values and χ * using all strategies, which is consistent with the conclusion from Theorem 1 and Corollary 1.

However, Q-OCBA leads to desirably much shorter CI's generally, with the shortest CI lengths in all settings and sometimes by a big margin.

For example, when r L = 2, the CI length derived by Q-OCBA is at least 80% less than those derived by all the other methods.

We also observe that Q-OCBA performs much more stably than RE and -greedy, the latter varying quite significantly for different values of r L .

When r L = 1, -greedy with = 0.01 can perform almost as well as Q-OCBA, with both CI lengths for Q being 2.45 − 2.46 and for χ * being 2.41 − 2.42.

But when r L = 2, -greedy with the same = 0.01 cannot even explore all (s, a) pairs.

The situation worsens when r L = 3, where none of the considered values of can explore all (s, a) pairs.

This observation on -greedy is consistent with Table 3 in the main paper where we consider the criterion using the probability of correct selection.

Regardless of using that or the current criteria, the performances of -greedy depend fundamentally on whether the (s, a) pairs that need to be explored more also tend to have larger Q-values.

Note that when changing r L , the corresponding changes in the Q-values would change the exploration "preference" for -greedy.

However, as the underlying stochasticity of the system does not change with r L , the states that need more exploration remain unchanged.

This misalignment leads us to observe quite different performances for -greedy when r L varies.

Lastly, again consistent with the results on the probability of correct selection shown in Table 3 of the main paper, we observe that Q-OCBA outperforms pure RE in all cases in Table 5 , with at least 40% shorter CI lengths for the χ * estimates.

This is attributed to the efficient use of variance information in the second stage of Q-OCBA.

a NA means that some (s, a) pair has never been visited.

In this section, we present the proofs of the main results.

In the proofs, we shall treat P as an N m s -dimensional vector following the index rule:

By Assumption 2, there exists an open neighborhood of Q, which we denote as Ω, such that ∀Q ∈ Ω, arg max j Q ((i − 1)m a + j) is still unique for each

has all its partial derivatives exist and continuous.

This implies that

Denote the partial derivatives of F as ∂F ∂(Q , r , P ) = ∂F ∂Q ∂F ∂r ∂F ∂P .

∂Q is an N ×N matrix.

Denote its element at the ((i−1)m a +j)-th row, ((k −1)m a +l)-th column by

.

Then we have

Putting all the elements together, we have

whereP is an N × N matrix with

Since all rows ofP sum up to one,P can be interpreted as the transition matrix of a Markov Chain with state space {(i, j) :

∂Q is invertible for any Q ∈ Ω. We can then apply the implicit function theorem to the equation F (Q, µ R , P ) = 0.

In particular, there exists an open set U around µ R × P ∈ R N × R N ms , and a unique continuously differentiable function φ: U → R N , such that for any r × P ∈ U φ(µ R , P ) = Q F (φ(r , P ), r , P ) = 0.

In addition, the partial derivatives of φ satisfy

It is also easy to verify that ∂F ∂r Q =Q,r =µ R ,P =P = I N ×N .

We also note that

which is an N × N m s matrix.

Next, for

by Assumption 1, 3 and Slutsky's theorem, we have

where

. . .

. . .

which is an N m s × N m s matrix.

By the continuous mapping theorem, we have φ(μ R,n ,P n ) − φ(µ R , P ) → 0 a.s.

as n → ∞, which impliesQ n → Q a.s..

In addition, using the delta method, we have

We also have

Proof of Corollary 1.

which is continuously differentiable in an open neighborhood of Q. Then we can apply the delta method to get

rearranging the index such thatP

(where " * " denotes a placeholder of some quan-

Thus,

Lastly, the asymptotic normality ofχ π * n follows from the continuous mapping theorem.

We next establish the asymptotic normality for the estimated value function under a given policyπ.

In this case, the value function Vπ satisfies a Bellman equation

Denote the the estimator of Vπ byVπ n .

In particular,Vπ n is the fixed point of the corresponding empirical Bellman equation that replaces (µ R , P ) by (μ R,n ,P n ).

We have:

Corollary 2.

Under Assumptions 1 and 3,

where

, Pπ is an m s × m s transition matrix with Pπ(i, j) = a P (j|s = i, a)π(a|s = i), W is an m s × m s diagonal matrix with

Proof of Corollary 2.

Similar to the proof of Theorem 1, define Fπ as a mapping from

Note that Fπ(Vπ, µ R , P ) = 0, Fπ is continuously differentiable and I − γPπ is invertible.

We can thus apply the implicit function theorem.

In particular, there exists an open set Uπ around µ R × P ∈ R N × R N ms , and a unique continuously differentiable function φπ: Uπ → R N , such that

Fπ(φπ(r , P ), r , P ) = 0 for any r × P ∈ Uπ.

For the partial derivatives of φπ, we have ∂φπ ∂(r , P ) r =µ R ,P =P = − ∂Fπ ∂V

and

, which is an N -dimensional row vector.

Applying the delta method, we have

T and the conclusion follows.

Proof of Theorem 2.

Write the MDP problem in its LP representation

The decision variables in the dual problem, x s,a 's, in particular represent the occupancy measures of the MDP.

If the MDP has non-unique optimal policies, the dual problem also has non-unique optimal solutions, which implies a degeneration of the primal problem.

Degeneration here means that some constraints are redundant at the primal optimal solution (i.e., the corner-point solution is at the intersection of more than m s hyperplanes).

Since the rows of the primal LP are linearly independent, we know that in this case, there are multiple (K > 1) choices for the set of basic variables (v B k ) 1≤k≤K at the optimal solution.

When the coefficients in the intersecting hyperlanes perturb slightly along a given direction, the objective value will change by a perturbation of the objective coefficients along a chosen set of basic variables v B k .

In other words, we can partition the set of directions U into subsets {U k } 1≤k≤K such that, if the direction of perturbation of (P, µ R ), say u, lies in U k , then the LP optimal value perturbs by fixing the basic variables as v B k .

Denote G ρ k as the gradient vector corresponding to this direction.

If some of the G ρ k 's are equal, we merge the corresponding U k 's into one partition set.

Thus, we have

Note that the argument so far focuses on the LP with objective value s ρ(s)V (s).

However, we can repeat the same argument for each V (s) by setting ρ(s) = e s .

For any u ∈ U and s ∈ S, denote the directional gradient of V (s) with respect to P, µ R by D u V (P, µ R )(s), thus we can define the directional Jacobian of V with respect to P, µ R as

Multiply √ n on both sides and notice that

is a continuous mapping of ( √ n(P n − P ), √ n(μ R,n − µ R )).

By taking n → ∞, we get the result by the continuous mapping theorem.

Proof of Theorem 3.

We use the LP representation of the constrained MDP.

Define x s,a as the occupancy measure

where P ρ denotes the distribution of S t 's with initial distribution ρ.

Then, x s,a satisfies the LP

(This is the dual formulation in the proof of Theorem 2 with an extra constraint.)

The objective and the first constraint correspond to the objective and constraint in the constrained MDP formulation.

The second constraint can be deduced by a one-step analysis on the definition of occupancy measure.

Once (11) is solved to obtain an optimal solution (x * s,a ) s,a , it can be translated into an optimal policy

Note that the number of structural constraints in (11) is m s + 1, and a corner-point optimal solution has m s + 1 basic variables.

Moreover, by our assumptions, the optimal solution is unique and the LP is non-degenerate, so that perturbing the parameters µ R (s, a), µ C (s, a), P (s|s , a) does not immediately imply an overshoot to negativity for the reduced costs of the non-basic variables.

Now consider two cases depending on whether the first constraint is non-binding or binding.

The first case corresponds to a deterministic optimal policy, i.e., for any s, x s,a > 0 for only one a. In this case a small perturbation of the parameters still retains the same basic and non-basic variables, and the derived perturbed policy still retains the non-binding first constraint.

In this case, the analysis reduces back to Corollary 2.

In the second case, x s,a > 0 for only one a, for all s except one state s r , where we can have x sr,a * 1 (sr) > 0 and x sr,a * 2 (sr) > 0 for two distinct actions a * 1 (s r ), a * 2 (s r ).

Again, perturbing the parameters retains these basic and non-basic variables.

In particular, the first constraint remains binding in the perturbation, so that the perturbed optimal policy π * is still split at the same state and satisfies s ρ(s)L π * (s) = η.

Now denote the mixing parameter by α * := π * (a * 1 (s r )|s r ), and so π * (a * 2 (s r )|s r ) = 1 − α * .

By applying the implicit function theorem to the Bellman equation, there exists a continuously differentiable function φ L such that L π * (s) = φ L (µ C , P, α * ).

By applying the implicit function theorem again to the equation s ρ(s)φ L (µ C , P, α * ) = η, we know α * is a continuously differentiable function of µ C and P .

Thus V * can be viewed as a function of µ R , P and α * , the latter in turn depending on µ C and P .

It can also be viewed as a function of µ R , µ C and P directly.

We use ∇ µ R ,µ C ,P V * (µ R , µ C , P ) to denote the Jacobian of V * with respect to µ R , µ C , P when viewing V * as a function of these variables.

We also use ∇ µ R ,µ C ,P,α V * (µ R , P, α * ) to denote the Jacobian of V * with respect to µ R , µ C , P, α * , this time viewing V * as a function of µ R , P, α * .

We use similar notations throughout, and their meanings should be clear from the context.

To facilitate derivations, we also distinguish the notation ∇ x f , which denotes the multi-dimensional Jacobian matrix, from ∂ x f , which is used to denote the Jacobian when x is a scalar (1-dimensional).

∇ µ R ,µ C ,P V * (µ R , µ C , P ) = ∇ µ R ,µ C ,P,α * V * (µ R , P, α * )[I, ∇ µ R ,µ C ,P α * (µ C , P ) T ] T = ∇ µ R ,µ C ,P V * (µ R , P, α * ) + ∂ α * V * (µ R , P, α * )∇ µ R ,µ C ,P α * (µ C , P ) (12) Differentiating ρ T L π * (µ R , µ C , P ) = η, we have

(µ C , P, α * )∇ µ R ,µ C ,P α * (µ C , P ) = 0.

By rearranging the equation, we have

Substituting this into (12), we get

Next, define an m s -dimensional vector r C by r C (s) = ma j=1 µ C (s, j)π * (j|s).

Then The derivation of ∇ µ R ,µ C ,P V * (µ R , P, α * ) and ∇ µ R ,µ C ,P L π * (µ C , P, α * ) follows exactly the same line of analysis as how we derive Gπ and Hπ V in the proof of Corollary 2.

Proof of Theorem 4.

Denote .

Changing the distribution of [μ R,n ,P n ] S\S0 will not change the distribution ofQ M n .

We can thus assign auxiliary random variables toμ R,n andP n for all i / ∈ S 0 , 1 ≤ j ≤ m a , 1 ≤ k ≤ m s .

In particular, we define independent random variables for each i / ∈ S 0 by lettinĝ for all i ∈ S. Thus, w is the stationary distribution of transition matrixP πw .

<|TLDR|>

@highlight

We investigate the large-sample behaviors of the Q-value estimates and proposed an efficient exploration strategy that relies on estimating the relative discrepancies among the Q estimates. 
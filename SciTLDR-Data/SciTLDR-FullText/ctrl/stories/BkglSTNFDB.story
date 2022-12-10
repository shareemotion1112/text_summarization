A fundamental question in reinforcement learning is whether model-free algorithms are sample efficient.

Recently,  Jin et al. (2018) proposed a Q-learning algorithm with UCB exploration policy, and proved it has nearly optimal regret bound for finite-horizon episodic MDP.

In this paper, we adapt Q-learning with UCB-exploration bonus to infinite-horizon MDP with discounted rewards \emph{without} accessing a generative model.

We show that the \textit{sample complexity of exploration} of our algorithm is bounded by $\tilde{O}({\frac{SA}{\epsilon^2(1-\gamma)^7}})$.

This improves the previously best known result of $\tilde{O}({\frac{SA}{\epsilon^4(1-\gamma)^8}})$ in this setting achieved by delayed Q-learning (Strehlet al., 2006),, and matches the lower bound in terms of $\epsilon$ as well as $S$ and $A$ up to logarithmic factors.

The goal of reinforcement learning (RL) is to construct efficient algorithms that learn and plan in sequential decision making tasks when the underlying system dynamics are unknown.

A typical model in RL is Markov Decision Process (MDP).

At each time step, the environment is in a state s.

The agent takes an action a, obtain a reward r, and then the environment transits to another state.

In reinforcement learning, the transition probability distribution is unknown.

The algorithm needs to learn the transition dynamics of MDP, while aiming to maximize the cumulative reward.

This poses the exploration-exploitation dilemma: whether to act to gain new information (explore) or to act consistently with past experience to maximize reward (exploit).

Theoretical analyses of reinforcement learning fall into two broad categories: those assuming a simulator (a.k.a.

generative model), and those without a simulator.

In the first category, the algorithm is allowed to query the outcome of any state action pair from an oracle.

The emphasis is on the number of calls needed to estimate the Q value or to output a near-optimal policy.

There has been extensive research in literature following this line of research, the majority of which focuses on discounted infinite horizon MDPs (Azar et al., 2011; Even-Dar & Mansour, 2003; Sidford et al., 2018b) .

The current results have achieved near-optimal time and sample complexities (Sidford et al., 2018b; a) .

Without a simulator, there is a dichotomy between finite-horizon and infinite-horizon settings.

In finite-horizon settings, there are straightforward definitions for both regret and sample complexity; the latter is defined as the number of samples needed before the policy becomes near optimal.

In this setting, extensive research in the past decade (Jin et al., 2018; Azar et al., 2017; Jaksch et al., 2010; Dann et al., 2017) has achieved great progress, and established nearly-tight bounds for both regret and sample complexity.

The infinite-horizon setting is a very different matter.

First of all, the performance measure cannot be a straightforward extension of the sample complexity defined above (See Strehl & Littman (2008) for detailed discussion).

Instead, the measure of sample efficiency we adopt is the so-called sample complexity of exploration (Kakade et al., 2003) , which is also a widely-accepted definition.

This measure counts the number of times that the algorithm "makes mistakes" along the whole trajectory.

See also (Strehl & Littman, 2008) for further discussions regarding this issue.

Several model based algorithms have been proposed for infinite horizon MDP, for example Rmax (Brafman & Tennenholtz, 2003) , MoRmax (Szita & Szepesvári, 2010) and UCRL-γ (Lattimore & Hutter, 2012) .

It is noteworthy that there still exists a considerable gap between the state-of-the-art algorithm and the theoretical lower bound (Lattimore & Hutter, 2012) regarding 1/(1 − γ) factor.

Though model-based algorithms have been proved to be sample efficient in various MDP settings, most state-of-the-art RL algorithms are developed in the model-free paradigm (Schulman et al., 2015; Mnih et al., 2013; 2016) .

Model-free algorithms are more flexible and require less space, which have achieved remarkable performance on benchmarks such as Atari games and simulated robot control problems.

For infinite horizon MDPs without access to simulator, the best model-free algorithm has a sample complexity of explorationÕ( SA 4 (1−γ) 8 ), achieved by delayed Q-learning (Strehl et al., 2006) .

The authors provide a novel strategy of argument when proving the upper bound for the sample complexity of exploration, namely identifying a sufficient condition for optimality, and then bound the number of times that this condition is violated.

However, the results of Delayed Q-learning still leave a quadratic gap in 1/ from the best-known lower bound.

This is partly because the updates in Q-value are made in an over-conservative way.

In fact, the loose sample complexity bound is a result of delayed Q-learning algorithm itself, as well as the mathematical artifact in their analysis.

To illustrate this, we construct a hard instance showing that Delayed Q-learning incurs Ω(1/ 3 ) sample complexity.

This observation, as well as the success of the Q-learning with UCB algorithm (Jin et al., 2018) in proving a regret bound in finite-horizon settings, motivates us to incorporate a UCB-like exploration term into our algorithm.

In this work, we propose a Q-learning algorithm with UCB exploration policy.

We show the sample complexity of exploration bound of our algorithm isÕ(

.

This strictly improves the previous best known result due to Delayed Q-learning.

It also matches the lower bound in the dependence on , S and A up to logarithmic factors.

We point out here that the infinite-horizon setting cannot be solved by reducing to finite-horizon setting.

There are key technical differences between these two settings: the definition of sample complexity of exploration, time-invariant policies and the error propagation structure in Q-learning.

In particular, the analysis techniques developed in (Jin et al., 2018) do not directly apply here.

We refer the readers to Section 3.2 for detailed explanations and a concrete example.

The rest of the paper is organized as follows.

After introducing the notation used in the paper in Section 2, we describe our infinite Q-learning with UCB algorithm in Section 3.

We then state our main theoretical results, which are in the form of PAC sample complexity bounds.

In Section 4 we present some interesting properties beyond sample complexity bound.

Finally, we conclude the paper in Section 5.

We consider a Markov Decision Process defined by a five tuple S, A, p, r, γ , where S is the state space, A is the action space, p(s |s, a) is the transition function, r : S ×A → [0, 1] is the deterministic reward function, and 0 ≤ γ < 1 is the discount factor for rewards.

Let S = |S| and A = |A| denote the number of states and the number of actions respectively.

Starting from a state s 1 , the agent interacts with the environment for infinite number of time steps.

At each time step, the agent observes state s t ∈ S, picks action a t ∈ A, and receives reward r t ; the system then transits to next state s t+1 .

Using the notations in Strehl et al. (2006) , a policy π t refers to the non-stationary control policy of the algorithm since step t. We use V πt (s) to denote the value function under policy π t , which is defined as

We also use V * (s) = sup π V π (s) to denote the value function of the optimal policy.

Accordingly, we define

as the Q function under policy π t ; Q * (s, a) is the Q function under optimal policy π * .

We use the sample complexity of exploration defined in Kakade et al. (2003) to measure the learning efficiency of our algorithm.

This sample complexity definition has been widely used in previous works Strehl et al. (2006) ; Lattimore & Hutter (2012); Strehl & Littman (2008) .

Definition 1.

Sample complexity of Exploration of an algorithm ALG is defined as the number of time steps t such that the non-stationary policy π t at time t is not -optimal for current state s t , i.e.

Roughly speaking, this measure counts the number of mistakes along the whole trajectory.

We use the following definition of PAC-MDP Strehl et al. (2006) .

Definition 2.

An algorithm ALG is said to be PAC-MDP (Probably Approximately Correct in Markov Decision Processes) if, for any and δ, the sample complexity of ALG is less than some polynomial in the relevant quantities (S, A, 1/ , 1/δ, 1/(1 − γ)), with probability at least 1 − δ.

Finally, recall that Bellman equation is defined as the following:

which is frequently used in our analysis.

Here we denote

In this section, we present the UCB Q-learning algorithm and the sample complexity bound.

Algorithm 1 Infinite Q-learning with UCB Parameters: , γ, δ

H+k .

for t = 1, 2, ... do Take action a t ← arg max a Q (s t , a ) Receive reward r t and transit to s t+1

c 2 is a constant and can be set to 4 √ 2

Our UCB Q-learning algorithm (Algorithm 1) maintains an optimistic estimation of action value function Q(s, a) and its historical minimum valueQ(s, a).

N t (s, a) denotes the number of times that (s, a) is experienced before time step t; τ (s, a, k) denotes the time step t at which (s t , a t ) = (s, a) for the k-th time; if this state-action pair is not visited that many times, τ (s, a, k) = ∞. Q t (s, a) andQ t (s, a) denotes the Q andQ value of (s, a) that the algorithm maintains when arriving at s t respectively.

Our main result is the following sample complexity of exploration bound.

Theorem 1.

For any > 0, δ > 0, 1/2 <

γ < 1, with probability 1 − δ, the sample complexity of exploration (i.e., the number of time steps t such that π t is not -optimal at s t ) of Algorithm 1 is at

whereÕ suppresses logarithmic factors of 1/ , 1/(1 − γ) and SA.

We first point out the obstacles for proving the theorem and reasons why the techniques in Jin et al. (2018) do not directly apply here.

We then give a high level description of the ideas of our approach.

One important issue is caused by the difference in the definition of sample complexity for finite and infinite horizon MDP.

In finite horizon settings, sample complexity (and regret) is determined in the first T timesteps, and only measures the performance at the initial state s 1 (i.e. (V * − V π )(s 1 )).

However, in the infinite horizon setting, the agent may enter under-explored regions at any time period, and sample complexity of exploration characterizes the performance at all states the agent enters.

The following example clearly illustrates the key difference between infinite-horizon and finitehorizon.

Consider an MDP with a starting state s 1 where the probability of leaving s 1 is o(T −1 ).

In this case, with high probability, it would take more than T timesteps to leave s 1 .

Hence, guarantees about the learning in the first T timesteps or about the performance at s 1 imply almost nothing about the number of mistakes the algorithm would make in the rest of the MDP (i.e. the sample complexity of exploration of the algorithm).

As a result, the analysis for finite horizon MDPs cannot be directly applied to infinite horizon setting.

This calls for techniques for counting mistakes along the entire trajectory, such as those employed by Strehl et al. (2006) .

In particular, we need to establish convenient sufficient conditions for being -optimal at timestep t and state s t , i.e. V * (s t ) − V πt (s t ) ≤ .

Then, bounding the number of violations of such conditions gives a bound on sample complexity.

Another technical reason why the proof in Jin et al. (2018) cannot be directly applied to our problem is the following: In finite horizon settings, Jin et al. (2018) decomposed the learning error at episode k and time h as errors from a set of consecutive episodes before k at time h + 1 using a clever design of learning rate.

However, in the infinite horizon setting, this property does not hold.

Suppose at time t the agent is at state s t and takes action a t .

Then the learning error at t only depends on those previous time steps such that the agent encountered the same state as s t and took the same action as a t .

Thus the learning error at time t cannot be decomposed as errors from a set of consecutive time steps before t, but errors from a set of non-consecutive time steps without any structure.

Therefore, we have to control the sum of learning errors over an unstructured set of time steps.

This makes the analysis more challenging.

Now we give a brief road map of the proof of Theorem 1.

Our first goal is to establish a sufficient condition so that π t learned at step t is -optimal for state s t .

As an intermediate step we show that a sufficient condition for

is small for a few time steps t within an interval [t, t + R] for a carefully chosen R (Condition 1).

Then we show the desired sufficient condition (Condition 2) implies Condition 1.

We then bound the total number of bad time steps on which V * (s t ) − Q * (s t , a t ) is large for the whole MDP; this implies a bound on the number of violations of Condition 2.

This in turn relies on a key technical lemma (Lemma 2).

The remaining part of this section is organized as follows.

We establish the sufficient condition for -optimality in Section 3.3.

The key lemma is presented in Section 3.4.

Finally we prove Theorem 1 in Section 3.5.

In this section, we establish a sufficient condition (Condition 2) for -optimality at time step t. For a fixed s t , let TRAJ(R) be the set of length-R trajectories starting from s t .

Our goal is to give a sufficient condition so that π t , the policy learned at step t, is -optimal.

For any 2 > 0, define R := ln

. . .

where the last inequality holds because

, which follows from the definition of R.

For any fixed trajectory of length R starting from s t , consider the sequence (∆ t ) t≤t <t+R .

Let X (i) t be the i-th largest item of (∆ t ) t≤t <t+R .

Rearranging Eq. (1), we obtain

We first prove that Condition 1 implies -optimality at time step t when 2 = /3.

(3) Claim 1.

If Condition 1 is satisfied at time step t, the policy π t is -optimal at state s t , i.e.

is monotonically decreasing with respect to i.

Therefore, E[X

where the last inequality follows from the fact that

Combining with Eq. 2, we have,

Next we show that given i, t, Condition 2 implies Eq. (3).

Claim 2.

Given i, t, Eq. (3) holds if Condition 2 is satisfied.

Proof.

The reason behind the choice of M is to ensure that η M > 1/(1 − γ)

1 .

It follows that, assuming Condition 2 holds, for

Therefore, if a time step t is not 2 -optimal, there exists 0 ≤ i < log 2 R and 2 ≤ j ≤ M such that

Now, the sample complexity can be bounded by the number of (t, i, j) pairs that Eq. (4)

In this section, we present two key lemmas.

Lemma 1 bounds the number of sub-optimal actions, which in turn, bounds the sample complexity of our algorithm.

Lemma 2 bounds the weighted sum of learning error, i.e. (Q t − Q * )(s, a), with the sum and maximum of weights.

Then, we show that Lemma 1 follows from Lemma 2.

Lemma 1.

For fixed t and η > 0, let B (t)

where I[·] is the indicator function.

Before presenting Lemma 2, we define a class of sequence that occurs in the proof.

Definition 3.

A sequence (w t ) t≥1 is said to be a (C, w)-sequence for C, w > 0, if 0 ≤ w t ≤ w for all t ≥ 1, and t≥1 w t ≤ C.

Lemma 2.

For every (C, w)-sequence (w t ) t≥1 , with probability 1 − δ/2, the following holds:

where (C) = ι(C) ln

is a log-factor.

Proof of Lemma 2 is quite technical, and is therefore deferred to supplementary materials.

Now, we briefly explain how to prove Lemma 1 with Lemma 2.

(Full proof can be found in supplementary materials.)

Note that sinceQ t ≥ Q * and a t = arg max aQt (s t , a),

We now consider a set J = {t :

−1 }, and consider the (|J|, 1)-weight sequence defined by w t = I [t ∈ J].

We can now apply Lemma 2 to weighted sum

.

On the one hand, this quantity is obviously at least |J|η(1 − γ) −1 .

On the other hand, by lemma 2, it is upper bounded by the weighted sum of (Q − Q * )(s t , a t ).

Thus we get

Now focus on the dependence on |J|.

The left-hand-side has linear dependence on |J|, whereas the left-hand-side has aÕ |J| dependence.

This allows us to solve out an upper bound on |J| with quadratic dependence on 1/η.

We prove the theorem by stitching Lemma 1 and Condition 2.

By lemma 1, for any 2 ≤ j ≤ M ,

HereP is a shorthand for polylog

] be a Bernoulli random variable, and {F t } t≥1 be the filtration generated by random variables {(s τ , a τ ) : 1 ≤ τ ≤ t}. Since A t is F t+R −measurable, for any 0 ≤ k < R, {A k+tR − E[A k+tR | F k+tR ]} t≥0 is a martingale difference sequence.

For now, consider a fixed 0 ≤ k < R. By Azuma-Hoeffiding inequality, after

ξi ln(RM L) time steps (if it happens that many times) with

we have t A k+tR ≥ C/2 i with probability at least 1 − δ/(2M RL).

On the other hand, if A k+tR happens, within [k + tR, k + tR + R − 1], there must be at least 2 i time steps at which V * (s t ) − Q * (s t , a t ) > η j−1 .

The latter event happens at most C times, and [k + tR, k + tR + R − 1] are disjoint.

Therefore,

i .

This suggests that the event described by (7) happens at most T times for fixed i and j. Via a union bound on 0 ≤ k < R, we can show that with probability 1 − δ/(2M L), there are at most RT time steps where

Thus, the number of sub-optimal steps is bounded by,

It should be stressed that throughout the lines,P is a shorthand for an asymptotic expression, instead of an exact value.

Our final choice of 2 and 1 are 2 = 3 , and 1 = 24RM ln

.

It is not hard to see that ln 1/ 1 = poly(ln 1 , ln 1 1−γ ).

This immediately implies that with probability 1 − δ, the number of time steps such that (

where hidden factors are poly(ln 1 , ln 1 1−γ , ln SA).

In this section, we discuss the implication of our results, and present some interesting properties of our algorithm beyond its sample complexity bound.

Lower bound To the best of our knowledge, the current best lower bound for worst-case sample complexity is Ω SA 2 (1−γ) 3 ln 1/δ due to Lattimore & Hutter (2012) .

The gap between our results and this lower bound lies only in the dependence on 1/(1−γ) and logarithmic terms of SA, 1/(1−γ) and 1/ .

Model-free algorithms Previously, the best sample complexity bound for a model-free algorithm isÕ SA 4 (1−γ) 8 (suppressing all logarithmic terms), achieved by Delayed Q-learning Strehl et al. (2006) .

Our results improve this upper bound by a factor of 1 2 (1−γ) , and closes the quadratic gap in 1/ between Delayed Q-learning's result and the lower bound.

In fact, the following theorem shows that UCB Q-learning can indeed outperform Delayed Q-learning.

Theorem 2.

There exists a family of MDPs with constant S and A, in which with probability 1 − δ, Delayed Q-learning incurs sample complexity of exploration of Ω −3 ln(1/δ) , assuming that ln(1/δ) < −2 .

The construction of this hard MDP family is given in the supplementary material.

Model-based algorithms For model-based algorithms, better sample complexity results in infinite horizon settings have been claimed Szita & Szepesvári (2010) .

To the best of our knowledge, the best published result without further restrictions on MDPs isÕ

Due to length limits, detailed discussion in this section is deferred to supplementary materials.

Finite horizon MDP The sample complexity of exploration bounds of UCB Q-learning implies O −2 PAC sample complexity and aÕ T 1/2 regret bound in finite horizon MDPs.

That is, our algorithm implies a PAC algorithm for finite horizon MDPs.

We are not aware of reductions of the opposite direction (from finite horizon sample complexity to infinite horizon sample complexity of exploration).

Regret The reason why our results can imply anÕ( √ T )

regret is that, after choosing 1 , it follows from the argument of Theorem 1 that with probability 1 − δ, for all 2 >Õ( 1 /(1 − γ)), the number of 2 -suboptimal steps is bounded by

In contrast, Delayed Q-learning Strehl et al. (2006) can only give an upper bound on 1 -suboptimal steps after setting parameter 1 .

Infinite-horizon MDP with discounted reward is a setting that is arguably more difficult than other popular settings, such as finite-horizon MDP.

Previously, the best sample complexity bound achieved by model-free reinforcement learning algorithms in this setting isÕ( -learning Strehl et al. (2006) .

In this paper, we propose a variant of Q-learning that incorporates upper confidence bound, and show that it has a sample complexity ofÕ(

.

This matches the best lower bound except in dependence on 1/(1 − γ) and logarithmic factors.

A PROOF OF LEMMA 1 Lemma 1.

For fixed t and η > 0, let B (t) η be the event that V * (s t ) − Q * (s t , a t ) > η 1−γ in step t. If η > 2 1 , then with probability at least 1 − δ/2,

where I[·] is the indicator function.

Proof.

When η > 1 the lemma holds trivially.

Now consider the case that η ≤ 1.

By lemma 2, with probability 1 − δ,

Suppose that |I| = SAk 2 η 2 (1−γ) 3 ln SA, for some k > 1.

Then it follows that for some constant

If k ≥ 10C ln C , then

which means violation of (9).

Therefore, since C ≥ 2

, 20 ln 2}.

It immediately follows that

B PROOF OF LEMMA 2

Lemma 2.

For every (C, w)-sequence (w t ) t≥1 , with probability 1 − δ/2, the following holds:

where (C) = ι(C) ln

is a log-factor.

(2) For any p, there exists p ≤ p such that

Proof.

Both properties are results of the update rule at line 11 of Algorithm 1.

Before proving lemma 2, we will prove two auxiliary lemmas.

Lemma 3.

The following properties hold for α i t :

t where ι(t) = ln(c(t+1)(t+2)), for every t ≥ 1, c ≥ 1.

Proof.

Recall that

(1 − α j ).

Properties 1-3 are proven by Jin et al. (2018) .

Now we prove the last property.

On the one hand,

where the last inequality follows from property 1.

The left-hand side is proven by induction on t. For the base case, when t = 1, α

Since function f (t) = ι(t)/t is monotonically decreasing for t ≥ 1, c ≥ 1, we have

Lemma 4.

With probability at least 1 − δ/2, for all p ≥ 0 and (s, a)-pair,

where t = N p (s, a), t i = τ (s, a, i) and β t = c 3 Hι(t)/((1 − γ) 2 t).

Proof.

Recall that

(1 − α j ).

From the update rule, it can be seen that our algorithm maintains the following Q(s, a):

Bellman optimality equation gives:

Subtracting the two equations gives

The identity above holds for arbitrary p, s and a. Now fix s ∈ S, a ∈ A and p ∈ N. Let t = N p (s, a), t i = τ (s, a, i).

The t = 0 case is trivial; we assume t ≥ 1 below.

Now consider an arbitrary fixed k. Define

Let F i be the σ-Field generated by random variables (s 1 , a 1 , ..., s ti , a ti ).

It can be seen that

1−γ .

Therefore, ∆ i is a martingale difference sequence; by the Azuma-Hoeffding inequality,

By choosing η, we can show that with probability 1 − δ/ [SA(k + 1)(k + 2)],

Here

.

By a union bound for all k, this holds for arbitrary k > 0, arbitrary s ∈ S, a ∈ A simultaneously with probability

Therefore, we conclude that (16) holds for the random variable t = N p (s, a) and for all p, with probability 1 − δ/2 as well.

Proof of the right hand side of (13): We also know that (

It is implied by (16) that

(Property 4 of lemma 3)

Proof of the left hand side of (13): Now, we assume that event that (16) holds.

We assert that Q p ≥ Q * for all (s, a) and p ≤ p .

This assertion is obviously true when p = 0.

Then

Therefore the assertion holds for p + 1 as well.

By induction, it holds for all p.

We now see that (13) holds for probability 1 − δ/2 for all p, s, a. SinceQ p (s, a) is always greater than Q p (s, a) for some p ≤ p, we know thatQ p (s, a) ≥ Q p (s, a) ≥ Q * (s, a), thus proving (14).

We now give a proof for lemma 2.

Recall the definition for a (C, w)-sequence.

A sequence (w t ) t≥1 is said to be a (C, w)-sequence for C, w > 0, if 0 ≤ w t ≤ w for all t ≥ 1, and t≥1 w t ≤ C.

Proof.

Let n t = N t (s t , a t ) for simplicity; we have

The last inequality is due to lemma 4.

Note that α 0 nt = I[n t = 0], the first term in the summation can be bounded by,

For the second term, define u(s, a) = sup t N t (s, a).

2 It follows that,

Where C s,a = t≥1,(st,at)=(s,a) w t .

Inequality (19) follows from rearrangement inequality, since ι(x)/x is monotonically decreasing.

Inequality (21) follows from Jensen's inequality.

For the third term of the summation, we have

We claim that w t+1 is a (C, (1 + 1 H )w)-sequence.

We now prove this claim.

By lemma 3, for any t ≥ 0,

First we define a mapping from a finite horizon MDP to an infinite horizon MDP so that our algorithm can be applied.

For an arbitrary finite horizon MDP M = (S, A, H, r h (s, a), p h (s | s, a)) where H is the length of episode, the corresponding infinite horizon MDPM = (S,Ā, γ,r(s,ā),p(s |s,ā)) is defined as, For any > 0, by running our algorithm onM forÕ( ) time steps, the starting state s 1 is visited at leastÕ( ) times, and at most 1/3 of them are not -optimal.

If we select the policy uniformly randomly from the policy π tH+1 for 0 ≤ t < T /H, with probability at least 2/3 we can get an -optimal policy.

Therefore the PAC sample complexity isÕ −2 after hiding S, A, H terms.

On the other hand, we want to show that for any K episodes,

The reason why our algorithm can have a better reduction from regret to PAC is that, after choosing 1 , it follows from the argument of theorem 1 that for all 2 >Õ( 1 /(1 − γ)), the number of 2 -suboptimal steps is bounded by .

It follows that,

SA ln 1/δ 2 · 2 i−2 ≤Õ √ SAT ln 1/δ with probability 1 − δ.

Note that theÕ notation hides the poly (1/(1 − γ), log 1/ 1 ) which is, by our reduction, poly (H, log T, log S, log A).

Recall that our MDP mapping from M = (S, A, H, r h (s, a), p h (s | s, a)) toM = (S,Ā, γ,r(s,ā),p(s |s,ā)) is defined as,

In this section, we prove Theorem 2 regarding the performance of Delayed Q-learning.

Theorem 2.

There exists a family of MDPs with constant S and A, in which with probability 1 − δ, Delayed Q-learning incurs sample complexity of exploration of Ω −3

ln(1/δ) , assuming that ln(1/δ) < −2 .

Proof.

For each 0 < < 1 10 , consider the following MDP (see also Fig. 1 ): state space is S = {a, b, c} while action set is A = {x, y}; transition probabilities are P (b|a, y) = 1 − 10 , P (c|a, y) = 10 , P (b|a, x) = 1, P (a|b, ·) = P (a|c, ·) = 1.

Rewards are all 1, except R(c, ·) = 0.

<|TLDR|>

@highlight

We adapt Q-learning with UCB-exploration bonus to infinite-horizon MDP with discounted rewards without accessing a generative model, and improves the previously best known result.

@highlight

This paper considered a Q-learning algorithm with UCB exploration policy for infinite-horizon MDP.
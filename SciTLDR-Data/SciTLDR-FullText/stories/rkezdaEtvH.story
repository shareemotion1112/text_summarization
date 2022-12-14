Reinforcement learning (RL) typically defines a discount factor as part of the Markov Decision Process.

The discount factor values future rewards by an exponential scheme that leads to theoretical convergence guarantees of the Bellman equation.

However, evidence from psychology, economics and neuroscience suggests that humans and animals instead have hyperbolic time-preferences.

Here we extend earlier work of Kurth-Nelson and Redish and propose an efficient deep reinforcement learning agent that acts via hyperbolic discounting and other non-exponential discount mechanisms.

We demonstrate that a simple approach approximates hyperbolic discount functions while still using familiar temporal-difference learning techniques in RL.

Additionally, and independent of hyperbolic discounting, we make a surprising discovery that simultaneously learning value functions over multiple time-horizons is an effective auxiliary task which often improves over state-of-the-art methods.

The standard treatment of the reinforcement learning (RL) problem is the Markov Decision Process (MDP) which includes a discount factor 0 ≤ γ ≤ 1 that exponentially reduces the present value of future rewards (Bellman, 1957; Sutton & Barto, 1998) .

A reward r t received in t-time steps is devalued to γ t r t , a discounted utility model introduced by Samuelson (1937) .

This establishes a timepreference for rewards realized sooner rather than later.

The decision to exponentially discount future rewards by γ leads to value functions that satisfy theoretical convergence properties (Bertsekas, 1995) .

The magnitude of γ also plays a role in stabilizing learning dynamics of RL algorithms (Prokhorov & Wunsch, 1997; Bertsekas & Tsitsiklis, 1996) and has recently been treated as a hyperparameter of the optimization (OpenAI, 2018; Xu et al., 2018) .

However, both the magnitude and the functional form of this discounting function establish priors over the solutions learned.

The magnitude of γ chosen establishes an effective horizon for the agent of 1/(1 − γ), far beyond which rewards are neglected (Kearns & Singh, 2002) .

This effectively imposes a time-scale of the environment, which may not be accurate.

Further, the exponential discounting of future rewards is consistent with a prior belief that there is a known constant per-time-step hazard rate (Sozou, 1998) or probability of dying of 1 − γ (Lattimore & Hutter, 2011).

Additionally, discounting future values exponentially and according to a single discount factor γ does not harmonize with the measured value preferences in humans 1 and animals (Mazur, 1985; Ainslie, 1992; Green & Myerson, 2004; Maia, 2009) .

A wealth of empirical evidence has been amassed that humans, monkeys, rats and pigeons instead discount future returns hyperbolically, where d k (t) = 1 1+kt , for some positive k > 0 (Ainslie, 1975; 1992; Mazur, 1985; Frederick et al., 2002; Green et al., 1981; Green & Myerson, 2004) .

This discrepancy between the time-preferences of animals from the exponential discounted measure of value might be presumed irrational.

But Sozou (1998) showed that hyperbolic time-preferences is mathematically consistent with the agent maintaining some uncertainty over the prior belief of the hazard rate in the environment.

Hazard rate h(t) measures the per-time-step risk the agent incurs as it acts in the environment due to a potential early death.

Precisely, if s(t) is the probability that the agent is alive at time t then the hazard rate is h(t) = − d dt lns(t).

We consider the case where there is a fixed, but potentially unknown hazard rate h(t) = λ ≥ 0.

The prior belief of the hazard rate p(λ) implies a specific discount function Sozou (1998) .

Under this formalism, the canonical case in RL of discounting future rewards according to d(t) = γ t is consistent with the belief that there exists a single hazard rate λ = e −γ known with certainty.

Further details are available in Appendix A. Figure 1: Hyperbolic versus exponential discounting.

Humans and animals often exhibit hyperbolic discounts (blue curve) which have shallower discount declines for large horizons.

In contrast, RL agents often optimize exponential discounts (orange curve) which drop at a constant rate regardless of how distant the return.

Common RL environments are also characterized by risk, but often in a narrower sense.

In deterministic environments like the original Arcade Learning Environment (ALE) (Bellemare et al., 2013) stochasticity is often introduced through techniques like no-ops (Mnih et al., 2015) and sticky actions (Machado et al., 2018) where the action execution is noisy.

Physics simulators may have noise and the randomness of the policy itself induces risk.

But even with these stochastic injections the risk to reward emerges in a more restricted sense.

In Section 2 we show that a prior distribution reflecting the uncertainty over the hazard rate, has an associated discount function in the sense that an MDP with either this hazard distribution or the discount function, has the same value function for all policies.

This equivalence implies that learning policies with a discount function can be interpreted as making them robust to the associated hazard distribution.

Thus, discounting serves as a tool to ensure that policies deployed in the real world perform well even under risks they were not trained under.

We propose an algorithm that approximates hyperbolic discounting while building on successful Qlearning (Watkins & Dayan, 1992) tools and their associated theoretical guarantees.

We show learning many Q-values, each discounting exponentially with a different discount factor γ, can be aggregated to approximate hyperbolic (and other non-exponential) discount factors.

We demonstrate the efficacy of our approximation scheme in our proposed Pathworld environment which is characterized both by an uncertain per-time-step risk to the agent.

Conceptually, Pathworld emulates a foraging environment where an agent must balance easily realizable, small meals versus more distant, fruitful meals.

We then consider higher-dimensional deep RL agents in the ALE, where we measure the benefits of hyperbolic discounting.

This approximation mirrors the work of Kurth-Nelson & Redish (2009); Redish & Kurth-Nelson (2010) which empirically demonstrates that modeling a finite set of µAgents simultaneously can approximate hyperbolic discounting function.

Our method then generalizes to other non-hyperbolic discount functions and uses deep neural networks to model the different Q-values from a shared representation.

Surprisingly and in addition to enabling new non-exponential discounting schemes, we observe that learning a set of Q-values is beneficial as an auxiliary task (Jaderberg et al., 2016) .

Adding this multi-horizon auxiliary task often improves over a state-of-the-art baseline, Rainbow (Hessel et al., 2018) in the ALE (Bellemare et al., 2013) .

This work questions the RL paradigm of learning policies through a single discount function which exponentially discounts future rewards through the following contributions:

1.

Hazardous MDPs.

We formulate MDPs with hazard present and demonstrate an equivalence between undiscounted values learned under hazards and (potentially nonexponentially) discounted values without hazard.

2.

Hyperbolic (and other non-exponential)-agent.

A practical approach for training an agent which discounts future rewards by a hyperbolic (or other non-exponential) discount function and acts according to this.

3.

Multi-horizon auxiliary task.

A demonstration of multi-horizon learning over many γ simultaneously as an effective auxiliary task.

To study MDPs with hazard distributions and general discount functions we introduce two modifications.

The hazardous MDP now is defined by the tuple < S, A, R, P, H, d >.

In standard form, the state space S and the action space A may be discrete or continuous.

The learner observes samples from the environment transition probability P (s t+1 |s t , a t ) for going from s t ∈ S to s t+1 ∈ S given a t ∈ A.

We will consider the case where P is a sub-stochastic transition function, which defines an episodic MDP.

The environment emits a bounded reward r : S × A → [r min , r max ] on each transition.

In this work we consider non-infinite episodic MDPs.

The first difference is that at the beginning of each episode, a hazard λ ∈ [0, ∞) is sampled from the hazard distribution H. This is equivalent to sampling a continuing probability γ = e −λ .

During the episode, the hazard modified transition function will be P λ , in that P λ (s |s, a) = e −λ P (s |s, a).

The second difference is that we now consider a general discount function d(t).

This differs from the standard approach of exponential discounting in RL with γ according to d(t) = γ t , which is a special case.

This setting makes a close connection to partially observable Markov Decision Process (POMDP) (Kaelbling et al., 1998) where one might consider λ as an unobserved variable.

However, the classic POMDP definition contains an explicit discount function γ as part of its definition which does not appear here.

A policy π :

S → A is a mapping from states to actions.

The state action value function Q

is the expected discounted rewards after taking action a in state s and then following policy π until termination.

where λ ∼ H and E π,P λ implies that s t+1 ∼ P λ (·|s t , a t ) and a t ∼ π(·|s t ).

In the hazardous MDP setting we observe the same connections between hazard and discount functions delineated in Appendix A. This expresses an equivalence between the value function of an MDP with a discount and MDP with a hazard distribution.

For example, there exists an equivalence between the exponential discount function d(t) = γ t to the undiscounted case where the agent is subject to a (1 − γ) per time-step of dying (Lattimore & Hutter, 2011) .

The typical Q-value (left side of Equation 2) is when the agent acts in an environment without hazard λ = 0 or H = δ(0) and discounts future rewards according to d(t) = γ t = e −λt which we denote as Q

The alternative Q-value (right side of Equation 2) is when the agent acts under hazard rate λ = − ln γ but does not discount future rewards which we denote as Q

where δ(x) denotes the Dirac delta distribution at x. This follows from P λ (s |s, a) = e −λ P (s |s, a)

We also show a similar equivalence between hyperbolic discounting and the specific hazard distribu-

For notational brevity later in the paper, we will omit the explicit hazard distribution H-superscript if the environment is not hazardous.

This formulation builds upon Sozou (1998)'s relate of hazard rate and discount functions and shows that this holds for generalized Q-values in reinforcement learning.

We now show how one can re-purpose exponentially-discounted Q-values to compute hyperbolic (and other-non-exponential) discounted Q-values.

The central challenge with using non-exponential discount strategies is that most RL algorithms use some form of TD learning (Sutton, 1988) .

This family of algorithms exploits the Bellman equation (Bellman, 1958) which, when using exponential discounting, relates the value function at one state with the value at the following state.

where expectation E π,P denotes sampling a ∼ π(·|s), s ∼ P (·|s, a), and a ∼ π(·|s ).

Being able to reuse TD methods without being constrained to exponential discounting is thus an important challenge.

We propose here a scheme to deduce hyperbolic as well as other non-exponentially discounted Q-values when our discount function has a particular form.

which we will refer to as the exponential weighting condition, then

Proof.

Applying the condition on d,

The exchange in the above proof is valid if

The exponential weighting condition is satisfied for hyperbolic discounting and other discounting that we might want to consider (see Appendix F for examples).

As an example, the hyperbolic discount can be expressed as the integral of a function f (γ, t) for γ = [0, 1) in Equation 9.

This equationn tells us an integral over a function f (γ, t) = 1 k γ 1/k+t−1 = w(γ)γ t yields the desired hyperbolic discount factor Γ k (t) = This prescription gives us a tool to produce general forms of non-exponentially discounted Q-values using our familiar exponentially discounted Q-values traditionally learned in RL (Sutton, 1988; Sutton & Barto, 1998) .

Section 3 describes an equivalence between hyperbolically-discounted Q-values and integrals of exponentially-discounted Q-values, however, the method required evaluating an infinite set of value functions.

We therefore present a practical approach to approximate discounting Γ(t) = 1 1+kt using a finite set of functions learned via standard Q-learning (Watkins & Dayan, 1992) .

To avoid estimating an infinite number of Q γ π -values we introduce a free hyperparameter (n γ ) which is the total number of Q γ π -values to consider, each with their own γ.

We use a practically-minded approach to choose G that emphasizes evaluating larger values of γ rather than uniformly choosing points and empirically performs well as seen in Section 5.

Our approach is described in Appendix G. Each Q γi π computes the discounted sum of returns according to that specific discount factor

We previously proposed two equivalent approaches for computing hyperbolic Q-values, but for simplicity we consider the one presented in Lemma 3.1.

The set of Q-values permits us to estimate the integral through a Riemann sum (Equation 11) which is described in further detail in Appendix I.

where we estimate the integral through a lower bound.

We consolidate this entire process in Figure 11 where we show the full process of rewriting the hyperbolic discount rate, hyperbolically-discounted Q-value, the approximation and the instantiated agent.

This approach is similar to that of KurthNelson & Redish (2009) where each µAgent models a specific discount factor γ.

However, this differs in that our final agent computes a weighted average over each Q-value rather than a sampling operation of each agent based on a γ-distribution.

The benefits of hyperbolic discounting will be greatest under two conditions: uncertain hazard and non-trivial intertemporal decisions.

The first condition can arise under a unobserved hazard-rate variable λ drawn independently at the beginning of each episode from H = p(λ).

The second condition emerges with a choice between a smaller nearby rewards versus larger distant rewards.

In the absence of both properties we would not expect any advantage to discounting hyperbolically.

To see why, if there is a single-true hazard rate λ env , than an optimal γ * = e −λenv exists and future rewards should be discounted exponentially according to it.

Further, if there is a single path through the environment with perfect alignment of short-and long-term objectives, all discounting schemes yield the same optimal policy.

We note two sources for discounting rewards in the future: time delay and survival probability (Section 2).

In Pathworld we train to maximize hyperbolically discounted returns ( t Γ k (t)R(s t , a t )) under no hazard (H = δ(λ − 0)) but then evaluate the undiscounted returns d(t) = 1.0 ∀ t with the paths subject to hazard H = 1 k exp(−λ/k).

Through this procedure, we are able to train an agent that is robust to hazards in the environment.

The agent makes one decision in Pathworld (Figure 2 ): which of the N paths to investigate.

Once a path is chosen, the agent continues until it reaches the end or until it dies.

This is similar to a multi-armed bandit, with each action subject to dynamic risk.

The paths vary quadratically in length with the index d(i) = i 2 but the rewards increase linearly with the path index r(i) = i. This presents ...

Figure 2: The Pathworld.

Each state (white circle) indicates the accompanying reward r and the distance from the starting state d. From the start state, the agent makes a single action: which which path to follow to the end.

Longer paths have a larger rewards at the end, but the agent incurs a higher risk on a longer path.

a non-trivial decision for the agent.

At deployment, an unobserved hazard λ ∼ H is drawn and the agent is subject to a per-time-step risk of dying of (1 − e −λ ).

This environment differs from the adjusting-delay procedure presented by Mazur (1987) and then later modified by Kurth-Nelson & Redish (2009) .

Rather then determining time-preferences through variable-timing of rewards, we determine time-preferences through risk to the reward.

Figure 3: In each episode of Pathworld an unobserved hazard λ ∼ p(λ) is drawn and the agent is subject to a total risk of the reward not being realized of Figure 3 showing that our approximation scheme well-approximates the true valueprofile.

Figure 3 validates that our approach well-approximates the true hyperbolic value of each path when the hazard prior matches the true distribution.

Agents that discount exponentially according to a single γ (the typical case in RL) incorrectly value the paths.

We examine further the failure of exponential discounting in this hazardous setting.

For this environment, the true hazard parameter in the prior was k = 0.05 (i.e. λ ∼ 20exp(−λ/0.05)).

Therefore, at deployment, the agent must deal with dynamic levels of risk and faces a non-trivial decision of which path to follow.

Even if we tune an agent's γ = 0.975 such that it chooses the correct arg-max path, it still fails to capture the functional form ( Figure 3 ) and it achieves a high error over all paths (Table 1) .

If the arg-max action was not available or if the agent was proposed to evaluate non-trivial intertemporal decisions, it would act sub-optimally.

In Appendix B we consider additional experiments where the agent's prior over hazard more realistically does not exactly match the environment true hazard rate and demonstrate the benefit of appropriate priors.

With our approach validated in Pathworld, we now move to the high-dimensional environment of Atari 2600, specifically, ALE.

We use the Rainbow variant from Dopamine (Castro et al., 2018) which implements three of the six considered improvements from the original paper: distributional RL, predicting n-step returns and prioritized replay buffers.

The agent (Figure 4 ) maintains a shared representation h(s) of state, but computes Q-value logits for each of the N γ i via Q

π (s, a) = W i h(s) + b i where W i and b i are the learnable parameters of the affine transformation for that head.

A ReLU-nonlinearity is used within the body of the network (Nair & Hinton, 2010) .

: Multi-horizon model predicts Q-values for n γ separate discount functions thereby modeling different effective horizons.

Each Q-value is a lightweight computation, an affine transformation off a shared representation.

By modeling over multiple time-horizons, we now have the option to construct policies that act according to a particular value or a weighted combination.

Hyperparameter details are provided in Appendix K and when applicable, they default to the standard Dopamine values.

We find strong performance improvements of the hyperbolic agent built on Rainbow (Hyper-Rainbow; blue bars) on a random subset of Atari 2600 games in Figure 5 .

To dissect the Hyper-Rainbow improvements, recognize that two properties from the base Rainbow agent have changed:

1.

Behavior policy, µ. The agent acts according to hyperbolic Q-values computed by our approximation described in Section 4 2.

Learn over multiple horizons.

The agent simultaneously learns Q-values over many γ rather than a Q-value for a single γ

On this subset of 19 games, Hyper-Rainbow improves upon 14 games and in some cases, by large margins.

But we seek here a more complete understanding of the underlying driver of this improvement in ALE through an ablation study.

The second modification can be regarded as introducing an auxiliary task (Jaderberg et al., 2016) .

Therefore, to attribute the performance of each properly we construct a Rainbow agent augmented with the multi-horizon auxiliary task (referred to as Multi-Rainbow and shown in orange) but have it still act according to the original policy.

That is, Multi-Rainbow acts to maximize expected rewards discounted by a fixed γ action but now learns over multiple horizons as shown in Figure 4 .

Figure 5 : We compare the Hyper-Rainbow (in blue) agent versus the Multi-Rainbow (orange) agent on a random subset of 19 games from ALE (3 seeds each).

For each game, the percentage performance improvement for each algorithm against Rainbow is recorded.

There is no significant difference whether the agent acts according to hyperbolically-discounted (Hyper-Rainbow) or exponentiallydiscounted (Multi-Rainbow) Q-values suggesting the performance improvement in ALE emerges from the multi-horizon auxiliary task.

We find that the Multi-Rainbow agent performs nearly as well on these games, suggesting the effectiveness of this as a stand-alone auxiliary task.

This is not entirely unexpected given the rather special-case of hazard exhibited in ALE through sticky-actions (Machado et al., 2018) .

We examine further and investigate the performance of this auxiliary task across the full Arcade Learning Environment (Bellemare et al., 2017) using the recommended evaluation by (Machado et al., 2018) .

Doing so we find strong empirical benefits of the multi-horizon auxiliary task over the state-of-the-art Rainbow agent as shown in Figure 6 .

Game Name Auxiliary Task Improvement for Rainbow Agent Figure 6 : Performance improvement over Rainbow using the multi-horizon auxiliary task in Atari Learning Environment (3 seeds each).

To understand the interplay of the multi-horizon auxiliary task with other improvements in deep RL, we test a random subset of 10 Atari 2600 games against improvements in Rainbow (Hessel et al., 2018) .

On this set of games we measure a consistent improvement with multi-horizon C51 (Multi-C51) in 9 out of the 10 games over the base C51 agent in Figure 7 .

Figure 7 indicates that the current implementation of Multi-Rainbow does not generally build successfully on the prioritized replay buffer.

On the subset of ten games considered, we find that four out of ten games (Pong, Venture, Gravitar and Zaxxon) are negatively impacted despite (Hessel et al., 2018) finding it to be of considerable benefit and specifically beneficial in three out of these

Hyperbolic discounting in economics.

Hyperbolic discounting is well-studied in the field of economics (Sozou, 1998; Dasgupta & Maskin, 2005) .

Dasgupta and Maskin (2005) proposes a softer interpretation than Sozou (1998) (which produces a per-time-step of death via the hazard rate) and demonstrates that uncertainty over the timing of rewards can also give rise to hyperbolic discounting and preference reversals, a hallmark of hyperbolic discounting.

Hyperbolic discounting was initially presumed to not lend itself to TD-based solutions (Daw & Touretzky, 2000) but the field has evolved on this point.

Maia (2009) proposes solution directions that find models that discount quasi-hyperbolically even though each learns with exponential discounting (Loewenstein, 1996) but reaffirms the difficulty.

Finally, Alexander and Brown (2010) proposes hyperbolically discounted temporal difference (HDTD) learning by making connections to hazard.

Behavior RL and hyperbolic discounting in neuroscience.

TD-learning has long been used for modeling behavioral reinforcement learning (Montague et al., 1996; Schultz et al., 1997; Sutton & Barto, 1998) .

TD-learning computes the error as the difference between the expected value and actual value (Sutton & Barto, 1998; Daw, 2003) where the error signal emerges from unexpected rewards.

However, these computations traditionally rely on exponential discounting as part of the estimate of the value which disagrees with empirical evidence in humans and animals (Strotz, 1955; Mazur, 1985; Ainslie, 1975; 1992) .

Hyperbolic discounting has been proposed as an alternative to exponential discounting though it has been debated as an accurate model (Kacelnik, 1997; Frederick et al., 2002) .

Naive modifications to TD-learning to discount hyperbolically present issues since the (2009) demonstrated that distributed exponential discount factors can directly model hyperbolic discounting.

This work proposes the µAgent, an agent that models the value function with a specific discount factor γ.

When the distributed set of µAgent's votes on the action, this was shown to approximate hyperbolic discounting well in the adjusting-delay assay experiments (Mazur, 1987) .

Using the hazard formulation established in Sozou (1998), we demonstrate how to extend this to other non-hyperbolic discount functions and demonstrate the efficacy of using a deep neural network to model the different Q-values from a shared representation.

Towards more flexible discounting in reinforcement learning.

RL researchers have recently adopted more flexible versions beyond a fixed discount factor (Feinberg & Shwartz, 1994; Sutton, 1995; Sutton et al., 2011; White, 2017) .

Optimal policies are studied in Feinberg & Shwartz (1994) where two value functions with different discount factors are used.

Introducing the discount factor as an argument to be queried for a set of timescales is considered in both Horde (Sutton et al., 2011) and γ-nets (Sherstan et al., 2018) .

Reinke et al. (2017) proposes the Average Reward Independent Gamma Ensemble framework which imitates the average return estimator.

Lattimore and Hutter (2011) generalizes the original discounting model through discount functions that vary with the age of the agent, expressing time-inconsistent preferences as in hyperbolic discounting.

The need to increase training stability via effective horizon was addressed in François-Lavet, Fonteneau, and Ernst (2015) who proposed dynamic strategies for the discount factor γ.

Meta-learning approaches to deal with the discount factor have been proposed in Xu, van Hasselt, and Silver (2018) .

Finally, Pitis (2019) characterizes rational decision making in sequential processes, formalizing a process that admits a state-action dependent discount rates.

Operating over multiple time scales has a long history in RL.

Sutton (1995) generalizes the work of Singh (1992) and Dayan and Hinton (1993) to formalize a multi-time scale TD learning model theory.

Previous work has been explored on solving MDPs with multiple reward functions and multiple discount factors though these relied on separate transition models (Feinberg & Shwartz, 1999; Dolgov & Durfee, 2005) .

Edwards, Littman, and Isbell (2015) considers decomposing a reward function into separate components each with its own discount factor.

In our work, we continue to model the same rewards, but now model the value over different horizons.

Recent work in difficult exploration games demonstrates the efficacy of two different discount factors (Burda et al., 2018) one for intrinsic rewards and one for extrinsic rewards.

Finally, and concurrent with this work, Romoff et al. (2019) proposes the TD(∆)-algorithm which breaks a value function into a series of value functions with smaller discount factors.

Auxiliary tasks in reinforcement learning.

Finally, auxiliary tasks have been successfully employed and found to be of considerable benefit in RL.

Suddarth and Kergosien (1990) used auxiliary tasks to facilitate representation learning.

Building upon this, work in RL has consistently demonstrated benefits of auxiliary tasks to augment the low-information coming from the environment through extrinsic rewards (Lample & Chaplot, 2017; Mirowski et al., 2016; Jaderberg et al., 2016; Veeriah et al., 2018; Sutton et al., 2011) 8 DISCUSSION AND FUTURE WORK This work builds on a body of work that questions one of the basic premises of RL: one should maximize the exponentially discounted returns via a single discount factor.

By learning over multiple horizons simultaneously, we have broadened the scope of our learning algorithms.

Through this we have shown that we can enable acting according to new discounting schemes and that learning multiple horizons is a powerful stand-alone auxiliary task.

Our method well-approximates hyperbolic discounting and performs better in hazardous MDP distributions.

This may be viewed as part of an algorithmic toolkit to model alternative discount functions.

However, this work still does not fully capture more general aspects of risk since the hazard rate may be a function of time.

Further, hazard may not be an intrinsic property of the environment but a joint property of both the policy and the environment.

If an agent purses a policy leading to dangerous state distributions then it will naturally be subject to higher hazards and vice-versathis creates a complicated circular dependency.

We would therefore expect an interplay between time-preferences and policy.

This is not simple to deal with but recent work proposing state-action dependent discounting (Pitis, 2019) Sozou (1998) formalizes time preferences in which future rewards are discounted based on the probability that the agent will not survive to collect them due to an encountered risk or hazard.

Definition A.1.

Survival s(t) is the probability of the agent surviving until time t.

s(t) = P (agent is alive|at time t)

A future reward r t is less valuable presently if the agent is unlikely to survive to collect it.

If the agent is risk-neutral, the present value of a future reward r t received at time-t should be discounted by the probability that the agent will survive until time t to collect it, s(t).

Consequently, if the agent is certain to survive, s(t) = 1, then the reward is not discounted per Equation 14.

From this it is then convenient to define the hazard rate.

Definition A.2.

Hazard rate h(t) is the negative rate of change of the log-survival at time t

or equivalently expressed as h(t) = − ds(t) dt 1 s(t) .

Therefore the environment is considered hazardous at time t if the log survival is decreasing sharply.

Sozou (1998) demonstrates that the prior belief of the risk in the environment implies a specific discounting function.

When the risk occurs at a known constant rate than the agent should discount future rewards exponentially.

However, when the agent holds uncertainty over the hazard rate then hyperbolic and alternative discounting rates arise.

We recover the familiar exponential discount function in RL based on a prior assumption that the environment has a known constant hazard.

Consider a known hazard rate of h(t) = λ ≥ 0.

Definition A.2 sets a first order differential equation

.

The solution for the survival rate is s(t) = e −λt which can be related to the RL discount factor γ

This interprets γ as the per-time-step probability of the episode continuing.

This also allows us to connect the hazard rate λ ∈ [0, ∞] to the discount factor γ ∈ [0, 1).

As the hazard increases λ → ∞, then the corresponding discount factor becomes increasingly myopic γ → 0.

Conversely, as the environment hazard vanishes, λ → 0, the corresponding agent becomes increasingly far-sighted γ → 1.

In RL we commonly choose a single γ which is consistent with the prior belief that there exists a known constant hazard rate λ = −ln(γ).

We now relax the assumption that the agent holds this strong prior that it exactly knows the true hazard rate.

From a Bayesian perspective, a looser prior allows for some uncertainty in the underlying hazard rate of the environment which we will see in the following section.

We may not always be so confident of the true risk in the environment and instead reflect this underlying uncertainty in the hazard rate through a hazard prior p(λ).

Our survival rate is then computed by weighting specific exponential survival rates defined by a given λ over our prior p(λ)

Sozou (1998) shows that under an exponential prior of hazard p(λ) = 1 k exp(−λ/k) the expected survival rate for the agent is hyperbolic

We denote the hyperbolic discount by Γ k (t) to make the connection to γ in reinforcement learning explicit.

Further, Sozou (1998) shows that different priors over hazard correspond to different discount functions.

We reproduce two figures in Figure 8 showing the correspondence between different hazard rate priors and the resultant discount functions.

The common approach in RL is to maintain a delta-hazard (black line) which leads to exponential discounting of future rewards.

Different priors lead to non-exponential discount functions.

There is a correspondence between hazard rate priors and the resulting discount function.

In RL, we typically discount future rewards exponentially which is consistent with a Dirac delta prior (black line) on the hazard rate indicating no uncertainty of hazard rate.

However, this is a special case and priors with uncertainty over the hazard rate imply new discount functions.

All priors have the same mean hazard rate E[p(λ)] = 1.

In Figure 9 we consider the case that the agent still holds an exponential prior but has the wrong coefficient k and in Figure 10 we consider the case where the agent still holds an exponential prior but the true hazard is actually drawn from a uniform distribution with the same mean.

Through these two validating experiments, we demonstrate the robustness of estimating hyperbolic discounted Q-values in the case when the environment presents dynamic levels of risk and the agent faces non-trivial decisions.

Hyperbolic discounting is preferable to exponential discounting even when the agent's prior does not precisely match the true environment hazard rate distribution, by coefficient ( Figure 9 ) or by functional form (Figure 10 ). .

Predictably, the mismatched priors result in a higher prediction error of value but performs more reliably than exponential discounting, resulting in a cumulative lower error.

Numerical results in Table 2 .

Table 2 : The average mean squared error (MSE) over each of the paths in Figure 9 .

As the prior is further away from the true value of k = 0.05, the error increases.

However, notice that the errors for large factor-of-2 changes in k result in generally lower errors than if the agent had considered only a single exponential discount factor γ as in Table 1 .

Figure 10: If the true hazard rate is now drawn according to a uniform distribution (with the same mean as before) the original hyperbolic discount matches the functional form better than exponential discounting.

Numerical results in Table 3 .

hyperbolic value 0.235 γ = 0.975 0.266 γ = 0.95 0.470 γ = 0.99 4.029 Table 3 : The average mean squared error (MSE) over each of the paths in Figure 10 when the underlying hazard is drawn according to a uniform distribution.

We find that hyperbolic discounting results is more robust to hazards drawn from a uniform distribution than exponential discounting.

Let's start with the case where we would like to estimate the value function where rewards are discounted hyperbolically instead of the common exponential scheme.

We refer to the hyperbolic Q-values as Q Γ π below in Equation 21

We may relate the hyperbolic Q Γ π -value to the values learned through standard Q-learning.

To do so, notice that the hyperbolic discount Γ t can be expressed as the integral of a certain function f (γ, t) for γ = [0, 1) in Equation 22.

The integral over this specific function f (γ, t) = γ kt yields the desired hyperbolic discount factor Γ k (t) by considering an infinite set of exponential discount factors γ over its domain γ ∈ [0, 1).

Recognize that the integrand γ kt is the standard exponential discount factor which suggests a connection to standard Q-learning (Watkins & Dayan, 1992) .

This suggests that if we could consider an infinite set of γ then we can combine them to yield hyperbolic discounts for the corresponding time-step t. We build on this idea of modeling many γ throughout this work.

where Γ k (t) has been replaced on the first line by as a weighting over exponentiallydiscounted Qvalues using the same weights :

can be expressed as a weighting over exponential discount functions with weights (see Table 1 ).

3.

The integral in box 2 can be approximated with a Riemann sum over the discrete intervals:

Following Section A we also show a similar equivalence between hyperbolic discounting and the specific hazard distribution

Where the first step uses Equation 19.

This equivalence implies that discount factors can be used to learn policies that are robust to hazards.

We expand upon three special cases to see how functions f (γ, t) = w(γ)γ t may be related to different discount functions d(t).

We summarize in Table 4 how a particular hazard prior p(λ) can be computed via integrating over specific weightings w(γ) and the corresponding discount function.

Table 4 : Different hazard priors H = p(λ) can be alternatively expressed through weighting exponential discount functions γ t by w(γ).

This table matches different hazard distributions to their associated discounting function and the weighting function per Lemma 3.1.

The typical case in RL is a Dirac Delta Prior over hazard rate δ(λ − k).

We only show this in detail for completeness; one would not follow such a convoluted path to arrive back at an exponential discount but this approach holds for richer priors.

The derivations can be found in the Appendix F.

Three cases:

For the three cases we begin with the Laplace transform on the prior p(λ) = ∞ λ=0 p(λ)e −λt dλ and then chnage the variables according to the relation between γ = e −λ , Equation 17.

A delta prior p(λ) = δ(λ − k) on the hazard rate is consistent with exponential discounting.

where δ(λ − k) is a Dirac delta function defined over variable λ with value k. The change of variable γ = e −λ (equivalently λ = − ln γ) yields differentials dλ = − 1 γ dγ and the limits λ = 0 → γ = 1 and λ = ∞ → γ = 0.

Additionally, the hazard rate value λ = k is equivalent to the γ = e −k .

where we define a γ k = e −k to make the connection to standard RL discounting explicit.

Additionally and reiterating, the use of a single discount factor, in this case γ k , is equivalent to the prior that a single hazard exists in the environment.

Again, the change of variable γ = e −λ yields differentials dλ = − 1 γ dγ and the limits λ = 0 → γ = 1 and λ = ∞ → γ = 0.

where p(·) is the prior.

With the exponential prior p(λ) = 1 k exp(−λ/k) and by substituting λ = −lnγ we verify Equation 9

Finally if we hold a uniform prior over hazard, Sozou (1998) shows the Laplace transform yields

Use the same change of variables to relate this to γ.

The bounds of the integral become λ = 0 → γ = 1 and λ = k → γ = e −k .

which recovers the discounting scheme.

We provide further detail for which γ we choose to model and motivation why.

We choose a γ max which is the largest γ to learn through Bellman updates.

If we are using k as the hyperbolic coefficient in Equation 19 and we are approximating the integral with n γ our γ max would be

However, allowing γ max → 1 get arbitrarily close to 1 may result in learning instabilities Bertsekas (1995).

Therefore we compute an exponentiation base of b = exp(ln(1 − γ 1/k max )/n γ ) which bounds our γ max at a known stable value.

This induces an approximation error which is described more in Appendix H.

Instead of evaluating the upper bound of Equation 9 at 1 we evaluate at γ max which yields γ kt max /(1+kt).

Our approximation induces an error in the approximation of the hyperbolic discount.

This approximation error in the Riemann sum increases as the γ max decreases as evidenced by Figure  12 .

When the maximum value of γ max → 1 then the approximation becomes more accurate as supported in Table 5 up to small random errors.

As discussed, we can estimate the hyperbolic discount in two different ways.

We illustrate the resulting estimates here and resulting approximations.

We use lower-bound Riemann sums in both cases for simplicity but more sophisticated integral estimates exist.

As noted earlier, we considered two different integrals for computed the hyperbolic coefficients.

Under the form derived by the Laplace transform, the integrals are sharply peaked as γ → 1.

The difference in integrals is visually apparent comparing in Figure 13 .

Figure 12: By instead evaluating our integral up to γ max rather than to 1, we induce an approximation error which increases with t. Numerical results in Table 5 .

Figure 12 .

(a) We approximate the integral of the function γ kt via a lower estimate of rectangles at specific γ-values.

The sum of these rectangles approximates the hyperbolic discounting scheme 1/(1 + kt) for time t. (b) Alternative form for approximating hyperbolic coefficients which is sharply peaked as γ → 1 which led to larger errors in estimation under our initial techniques.

J PERFORMANCE OF DIFFERENT REPLAY BUFFER PRIORITIZATION SCHEME As found through our ablation study in Figure 7 , the Multi-Rainbow auxiliary task interacted poorly with the prioritized replay buffer when the TD-errors were averaged evenly across all heads.

As an alternative scheme, we considered prioritizing according to the largest γ, which is also the γ defining the Q-values by which the agent acts.

The (preliminary 5 ) results of this new prioritization scheme is in Figure 14 .

-10 3 -10 2 -10 1 -10 0 0 10 0 10 1 10 2 10 3 Percent Improvement (log) Game Name Multi-Rainbow Improvement over Rainbow (prioritize-largest) Figure 14 : The (preliminary) performance improvement over Rainbow using the multi-horizon auxiliary task in Atari Learning Environment when we instead prioritize according to the TD-errors computed from the largest γ (3 seeds each).

To this point, there is evidence that prioritizing according to the TD-errors generated by the largest gamma is a better strategy than averaging.

Final results of the multi-horizon auxiliary task on Rainbow (Multi-Rainbow) in Table 7 .

@highlight

A deep RL agent that learns hyperbolic (and other non-exponential) Q-values and a new multi-horizon auxiliary task.
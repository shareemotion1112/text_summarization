We investigate the combination of actor-critic reinforcement learning algorithms with uniform large-scale experience replay and propose solutions for two challenges: (a) efficient actor-critic learning with experience replay (b) stability of very off-policy learning.

We employ those insights to accelerate hyper-parameter sweeps in which all participating agents run concurrently and share their experience via a common replay module.



To this end we analyze the bias-variance tradeoffs in V-trace, a form of importance sampling for actor-critic methods.

Based on our analysis, we then argue for mixing experience sampled from replay with on-policy experience, and propose a new trust region scheme that scales effectively to data distributions where V-trace becomes unstable.



We provide extensive empirical validation of the proposed solution.

We further show the benefits of this setup by demonstrating state-of-the-art data efficiency on Atari among agents trained up until 200M environment frames.

Value-based and actor-critic policy gradient methods are the two leading techniques of constructing general and scalable reinforcement learning agents (Sutton et al., 2018) .

Both have been combined with non-linear function approximation (Tesauro, 1995; Williams, 1992) , and have achieved remarkable successes on multiple challenging domains; yet, these algorithms still require large amounts of data to determine good policies for any new environment.

To improve data efficiency, experience replay agents store experience in a memory buffer (replay) (Lin, 1992) , and reuse it multiple times to perform reinforcement learning updates (Riedmiller, 2005) .

Experience replay allows to generalize prioritized sweeping (Moore & Atkeson, 1993) to the non-tabular setting (Schaul et al., 2015) , and can also be used to simplify exploration by including expert (e.g., human) trajectories (Hester et al., 2017) .

Overall, experience replay can be very effective at reducing the number of interactions with the environment otherwise required by deep reinforcement learning algorithms (Schaul et al., 2015) .

Replay is often combined with the value-based Q-learning (Mnih et al., 2015) , as it is an off-policy algorithm by construction, and can perform well even if the sampling distribution from replay is not aligned with the latest agent's policy.

Combining experience replay with actor-critic algorithms can be harder due to their on-policy nature.

Hence, most established actor-critic algorithms with replay such as (Wang et al., 2017; Gruslys et al., 2018; Haarnoja et al., 2018) employ and maintain Q-functions to learn from the replayed off-policy experience.

In this paper, we demonstrate that off-policy actor-critic learning with experience replay can be achieved without surrogate Q-function approximators using V-trace by employing the following approaches: a) off-policy replay experience needs to be mixed with a proportion of on-policy experience.

We show experimentally ( Figure 2 ) and theoretically that the V-trace policy gradient is otherwise not guaranteed to converge to a locally optimal solution.

b) a trust region scheme (Conn et al., 2000; Schulman et al., 2015; can mitigate bias and enable efficient learning in a strongly off-policy regime, where distinct agents share experience through a commonly shared replay module.

Sharing experience permits the agents to benefit from parallel exploration (Kretchmar, 2002) (Figures 1 and 3 ).

Our paper is structured as follows: In Section 2 we revisit pure importance sampling for actor-critic agents (Degris et al., 2012 ) and V-trace, which is notable for allowing to trade off bias and variance in its estimates.

We recall that variance reduction is necessary (Figure 4 left) but is biased in V-trace.

We derive proposition 2 stating that off-policy V-trace is not guaranteed to converge to a locally optimal solution -not even in an idealized scenario when provided with the optimal value function.

Through theoretical analysis (Section 3) and experimental validation (Figure 2 ) we determine that mixing on-policy experience into experience replay alleviates the problem.

Furthermore we propose a trust region scheme (Conn et al., 2000; Schulman et al., 2015; in Section 4 that enables efficient learning even in a strongly off-policy regime, where distinct agents share the experience replay module and learn from each others experience.

We define the trust region in policy space and prove that the resulting estimator is correct (i.e. estimates an improved return).

As a result, we present state-of-the-art data efficiency in Section 5 in terms of median human normalized performance across 57 Atari games (Bellemare et al., 2013) , as well as improved learning efficiency on DMLab30 (Beattie et al., 2016) (Table 1 ).

Figure 1: Sharing experience between agents leads to more efficient hyper-parameter sweeps on 57 Atari games.

Prior art results are presented as horizontal lines (with scores cited from Gruslys et al. (2018) , Hessel et al. (2017) and Mnih et al. (2013) ).

Note that the only previous agent "R2D2" that achieved a score beyond 400% required more than 3,000 million environment steps (see Kapturowski et al. (2019) , page 14, Figure 9 ).

We present the pointwise best agent from hyper-parameter sweeps with and without experience replay (shared and not shared).

Each sweep contains 9 agents with different learning rate and entropy cost combinations.

Replay experiment were repeated twice and ran for 50M steps.

To report scores at 200M we ran the baseline and one shared experience replay agent for 200M steps.

Table 1 : Comparison of state-of-the-art agents on 57 Atari games trained up until 200M environment steps (per game) and DMLab-30 trained until 10B steps (multi-task; all games combined).

The first two rows are quoted from Xu et al. (2018) and Hessel et al. (2019) , the third is our implementation of a pixel control agent from Hessel et al. (2019) and the last two rows are our proposed LASER (LArge Scale Experience Replay) agent.

All agents use hyper-parameter sweeps expect for the marked.

V-trace importance sampling is a popular off-policy correction for actor-critic agents (Espeholt et al., 2018) .

In this section we revisit how V-trace controls the (potentially infinite) variance that arises from naive importance sampling.

We note that this comes at the cost of a biased estimate (see Proposition 1) and creates a failure mode (see Proposition 2) which makes the policy gradient biased.

We discuss our solutions for said issues in Section 4.

Figure 2: Left: Learning entirely off-policy from experience replay fails, while combining on-policy data with experience replay leads to improved data efficiency: We present sweeps on DMLab-30 with experience replays of 10M capacity.

A ratio of 87.5% implies that there are 7 replayed transitions in the batch for each online transition.

Furthermore we consider an agent identical to "LASER 87.5% replay" which however draws all samples from replay.

Its batch thus does not contain any online data and we observe a significant performance decrease (see Proposition 2 and 3).

The shading represents the point-wise best and worst replica among 3 repetitions.

The solid line is the mean.

Right: The effect of capacity in experience replay with 87.5% replay data per batch on sweeps on DMLab-30.

Data-efficiency improves with larger capacity.

Figure 3: Left: Naively sharing experience between distinct agents in a hyper-parameter sweep fails (green) and is worse than the no-replay baseline (blue).

The proposed trust region estimator mitigates the issue (red).

Right: Combining population based training with trust region estimation improves performance further.

All replay experiments use a capacity of 10 million observations and 87.5% replay data per batch.

We follow the notation of Sutton et al. (2018) where an agent interacts with its environment, to collect rewards.

On each discrete time-step t, the agent selects an action a t ; it receives in return a reward r t and an observation o t+1 , encoding a partial view of the environment's state s t+1 .

In the fully observable case, the RL problem is formalized as a Markov Decision Process (Bellman, 1957) : a tuple (S, A, p, γ), where S, A denotes finite sets of states and actions, p models rewards and state transitions (so that r t , s t+1 ∼ p(s t , a t )), and γ is a fixed discount factor.

A policy is a mapping π(a|s) from states to action probabilities.

The agent seeks an optimal policy π * that maximizes the value, defined as the expectation of the cumulative discounted returns

Off-policy learning is the problem of finding, or evaluating, a policy π from data generated by a different policy µ.

This arises in several settings.

Experience replay (Lin, 1992) mixes data from multiple iterations of policy improvement.

In large-scale RL, decoupling acting from learning (Nair et al., 2015; Espeholt et al., 2018) causes the experience to lag behind the latest agent policy.

Finally, it is often useful to learn multiple general value functions (Sutton et al., 2011; Mankowitz et al., 2018; Lample & Chaplot, 2016; Mirowski et al., 2017; Jaderberg et al., 2017b) or options (Sutton et al., 1999; Bacon et al., 2017 ) from a single stream of experience.

On-policy n-step bootstraps give more accurate value estimates in expectation with larger n (Sutton et al., 2018) .

They are used in many reinforcement learning agents (Mnih et al., 2016; Schulman et al., 2017; Hessel et al., 2017) .

Unfortunately n must be chosen suitably as the estimates variance increases with n too.

It is desirable to obtain benefits akin to n-step returns in the off-policy case.

To this end multi-step importance sampling (Kahn, 1955) can be used.

This however adds another source of (potentially infinite (Sutton et al., 2018) ) variance to the estimate.

Importance sampling can estimate the expected return V π from trajectories sampled from µ = π, as long as µ is non-zero whereever π is.

We employ a previously estimated value function V as a bootstrap to estimate expected returns.

Following Degris et al. (2012) , a multi-step formulation of the expected return is

where E µ denotes the expectation under policy µ up to an episode termination, δ t V = r t + γV (s t+1 ) − V (s t ) is the temporal difference error in consecutive states s t+1 , s t , and π t = π t (a t |s t ).

Importance sampling estimates can have high variance.

Tree Backup (Precup et al., 2000) , and Q(λ) (Sutton et al., 2014) address this, but reduce the number of steps before bootstrapping even when this is undesirable (as in the on-policy case).

RETRACE (Munos et al., 2016 ) makes use of full returns in the on-policy case, but it introduces a zero-mean random variable at each step, adding variance to empirical estimates in both on-and off-policy cases.

V-trace (Espeholt et al., 2018) reduces the variance of importance sampling by trading off variance for a biased estimate of the return -resulting in a failure mode (see Proposition 2).

It uses clipped importance sampling ratios to approximate

i=0 c i ρ t δ t+k V where V is a learned state value estimate used to bootstrap, and ρ t = min [π t /µ t ,ρ], c t = min [π t /µ t ,c] are the clipped importance ratios.

Note that, differently from RETRACE, V-trace fully recovers the Monte Carlo return when on policy.

It similarly reweights the policy gradient as:

Note that ∇Vπ(s t ) recovers the naively importance sampled policy gradient forρ → ∞. In the literature, it is common to subtract a baseline from the action-value estimate r t + γVπ(s t+1 ) to reduce variance (Williams, 1992) , omitted here for simplicity.

The constantsρ ≥c ≥ 1 (typically chosenρ =c = 1) define the level of clipping, and improve stability by ensuring a bounded variance.

For any givenρ, the bias introduced by V-trace in the value and policy gradient estimates increases with the difference between π and µ. We analyze this in the following propositions.

Proposition 1.

The V-trace value estimate Vπ is biased: It does not match the expected return of π but the return of a related implied policyπ defined by equation 3 that depends on the behaviour policy µ:π

Proof.

See Espeholt et al. (2018) .

Note that the biased policyπ µ can be very different from π.

Hence the V-trace value estimate Vπ may be very different from V π as well.

As an illustrative example, consider two policies over a set of two actions, e.g. "left" and "right" represented as a tuple of probabilities.

Let us investigate µ = (φ, 1 − φ) and π = (1 − φ, φ) defined for any suitably small φ ≤ 1.

Observe that π and µ share no trajectories (state-action sequences) in the limit as φ → 0 and they get more focused on one action.

A practical example of this could be two policies, one almost always taking a left turn and one always taking the right.

Given sufficient data of either policy it is possible to estimate the value of the other e.g. with naive importance sampling.

However observe that V-trace withρ = 1 will always estimate a biased value -even given infinite data.

Observe that min [µ(a|x), π(a|x)] = min [φ, 1 − φ] for both actions.

Thusπ µ is uniform rather than resembling π the policy.

The V-trace estimate Vπ would thus compute the average value of "left" and "right" -poorly representing the true V π .

Proposition 2.

The V-trace policy gradient is biased: given the the optimal value function V * the V-trace policy gradient does not converge to a locally optimal π * for all off-policy behaviour distributions µ.

Proof.

See Appendix C.

In Proposition 2 we presented a failure mode in V-trace where the variance reduction biases the value estimate and policy gradient.

V-trace computes biased Q-estimates Q ω = Q resulting in a wrong local policy gradient:

The question of how biased the resulting policy will be depends on whether the distortion changes the argmax of the Q-function.

Little distortions that do not change the argmax will result in the same local fixpoint of the policy improvement.

The policy will continue to select the optimal action and it will not be biased at this state.

The policy will however be biased if the Q-function is distorted too much.

For example consider a ω(s, a) that swaps the argmax for the 2nd largest value, the regret will then be the difference between the maximum and the 2nd largest value.

Intuitively speaking the more distorted the Q ω , the larger will be the regret compared to the optimal policy.

More precisely, the regret of learning a policy that maximizes the distorted Q ω at state s is:

where a * = argmax b (Q, b) is the optimal action according to the real Q and

, is the optimal action according to the distorted Q ω .

For generality, we denote A * as the set of best actions -covering the case with multiple with identical optimal Q-values.

Proposition 3 provides a mitigation: Clearly the V-trace policy gradient will converge to the same solution as the true on-policy gradient if the argmax of the Q-function is preserved at all states in a tabular setting.

We show that this can be achieved by mixing a sufficient proportion α of on-policy experience into the computation.

We show in equation 13 in the Appendix that choosing α such that

will result in a policy that correctly chooses the best action at state s. Note that

Intuitively: the larger the action value gap of the real Q-function Q(s, a * ) − Q(s, b) the lower the right hand side and the less on-policy data is required.

is negative, then α may be as small as zero and we enabling even pure off-policy learning.

Finally note that the right hand side decreases due to d µ (s)/d π (s) if π visits the state s more often than µ.

All of those conditions can be computed and checked if an accurate Q-function and state distribution is accessible.

How to use imperfect Q-function estimates to adaptively choose such an α remain a question for future research.

We provide experimental evidence for these results with function approximators in the 3-dimensional simulated environment DMLab-30 with various α ≥ 1/8 in Section 5.3 and Figure 2 .

We observe that α = 1/8 is sufficient to facilitate stable learning.

Furthermore it results in better data-efficiency than pure on-policy learning as it utilizes off-policy replay experience.

Proposition 3.

Mixing on-policy data into the V-trace policy gradient with the ratio α reduces the bias by providing a regularization to the implied state-action values.

In the general function approximation case it changes the off-policy V-trace policy gradient from

is a regularized stateaction estimate and d π , d µ are the state distributions for π and µ. Note that there exists α ≤ 1 such that Q α has the same argmax (i.e. best action) as Q.

Proof.

See Appendix C.

Mixing online data with replay data has also been argued for by Zhang & Sutton (2017) , as a heuristic way of reducing the sensitivity of reinforcement learning algorithms to the size of the replay memory.

Proposition 3 grounds this in the theoretical properties of V-trace.

To mitigate the bias and variance problem of V-trace and importance sampling we propose a trust region scheme that adaptively selects only suitable behaviour distributions when estimating the state-value of π.

To this end we introduce a behaviour relevance function that classifies behaviour as relevant.

We then define a trust-region estimator that computes expectations (such as expected returns, or the policy gradient) only on relevant transitions.

In proposition 4 and 5 we show that this trust region estimator indeed computes new state-value estimates that improve over the current value function.

While our analysis and proof is general we propose a suitable behaviour relevance function in section 4.3 that employs the Kullback Leibler divergence between target policy π and implied policyπ µ : KL (π(·|s)||π µ (·|s)).

We provide experimental validation in Figure 3 .

In off-policy learning we often consider a family of behaviour policies either indexed by training iteration t: M T = {µ t |t < T } for experience replay, or by a different agent k: M K = {µ k |k ∈ K} when training multiple agents.

In the classic experience replay case we then sample a time t and locate the transition τ that was generated earlier via µ t .

This extends naturally to the multiple agent case where we sample an agent index k and then obtain a transition for such agent or tuples of (k, t).

Without loss of generality we simplify this notation and index sampled behaviour policies by a random variable z ∼ Z that represents the selection process.

While online reinforcement learning algorithms process transitions τ ∼ π, off-policy algorithms process τ ∼ µ z for z ∼ Z. In this notation, given equation (1) and a bootstrap V , the expectation of importance sampled off-policy returns at state s t is described by:

where

Above E µz|z represents the expectation of sampling from a given µ z .

The conditioning on z is a notational reminder that this expectation does not sample z or µ z but experience from µ z .

For any sampled z we obtain a µ z and observe that the inner expectation wrt.

experience of µ z in equation (4) recovers the expected on-policy return in expectation:

Thus

.

This holds provided that µ z is non-zero wherever π is.

This fairly standard assumption leads us straight to the core of the problem: it may be that some behaviours µ z are ill-suited for estimating the inner expectation.

However, standard importance sampling applied to very off-policy experience divides by small µ resulting in high or even infinite variance.

Similarly, V-trace attempts to compute an estimate of the return following π resulting in limited variance at the cost of a biased estimate in turn.

The key idea of our proposed solution is to compute the return estimate for π at each state only from a subset of suitable behaviours µ z :

M β,π (s) = {µ z |z ∈ Z and β(π, µ, s) < b} as determined by a behaviour relevance function β(π, µ, s) : (M Z , M Z , S) → R and a threshold b. The behaviour relevance function decides if experience from a behaviour is suitable to compute an expected return for π.

It can be chosen to control properties of V π mix by restricting the expectation on subsets of Z. In particular it can be used to control the variance of an importance sampled estimator: Observe that the inner expectation E µz G π,µ (s t ) z in equation (4) already matches the expected return V π .

Thus we can condition the expectation on arbitrary subsets of Z without changing the expected value of V π mix .

This allows us to reject high variance G π,µ without introducing a bias in V π mix .

The same technique can be applied to V-trace where we can reject return estimates with high bias.

Using a behaviour relevance function β(s) we can define a trust region estimator for regular importance sampling (IS) and V-trace and show their correctness.

We define the trust region estimator as the conditional expectation

with λ-returns G, chosen as G IS for importance sampling and G Vtrace for V-trace:

where λ π,µ (s t ) is designed to constraint Monte-Carlo bootstraps to relevant behaviour: λ π,µ (s t ) = 1 β(π,µ,st)<b and ρ z,t+k = min πt+i µz,t+i ,ρ and c z,t+k are behaviour dependent clipped importance rations.

Thus both G π,µz IS and G

Vtrace are a multi-step return estimators with adaptive length.

Note that only estimators with length ≥ 1 are used in V π trusted .

Due to Minkowski's inequality the trust region estimator thus shows at least the same contraction as a 1-step bootstrap, but can be faster due to its adaptive nature:

be a set of importance sampling estimators as defined in equation 7.

Note that they all have the same fix point V π and contract with at least γ.

Then the contraction properties carry over to V π trusted .

In particular

Proof.

See Appendix C.

Vtrace be a set of V-trace estimators (see equation 8) with corresponding fixed points V z (see equation 3) to which they contract at a speed of an algorithm and behaviour specific

Proof.

See Appendix C.

Note how the choice of β and thus M β,π enables us to discard ill-suited G π,µz

Vtrace from the estimation of V π trusted .

Recall that V-trace fixed points V z are biased.

Thus β allows us to selectively create the V-trace target

and control its bias and the shrinkage

Similarly it can control cases where we can not use the exact importance sampled estimator.

The same approach based on nested expectations can be applied to the expectation of the policy gradient estimate and allows to control the bias and greediness (see Proposition 2) there as well.

In Proposition 5 we have seen that the quality of the trust region V-trace return estimator depends on β.

A suitable choice of β can move the return estimate V β closer to V π and improve the shrinkage by

Hence, we employ a behaviour relevance function β KL that rejects high bias transitions by estimating the Kulback-Leibler divergence between the target policy π and the implied policyπ µz for a sampled behaviour µ z .

Recall from Proposition 1 thatπ µz determines the fixed point of the V-trace estimator for behaviour µ z and thus determines the bias in V z .

Note that the behaviour probabilities µ z can be evaluated and saved to the replay when the agent executes the behaviour, similarly the target policy π is represented by the agents neural network.

Using both and equation 3,π µ can be computed.

For large or infinite action spaces a Monte Carlo estimate of the KL divergence can be computed.

It is possible to define separate behaviour relevance functions for the policy and value estimate.

For simplicity we reject transitions entirely for all estimates and do not consider rejected transitions for the policy gradient and value gradient updates or auxiliary tasks.

As described above we stop the Monte-Carlo bootstraps once they reach undesirable state-behaviour pairs.

Note that this censoring procedure is computed from state dependent β(π, µ, s) and ensures that the choice of bootstrapping does not depend on the sampled actions.

Note that rejection by an action-based criteria such as small π(a|s)/µ(a|s) would introduce an additional bias which we avoid by choosing β KL .

We present experiments to support the following claims:

• Section 5.2: Uniform experience replay obtains comparable results as prioritized experience replay, while being simpler to implement and tune.

• Section 5.3: Using fresh experience before inserting it in experience replay is better than learning purely off-policy from experience replay -in line with Proposition 3.

• Section 5.4: Sharing experience without trust region performs poorly as suggested by Proposition 2.

Off-Policy Trust-Region V-trace solves this issue.

• Section 5.5: Sharing experience can take advantage of parallel exploration and obtains state-of-the-art performance on Atari games, while also saving memory through sharing a single experience replay.

We use the V-trace distributed reinforcement learning agent (Espeholt et al., 2018) (Pascanu et al., 2012) .

Updates are computed on mini-batches of 32 (regular) and 128 (replay) trajectories, each corresponding to 19 steps in the environment.

In the context of DeepMind Lab, we consider the multi-task suite DMLab-30 (Espeholt et al., 2018) , as the visuals and the dynamics are more consistent across tasks.

Furthermore the multi-task regime is particularly suitable for the investigation of strongly off-policy data distributions arising from sharing the replay across agents, as concurrently learning agents can easily be stuck in different policy plateaus, generating substantially different data (Schaul et al., 2019) .

As in Espeholt et al. (2018) , in the multi-task setting each agent trains simultaneously on a uniform mixture of all tasks rather than individually on each game.

The score of an agent is thus the median across all 30 tasks.

Following Hessel et al. (2019), we augment our agent with multi-task Pop-Art normalization and PixelControl.

We use a PreCo LSTM (Amos et al., 2018) instead of the vanilla one (Hochreiter & Schmidhuber, 1997) .

Updates are computed on mini-batches of multiple trajectories chosen as above, each corresponding to 79 steps in the environment.

In early experiments we found that computing the entropy cost only on the online data provided slightly better results, hence we have done so throughout our experiments.

In all our experiments, experience sampled from memory is mixed with online data within each minibatch -following Proposition 3.

Episodes are removed in a first in first out order, so that replay always holds the most recent experience.

Unless explicitly stated otherwise we consider hyper-parameter sweeps, some of which share experience via replay.

In this setting multiple agents start from-scratch, run concurrently at identical speed, and add their new experience into a common replay buffer.

All agents will then draw uniform samples from the replay buffer.

On DMLab-30 we consider both regular hyper-parameter sweeps and sweeps with population based training (PBT) (Jaderberg et al., 2017a) .

On DMLab-30 sweeps contain 10 agents with hyper-parameters sampled similar as Espeholt et al. (2018) but fixed RMSProp = 0.1.

On Atari sweeps contain 9 agents with different constant learning rate and entropy cost combinations {3 · 10 −4 , 6 · 10 −4 , 1.2 · 10 −3 } × {5 · 10 −3 , 1 · 10 −2 , 2 · 10 −2 } (distributed by factors {1/2, 1, 2} around the initial parameters reported in Espeholt et al. (2018) ).

Although our focus is on efficient hyper-parameter sweeps given crude initial parameters, we also present a single-agent LASER experiment using the same tuned schedule as Espeholt et al. (2018) , a 87.5% replay ratio and a 15M replay.

We store the entire episodes in the replay buffer and replay each episode from the beginning, using the most recent network parameters to recompute the LSTM states along the way: this is particularly critical when sharing experience between different agents, which may have arbitrarily different state representations.

Prioritized experience replay has the potential to provide more efficient learning compared to uniform experience replay (Schaul et al., 2015; .

However, it also introduces a number of new hyper-parameters and design choices: the most critical are the priority metric, how strongly to bias the sampling distribution, and how to correct for the resulting bias.

Uniform replay is instead almost parameter-free, requires little tuning and can be easily shared between multiple agents.

Experiments provided in Figure 4 in the appendix showed little benefit of actor critic prioritized replay on DMLab-30.

Furthermore priorities are typically computed from the agent specific metrics such as the TD-error, which are ill-defined when replay is shared among multiple agents.

Hence we used uniform replay for our further investigations.

Figure 2 (left) shows that performance degrades significantly when online data is not present in the batch.

This experimentally validates Propositions 2 and 3 that highlight difficulties of learning purely off-policy.

Furthermore Figure 2 (right) shows that best results are obtained with experience replay of 10M capacity and 87.5% ratio.

A ratio of 87.5% = 7/8 corresponds to 7 replay samples for each online sample.

We have considered ratios of 1/2, 3/4, and 7/8 and observed stable training for all of them.

Observe that among those values, larger ratios are more data-efficient as they take advantage of more replayed experience per training step.

In line with proposition 2 we observe in Figure 3 (left) that hyper-parameter sweeps without trustregion are even surpassed by the baseline without experience replay.

State-of-the-art results are obtained in Figure 3 (right) when experience is shared with trust-region in a PBT sweep.

Observe that this indicates parallel exploration benefits and saves memory at the same time: in our sweep of 10 replay agents the difference between 10 × 10M (separate replays) and 10M (shared replay) is 10-fold.

This effect would be even more pronounced with larger sweeps.

As discussed in section 2.3, the bias in V-trace occurs due to the clipping of importance ratios.

A potential solution of reducing the bias would be to increase theρ threshold to clip less aggressively and accept increased variance.

Figure 4 in the appendix shows that this is not a solution.

We apply our proposed agent to Atari which has been a long established suite to evaluate reinforcement algorithms (Bellemare et al., 2013 ).

Since we focus on sample-efficient learning we present our results in comparison to prior work at 200M steps (Figure 1 ).

Shared experience replay obtains even better performance than not shared experience replay.

This confirms the efficient use of parallel exploration (Kretchmar, 2002) .

The fastest prior agent to reach 400% is presented by Kapturowski et al. (2019) requiring more than 3,000M steps.

LASER with shared replay achieves 423% in 60M per agent.

Given 200M steps it achieves 448%.

We also present a single (no sweep) LASER agent that achieves 431% in 200M steps.

We have presented LASER -an off-policy actor-critic agent which employs a large and shared experience replay to achieve data-efficiency.

By sharing experience between concurrently running experiments in a hyper-parameter sweep it is able to take advantage of parallel exploration.

As a result it achieves state-of-the-art data efficiency on 57 Atari games given 200M environment steps.

Furthermore it achieves competitive results on both DMLab-30 and Atari under regular, not shared experience replay conditions.

To facilitate this algorithm we have proposed two approaches: a) mixing replayed experience and on-policy data and b) a trust region scheme.

We have shown theoretically and demonstrated through a series of experiments that they enable learning in strongly off-policy settings, which present a challenge for conventional importance sampling schemes.

Increasing the clipping constantρ in V-trace reduces bias in favour of increased variance.

We investigate if reducing bias in this manner enables sharing experience replay between multiple agents in a hyper-parameter sweep.

Figure 4 (left) shows that this is not a solution, thus motivating our trust region scheme.

In fact sharing experience replay in this particular way is worse than pure online learning.

This motivates the use of our proposed trust region scheme.

On a side note, increased clipping thresholds resulting in worse performance verifies the importance of variance reduction through clipping.

Right: Median human normalized performance across 30 tasks for the best agent in a sweep, averaged across 2 replicas.

All replay experiments use 50% replay ratio and a capacity of 3 million observations.

We investigate if uncorrected LSTM states can be used in combination with different replay modes.

We consider uniform sampling and prioritization via the critic's loss, and include both full (β = 1) and partial (β = 0.5) importance corrections A.2 PRIORITIZED AND UNIFORM EXPERIENCE REPLAY, LSTM STATES With prioritized experience replay each transition τ is sampled with probability P (τ ) ∝ p α τ , for a suitable unnormalized priority score p τ and a global tunable parameter α.

It is common (Schaul et al., 2015; Hessel et al., 2017) to then weight updates computed from that sample by 1/P (τ ) β for 0 < β ≤ 1, where β = 1 fully corrects for the bias introduced in the state distribution.

In one step temporal difference methods, typical priorities are based on the immediate TD-error, and are typically recomputed after a transition is sampled from replay.

This means low priorities might stay low and get stale -even if the transition suddenly becomes relevant.

To alleviate this issue, the sampling distribution is mixed with a uniform, as controlled by a third hyper parameter .

The performance of agents with prioritized experience replay can be quite sensitive to the hyperparameters α, β, and .

A critical practical consideration is how to implement random access for recurrent memory agents such as agents using an LSTM.

Prioritized agents sample a presumably interesting transition from the past.

This transition may be at any position within the episode.

To infer the correct recurrent memory-state at this environment-state all earlier environment-states within that episode would need to be replayed.

A prioritized agent with a random access pattern would thus require costly LSTM refreshes for each sampled transition.

If LSTM states are not recomputed representational missmatch (Kapturowski et al., 2019) occurs.

Sharing experience between multiple agents amplifies the issue of LSTM state representation missmatch.

Here each agent has its own network parameters and the state representations between agents may be arbitrarily different.

As a mitigation Kapturowski et al. (2019) use a burn-in window or to initialize with a constant starting state.

We note that those solutions can only partially mitigate the fundamental issue and that counter examples such as arbitrarily long T-Mazes (Tolman, 1948; Olton, 1979) can be constructed easily.

We thus advocate for uniform sampling.

In our implementation we uniformly sample an episode.

Then we replay each episode from the beginning, using the most recent network parameters to recompute the LSTM states along the way: this is particularly critical when sharing experience between different agents, which may have arbitrarily different state representations.

This solution is exact and cost-efficient as it only requires one additional forward pass for each learning step (forward + backward pass).

An even more cost efficient approach would be to not refresh LSTM states at all.

Naturally this comes at the cost of representational missmatch.

However it would allow for an affordable implementation of prioritized experience replay.

We investigate this in Figure 4 (right) and observe that it is not viable.

We compare a baseline V-trace agent with no experience replay, one with uniform experience replay, and two different prioritized replay agents.

We do not refresh LSTM states for any of the agents.

The uniform replay agent is more data efficient then the baseline, and also saturates at a higher level of performance.

The best prioritized replay agent uses full importance sampling corrections (β = 1).

However it performs no higher than with uniform replay.

We therefore we used uniform replay with full state correction for all our investigations in the paper.

For evaluation, we average episode returns within buckets of 1M (Atari) and 10M (DMLab) environment steps for each agent instance, and normalize scores on each game by using the scores of a human expert and a random agent (van Hasselt et al., 2016) .

In the multi-task setting, we then define the performance of each agent as the median normalized score of all levels that the agent trains on.

Given the use of population based training, we need to perform the comparisons between algorithms at the level of sweeps.

We do so by selecting the best performing agent instance within each sweep at any time.

Note that for the multi-task setting, our approach of first averaging across many episodes, then taking the median across games, on DMLab further downsampling to 100M env steps, and only finally selecting the maximum within the sweep, results in substantially lower variance than if we were to compute the maximum before the median and smoothing.

All DMLab-30 sweeps are repeated 3× with the exception of ρ = 2 and ρ = 4 in Figure 4 .

We then plot a shaded area between the point-wise best and worst replica and a solid line for the mean.

Atari sweeps having 57 games are summarized and plotted by the median of the human-normalized scores.

We present algorithm pseudocode for LASER with trust region (Algorithm 1).

For clarity we present a version without LSTM and focus on the single agent case.

The multi-agent case is a simple extension where all agents save to the same replay database and also sample from the same replay.

Also each agent starts with different network parameters and hyper-parameters.

The LSTM state recomputation can be achieved with Replayer Threads (nearly identical to Actor Threads) that sample entire epsiodes from replay, step through them while reevaluating the LSTM state and slice the experience into trajectories of length T .

Similar to regular LSTM Actor Threads from Espeholt et al. (2018) the Replayer Threads send each trajectory together with an LSTM state to the learning thread via a queue.

The Learner Thread initializes the LSTM with the transmitted state when the LSTM is unrolled over the trajectory.

Initialize parameter vectors θ.

Initialize π 1 = π θ .

Actor Thread: while training is ongoing do Sample trajectory unroll u = {τ t } t∈{1,...,T } of length T by acting in the environment using the latest π k where τ t = (s t , a t , r t , µ t = π k (s t |·)).

Compute trust-region V-trace return V t,b using 8 where

, where ρ is the clipped v-trace importance sampling ratio.

Perform gradient update to θ using

, denote the resulting π θ as π k+1 . end for

We have stated five propositions in our paper for which we provide proofs below.

Proposition 1.

The V-trace value estimate Vπ is biased: It does not match the expected return of π but the return of a related implied policyπ defined by equation 9 that depends on the behaviour policy µ:π Proof.

See Espeholt et al. (2018) .

Proposition 2.

The V-trace policy gradient is biased: given the the optimal value function V * the V-trace policy gradient does not converge to a locally optimal π * for all off-policy behaviour distributions µ.

Consider a tabular counter example with a single (locally) optimal policy at s t given by π * (s t ) = argmax π a∈A π(a|s t )Q * (a, s t ) that always selects the action argmax a Q * (a, s t ).

Even in this ideal tabular setting V-trace policy gradient estimates a differentπ * rather than the optimal π * as follows ∇V * ,π (s t ) = E µ [ρ t (r t + γV * (s t+1 )∇ log π(a t |s t )] = E µ [ρ t Q * (s t , a t )∇ log π(a t |s t )] = E µ min π(a t |s t ) µ(a t |s t ) ,ρ Q * (s t , a t )∇ log π(a t |s t ) = E µ π(a t |s t ) µ(a t |s t ) min 1,ρ µ(a t |s t ) π(a t |s t ) Q * (s t , a t )∇ log π(a t |s t )

= E π min 1,ρ µ(a t |s t ) π(a t |s t ) Q * (s t , a t )∇ log π(a t |s t )

= E π [ω(s t , a t )Q * (s t , a t )∇ log π(a t |s t )]

= E π [Q * ,ω (s t , a t )∇ log π(a t |s t )]

Observe how the optimal Q-function Q * is scaled by ω(s t , a t ) = min 1,ρ µ(at|st) π(at|st) ≤ 1 resulting in implied state-action values Q * ,ω .

This penalizes actions where µ(a t |s t )ρ < π(a t |s t ) and makes V-trace greedy w.r.t.

to the remaining ones.

Thus µ can be chosen adversarially to corrupt the optimal state action value.

Note thatρ is a constant typically chosen to be 1.

To prove the lemma consider a counter example such as an MDP with two actions and Q * = (2, 5) and µ = (0.9, 0.1) and initial π = (0.5, 0.5).

Here the second action with expected return 5 is clearly favourable.

Abusing notation µ/π = (1.8, 0.2).

Thus Qπ ,ω = (2 * 1, 5 * 0.2) = (2, 1).

Thereforẽ π * = (1, 0) wrongly selects the first action.

with the V-trace distortion factor ω(s t , a t ) = min 1,ρ µ(at|st) π(at|st) ≤ 1 that can de-emphasize action values and Q ω (s, a) = ω(s, a)Q(s, a).

We display the Atari per-level performance of various agents at 50M and 200M environment steps in Table 2 .

The scores correspond to the agents presented in Figure 1 .

The LASER scores are computed by averaging the last 100 episode returns before 50M or respectively 200M environment frames have been experienced.

Following the procedure defined by Mnih et al. (2015) we initialize the environment with a random number of no-op actions (up to 37 in our case).

Again following Mnih et al. (2015) episodes are terminated after 30 minutes of gameplay.

Note that Xu et al. (2018) have not published per-level scores.

Rainbow scores are obtained from Hessel et al. (2017) .

@highlight

We investigate and propose solutions for two challenges in reinforcement learning: (a) efficient actor-critic learning with experience replay (b) stability of very off-policy learning.
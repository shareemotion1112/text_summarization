Many practical reinforcement learning problems contain catastrophic states that the optimal policy visits infrequently or never.

Even on toy problems, deep reinforcement learners periodically revisit these states, once they are forgotten under a new policy.

In this paper, we introduce intrinsic fear, a learned reward shaping that accelerates deep reinforcement learning and guards oscillating policies against periodic catastrophes.

Our approach incorporates a second model trained via supervised learning to predict the probability of imminent catastrophe.

This score acts as a penalty on the Q-learning objective.

Our theoretical analysis demonstrates that the perturbed objective yields the same average return under strong assumptions and an $\epsilon$-close average return under weaker assumptions.

Our analysis also shows robustness to classification errors.

Equipped with intrinsic fear, our DQNs solve the toy environments and improve on the Atari games Seaquest, Asteroids, and Freeway.

Following success on Atari games BID20 and the board game Go BID28 , many researchers have begun exploring practical applications of deep reinforcement learning (DRL).

Some investigated applications include robotics BID15 , dialogue systems BID6 BID17 , energy management BID23 , and self-driving cars BID26 .

Amid this push to apply DRL, we might ask, can we trust these agents in the wild?

Agents acting in real-world environments might possess the ability to cause catastrophic outcomes.

Consider a self-driving car that might hit pedestrians or a domestic robot that might injure a child.

We might hope to prevent DRL agents from ever making catastrophic mistakes.

But doing so requires extensive prior knowledge of the environment in order to constrain the exploration of policy space BID7 .Many conflicting definitions of safety and catastrophe exist, a problem that invites further philosophical consideration.

In this paper, we introduce a specific but plausible notion of avoidable catastrophes.

These are states that prior knowledge dictates an optimal policy should never visit.

For example, we might believe that an optimal self-driving algorithm would never hit a pedestrian.

Moreover, we assume that an optimal policy never even comes near an avoidable catastrophe state.

We define proximity in trajectory space, and not by the geometry of feature space.

We denote states proximal to avoidable catastrophes as danger states.

While we don't assume prior knowledge of which states are dangerous, we do assume the existence of a catastrophe detector.

After encountering a catastrophic state, an agent can realize this and take action to avoid dangerous states in the future.

Given this definition, we address two challenges: First, can we expect DRL agents, after experiencing some number of catastrophic failures, to avoid perpetually making the same mistakes?

Second, can we use our prior knowledge that catastrophes should be kept at a distance to accelerate learning of a DRL agent?

Our experiments show that even on toy problems, the deep Q-network (DQN), a basic algorithm behind many of today's state-of-the-art DRL systems, struggles on both counts.

Even in toy environments, DQNs may encounter thousands of catastrophes before learning to avoid them and are susceptible to repeating old errors.

We call this latter problem the Sisyphean curse.

This poses a formidable obstacle to using DQNs in the real world.

How can we hand over responsibility for consequential actions (control of a car, say) to a DRL agent if it may be doomed to periodically remake every kind of mistake, however grave, so long as it continues to learn?

Imagine a self-driving car that had to periodically hit a few pedestrians in order to remember that is undesirable.

In the tabular setting, an RL agent never forgets the learned dynamics of its environment, even as its policy evolves.

Moreover, if the Markovian assumption holds, eventual convergence to a globally optimal policy is guaranteed.

Unfortunately, the tabular approach becomes infeasible in high-dimensional, continuous state spaces.

The trouble for DQNs owes to the use of function approximation BID22 .

When training a DQN, we successively update a neural network based on experiences.

These experiences might be sampled in an online fashion, from a trailing window (experience replay buffer), or uniformly from all past experiences.

Regardless of which mode we use to train the network, eventually, states that a learned policy never encounters will come to form an infinitesimally small region of the training distribution.

At such times, our networks are subject to the classic problem of catastrophic interference BID19 BID18 .

Nothing prevents the DQN's policy from drifting back towards a policy that revisits long-forgotten catastrophic mistakes.

More formally, we characterize the problem as unfolding in the following steps: (i) Training under distribution D, our agent produces a safe policy ?? s that avoids catastrophes (ii) Collecting data generated under ?? s yields a new distribution of transitions D (iii) Training under D , the agent produces ?? d , a policy that once again experiences avoidable catastrophes.

To illustrate the brittleness of modern DRL algorithms, we introduce a simple pathological problem called Adventure Seeker.

This problem consists of a one-dimensional continuous state, two actions, simple dynamics, and a clear analytic solution.

Nevertheless, the DQN fails.

We then show that similar dynamics exist in the classic RL environment Cart-Pole.

In this paper, to combat these problems, we propose intrinsic fear.

In this approach, we train a supervised fear model that predicts which states are likely to lead to a catastrophe within k r steps.

The output of the fear model (a probability), scaled by a fear factor penalizes the Q-learning target.

Our approach draws inspiration from intrinsic motivation BID5 .

However, instead of perturbing the reward function to encourage the discovery of novel states, we perturb it to discourage revisiting catastrophic states.

We validate the approach both empirically and theoretically.

Our experiments address both our Adventure Seeker problem and Cartpole as well as the Atari games Seaquest and Asteroids, and Freeway.

For these environments, we label each loss of a life as a catastrophic state.

On the toy environments, the intrinsic fear agent learns to avoid death indefinitely, achieving unbounded reward per episode.

On Seaquest and Asteroids, the intrinsic fear agent improves markedly and on Freeway the improvement is dramatic.

Theoretically, we demonstrate the following: First, we prove that when the reward is bounded and the optimal policy rarely visits the catastrophic states, the policy learned on the altered value function has return similar to the optimal policy on the original value function.

Second we prove that the method is robust to noise in the danger model.

Over a series of turns, an agent interacts with its environment via a Markov decision process, or MDP, (S, A, T , R, ??).

At each step t, an agent observes a state s ??? S. The agent then chooses an action a ??? A according to some policy ??.

In turn, the environment transitions to a new state s t+1 ??? S according to transition dynamics T (s t+1 |s t , a t ) and generates a reward r t with expectation R(s, a).

This cycle continues until each episode terminates.

The goal of an agent is to maximize the cumulative discounted return T t=0 ?? t r t .

Temporaldifferences (TD) methods BID30 such as Q-learning BID33 model the Q-function, which gives the optimal discounted total reward of a state-action pair; the greedy policy w.r.t.

the Q-function is optimal BID31 .

Problems of practical interest tend to have large state spaces, thus the Q-function is typically approximated by parametric models such as neural networks.

In Q-learning with function approximation, an agent alternately collects experiences by acting greedily with respect to Q(s, a; ?? Q ) and updates its parameters ?? Q .

Updates proceed as follows.

For a given experiences (s t , a t , r t , s t+1 ), we minimize the squared Bellman error: DISPLAYFORM0 (1) for y t = r t + ?? ?? max a Q(s t+1 , a ; ?? Q ).

Traditionally, the parameterised Q(s, a; ??) is trained by stochastic approximation, estimating the loss on each experience as it is encountered, yielding the update: DISPLAYFORM1 Q-learning methods also require an exploration strategy for action selection.

For simplicity, we consider only the -greedy heuristic.

A few tricks help to stabilize Q-learning with function approximation.

Of particular relevance to this work is experience replay BID16 : the RL agent maintains a buffer of past experiences, applying TD-learning on randomly selected mini-batches of experience to update the Q-function.

In this paper, we propose a new formulation of the safety problem.

We suppose there exists a subset C ??? S of states that an optimal policy encounters them very rarely or never and denote them catastrophic states.

Moreover, we assume that for some environments, optimal policies are rarely within a short distance of a catastrophic state.

As a measure of distance, we consider steps in trajectory space.

We define the distance d(s i , s j ) to be length N of the smallest sequence of transitions {(s t , a t , r t , s t+1 )} N t=1 that traverses state space from s i to s j .

Definition 2.1.

Suppose that we are given a priori knowledge that acting according to the optimal policy ?? * , an agent never encounters states s ??? S for which lie within distance d(s, c) < k ?? for any catastrophe state c ??? C. Then each state s for which ???c ??? C s.t.

d(s, c) < k ?? is a danger state.

We also suppose that the agent can recognize the catastrophe states as they are encountered.

Definition 2.2.

A catastrophe detector is a function f : S ??? {0, 1} that returns 1 if and only if a state is a catastrophe state.

We propose Intrinsic Fear (IF) (Algorithm 1), a novel algorithm for avoiding catastrophes when learning online with function approximation.

In our approach, we maintain both a DQN and a separate, supervised fear model F : S ??? [0, 1].

Our fear model F provides an auxiliary source of reward, penalizing the Q-learner for entering possibly dangerous states.

The goal in modeling danger states is twofold.

First, by shaping rewards away from suboptimal states, we encode prior knowledge about the environment and can thus accelerates learning.

Second, when catastrophic states correspond to especially undesirable outcomes, the learned reward shaping can protect DQNs, which are susceptible to catastrophic forgetting, from drifting close to catastrophic states.

Owing to this self-assigned reward, once the fear model is trained, a Q-learner might update to avoid catastrophes without having to actually repeat them, so long as the fear model is not itself susceptible to catastrophic forgetting.

We draw some inspiration from the idea of a parent scolding a child for running around with a knife.

The child can learn to adjust its behavior without actually having to stab someone.

We also draw inspiration from the way humans appear to process traumatic experience, remembering especially bad events vividly even as most other memories from the same time period fade.

Perhaps this selective memorization of bad events confers a benefit for avoiding similar outcomes in the future.

Our instantiation of intrinsic fear works as follows: In addition to the DQN, we maintain a binary classifier that we term a fear model.

In our case, we use a neural network of the same architecture as the DQN (but for the output layer).

The fear model's purpose is to predict the probability that any state will lead to catastrophe within k moves.

Over the course of training, our agent adds each experience (s, a, r, s ) to its experience replay buffer.

As each catastrophe is reached at the n th turn of an episode, we add the k r (fear radius) states leading up to the catastrophe to a list of danger states.

We add the preceding n ??? k r states to a list of safe states.

When n < k r , all states for that episode are added to the list of danger states.

Then after each turn, in addition to making one update to the Q-network, we make one mini-batch update to the fear model.

To make this update, we sample 50% of samples in the batch from the danger states, assigning them label 1 and the remaining 50% from the safe states, assigning them label 0.For each update to the DQN, we perturb the TD target y t .

Instead of updating Q(s t , a t ; ?? Q ) towards r t + max a Q(s t+1 , a ; ?? Q ), we introduce the intrinsic fear to the model via the target: With probability select random action a t DISPLAYFORM0

Otherwise, select action a t = argmax a Q(s t , a ; ?? Q )

Execute action a t in environment, observing reward r t and successor state s t+1 10: DISPLAYFORM0 if s t+1 is a catastrophe state then Sample random minibatch of transitions (s ?? , a ?? , r ?? , s ?? +1 ) from D 16: DISPLAYFORM1 Sample random mini-batch s j with 50% of examples from D D and 50% from D S 20: DISPLAYFORM2 where F (s; ?? F ) is the fear model and ?? is a fear factor determining the scale of the impact of intrinsic fear on the Q-function update.

Note that IF perturbs the objective function.

Thus, one might be concerned that the perturbed reward might indicate a different optimal policy.

Fortunately, if the labeled catastrophe states and danger zone do not violate our assumptions, and if the fear model reaches arbitrarily high accuracy, then this will not happen.

For an MDP, M = S, A, T , R, ?? , with 0 ??? ?? ??? 1, the average reward return is as follows: DISPLAYFORM3 The optimal policy ?? * of the model M is the policy which maximizes the average reward return, ?? * = max ?????P ??(??) where P is a set of stationary polices.

Theorem 1.

For a given MDP, M , with ?? ??? [0, 1] and a catastrophe detector f , let ?? * denote an optimal policy of M , and?? denote an optimal policy of M equipped with fear model F and ??.

If the probability ?? * visits the states in the danger zone is at most , and R min ??? R(s, a) ??? R max , then DISPLAYFORM4 Proof.

Appendix A.It is worth noting that when at least one of the optimal policies of M , does not visit the fear zone ( = 0), then ?? * M = ?? M,F (??) and the fear signal can boost up the process of learning the optimal policy.

Since we learn the catastrophe detector f and fear model F empirically using the collected data, our RL agent has access to an imperfect detectorf and imperfect fear modelF , and therefore assumes the fear model isF .

In this case, the RL agent trains with intrinsic fear generated byf , learning a different value function than the RL agent with perfect f .

To show robustness against modeling errors, we are interested in the average deviation in the value functions of the two agents.

In general, in practical RL problems, we use discount factors ?? < 1 ( BID14 in order to reduce the planing horizon, and computation cost.

Moreover, BID12 suggests that when we have estimation (up to the confidence intervals) of our MDP model, it is better to use smaller discount factors in order to prevent over-fitting to the estimated model.

We show that under modeling errors, if the actual objective function to optimize for Eq. 4 has with discount factor ?? eval , it's better to use some ?? ??? ?? eval because it reduces the average deviation in the value functions.

For a given environment, with fear model F 1 and discount factor ?? 1 , let V ?? * F 2 ,?? 2 F1,??1 (s), s ??? S, denote the state value function under the optimal policy of a environment with fear model F 2 and the discount factor ?? 2 .

On the same environment, let ?? DISPLAYFORM5 (s) denote the stationary distribution over states.

Therefore we are interested in the average deviation on value functions caused by imperfect classifier: DISPLAYFORM6 Theorem 2.

For a given MDP model, the average deviation on the value functions, L(F, F , ?? eval , ??), F,F ??? F, vanishes as the number of samples N increases DISPLAYFORM7 with probability at least 1 ??? ??.

VC(F) is the VC dimension of the hypothesis class F.Proof.

Appendix B Thm.

2, holds for both tabular MDPs and continuous state-action MDPs.

In addition to proofs of these results, we provide a deeper theoretical analysis on deterministic and stochastic fear models in the tabular setting in Appendix B.Over the course of our experiments, we discovered the following pattern: Intrinsic fear models are more effective when the fear radius k r is large enough that the model can experience danger states at a safe distance and correct the policy, without experiencing many catastrophes.

When the fear radius is too small, the danger probability is only nonzero at states from which catastrophes are inevitable anyway and intrinsic fear seems not to help.

We also found that wider fear factors train more stably when phased in over the course of many episodes.

So, in all of our experiments we gradually phase in the fear factor ?? from 0 to ?? reaching full strength at predetermined time step k ?? .

In our Cart-Pole experiments, we phase ?? in over 1M steps.

We demonstrate our algorithms on three environments.

These include Adventure Seeker, a toy pathological environments which we designed to demonstrate the Sisyphean curse; Cartpole, a classic reinforcement learning environment; and three Atari games, Seaquest, Asteroids, and Freeway, simulated in the Arcade Learning Environment BID1 .Adventure Seeker We imagine a player placed on a hill, sloping upward to the right FIG1 .

At each turn, the player can move to the right (up the hill) or left (down the hill).

The environment adjusts the player's position accordingly, adding some random noise.

Between the left and right edges of the hill, the player gets more reward for spending time higher on the hill.

But if the player goes too far to the right, he/she will fall off (a catrastrophic state), terminating the episode and receiving a return of 0.

state s t , T (s t+1 |s t , a t ) gives successor state s t+1 ??? s t + .01 ?? a t + ?? where ?? ??? N (0, .01 2 ).

The reward at each turn is equal to s t (proportional to height).

The player falls off the hill, entering the catastrophic terminating state, whenever s t+1 > 1.0 or s t+1 < 0.0.This game admits an obvious analytic solution; There exists some threshold above which the agent should always choose to go left, and below which it should always go right.

And yet a state-of-the-art DQN model learning online or with experience replay successively plunges to its death.

To be clear, the DQN does learn a near-optimal thresholding policy quickly.

But over the course of continued training, the agent oscillates between a reasonable thresholding policy and one which always moves right, regardless of the state.

The pace of this oscillation evens out and all networks (over multiple runs) quickly reach a constant catastrophe per turn rate that does not attenuate with continued training.

How could we trust a system that can't solve Adventure Seeker to make consequential decisions?Cart-Pole In this classic RL environment, an agent balances a pole atop a cart FIG1 .

Qualitatively, the game exhibits four distinct catastrophe modes.

The pole could fall down to the right or fall down to the left.

Additionally, the cart could run off the right boundary of the screen or run off the left.

Formally, at each time, the agent observes a four-dimensional state vector (x, v, ??, ??) consisting respectively of the cart position, cart velocity, pole angle, and the pole's angular velocity.

At each time step, the agent chooses an action, applying a force of either ???1 or +1.

For every time step that the pole remains upright and the cart remains on the screen, the agent receives a reward of 1.

If the pole falls, the episode terminates, giving a return of 0 from the penultimate state.

In experiments, we use the implementation CartPole-v0 contained in the openAI gym BID4 .

Like Adventure Seeker, this problem admits an analytic solution.

A perfect policy should never drop the pole.

But, as with Adventure Seeker, a DQN converges to a constant rate of catastrophes per turn.

Atari games In addition to these pathological cases, we address Freeway, Asteroids, and Seaquest, games from the Atari Learning Environment.

In Freeway, the agent controls a chicken with a goal of crossing the road while dodging traffic.

The chicken loses a life and starts from the original location if hit by a car.

Points are only rewarded for successfully crossing the road.

In Asteroids, the agent pilots a ship and gains points from shooting the asteroids.

She must avoid colliding with asteroids which cost it lives.

In Seaquest, a player swims under water.

Periodically, as the oxygen gets low, she must rise to the surface for oxygen.

Additionally, fishes swim across the screen.

The player gains points each time she shoots a fish.

Colliding with a fish or running out of oxygen result in death.

In all three games, the agent has 3 lives, and the final death is a terminal state.

We label each loss of a life as a catastrophe state.

To assess the effectiveness of the intrinsic fear model, we evaluate both a standard DQN (DQNNoFear) and one enhanced by intrinsic fear (DQN-Fear).

In both cases, we use multilayer perceptrons (MLPs) with a single hidden layer and 128 hidden nodes.

We train all MLPs by stochastic gradient descent using the Adam optimizer BID13 to adaptively tune the learning rate.

Because, for the Adventure Seeker problem, an agent can escape from danger with only a few time steps of notice, we set the fear radius k r to 5.

We phase in the fear factor quickly, reaching full strength in just 1000 moves.

On this problem we set the fear factor ?? to 40.For Cart-Pole, we set a wider fear radius of k r = 20.

We initially tried training this model with a shorter fear radius but made the following observation.

Some models would learn well surviving for millions of experiences, with just a few hundred catastrophes.

This compared to a DQN ( FIG2 ) which would typically suffer 4000-5000 catastrophes.

When examining the output from the fear models on successful vs unsuccessful runs, we noticed that the unsuccessful models would output danger of probability greater than .5 for precisely the 5 moves before a catastrophe.

But by that time it would be too late for an agent to correct course.

In contrast, on the more successful runs, the fear model typically outputs predictions in the range .1 ??? .5.

We suspect that the gradation between mildly dangerous states and those with imminent danger provides a richer reward signal to the DQN.On both the Adventure Seeker and Cart-Pole environments, the DQNs augmented by intrinsic fear far outperform their otherwise identical counterparts FIG2 ).

We cannot plot the reward per episode for the intrinsic fear models on these environments because after the first several deaths, the episodes never terminate.

In contrast, both the DQN and related approaches like expected SARSA continue to visit the catastrophic states regularly.

We compared our approach against some traditional approaches for mitigating catastrophic forgetting.

For example, we tried a memory-based method in which we preferentially sample the catastrophic states for updating the model, but they did not improve over the DQN.

It seems that the notion of a danger zone is necessary here.

For Seaquest, Asteroids, and Freeway, we use a fear radius of 5 and a fear factor of .5.

For all Atari games, the IF models outperform their DQN counterparts.

Interestingly while for all games, the IF models achieve higher reward, on Seaquest, models trained with Intrinsic Fear have similar catastrophe rates.

More precisely, they appear to have fewer catastrophes early on but eventually enter a different reward regime, exchanging more catastrophes for higher reward.

This result suggests an interplay between the various reward signals that warrants further exploration.

For Asteroids and Freeway, the improvements are more dramatic.

Over just a few thousand episodes of Freeway, a randomly exploring DQN achieves zero reward.

However, the reward shaping of intrinsic fear leads to rapid improvement.

The paper addresses safety in RL, intrinsically motivated RL, and the stability of Q-learning with function approximation under distributional shift.

Our work also has some connection to reward shaping.

We attempt to highlight the most relevant papers here.

Several papers address safety in RL.

BID7 provide a thorough review on the topic, identifying two main classes of methods: those that perturb the objective function and those that use external knowledge to improve the safety of exploration.

While a typical reinforcement learner optimizes expected return, some papers suggest that a safely acting agent should also minimize risk.

BID10 ) defines a fatality as any return below some threshold ?? .

They propose a solution comprised of a safety function, which identifies unsafe states, and a backup model, which navigates away from those states.

Their work, which only addresses the tabular setting, suggests that an agent should minimize the probability of fatality instead of maximizing the expected return.

BID11 suggests an alternative Q-learning objective concerned with the minimum (vs expected) return.

Other papers suggest modifying the objective to penalize policies with high-variance returns BID7 .

Maximizing expected returns while minimizing their variance is a classic problem in finance, where a common objective is the ratio of expected return to its standard deviation BID27 . (Moldovan and Abbeel, 2012) gives a definition of safety based on ergodicity.

They consider a fatality to be a state from which one cannot return to the start state.

Shalev-Shwartz et al. FORMULA1 theoretically analyzes how strong a penalty should be to discourage accidents.

They also consider hard constraints to ensure safety.

None of the above works address the case where distributional shift dooms an agent to perpetually revisit known catastrophic failure modes.

Other papers incorporate external knowledge into the exploration process.

Typically, this requires access to an oracle or extensive prior knowledge of the environment.

In the extreme case, some papers suggest confining the policy search to the subset of policies known to be safe.

For reasonably complex environments or classes of policies this seems infeasible.

The potential oscillatory or divergent behavior of Q-learners with function approximation has been previously identified BID3 BID0 BID8 .

Outside of RL, the problem of covariate shift has been extensively studied BID29 .

BID22 addresses the problem of catastrophic forgetting owing to distributional shift in RL with function approximation, proposing a memory-based solution.

Many papers address intrinsic rewards, which are internally assigned, vs the standard (extrinsic) reward.

Typically, intrinsic rewards are used to encourage exploration BID25 BID2 and to acquire a modular set of skills BID5 .

Some papers refer to the intrinsic reward for discovery as curiosity.

Like classic work on intrinsic motivation, our methods perturb the reward function.

But instead of assigning bonuses to encourage discovery of novel transitions, we assign penalties to discourage catastrophic transitions.

Key differences In this paper, we undertake a novel treatment of safe reinforcement learning, While the literature offers several notions of safety in reinforcement learning, we see the following problem: Existing safety research that perturbs the reward function requires little foreknowledge, but fundamentally changes the objective globally.

On the other hand, processes relying on expert knowledge may presume an unreasonable level of foreknowledge.

Moreover, little of the prior work on safe reinforcement learning, to our knowledge, specifically addresses the problem of catastrophic forgetting.

This paper proposes a new class of algorithms for avoiding catastrophic states and a theoretical analysis supporting its robustness.

Our experiments demonstrate that DQNs are susceptible to periodically repeating mistakes, however bad, raising questions about their real-world utility when harm can come of actions.

While it's easy to visualize these problems on toy examples, similar dynamics are embedded in more complex domains.

Consider a domestic robot acting as a barber.

The robot might receive positive feedback for giving a closer shave.

This reward encourages closer contact at a steeper angle.

Of course, the shape of this reward function belies the catastrophe lurking just past the optimal shave.

Similar dynamics might be imagines in a vehicle that is rewarded for traveling faster but could risk an accident with excessive speed.

Our results with the intrinsic fear model suggest that with only a small amount of prior knowledge (the ability to recognize catastrophe states after the fact), we can simultaneously accelerate learning and avoid catastrophic states.

This work represents a first step towards combating some issues relating to safety in RL stemming from catastrophic forgetting.

The average return of the reward under a policy ?? is as follows: DISPLAYFORM0 Let us assume that any stationary policy ?? induces a stationary distribution ?? ?? (s), s ??? S.

Therefore we can rewrite Eq. 7 in terms of stationary distribution BID24 .

DISPLAYFORM1 In RL, we are interested in a policy ?? * that maximizes the expected average reward: DISPLAYFORM2 .

In a first place, the optimization in Eq. 7 looks linear in ?? but actually the policy ?? derives the stationary distribution ?? ?? (??), which makes the optimization problem harder.

Given the policy ??, let's define the joint distribution in (s, a) as follows: DISPLAYFORM3 Then we can rewrite the optimization problem in terms of the joint probability distribution ?? ?? .

DISPLAYFORM4 We can see that this new formalization, turns our optimization objective into a linear function of ?? ?? .Since ?? ?? is a join distribution of (s, a) under the model dynamics T , it can not take any arbitrary value.

Let ??? denote the set of feasible value for ?? ?? , then DISPLAYFORM5 T (s |s, a)??(s, a), ???s ??? S} .Clearly, ??? is a polytope on the simplex in R S??A .

Now, we can rewrite the optimization problem Eq. 7 as a constrained linear program on ?? ?? : DISPLAYFORM6 This change of variable allows us to analyze the introduction of intrinsic fear in different situations.

Among all the optimal policies of Eq. 7, consider the one which minimizes := s???C,a ?? ?? * (s, a).let's assume that the negative reward assigned to the states in the danger zone is ??(R max ??? R min ) and the optimal policy?? of the environment with the intrinsic fear has return of ?? M,F (??).Applying policy ?? * on the environment with intrinsic fear gives a return of ?? * M ??? ?? (R max ??? R min ).

Therefore, the return of?? on the environment with intrinsic fear, ?? M,F , is lower bounded by ?? * M ??? ?? (R max ??? R min ).

Therefore, applying?? on the environment without intrinsic fear gives a return of ?? M (??) which is lower bounded by ?? * DISPLAYFORM7 A.1 DISCOUNTED CUMULATIVE REWARD For the ??-discounted setting, we are interested in DISPLAYFORM8 The above mentioned equations hold in this setting as well, i.e., ?? * DISPLAYFORM9

In the previous section, we assumed that we have access to the perfect classifier F which can exactly label the danger zone.

This assumption does not hold in real world where we train the classifier.

In this section we derive an analysis in order to show that imperfect classifier F can not change the overall performance by much.

In general, in practical RL problems, we use discount factors ?? eval < 1 ( BID14 in order to reduce the planing horizon, and computation cost.

Moreover, BID12 suggest that when we have an estimation (up to the confidence intervals) of our MDP model, it is better to use ?? ??? ?? eval .

They show that since larger discount factor enriches the class of optimal policies for a given set of plausible models, large discount factors enrich models and end up over fitting to the noisy estimate of the environment.

In this section, we show how to choose the discount factor ?? ??? ?? eval such that the learned Value function stays close to the Value function under the perfect classifier F is perfect.

Let, V ?? * F 1 ,?? 1 F2,??2 (s), s ??? S, denote the state value under the optimal policy of model with classifier F 1 under the discount factor ?? 1 on the environment equipped with classifier F and discount factor ?? 2 .

On the same environment, ?? DISPLAYFORM0

(s) denotes the stationary distribution over states.

We are interested in the average deviation on value functions caused by the imperfect classifier: DISPLAYFORM0 This quantity can be upper bounded by DISPLAYFORM1 The goal is to find an ?? * which minimizes this loss, i.e. ?? * = argmin ??????? eval L with high probability.

For simplicity and without loss of generality, let's assume that all the rewards, including the intrinsic fears, are in [0, 1] and call ?? , the transformed version of ?? 2 .

One can decompose the upper bound in Eq. 12 as follows: DISPLAYFORM2 The first term is the deviation on value function when applying same policy on the same environment but with different discount factors.

Since ?? ??? ?? eval we have V DISPLAYFORM3 The second part of Eq. 13 is the deviation in value function under different policies and different DISPLAYFORM4 where the last inequality is due to the optimality of ?? * F,?? on the environment of F, ??.

To bound this part we exploit the proof trick used in BID12 .

DISPLAYFORM5 since the middle term is negative we have DISPLAYFORM6 This quantity V ?? F ,?? DISPLAYFORM7 is the difference between the performance of the same policy on two different environments.

These two values functions should satisfy the following bellman equations: As i tends to infinity, these two dynamics updates converge to V DISPLAYFORM8 As i tends to infinity, we have (1 ??? ??)

If we consider the fear model as a lookup table, and deterministic, then observing each state once is enough to exactly recover the classifier.

with probability ?? where N (s) is the number visits to a state s at time step N .

The trajectory produced by algorithm does not produce i.i.d.

samples of state.

Therefore, for Eq. 20 we use Hoeffding's inequality accompanied with union bound over time N .

In order to have this bound to hold for all the states at once, we need another union bounds over states and all possibly optimal policies ?? ?? under noisy classifier , which requires to replace ?? ??? ??/SA?? ?? .

Let's assume a minimum number of visit N to each state, DISPLAYFORM0 Finally, adding Eq. 14 and Eq. 21, the upper bound on L is as follows: DISPLAYFORM1

Let F denote a set of given binary classifiers and F ??? F. In this case, let's assume that we are given a set of N i.i.d samples from the stationary distribution ?? ?? * F ,??

.

Given a policy ??, the MDP transition process reduces to a Markov chain with transition probability T ?? .

Now we rewrite the Eq. 19 in a matrix format where V ?? i , F ??? R S are vectors of concatenation of V ?? i (s) and F (s), ???s ??? S respectively.

DISPLAYFORM0 as i goes to infinity we have DISPLAYFORM1 Using PAC analysis of binary classification in BID9 a follow up to BID32 , we have DISPLAYFORM2 ??? 3200 VC(F) + log 1 ?? N with probability at least 1 ??? ?? where VC(F) is the VC dimension of the hypothesis class and | ?? | is entry-wise absolute value.

Since ?? < 1, then ?? max , the maximum eigenvalue of (1 ??? ??T ?? ) ???1 , is bounded above and we have The same analysis, up to a slight modification 3 , holds for the continuous state and action spaces.

@highlight

Shape reward with intrinsic motivation to avoid catastrophic states and mitigate catastrophic forgetting.

@highlight

An RL algorithm that combines the DQN algorithm with a fear model trained in parallel to predict catastropohic states.

@highlight

The paper studies catastrophic forgetting in RL, by emphasizing tasks where a DQN is able to learn to avoid catastrophic events as long as it avoids forgetting.
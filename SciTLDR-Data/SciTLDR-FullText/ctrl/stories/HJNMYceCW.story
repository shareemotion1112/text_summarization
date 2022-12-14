We consider reinforcement learning and bandit structured prediction problems with very sparse loss feedback: only at the end of an episode.

We introduce a novel algorithm, RESIDUAL LOSS PREDICTION (RESLOPE), that solves such problems by automatically learning an internal representation of a denser reward function.

RESLOPE operates as a reduction to contextual bandits, using its learned loss representation to solve the credit assignment problem, and a contextual bandit oracle to trade-off exploration and exploitation.

RESLOPE enjoys a no-regret reduction-style theoretical guarantee and outperforms state of the art reinforcement learning algorithms in both MDP environments and bandit structured prediction settings.

Current state of the art learning-based systems require enormous, costly datasets on which to train supervised models.

To progress beyond this requirement, we need learning systems that can interact with their environments, collect feedback (a loss or reward), and improve continually over time.

In most real-world settings, such feedback is sparse and delayed: most decisions made by the system will not immediately lead to feedback.

Any sort of interactive system like this will face at least two challenges: the credit assignment problem (which decision(s) did the system make that led to the good/bad feedback?); and the exploration/exploitation problem (in order to learn, the system must try new things, but these could be bad).We consider the question of how to learn in an extremely sparse feedback setting: the environment operates episodically, and the only feedback comes at the end of the episode, with no incremental feedback to guide learning.

This setting naturally arises in many classic reinforcement learning problems ( §4): a barista robot will only get feedback from a customer after their cappuccino is finished 1 .

It also arises in the context of bandit structured prediction BID41 BID9 ( §2.2), where a structured prediction system must produce a single output (e.g., translation) and observes only a scalar loss.

We introduce a novel reinforcement learning algorithm, RESIDUAL LOSS PREDICTION (RESLOPE) ( § 3), which aims to learn effective representations of the loss signal.

By effective we mean effective in terms of credit assignment.

Intuitively, RESLOPE attempts to learn a decomposition of the episodic loss into a sum of per-time-step losses.

This process is akin to how a person solving a task might realize before the task is complete when and where they are likely to have made suboptimal choices.

In RESLOPE, the per-step loss estimates are conditioned on all the information available up to the current point in time, allowing it to learn a highly non-linear representation for the episodic loss (assuming the policy class is sufficiently complex; in practice, we use recurrent neural network policies).

When the system receives the final episodic loss, it uses the difference between the observed loss and the cumulative predicted loss to update its parameters.

Algorithmically, RESLOPE operates as a reduction ( §3.3) to contextual bandits (Langford & Zhang, 2008) , allowing the bandit algorithm to handle exploration/exploitation and focusing only on the credit assignment problem.

RESIDUAL LOSS PREDICTION is theoretically motivated by the need for variance reduction techniques when estimating counterfactual costs (Dudík et al., 2014) and enjoys a no-regret bound ( §3.3) when the underlying bandit algorithm is no-regret.

Experimentally, we show the efficacy of RESLOPE on four benchmark reinforcement problems and three bandit structured prediction problems ( § 5.1), comparing to several reinforcement learning algorithms: Reinforce, Proximal Policy Optimization and Advantage Actor-Critic.

We focus on finite horizon, episodic Markov Decision Processes (MDPs) in this paper, which captures both traditional reinforcement learning problems ( § 4) and bandit structured prediction problems ( § 2.2).

Our solution to this problem, RESIDUAL LOSS PREDICTION ( § 3) operates in a reduction framework.

Specifically, we assume there exists "some" machine learning problem that we know how to solve, and can treat as an oracle.

Our reduction goal is to develop a procedure that takes the reinforcement learning problem described above and map it to this oracle, so that a good solution to the oracle guarantees a good solution to our problem.

The specific oracle problem we consider is a contextual bandit learning algorithm, relevant details of which we review in §2.1.Formally, we consider a (possibly virtual) learning agent that interacts directly with its environment.

The interaction between the agent and the environment is governed by a restricted class of finitehorizon Markov Decision Processes (MDP), defined as a tuple {S, s 0 , A, P, L, H} where: S is a large but finite state space, typically S ⊂ R d ; s 0 ∈ S is a start state; A is a finite action space 2 of size K; P = { P(s |s, a) : s, s ∈ S, a ∈ A } is the set of Markovian transition probabilities; L ∈ R |S| is the state dependent loss function, defined only at terminal states s ∈ S; H is the horizon (maximum length of an episode).The goal is to learn a policy π, which defines the behavior of the agent in the environment.

We consider policies that are potentially functions of entire trajectories 3 , and potentially produce distributions over actions: π(s) ∈ ∆ A , where ∆ A is the A-dimensional probability simplex.

However, to ease exposition, we will present the background in terms of policies that depend only on states; this can be accomplished by simply blowing up the state space.

Let d π h denote the distribution of states visited at time step h when starting at state s 0 and operating according to π: d π h+1 (s ) = E s h ∼d π h ,a h ∼π(s h ) P(s | s = s h , a = a h ) The quality of the policy π is quantified by its value function or q-value function: V π (s) ∈ R associates each state with the expected future loss for starting at this state and following π afterwards; Q π (s, a) ∈ R associates each state/action pair with the same expected future loss: DISPLAYFORM0 The learning goal is to estimate a policy π from a hypothesis class of policies Π with minimal expected loss: J(π) = V π (s 0 ).

The contextual bandit learning problem (Langford & Zhang, 2008) can be seen as a tractable special case of reinforcement learning in which the time horizon H = 1.

In particular, the world operates episodically.

At each round t, the world reveals a context (i.e. feature vector) x t ∈ X ; the system chooses an action a t ; the world reveals a scalar loss t (x t , a t ) ∈ R + , a loss only for the selected action that may depend stochastically on x t and a t .

The total loss for a system over T rounds is the sum of losses: T t=1 t (x t , a t ).

The goal in policy optimization is to learn a policy π : x → A from a policy class Π that has low regret with respect to the best policy in this class.

Assuming the learning algorithm produces a sequence of policies π 1 , π 2 , . . .

, π T , its regret is: DISPLAYFORM0 The particular contextual bandit algorithms we will use in this paper perform a second level of reduction: they assume access to an oracle supervised learning algorithm that can optimize a cost-sensitive loss (Appendix C), and transform the contextual bandit problem to a cost-sensitive classification problem.

Algorithms in this family typically vary along two axes: how to explore (faced with a new x how does the algorithm choose which action to take); and how to update (Given the observed loss t , how does the algorithm construct a supervised training example on which to train).

More details are in Appendix A.

In structured prediction, we observe structured input sequences x SP ∈ X and the goal is to predict a set of correlated output variables y SP ∈ Y. For example, in machine translation, the input x SP is a sentence in an input language (e.g., Tagalog) and the output y SP is a sentence in an output language (e.g., Chippewa).

In the fully supervised setting, we have access to samples (x SP , y SP ) from some distribution D over input/output pairs.

Structured prediction problems typically come paired with a structured loss (y SP ,ŷ SP ) ∈ R + that measures the fidelity of a predicted outputŷ SP to the "true" output y SP .

The goal is to learn a function f : X → Y with low expected loss under D: DISPLAYFORM0 Recently, it has become popular to solve structured prediction problems incrementally using some form of recurrent neural network (RNN) model.

When the output y SP contains multiple parts (e.g., words in a translation), the RNN can predict each word in sequence, conditioning each prediction on all previous decisions.

Although typically such models are trained to maximize cross-entropy with the gold standard output (in a fully supervised setting), there is mounting evidence that this has similar drawbacks to pre-RNN techniques, such as overfitting to gold standard prefixes (the model never learns what to do once it has made an error) and sensitivity to errors of different severity (due to error compounding).

In order to achieve this we must formally map from the structured prediction problem to the MDP setting; this mapping is natural and described in detail in Appendix B.Our focus in this paper is on the recently proposed bandit structured prediction setting BID9 BID41 , at training time, we only have access to input x SP from the marginal distribution D X .

For example, a Chippewa speaker sees an article in Tagalog, and asks for a translation.

A system then produces a single translationŷ SP , on which a single "bandit" loss (ŷ SP | x SP ) is observed.

Given only this bandit feedback, without ever seeing the "true" translation, the system must learn.

Our goal is to learn a good policy in a Markov Decision Process ( § 2) in which losses only arrive at the end of episodes.

Our solution, RESIDUAL LOSS PREDICTION (RESLOPE), automatically deduces per-step losses based only on the episodic loss.

To gain an intuition for how this works, suppose you are at work and want to meet a colleague at a nearby coffee shop.

In hopes of finding a more efficient path to the coffee shop, you take a different path than usual.

While you're on the way, you run into a friend and talk to them for a few minutes.

You then arrive at the coffee shop and your colleague tells you that you are ten minutes late.

To estimate the value of the different path, you wonder: how much of this ten minutes is due to taking the different path vs talking to a friend.

If you can accurately estimate that you spent seven minutes talking to your friend (you lost track of time), you can conclude that the disadvantage for the different path is three minutes.

RESLOPE addresses the problem of sparse reward signals and credit assignment by learning a decomposition of the reward signal, essentially doing automatic reward shaping (evaluated in §5.3).

Finally, it addresses the problem of exploration vs exploitation by relying on a strong underlying contextual bandit learning algorithm with provably good exploration behavior.

Akin to the coffee shop example, RESLOPE learns a decomposition of the episodic loss (i.e total time spent from work to the coffee shop) into a sum of per-time-step losses (i.e. timing activities along the route).

RESLOPE operates as a reduction from reinforcement learning with episodic loss to contextual bandits.

In this way, RESLOPE solves the credit assignment problem by predicting residual losses, and relies on the underlying contextual bandit oracle to solve explore/exploit.

RES-LOPE operates online, incrementally updating a policy π learn once per episode.

In the structured

Figure 1: RESIDUAL LOSS PREDICTION: the system assigns a part-of-speech tag sequence to the sentence "International Conference for Learning Representations".

Each state represents a partial labeling.

The end state e = [Noun, Noun, Preposition, Verb, Noun].

The end state e is associated with an episodic loss (e), which is the total hamming loss in comparison to the optimal output structure e * = [Adjective, Noun, Preposition, Noun, Noun].

We emphasize that our algorithm doesn't assume access to neither the optimal output structure, nor the hamming loss for every time step.

Only the total hamming loss is observed in this case ( (e) = 2).contextual bandit setting, we assume access to a reference policy, π ref , that was perhaps pretrained on supervised data, and which we wish to improve; a hyperparameter β controls how much we trust π ref .As π learn improves, we replace π ref with π learn .

In the RL setting, we set β = 0.We initially present a simplified variant of RESLOPE that mostly follows the learned policy (and the reference policy as appropriate), except for a single deviation per episode.

This algorithm closely follows the bandit version of the Locally Optimal Learning to Search (LOLS) approach of BID9 , with three key differences: (1) residual loss prediction; (2) alternative exploration strategies; (3) alternative parameter update strategies.

We assume access to a contextual bandit oracle CB that supports the following API:1.

CB.ACT(π learn , x), where x is the input example; this returns a tuple (a, p), where a is the selected action, and p is the probability with which that action was selected.2.

CB.COST(π learn , x, a) returns the estimated cost of taking action a in the context.3.

CB.UPDATE(π learn , x, a, p, c), where x is the input example, a ∈ [K] is the selected action, p ∈ (0, 1] is the probability of that action, and c ∈ R is the target cost.

The requirement that the contextual bandit algorithm also predicts costs (CB.COST) is somewhat non-standard, but is satisfied by many contextual bandit algorithms in practice, which often operate by regressing on costs and picking the minimal predicted cost action.

We describe the specific contextual bandit approaches we use in §3.2.Algorithm 1 shows how our reduction is constructed formally.

It uses a MAKEENVIRONMENT(t) function to construct a new environment (randomly in RL and by selecting the tth example in bandit structured prediction).

To learn a good policy, RESLOPE reduces long horizon trajectories to singlestep contextual bandit training examples.

In each episode, RESLOPE picks a single time step to deviate.

Prior to the deviation step, it executes π learn as a roll-in policy and after the deviation step, it executes a β mixture of π learn and π ref ( Figure 5 ).

At the deviation step, it calls CB.ACT to handle the exploration and choose an action.

At every step, it calls CB.COST to estimate the cost of that action.

Finally, it constructs a single contextual bandit training example for the deviation step, whose input was the observation at that step, whose action and probability are those that were selected by CB.ACT, and whose cost is the observed total cost minus the cost of every other action taken in this trajectory.

This example is sent to CB.UPDATE.

When the contextual bandit policy is an RNN (as in our setting), this will then compute a loss which is back-propagated through the RNN.

Choose rollout policy π mix to be π ref with probability β or π learn t−1 with probability 1 − β 8:for all time steps h = 1 . . .

env.

H do 9:x ← env.

STATEFEATURES {computed by an RNN} 10: DISPLAYFORM0 a ← a

The contextual bandit oracle receives examples where the cost for only one predicted action is observed, but no others.

It learns a policy for predicting actions minimizing expected loss by estimating the unobserved target costs for the unpredicted actions and exploring different actions to balance the exploitation exploration trade-off ( § 3.2).

The contextual bandit oracle then calls a cost-sensitive multi-class oracle (Appendix C) given the target costs and the selected action.

CB.UPDATE: Cost Estimation Techniques.

The update procedure for our contextual bandit oracles takes an example x, action a, action probability p and cost c as input and updates its policy.

We do this by reducing to a cost-sensitive classification oracle (Appendix C), which expects an example x and a cost vector y ∈ R K that specifies the cost for all actions (not just the selected one).

The reduction challenge is constructing this cost-sensitive classification example given the input to CB.UPDATE.

We consider three methods: inverse propensity scoring BID18 , doubly robust estimation (Dudík et al., 2014) and multitask regression (Langford & Agarwal, 2017) .Inverse Propensity Scoring (IPS): IPS uses the selected action probability p to correct for the shift in action proportions predicted by the policy π learn .

IPS estimates the target cost vector y as: DISPLAYFORM0 , where 1 is an indicator function and where a is the selected action and c is the observed cost.

While IPS yields an unbiased estimate of costs, it typically has a large variance as p → 0.

The doubly robust estimator uses both the observed cost c as well as its own predicted costsŷ(i) for all actions, forming a target that combines these two sources of information.

DR estimates the target cost vector y as: DISPLAYFORM0 The DR estimator remains unbiased, and the estimated loss y helps decrease its variance.

The multitask regression estimator functions somewhat differently from IPS and DR.

Instead of reducing to to cost-sensitive classification, MTR reduces directly to importance-weighted regression.

MTR maintains K different regressors for predicting costs given input/action pairs.

Given x, a, c, p, MTR constructs a regression example, whose input is (x, a), whose target output is c and whose importance weight is 1/p.

Uniform: explores randomly with probability and otherwise acts greedily BID42 .Boltzmann: varies action probabilities where action a is chosen with probability proportional to exp[−c(a)/temp], where temp ∈ R + is the temperature, and c(a) is the predicted cost of action a.

Bootstrap Exploration: BID0 trains a bag of multiple policies simultaneously.

Each policy in the bag votes once on its predicted action, and an action is sampled from this distribution.

To train, each example gets passed to each policy Poisson(λ = 1)-many times, which ensures diversity .

Bootstrap can operate in "greedy update" and "greedy prediction" mode (Bietti et al., 2017) .

In greedy update, we always update the first policy in the bag exactly once.

In greedy prediction, we always predict the action from the first policy during exploitation.

For simplicity, we first consider the case where we have access to a good reference policy π ref but do not have access to good Q-value estimates under π ref .The only way one can obtain a Q-value estimate is to do a roll-out, but in a non-resettable environment, we can only do this once.

We will subsequently consider the case of suboptimal (or missing) reference policies, in which the goal of the analysis will change from competing with π ref to competing with both π ref and a local optimality guarantee.

Theorem 1.

Setting β = 1, running RESLOPE for N episodes with a contextual bandit algorithm, the average returned policyπ = E n π n has regret equal to the suboptimality of π ref , namely: DISPLAYFORM0 where CB (N ) is the cumulative regret of the underlying contextual bandit algorithm after N steps, and approx is an approximation error term that goes to zero as N → ∞ so long as the contextual bandit algorithm is no-regret and assuming all costs are realizable under the hypothesis class used by RESLOPE.In particular, when the problem is realizable and the contextual bandit algorithm is no-regret, RES-LOPE is also no-regret.

The realizability assumption is unfortunate, but does not appear easy to remove (see Appendix D for the proof).In the case that π ref is not known to be optimal, or not available, we follow the LOLS analysis and obtain a regret to a convex combination of π ref and the learned policy's one-step deviations (a form of local optimality) and can additionally show the following (proof in Appendix E): Theorem 2.

For arbitrary β, define the combined regret ofπ as: DISPLAYFORM1 The first term is suboptimality to π ref ; the second term is suboptimality to the policy's own realizable one-step deviations.

Given a contextual bandit learning algorithm, and under a realizability assumption, the combined regret ofπ satisfies: DISPLAYFORM2 Again, if the contextual bandit algorithm is no regret, then CB /N → 0 as N → ∞; see Appendix E for the proof.

Finally, we present the multiple deviation variant of RESLOPE.

Algorithm 2 shows how RESLOPE operates under multiple deviations.

The difference between the single and multiple deviation mode is twofold: 1.

Instead of deviating at a single time step, multi-dev RESLOPE performs deviations at each time step in the horizon; 2.

Instead of generating a single contextual bandit example per episode, multi-dev RESLOPE generates H examples, where H is the length of the time horizon, effectively updating the policy H times.

These two changes means that we update the learned policy π learn multiple times per episode.

Empirically, we found this to be crucial for achieving superior performance.

Although, the generated samples for the same episode are not independent, this is made-up for by the huge increase in the (a DISPLAYFORM0 env.

STEP(a dev h ) {updates environment and internal state of the RNN } 10:end for 11: DISPLAYFORM1 ) for all h 13: end for 14: Return average policyπ = 1 T t π learn t number of available samples for training (i.e. T×H samples for multiple deviations versus only T samples in the single deviation mode).

The theoretical analysis that precedes still holds in this case, but only makes sense when β = 0 because there is no longer any distinction between roll-in and roll-out, and so the guarantee reduces to a local optimality guarantee.

We conduct experiments on both reinforcement learning and structured prediction tasks.

Our goal is to evaluate how quickly different learning algorithms learn from episodic loss.

We implement our models on top of the DyNet neural network optimization package BID30 .

4 Reinforcement Learning Environments We perform experiments in four standard reinforcement learning environments:

Blackjack (classic card game), Hex (two-player board game), Cartpole (aka "inverted pendulum") and Gridworld.

Our implementations of these environments are described in Appendix F and largely follows the AI Gym BID7 implementations.

We report results in terms of cumulative loss, where loss is −1×reward for consistency with the loss-based exposition above and the loss-based evaluation of bandit structured prediction ( §2.2).

We also conduct experiments on structured prediction tasks.

The evaluation framework we consider is the fully online setup described in ( § 2.2), measuring the degree to which various algorithms can effectively improve by observing only the episodic loss, and effectively balancing exploration and exploitation.

We learn from one structured example at a time and we do a single pass over the available examples.

We measure performance in terms of average cumulative loss on the online examples as well as on a held-out evaluation dataset.

The loss on the online examples measures how much the algorithm is penalized for unnecessary exploration.

We perform experiments on the three tasks described in detail in Appendix G: English Part of Speech Tagging, English Dependency Parsing and Chinese Part of Speech Tagging.

We compare against three common reinforcement learning algorithms: Reinforce BID47 with a baseline whose value is an exponentially weighted running average of rewards; Proximal Policy Optimization (PPO) BID39 ; and Advantage Actor-Critic (A2C) BID26 .

For the structured prediction experiments, since the bandit feedback is simulated based on labeled data, we can also estimate an "upper bound" on performance by running a supervised learning algorithm that uses full information (thus forgoing issues of both exploration/exploitation and credit assignment).

We run supervised DAgger to obtain such an upper bound.

In all cases, our policy is a recurrent neural network BID14 ) that maintains a real-valued hidden state and combines: (a) its previous hidden state, (b) the features from the environment (described for each environment in the preceding sections), and (c) an embedding of its previous action.

These form a new hidden state, from which a prediction is made.

Formally, at time step h, v h is the hidden state representation, f (state h ) are the features from the environment and a h is the action taken.

The recursion is: DISPLAYFORM0 Here, A is a learned matrix, const is an initial (learned) state, emb is a (learned) action embedding function, and ReLU is a rectified linear unit applied element-wise.

Given the hidden state v h , an action must be selected.

This is done using a simple feedforward network operating on v h with either no hidden layers (in which case the output vector is o h = Bv h ) or a single hidden layer (where o h = B 2 ReLU(B 1 v h )).

In the case of RESLOPE and DAgger, which expect cost estimates as the output of the policy, the output values o h are used as the predicted costs (and a h might be the argmin of these costs when operating greedily).

In the case of Reinforce, PPO and A2C, which expect action probabilities, these are computed as softmax(−o h ) from which, for instance, an action a h is sampled.

Details on optimization, hyperparameters and "deep learning tricks" are reported in Appendix H.

We study several questions empirically: 1.

How does RESIDUAL LOSS PREDICTION compare to policy gradient methods on reinforcement learning and bandit structured prediction tasks? ( § 5.1) 2.

What's the effect of ablating various parts of the RESLOPE approach, including multiple deviations? ( §5.2) 3.

Does RESLOPE succeed in learning a good representation of the loss? ( §5.3)

In our first set of experiments, we compare RESLOPE to the competing approaches on the four reinforcement learning tasks described above.

FIG1 shows the results.

In Blackjack, Hex and Grid, RESLOPE outperforms all the competing approaches with lower loss earlier in the learning process (though for Hex and Grid they all finish at the same near-optimal policy).

For Cartpole, RESLOPE significantly underperforms both Reinforce and PPO.

5 Furthermore, in both Blackjack and Grid, the bootstrap exploration significantly improves upon Boltzmann exploration.

In general, both RESLOPE performs quite well.

In our second set of experiments, we compare the same algorithms plus the fully supervised DAgger algorithm on the three structured prediction problems; the results are in FIG2 .

Here, we can observe RESLOPE significantly outperforming all alternative algorithms (except, of course, DAgger) on training loss (also on heldout (development) loss; see Figure 9 in the appendix).

There is still quite a gap to fully supervised learning, but nonetheless RESLOPE is able to reduce training error significantly on all tasks: by over 25% on English POS, by about half on English dependency parsing, and by about 10% on Chinese POS tagging.

In our construction of RESLOPE, there are several tunable parameters: which contextual bandit learner to use (IPS, DR, MTR), which exploration strategy (Uniform, Boltzmann, Bootstrap), and, for Bootstrap, whether to do greedy prediction and/or greedy update.

In Table 1 (in the Appendix), we show the results on all tasks for ablating these various parameters.

For the purpose of the ablation, we fix the "baseline" system as: DR, Bootstrap, and with both greedy prediction and greedy updates, though this is not uniformly the optimal setting (and therefore these numbers may differ slightly from the preceding figures).

The primary take-aways from these results are: (1) MTR and DR are competitive, but IPS is much worse; (2) Bootstrap is much better than either other exploration method (especially uniform, not surprisingly); (3) Greedy prediction is a bit of a wash, with only small differences either way; (4) Greedy update is important.

In Appendix I, we consider the effect of single vs multiple deviations and observe that significant importance of multiple deviations for all algorithms, with Reinforce and PPO behaving quite poorly with only single deviations.

In our final set of experiments, we study RESLOPE's performance under different-and especially non-additive-loss functions.

Our goal is to investigate RESLOPE's ability to learn good representations for the episodic loss.

We consider the following different incremental loss functions for each time step: Hamming (0/1 loss at each position), Time-Sensitive (cost for an error at position h is equal to h) and Distance-Sensitive (cost for predictingâ instead of a is |â − a|).

To combine these per-stop losses into a per-trajectory loss τ of length H, we compute the H-dimensional loss vector suffered by RESLOPE along this trajectory.

To consider both additive and non-additive combinations, we consider Lp norms of this loss vector.

When the norm is L1, this is simple additive loss.

More generally, we consider (τ ) = p t=H t=1 p (t) for any p > 0.Reinforce.

We also have conducted experiments with PPO with larger minibatches; these results are reported in the appendix in FIG6 .

In those experiments, we adjusted the minibatch size and number of epochs to match exactly with the PPO algorithm described in BID39 .

In each iteration, each of N actors collect T timesteps of data.

Then we construct the surrogate loss on these NT time steps of data, and optimize it with minibatch Adam for K epochs.

With these adjustments, PPO's performance falls between RESLOPE and Reinforce on Blackjack, slightly superior to RESLOPE on Hex, better than everything on Cartpole, and roughly equivalent to RESLOPE on Gridworld.

We were, unfortunately, unable to conduct these experiments in the structured prediction setting, because the state memoization necessary to implement PPO with large/complex environments overflowed our system's memory quite quickly. .

The x-axis shows the number of episodes and the y-axis measures the incremental loss using the true loss function (light colors) and using RESLOPE (dark colors).

If RESLOPE worked perfectly, these would coincide.

We run six different experiments using different incremental and episodic loss functions.

For each incremental loss function (i.e. hamming, time sensitive, distance sensitive) we run two experiments: using the total hamming loss (additive) and an Lp norm of five (non-additive).

Results are presented in FIG3 .

We observe the following.

RESLOPE can always learn the optimal representation for the incremental loss when the episodic loss function is additive.

This is the case for all the three incremental loss functions: hamming, time sensitive, and distance sensitive.

Learning is faster when the episodic loss function is additive.

While RESLOPE is still able to learn a good representation even when using the L5 norm loss, this happens much later in comparison to the additive loss function (40k time steps for L5 norm vs 20k for total hamming loss).

Not surprisingly, performance degrades as the episodic loss function becomes non-additive.

This is most acute when using L-5 norm with the incremental hamming loss.

This is expected as in the distance and time sensitive loss functions, RESLOPE observes a smoother loss function and learns to distinguish between different time steps based on the implicit encoding of time and distance information in the observed loss.

RESLOPE can still learn a good representation for smoother episodic loss functions.

This is shown empirically for time and distance sensitive loss functions.

RESIDUAL LOSS PREDICTION builds most directly on the bandit learning to search frameworks LOLS BID9 and BLS BID40 .

The "bandit" version of LOLS was analyzed theoretically but not empirically in the original paper; BID40 found that it failed to learn empirically.

They addressed this by requiring additional feedback from the user, which worked well empirically but did not enjoy any theoretical guarantees.

RESLOPE achieves the best of both worlds: a strong regret guarantee, good empirical performance, and no need for additional feedback.

The key ingredient for making this work is using the residual loss structure together with strong base contextual bandit learning algorithms.

A number of recent algorithms have updated "classic" learning to search papers with deep learning underpinnings BID48 BID21 .

These aim to incorporate sequencelevel global loss function to mitigate the mismatch between training and test time discrepancies, but only apply in the fully supervised setting.

Mixing of supervised learning and reinforcement signals has become more popular in structured prediction recently, generally to do a better job of tuning for a task-specific loss using either Reinforce BID35 or Actor-Critic BID2 .

The bandit variant of the structured prediction problem was studied by BID41 , who proposed a reinforce method for optimizing different structured prediction models under bandit feedback in a log-linear structured prediction model.

A standard technique for dealing with sparse and episodic reward signals is reward shaping BID31 : supplying additional rewards to a learning agent to guide its learning process, beyond those supplied by the underlying environment.

Typical reward shaping is hand-engineered; RESLOPE essentially learns a good task-specific reward shaping automatically.

The most successful baseline approach we found is Proximal Policy Optimization (PPO, BID39 ), a variant of Trust Region Policy Optimization (TRPO, BID38 ) that is more practical.

Experimentally we have seen RESLOPE to typically learn more quickly than PPO.

Theoretically both have useful guarantees of a rather incomparable nature.

Since RESLOPE operates as a reduction to a contextual bandit oracle, this allows it to continually improve as better contextual bandit algorithms become available, for instance work of Syrgkanis et al. (2016b) and BID0 .

Although RESLOPE is quite effective, there are a number of shortcomings that need to be addressed in future work.

For example, the bootstrap sampling algorithm is prohibitive in terms of both memory and time efficiency.

One approach for tackling this would be using the amortized bootstrap approach by BID27 , which uses amortized inference in conjunction with implicit models to approximate the bootstrap distribution over model parameters.

There is also a question of whether the reduction to contextual bandits creates "reasonable" contextual bandit problems in conjunction with RNNs.

While some contextual bandit algorithms assume strong convexity or linearity, the ones we employ operate on arbitrary policy classes, provided a good cost-sensitive learner exists.

The degree to which this is true will vary by neural network architecture, and what can be guaranteed (e.g., no regret full-information online neural learning).

A more significant problem in the multi-deviation setting is that as RESLOPE learns, the residual costs will change, leading to a shifting distribution of costs; in principle this could be addressed using CB algorithms that work in adversarial settings BID43 BID16 , but largely remains an open challenge.

RESLOPE is currently designed for discrete action spaces.

Extension to continuous action spaces BID22 BID23 remains an open problem.

We thank Paul Mineiro and the anonymous reviewers 7 for very helpful comments and insights (especially to reviewer #3 whose patient comments on the analysis section of this paper were incredibly helpful 8 ).

We also thank Khanh Nguyen, Shi Feng, Kianté Brantley, Moustafa Meshry, and Sudha Rao for reviewing earlier drafts for this work and Alekh Agarwal, Nan Jiang, and Adith Swaminathan for helpful discussions and comments.

This work was partially funded by an Amazon Research Award.

This material is based upon work supported by the National Science Foundation under Grant No. 1618193.

Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

We assume that contexts are chosen i.i.d from an unknown distribution D(x), the actions are chosen from a finite action set A, and the distribution over loss D( |a, x) is fixed over time, but is unknown.

In this context, the key challenge in contextual bandit learning is the exploration/exploitation problem.

Classic algorithms for the contextual bandit problem such as EXP4.P BID5 can achieve a √ T regret bound; in particular: DISPLAYFORM0 where K = |A|.

When the regret is provably sublinear in T , such algorithms are often called "no regret" because their average regret per time step goes to zero as T → ∞.The particular contextual bandit algorithms we will use in this paper perform a second level of reduction: they assume access to an oracle supervised learning algorithm that can optimize a costsensitive loss, and transform the contextual bandit problem to a cost-sensitive classification problem.

Algorithms in this family typically vary along two axes:1.

How to explore?

I.e., faced with a new x how does the algorithm choose which action to take; 2.

How to update?

Given the observed loss t , how does the algorithm construct a supervised training example on which to train.

As a simple example, an algorithm might explore uniformly at random on 10% of the examples and return the best guess action on 90% of examples ( -greedy exploration).

A single round to such an algorithm consists of a tuple (x, a, p), where p is the probability with which the algorithm took action a. (In the current example, this would be DISPLAYFORM1 K for all actions except π(x) and 0.9 + 0.1 K for a = π(x).)

If the update rule were "inverse propensity scaling" (IPS) BID18 , the generated cost-sensitive learning example would have x as an input, and a cost vector c ∈ R K with zeros everywhere except in position a where it would take value p .

The justification for this scaling is that in expectation over a ∼ p, the expected value of this cost vector is equal to the true costs for each action.

Neither of these choices is optimal (IPS has very high variance as p gets small); we discuss alternative exploration strategies and variance reduction strategies ( §3.2).

Recently, it has become popular to solve structured prediction problems incrementally using some form of recurrent neural network (RNN) model.

When the output y contains multiple parts (e.g., words in a translation), the RNN can predict each word in sequence, conditioning each prediction on all previous decisions.

Although typically such models are trained to maximize cross-entropy with the gold standard output (in a fully supervised setting), there is mounting evidence that this has similar drawbacks to pre-RNN techniques, such as overfitting to gold standard prefixes (the model never learns what to do once it has made an error) and sensitivity to errors of different severity (due to error compounding).By casting the structured prediction problem explicitly as a sequential decision making problem BID11 BID10 BID37 BID29 , we can avoid these problems by applying imitation-learning style algorithms to their solution.

This "Learning to Search" framework ( Figure 5 ) solves structured prediction problems by:1.

converting structured and control problems to search problems by defining a search space of states S and an action set A; 2. defining structured features over each state to capture the inter-dependency between output variables; 3. constructing a reference policy π ref based on the supervised training data;4.

learning a policy π learn that imitates or improves upon the reference policy.

In the bandit structured prediction setting, this maps nicely to the type of MDPs described at the beginning of this section.

The formal reduction, following BID11 is to ignore the Figure 5 : An example for a search space defined by a Learning to Search (L2S) algorithm.

A search space is defined in terms of the set of states X , and the set of actions A. The agent starts at the initial state S, and queries the roll-in policy π in twice, next, at state R, the agent considers all three actions as possible one-step deviations.

The agent queries the roll-out policy π out to generate three different trajectories from the set of possible output structures Y.first action a 0 and to transition to an "initial state" s 1 by drawing an input x SP ∼ D X .

The search space of the structured prediction task then generates the remainder of the state/action space for this example.

The episode terminates when a state, s H that corresponds to a "final output" is reached, at which point the structured prediction loss (ŷ s H | x SP ) is computed on the output that corresponds to s H .

This then becomes the loss function L in the MDP.

Clearly, learning a good policy under this MDP is equivalent to learning a structured prediction model with low expected loss.

Many of the contextual bandit approaches we use in turn reduce the contextual bandit problem to a cost-sensitive classification problem.

Cost-sensitive classification problems are defined by inputs x and cost vectors y ∈ R K , where y(i) is the cost of choosing class i on this example.

The goal in cost-sensitive classification is to learn a classifier f : DISPLAYFORM0 is small.

A standard strategy for solving cost-sensitive classification is via reduction to regression in a one-against-all framework BID4 .

Here, a regression function g(x, i) ∈ R is learned that predicts costs given input/class pairs.

A predicted class on an input x is chosen as argmin i g(x, i).

This cost-sensitive one-against-all approach achieves low regret when the underlying regressor is good.

In practice, we use regression against Huber loss.

In a now-classic lemma, BID20 and BID1 show that the difference in total loss between two policies can be computed exactly as a sum of per-time-step advantages of one over the other: Lemma 1 BID1 ; BID20 ).

For all policies π and π : DISPLAYFORM0 Proof of Theorem 1.

Let π n be the nth learned policy andπ be the average learned policy.

We wish to bound J(π) − J(π * ).

We proceed as follows, largely following the AggreVaTe analysis BID36 .

We begin by noting that DISPLAYFORM1 and will concern ourselves with bounding the first difference.

DISPLAYFORM2 Fix an n, and consider the sum above for a fixed deviation time step h dev .

In what follows, we consider π n to represent both the learned policy as well as the contextual bandit cost estimator, CB.COST.

DISPLAYFORM3 DISPLAYFORM4 where Residual(π n , h dev , s) is the estimated residual on this example.

Since the above analysis holds for an arbitrary n, it holds in expectation over n; thus: DISPLAYFORM5 In the first line, the term in square brackets is exactly the cost being minimized by the contextual bandit algorithm and thus reduces to the regret of the CB algorithm.

In Eq (13), we have H-many regret minimizing online learners: one estimating the policy and one estimating estimating the H − 1-many costs.

BID8 (Theorem 7.3) proves that in a K-player game, if each player minimizes its internal regret, then the overall values convergence in time-average to the value of the game.

In order to apply this result to our setting we need to convert from external regret (which we are assuming about the underlying learners) to internal regret (which the theorem requires).

This can be done using, for instance, the algorithm of which gives a general reduction from an algorithm that minimizes internal regret to one that minimizes external regret.

From there, by the strong realizability assumption, and the fact that multiple no-regret minimizers will achieve a time-averaged minimax value, we can conclude that as N → ∞, the approximation error term will vanish.

Moreover, the term in the round parentheses (. . . )

is exactly the expected value of the target of the contextual bandit cost.

Therefore, If the CB algorithm has regret sublinear in N , both CB (N ) and the approximation error term go to zero as N → ∞. This completes the proof that the overall algorithm is no-regret.

First, we observe (LOLS Eq 6): DISPLAYFORM6 Then (LOLS Eq 7): DISPLAYFORM7 So far nothing has changed.

It will be convenient to define DISPLAYFORM8 .

For each n fix the deviation time step h dev n .

We plug these together ala LOLS and get: DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 The final step follows because the inner-most expectation is exactly what the contextual bandit algorithm is estimating, and Q πn β (s dev ) is exactly the expectation of the observed loss.

At this point the rest of the proof follows that of Theorem 1, relying on the same internal-to-external regret transformation, and the joint no-regret minimization of all "players."

Blackjack is a card game where the goal is to obtain cards that sum to as near as possible to 21 without going over.

Players play against a fixed dealer who hits until they have at least 17.

Face cards (Jack, Queen, King) have a point value of 10.

Aces can either count as 11 or 1, and a card is called "usable" at 11.

The reward for winning is +1, drawing is 0, and losing is −1.

The world is partially visible: the player can see one their own cards and one of the two initial dealer cards.

Hex is a classic two-player board game invented by Piet Hein and independently by John Nash (Hayward & Van Rijswijck, 2006; BID28 .

The board is an n×n rhombus of hexagonal cells.

Players alternately place a stone of their color on any empty cell.

To win, a player connects her two opposing sides with her stones.

We use n = 5; the world is fully visible to the agent, with each hexagon showing as unoccupied, occupied with white or occupied with black.

The reward is +1 for winning and −1 for losing.

Cart Pole is a classic control problem variously referred to as the "cart-pole", "inverted pendulum", or "pole balancing" problem BID3 .

Is is an example of an inherently unstable dynamic system, in which the objective is to control translational forces that position a cart at the center of a finite width track while simultaneously balancing a pole hinged on the cart's top.

In this task, a pole is attached by a joint to a cart which moves along a frictionless track (Figure 6c ).

The system is controlled by applying a force of +1 or −1 to the cart, thus, we operate in a discrete action space with only two actions.

The pendulum starts upright, and the goal is to prevent it from falling over.

The episode ends when the pole is more than 15 degrees from the vertical axis, or the cart moves more than 2.4 units from the center.

The state is represented by four values indicating the poles position, angle to the vertical axis, and the linear and angular velocities.

The total cumulative reward at the end of the episode is the total number of time steps the pole remained upright before the episode terminates.

Grid World consists of a simple 3×4 grid, with a +1 reward in the upper-right corner and −1 reward immediately below it; the cell at (1, 1) is blocked (Figure 6d ).

The agent starts at a random unoccupied square.

Each step costs 0.05 and the agent has a 10% chance of misstepping.

The agent only gets partial visibility of the world: it gets an indicator feature specifying which directions it can step.

The only reward observed is the complete sum of rewards over an episode.

English POS Tagging we conduct POS tagging experiments over the 45 Penn Treebank BID25 tags.

We simulate a domain adaptation setting by training a reference policy on the TweetNLP dataset BID33 which achieves good accuracy in domain, but performs badly out of domain.

We simulate bandit episodic loss over the entire Penn Treebank Wall Street Journal (sections 02 → 21 and 23), comprising 42k sentences and about one million words.

The measure of performance is the average Hamming loss.

We define the search space by sequentially selecting greedy part-of-speech tags for words in the sentence from left to right.

Chinese POS Tagging we conduct POS tagging experiments over the Chinese Penn Treebank (3.0) BID49 tags.

We simulate a domain adaptation setting by training a reference policy on the Newswire domain from the Chinese Treebank Dataset BID50 and simulate bandit episodic feedback from the spoken conversation domain.

We simulate bandit episodic loss over 40k sentences and about 300k words.

The measure of performance is the average Hamming loss.

We define the search space by sequentially selecting greedy part-of-speech tags for words in the sentence from left to right.

English Dependency Parsing For this task, we assign a grammatical head (i.e. parent) for each word in the sentence.

We train an arc-eager dependency parser BID32 which chooses among (at most) four actions at each state: Shift, Reduce, Left or Right.

The reference policy is trained on the TweetNLP dataset and evaluated on the Penn Treebank corpus.

The loss is the unlabeled attachment score (UAS), which measures the fraction of words that are assigned the correct parent.

In all structured prediction settings, the feature representation begins with pretrained (and nonupdated) embeddings.

For English, these are the 6gb Glove embeddings BID34 ; for Chinese, these are the FastText embeddings BID19 .

We then run a bidirectional LSTM BID17 over the input sentence.

The input features for labeling the nth word in POS tagging experiments are the biLSTM representations at position n. The input features for dependency actions are a concatenation of the biLSTM features of the next word on the buffer and the two words on the top of the stack.

We optimize all parameters of the model using the Adam 9 optimizer (Kingma & Ba, 2014) , with a tuned learning rate, a moving average rate for the mean of β 1 = 0.9 and for the variance of β 2 = 0.999; epsilon (for numerical stability) is fixed at 1e − 8 (these are the DyNet defaults).

The learning rate is tuned in the range {0.050.01, 0.005, 0.001, 0.0005, 0.0001}.For the structured prediction experiments, the following input features hyperparameters are tuned:• Word embedding dimension ∈ {50, 100, 200, 300} (for the Chinese embeddings, which come only in 300 dimensional versions, we took the top singular vectors to reduce the dimensionality).• BiLSTM dimension ∈ {50, 150, 300}• Number of BiLSTM layers ∈ {1, 2}• Pretraining: DAgger or AggreVaTe initialization with probability of rolling in with the reference policy ∈ {0.0, 0.999 N , 0.99999 N , 1.0}, where N is the number of examples• Policy RNN dimension ∈ {50, 150, 300}• Number of policy layers ∈ {1, 2}• Roll-out probability β ∈ {0.0, 0.5, 1.0}For each task, the network architecture that was optimal for supervised pretraining was fixed and used for all bandit learning experiments 10 .For the reinforcement learning experiments, we tuned:• Policy RNN dimension ∈ {20, 50, 100}• Number of policy layers ∈ {1, 2} Some parameters we do not tune: the nonlinearities used, the size of the action embeddings (we use 10 in all cases), the input RNN form for the text experiments (we always use LSTM instead of RNN or GRU based on preliminary experiments).

We do not regularize our models (weight shrinkage only reduced performance in initial experiments) nor do we use dropout.

Pretraining of the structured prediction models ran for 20 passes over the data with early stopping based on held-out loss.

The state of the optimizer was reset once bandit learning began.

The variance across difference configurations was relatively small across RL tasks, so we chose a two layer policy with 20 dimensional vectors for all RL tasks.

Each algorithm also has a set of hyperparameters; we tune them as below:• Reinforce: with baseline or without baseline 9 We initially experimented also with RMSProp BID46 and AdaGrad BID12 but Adam consistently performed as well or better than the others on all tasks.10 English POS tagging and dependency parsing: DAgger 0.99999 N , 300 dim embeddings, 300 dim 1 layer LSTM, 2 layer 300 dimensional policy; Chinese POS tagging: DAgger 0.999 N , 300 dim embeddings, 50 dim 2 layer LSTM, 1 layer 50 dimensional policy).

Table 1 : Results of ablating various parts of the RESIDUAL LOSS PREDICTION approach.

Columns are tasks.

The first two rows are the cumulative average loss over multiple runs and its standard deviation.

The numbers in the rest of the column measure how much it hurts (positive number) or helps (negative number) to ablate the corresponding parameter.

To keep the numbers on a similar scale, the changes are reported as multiples of the standard deviation.

So a value of 2.0 means that the cumulative loss gets worse by an additive factor of two standard deviations.• A2C: a multiplier on the relative importance of actor loss and critic loss ∈ {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}• PPO: with baseline or without baseline; and epsilon parameter ∈ {0.01, 0.05, 0.1, 0.2, 0.4, 0.8}• RESLOPE: update strategy (IPS, DR, MTR) and exploration strategy (uniform, Boltzmann or Bootstrap) In each reinforcement/bandit experiment, we optimistically pick algorithm hyperparameters and learning rate based on final evaluation criteria, noting that this likely provides unrealistically optimistic performance for all algorithms.

We perform 100 replicates of every experiment in the RL setting and 20 replicates in the structured prediction setting.

We additionally ablate various aspects of RESLOPE in §5.2.We employ only two "tricks," both of which are defaults in dynet: gradient clipping (using the default dynet settings) and smart parameter initialization (dynet uses Glorot initialization BID15 ).

Next, we consider the single-deviation version of RESLOPE (1) versus the multiple-deviation version (2).

To enable comparison with alternative algorithms, we also experiment with variants of Reinforce, PPO and DAgger that are only allowed single deviations as well (also chosen uniformly Figure 9: Average loss (top) and heldout loss (bottom) during learning for three bandit structured prediction problems.

Also included are supervised learning results with DAgger.

The main difference between the training loss and the development loss is that in the development data, the system needn't explore, and so the gaps in algorithms which explore different amounts (e.g., especially on English POS tagging) disappear.

at random).

The results are shown in FIG8 .

Not surprisingly, all algorithms suffer when only allowed single deviations.

PPO makes things worse over time (likely because its updates are very conservative, such that even in the original PPO paper the authors advocate multiple runs over the same data), as does Reinforce.

DAgger still learns, though more slowly, when only allowed a single deviation.

RESLOPE behaves similarly though not quite as poorly.

Overall, this suggests that even though the samples generated with multiple deviations by RESLOPE are no longer independent, the gain in number of samples more than makes up for this.

Experiments were conducted on a synthetic sequence labeling dataset.

Input sequences are random integers (between one and ten) of length 6.

The ground truth label for the hth word is the corresponding input mod 4.

We generate 16k training sequences for this experiment.

We run RESLOPE with bootstrap sampling in multiple deviation mode.

We use the MTR cost estimator, and optimize the policies using ADAM with a learning rate of 0.01. .

The x-axis shows the number of episodes and the y-axis measures the incremental loss using the true loss function (light colors) and using RESLOPE (dark colors).

If RESLOPE worked perfectly, these would coincide.

In this section, we study RESLOPE's performance under different-and especially nonadditive-loss functions.

This experiment is akin to the experimental setting in section 5.3, however it's performed on the grid world reinforcement learning environment, where the quantitative aspects of the loss function is well understood.

We study a simple 4×4 grid, with a +1 reward in the upper-right corner and −1 reward immediately below it; the cells at (1, 1) and (2, 1) are blocked.

The agent starts at a random position in the grid.

Each step costs +0.05 and the probability of success is 0.9.

The agent has full visibility of the world: it knows its horizontal and vertical position in the grid.

We consider two different episodic reward settings:1.

The only reward observed is the complete sum of losses over an episode. (additive setting);2.

The only reward observed is the L5 norm of the vector of losses over an episode (nonadditive setting).Results are shown in FIG9 .

Results are very similar to the structured prediction setting (section 5.3).

Performance is better when the loss is additive (blue) vs non-additive (green).

<|TLDR|>

@highlight

We present a novel algorithm for solving reinforcement learning and bandit structured prediction problems with very sparse loss feedback.
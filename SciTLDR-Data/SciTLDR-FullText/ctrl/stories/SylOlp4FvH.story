Some of the most successful applications of deep reinforcement learning to challenging domains in discrete and continuous control have used policy gradient methods in the on-policy setting.

However, policy gradients can suffer from large variance that may limit performance, and in practice require carefully tuned entropy regularization to prevent policy collapse.

As an alternative to policy gradient algorithms, we introduce V-MPO, an on-policy adaptation of Maximum a Posteriori Policy Optimization (MPO) that performs policy iteration based on a learned state-value function.

We show that V-MPO surpasses previously reported scores for both the Atari-57 and DMLab-30 benchmark suites in the multi-task setting, and does so reliably without importance weighting, entropy regularization, or population-based tuning of hyperparameters.

On individual DMLab and Atari levels, the proposed algorithm can achieve scores that are substantially higher than has previously been reported.

V-MPO is also applicable to problems with high-dimensional, continuous action spaces, which we demonstrate in the context of learning to control simulated humanoids with 22 degrees of freedom from full state observations and 56 degrees of freedom from pixel observations, as well as example OpenAI Gym tasks where V-MPO achieves substantially higher asymptotic scores than previously reported.

Deep reinforcement learning (RL) with neural network function approximators has achieved superhuman performance in several challenging domains (Mnih et al., 2015; .

Some of the most successful recent applications of deep RL to difficult environments such as Dota 2 (OpenAI, 2018a), Capture the Flag , Starcraft II (Vinyals et al., 2019) , and dexterous object manipulation (OpenAI, 2018b) have used policy gradient-based methods such as Proximal Policy Optimization (PPO) (Schulman et al., 2017) and the Importance-Weighted Actor-Learner Architecture (IMPALA) , both in the approximately on-policy setting.

Policy gradients, however, can suffer from large variance that may limit performance, especially for high-dimensional action spaces (Wu et al., 2018) .

In practice, moreover, policy gradient methods typically employ carefully tuned entropy regularization in order to prevent policy collapse.

As an alternative to policy gradient-based algorithms, in this work we introduce an approximate policy iteration algorithm that adapts Maximum a Posteriori Policy Optimization (MPO) (Abdolmaleki et al., 2018a; b) to the on-policy setting.

The modified algorithm, V-MPO, relies on a learned state-value function V (s) instead of the state-action value function used in MPO.

Like MPO, rather than directly updating the parameters in the direction of the policy gradient, V-MPO first constructs a target distribution for the policy update subject to a sample-based KL constraint, then calculates the gradient that partially moves the parameters toward that target, again subject to a KL constraint.

As we are particularly interested in scalable RL algorithms that can be applied to multi-task settings where a single agent must perform a wide variety of tasks, we show for the case of discrete actions that the proposed algorithm surpasses previously reported performance in the multi-task setting for both the Atari-57 (Bellemare et al., 2012) and DMLab-30 (Beattie et al., 2016) benchmark suites, and does so reliably without population-based tuning of hyperparameters (Jaderberg et al., 2017a) .

For a few individual levels in DMLab and Atari we also show that V-MPO can achieve scores that are substantially higher than has previously been reported in the single-task setting, especially in the challenging Ms. Pacman.

V-MPO is also applicable to problems with high-dimensional, continuous action spaces.

We demonstrate this in the context of learning to control both a 22-dimensional simulated humanoid from full state observations-where V-MPO reliably achieves higher asymptotic performance than previous algorithms-and a 56-dimensional simulated humanoid from pixel observations (Tassa et al., 2018; Merel et al., 2019) .

In addition, for several OpenAI Gym tasks (Brockman et al., 2016) we show that V-MPO achieves higher asymptotic performance than has previously been reported.

We consider the discounted RL setting, where we seek to optimize a policy ?? for a Markov Decision Process described by states s, actions a, initial state distribution ?? env 0 (s 0 ), transition probabilities P env (s t+1 |s t , a t ), reward function r(s t , a t ), and discount factor ?? ??? (0, 1).

In deep RL, the policy ?? ?? (a t |s t ), which specifies the probability that the agent takes action a t in state s t at time t, is described by a neural network with parameters ??.

We consider problems where both the states s and actions a may be discrete or continuous.

Two functions play a central role in RL: the state-value function V ?? (s t ) = E at,st+1,at+1,... ??? k=0 ?? k r(s t+k , a t+k ) and the state-action value function Q ?? (s t , a t ) = E st+1,at+1,... ??? k=0 ?? k r(s t+k , a t+k ) = r(s t , a t ) + ??E st+1 V ?? (s t+1 ) , where s 0 ??? ?? env 0 (s 0 ), a t ??? ??(a t |s t ), and s t+1 ??? P env (s t+1 |s t , a t ).

In the usual formulation of the RL problem, the goal is to find a policy ?? that maximizes the expected return given by J(??) = E s0,a0,s1,a1,... ??? t=0 ?? t r(s t , a t ) .

In policy gradient algorithms (Williams, 1992; Sutton et al., 2000; Mnih et al., 2016) , for example, this objective is directly optimized by estimating the gradient of the expected return.

An alternative approach to finding optimal policies derives from research that treats RL as a problem in probabilistic inference, including Maximum a Posteriori Policy Optimization (MPO) (Levine, 2018; Abdolmaleki et al., 2018a; b) .

Here our objective is subtly different, namely, given a suitable criterion for what are good actions to take in a certain state, how do we find a policy that achieves this goal?

As was the case for the original MPO algorithm, the following derivation is valid for any such criterion.

However, the policy improvement theorem (Sutton & Barto, 1998) tells us that a policy update performed by exact policy iteration,

, can improve the policy if there is at least one state-action pair with a positive advantage and nonzero probability of visiting the state.

Motivated by this classic result, in this work we specifically choose an exponential function of the advantages

Notation.

In the following we use s,a to indicate both discrete and continuous sums (i.e., integrals) over states s and actions a depending on the setting.

A sum with indices only, such as s,a , denotes a sum over all possible states and actions, while s,a???D , for example, denotes a sum over sample states and actions from a batch of trajectories (the "dataset") D.

3 RELATED WORK V-MPO shares many similarities, and thus relevant related work, with the original MPO algorithm (Abdolmaleki et al., 2018a; b) .

In particular, the general idea of using KL constraints to limit the size of policy updates is present in both Trust Region Policy Optimization (TRPO; Schulman et al., 2015) and Proximal Policy Optimization (PPO) (Schulman et al., 2017) ; we note, however, that this corresponds to the E-step constraint in V-MPO.

It is worth noting here the following main differences with MPO, which is conceptually quite similar to V-MPO.

MPO is primarily designed to be a sample-efficient off-policy algorithm in which the Figure 1 : (a) Actor-learner architecture with a target network, which is used to generate agent experience in the environment and is updated every T target learning steps from the online network.

(b) Schematic of the agents, with the policy (??) and value (??) networks sharing most of their parameters through a shared input encoder and LSTM [or Transformer-XL (TrXL) for single Atari levels].

The agent also receives the action and reward from the previous step as an input to the LSTM.

For DMLab an additional LSTM is used to process simple language instructions.

E-step constructs a conditional target distribution q(a|s), which requires a state-action value function Q(s, a) that can evaluate multiple sampled actions for a given state.

In contrast, V-MPO is primarily (though not exclusively) designed to be an on-policy algorithm in which the E-step constructs a joint distribution ??(s, a), and in the absence of a learned Q-function only one action per state is used.

In this regard V-MPO can also be compared to Fitted Q-iteration by Advantage Weighted Regression (Neumann & Peters, 2009 ), which learns a Q-function but uses only one action per state.

V-MPO can also be related to Relative Entropy Policy Search (REPS) (Peters et al., 2008) .

Two distinguishing features of V-MPO from REPS are the introduction of the M-step KL constraint and the use of top-k advantages.

Moreover, in REPS the value function is a linear function of a learned feature representation whose parameters are trained by matching the feature distributions under the policy's stationary state distribution.

In V-MPO, the nonlinear neural network value function is instead learned directly from n-step returns.

Interestingly, previous attempts to use REPS with neural network function approximators reported very poor performance, being particularly prone to local optima (Duan et al., 2016) .

In contrast, we find that the principles of EM-style policy optimization, when combined with this learned value function and appropriate constraints, can reliably train powerful neural networks, including transformers, for RL tasks.

Like V-MPO, Supervised Policy Update (SPU) (Vuong et al., 2019) seeks to exactly solve an optimization problem and fit the parametric policy to this solution.

As we argue in Appendix D, however, SPU uses this nonparametric distribution quite differently from V-MPO; as a result, the final algorithm is closer to a policy gradient algorithm such as PPO.

V-MPO is an approximate policy iteration (Sutton & Barto, 1998) algorithm with a specific prescription for the policy improvement step.

In general, policy iteration uses the fact that the true state-value function V ?? corresponding to policy ?? can be used to obtain an improved policy ?? .

Thus we can 1.

Generate trajectories ?? from an old policy ?? ??old (a|s) whose parameters ?? old are fixed.

To control the amount of data generated by a particular policy, we use a target network which is fixed for T target learning steps (Fig. 1a) .

2.

Evaluate the policy ?? ??old (a|s) by learning the value function V ?? ?? old (s) from empirical returns and estimating the corresponding advantages A ?? ?? old (s, a) for the actions that were taken (Section 4.1).

3.

Based on A ?? ?? old (s, a), estimate an improved policy ?? ?? (a|s) which we call the "online" policy to distinguish it from the fixed target network (Section 4.2).

The first two steps are standard, and describing V-MPO's approach to step 3 is the essential contribution of this work.

At a high level, our strategy is to first construct a nonparametric target distribution for the policy update, then partially move the parametric policy towards this distribution subject to a KL constraint.

We first review policy evaluation (step 2) in Section 4.1, then derive the V-MPO policy improvement (step 3) in Section 4.2.

Ultimately, we use gradient descent to optimize a single, relatively simple loss, which is given in Eq. 10 following the derivation.

A summary of the full algorithm is also presented in Algorithm 1.

In the present setting, policy evaluation means learning an approximate state-value function V ?? (s) given a policy ??(a|s), which we keep fixed for T target learning steps (i.e., batches of trajectories).

We note that the value function corresponding to the target policy is instantiated in the "online" network receiving gradient updates; bootstrapping uses the online value function, as it is the best available estimate of the value function for the target policy.

Thus in this section ?? refers to ?? ??old , while the value function update is performed on the current ??, which may share parameters with the current ??.

We fit a parametric value function V ?? ?? (s) with parameters ?? by minimizing the squared loss

where

is the standard n-step target for the value function at state s t at time t (Sutton & Barto, 1998) .

This return uses the actual rewards in the trajectory and bootstraps from the value function for the rest: for each = t, . . .

, t + n ??? 1 in an unroll,

The advantages, which are the key quantity of interest for the policy improvement step in V-MPO, are then given by

for each s t , a t in the batch of trajectories.

PopArt normalization.

As we are interested in the multi-task setting where a single agent must learn a large number of tasks with differing reward scales, we used PopArt (van Hasselt et al., 2016; Hessel et al., 2018) for the value function, even when training on a single task.

We observed benefits in using PopArt even in the single-task setting, partly due to the fact that we do not tune the relative weighting of the policy evaluation and policy improvement losses despite sharing most parameters for the policy and value networks.

Specifically, the value function outputs a separate value for each task in normalized space, which is converted to actual returns by a shift and scaling operation, the statistics of which are learned during training.

We used a scale lower bound of 10 ???2 , scale upper bound of 10 6 , and learning rate of 10 ???4 for the statistics.

The lower bound guards against numerical issues when rewards are extremely sparse.

Importance-weighting for off-policy data.

It is possible to importance-weight the samples using V-trace to correct for off-policy data , for example when data is taken from a replay buffer.

For simplicity, however, no importance-weighting was used for the experiments presented in this work, which were mostly on-policy.

In this section we show how, given the advantage function A ?? ?? old (s, a) for the state-action distribution p ??old (s, a) = ?? ??old (a|s)p(s) induced by the old policy ?? ??old (a|s), we can estimate an improved policy ?? ?? (a|s).

More formally, let I denote the binary event that the new policy is an improvement (in a sense to be defined below) over the previous policy: I = 1 if the policy is successfully improved and 0 otherwise.

Then we would like to find the mode of the posterior distribution over parameters ?? conditioned on this event, i.e., we seek the maximum a posteriori (MAP) estimate

where we have written p(I = 1|??) as p ?? (I = 1) to emphasize the parametric nature of the dependence on ??.

We use the well-known identity log p(

is the Kullback-Leibler divergence between ??(Z) and p(Z|X) with respect to Z, and the first term is a lower bound because the KL divergence is always non-negative.

Then considering s, a as latent variables,

Policy improvement in V-MPO consists of the following two steps which have direct correspondences to the expectation maximization (EM) algorithm (Neal & Hinton, 1998) : In the expectation (E) step, we choose the variational distribution ??(s, a) such that the lower bound on log p ?? (I = 1) is as tight as possible, by minimizing the KL term.

In the maximization (M) step we then find parameters ?? that maximize the corresponding lower bound, together with the prior term in Eq. 2.

In the E-step, our goal is to choose the variational distribution ??(s, a) such that the lower bound on log p ?? (I = 1) is as tight as possible, which is the case when the KL term in Eq. 3 is zero.

Given the old parameters ?? old , this simply leads to ??(s, a) = p ??old (s, a|I = 1), or

Intuitively, this solution weights the probability of each state-action pair with its relative improvement probability p ??old (I = 1|s, a).

We now choose a distribution p ??old (I = 1|s, a) that leads to our desired outcome.

As we prefer actions that lead to a higher advantage in each state, we suppose that this probability is given by

for some temperature ?? > 0, from which we obtain the equation on the right in Eq. 12.

This probability depends on the old parameters ?? old and not on the new parameters ??.

Meanwhile, the value of ?? allows us to control the diversity of actions that contribute to the weighting, but at the moment is arbitrary.

It turns out, however, that we can tune ?? as part of the optimization, which is desirable since the optimal value of ?? changes across iterations.

The convex loss that achieves this, Eq. 13, is derived in Appendix A by minimizing the KL term in Eq. 3 subject to a hard constraint on ??(s, a).

Top-k advantages.

We found that learning improves substantially if we take only the samples corresponding to the highest 50% of advantages in each batch for the E-step, corresponding to the use ofD rather than D in Eqs. 12, 13.

Importantly, these must be consistent between the maximum likelihood weights in Eq. 12 and the temperature loss in Eq. 13, since, mathematically, this corresponds to a specific choice of the policy improvement probability in Eq. 5 to only use the top half of the advantages.

This is similar to the technique used in the Cross Entropy Method (CEM) (Mannor et al., 2003) and Covariance Matrix Adaptation -Evolutionary Strategy (CMA-ES) (Hansen et al., 1997; Abdolmaleki et al., 2017) , and is a special case of the more general feature that any rank-preserving transformation is allowed under this formalism.

For example, in Fig. 8 of the Appendix we show an example of an agent trained with uniform weights given to the top-k samples, instead of optimizing the temperature.

Other choices are possible, and in future work we will investigate the suitability of different choices for specific applications.

Importance weighting for off-policy corrections.

As for the value function, importance weights can be used in the policy improvement step to correct for off-policy data.

While not used for the experiments presented in this work, details for how to carry out this correction are given in Appendix E.

In the E-step we found the nonparametric variational state-action distribution ??(s, a), Eq. 4, that gives the tightest lower bound to p ?? (I = 1) in Eq. 3.

In the M-step we maximize this lower bound together with the prior term log p(??) with respect to the parameters ??, which effectively leads to a constrained weighted maximum likelihood problem.

Thus the introduction of the nonparametric distribution in Eq. 4 separates the RL procedure from the neural network fitting.

We would like to find new parameters ?? that minimize

Note, however, that so far we have worked with the joint state-action distribution ??(s, a) while we are in fact optimizing for the policy, which is the conditional distribution ?? ?? (a|s).

Writing p ?? (s, a) = ?? ?? (a|s)p(s) since only the policy is parametrized by ?? and dropping terms that are not parametrized by ??, the first term of Eq. 6 is seen to be the weighted maximum likelihood policy loss

In the sample-based computation of this loss, we assume that any state-action pairs not in the batch of trajectories have zero weight, leading to the normalization in Eq. 12.

As in the original MPO algorithm, a useful prior is to keep the new policy ?? ?? (a|s) close to the old policy ?? ??old (a|s): log p(??) ??? ?????E s???p(s) D KL ?? ??old (a|s) ?? ?? (a|s) .

While intuitive, we motivate this more formally in Appendix B.

It is again more convenient to specify a bound on the KL divergence instead of tuning ?? directly, so we solve the constrained optimization problem

Intuitively, the constraint in the E-step expressed by Eq. 18 in Appendix A for tuning the temperature only constrains the nonparametric distribution; it is the constraint in Eq. 8 that directly limits the change in the parametric policy, in particular for states and actions that were not in the batch of samples and which rely on the generalization capabilities of the neural network function approximator.

To make the constrained optimization problem amenable to gradient descent, we use Lagrangian relaxation to write the unconstrained objective as

which we can optimize by following a coordinate-descent strategy, alternating between the optimization over ?? and ??.

Since ?? and ?? are Lagrange multipliers that must be positive, after each gradient update we project the resulting ?? and ?? to a small positive value which we choose to be ?? min = ?? min = 10 ???8 throughout the results presented below.

KL constraints in both the E-step and M-step are generally well satisfied, especially for the E-step since the temperature optimization is convex.

Fig. 7 in the Appendix shows an example of how the KL constraints behave in the Atari Seaquest experiment presented below.

We note, in particular, that it is desirable for the bounds to not just be satisfied but saturated.

In this section we provide the full loss function used to implement V-MPO, which is perhaps simpler than is suggested by the derivation.

Consider a batch of data D consisting of a number of trajectories, with |D| total state-action samples.

Each trajectory consists of an unroll of length n of the form ?? = (s t , a t , r t+1 ), . . .

, (s t+n???1 , a t+n???1 , r t+n ), s t+n including the bootstrapped state s t+n , where r t+1 = r(s t , a t ).

The total loss is the sum of a policy evaluation loss and a policy improvement loss,

where ?? are the parameters of the value network, ?? the parameters of the policy network, and ?? and ?? are Lagrange multipliers.

In practice, the policy and value networks share most of their parameters in the form of a shared convolutional network (a ResNet) and recurrent LSTM core, and are optimized together (Fig. 1b) (Mnih et al., 2016) .

We note, however, that the value network parameters ?? are considered fixed for the policy improvement loss, and gradients are not propagated.

The policy evaluation loss for the value function, L V (??), is the standard regression to n-step returns and is given by Eq. 1 above.

The policy improvement loss L V-MPO (??, ??, ??) is given by

Here the policy loss is the weighted maximum likelihood loss

where the advantages A target (s, a) for the target network policy ?? ??target (a|s) are estimated according to the standard method described above.

The tilde over the dataset,D, indicates that we take samples corresponding to the top half advantages in the batch of data.

The ??, or "temperature", loss is

We perform the alternating optimization over ?? and ?? while keeping a single loss function by alternately applying a "stop-gradient" to the Lagrange multiplier and KL term.

Then the KL constraint, which can be viewed as a form of trust-region loss, is given by

where sg [[??] ] indicates a stop gradient, i.e., that the enclosed term is assumed constant with respect to all variables.

Note that here we use the full batch D, notD.

For continuous action spaces parametrized by Gaussian distributions, we use decoupled KL constraints for the M-step in Eq. 14 as in Abdolmaleki et al. (2018b) ; the precise form is given in Appendix C.

We used the Adam optimizer (Kingma & Ba, 2015) with default TensorFlow hyperparameters to optimize the total loss in Eq. 10.

In particular, the learning rate was fixed at 10 ???4 for all experiments.

given Batch size B, unroll length n, T target , KL bounds ?? , ?? . initialize Network parameters ?? online , ?? online , Lagrange multipliers ??, ??.

repeat ?? target ??? ?? online for i = 1, . . .

, T target do Use policy ?? ??target to act in the environment and collect B trajectories ?? of length n. Update ?? online , ?? online , ??, ?? using Adam to minimize the total loss in Eq. 10.

?? ??? max(??, ?? min ) ?? ??? max(??, ?? min ) end for until Fixed number of steps.

Details on the network architecture and hyperparameters used for each task are given in Appendix F.

DMLab.

DMLab-30 (Beattie et al., 2016 ) is a collection of visually rich, partially observable 3D environments played from the first-person point of view.

Like IMPALA, for DMLab we used pixel control as an auxiliary loss for representation learning (Jaderberg et al., 2017b; Hessel et al., 2018) .

However, we did not employ the optimistic asymmetric reward scaling used by previous IMPALA Multi-task Atari-57.

In the IMPALA experiment, hyperparameters were evolved with PBT.

For V-MPO each of the 24 lines represents a set of hyperparameters that were fixed throughout training, and all runs achieved a higher score than the best IMPALA run.

Data for IMPALA ("Pixel-PopArt-IMPALA" for DMLab-30 and "PopArt-IMPALA" for Atari-57) was obtained from the authors of Hessel et al. (2018) .

Each agent step corresponds to 4 environment frames due to the action repeat.

experiments to aid exploration on a subset of the DMLab levels, by weighting positive rewards more than negative rewards Hessel et al., 2018; Kapturowski et al., 2019) .

Unlike in Hessel et al. (2018) we also did not use population-based training (PBT) (Jaderberg et al., 2017a) .

Additional details for the settings used in DMLab can be found in Table 5 of the Appendix.

Fig. 2a shows the results for multi-task DMLab-30, comparing the V-MPO learning curves to data obtained from Hessel et al. (2018) for the PopArt IMPALA agent with pixel control.

We note that the result for V-MPO at 10B environment frames across all levels matches the result for the Recurrent Replay Distributed DQN (R2D2) agent (Kapturowski et al., 2019 ) trained on individual levels for 10B environment steps per level.

Fig. 3 shows example individual levels in DMLab where V-MPO achieves scores that are substantially higher than has previously been reported, for both R2D2 and IMPALA.

The pixel-control IMPALA agents shown here were carefully tuned for DMLab and are similar to the "experts" used in ; in all cases these results match or exceed previously published results for IMPALA Kapturowski et al., 2019) .

Atari.

The Atari Learning Environment (ALE) (Bellemare et al., 2012 ) is a collection of 57 Atari 2600 games that has served as an important benchmark for recent deep RL methods.

We used the standard preprocessing scheme and a maximum episode length of 30 minutes (108,000 frames), see Table 6 in the Appendix.

For the multi-task setting we followed Hessel et al. (2018) in setting the discount to zero on loss of life; for the example single tasks we did not employ this trick, since it can prevent the agent from achieving the highest score possible by sacrificing lives.

Similarly, while in the multi-task setting we followed previous work in clipping the maximum reward to 1.0, no such clipping was applied in the single-task setting in order to preserve the original reward structure.

Additional details for the settings used in Atari can be found in Table 6 in the Appendix.

Fig. 2b shows the results for multi-task Atari-57, demonstrating that it is possible for a single agent to achieve "superhuman" median performance on Atari-57 in approximately 4 billion (???70 million per level) environment frames.

Again, while we did not employ PBT in order to demonstrate that individual V-MPO runs can exceed the performance of a population of IMPALA agents, Fig. 6 shows that with population-based tuning of hyperparameters even higher performance is possible.

We also compare the performance of V-MPO on a few individual Atari levels to R2D2 (Kapturowski et al., 2019) , which previously achieved some of the highest scores reported for Atari.

Again, V-MPO can match or exceed previously reported scores while requiring fewer interactions with the environment.

In Ms. Pacman, the final performance approaches 300,000 with a 30-minute timeout (and the maximum 1M without).

Inspired by the argument in Kapturowski et al. (2019) that in a fully observable environment LSTMs enable the agent to utilize more useful representations than is available in the immediate observation, for the single-task setting we used a Transformer-XL (TrXL) (Dai et al., 2019) to replace the LSTM core.

Unlike previous work for single Atari levels, we did not employ any reward clipping (Mnih et al., 2015; or nonlinear value function rescaling (Kapturowski et al., 2019) .

To demonstrate V-MPO's effectiveness in high-dimensional, continuous action spaces, here we present examples of learning to control both a simulated humanoid with 22 degrees of freedom from full state observations and one with 56 degrees of freedom from pixel observations (Tassa et al., 2018; Merel et al., 2019) .

As shown in Fig. 5a , for the 22-dimensional humanoid V-MPO reliably achieves higher asymptotic returns than has previously been reported, including for Deep Deterministic Policy Gradients (DDPG) , Stochastic Value Gradients (SVG) , and MPO.

These algorithms are far more sample-efficient but reach a lower final performance.

In the "gaps" task the 56-dimensional humanoid must run forward to match a target velocity of 4 m/s and jump over the gaps between platforms by learning to actuate joints with position-control (Merel et al., 2019) .

Previously, only an agent operating in the space of pre-learned motor primitives was able to solve the task from pixel observations ; here we show that V-MPO can learn a challenging visuomotor task from scratch (Fig. 5b) .

For this task we also demonstrate the importance of the parametric KL constraint, without which the agent learns poorly.

In Figs. 5c-d we also show that V-MPO achieves the highest asymptotic performance reported for two OpenAI Gym tasks (Brockman et al., 2016) .

Again, MPO and Stochastic Actor-Critic (Haarnoja et al., 2018) are far more sample-efficient but reach a lower final performance.

These experiments are presented to demonstrate the existence of higher-return solutions than have previously been reported, and an algorithm, V-MPO, that can reliably converge to these solutions.

However, in the future we desire algorithms that can do so while using fewer interactions with the environment.

In this work we have introduced a scalable on-policy deep reinforcement learning algorithm, V-MPO, that is applicable to both discrete and continuous control domains.

For the results presented in this work neither importance weighting nor entropy regularization was used; moreover, since the size of neural network parameter updates is limited by KL constraints, we were also able to use the same learning rate for all experiments.

This suggests that a scalable, performant RL algorithm may not require some of the tricks that have been developed over the past several years.

Interestingly, both the original MPO algorithm for replay-based off-policy learning (Abdolmaleki et al., 2018a; b) and V-MPO for on-policy learning are derived from similar principles, providing evidence for the benefits of this approach as an alternative to popular policy gradient-based methods.

In this section we derive the E-step temperature loss in Eq. 22.

To this end, we explicitly commit to the more specific improvement criterion in Eq. 5 by plugging into the original objective in Eq. 3.

We seek ??(s, a) that minimizes

where ?? = ?? log p ??old (I = 1) after multiplying through by ??, which up to this point in the derivation is given.

We wish to automatically tune ?? so as to enforce a bound ?? on the KL term D KL ??(s, a) p ??old (s, a) multiplying it in Eq. 16, in which case the temperature optimization can also be viewed as a nonparametric trust region for the variational distribution with respect to the old distribution.

We therefore consider the constrained optimization problem

s.t.

s,a ??(s, a) log ??(s, a) p ??old (s, a) < ?? and s,a ??(s, a) = 1.

We can now use Lagrangian relaxation to transform the constrained optimization problem into one that maximizes the unconstrained objective

with ?? ??? 0.

(Note we are re-using the variables ?? and ?? for the new optimization problem.)

Differentiating J with respect to ??(s, a) and setting equal to zero, we obtain

Normalizing over s, a (using the freedom given by ??) then gives

which reproduces the general solution Eq. 4 for our specific choice of policy improvement in Eq. 5.

However, the value of ?? can now be found by optimizing the corresponding dual function.

Plugging Eq. 21 into the unconstrained objective in Eq. 19 gives rise to the ??-dependent term

Replacing the expectation with samples from p ??old (s, a) in the batch of trajectories D leads to the loss in Eq. 13.

Here we give a somewhat more formal motivation for the prior log p(??).

Consider a normal prior N (??; ??, ??) with mean ?? and covariance ??. We choose ?? ???1 = ??F (?? old ) where ?? is a scaling parameter and F (?? old ) is the Fisher information for ?? ?? (a|s) evaluated at ?? = ?? old .

Then

T F (?? old )(?? ??? ?? old ) + {term independent of ??}, where the first term is precisely the second-order approximation to the KL divergence D KL (?? old ??).

We now follow TRPO (Schulman et al., 2015) in heuristically approximating this as the state-averaged expression, E s???p(s) D KL ?? ??old (a|s) ?? ?? (a|s) .

We note that the KL divergence in either direction has the same second-order expansion, so our choice of KL is an empirical one (Abdolmaleki et al., 2018a) .

As in Abdolmaleki et al. (2018b) , for continuous action spaces parametrized by Gaussian distributions we use decoupled KL constraints for the M-step.

This uses the fact that the KL divergence between two d-dimensional multivariate normal distributions with means ?? 1 , ?? 2 and covariances ?? 1 , ?? 2 can be written as

where | ?? | is the matrix determinant.

Since the first distribution and hence ?? 1 in the KL divergence of Eq. 9 depends on the old target network parameters, we see that we can separate the overall KL divergence into a mean component and a covariance component:

With the replacement D KL ?? ??old ?? ?? ??? D C KL ?? ??old ?? ?? for C = ??, ?? and corresponding ?? ??? ?? ?? , ?? ?? , we obtain the total loss

where L ?? (??) and L ?? (??) are the same as before.

Note, however, that unlike in Abdolmaleki et al. (2018a) we do not decouple the policy loss.

We generally set ?? to be much smaller than ?? (see Table 7 ).

Intuitively, this allows the policy to learn quickly in action space while preventing premature collapse of the policy, and, conversely, increasing "exploration" without moving in action space.

Like V-MPO, Supervised Policy Update (SPU) (Vuong et al., 2019) adopts the strategy of first solving a nonparametric constrained optimization problem exactly, then fitting a neural network to the resulting solution via a supervised loss function.

There is, however, an important difference from V-MPO, which we describe here.

In SPU, the KL loss, which is the sole loss in SPU, leads to a parametric optimization problem that is equivalent to the nonparametric optimization problem posed initially.

To see this, we observe that the SPU loss seeks parameters (note the direction of the KL divergence)

= arg min

= arg min

(29) Multiplying by ?? since it can be treated as a constant up to this point, we then see that this corresponds exactly to the (Lagrangian form) of the problem

s.t.

which is the original nonparametric problem posed in Vuong et al. (2019) .

The network that generates the data may lag behind the target network in common distributed, asynchronous implementations .

We can compensate for this by multiplying the exponentiated advantages by importance weights ??(s, a):

where ?? D are the parameters of the behavior policy that generated D and which may be different from ?? target .

The clipped importance weights ??(s, a) are given by

As was the case with V-trace for the value function, we did not find it necessary to use importance weighting and all experiments presented in this work did not use them for the sake of simplicity.

For DMLab the visual observations were 72??96 RGB images, while for Atari the observations were 4 stacked frames of 84??84 grayscale images.

The ResNet used to process visual observations is similar to the 3-section ResNet used in Hessel et al. (2018) , except the number of channels was multiplied by 4 in each section, so that the number of channels were (64, 128, 128) (Schmitt et al., 2019) .

For individual DMLab levels we used the same number of channels as Hessel et al. (2018), i.e., (16, 32, 32) .

Each section consisted of a convolution and 3 ?? 3 max-pooling operation (stride 2), followed by residual blocks of size 2, i.e., a convolution followed by a ReLU nonlinearity, repeated twice, and a skip connection from the input residual block input to the output.

The entire stack was passed through one more ReLU nonlinearity.

All convolutions had a kernel size of 3 and a stride of 1.

For the humanoid control tasks from vision, the number of channels in each section were (16, 32, 32) .

Since some of the levels in DMLab require simple language processing, for DMLab the agents contained an additional 256-unit LSTM receiving an embedding of hashed words as input.

The output of the language LSTM was then concatenated with the output of the visual processing pathway as well as the previous reward and action, then fed to the main LSTM.

For multi-task DMLab we used a 3-layer LSTM, each with 256 units, and an unroll length of 95 with batch size 128.

For the single-task setting we used a 2-layer LSTM.

For multi-task Atari and the 56-dimensional humanoid-gaps control task a single 256-unit LSTM was used, while for the 22-dimensional humanoid-run task the core consisted only of a 2-layer MLP with 512 and 256 units (no LSTM).

For single-task Atari a Transformer-XL was used in place of the LSTM.

Note that we followed Radford et al. (2019) in placing the layer normalization on only the inputs to each sub-block.

For Atari the unroll length was 63 with a batch size of 128.

For both humanoid control tasks the batch size was 64, but the unroll length was 40 for the 22-dimensional humanoid and 63 for the 56-dimensional humanoid.

In all cases the policy logits (for discrete actions) and Gaussian distribution parameters (for continuous actions) consisted of a 256-unit MLP followed by a linear readout, and similarly for the value function.

For discrete actions we initialized the linear policy layer with zero weights and biases to ensure a uniform policy at the start of training.

The initial values for the Lagrange multipliers in the V-MPO loss are given in Table 1 Implementation note.

We implemented V-MPO in an actor-learner framework that utilizes TF-Replicator (Buchlovsky et al., 2019) for distributed training on TPU 8-core and 16-core configurations (Google, 2018) .

One practical consequence of this is that a full batch of data D was in fact split into 8 or 16 minibatches, one per core/replica, and the overall result obtained by averaging the computations performed for each minibatch.

More specifically, the determination of the highest advantages and the normalization of the nonparametric distribution, Eq. 12, is performed within minibatches.

While it is possible to perform the full-batch computation by utilizing crossreplica communication, we found this to be unnecessary.

DMLab action set.

Ignoring the "jump" and "crouch" actions which we do not use, an action in the native DMLab action space consists of 5 integers whose meaning and allowed values are given in Table 2 .

Following previous work on DMLab (Hessel et al., 2018) , we used the reduced action set given in Table 3 with an action repeat of 4.

Table 7 : Settings for continuous control.

For the humanoid gaps task from pixels the physics time step was 5 ms and the control time step 30 ms.

Figure 6: Multi-task Atari-57 with population-based training (PBT) (Jaderberg et al., 2017a) .

All settings of the PBT experiment were the same as without except the learning rates were also sampled log-uniformly from [8 ?? 10 ???5 , 3 ?? 10 ???4 ) and ?? from [0.05, 0.5).

Along with ?? sampled loguniformly from [0.001, 0.01) as in the original experiment, hyperparameters were evolved via copy and mutation operators roughly once every 4 ?? 10 8 environment frames.

Fig. 2a (multi-task DMLab-30), but trained without top-k, i.e., all advantages are used in the E-step.

Note the small dip in the middle is due to a pause in the experiment and resetting of the human-normalized scores.

Figure 10: Example frame from the humanoid gaps task, with the agent's 64??64 first-person view on the right.

The proprioceptive information provided to the agent in addition to the primary pixel observation consisted of joint angles and velocities, root-to-end-effector vectors, root-frame velocity, rotational velocity, root-frame acceleration, and the 3D orientation relative to the z-axis.

<|TLDR|>

@highlight

A state-value function-based version of MPO that achieves good results in a wide range of tasks in discrete and continuous control.

@highlight

This paper presents an algorithm for on-policy reinforcement learning that can handle both continuous/discrete control, single/multi-task learning and use both low dimensional states and pixels.

@highlight

The paper proposes an online variant of MPO, V-MPO, which learns the V-function and updates the non-parametric distribution towards the advantages.
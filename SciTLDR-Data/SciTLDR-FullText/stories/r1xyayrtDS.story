Reinforcement learning in an actor-critic setting relies on accurate value estimates of the critic.

However, the combination of function approximation, temporal difference (TD) learning and off-policy training can lead to an overestimating value function.

A solution is to use Clipped Double Q-learning (CDQ), which is used in the TD3 algorithm and computes the minimum of two critics in the TD-target.

We show that CDQ induces an underestimation bias and propose a new algorithm that accounts for this by using a weighted average of the target from CDQ and the target coming from a single critic.

The weighting parameter is adjusted during training such that the value estimates match the actual discounted return on the most recent episodes and by that it balances over- and underestimation.

Empirically, we obtain more accurate value estimates and demonstrate state of the art results on several OpenAI gym tasks.

In recent years it was shown that reinforcement learning algorithms are capable of solving very complex tasks, surpassing human expert performance in games like Go , Starcraft (DeepMind) or Dota (OpenAI).

However, usually a large amount of training time is needed to achieve these results (e.g. 45,000 years of gameplay for Dota).

For many important problems (e.g. in robotics) it is prohibitively expensive for the reinforcement learning agent to interact with its environment that much.

This makes it difficult to apply such algorithms in the real world.

Off-policy reinforcement learning holds the promise of being more data-efficient than on-policy methods as old experience can be reused several times for training.

Unfortunately, the combination of temporal-difference (TD) learning, function approximation and off-policy training can be unstable, which is why it has been called the deadly triad (Sutton & Barto, 2018; van Hasselt et al., 2018) .

If the action space is discrete, solutions like Double DQN (Van Hasselt et al., 2016) are very effective at preventing divergence of the value estimates by eliminating an otherwise prevailing overestimation bias.

For continuous action spaces, which characterize many tasks, it was shown that Double DQN can not solve the overestimation problem Fujimoto et al. (2018) .

In an actor-critic setting it is important that the value estimates of the critic are accurate in order for the actor to learn a policy from the critic.

The TD3 Fujimoto et al. (2018) algorithm uses Clipped Double Q-learning (CDQ) to produce a critic without an overestimation bias, which greatly improved the performance of the algorithm.

In CDQ two critics are trained at the same time and the TD target for both of them is the minimum over the two single TD targets.

While the authors note that the CDQ critic update tends to underestimate the true values, this is not further examined.

We show that this underestimation bias occurs in practice and propose a method that accounts for over-and underestimation of the critic at the same time.

Similarly to CDQ we train two function approximators for the Q-values, but we regress them not on the same quantity.

The TD target for each of the two critics is a weighted average of the single TD target for that critic and the TD target from CDQ.

The weighting parameter is learned by comparing the value estimates for the most recent state-action pairs with the observed discounted returns for these pairs.

As the one term of the average has an underestimation bias while the other one has an overestimation bias, the weighted average balances these biases and we show empirically that this method obtains much more accurate estimates of the Q-values.

We verify that the more accurate critics improve the performance of the reinforcement learning agent as our method achieves state of the art results on a range of continuous control tasks from OpenAi gym Brockman et al. (2016) .

To guarantee reproducibility we open source our code which is easy to execute and evaluate our algorithm on a large number of different random seeds.

The Deterministic Policy Gradient algorithm (DPG) Silver et al. (2014) learns a deterministic policy in an actor-critic setting.

This work was extended to the Deep Deterministic Policy Gradient algorithm Lillicrap et al. (2015) by using multi-layer neural networks as function approximators.

The Twin Delayed Deep Deterministic policy gradient algorithm (TD3) Fujimoto et al. (2018) adds three more components to DDPG and achieves state of the art results.

First, the actor is updated less frequently than the critic, to allow for more accurate critic estimates before they are used for the actor.

Second, in the critic update noise is added to the actions proposed by the actor.

While these two extensions are introduced to decrease the variance in the policy gradient, the third one, Clipped Double Q-learning, aims at preventing an overestimation bias.

The use of two Q-estimators was first proposed in the Double Q-learning algorithm Hasselt (2010).

The two estimates are combined in the TD target such that determining the maximizing action is decoupled from computing the value for that action.

Later it was proposed to use the target network (whose parameters are periodically set to the current parameters or are an exponentially weighted moving average of them) for one of the two value estimates Van Hasselt et al. (2016) .

This eliminates the need to train two networks.

While this works well for discrete actions, versions of double Qlearning adapted to the actor-critic setting were shown to still suffer from an overestimation bias Fujimoto et al. (2018) .

Other approaches that aim at preventing overestimation bias in Q-learning have averaged the Q-estimates obtained from snapshots of the parameters from different training steps Anschel et al. (2017) or used a bias correction term Lee et al. (2013) .

Balancing between over-and underestimating terms in the Q-estimates has been done for a discrete action space Zhang et al. (2017) .

The work investigates multi-armed bandit problems and an underestimation bias is reported for Double Q-learning, while Q-learning with a single estimator is reported to overestimate the true values.

Similarly to our approach a weighting is introduced in the TD target.

Different to us, the weighting parameter is not learned by taking actual samples for the value estimator into account, but the parameter is set individually for each state-action pair used to train the Q-network according to a function that computes the minimum and maximum Q-value over all actions for the given state.

Finding these optimal actions for every transition on which the Q-networks are trained becomes infeasible for continuous action spaces.

Divergence of Q-values has been investigated in several recent works van Hasselt et al. (2018) Achiam et al. (2019) Fu et al. (2019) .

Of them only in Achiam et al. (2019) the case of a continuous action space is considered.

In their analysis it is investigated under which conditions a certain approximation of the Q-value updates is a contraction in the sup norm.

From that an algorithm is derived that does not need multiple critics or target networks.

The downside is that it is very compute intensive.

We consider model-free reinforcement learning for episodic tasks with continuous action spaces.

An agent interacts with its environment by selecting an action a t ??? A in state s t ??? S for every discrete time step t. The agent receives a scalar reward r t and observes the new state s t+1 .

The goal is to learn a policy ?? : S ??? A that selects the agents actions in order to maximize the sum of future discounted rewards

For a given state-action pair (s, a) the value function is defined as Q ?? (s, a) := E si???p??,ai????? [R t |s, a], which is the expected return when executing action a in state s and following ?? afterwards.

We write ?? ?? for the policy with parameters ??, that we learn in order to maximize the expected return J(??) = E si???p??,ai????? [R 0 ].

The parameters can be optimized with the gradient of J w.r.t.

the policy parameters ??.

The deterministic policy gradient Silver et al. (2014) is given by

In practice the value function Q ?? is not given and has to be approximated.

This setting is called actor-critic, the policy is the actor and the learned value function has the role of a critic.

The Q-learning algorithm Watkins (1989) tries to learn the value function with TD learning Sutton (1988) , which is an incremental update rule and aims at satisfying the Bellman equation Bellman (1957) .

Deep Q-learning Mnih et al. (2015) is a variant of this algorithm and can be used to learn the parameters ?? of an artificial neural network Q ?? : S ?? A ??? R that approximates the value function.

The network is updated by regressing its value at (s t , a t ) to its 1-step TD targets

where Q??, ???? are the target networks of Q ?? , ?? ?? and the corresponding parameters?? and?? are updated according to a exponential moving average:?? ??? ?? ?? + (1 ??? ?? )??, similarly for ??.

If a learned critic Q ?? is used in the deterministic policy gradient (eq. 1), the actor ?? ?? is updated through Q ?? , which in turn is learned with the rewards obtained from the environment.

This means that the actor requires a good critic to be able to learn a well performing policy.

Recently, it was shown, that using Q-learning in such an actor-critic setting can lead to an overestimation of the Q-values Fujimoto et al. (2018) .

This is problematic if the overestimation in the critic occurs for actions that lead to low returns.

To avoid this, Clipped Double Q-learning was proposed Fujimoto et al. (2018) .

In this approach two Q-networks (Q ??1 , Q ??2 ) are learned in parallel.

They are trained on the same TD target, which is defined via the minimum of the two Q-networks

The authors note that this can lead to an underestimation bias for the Q-values, but argue that this is not as bad as overestimating.

A big advantage of CDQ is that the Q-estimates do not explode, which otherwise can sometimes happen and is usually followed by a breakdown in performance.

Apart from that, an over-or underestimation bias would not be problematic if all values are biased by the same constant value.

It becomes a problem if the bias of the value estimates for different state-action pairs differs.

Then the critic might reinforces the wrong actions.

If this happens and in a given state an action is erroneously given a high value by the critic, the actor is reinforced to choose the corresponding action.

This increases the probability that the agent selects that action the next time when it is in that (or a similar) state.

The agent will receive a low reward, which leads to a decrease of performance.

But the critic can correct itself on the new experience which will eventually be propagated through to the actor.

If on the other hand, the critic underestimates the Q-value of a good action, the actor is trained to never try this action.

In this case the critic might never be corrected as experience opposing the critics believe is never encountered.

While this is a simplistic picture of the ongoing learning dynamics, it can give a good intuition, why both cases should be prevented if possible.

It is obvious that taking the minimum over two estimates can lead to an underestimation bias.

To check if this also occurs in practice, we conducted an experiment, where we examined the Q-value estimates of different agents.

We trained an TD3 agent that uses CDQ as defined in eq. 3 and one TD3 agent that uses instead of CDQ the critic updates of DDPG Lillicrap et al. (2015) as defined in eq. 2.

We trained on three different environments from OpenAi gym Brockman et al. (2016) .

Periodically, we sampled 1000 state-action pairs from the replay-buffer and computed the value estimate of the critic.

We approximated the true values for each state-action pair by rolling out the current policy 50 times from that pair onwards and averaged the observed discounted return.

The results for the average value of each time step are shown in the first row of Figure 1 .

Similarly to previous work Fujimoto et al. (2018) , we observe that the DDPG-style updates of the Q-network lead to an overestimation bias.

For CDQ we indeed observe an underestimation bias as the value estimates are significantly lower than the true values.

We propose Balanced Clipped Double Q-learning (BCDQ), a new algorithm to learn the critic with the goal of reducing the bias.

We adopt the idea of two Q-networks, but train them on different TD Figure 1: Measuring estimation bias in the Q-value estimates of DDPG, CDQ and Balanced Clipped Double Q-learning (BCDQ) on three different OpenAI gym environments.

The first row shows the estimates of DDPG and CDQ and it can be seen that DDPG leads to an overestimation bias, while CDQ leads to an underestimation bias.

In the second row the value estimates of BCDQ are shown.

It can be observe that the BCDQ estimates are more accurate and do not exhibit a clear bias in any direction.

Hopper-v3 HalfCheetah-v3

Figure 2: The plots show the average over 10 runs of the weighting parameter ?? for three OpenAI gym environments.

targets.

The TD target y k for the k-th Q-network Q ?? k , k ??? {1, 2} is defined as a weighted average of the network itself and the minimum of both networks

where ?? ??? [0, 1].

The first term corresponds to the TD target according to DDPG and second term corresponds to the TD target of CDQ.

While the first term tends to overestimate, the second term tends to underestimate the true Q-values.

Correctly weighting between them can correct for this bias.

However, setting ?? manually is difficult.

The perfect ?? that maximally reduces bias may change from environment to environment and also over the time of the training process.

Consequently, we adjust ?? over the course of the training.

As the goal is to minimize bias and since ?? controls in which direction more bias is introduced, we use samples of the Q-values to learn ??.

After every episode we compute for every seen state-action pair (s t , a t ) the actual discounted future return from that pair onwards R t = T i=t ?? i???t r i , which is a sample for the quantity the Q-networks Q ?? k (s t , a t ) try to estimate.

If the Q-estimates are higher than R t , they overestimated and ?? should be decreased to give the "min" term in eq. 4 more weight.

If on the other hand the Q-estimates are lower, we observe the case of underestimation and ?? should be increased.

This behaviour can be achieved by

Initialize critic networks Q ??1 , Q ??2 , and actor network ?? ?? with random parameters ?? 1 , ?? 2 , ?? Initialize target networks?? 1 ??? ?? 1 ,?? 2 ??? ?? 2 ,?? ??? ??, set k = 0 and ?? ??? [0, 1] Initialize replay buffer B for t = 1 to total timesteps do Select action with exploration noise a ??? ?? ?? (s) + , ??? N (0, ??) and observe reward r, new state s and binary value d indicating if the episode ended Store transition tuple (s, a, r, s , d) in B and set k ??? k + 1 if k ??? beta update rate and

if t mod actor delay rate = 0 then // Update ?? by the deterministic policy gradient:

minimizing the following objective w.r.t.

??:

where we restrict ?? to be in the interval [0, 1], E is the number of episodes we optimize over, s (j) t is the t-th state in the j-th considered episode (similarly for the actions a (j) t ), T j is the number of time steps in episode j and R (j) t are the future discounted returns.

The parameter ?? is updated every time the sum over all time steps in the episodes since the last update, E j=1 T j , exceeds a fixed threshold.

We set this threshold to be the maximum number of episode steps that are possible.

To optimize ?? we use stochastic gradient descent.

We note that learning ?? increases the computational complexity only minimal, as it is just one parameter that has to be optimized.

To evaluate the objective in eq. 5, a further forward path through the Q-network is performed, but no backward path is needed in the training.

We evaluated the accuracy of the value estimates of BCDQ and report the results in the second row of Figure 1 .

It can be seen, that compared to the other methods BCDQ approximates the true Qvalues much better.

This indicates that the weighting parameter ?? can indeed be adjusted over the course of the training such that the two opposing biases cancel each other out.

The behaviour of ?? is visualized in Figure 2 as an average over 10 runs per environment.

For the Hopper task it can be observed that after some time the parameter ?? gets very close to zero, which corresponds to using only CDQ to update the critic.

For HalfCheetah, the CDQ term is not weighted very high.

This is explainable as in Figure 1 it can be seen that CDQ induces a large bias on this

-2.66 ?? 0.17 -2.68 ?? 0.16 -6.75 ?? 0.48 -3.61 ?? 0.23 task.

Adjusting ?? over time allows to put more weight on the term that currently gives a more accurate estimate.

This prevents the accumulation of errors introduced by bootstrapping in the TD target.

From the plots it can also be seen that treating ?? as an hyperparameter might be difficult, as it would have to be tuned for every environment.

Furthermore, leaving ?? fixed could not account for the changing learning dynamics over the course of training.

In Figure 2 it can be seen that different drifts exist in each environment.

For example in Walker2d the learned ?? decreases on average after an initial stabilization period.

Since the two Q-networks are not trained on the same target as it is the case for CDQ, the difference between the predictions of the two Q-networks will be higher.

This suggest that -similarly to ensemble methods -the average of the two predictions might be an even better estimator.

Following that rationale, in our algorithm the critic that teaches the actor is the average of the predictions of the two Q-networks.

As a result of the above discussion, we propose the Balanced Twin Delayed Deep Deterministic policy gradient algorithm (BTD3), which builds on TD3 Fujimoto et al. (2018) .

Differently to TD3, our algorithm uses BTDQ instead of CDQ to update the critics.

For the learning of the actor the predictions of the two critics are averaged instead of using only the first critic.

The BTD3 algorithm is shown in Algorithm 1.

We evaluate our algorithm on a range of challenging continuous control tasks from OpenAI Gym Brockman et al. (2016) , which makes use of the physics engine MuJoCo Todorov et al. (2012) (version 2.0).

To guarantee an apples-to-apples comparison with TD3, we extended the original source code of TD3 with our method and evaluate our algorithm with the default hyperparameters of TD3 for all tasks except for Humanoid-v3.

We observed that TD3 does not learn a successful policy on Humanoid-v3 with the default learning rate of 0.001, but we found that TD3 does learn if the learning rate for both actor and critic is reduced.

Consequently, we set it to 0.0001 for this task and did the same for BTD3.

We set the learning rate for the weighting parameter ?? to 0.05 and initialize ?? = 0.5 at the beginning of the training for all environments.

As is done in TD3, we reduce the dependency on the initial parameters by using a random policy for the first 10000 steps for HalfCheetah-v3, Ant-v3, Humanoid-v3 and the first 1000 steps for the remaining environments.

After that period we add Gaussian noise N (0, 0.1) to each action in order to ensure enough exploration.

During training the policy is evaluated every 5000 environment steps by taking the average over the episode reward obtained by rolling out the current policy without exploration noise 10 times.

For each task and algorithm we average the results of 10 trials each with a different random seed, except for Humanoid-v3, where we used 5 trials.

We compare our algorithm to the state of the art continuous control methods SAC Haarnoja et al. (2018a) (with learned temperature parameter Haarnoja et al. (2018b) ), TD3 Fujimoto et al. (2018) and to DDPG Lillicrap et al. (2015) .

For both, SAC and TD3, we used the source code published by Hopper-v3 Ant-v3

Humanoid-v3 Reacher-v2 The learning curves are shown in Figure 3 .

For all tasks BTD3 matches or outperforms TD3.

Furthermore, it performs significantly better than SAC and DDPG.

In Table 1 the results are presented in terms of the average maximum episode reward.

In order to compute that statistic, for each trial we computed the maximum over the evaluations that were executed all 5000 time steps, where the evaluations are itself the average over 10 rollouts of the current policy.

Afterwards, we computed the average of this value over the different trials.

The results show that the best policies of BTD3 achieve significantly higher episode rewards than the best policies of the other methods.

To further understand the influence of the dynamic weighting scheme we trained BTD3 with a fixed value for ??.

We evaluated for the values ?? ??? {0.00, 0.25, 0.50, 0.75, 1.00}, where ?? = 0.00 corresponds to TD3 and ?? = 1.00 corresponds to DDPG.

The averaged results over 10 runs are shown in Figure 4 .

From the plots we can make two observations.

First, it is essential that ?? is adjusted during the training.

For any of the considered values of ?? leaving it fixed leads to a worse performance compared to BTD3 and in most cases also worse than TD3.

In Figure 2 it was shown that the adjusted weighting parameter is on average over many runs attracted to different values depending not only on the environment but also on the timestep during training.

The dynamic adjustment to prevent accumulating errors is not possible when ?? is fixed.

Second, it is surprising that fixed values for ?? that would seem promising to try from Figure 2 can perform worse than other fixed values.

For example inspecting the plots in Figure 2 the value ?? = 0.75 seems a good fit for the HalfCheetah environment.

But the evaluation shows that ?? = 0.25 and ?? = 0.50 perform better.

This further supports the hypothesis that the most important part about BCDQ is the dynamic adjustment of the weighting parameter.

Figure 4: Learning curves for four different continuous control tasks from OpenAi gym over 10 random seeds each.

The show algorithms are BTD3 and versions of it with a fixed value of ??.

For each algorithm the curves show the mean over 10 runs with different random seeds and are filtered with a uniform filter of size 15.

We showed that Clipped Double Q-learning (CDQ) induces an underestimation bias in the critic, while an overestimation bias occurs if just one Q-network is used.

From that we derived the Balanced Clipped Double Q-learning algorithm (BCDQ) that updates the critic through a weighted average of the two mentioned update mechanisms.

The weighting parameter is adjusted over the course of training by comparing the Q-values of recently visited state-action pairs with the actual discounted return observed from that pair onwards.

It was shown that BCDQ achieves much more accurate value estimates by adjusting the weighting parameter.

Replacing CDQ with BCDQ leads to the Balanced Twin Delayed Deep Deterministic policy gradient algorithm (BTD3).

Our method achieves state of the art performance on a range of continuous control tasks.

Furthermore, BCDQ can be added to any other actor-critic algorithm while it only minimally increases the computational complexity compared to CDQ.

It is also be possible to use BCDQ for discrete action spaces.

Evaluating that approach is an interesting area for future research.

@highlight

A method for more accurate critic estimates in reinforcement learning.
Combining multiple function approximators in machine learning models typically leads to better performance and robustness compared with a single function.

In reinforcement learning, ensemble algorithms such as an averaging method and a majority voting method are not always optimal, because each function can learn fundamentally different optimal trajectories from exploration.

In this paper, we propose a Temporal Difference Weighted (TDW) algorithm, an ensemble method that adjusts weights of each contribution based on accumulated temporal difference errors.

The advantage of this algorithm is that it improves ensemble performance by reducing weights of Q-functions unfamiliar with current trajectories.

We provide experimental results for Gridworld tasks and Atari tasks that show significant performance improvements compared with baseline algorithms.

Using ensemble methods that combine multiple function approximators can often achieve better performance than a single function by reducing the variance of estimation (Dietterich (2000) ; Kuncheva (2014) ).

Ensemble methods are effective in supervised learning, and also reinforcement learning (Wiering & Van Hasselt (2008) ).

There are two situations where multiple function approximators are combined: combining and learning multiple functions during training (Freund & Schapire (1997) ) and combining individually trained functions to jointly decide actions during testing (Breiman (1996) ).

In this paper, we focus on the second setting of reinforcement learning wherein each function is trained individually and then combined them to achieve better test performance.

Though there is a body of research on ensemble algorithms in reinforcement learning, it is not as sizeable as the research devoted to ensemble methods for supervised learning.

Wiering & Van Hasselt (2008) investigated many ensemble approaches combining several agents with different valuebased algorithms in Gridworld settings.

Faußer & Schwenker (2011; 2015a) have shown that combining value functions approximated by neural networks improves performance greater than using a single agent.

Although previous work dealt with each agent equally contributing to the final output, weighting each contribution based on its accuracy is also a known and accepted approach in supervised learning (Dietterich (2000) ).

However, unlike supervised learning, reinforcement learning agents learn from trajectories resulting from exploration, such that each agent learns from slightly different data.

This characteristic is significant in tasks with high-dimensional state-space, where there are several possible optimal trajectories to maximize cumulative rewards.

In such a situation, the final joint policy function resulting from simple averaging or majority voting is not always optimal if each agent learned different optimal trajectories.

Furthermore, it is difficult to decide constant weights of each contribution as it is possible that agents with poor episode rewards have better performance in specific areas.

In this paper, we propose the temporal difference weighted (TDW) algorithm, an ensemble method for reinforcement learning at test time.

The most important point of this algorithm is that confident agents are prioritized to participate in action selection while contributions of agents unfamiliar with the current trajectory are reduced.

To do so in the TDW algorithm, the weights of the contributions at each Q-function are calculated as softmax probabilities based on accumulated TD errors.

Extending an averaging method and a majority voting method, actions are determined by weighted average or voting methods according to the weights.

The advantage of the TDW algorithm is that arbitrary training algorithms can use this algorithm without any modifications, because the TDW algorithm only cares about the joint decision problem, which could be easily adopted in competitions and development works using reinforcement learning.

In our experiment, we demonstrate that the TDW retains performance in tabular representation Gridworld tasks with multiple possible trajectories, where simple ensemble methods are significantly degraded.

Second, to demonstrate the effectiveness of our TDW algorithm in high-dimensional state-space, we also show that our TDW algorithm can achieve better performance than baseline algorithms in Atari tasks (Bellemare et al. (2013) ).

Ensemble methods that combine multiple function approximators during training rather than evaluation have been studied in deep reinforcement learning.

Bootstrapped deep Q-network (DQN) (Osband et al. (2016) ) leverages multiple heads that are randomly initialized to improve exploration, because each head leads to slightly different states.

Averaged-DQN (Anschel et al. (2017) ) reduces the variance of a target approximation by calculating an average value of last several learned Qnetworks.

Using multiple value functions to reduce variance of target estimation is also utilized in the policy gradients methods (Fujimoto et al. (2018) ; Haarnoja et al. (2018) ).

In contrast, there has been research focused on joint decision making in reinforcement learning.

Using multiple agents to jointly select an action achieves better performance than a single agent (Wiering & Van Hasselt (2008); Faußer & Schwenker (2015a; ).

However, such joint decision making has been limited to relatively small tasks such as Gridworld and Maze.

Therefore, it is not known whether joint decision making with deep neural networks can improve performance in high-dimensional state-space tasks such as Atari 2600 (Bellemare et al. (2013) ).

Doya et al. (2002) proposes a multiple mode-based reinforcement learning (MMRL), a weighted ensemble method for model-based reinforcement learning, which determines each weight based by using prediction models.

The MMRL gives larger weights to reinforcement learning controllers with small errors of special responsibility predictors.

Unlike MMRL, our method does not require additional components to calculate weights.

Our method is not the first one to use TD errors in combining multiple agents.

Ring & Schaul (2011) proposes a module selection mechanism that chooses the module with smallest TD errors to learn current states, which will eventually assign each module to a small area of a large task.

As a joint decision making method, a selective ensemble method is proposed to eliminate agents with less confidence at the current state by measuring TD errors (Faußer & Schwenker (2015b) ), which is the closest approach to our method.

This selection drops all outputs whose TD errors exceeds a threshold, which can be viewed as a hard version of our method that uses a softmax of all weighted outputs instead of elimination.

The threshold is not intuitively determined.

Because the range of TD errors varies by tasks and reward settings, setting the threshold requires sensitive tuning.

We formulate standard reinforcement learning setting as follows.

At time t, an agent receives a state s t ∈ S, and takes an action a t ∈ A based on a policy function a t = π(s t ).

The next state s t+1 is given to the agent along with a reward r t+1 .

The return is defined as a discounted cumulative reward

, where γ ∈ [1, 0] is a discount factor.

The true value of taking an action a t at a state s t is described as follows:

where Q π (s t , a t ) is an action-value under the policy π.

The optimal value is Q * (s t , a t ) = max π Q π (s t , a t ).

With such an optimal Q-function, optimal actions can be determined based on the highest action-values at each state.

DQN (Mnih et al. (2015) ) is a deep reinforcement learning method that approximates an optimal Qfunction with deep neural networks.

The Q-function Q(s t , a t |θ) with a parameter θ is approximated by a Q-learning style update (Watkins & Dayan (1992) ).

The parameter θ is learned to minimize squared temporal difference errors.

where y t = r t+1 + γ max a Q(s t+1 , a|θ ) with a target network parameter θ .

The target network parameter θ is synchronized to the parameter θ in a certain interval.

DQN also introduces use of the experience replay (Lin (1992) ), which randomly samples past state transitions from the replay buffer to compute the squared TD error (1).

Assume there are N sets of trained Q-function Q(s, a|θ i ) where i denotes an index of the function.

The final policy π(s t ) is determined by combining the N Q-functions.

We formulate two baseline methods commonly used in ensemble algorithms: Average policy and Majority Voting (MV) policy (Faußer & Schwenker (2011; 2015a) ; Kuncheva (2014)).

Majority Voting (MV) policy is an approach to decide the action based on greedy selection according to the formula:

where v i (s, a) is a binary function that outputs 1 for the most valued action and 0 for others:

Contributions of each function to the final output are completely equal.

Average policy is a method that averages all the outputs of the Q-functions, and the action is greedily determined:

Averaging outputs from multiple approximated functions reduces variance of prediction.

Unlike MV policy, Average policy leverages all estimated values as well as the highest values.

In this section, we explain the TDW ensemble algorithm that adjusts weights of contributions based on accumulated TD errors.

The TDW algorithm is especially powerful in the complex situation such as high-dimensional state-space where it is difficult to cover whole state-space with a single agent.

Section 4.1 describes the error accumulation mechanism.

Section 4.2 introduces joint action selection using the weights computed with the accumulated errors.

We consider that a squared TD error δ

2 fundamentally consists of two kinds of errors:

where δ p is a prediction error of approximated function, and δ u is an error at states where the agent rarely experienced.

In a tabular-based value function, δ p will be near 0 at frequently visited states.

In contrast, δ u will be extremely large at less visited states with both a tabular-based value function and a function approximator because TD errors are not sufficiently propagated such a state.

There are two causes of unfamiliar states: (1) states are difficult to visit due to hard exploration, and (2) states are not optimal to the agent according to learned state transitions.

For combining multiple agents at a joint decision, the second case is noteworthy because each agent may be optimized at different optimal trajectories.

Thus, some of the agents will produce larger δ u when they face such states as a result of an ensemble, and contributions of less confident agents can be reduced based on the TD error δ u . (8) or (9).

end for

To measure uncertainty of less confident agents, we define u i t as a uncertainty of an agent:

where α ∈ [0, 1] is a constant factor decaying the uncertainty at a previous step.

With a large α, the uncertainty u i is extremely large during unfamiliar trajectories, which makes it possible to easily distinguish confident agents from the others.

However, a trade-off arises when prediction error δ p is accumulated for a long horizon, which increases correlation between agents.

To reduce contributions of less confident agents, each contribution at joint decision is weighted based on uncertainty u i t .

Using the uncertainty u i t , a weight w i t of each agent is calculated as a probability by the softmax function:

When the agent has a small uncertainty value u i t , the weight w i t becomes large.

We consider two weighted ensemble methods corresponding to the Average policy and the MV policy based on the weights w i t .

As a counterpart of the Average policy, our TDW Average policy is as follows:

For the MV policy, TDW Voting policy is as follows:

Unlike the averaging method, because TDW Voting policy directly uses probabilities calculated by (7), the correlation between agents can be increased significantly with large decay factor α, leading to worse performance.

Although these weighted ensemble algorithms are simple enough to extend to arbitrary ensemble methods, we leave more advanced applications for future work so that we may demonstrate the effectiveness of our approach in a simpler setting.

The complete TDW ensemble algorithm is described in Algorithm 1.

In this section, we describe the experiments performed on the Gridworld tasks and Atari tasks (Bellemare et al. (2013) ) in Section 5.2.

To build the trained Q-functions, we used the table-based Qlearning algorithm (Watkins & Dayan (1992) ) and DQN (Mnih et al. (2015) ) with a standard model, respectively.

In each experiment, we evaluated our algorithm to address performance improvements from there baselines as well as the effects of selecting the decay factor α.

We first evaluated the TDW algorithms with a tabular representation scenario to show their effectiveness in the situation where it is difficult to cover a whole state-space with the single agent.

We built two Gridworld environments as shown in Figure 1 .

Each environment is designed to induce bias of learned trajectories by setting multiple slits.

As a result of exploration, once an agent gets through one of the slits to the goal, the agent is easily biased to aim for the same slit due to the max operator of Q-learning.

The state-representation is a discrete index of a table with size of 13 × 13.

There are four actions corresponding to steps of up, down, left and right.

If a wall exists where the agent tries to move, the next state remains the same as the current state.

The agent always starts from S depicted in Figure  1 .

At every timestep, the agent receives a reward of −0.1 or +100 at goal states.

The agent starts a new episode if either the agent arrives at the goal states or the timestep reaches 100 steps.

We trained N = 10 agents with different random seeds for -greedy exploration with = 0.3.

Each training continues until 1M steps have been simulated.

We set the learning rate to 0.01 and γ = 0.95.

After training, we evaluated TDW ensemble algorithms for 20K episodes.

As baselines, we also evaluate each single agent as well as ensemble methods of Average policy and MV policy for 20K episodes each.

The evaluation results on the Gridworld environments are shown in Table 1 .

Four-slit Gridworld is significantly more difficult than Two-slit Gridworld because each Q-function is not only horizontally biased, but also vertically biased.

In both the Two-slit Gridworld and Four-slit Gridworld environments, the TDW ensemble methods achieve better performance than their corresponding Average policy and the MV policy baselines.

Additionally, the results of both of the Average policy and the MV policy were worse than the single models.

It should be noted that Average policy degrades original performance more than MV policy.

For the selection of the decay factor α, a larger α tends to increase performance in TDW Average policy.

In contrast, the larger α leads to poor performance in TDW Voting policy especially in Fourslit Gridworld.

We believe that the large α significantly reduces contributions of most Q-functions, which would ignore votes of actions that would be the best in equal voting.

In contrast, TDW Average policy leverages values of all actions, exploiting all contributions to select the best action.

To demonstrate effectiveness in high-dimensional state-space, we evaluated TDW algorithm in Atari tasks.

We trained DQN agents across 6 Atari tasks (Asterix, Beamrider, Breakout, Enduro, MsPacman and SpaceInvaders) through OpenAI Gym (Brockman et al. (2016) ).

At each task, N = 10 agents were trained with different random seeds for neural network initialization, exploration and environments in order to vary the learned Q-function.

The training continued until 10M steps (40M game frames) with frame skipping and termination on loss of life enabled.

The of exploration is linearly decayed from 1.0 to 0.1 through 1M steps.

The hyperparameters of neural networks are same as (Mnih et al. (2015) ).

After training, evaluation was conducted with each Q-function, TDW Average policy, TDW Voting policy and the two baselines.

We additionally evaluated weighted versions of the baselines whose Q-functions were weighted based on their evaluation performance.

The evaluation continued for 1000 episodes with = 0.05.

The experimental results are shown in Table 2 .

Interestingly, both of Average policy and MV policy improved performance from mean performance of single agents, though the simple ensemble algorithms had not been investigated well in the domain of deep reinforcement learning.

In the games of Asterix, Beamrider, Breakout and Enduro, the TDW algorithms achieve additional performance improvements as compared with the non-weighted and weighted baselines.

Even in MsPacman and SpaceInvaders, the TDW algorithms perform significantly better than non-weighted baselines and the best single models.

In most of the cases, the globally weighted ensemble baselines performed worse than non-weighted versions.

We believe this is because these globally weighted ensemble methods will ignore local performance, which is significant in high-dimensional state-space because it is difficult to cover all possible states with single models.

The TDW algorithms with small α tend to achieve better performance than those with a large α, which suggests that a significantly large α can increase correlation between Q-functions and reduce contributions of less confident Q-functions.

To analyze changes of weights through an episode, we plot entropies during a sample episode on Breakout (TDW Average policy, α = 0.8) in Figure 2 (a) .

If an entropy is low (high in negative scale), some of Q-functions have large weights, while others have extremely small weights.

Extreme low entropies are observed when the ball comes closely to the pad as shown in Figure 2 (b) where the value should be close to 0 for the non-optimal actions because missing the ball immediately ends its life.

It is easy for sufficiently learned Q-functions to estimate such a terminal state so that the entropy becomes low due to the gap between optimal Q-functions for the current states and the others.

The entropies tend to be low during latter steps as there are many variations of the remaining blocks.

We consider that the reason why the TDW algorithms with α = 0.8 achieved the best performance in Breakout is that the large α value reduces influence of the Q-functions which cannot correctly predict values with unseen variations of remaining blocks.

In contrast to Breakout where living long leads to higher scores, in SpaceInvaders we observe that the low entropies appear at dodging beams rather than shooting invaders, because shooting beams requires long-term value prediction which does not induce large TD errors.

Therefore, performance improvements on SpaceInvaders are not significantly better than weighted baselines.

To analyze correlation between the decay factor α and entropies, plots of the number of observations with a certain entropy are shown in Figure 3 .

In most games, higher decay factors increase the presence of low entropy states and decreases the presence of high entropy states.

In the games with frequent reward occurences such as Enduro and MsPacman, there are more low-entropy observations than BeamRider and SpaceInvaders, where reward occurences are less frequent.

Especially with regards to MsPacman, we believe that the TDW Average policy with larger α values results in worse performance because the agent frequently receives positive rewards at almost every timestep, which often induces prediction error δ p , and increases uncertainty in all Q-functions.

Thus, globally weighted ensemble methods achieve better performance than TDW algorithms because it is difficult to consistently accumulate uncertainties on MsPacman.

In this paper, we have introduced the TDW algorithm: an ensemble method that accumulates temporal difference errors as an uncertainties in order to adjust weights of each Q-function, improving performance especially in high-dimensional state-space or situations where there are multiple optimal trajectories.

We have shown performance evaluations in Gridworld tasks and Atari tasks, wherein the TDW algorithms have achieved significantly better performance than non-weighted algorithms and globally weighted algorithms.

However, it is difficult to correctly measure uncertainties with frequent reward occurrences because the intrinsic prediction errors are also accumulated.

Thus, these types of games did not realize the same performance improvements.

In future work, we intend to investigate an extension of this work into continuous action-space tasks because only the joint decision problem of Q-functions is considered in this paper.

We believe a similar algorithm can extend a conventional ensemble method (Huang et al. (2017) ) of Deep Deterministic Policy Gradients (Lillicrap et al. (2015) ) by measuring uncertainties of pairs of a policy function and a Q-function.

We will also consider a separate path, developing an algorithm that measures uncertainties without rewards because reward information is not always available especially in the case of real world application.

A Q-FUNCTION TABLES OBTAINED ON GRIDWORLDS   table 1   0   20   40

60   80   100   table 2   0   20   40

60   80   100   table 3   0   20   40

60   80   100   table 4   0   20   40

60   80   100   table 5   0

<|TLDR|>

@highlight

Ensemble method for reinforcement learning that weights Q-functions based on accumulated TD errors.
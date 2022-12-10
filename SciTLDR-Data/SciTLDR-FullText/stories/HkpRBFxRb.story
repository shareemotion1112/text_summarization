Reinforcement Learning (RL) can model complex behavior policies for goal-directed sequential decision making tasks.

A hallmark of RL algorithms is Temporal Difference (TD) learning: value function for the current state is moved towards a bootstrapped target that is estimated using the next state's value function.

lambda-returns define the target of the RL agent as a weighted combination of rewards estimated by using multiple many-step look-aheads.

Although mathematically tractable, the use of  exponentially decaying weighting of n-step returns based targets in lambda-returns is a rather ad-hoc design choice.

Our major contribution  is that we propose a generalization of lambda-returns called Confidence-based Autodidactic Returns (CAR), wherein the RL agent learns the weighting of the n-step returns in an end-to-end manner.

In contrast to lambda-returns wherein the RL agent is restricted to use an exponentially decaying weighting scheme, CAR allows the agent to learn to decide how much it wants to weigh the n-step returns based targets.

Our experiments, in addition to showing the efficacy of CAR, also empirically demonstrate that using sophisticated weighted mixtures of multi-step returns (like CAR and lambda-returns) considerably outperforms the use of n-step returns.

We perform our experiments on the  Asynchronous Advantage Actor Critic (A3C) algorithm in the Atari 2600 domain.

Reinforcement Learning (RL) BID21 ) is often used to solve goal-directed sequential decision making tasks wherein conventional Machine Learning methods such as supervised learning are not suitable.

Goal-directed sequential decision making tasks are modeled as Markov Decision Process (MDP) BID11 .

Traditionally, tabular methods were extensively used for solving MDPs wherein value function or policy estimates were maintained for every state.

Such methods become infeasible when the underlying state space of the problem is exponentially large or continuous.

Traditional RL methods have also used linear function approximators in conjunction with hand-crafted state spaces for learning policies and value functions.

This need for hand-crafted task-specific features has limited the applicability of RL, traditionally.

Recent advances in representation learning in the form of deep neural networks provide us with an effective way to achieve generalization BID1 BID6 .

Deep neural networks can learn hierarchically compositional representations that enable RL algorithms to generalize over large state spaces.

The use of deep neural networks in conjunction with RL objectives has shown remarkable results such as learning to solve the Atari 2600 tasks from raw pixels BID0 BID8 BID16 BID4 , learning to solve complex simulated physics tasks BID24 BID13 BID7 and showing super-human performance on the ancient board game of Go .

Building accurate and powerful (in terms of generalization capabilities) state and action value function BID21 estimators is important for successful RL solutions.

This is because many practical RL solutions (Q-Learning (Watkins & Dayan, 1992) , SARSA (Rummery & Niranjan, 1994) and Actor-Critic Methods BID5 ) use Temporal Difference (TD) Learning BID20 .

In TD learning, a n-step return is used as an estimate of the value function by means of bootstrapping from the n th state's value function estimate.

On the other hand, in Monte Carlo learning, the cumulative reward obtained in the entire trajectory following a particular state is used as an estimate for the value function of that state.

The ability to build better estimates of the value functions directly results in better policy estimates as well as faster learning.

λ-returns (LR) BID21 are very effective in this regard.

They are effective for faster propagation of delayed rewards and also result in more reliable learning.

LR provide a trade-off between using complete trajectories (Monte Carlo) and bootstrapping from n-step returns (TD learning).

They model the TD target using a mixture of n-step returns, wherein the weights of successively longer returns are exponentially decayed.

With the advent of deep RL, the use of multi-step returns has gained a lot of popularity BID9 .

However, it is to be noted that the use of exponentially decaying weighting for various n-step returns seems to be an ad-hoc design choice made by LR.

In this paper, we start off by extensively benchmarking λ-returns (our experiments only use truncated λ-returns due to the nature of the DRL algorithm (A3C) that we work with and we then propose a generalization called the Confidence-based Autodidactic Returns (CAR), In CAR, the DRL agent learns in an end-to-end manner, the weights to assign to the various n-step return based targets.

Also in CAR, it's important to note that the weights assigned to various n-step returns change based on the different states from which bootstrapping is done.

In this sense, CAR weights are dynamic and using them represents a significant level of sophistication as compared to the usage of λ-returns.

In summary, our contributions are:1.

To alleviate the need for some ad-hoc choice of weights as in the case of λ-returns, we propose a generalization called Autodidactic Returns and further present a novel derivative of it called Confidence-based Autodidactic Returns (CAR) in the DRL setting.2.

We empirically demonstrate that using sophisticated mixtures of multi-step return methods like λ-returns and Confidence-based Autodidactic Returns leads to considerable improvement in the performance of a DRL agent.3.

We analyze how the weights learned by CAR are different from that of λ-returns, what the weights signify and how they result in better estimates for the value function.

In this section, we present some basic concepts required to understand our work.

An MDP (Puterman, 2014) is defined as the tuple S, A, r, P, γ , where S is the set of states in the MDP, A is the set of actions, r :

S × A → R is the reward function, P : S × A × S → [0, 1] is the transition probability function such that s p(s, a, s ) = 1, p(s, a, s ) ≥ 0, and γ ∈ [0, 1) is the discount factor.

We consider a standard RL setting wherein the sequential decision-making task is modeled as an MDP and the agent interacts with an environment E over a number of discrete time steps.

At a time step t, the agent receives a state s t and selects an action a t from the set of available actions A. Given a state, the agent could decide to pick its action stochastically.

Its policy π is in general a mapping defined by: π : S × A → [0, 1] such that a∈A π(s, a) = 1, π(s, a) ≥ 0 ∀s ∈ S, ∀a ∈ A.

At any point in the MDP, the goal of the agent is to maximize the return, defined as: DISPLAYFORM0 γ k r t+k which is the cumulative discounted future reward.

The state value function of a policy π, V π (s) is defined as the expected return obtained by starting in state s and picking actions according to π.

Actor Critic algorithms BID5 are a class of approaches that directly parameterize the policy (using an actor) π θa (a|s) and the value function (using a critic) V θc (s).

They update the policy parameters using Policy Gradient Theorem BID22 BID17 based objective functions.

The value function estimates are used as baseline to reduce the variance in policy gradient estimates.

Asynchronous Advantage Actor Critic(A3C) BID9 ) introduced the first class of actor-critic algorithms which worked on high-dimensional complex visual input space.

The key insight in this work is that by executing multiple actor learners on different threads in a CPU, the RL agent can explore different parts of the state space simultaneously.

This ensures that the updates made to the parameters of the agent are uncorrelated.

The actor can improve its policy by following an unbiased low-variance sample estimate of the gradient of its objective function with respect to its parameters, given by: DISPLAYFORM0 In practice, G t is often replaced with a biased lower variance estimate based on multi-step returns.

In the A3C algorithm n-step returns are used as an estimate for the target G t , where n ≤ m and m is a hyper-parameter (which controls the level of rolling out of the policies).

A3C estimates G t as: DISPLAYFORM1 and hence the objective function for the actor becomes: DISPLAYFORM2 is the j-step returns based TD error.

The critic in A3C models the value function V (s) and improves its parameters based on sample estimates of the gradient of its loss function, given as: DISPLAYFORM3

Weighted average of n-step return estimates for different n's can be used for arriving at TD-targets as long as the sum of weights assigned to the various n-step returns is 1 BID21 ).In other words, given a weight vector DISPLAYFORM0 , and n-step returns for n ∈ {1, 2, · · · , h}: G DISPLAYFORM1 t , we define a weighted return as DISPLAYFORM2 Note that the n-step return G (n) t is defined as: DISPLAYFORM3

A special case of G w t is G λ t (known as λ-returns) which is defined as: DISPLAYFORM0 What we have defined here are a form of truncated λ-returns for TD-learning.

These are the only kind that we experiment with, in our paper.

We use truncated λ-returns because the A3C algorithm is designed in a way which makes it suitable for extension under truncated λ-returns.

We leave the problem of generalizing our work to the full λ-returns as well as eligibility-traces (λ-returns are the forward view of eligibility traces) to future work.3 λ-RETURNS AND BEYOND: AUTODIDACTIC RETURNS

Autodidactic returns are a form of weighted returns wherein the weight vector is also learned alongside the value function which is being approximated.

It is this generalization which makes the returns autodidactic.

Since the autodidactic returns we propose are constructed using weight vectors that are state dependent (the weights change with the state the agent encounters in the MDP), we denote the weight vector as w(s t ).

The autodidactic returns can be used for learning better approximations for the value functions using the TD(0) learning rule based update equation: DISPLAYFORM0 In contrast with autodidactic returns, λ-returns assign weights to the various n-steps returns which are constants given a particular λ.

We reiterate that the weights assigned by λ-returns don't change during the learning process.

Therefore, the autodidactic returns are a generalization and assign weights to returns which are dynamic by construction.

The autodidactic weights are learned by the agent, using the reward signal it receives while interacting with the environment.

All the n-step returns for state s t are estimates for V (s t ) bootstrapped using the value function of corresponding n th future state (V (s t+n )).

But all those value functions are estimates themselves.

Hence, one natural way for the RL agent to weigh an n-step return G (n) t would be to compute this weight using some notion of confidence that the agent has in the value function estimate, V (s t+n ), using which the n-step return was estimated.

The agent can weigh the n-returns based on how confident it is about bootstrapping from V (s t+n ) in order to obtain a good estimate for V (s t ).

We denote this confidence on V (s t+n ) as c(s t+n ).

Given these confidences, the weight vector w(s t ) can computed as: DISPLAYFORM0 where w(s t ) (i) is given by: DISPLAYFORM1 The idea of weighing the returns based on a notion of confidence has been explored earlier BID26 BID23 .

In these works, learning or adapting the lambda parameter based on a notion of confidence/certainty under the bias-variance trade-off has been attempted, but the reason why only a few successful methods have emerged from that body of work is due to the difficult of quantifying, measuring and optimizing this certainty metric.

In this work, we propose a simple and robust way to model this and we also address the question of what it means for a particular state to have a high value of confidence and how this leads to better estimates of the value function.

λ-returns have been well studied in literature BID10 BID21 BID15 and have been used in DRL setting as well BID14 BID2 .

We propose a straightforward way to incorporate (truncated) λ-returns into the A3C framework.

We call this combination as LRA3C.The critic in A3C uses n-step return for arriving at good estimates for the value function.

However, note that the TD-target can in general be based on any n-step return (or a mixture thereof).

The A3C algorithm in specific is well suited for using weighted returns such as λ-returns since the algorithm already uses n-step return for bootstrapping.

Using eqs.(1) to (3) makes it very easy to incorporate weighted returns into the A3C framework.

The sample estimates for the gradients of the actor and the critic respectively become: DISPLAYFORM0 Figure 1: CARA3C network -Confidence-based weight vector calculation for state s 1 .

We propose to use autodidactic returns in place of normal n-step returns in the A3C framework.

We call this combination as CARA3C.

In a generic DRL setup, a forward pass is done through the network to obtain the value function of the current state.

The parameters of the network are progressively updated based on the gradient of the loss function and the value function estimation (in general) becomes better.

For predicting the confidence values, a distinct neural network is created which shares all but the last layer with the value function estimation network.

So, every forward pass of the network on state s t now outputs the value function V (s t ) and the confidence the network has in its value function prediction, c(s t ).

Figure 1 shows the CARA3C network unrolled over time and it visually demonstrates how the confidence values are calculated using the network.

Next, using eqs. FORMULA7 to (5) the weighted average of n-step returns is calculated and used as a target for improving V (s t ).

Algorithm 1, in Appendix F , presents the detailed pseudo-code for training a CARA3C agent.

The policy improvement is carried out by following sample estimates of the loss function's gradient, given by: ∇ θa log π θa (a t |s t )δ t , where δ t is now defined in terms of the TD error term obtained by using autodidactic returns as the TD-target.

Overall, the sample estimates for the gradient of the actor and the critic loss functions respectively are: DISPLAYFORM0

The LSTM-A3C neural networks for representing the policy and the value function share all but the last output layer.

In specific, the LSTM BID3 controller which aggregates the observations temporally is shared by the policy and the value networks.

As stated in the previous sub-section, we extend the A3C network to predict the confidence values by creating a new output layer which takes as input the LSTM output vector (LSTM outputs are the pre-final layer).

Figure 1 contains a demonstration of how w(s 1 ) is computed.

Since all the three outputs (policy, value function, confidence on value function) share all but the last layer, G w(st) t depends on the parameters of the network which are used for value function prediction.

Hence, the autodidactic returns also influence the gradients of the LSTM controller parameters.

However, it was observed that when the TD target, G w t , is allowed to move towards the value function prediction V (s t ), it makes the learning unstable.

This happens because the L 2 loss between the TD-target and the value function prediction can now be minimized by moving the TD-target towards erroneous value function predictions V (s t ) instead of the other way round.

To avoid this instability we ensure that gradients do not flow back from the confidence values computation's last layer to the LSTM layer's outputs.

In effect, the gradient of the critic loss with respect to the parameters utilized for the computation of the autodidactic return can no longer influence the gradients of the LSTM parameters (or any of the previous convolutional layers).

To summarize, during back-propagation of gradients in the A3C network, the parameters specific to the computation of the autodidactic return do not contribute to the gradient which flows back into the LSTM layer.

This ensures that the parameters of the confidence network are learned while treating the LSTM outputs as fixed feature vectors.

This entire scheme of not allowing gradients to flow back from the confidence value computation to the LSTM outputs has been demonstrated in Figure 1 .

The forward arrows depict the parts of the network which are involved in forward propagation whereas the backward arrows depict the path taken by the back-propagation of gradients.

We performed general game-play experiments with CARA3C and LRA3C on 22 tasks in the Atari domain.

All the networks were trained for 100 million time steps.

The hyper-parameters for each of the methods were tuned on a subset of four tasks: Seaquest, Space Invaders, Gopher and Breakout.

The same hyper-parameters were used for the rest of the tasks.

The baseline scores were taken from BID16 .

All our experiments were repeated thrice with different random seeds to ensure that our results were robust to random initialization.

The same three random seeds were used across experiments and all results reported are the average of results obtained by using these three random seeds.

Since the A3C scores were taken from BID16 , we followed the same training and testing regime as well.

Appendix A contains experimental details about the training and testing regimes.

Appendix G documents the procedure we used for picking important hyper-parameters for our methods.

Evolution of the average performance of our methods with training progress has been shown in FIG0 .

An expanded version of the graph for all the tasks can be found in Appendix C. TAB0 Figure 3: Percentage improvement achieved by CARA3C and LRA3C over A3C.shows the mean and median of the A3C normalized scores of CARA3C and LRA3C.

If the scores obtained by one of the methods and A3C in a task are p and q respectively, then the A3C normalized score is calculated as: p q .

As we can see, both CARA3C and LRA3C improve over A3C with CARA3C doing the best: on an average, it achieves over 4× the scores obtained by A3C.

The raw scores obtained by our methods against A3C baseline scores can be found in TAB1 (in Appendix B).

Figure 3 shows the percentage improvement achieved by sophisticated mixture of n-step return methods (CARA3C and LRA3C) over A3C.

If the scores obtained by one of the methods and A3C in a task are p and q respectively, then the percentage improvement is calculated as: p−q q × 100 .

As we can see, CARA3C achieves a staggering 67× performance in the task Kangaroo.

LRA3C ) to various n-step returns over the duration of an episode by fully trained DRL agents.

The four games shown here are games where CARA3C achieves large improvements in performance over LRA3C.

It can be seen that for all the four tasks, the difference in weights evolve in a dynamic fashion as the episode goes on.

The motivation behind this analysis is to understand how different the weights used by CARA3C are as compared to LRA3C and as it can be seen, it's very different.

The agent is clearly able to weighs it's returns in a way that is very different from LRA3C and this, in fact, seems to give it the edge over both LRA3C and A3C in many games.

These results once again reiterates the motivation for using dynamic Autodidactic Returns.

An expanded version of the graphs for all the tasks can be found in Appendix D. Figure 5 : Relation between the confidence assigned to a state and the percent change in their value estimate.

Percentage change in value estimates were obtained by calculating the value estimate of a state just before and after a batch gradient update step.

In Figure 5 each bin (x, y) denotes the average confidence value assigned by the network to the states that were encountered during the training time between y million and y + 1 million steps and whose value estimates were changed by a value between x and x + 1 percent, where 0 ≤ x < 100 and 0 ≤ y < 100.

In all the graphs we can see that during the initial stages of the training(lower rows of the graph), the confidence assigned to the states is approximately equal irrespective of the change is value estimate.

As training progresses the network learns to assign relatively higher confidence values to the states whose value function changes by a small amount than the ones whose value function changes more.

So, the confidence value can be interpreted as a value that quantifies the certainty the network has on the value estimate of that state.

Thus, weighing the n-step returns based on the confidence values will enable the network to bootstrap the target value better.

The confidence value depends on the certainty or the change in value estimate and it is not the other way round, i.e., having high confidence value doesn't make the value estimate of that state to change less.

This is true because the confidence value can not influence the value estimate of a state as the gradients obtained from the confidence values are not back propagated beyond the dense layer of confidence output.

Figure 6: Evolution of confidence over an episode along with game frames for certain states with high confidence values.

Figure 6 shows the confidence values assigned to states over the duration of an episode by a fully trained CARA3C DRL agent for two games where CARA3C achieves large improvements over LRA3C and A3C: Kangaroo and Beam Rider.

For both the games it's clear that the confidence values change dynamically in response to the states encountered.

Here, it's important to note that the apparent periodicity observed in the graphs is not because of the nature of the confidences learnt but is instead due to the periodic nature of the games themselves.

From the game frames shown one can observe that the frames with high confidence are highly recurring key states of the game.

In case of Beam Rider, the high confidence states correspond to the initial frames of every wave wherein a new horde of enemies (often similar to the previous wave) come in after completion of the penultimate wave in the game.

In the case of Kangaroo, apart from many other aspects to the game, there is piece of fruit which keeps falling down periodically along the left end of the screen (can be seen in the game's frames).

Jumping up at the appropriate time and punching the fruit gives you 200 points.

By observing game-play, we found that the policy learnt by the CARA3C agent identifies exactly this facet of the game to achieve such large improvements in the score.

Once again, these series of states encompassing this transition of the fruit towards the bottom of the screen form a set of highly recurring states, Especially states where the piece of fruit is "jumping-distance" away the kangaroo and hence are closer to the reward are found to form the peaks.

These observations reiterate the results obtained in Section 4.3 as the highly recurring nature of these states (during training over multiple episodes) would enable the network to estimate these value functions better.

The better estimates of these value functions are then suitably used to obtain better estimates for other states by bootstrapping with greater attention to the high confidence states.

We believe that this ability to derive from a few key states by means of an attention mechanism provided by the confidence values enables CARA3C to obtain better estimates of the value function.

An expanded version of the graphs for all the tasks can be found in Appendix E. In this paper, we propose two methods for learning value functions in a more sophisticated manner than using n-step returns.

Hence, it is important to analyze the value functions learned by our methods and understand whether our methods are indeed able to learn better value functions than baseline methods or not.

For this sub-section we trained a few A3C agents to serve as baselines.

To verify our claims about better learning of value functions, we conducted the following experiment.

We took trained CARA3C, LRA3C and A3C agents and computed the L 2 loss between the value function V (s t ) predicted by a methods and the actual discounted sum of returns ( T −t k=0 γ k r t+k ).

We averaged this quantity over 10 episodes and plotted it as a function of time steps within an episode.

FIG4 demonstrates that our novel method CARA3C learns a much better estimate of the Value function V (s t ) than LRA3C and A3C.

The only exception to this is the game of Kangaroo.

The reason that A3C and LRA3C critics manage to estimate the value function well in Kangaroo is because the policy is no better than random and in fact their agents often score just around 0 (which is easy to estimate).

We propose a straightforward way to incorporate λ-returns into the A3C algorithm and carry out a large-scale benchmarking of the resulting algorithm LRA3C.

We go on to propose a natural generalization of λ-returns called Confidence-based Autodidactic returns (CAR).

In CAR, the agent learns to assign weights dynamically to the various n-step returns from which it can bootstrap.

Our experiments demonstrate the efficacy of sophisticated mixture of multi-steps returns with at least one of CARA3C or LRA3C out-performing A3C in 18 out of 22 tasks.

In 9 of the tasks CARA3C performs the best whereas in 9 of them LRA3C is the best.

CAR gives the agent the freedom to learn and decide how much it wants to weigh each of its n-step returns.

The concept of Autodidactic Returns is about the generic idea of giving the DRL agent the ability to model confidence in its own predictions.

We demonstrate that this can lead to better Under review as a conference paper at ICLR 2018 TD-targets, in turn leading to improved performances.

We have proposed only one way of modeling the autodidactic weights wherein we use the confidence values that are predicted alongside the value function estimates.

There are multiple other ways in which these n-step return weights can be modeled.

We believe these ways of modeling weighted returns can lead to even better generalization in terms how the agent perceives it's TD-target.

Modeling and bootstrapping off TD-targets is fundamental to RL.

We believe that our proposed idea of CAR can be combined with any DRL algorithm BID8 BID4 BID16 wherein the TD-target is modeled in terms of n-step returns.

Since the baseline scores used in this work are from BID16 , we use the same training and evaluation regime as well.

We used the LSTM-variant of A3C BID9 ] algorithm for the CARA3C and LRA3C experiments.

The async-rmsprop algorithm BID9 ] was used for updating parameters with the same hyper-parameters as in BID9 .

The initial learning rate used was 10 −3 and it was linearly annealed to 0 over 100 million time steps, which was the length of the training period.

The n used in n-step returns was 20.

Entropy regularization was used to encourage exploration, similar to BID9 .

The β for entropy regularization was found to be 0.01 after hyper-parameter tuning, both for CARA3C and LRA3C, separately.

The β was tuned in the set {0.01, 0.02}. The optimal initial learning rate was found to be 10 −3 for both CARA3C and LRA3C separately.

The learning rate was tuned over the set {7 × 10 −4 , 10 −3 , 3 × 10 −3 }.

The discounting factor for rewards was retained at 0.99 since it seems to work well for a large number of methods BID9 BID16 BID4 .

The most important hyper-parameter in the LRA3C method is the λ for the λ-returns.

This was tuned extensively over the set {0.05, 0.15, 0.5, 0.85, 0.9, 0.95, 0.99}. The best four performing models have been reported in Figure 11b .

The best performing models had λ = 0.9.All the models were trained for 100 million time steps.

This is in keeping with the training regime in BID16 to ensure fair comparisons to the baseline scores.

Evaluation was done after every 1 million steps of training and followed the strategy described in BID16 to ensure fair comparison with the baseline scores.

This evaluation was done after each 1 million time steps of training for 100 episodes , with each episode's length capped at 20000 steps, to arrive at an average score.

The evolution of this average game-play performance with training progress has been demonstrated for a few tasks in FIG0 .

An expanded version of the figure for all the tasks can be found in Appendix C. TAB1 in Appendix B contains the raw scores obtained by CARA3C, LRA3C and A3C agents on 22 Atari 2600 tasks.

The evaluation was done using the latest agent obtained after training for 100 million steps, to be consistent with the evaluation regime presented in BID16 and BID9 .

We used a low level architecture similar to BID9 ; BID16 which in turn uses the same low level architecture as BID8 .

Figure 1 contains a visual depiction of the network used for CARA3C.

The common parts of the CARA3C and LRA3C networks are described below.

The first three layers of both the methods are convolutional layers with same filter sizes, strides, padding and number of filters as BID8 ; BID16 .

These convolutional layers are followed by two fully connected (FC) layers and an LSTM layer.

A policy and a value function are derived from the LSTM outputs using two different output heads.

The number of neurons in each of the FC layers and the LSTM layers is 256.

These design choices have been taken from BID16 to ensure fair comparisons to the baseline A3C model and apply to both the CARA3C and LRA3C methods.

Similar to BID9 the Actor and Critic share all but the final layer.

In the case of CARA3C, Each of the three functions: policy, value function and the confidence value are realized with a different final output layer, with the confidence and value function outputs having no non-linearity and one output-neuron and with the policy and having a softmax-non linearity of size equal to size of the action space of the task.

This non-linearity is used to model the multinomial distribution.

All the evaluations were done using the agent obtained after training for 100 million steps, to be consistent with the evaluation paradigm presented in BID16 and BID9 .

Both CARA3C and LRA3C scores are obtained by averaging across 3 random seeds.

The scores for A3C column were taken from Table 4 of BID16 .

The evaluation strategy described in Appendix A was executed to generate training curves for all the 22 Atari tasks.

This appendix contains all those training curves.

These curves demonstrate how the performance of the CARA3C and LRA3C agents evolves with time.

This appendix presents the expanded version of the results shown in Section 4.2.

These plots show the vast differences in weights assigned by CARA3C and LRA3C to various n-step returns over the duration of a single episode.

This appendix presents the expanded version of the results shown in Section 4.4.

The aim is to show the how the confidence values dynamically change over the duration of an episode and show the presence of explicit peaks and troughs in many games.

Here, it's important to note that the apparent periodicity observed in some graphs is not because of the nature of the confidences learnt but is instead due to the periodic nature of the games themselves.

Figure 10: Evolution of confidence values over an episode.

@highlight

A novel way to generalize lambda-returns by allowing the RL agent to decide how much it wants to weigh each of the n-step returns.

@highlight

Extends the A3C algorithm with lambda returns, and proposes an approach for learning the weights of the returns.

@highlight

The authors present confidence-based autodidactic returns, a Deep learning RL method to adjust the weights of an eligibility vector in TD(lambda)-like value estimation to favour more stable estimates of the state.
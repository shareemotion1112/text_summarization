One of the key challenges of session-based recommender systems is to enhance users’ purchase intentions.

In this paper, we formulate the sequential interactions between user sessions and a recommender agent as a Markov Decision Process (MDP).

In practice, the purchase reward is delayed and sparse, and may be buried by clicks, making it an impoverished signal for policy learning.

Inspired by the prediction error minimization (PEM) and embodied cognition, we propose a simple architecture to augment reward, namely Imagination Reconstruction Network (IRN).

Speciﬁcally, IRN enables the agent to explore its environment and learn predictive representations via three key components.

The imagination core generates predicted trajectories, i.e., imagined items that users may purchase.

The trajectory manager controls the granularity of imagined trajectories using the planning strategies, which balances the long-term rewards and short-term rewards.

To optimize the action policy, the imagination-augmented executor minimizes the intrinsic imagination error of simulated trajectories by self-supervised reconstruction, while maximizing the extrinsic reward using model-free algorithms.

Empirically, IRN promotes quicker adaptation to user interest, and shows improved robustness to the cold-start scenario and ultimately higher purchase performance compared to several baselines.

Somewhat surprisingly, IRN using only the purchase reward achieves excellent next-click prediction performance, demonstrating that the agent can "guess what you like" via internal planning.

A good recommender system can enhance both satisfaction for users and profit for content providers BID7 .

In many real-world scenarios, the recommender systems make recommendations based only on the current browsing session, given the absence of user profiles (because the user is new or not tracked or not logged in, till the final purchase step).

A session is a group of sequential interactions between a user and the system within a short period of time.

To model this phenomenon, Recurrent Neural Networks (RNNs) were recently employed as session-based recommenders BID9 BID12 .

For instance, GRU4Rec BID9 utilizes the session-parallel mini-batch training to handle the variable lengths of sessions, and predicts the next action given the sequence of items in the current session.

However, these approaches primarily focus on next-click prediction and model the session data via sequential classification, and thus cannot distinguish the different effects of user clicks and purchases.

In this paper, we consider the session-based recommendation as a Markov Decision Process (MDP), which can take into account both the click reward and the purchase reward (see FIG0 , and leverage Reinforcement Learning (RL) to learn the recommendation strategy.

In practice, several challenges need to be addressed.

First, the recommender systems involve large numbers of discrete actions (i.e., items), making current RL algorithms difficult to apply .

This requires the agent to explore its environment for action feature learning and develop an ability to generalize over unseen actions.

Second, we found it difficult to specify the click reward and the purchase reward; the policy may be biased by long sessions that contain many user clicks, as RL algorithms maximize the accumulated reward.

Besides, real-world recommender systems require quick adaptation to user interest and robustness to the cold-start scenario (i.e., enhancing the purchase performance of short sessions).

Therefore, we will be particularly interested in a case where only the purchase is used as reward (click sequences are used as inputs of the imagination core for Under review as a conference paper at ICLR 2019 exploration).

1 However, the purchase reward is delayed and sparse (one session may contain only one purchase), making it a difficult signal for policy learning.

To augment reward and encourage exploration, we present the Imagination Reconstruction Network (IRN), which is inspired by the prediction error minimization (PEM) BID10 BID5 BID14 and embodied cognition BID2 BID1 BID23 BID3 from the neuroscience literature.

The PEM is an increasingly influential theory that stresses the importance of brain-body-world interactions in cognitive processes, involving perception, action and learning.

In particular, IRN can be regarded as a proof-of-concept for the PEM from the recommendation perspective, following the ideas in BID1 and BID23 -the brain utilizes active sensorimotor predictions (or counterfactual predictions) to represent states of affairs in the world in an action-oriented manner.

Specifically, the imagination core of IRN that predicts the future trajectories (i.e., a set of imagined items that user may purchase) conditioned on actions sampled from the imagination policy, can be considered as the generative model of the brain that simulates sensorimotor predictions.

To update the action policy, the imagination-augmented executor minimizes the intrinsic imagination error of predicted trajectories by self-supervised reconstruction, while maximizing the extrinsic reward using RL, with shared input state or output action representations for predictive learning.

This simulates the active perception (a key aspect of embodied cognition) of the body under the PEM framework, which adapts the agent to possible changes that arise from the ongoing exploratory action.

Note that the imagination policy imitates the action policy through distillation or a delayed target network, and thus IRN constructs a loop between brain and body, encouraging the agent to perform actions that can reduce the error in the agent's ability to predict the future events BID20 .

IRN equips the agent with a planning module, trajectory manager, that controls the granularity of imagined trajectories using the planning strategies (e.g., breadth-n and depth-m).

Besides, IRN is a combination of model-based planning and self-supervised RL, as the imagined trajectories provide dense training signals for auxiliary task learning (see section 2).The key contributions of this paper are summarized as follows:• We formulate the session-based recommendation as a MDP, and leverage deep RL to learn the optimal recommendation policy, and also discuss several challenges when RL is applied.• We consider a special case where only the purchase is used as reward, and then propose the IRN architecture to optimize the sparser but more business-critical purchase signals, which draws inspiration from the theories of cognition science.• We present a self-supervised reconstruction method for predictive learning, which minimizes the imagination error of simulated trajectories over time.

IRN achieves excellent click and purchase performance even without any external reward (predictive perception BID23 ).•

We conduct a comprehensive set of experiments to demonstrate the effectiveness of IRN.

Compared to several baselines, IRN improves data efficiency, promotes quicker adaptation to user interest, and shows improved robustness to the cold-start scenario and ultimately higher purchase performance.

These are highly valuable properties in an industrial context.

Session-based Recommenders Classical latent factor models (e.g., matrix factorization) break down in the session-based setting, given the absence of user profiles.

A natural solution is the neighborhood approach like item-to-item recommendation BID15 .

In this setting, an item similarity matrix can be precomputed based on co-occurrences of clicked items in sessions.

However, this method only considers the last clicked item of the browsing session for recommendations, ignoring the sequential information of the previous events.

Previous works also attempt to apply MDPs in the recommendation systems BID24 BID30 .

The main issue is that the state space quickly becomes unmanageable due to the large number of items (IRN employs deep learning to overcome this problem and thus generalizes well to unseen states).

Recently, RNNs have been used with success in this area BID9 BID8 BID12 .

GRU4Rec BID9 is the first application of RNNs to model the session data, which can provide recommendations after each click for new sessions.

Under review as a conference paper at ICLR 2019 where the purchase is used as the reward signal for recommender agent and user 5 sessions are treated as external environment.

To handle the sparse purchase reward, 6 we propose a simple architecture, namely Imagination Reconstruction Network 7 (IRN), which is inspired from the neu-science Prediction Error Minimization.

IRN distill a roll-out policy from action policy and combine it with a static 9 environment model to predict the future observations.

Imaginations are then 10 used as self-supervised reconstruction signal for action policy when there is no Abstract mmendations, it is important to enhance the purchase willing of ular session-based models use classification methods to model ns classification, which do not explicit differentiate the click and In this paper, we formulate sequential recommendation as MDP, e is used as the reward signal for recommender agent and user d as external environment.

To handle the sparse purchase reward, ple architecture, namely Imagination Reconstruction Network nspired from the neu-science Prediction Error Minimization.

-out policy from action policy and combine it with a static el to predict the future observations.

Imaginations are then vised reconstruction signal for action policy when there is no y minimizing the intrinsic imagination error and maximizing the probability, IRN can automatically balance EE and learn a better ntations, resulting in a more robust action policy.

Experiments significantly enhance the purchase, and achieve a satisfactory even when we do not treat click as rewards.

start scenario (short sessions).

fficiency.

en item space is huge.ation (end with buy).

using GRU4REC classification problem do not urchase, in practice, we may pay more attention to the improvements of -art methods that consider users' sequential behavior for recommendation, ers with recurrent neural networks (RNN) or Markov chains, our method onsistently better performance on four real-world datasets.tions, users' current interests are influenced by their historical behaviors.

n, previous sequential recommenders with user historical records.

For However, GRU4Rec utilizes the session-parallel mini-batch training to handle the variable lengths of sessions; this trick cannot effectively capture sequentiality of sessions, since the network is trained using the BP algorithm (not BPTT for RNNs).

These models primarily focus on next-click prediction and model the click-streams via sequential classification, while here we aim at modeling the purchase behavior and enhancing users' purchase intentions.

Besides, IRN is built on RL, which encodes sequentiality of states into the value function.

Imagination-augmented Agents All approaches incorporating off-policy experience (e.g., imagined trajectories) generated by a learned model can be categorized into model-based reinforcement learning BID19 BID26 BID28 .

By using an internal model of the world, the agent can generalize to unseen states, remain valid in the real environment, and exploit additional training signals to improve data efficiency.

However, the performance of model-based agents usually suffers from model errors resulting from function approximation.

I2As were proposed to address this issue.

I2As augment model-free agents with imagination and use an interpretation module to handle imperfect predictions.

The imagined trajectories of I2As are provided as additional context (i.e., input features) to a policy network, while the proposed IRN uses the trajectories as additional training signals for self-supervised reconstruction.

Self-supervised Reinforcement Learning In many real-world scenarios, reward is extremely sparse and delayed, and the agent updates its policy only if it reaches a pre-defined goal state.

To model this phenomenon, self-supervised reinforcement learning have often been used, which accelerates the acquisition of a useful representation with auxiliary task learning BID11 BID20 BID15 BID25 .

Specifically, auxiliary tasks provide additional losses for feature learning, and can be trained instantaneously using the self-supervision from the environment.

For instance, UNREAL BID11 ) maximizes many other pseudo-reward functions simultaneously, e.g., pixel change control, with a common representation shared by all tasks.

In contrast, the proposed IRN do not require the external supervision from the environment, i.e., self-supervised reconstruction is performed on internal imagined trajectories.

Reward R:

After the RA takes an action a t at the state s t , i.e., recommending an item to a user, the user browses this item and provides her feedback (click or purchase).

The agent receives a scalar reward r(s t , a t ) according to the user's feedback.

We also define the k-step return starting from state DISPLAYFORM0 Transition probability P : Transition probability p(s t+1 |s t , a t ) defines the probability of state transition from s t to s t+1 when the RA takes action a t .

In this case, the state transition is deterministic after taking the ground-true action a t = i t , i.e., p(s t+1 |s t , i t ) = 1 and s t+1 = s t ∪ {i t }.The goal of the RA is to find an optimal policy π * , such that V π * (s 1 ) ≥ V π (s 1 ) for all policies π and start state s 1 , where V π (s t ) is the expected return for a state s t when following a policy π, i.e., DISPLAYFORM1 Asynchronous Advantage Actor-Critic.

This paper builds upon the A3C algorithm, an actorcritic approach that constructs a policy network π(a|s; θ) and a value function network V (s; θ v ), with all non-output layers shared .

The policy and the value function are adjusted towards the bootstrapped k-step return 1993 ) is computed as the difference of the bootstrapped k-step return and the current state value estimate: DISPLAYFORM2 DISPLAYFORM3 where θ − v are the parameters of the previous target network.

To increase the probability of rewarding actions, A3C applies an update g(θ) to the parameters θ using an unbiased estimation BID29 : DISPLAYFORM4 The value function V (s; θ v ) is updated following the recursive definition of the Bellman Equation, DISPLAYFORM5 is obtained by minimizing a squared error between the target return and the current value estimate: DISPLAYFORM6 In A3C multiple agents interact in parallel, with multiple instances of the environment.

The asynchronous execution accelerates and stabilizes learning.

In practice, we combine A3C with the session-parallel mini-batches proposed in BID9 .

Each instance of the agent interacts with multiple sessions simultaneously, gathering M samples from different sessions at a time step.

After k steps, the agent updates its policy and value network according to Eq. (2)(3), using k * M samples.

This decorrelates updates between samples of one session in the instance level.

Besides, to build the A3C agent, we employ an LSTM that jointly approximates both policy π and value function V , given the one-hot vectors of previous items clicked/purchased as inputs.

In this section we incorporate the imagination reconstruction module into the model-free agents (e.g., A3C) in order to enhance data efficiency, promote more robust learning and ultimately higher performance under the sparse extrinsic reward.

Our IRN implements an imagination-augmented policy via three key components (Figure 2 ).

The imagination core (IC) predicts the next time steps conditioned on actions sampled from the imagination policyπ.

At a time step t, the trajectory manager (TM) determines how to roll out the IC under the planning strategy, and then produces imagined trajectoriesT 1 , . . .

,T n of an observable world state s t .

Each trajectoryT j is a sequence of items {î j,t ,î j,t+1 , . . .}, that users may purchase (or click) from the current time t. The Imaginationaugmented Executor (IAE) aggregates the internal data resulting from imagination and external rewarding data to update its action policy π.

Specifically, the IAE optimizes the policy π by maximizing the extrinsic reward while minimizing the intrinsic imagination error.

In principle, IRN encourages exploration and learns predictive representations via imagination rollouts, which promotes quick adaptation to user interest and robustness to the cold-start scenario.

Figure 2: IRN architecture.

s shown in Figure 2 , where IC is imagination core to generate trajectories, and N-step imagination rollout, using these trajectories to perform encoder-decoder.

In this section we incorporate the imagination reconstruction module into the model-free agents 117 (e.g., A3C) in order to enhance data efficiency, promote more robust learning and ultimately higher 118 performance under the sparse extrinsic rewards.

Our IRN implements an imagination-augmented 119 policy via three key components (Fig 2) .

cost] .

Besides, the predictors may learn a trivial identical function, the state transition in agent trajectories is deterministic, i.e., s t+1 = s t [ {i t } and a t = i t .

In ork, we derive a static environment model from the state transition:ŝ t+⌧ +1 =ŝ t+⌧ [ {î t+⌧ }, =â t+⌧ andŝ t = s t , where ⌧ is the length of the imagined rollout,â t+⌧ the output action of the ination policy⇡.

During training, the generated itemî t+⌧ may not be the true purchase, but we se it for self-supervised reconstruction of I2E.

This makes the action policy ⇡ more robust to sic errors and forces the imagination policy⇡ to generate more accurate actions.

ctice, the imagination policy⇡ can be obtained from policy distillation [] or fixed target network QN [].

The former distills the action policy ⇡(s t ; ✓) into a smaller rollout network⇡(s t ;✓), a cross-entropy loss, l ⇡,⇡ (s t ) = P a ⇡(a|s t )log⇡(a|s t ;✓).

The latter uses a shared but slowly ging target network⇡(s t ; ✓ ), where ✓ are previous parameters in ⇡(s t ; ✓).

By imitating the policy ⇡, the imagined trajectories will be similar to agent experiences in the real environment; lso helps I2E learn a predictive representation of rewarding states, and in turn should allow the learning of the action policy under the sparse reward signals.adaptation to user interest.

ination Core.

In order to simulate imagined trajectories, we rely on environment models that, the present state and a candidate action, make predictions about the future states.

2 In general, an employ an environment model that build on recent popular action-conditional next-step ctors [], and train it in an unsupervised fashion from agent experiences.

However, the predictors ly suffer from model errors, resulting in poor agent performance, and need extra computational e.g., pre-training) [error][

, since it is not useful as reported in [].

and in RS, rewards of different actions rd to specify 4 extrinsic rewards while minimizing the intrinsic imagination errors, L = iple, IRN encourages exploration (on the output layer, i.e., item representation) plan ("guess what you like") via imagination rollouts, and therefore promotes r interest.

n order to simulate imagined trajectories, we rely on environment models that, and a candidate action, make predictions about the future states.2 In general, vironment model that build on recent popular action-conditional next-step it in an unsupervised fashion from agent experiences.

However, the predictors del errors, resulting in poor agent performance, and need extra computational ) [error] [cost].

Besides, the predictors may learn a trivial identical function, n in agent trajectories is deterministic, i.e., s t+1 = s t [ {i t } and a t = i t .

In static environment model from the state transition:ŝ t+⌧ +1 =ŝ t+⌧ [ {î t+⌧ }, s t , where ⌧ is the length of the imagined rollout,â t+⌧ the output action of the During training, the generated itemî t+⌧ may not be the true purchase, but we rvised reconstruction of I2E.

This makes the action policy ⇡ more robust to ces the imagination policy⇡ to generate more accurate actions.

tion policy⇡ can be obtained from policy distillation [] or fixed target network er distills the action policy ⇡(s t ; ✓) into a smaller rollout network⇡(s t ;✓), oss, l ⇡,⇡ (s t ) = P a ⇡(a|s t )log⇡(a|s t ;✓).

The latter uses a shared but slowly rk⇡(s t ; ✓ ), where ✓ are previous parameters in ⇡(s t ; ✓).

By imitating the agined trajectories will be similar to agent experiences in the real environment; n a predictive representation of rewarding states, and in turn should allow the tion policy under the sparse reward signals.

we propose a simple architecture, namely Imagination Reconstruction Network 7 (IRN), which is inspired from the neu-science Prediction Error Minimization.conditioned on actions sampled from the imagination policy⇡.

At a time step t, the trajectory 121 manager (TM) determines how to roll out the IC under the planning strategy, and then produces 122 imagined trajectoriesT 1 , . . . ,T n of an observable world state s t .

Each trajectoryT is a sequence 123 of items {î t ,î t+1 , . . .}, that the agent may purchase (or click) from the current time t. Finally, 124 the Imagination-augmented Executor (I2E) aggregates the internal data resulting from imagination 125 and external rewarding data to update its action policy ⇡.

Specifically, I2E optimizes the

IRN distill a roll-out policy from action policy and combine it with a static 9 environment model to predict the future observations.

Imaginations are then 10 used as self-supervised reconstruction signal for action policy when there is no 11 purchase reward.

By minimizing the intrinsic imagination error and maximizing the 12 extrinsic purchase probability, IRN can automatically balance EE and learn a better 13 predictive representations, resulting in a more robust action policy.

Experiments 14 show that IRN can significantly enhance the purchase, and achieve a satisfactory 15 click performance even when we do not treat click as rewards.

robust to the cold-start scenario (short sessions).

improve the data efficiency.

Compared with state-of-the-art methods that consider users' sequential behavior for recommenda 29 e.g., sequential recommenders with recurrent neural networks (RNN) or Markov chains, our met 30 achieves significantly and consistently better performance on four real-world datasets.

In many real-world applications, users' current interests are influenced by their historical behav

To model this phenomenon, previous sequential recommenders with user historical records.

Submitted to 32nd Conference on Neural Information Processing Systems (NIPS 2018).

Do not distribute.

purchase reward.

By minimizing the intrinsic imagination error and maximizing the 12 extrinsic purchase probability, IRN can automatically balance EE and learn a better 13 predictive representations, resulting in a more robust action policy.

Experiments 14 show that IRN can significantly enhance the purchase, and achieve a satisfactory 15 click performance even when we do not treat click as rewards.

robust to the cold-start scenario (short sessions).

improve the data efficiency.

used as self-supervised reconstruction signal for action policy when there is no 11 purchase reward.

By minimizing the intrinsic imagination error and maximizing the 12 extrinsic purchase probability, IRN can automatically balance EE and learn a better 13 predictive representations, resulting in a more robust action policy.

Experiments 14 show that IRN can significantly enhance the purchase, and achieve a satisfactory 15 click performance even when we do not treat click as rewards.

IRN distill a roll-out policy from action policy and combine it with a static 9 environment model to predict the future observations.

Imaginations are then 10 used as self-supervised reconstruction signal for action policy when there is no 11 purchase reward.

By minimizing the intrinsic imagination error and maximizing the 12 extrinsic purchase probability, IRN can automatically balance EE and learn a better 13 predictive representations, resulting in a more robust action policy.

Experiments 14 show that IRN can significantly enhance the purchase, and achieve a satisfactory 15 click performance even when we do not treat click as rewards.

Compared with state-of-the-art methods that consider users' sequential behavior for recommendation, 26 e.g., sequential recommenders with recurrent neural networks (RNN) or Markov chains, our method 27 achieves significantly and consistently better performance on four real-world datasets.

In many real-world applications, users' current interests are influenced by their historical behaviors.

oned on actions sampled from the imagination policy⇡.

At a time step t, the trajectory er (TM) determines how to roll out the IC under the planning strategy, and then produces ed trajectoriesT 1 , . . .

,T n of an observable world state s t .

Each trajectoryT is a sequence s {î t ,î t+1 , . . .}, that the agent may purchase (or click) from the current time t. Finally, gination-augmented Executor (I2E) aggregates the internal data resulting from imagination ternal rewarding data to update its action policy ⇡.

Specifically, I2E optimizes the policy aximizing the extrinsic rewards while minimizing the intrinsic imagination errors, L = L IRN .

In principle, IRN encourages exploration (on the output layer, i.e., item representation) licitly learns to plan ("guess what you like") via imagination rollouts, and therefore promotes daptation to user interest.

ation Core.

In order to simulate imagined trajectories, we rely on environment models that, he present state and a candidate action, make predictions about the future states.

In general, employ an environment model that build on recent popular action-conditional next-step ors [], and train it in an unsupervised fashion from agent experiences.

However, the predictors suffer from model errors, resulting in poor agent performance, and need extra computational g., pre-training) [error] [cost].

Besides, the predictors may learn a trivial identical function, e state transition in agent trajectories is deterministic, i.e., s t+1 = s t [ {i t } and a t = i t .

In rk, we derive a static environment model from the state transition:ŝ t+⌧ +1 =ŝ t+⌧ [ {î t+⌧ }, â t+⌧ andŝ t = s t , where ⌧ is the length of the imagined rollout,â t+⌧ the output action of the ation policy⇡.

During training, the generated itemî t+⌧ may not be the true purchase, but we it for self-supervised reconstruction of I2E.

This makes the action policy ⇡ more robust to c errors and forces the imagination policy⇡ to generate more accurate actions.

tice, the imagination policy⇡ can be obtained from policy distillation [] or fixed target network N [].

The former distills the action policy ⇡(s t ; ✓) into a smaller rollout network⇡(s t ;✓), cross-entropy loss, l ⇡,⇡ (s t ) = P a ⇡(a|s t )log⇡(a|s t ;✓).

The latter uses a shared but slowly g target network⇡(s t ; ✓ ), where ✓ are previous parameters in ⇡(s t ; ✓).

By imitating the olicy ⇡, the imagined trajectories will be similar to agent experiences in the real environment; o helps I2E learn a predictive representation of rewarding states, and in turn should allow the arning of the action policy under the sparse reward signals.do not predict the rewards, since it is not useful as reported in [].

and in RS, rewards of different actions to specify 4 n observable world state s t .

Each trajectory T is a sequence t may purchase (or click) from the current time t. Finally, (I2E) aggregates the internal data resulting from imagination e its action policy ⇡.

Specifically, I2E optimizes the policy rds while minimizing the intrinsic imagination errors, L = rages exploration (on the output layer, i.e., item representation) hat you like") via imagination rollouts, and therefore promotes late imagined trajectories, we rely on environment models that, e action, make predictions about the future states.

In general, el that build on recent popular action-conditional next-step vised fashion from agent experiences.

However, the predictors lting in poor agent performance, and need extra computational Besides, the predictors may learn a trivial identical function, ctories is deterministic, i.e., s t+1 = s t [ {i t } and a t = i t .

In ent model from the state transition:ŝ t+⌧ +1 =ŝ t+⌧ [ {î t+⌧ }, he length of the imagined rollout,â t+⌧ the output action of the , the generated itemî t+⌧ may not be the true purchase, but we uction of I2E.

This makes the action policy ⇡ more robust to tion policy⇡ to generate more accurate actions.

n be obtained from policy distillation [] or fixed target network action policy ⇡(s t ; ✓) into a smaller rollout network⇡(s t ;✓), P a ⇡(a|s t )log⇡(a|s t ;✓).

The latter uses a shared but slowly here ✓ are previous parameters in ⇡(s t ; ✓).

By imitating the ies will be similar to agent experiences in the real environment; epresentation of rewarding states, and in turn should allow the er the sparse reward signals.

Figure 2: IRN architecture: a) the imagination core (IC) predicts the next time step and then generates the imagined trajectoriesT ; b) the trajectory manager (TM) employs various planning strategies (e.g., depth-m here) to control the granularity ofT ; c) the imagination-augmented executor (IAE) optimizes the network using the internal imagination data and external rewarding data (e.g., purchases).Imagination Core In order to simulate imagined trajectories, we rely on environment models that, given the present state and a candidate action, make predictions about the future states.

In general, we can employ an environment model that build on action-conditional next-step predictors BID18 , and train it in an unsupervised fashion from agent experiences.

However, the predictors usually suffer from model errors, resulting in poor agent performance, and require extra computational cost (e.g., pre-training).

Besides, the predictors may learn a trivial identical function, since the state transition in agent trajectories (or session data) is deterministic, i.e., s t+1 = s t ∪ {i t } and i t = a t .

In this work, we derive a static environment model from the state transition:ŝ t+τ +1 =ŝ t+τ ∪ {î t+τ }, i t+τ =â t+τ andŝ t = s t , where τ is the length of the imagined rollout,â t+τ the output action of the imagination policyπ.

During training, the generated itemî t+τ may not be the true purchase/click, but we still use it for self-supervised reconstruction.

This makes the action policy π more robust to intrinsic errors and forces the imagination policyπ to generate more accurate actions.

In practice, the imagination policyπ can be obtained from policy distillation or a fixed target network like DQN BID16 .

The former distills the action policy π(s t ; θ) into a smaller rollout networkπ(s t ;θ), using a cross-entropy loss, l π,π (s t ) = a π(a|s t )logπ(a|s t ;θ).

The latter uses a shared but slowly changing networkπ(s t ; θ − ), where θ − are previous parameters in π(s t ; θ).

By imitating the action policy π, the imagined trajectories will be similar to agent experiences in the real environment; this also helps IAE learn predictive representations of rewarding states, and in turn should allow the easy learning of the action policy under the sparse reward signals.

Trajectory Manager The TM rolls out the IC over multiple time steps into the future, generating multiple imagined trajectories with the present information.

Additionally, various planning strategies are supported for trajectory simulation: breadth-n, depth-m and their combination.

For breadth-n imagination, the TM generates n trajectories,T 1 , . . .

,T n , over one time step from the current state s t , i.e.,T j = {î j,t }.

Empirically, the IAE using breadth-n imagination will motivate the agent to focus on short-term events and predict the next step more accurately (e.g., enhancing the next-click prediction performance even when we do not formalize the click event as reward).

For depth-m imagination, the TM generates only one trajectoryT 1 through m time steps, i.e.,T 1 = {î 1,t , . . .

,î 1,t+m−1 }.

This enables the agent to learn to plan the long-term future, and thus recommend items that yield high rewards (purchases).

Finally, we can also achieve the trade-off between breadth-n and depth-m to balance the long-term rewards and short-term rewards.

Specifically, we generate n trajectories, and each has a depth m, i.e., {T } = {{î 1,t , . . .

,î 1,t+m−1 }, . . .

, {î n,t , . . .

,î n,t+m−1 }}.

As mentioned before, the IAE uses external rewarding data and internal imagined trajectories to update its action policy π.

For the j-th trajectory,T j = {î j,t , . . . ,î j,t+m−1 }, we define a multi-step reconstruction objective using the mean squared error:Under review as a conference paper at ICLR 2019 DISPLAYFORM0 whereT j,τ is the τ -th imagined item, φ(·) is the input encoder shared by π (for joint feature learning), AE is the autoencoder that reconstructs the input feature, and the discounting factor γ is used to mimic Bellman type operations.

In practice, we found that action representation learning (i.e., the output weights of π) is crucial to the final performance due to the large size of candidate items.

Therefore, we use the one-hot transformation as φ(·) and replace AE with the policy π (excluding the final softmax function), and only back-propagate errors in the positions of imagined items.

Specifically, for an imagined item, the mean squared error is computed between one and its activation value through π; errors for other items are turned to be zero.

In this case, the policy π is optimized not only to predict purchases accurately but also to minimize the reconstruction error of imagined items over time.

Take a session for example, {i 0 , i 1 , ..., i q−1 , i q } (i q is the final purchased item), π is trained t + 1 times using imagination reconstruction and once using A3C updating (for the purchase event); the overall reconstruction loss for this session is defined as L IRN = q t=0 n j=1 L j (s t ).

There are several advantages associated with the imagination reconstruction.

First, imagined trajectories provide auxiliary signals for reward augmentation.

This speeds up policy learning when extrinsic reward is delayed and sparse.

Second, by using a shared policy network, IAE enables exploration and exploitation, and thus improves feature learning when the number of actions is large.

Third, compared with agents that predict the next observations for robust learning BID15 , our IAE reconstructs the imagined trajectories generated by the TM over time for predictive learning.

When external reward is provided, IAE can be considered as a process of goal-oriented learning or semi-supervised learning.

This self-supervised reconstruction approach also achieves excellent click and purchase prediction performance even without any external reward (unsupervised learning in this case, where inputs and output targets used for training π are all counterfactual predictions, and the input states are transformed through actions in order to match predictions, i.e., predictive perception in Seth (2014)).

We evaluate the proposed model on the dataset of ACM RecSys 2015 Challenge 2 , which contains click-streams that sometimes end with purchase events.

The purchase reward and the click reward (if used) are empirically set as 5 and 1, respectively.

Focusing on the most recent events has shown to be effective BID12 ; therefore we collect the latest one month of data and keep sessions that contain purchases.

We follow the preprocessing steps in BID9 and use the sessions of the last three day for testing (we also trained IRN and baselines on the full six month training set, with slightly poorer results; the relative improvements remained similar).

The training set contains 72274 sessions of 683530 events, and the test set contains 7223 sessions of 63100 events, and the number of items is 9167.

We also derive a separate validation set from the training set, with sessions of the last day in the training set.

The evaluation is done by incrementally adding the previous observed event to the session and checking the rank of the next event.

We adopt Recall and Mean Reciprocal Rank (MRR) for top-K evaluations, and take the averaged scores over all events in the test set.

We repeat this procedure 5 times and report the average performance.

Without special mention, we set K to 5 for both metrics.

Besides, we build an environment using session-parallel mini-batches, where the agent interacts with multiple sessions simultaneously (see section 3).Baselines We choose various baseline agents for comparison, including: (1) BPR BID22 ), a pairwise ranking approach, widely applied as a benchmark; (2) GRU4Rec BID9 , a RNN-based approach for session-based recommendations with a BPR-max loss function (note that original GRU4Rec gives much lower purchase performance, thus we only use the clicked items from the same mini-batch as negative examples); (3) CKNN BID12 , a session-based KNN method, which incorporates heuristics to sample similar past sessions as neighbors; (4) A3C-F and A3C-P, the base agents without imagination, using the click and purchase reward (-F) or only the purchase reward (-P); (5) IRN-F and IRN-P, the proposed models that augment A3C with imagiantion; (6) PRN-P, an A3C agent that reconstructs the previous observed trajectories (i.e., click/purchase sequences), using the purchase reward.

Architecture We implemented IRN via Tensorflow 3 , which will be released publicly upon acceptance.

We use grid search to tune hyperparameters of IRN and compared baselines on the validation set.

Specifically, the input state s t is passed through a LSTM with 256 units which takes in the one-hot representation of recent clicked/purchased items.

The output of the LSTM layer is fed into two separate fully connected layers with linear projections, to predict the value function and the action.

A softmax layer is added on top of the action output to generate the probability of 9167 actions.

The discounting value γ is 0.99.

The imagination policyπ is obtained from π using the fixed target network, and the weights ofπ are updated after every 500 iterations.

Without special mentioned, TM employs the combination of breadth-2 and width-2 for internal planning.

The imagination reconstruction is performed every one environment step.

The A3C updating is performed with immediate purchase reward (when found) or 3-step returns (when click reward is used).

Besides, weights of IRN are initialized using Xavier-initializer BID6 and trained via Adam optimizer BID13 with the learning rate and the batch size set to 0.001 and 128, respectively.

We first evaluate the top-K recommendation performance.

The experimental results are summarized in TAB8 .

From the purchase performance comparison, we get:• A3C-P has already outperformed classical session-based recommenders (BPR, CKNN and GRU4Rec) on Recall metrics and achieved comparable results on MRR metrics.

GRU4Rec gives poor purchase performance, as it focuses on next-click prediction.• Comparing IRN-P with A3C-P, we can see that the purchase (and click) performance can be significantly improved with imagination reconstruction, demonstrating that IRN-P can guess what you like via internal planning and learn predictive representations.• IRN-P consistently outperforms IRN-F, and A3C-P also outperforms A3C-F for purchase prediction.

This demonstrates that purchase events can better characterize user interest, and the agents may be biased if clicks are used as reward.

• Comparing PRN-P with A3C-P and IRN-P, we found that reconstructing the previous actual trajectories (i.e., click-streams) also improves the purchase performance (compared to A3C-P).

This is because that PRN-P can learn better representations for clicked items, and user purchases are sometimes contained in the click-streams.

Besides, IRN-P outperforms PRN-P, since PRN-P introduces stronger supervision and may not know what is the final goal, while the imagination reconstruction (without any real trajectories) performs semi-supervised learning, which promotes more robust policy learning.

DISPLAYFORM0 From the click performance comparison, we get:• GRU4Rec achieves excellent next-click performance (e.g., top-5 and top-10) compared to BPR and CKNN, as it models the session data via sequential classification.• A3C-F performs much better than A3C-P and GRU4Rec.

This indicates that RL-based recommenders trained on clicks can generate actions that better preserve the sequential property, possibly due to the accumulated click reward (of longer sessions).• Somewhat interesting, IRN-P significantly outperforms A3C-P, and gets comparable results like IRN-F and A3C-F. This demonstrates that the IRN-P agent may learn to plan and reconstruct the previous clicked trajectories even when only the purchase reward is provided.

Varying the degree of purchase reward sparsity We now explore the robustness of four RLbased recommenders to different purchase reward density.

We randomly sample a d proportion of purchase events from the training set.

The click events remain unchanged.

As shown in TAB9 , A3C-F and IRN-F are robust to different purchase sparsity, since purchases are sometimes contained in the click sequences.

IRN using only the click reward for policy learning can also enhance the purchase prediction performance (see d = 0).

While the performance of A3C-P degrades with sparser purchase reward, the proposed IRN-P achieves comparable performance; the imagination reconstruction promotes predictive learning of rewarding states.

To our surprise, we have found that IRN-P performs well even without any external reward from the environment (i.e., predictive perception, see A3C-F and IRN-P with d = 0).

Minimizing the imagination error of predictive trajectories over time enables the agent to learn sequential patterns in an unsupervised fashion.

FIG7 compares the performance of IRN-P on different reward sparsity setting, where one epoch contains nearly 5000 iterations.

We can observe that the performance of all models is gradually improved, and IRN-P with a larger d learns faster, indicating better exploration and exploitation.

Note that IRN-P with d = 0 will adversely decrease the performance due to the local over-training.

In extreme cases, a final purchase decision would be unknown, the imagination reconstruction may be applied without external reward, but we can use the click prediction performance for validation and early stopping.

We then analyze the effectiveness of different planners of the TM.

TAB11 shows the best results obtained with IRN-P when using alternative measurements.

Note that the purchase event in one session is usually the last user interaction, and "First" means that the second event is evaluated separately (the first clicked item is used as the initial state).

We can observe that, different planners equip the agent with different prediction capacity.

For instance, IRN-P with a larger n performs better on First and Click metrics, indicating that the agent with breadth-n planning focuses more on short-term rewards.

On the contrary, a larger m can improve the purchase performance at a cost of lower First and Click results, since depth-m planning enables the agent to imagine the longer future.

The combination of breadth-n and depth-m can better balance the long-term rewards and short-term rewards.

Besides, for IRN-P without any external reward (d = 0.0), the depth-2 planner gives better performance than depth-1 and breadth-2 on three measurements (by 2-5%), possibly due to the more predictive representations learned after unsupervised training.

However, for IRN with purchase reward (semi-supervised learning), the purchase performance cannot be improved using longer imagined trajectories.

One possible reason is that two steps of imagination reconstruction is sufficient for learning to predict the future events recursively; the first step of IRN learns to capture the difference of adjacent input states, and the second step learns to look ahead the future purchase signal accurately.

Robustness to the cold-start scenario We simulate a cold-start scenario using the test set.

Specifically, we use a parameter c to control the number of items in the input state (a set of one-hot vectors of clicked items), i.e., new events will not be added to the input state if the number of items exceeds c, but are still used for evaluations.

FIG8 (a,b) shows the purchase performance w.r.t.

the cold-start parameter c. We can see that IRN-P outperforms A3C-P and A3C-F over all ranges of c, verifying the effectiveness of imagination reconstruction.

In other words, IRN-P can guess what you like (or learn predictive representations) and obtain a better user (or session) profile.

Besides, A3C-F achieves slightly better results than A3C-P, which is different from that in TAB8 .

A3C-F that trained with the click reward can preserve the sequential property of sessions, and thus provide auxiliary (implicit) information under the cold-start setting (in the warm-start setting, the agent using more clicked items as input may be biased and thus focuses on next-click prediction).Adaptation to user interest To demonstrate that IRN can improve data efficiency and promote quick adaptation to user interest, we create a more realistic scenario for online learning.

Specifically,

@highlight

We propose the IRN architecture to augment sparse and delayed purchase reward for session-based recommendation.

@highlight

The paper proposes improving the performance of recommendation systems through reinforcement learning by using an Imagination Reconstruction Network.

@highlight

The paper presents a session-based recommendation approach by focusing on user purchases instead of clicks. 
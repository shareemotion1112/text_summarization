Reinforcement learning typically requires carefully designed reward functions in order to learn the desired behavior.

We present a novel reward estimation method that is based on a finite sample of optimal state trajectories from expert demon- strations and can be used for guiding an agent to mimic the expert behavior.

The optimal state trajectories are used to learn a generative or predictive model of the “good” states distribution.

The reward signal is computed by a function of the difference between the actual next state acquired by the agent and the predicted next state given by the learned generative or predictive model.

With this inferred reward function, we perform standard reinforcement learning in the inner loop to guide the agent to learn the given task.

Experimental evaluations across a range of tasks demonstrate that the proposed method produces superior performance compared to standard reinforcement learning with both complete or sparse hand engineered rewards.

Furthermore, we show that our method successfully enables an agent to learn good actions directly from expert player video of games such as the Super Mario Bros and Flappy Bird.

Reinforcement learning (RL) deals with learning the desired behavior of an agent to accomplish a given task.

Typically, a scalar reward signal is used to guide the agent's behavior and the agent learns a control policy that maximizes the cumulative reward over a trajectory, based on observations.

This type of learning is referred to as "model-free" RL since the agent does not know apriori or learn the dynamics of the environment.

Although the ideas of RL have been around for a long time BID24 ), great achievements were obtained recently by successfully incorporating deep models into them with the recent success of deep reinforcement learning.

Some notable breakthroughs amongst many recent works are, the work from BID12 who approximated a Q-value function using as a deep neural network and trained agents to play Atari games with discrete control; who successfully applied deep RL for continuous control agents achieving state of the art; and BID22 who formulated a method for optimizing control policies with guaranteed monotonic improvement.

In most RL methods, it is very critical to choose a well-designed reward function to successfully learn a good action policy for performing the task.

However, there are cases where the reward function required for RL algorithms is not well-defined or is not available.

Even for a task for which a reward function initially seems to be easily defined, it is often the case that painful hand-tuning of the reward function has to be done to make the agent converge on an optimal behavior.

This problem of RL defeats the benefits of automated learning.

In contrast, humans often can imitate instructor's behaviors, at least to some extent, when accomplishing a certain task in the real world, and can guess what actions or states are good for the eventual accomplishment, without being provided with the detailed reward at each step.

For example, children can learn how to write letters by imitating demonstrations provided by their teachers or other adults (experts).

Taking inspiration from such scenarios, various methods collectively referred to as imitation learning or learning from experts' demonstrations have been proposed BID21 ) as a relevant technical branch of RL.

Using these methods, expert demonstrations can be given as input to the learning algorithm.

Inverse reinforcement learning BID15 ; BID1 ; BID28 ), behavior cloning BID20 ), imitation learning BID6 ; BID5 ), and curiosity-based exploration ) are examples of research in this direction.

While most of the prior work using expert demonstrations assumes that the demonstration trajectories contain both the state and action information (τ = {(s t )}) to solve the imitation learning problem, we, however, believe that there are many cases among real world environments where action information is not readily available.

For example, a human teacher cannot tell the student what amount of force to put on each of the fingers when writing a letter.

As such, in this work, we propose a reward estimation method that can estimate the underlying reward based only on the expert demonstrations of state trajectories for accomplishing a given task.

The estimated reward function can be used in RL algorithms in order to learn a suitable policy for the task.

The proposed method has the advantage of training agents based only on visual observations of experts performing the task.

For this purpose, it uses a model of the distribution of the expert state trajectories and defines the reward function in a way that it penalizes the agent's behavior for actions that cause it to deviate from the modeled distribution.

We present two methods with this motivation; a generative model and a temporal sequence prediction model.

The latter defines the reward function by the similarity between the state predicted by the temporal sequence model trained based on the expert's demonstrations and the currently observed state.

We present experimental results of the methods on multiple environments and with varied settings of input and output.

The primary contribution of this paper is in the estimation of the reward function based on state similarity to expert demonstrations, that can be measured even from raw video input.

Model-free Reinforcement Learning (RL) methods learn a policy π(a t |s t ) that produces an action from the current observation.

BID12 showed that a q-value function q(s t , a t ) can be approximated with a deep neural network, which is trained using hand-engineered scalar reward signals given to the agent based on its behavior.

Similarly, actor-critic networks in Deep Deterministic Policy Gradients (DDPG) can enable state of the art continuous control, e.g. in robotic manipulation by minimizing the distance between the end effector and the target position.

Since the success with DDPG, other methods such as Trust Region Policy Optimization (TRPO) BID22 ) and Proximal Policy Optimization (PPO) BID23 ) have been proposed as further improvements for model-free RL in continuous control problems.

Although RL enables an agent to learn an optimal policy without supervised training data, in the standard case, it requires a difficult task of hand-tuning good reward functions for each environment.

This has been pointed out previously in the literature BID1 ).

Several kinds of approaches have been proposed to workaround or tackle this problem.

An approach that does not require reward hand-tuning is behavior cloning based on supervised learning instead of RL.

It learns the conditional distribution of actions given states in a supervised manner.

Although it has an advantage of fast convergence BID5 ) (as behavior cloning learns a single action from states in each step), it typically results in compounding of errors in the future states.

An alternate approach is Inverse Reinforcement Learning (IRL) proposed in the seminal work by BID15 .

In this work, the authors try to recover the optimal reward function as a best description behind the given expert demonstrations from humans or other expert agents, using linear programming methods.

It is based on the assumption that expert demonstrations are solutions to a Markov Decision Process (MDP) defined with a hidden reward function BID15 ).

It demonstrated successful estimation of the reward function in case of relatively simple environments such as the grid world and the mountain car problem.

Another use of the expert demonstrations is initializing the value function; this was described by BID27 .

Extending the work by BID15 , entropy-based methods that compute the suitable reward function by maximizing the entropy of the expert demonstrations have been proposed by BID29 .

In the work by BID1 , a method was proposed for recovering the cost function based on expected feature matching between observed policy and the agent behavior.

Furthermore, they showed this to be the necessary and sufficient condition for the agent to imitate the expert behavior.

More recently, there was some work that extended this framework using deep neural networks as non-linear function approximator for both policy and the reward functions BID28 ).

In other relevant work by BID6 , the imitation learning problem was formulated as a two-players competitive game where a discriminator network tries to distinguish between expert trajectories and agent-generated trajectories.

The discriminator is used as a surrogate cost function which guides the agent's behavior in each step to imitate the expert behavior by updating policy parameters based on Trust Region Policy Optimization (TRPO) BID22 ).

Related recent works also include model-based imitation learning BID2 ) and robust imitation learning BID26 ) using generative adversarial networks.

All the above-mentioned methods, however, rely on both state and action information provided by expert demonstrations.

Contrarily, we learn only from expert state trajectories in this work.

A recent line of work aims at learning useful policies for agents even in the absence of expert demonstrations.

In this regard, trained an agent with a combination of reward inferred with intrinsic curiosity and a hand-engineered, complete or even very sparse scalar reward signal.

The curiosity-based reward is designed to have a high value when the agent encounters unseen states and a low value when it is in a state similar to the previously explored states.

The work reported successful navigation in games like Mario and Doom without any expert demonstrations.

In this paper, we compare our proposed methods with the curiosity-based approach and show the advantage over it in terms of the learned behaviors.

However, our methods assumed state demonstrations are available as expert data while the curiosity-based method did not use any demonstration data.

We consider an incomplete Markov Decision Process (MDP), consisting of states S and action space A, where the reward signal r : S × A → R, is unknown.

An agent can act in an environment defined by this MDP following a policy π(a t |s t ).

Here, we assume that we have knowledge of a finite set of optimal or expert state trajectories DISPLAYFORM0 ., n}. These trajectories can represent joints angles, raw images or any other information depicting the state of the environment.

Since the reward signal is unknown, our primary goal is to find a reward signal that enables the agent to learn a policy π, that can maximize the likelihood of these set of expert trajectories τ .

In this paper, we assume that the reward signal can be inferred entirely based on the current state and next state information, r : S × S → R. More formally, we would like to find a reward function that maximizes the following objective: DISPLAYFORM1 where r(s t+1 |s t ) is the reward function of the next state given the current state and p(s t+1 |s t ) is the transition probability.

We assume the performing to maximize the likelihood of next step prediction in equation 1 will be leading the maximizing the future reward when the task is deterministic.

Because this likelihood is based on similarity with demonstrations which are obtained while an expert agent is performing by maximizing the future reward.

Therefore we assume the agent will be maximizing future reward when it takes the action that gets the similar next step to expert demonstration trajectory data τ .

Let, τ = {s i t } i=1:M,t=1:N be the optimal states visited by the expert agent, where M is the number of demonstration episodes, and N is the number of steps within each episode.

We estimate an appropriate reward signal based on the expert state trajectories τ , which in turn is used to guide a reinforcement learning algorithm and learn a suitable policy.

We evaluate two approaches to implement this idea.

A straightforward approach is to first train a generative model using the expert trajectories τ .

Rewards can then be estimated based on similarity measures between a reconstructed state value and the actual currently experienced state value of the agent.

This method constrains exploration to the states that have been demonstrated by an expert and enables learning a policy that closely matches the expert.

However, in this approach, the temporal order of states are ignored or not readily accounted for.

This temporal order of the next state in the sequence is important for estimating the state transition probability function.

As such, the next approach we take is to consider a temporal sequence prediction model that can be trained to predict the next state value given current state, based on the expert trajectories.

Once again the reward value can be estimated as a function of the similarity measure between the predicted next state and the one actually visited by the agent.

The following sub-sections describes both these approaches in detail.

We train a deep generative model (three-layered fully connected auto-encoder) using the state values s i t for each step number t, sampled from the expert agent trajectories τ .

The generative model is trained to minimize the following reconstruction loss (maximize the likelihood of the training data): DISPLAYFORM0 where θ * g represents the optimum parameters of the generative model.

Following typical settings, we assume p(s i t ; θ g ) to be a Gaussian distribution, such that equation FORMULA2 DISPLAYFORM1 The reward value is estimated as a function of the difference between the actual state value s t+1 and the generated output g(s DISPLAYFORM2 where s t is the current state value, and ψ can be a linear or nonlinear function, typically hyperbolic tangent or gaussian function.

In this formulation, if the current state is similar to the reconstructed state value, i.e. g(s t ; θ g ), the estimated reward value will be higher.

However, if the current state is not similar to the generated state, the reward value will be estimated to be low.

Moreover, as a reward value is estimated at each time step, this approach can be used even in problems which originally had a highly sparse engineered reward structure.

In this approach, we learn a temporal sequence prediction model (the specific networks used are mentioned in the corresponding experiments sections) such that we can maximize the likelihood of the next state given the current state.

As such the network is trained using the following objective function, DISPLAYFORM0 where θ * h represents the optimal parameters of the prediction model.

We also assume the probability of the next state given the previous state value, p(s i t+1 |s i t ; θ h ) to be a Gaussian distribution.

As such the objective function can be seen to be minimizing the mean square error, DISPLAYFORM1 DISPLAYFORM2 The estimated reward here, can also be interpreted akin to the generative model case.

Here, if the agent's policy takes an action that changes the environment towards states far away from the expert trajectories, the corresponding estimated reward value is low.

If the actions of agent bring it close to the expert demonstrated trajectories, thereby making the predicted next state match with the actual visited state value, the reward is estimated to be high.

This process of reward shaping or guidance can enable the agent to learn a policy that is optimized based on the expert demonstration trajectories.

Algorithm 1 explains the step by step flow of the proposed methods.

Given trajectories τ from expert agent 3: DISPLAYFORM3 end for 6: end procedure 7: procedure REINFORCEMENT LEARNING 8: DISPLAYFORM4 Observe state s t

Select/execute action a t , and observe state s t+1 11: DISPLAYFORM0 Update the deep reinforcement learning network using the tuple (s t , a t , r t , s t+1 )

end for 14: end procedure

In order to evaluate our reward estimation methods, we conducted experiments across a range of environments.

We consider five different tasks, namely: robot arm reaching task (reacher) to a fixed target position, robot arm reaching task to a random target position, controlling a point agent for reaching a target while avoiding an obstacle, learning an agent for longest duration of flight in the Flappy Bird video game, and learning an agent for maximizing the traveling distance in Super Mario Bros video game.

We consider a 2-DoF robot arm in the x-y plane that has to learn to reach with the end-effector a point target.

The first arm of the robot is a rigidly linked (0, 0) point, with the second arm linked to its edge.

It has two joint values θ = (θ 1 , θ 2 ), θ 1 ∈ (−∞, +∞), θ 2 ∈ [−π, +π] and the lengths of arms are 0.1 and 0.11 units, respectively.

The robot arm was initialized by random joint values at the initial step for each episode.

In the following experiments, we have two settings: fixed point target, and a random target.

The applied continuous action values a t is used to control the joint angles, such that,θ = θ t − θ t−1 = 0.05 a t .

Each action value has been clipped the range [−1, 1].

The reacher task is enabled using the physics engine within the roboschool environment BID3 BID16 ).

Figure 1 describes the roboshool environment.

The robot arms are in blue, the blue-green point is the end-effector, and the pink dot is the desired target location.

In this experiment, the target point p tgt is always fixed at (0.1, 0.1).

The state vector s t consists of the following values: absolute end position of first arm (p 2 ), joint value of elbow (θ 2 ), velocities of the joints (θ 1 ,θ 2 ), absolute target position (p tgt ), and the relative end-effector position from target (p ee − p tgt ).

We used DDPG ) for this task, with the number of steps for each episode being 500 in this experiment 1 .

The reward functions used in this task were as follows:Dense reward : DISPLAYFORM0 Sparse reward : r t = − tanh(α p ee − p tgt 2 ) + r env t , DISPLAYFORM1 DISPLAYFORM2 GM with r env t : rGM (1000 episodes) : DISPLAYFORM3 GM with action : DISPLAYFORM4 where r env t is the environment specific reward, which is calculated based on the cost for current action, − a t 2 .

This regularization is required for finding the shortest path to reach the target.

As this cost is critical for fast convergence, we use this in all cases.

The dense reward is a distance between end-effector and the target, and the sparse reward is based on a bonus for reaching.

The generative model parameters θ 2k is trained by τ 2k trajectories that contains only states of 2000 episodes from an agent trained during 1k episodes with dense reward.

The generative network has 400, 300 and 400 units fully-connected layers, respectively.

They also have ReLU activation function, with the batch size being 16, and number of epochs being 50.

θ 1k is trained from τ 1k trajectories that is randomly picked 1000 episodes from τ 2k .

The GM with action uses a generative model θ 2k,+a that is trained pairs of state and action for 2000 episodes for same agents as τ 2k .

We use a tanh nonlinear function for the estimated reward in order to keep a bounded value.

The α, β change sensitiveness of distance or reward, were both set to 100 2 .

Here, we also compare our results with behavior cloning (BC) method BID20 where the trained actor networks directly use obtained pairs of states and actions.

DISPLAYFORM5 Figure 1: The environment of reacher task.

The reacher has two arms, and objective of agent is reaching endeffector (green) to target point (red).

FIG3 shows the difference in performance by using the different reward functions 3 .

All methods are evaluated by a score (y-axis), which was normalized to a minimum (0) and a maximum (1) value.

The proposed method, especially "GM with r env t ", manages to achieve a score nearing that of the dense reward, with the performance being much better as compared to the sparse reward setting.

Moreover, the learning curves based on the rewards estimated with the generative model show a faster convergence rate.

However, the result without environment specific reward, i.e. with the additional action regularization term, takes a longer time to converge.

This is primarily because of the fact that GM reward is reflective of the distance between target and end-effector, and cannot directly account for the action regularization.

The GM reward based on τ 1k underperforms as compared with GM reward based on τ 2k because of the lack of demonstration data.

FIG4 shows the reward value of each end-effector point.

GM estimated reward using τ 2k has better reward map as compared to GM estimated reward using τ 1k .

However, these demonstrations data FIG11 ) are biased by the robot trajectories.

Thus, a method of generating demonstration data that normalizes or avoids such bias will further improve the reward structure.

If the demonstrations contain the action information in addition to state information, behavior cloning achieves good performance.

Surprisingly, however, when using both state and action information in the generative model, "GM [state, action]", the performance of the agent is comparatively poor.

In this experiment, the target point p tgt is initialized by a random uniform distribution of [−0.27, +0.27] , that includes points outside of the reaching range of the robot arms.

Furthermore, we removed the relative position of the target from the input state information.

This makes the task more difficult.

The state vector s t has the following values: p 2 , θ 2 ,θ 1 ,θ 2 , p tgt .

In the previous experiment, the distribution of states in expert trajectories is expected to be similar to the reward structure due to a fixed target location.

However, when the target position changes randomly, this distribution is not fixed.

We, therefore, evaluate with the temporal sequence prediction model h(s t ; θ h ) in this experiment.

The RL setting is same as the previous experiment, however we changed the total number of steps within each episode to 400.

The reward functions used in this experiment were calculated as follows: DISPLAYFORM0 Sparse reward : r t = tanh(−α p ee − p tgt 2 ) + r env t , DISPLAYFORM1 LSTM reward : r t = tanh(−γ s t+1 − h(s t:t-n ; θ lstm ) 2 ) + r DISPLAYFORM2 The expert demonstrations τ were obtained using the states of 2000 episodes running a trained agent with dense hand-engineered reward.

The GM estimated reward uses the same setting as in the previous experiment.

NS is a model that predicts the next state given current state, and was trained using demonstration data τ 4 .

The LSTM model uses Long short-term memory BID7 ) as a temporal sequence prediction model.

The state in reacher task does not contain time sequence data, hence we use a finite state history as input.

LSTM model has three layers 5 and one fully-connected layer with 40 ReLU activation units.

The forward model based reward estimation is based on predicting the next state given both the current state and action 6 .

Here, we also compared with the baseline behavior cloning method.

In this experiment, α is 100, β is 1, and γ is 10.

FIG3 shows the performance of the trained agent using the different reward functions.

In all cases using estimated rewards performances significantly better than the sparse reward case.

The LSTM based prediction method gives the best results, reaching close to the performance obtained with dense hand engineered reward function.

As expected, the GM based reward estimation fails to work well in this relatively complex experimental setting.

The NS model estimated reward, which predicts next state given only the current state information, has comparable performance with LSTM based prediction model during the initial episodes.

The FM based reward function also performs poorly in this experiment.

Comparatively, the direct BC works relatively well.

This indicates that it is better to use behavior cloning than reward prediction when both state and action information are available from demonstration data.

Starting with this experiment, we evaluate using only the temporal sequence prediction method.

As such, here we use a finite history of the state values in order to predict the next state value.

We assume that predicting a part of the state that is related to a given action allows the model to make a better estimate of the reward function.

Former work by predicts a function of the next state, φ(s t+1 ) rather than predicting the raw value s t+1 , as in this paper.

In this experiment, we also changed the non-linear function ψ in the proposed prediction method to a Gaussian function (as compared to the hyperbolic tangent function used in previous experiments).

This allows us to compare the robustness of our proposed method for reward estimation to different non-linear functions.

We develop a new environment that adds an obstacle to the reaching task.

This reacher is a twodimensional point (x, y) that uses position control.

In FIG5 we show the modified environment setup.

The agent's goal is to reach the target while avoiding the obstacle in this case.

The initial position of agent, the target position, and an obstacle position were initialized randomly.

The state value contains the agent absolute position (p t ), current velocity of the agent (ṗ t ), a target absolute position (p tgt ), an obstacle absolute position (p obs ), and the relative location of target and obstacle with respect to the agent (p t −p tgt , p t −p obs ).

Once again the RL algorithm used in this experiment was DDPG ) for continuous control 7 .

The number of steps for each episode set to 500.

Here, we used the following reward functions: DISPLAYFORM0 LSTM reward : DISPLAYFORM1 DISPLAYFORM2 where h (s t:t-n ; θ lstm ) is a network that predicts a selected part of state values given a finite history of state information.

The dense reward is composed of both, the target distance cost and the obstacle distance bonus.

The optimal state trajectories τ contains 800 "human guided" demonstration data.

In this case, the LSTM network consisted of two layers, each with 256 units with ReLU activations.

In this experiment, σ 1 is 0.005, and σ 2 is 0.002 8 .

FIG6 shows the performance with the different estimated or hand-engineered reward settings.

The LSTM based prediction method learns to reach the target faster than the dense reward, while LSTM (s ) has the best overall performance by learning with human-guided demonstration data.

The agent (green) will move to reach the target (red), while avoiding the obstacle (pink).

DISPLAYFORM3

We use a re-implementation BID10 ) of Android game, "Flappy Bird", in python (pygame).

The objective of this game is to pass through the maximum number of pipes without collision.

The control is a single discrete command of whether to flap the bird wings or not.

The state has four consecutive gray frames (4 x 80 x 80).

The RL is trained by DQN BID12 ) 9 , and the update frequency of deep network is 100 steps.

The used rewards are, DISPLAYFORM0 if pass through a pipe; −1 if collide to a pipe.

LSTM reward : DISPLAYFORM1 which s t+1 is an absolute position of the bird, which can be given from simulator or it could be processed by pattern matching or CNN from raw images, h (s t ; θ lstm ) is a predicted the absolute position.

Hence, LSTM is trained for predicting absolute position of bird location given images.

The τ of this experiment is 10 episodes data from a trained agent in the repository by BID10 .

Also, we also compared with the baseline behavior cloning method.

In this experiment, σ is 0.02.

FIG8 shows the result of LSTM reward is better than normal "hand-crafted" reward.

The reason for this situation is, the normal dense reward just describes the traveling distance, but our LSTM reward will teach which absolute transition of bird is good.

Also, LSTM has better convergence than BC result; the reason is the number of demonstrations is not enough for behavior cloning method.

In the final task, we consider a more complex environment in order to evaluate our proposed reward estimation method using only state information.

Here we use the Super Mario Bros classic Nintendo video game environment (Paquette (2017)).

Our proposed method estimates reward values based on expert gameplay video data (using only the state information in the form of image frames).In this experiment, we also benchmarked against the recently proposed curiosity-based method ) using the implementation provided by the same authors BID17 ).

This was used as the baseline reinforcement learning technique.

Unlike in the actual game, here we always initialize Mario to the starting position rather than a previously saved checkpoint.

This is a discrete control setup, where, Mario can make 14 types of actions 10 .

The state information consists of sequential input of four 42 x 42 gray image frames 11 .

Here we used the A3C RL algorithm BID13 ).

We used gameplay of stage "1-1" for this experiment, with the objective of the agent being to travel as far as possible and achieve as high a score as possible.

The rewards functions used in this experiment were as follows: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 Curiosity DISPLAYFORM3 DISPLAYFORM4 where position t is the current position of Mario at time t, score t is the current score value at time t, and s t are screen images from the Mario game at time t.

The position and score information are obtained using the Mario game emulator.

In this experiment, we use a three-dimensional convolutional neural network BID8 ) (3D-CNN) for our temporal sequence prediction method.

In order to capture expert demonstration data, we took 15 game playing videos by five different people 12 .

In total, the demonstration data consisted of 25000 frames.

The length of skipped frames in input to the temporal sequence prediction model was 36, as humans cannot play as fast as an RL agent; however, we did not change the skip frame rate for the RL agent.

The 3D-CNN consisted of 4 layers 13 and a final layer to reconstruct the image.

The agent was trained using 50 epochs with a batch size of 8.

We implemented two prediction methods for reward estimation.

In the naïve method the Mario agent will end up getting positive rewards if it sits in a fixed place without moving.

This is because it can avoid dying by just not moving.

However, this is clearly a trivial suboptimal policy.

Hence, we implemented the alternate reward function based on the same temporal sequence prediction model, but we apply a threshold value that prevents the agent from converging onto such a trivial solution.

Here, the value of ζ is 0.025, which was calculated based on the reward value obtained by just staying fixed at the initial position.

FIG10 shows the performance with the different reward functions.

Here, the graphs directly show the average results over multiple trials.

As observed, the agent was unable to reach large distances even while using "hand-crafted" dense rewards and did not converge to the goal every time 14 ; this behavior was also observed by for their reward case.

As observed from the average curves of FIG10 , the proposed 3D-CNN method learns relatively faster as compared to the curiosity-based agent ).

As expected the 3D-CNN (naïve) method converged to a solution of remaining fixed at the initial state.

As future work, we hope to improve the performance in this game setting using deeper RL networks, as well as large input image sizes.

Overall estimating reward from φ(s t ) without the need of action data, allows an agent to learn suitable policy directly from raw video data.

The abundance of visual data creates ample opportunity for this type of reward estimation method to be explored further in different video game settings.10 A single action is repeated for six consecutive frames.

Please refer to ) for details.

11 Every next six frames were skipped.

12 All videos consisted of games where the player succeeded in clearing the stage.

13 Two layers with (2 x 5 x 5), two layers with (2 x 3 x 3) kernels, all have 32 filters, and every two layers with (2, 1, 1) stride.14 By our experiment, even if it trained long steps, such as 3.0M; it just reached around 600 -700 averagely.

In this paper, we proposed two variations of a reward estimation method via state prediction by using state-only trajectories of the expert; one based on an autoencoder-based generative model and one based on temporal sequence prediction using LSTM.

Both the models were for calculating similarities between actual states and predicted states.

We compared the methods with conventional reinforcement learning methods in five various environments.

As overall trends, we found that the proposed method converged faster than using hand-crafted reward in many cases, especially when the expert trajectories were given by humans, and also that the temporal sequence prediction model had better results than the generative model.

It was also shown that the method could be applied to the case where the demonstration was given by videos.

However, detailed trends were different for the different environments depending on the complexity of the tasks.

Neither model of our proposed method was versatile enough to be applicable to every environment without any changes of the reward definition.

As we saw in the necessity of the energy term of the reward for the reacher task and in the necessity of special handling of the initial position of Mario, the proposed method has a room of improvements especially in modeling global temporal characteristics of trajectories.

We would like to tackle these problems as future work.

The DDPG's actor network has 400 and 300 unites fully-connected (fc) layers, the critic network has also 400 and 300 fully-connected layers, and each layer has a ReLU (Nair & Hinton FORMULA1 ) activation function.

We put the tanh activation function at the final layer of actor network.

Without this modification, the normal RL takes a long time to converge.

Also, initial weights will be set from a uniform distribution U (−0.003, 0.003).

The exploration policy is Ornstein-Uhlenbeck process BID25 ) (θ = 0.15, µ = 0, σ = 0.01), size of reply memory is 1M, and optimizer is Adam (Kingma & Ba FORMULA1 ).

We implemented these experiments by Keras-rl (Plappert (2016)), Keras BID4 ), and Tensorflow BID0 ) libraries. : These are scatter-plots of end-effector positions (blue) for each state of captured demonstration τ 500 , τ 1k , τ 2k , each point is drawn by α is 0.01.

And the fixed target position is also plotted (red).

Notes that this is just plotting end-effector position, there is more variation in other state values.

For example, even if the end-effector position were same, arms' pose (joint values) might be different.

Note that τ 500 is not used in the experiment.

The DDPG's actor network has 64 and 64 unites fully-connected layers, the critic network has also 64 and 64 fully-connected layers, and each layer has a ReLU activation function.

Initial weights will be set from a uniform distribution U (−0.003, 0.003).

The exploration policy is Ornstein-Uhlenbeck process BID25 ) (θ = 0.15, µ = 0, σ = 0.01), size of reply memory is 500k, and optimizer is Adam.

<|TLDR|>

@highlight

Reward Estimation from Game Videos
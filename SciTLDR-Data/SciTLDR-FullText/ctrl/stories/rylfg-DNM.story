We propose to extend existing deep reinforcement learning  (Deep RL) algorithms by allowing them to additionally choose sequences of actions as a part of their policy.

This modification forces the network to anticipate the reward of action sequences, which, as we show, improves the exploration leading to better convergence.

Our proposal is simple, flexible, and can be easily incorporated into any Deep RL framework.

We show the power of our scheme by consistently outperforming the state-of-the-art GA3C algorithm on several popular Atari Games.

Basic reinforcement learning has an environment and an agent.

The agent interacts with the environment by taking some actions and observing some states and rewards.

At each time step t, the agent observes a state s t and performs an action a t based on a policy π(a t |s t ; θ).

In return to the action, the environment provides a reward r t and the next state s t+1 .

This process goes on until the agent reaches a terminal state.

The learning goal is to find a policy that gives the best overall reward.

The main challenges here are that the agent does not have information about the reward and the next state until the action is performed.

Also, a certain action may yield low instant reward, but it may pave the way for a good reward in the future.

Deep Reinforcement Learning BID6 has taken the success of deep supervised learning a step further.

Prior work on reinforcement learning suffered from myopic handcrafted designs.

The introduction of Deep Q-Learning Networks (DQN) was the major advancement in showing that Deep Neural Networks (DNNs) can approximate value and policy functions.

By storing the agent's data in an experience replay memory, the data can be batched BID8 BID9 or randomly sampled BID4 BID12 from different time-steps and learning the deep network becomes a standard supervised learning task with several input-output pairs to train the parameters.

As a consequence, several video games could be played by directly observing raw image pixels BID1 and demonstrating super-human performance on the ancient board game Go .In order to solve the problem of heavy computational requirements in training DQN, several followups have emerged leading to useful changes in training formulations and DNN architectures.

Methods that increase parallelism while decreasing the computational cost and memory footprint were also proposed BID7 BID6 , which showed impressive performance.

A breakthrough was shown in BID6 , where the authors propose a novel lightweight and parallel method called Asynchronous Advantage Actor-Critic (A3C).

A3C achieves the stateof-the-art results on many gaming tasks.

When the proper learning rate is used, A3C learns to play an Atari game from raw screen inputs more quickly and efficiently than previous methods.

In a remarkable followup to A3C, BID0 proposed a careful implementation of A3C on GPUs(called GA3C) and showed the A3C can accelerated significantly over GPUs, leading to the best publicly available Deep RL implementation, known till date.

Slow Progress with Deep RL:

However, even for very simple Atari games, existing methods take several hours to reach good performance.

There is still a major fundamental barrier in the current Deep RL algorithms, which is slow progress due to poor exploration.

During the early phases, when the network is just initialized, the policy is nearly random.

Thus, the initial experience are primarily several random sequences of actions with very low rewards.

Once, we observe sequences which gives high rewards, the network starts to observe actions and associate them with positive rewards and starts learning.

Unfortunately, finding a good sequence via network exploration can take a significantly long time, especially when the network is far from convergence and the taken actions are near random.

The problem becomes more severe if there are only very rare sequence of actions which gives high rewards, while most others give on low or zero rewards.

The exploration can take a significantly long time to hit on those rare combinations of good moves.

In this work, we show that there is an unusual, and surprising, opportunity of improving the convergence of deep reinforcement learning.

In particular, we show that instead of learning to map the reward over a basic action space A for each state, we should force the network to anticipate the rewards over an enlarged action space A + = K k=1 A k which contains sequential actions like (a 1 , a 2 , ..., a k ).

Our proposal is a strict generalization of existing Deep RL framework where we allow to take a premeditated sequence of action at a given state s t , rather than only taking a single action and re-deciding the next action based on the outcome of the first action and so on.

Thus the algorithm can pre-decide on a sequence of actions, instead of just the next best action, if the anticipated reward of the sequence is good enough.

Our experiments shows that by simply making the network anticipate the reward for a sequence of action, instead of just the next best actions, the network shows significantly better convergence behavior consistently.

We even outperform the fastest known implementation, the GPU accelerated version of A3C (GA3C).

The most exciting part is that that anticipation can be naturally incorporated in any existing implementation, including Deep Q Network and A3C.

We simply have to extend the action set to also include extra sequences of actions and calculate rewards with them for training, which is quite straightforward.

Methods for reinforcement learning can be classified into three broad classes of solutions: Valuebased, Policy-based and Actor-Critic.

The main idea in Value based methods is to define a function called Q-function (Q stands for Quality) which estimates the future reward for a given state-action pair.

One popular way to construct and learn a Q-function is called Deep-Q learning BID5 .

The Q-function is iteratively learned by minimizing the following loss function DISPLAYFORM0 Here, s is the current state, a is the action, r is the reward earned for action a and s is next state that we end up.

The recursive definition DISPLAYFORM1 comes from the Bellman equation in Dynamic Programming.

This is called 1-step Q-Learning as we only perform one action and observe the reward.

If we instead observe a sequence of k actions and the states resulting from those actions, we can define the Q function as follows DISPLAYFORM2

In policy-based model-free methods, a function approximator such as a neural network computes the policy π(a t |s t ; ), where θ is the set of parameters of the function.

θ is updated by maximizing the cumulative reward as per Bellman Equation given by DISPLAYFORM0 One of the popular approaches in policy-based methods is REINFORCE BID13 ).

An intuitive baseline is the mean of all previous rewards.

If the current reward is higher than the mean of all previous rewards, then the current action is 'good'.

Otherwise, it is 'bad'.

That is encapsulated in the loss function directly.

Baseline b t being independent of current state s t is not the beneficial because it has no context of the current state.

Hence, we would like to redefine it as b t (s t ).

One such popular function is DISPLAYFORM0 Here, V π is the Value function.

This approach marks the transition from pure Policy-Based Methods to a blend of Policy-based and Value-based methods.

Here, the policy function acts as an actor because it is responsible for taking actions and the Value function is called the critic because it evaluates the actions taken by the actor.

This approach is called the Actor-Critic Framework (Sutton & Barto).

We still solve the parameters for policy function but use a Value function to decide on the 'goodness' of a reward.

A3C BID6 ) is currently the state-of-the-art algorithm on several popular games.

It uses an Asynchronous framework in which multiple agents access a common policy, called central policy, and play simultaneously.

They communicate the gradients after atmost t max actions.

All the communicated gradients from multiple agents are then used to update the central policy.

Once the policy parameters are updated, they are communicated back to all the agents playing.

The framework uses a shared neural network which gives 2 outputs, one is the policy distribution, and the other is the Value function.

Policy π(a t |s t , θ) is the output of softmax(because it is a distribution) and Value function V (s t , θ) is the output of a linear layer.

The objective function for policy update of A3C is as follows(note that we maximize the policy objective) DISPLAYFORM0 Here, the first part is typical actor-critic framework except that the Value function now shares parameters θ.

The second part is the entropy over the policy distribution of actions.

From information theory, we know that entropy is maximum when all actions are equally likely.

Hence, this term favors exploration of new actions by enforcing some probability to unlikely actions.

The weight β decides how much priority we give to exploration.

Please note that A3C pseudocode in the original paper doesn't mention anything about entropy, but we include it here as it is discussed in various other references.

Since, V (s t ; θ) is also a function of θ, we also get value-function-gradients from V by minimizing the DQN-type loss function DISPLAYFORM1 Both the gradients are calculated and stored by each agent until they terminate or perform t max actions.

The collection of gradients is then communicated, and the updated central network is now available for all agents.

The major concern with A3C is that it relies on sequential training.

More generally, all Reinforcement Learning paradigms are plagued by the fact that we do not have a pre-decided training and testing data and we have to leverage information while training.

That renders GPUs and other parallelizations useless for implementing RL algorithms, particularly A3C.

GA3C BID0 was proposed as a follow-up and an alternative framework for A3C that enables the usage of GPU.

The broad idea of GA3C is to use larger batches of inputoutput(output in our case refers to reward) pairs to facilitate better usage of GPUs like usual supervised learning.

Since we need to perform actions and observe rewards, every agent GA3C maintains two queues called P redictionQueue and T rainingQueue.

Every Agent queues up Policy requests in P redictionQueue and submits a batch of input-reward pairs to the T rainingQueue.

Instead of having a central policy that every agent uses to predict, GA3C has a predictor that takes P redictionQueues from as many agents as possible and sends an inference query to GPU(this is where the batch size increases thereby making use of GPU).

Predictor then sends updated policy to all agents that sent their P redictionQueues.

On the other hand, there's a trainer component of GA3C which takes the input-reward batches from as many agents as possible and updates model parameters by sending the batches to a GPU.GA3C presents new challenges as it has to deal with trade-offs like size of data trasfer vs number of data transfers to GPU, number of predictors N P vs size of prediction batches etc.

While we build our idea on GA3C, we set most of these parameters to their defaults.

Our proposal is an unusually straightforward extension, and a strict generalization of the existing deep reinforcement learning algorithms.

At high level, by anticipation we extend the basic action set A to an enlarged action space DISPLAYFORM0 A k , which also includes sequences of actions up to length K. As an illustration, let us say A = {L, R} and we allow 2-step anticipation, therefore our new action space is A + = A ∪ A 2 = {L, R, LL, LR, RL, RR}. Each element a + belonging to A + is called a meta-action, which could be a single basic action or a sequence of actions.

Typical deep reinforcement learning algorithms have a DNN to output the estimated Q values or policy distributions according to basic action set A. In our algorithm, we instead let the DNN output values for each meta-action in the enlarged action set A + .

Overall, we are forcing the network to anticipate the "goodness" of meta-actions a little further, and have a better vision of the possibilities earlier in the exploration phase.

From human observations and experiences in both sports and video games, we know the importance of "Combo" actions.

Sometimes single actions individually do not have much power, but several of common actions could become very powerful while performed in a sequential order.

For example, in the popular game CounterStrike, jump-shoot combo would be a very good action sequence.

This kind of observation inspires us to explore the potential of "Combos", i.e. multi-step anticipatory actions in reinforcement learning.

Moreover, the advantage of anticipatory actions over the standard ones for improving exploration is analogous to how higher n-grams statistics help in better modeling compared to just unigrams in NLP.Another subtle advantage of anticipating rewards for sequence of actions is better parameter sharing which is linked with multi-task learning and generalization.

Parameter Sharing: Back in 1997, BID3 showed the advantage of parameter sharing.

In particular, it showed that a single representation for several dependent tasks is better for generalization of neural networks than only learning from one task.

With the addition of meta-action (or extra actions sequences), we are forcing the network layers to learn a representation which is not only useful in predicting the best actions but also predicts the suitability of meta-actions, which is a related task.

A forced multi-task learning is intrinsically happening here.

As illustrated in Figure 1 , the black box parameters are a shared representation which is simultaneously learned from the gradients of basic actions as well as meta-actions.

This additional constraint on the network to predict more observable behaviors regularizes the representation, especially in the early stages.

Anticipatory Deep Q Network: Although our main proposal is A4C which improves the current state-of-the-art A3C algorithm, to illustrate the generality of our idea, we start with a simpler algorithm -Anticipatory Deep Q Network (ADQN).

DQN is a value-based algorithm whose network approximates Q values for each action.

If we see each gradient update as a training sample sent to the network, DQN generates 1 training sample for each action-reward frame.

We believe one frame could provide more information than that.

With meta-action, i.e., ADQN algorithm, instead we force the network to output Q values for each meta-action in the enlarged action space.

For example, in CartPole game, the basic actions are L, R. In ADQN, we let the output values be over A + = {L, R, LL, LR, RL, RR}. For an experience sequence (..., s i , a i , r i , s i+1 , a i+1 , r i+1 , s i+2 , ...), we will get two updates for state s i : DISPLAYFORM0 In this way, we could abtain two gradient updates for each state.

This update improves the intermediate representaion (parameter sharing) aggresively leading to superior convergence.

In practice, we could organize them into one single training vector, as illustrated in the Figure 1 .

This algorithm performs very well on CartPole game (see Section 4.1).Figure 1: A toy example for ADQN with an enlarged action set {L, R, LL, LR, RL, RR}. For input s 0 , we have 2 gradients, one for action L and other for action LR.

In the previous section, we have shown that anticipation can be used on value-based reinforcement methods like DQN.

However, DQN is not the state-of-art algorithm, and it converges relatively slowly on more complex tasks like Atari games.

Due to the simplicity and generality of our method of anticipation, it is also directly applicable to -Asynchronous Advantage Actor-Critic (A3C) algorithm.

As mentioned earlier, A3C uses a single deep neural network with |A| policy nodes and 1 value node.

To enforce anticipation, we can just enlarge the number of policy nodes in the layer without changing other network architecture.

Generally, if we want to support up to K steps of action sequences, we need |A + | policy nodes for the output layer, where DISPLAYFORM0 The new action space A + contains both basic single actions and sequences of actions.

This improved algorithm is called Anticipatory asynchronous advantage actor-critic (A4C).In A4C algorithm, the neural network is used for two parts: prediction and training.

In the prediction part, A4C lets the neural network output a distribution of actions from A + .

For each state, we choose a meta-action a + according to the output distribution.

If a + contains only one action, this single action will be executed.

If a + corresponds to an action sequence (a 1 , a 2 , ..., a k ), these actions will be executed one by one in order.

A4C is a strict generalization of A3C, and it allows for three kinds of gradient updates for given action-reward frame: dependent updating (DU), independent updating (IU), and switching.

A meta-action a + can be viewed as a combination of single actions.

On the other hand, several basic actions taken sequentially could be viewed as a meta-action.

From here comes our intuition of dependent updating, where each meta-action has its dependent basic actions.

When we take a meta-action and get rewards, we not only calculate the gradients for this meta-action, but also for its corresponding basic actions.

And for a sequence of basic actions, even if they were not taken as a meta-action, we also update the network as it takes the corresponding meta-action.

For example, in a 2-step anticipation setting, we get an experience queue of (s 0 , a 0 , r 0 , s 1 , a 1 , r 1 , s 2 , ...).

No matter (a 0 ) was taken as a basic action or (a 0 , a 1 ) was taken as a meta-action, we will update both of them for state s 0 .

In this case, we get 2 times more gradient updates as A3C for the same amount of episodes, resulting in aggressive updates which lead to accelerated convergence, especially during the initial phases of the learning.

We call this kind of dependent updating version of A4C as DU-A4C.

Our pseudocode for DU-A4C is presented in Algorithm 1.Algorithm 1 Anticipatory asynchronous advantage actor-critic with Dependent Updating (DU-A4C) -pseudocode for each actor learner thread // Assume global shared parameter vectors θ and θ v and global shared T = 0 // Assume thread-specific parameter vector θ and θ v // Assume a basic set A = {a i } and the corresponding enlarged action set A + = {a DISPLAYFORM0 Initialize thread step counter t ← 1 repeat Reset gradients: dθ ← 0 and dθ v ← 0 Synchronize thread-specific parameters θ = θ and θ v = θ t start = t Get state s t repeat Choose a + t according to policy π(a + t |s t ; θ ) for a i in the basic action sequence (a 1 , a 2 , ...) corresponding to a + t do Perform a i , receive reward r t and new state s t+1 t ← t + 1 T ← T + 1 end for until terminal s t or t − t start >= t max R = 0 for terminal s t V (s t , θ v ) for non-terminal s t for i ∈ {t − 1, ..., t start } do R ←

r i + γR for j ∈ {i, ..., min(i + K, t − 1)} do Let a + ij be the meta-action corresponding to the sequence (a i , ..., a j ) Accumulate gradients wrt θ : dθ ← dθ + ∇ θ log π(a DISPLAYFORM1 2 /∂θ v end for Perform asynchronous update of θ using dθ and of θ v using dθ v .

until T > T max

Independent update is a very simple and straightforward updating method that we could just view each meta-action a + as a separate action offered by the environment.

The reward of a + is the sum of rewards of taking all the basic actions in a + one by one in order.

The next state of a + is the state after taking all the actions in the sequence.

While updating, we only use the information of reward, and the next state of a + without regards to the dependencies and relations between meta-actions.

The pseudocode is in Algorithm 2 (in Supplementary Materterials).Clearly, IU leads to less aggressive updates compared to DU.

Even though independent updating makes no use of extra information from the intrinsic relations of meta-actions, it still has superior performance in experiments.

The reason is that there exist some patterns of actions that yield high rewards consistently and anticipatory action space enables the network to explore this kind of action patterns.

Our experiment suggests, DU-A4C converges faster over Atari games for the first few hours of training.

DU-A4C shows a big gap over the speed of original A3C.

However, after training for a longer time, we observe that aggressive updates cause the network to saturate quickly.

This phenomenon is analogous to Stochastic Gradient Descent (SGD) updates where initial updates are aggressive but over time we should decay the learning rate BID2 .Technically, dependent updating makes good use of information from the anticipatory actions and yields fast convergence.

Independent updating method offers a less aggressive way of updating but it can sustain the growth for longer durations.

Thus, we propose a switching method to combine the advantages of these two updating methods.

Switching is simple: we first use dependent updating method to train the network for a while, then switch over to independent updating method from there on.

Since the network is the same for both updating methods, the switching process is quite trivial to implement.

We notice that this approach consistently stabilizes training on many scenarios(explained in the Section 4).

As mentioned, switching is analogous to decaying learning rate with epochs in a typical Neural Network training, the difference being our approach is a hard reduction while learning rate decay is a soft reduction.

The tricky part is when should we switch.

Currently, it is more of a heuristic way: for each game, we typically switch half-way.

The reason we choose half is that we want to utilize DU to converge quickly in first half and IU to stabilize and continuously increase in the second half.

In our experiments, we realize that switching seems to have robust performance in experiments with regards to different choice of switching points.

4.1 STUDY OF EXPLORATION USING CARTPOLE GAME Figure 2 : Results and analysis of CartPole game.

Left: ADQN vs DQN on CartPole-v0; Right: The performed action distributions at different training stages.

We divide the total 5000 episodes into 5 stages, and plot the distribution at each stage.

To understand the dynamics of the Anticipatory network, we use a simple, classic control game Cartpole.

Cartpole game has only 2 basic actions Left and Right, and its state space is R 4 .

We perform a 2-step Anticipatory DQN(mentioned in section 3.1) with Dependent Updates(DU) and compare against regular DQN.

Owing to the simplicity of CartPole, we do not compare A4C vs A3C here, which we reserve for Atari games.

We notice a significant jump in the score by using metaactions space given by {L, R, LL, LR, RL, RR}, instead of just {L, R}. Although CartPole game In the right plot (Figure 2(b) ), we also show the probability (frequency) distributions of 6 metaactions in different learning periods.

It is clear from the plots that as learning goes on, the probability of basic actions increases and the probability of multi-step action drops.

This trend shows that multi-step actions help the agent to better explore initially, with the anticipated vision of the future, obtaining better rewarding actions.

Once the network has seen enough good actions, it figures our the right policy and seems to select basic actions only.

Next, we demonstrate out A4C experiments on 4 popular Atari-2600 games namely Pong, Qbert, BeamRider, and SpaceInvaders.

We use the environments provided by OPENAI GYM for both these classes of games.

Atari-2600 games are the standard benchmarks for Reinforcement Learning Algorithms.

We compare our results against the state-of-the-art GPU based Asynchronous Actor-Critic (GA3C) framework from NVIDIA whose code is publicly available(at https://github.com/ NVlabs/GA3C)).

In order to have uniform playing fields for both A4C and GA3C, we ran the baseline GA3C code on various games on our machine with a single P-100 GPU.

We ran each experiment for 3 times on each game and plotted the average scores.

To test the robustness of approach, we experimented with various hyperparameter values like minimum training batch size, max norm of gradient (whether to clip gradients; if so MaxNorm=40 by default), learning rate and even the time instant where switching begins.

We noticed that our approach is better than baseline on all the settings.

Nevertheless, the plots we present are for the optimal setting (MinTrainingBatchSize=40, GradientClipping=False, LearningRate=0.003).

These values are also suggested to be optimal in the GA3C code.

Note that we compare results using the same setting for all 3 variants of A4C and also for the baseline GA3C.

FIG0 shows the comparison of three variants of A4C updates against GA3C for four games.

Note that the baseline GA3C plots(in red) are very similar to the ones reported in the original paper.

We notice that the Independent Updates(IU) performs significantly better than GA3C on all occasions except Qbert, where it is very similar GA3C.

In particular, IU achieves a score of 4300 on BeamRider game which is way better than the best result mentioned in GA3C paper.

IU crosses 3000 score in just 12.5 hrs while it takes 21 hrs for GA3C to achieve the same score.

IU also achieves a score of > 750 on SpaceInvaders game where the best result in GA3C paper achieves < 600.

At the same time, the Dependent Updates(DU) method (in blue) starts to rise faster than GA3C but doesn't sustain the growth after sometime owing to reasons mentioned in Section 3.2.3.

The only case where DU maintains the growth is Pong.

The hybrid switching method(Sw) performs remarkably well consistently on all the games, achieving higher scores than the best of GA3C.

For example, on Qbert game, the hybrid Sw method achieves a score of 12000 in just 7 hrs.

The best result mentioned in original GA3C paper achieves similar score in 20 hrs.

The other re-runs of Qbert in GA3C paper stall at a score of 8000.

Sw outperforms GA3C on other games as well, but it is still behind IU on BeamRider and SpaceInvaders games.

In all, we notice that Switching from DU to IU after few hours is the most robust method while IU alone is good on 2 games.

We propose a simple yet effective technique of adding anticipatory actions to the state-of-the-art GA3C method for reinforcement learning and achieve significant improvements in convergence and overall scores on several popular Atari-2600 games.

We also identify issues that challenge the sustainability of our approach and propose simple workarounds to leverage most of the information from higher-order action space.

There is scope for even higher order actions.

However, the action space grows exponentially with the order of anticipation.

Addressing large action space, therefore, remains a pressing concern for future work.

We believe human behavior information will help us select the best higher order actions.

<|TLDR|>

@highlight

Anticipation improves convergence of deep reinforcement learning.
We propose Episodic Backward Update - a new algorithm to boost the performance of a deep reinforcement learning agent by fast reward propagation.

In contrast to the conventional use of the replay memory with uniform random sampling, our agent samples a whole episode and successively propagates the value of a state into its previous states.

Our computationally efficient recursive algorithm allows sparse and delayed rewards to propagate effectively throughout the sampled episode.

We evaluate our algorithm on 2D MNIST Maze Environment and 49 games of the Atari 2600 Environment and show that our agent improves sample efficiency with a competitive computational cost.

Recently, deep reinforcement learning (RL) has been very successful in many complex environments such as the Arcade Learning Environment (Bellemare et al., 2013) and Go .

Deep Q-Network (DQN) algorithm BID11 with the help of experience replay BID8 BID9 enjoys more stable and sample-efficient learning process, so is able to achieve super-human performance on many tasks.

Unlike simple online reinforcement learning, the use of experience replay with random sampling breaks down the strong ties between correlated transitions and also allows the transitions to be reused multiple times throughout the training process.

Although DQN has shown impressive performances, it is still impractical in terms of data efficiency.

To achieve a human-level performance in the Arcade Learning Environment, DQN requires 200 million frames of experience for training, which is approximately 39 days of game play in real time.

Remind that it usually takes no more than a couple of hours for a skilled human player to get used to such games.

So we notice that there is still a tremendous amount of gap between the learning process of humans and that of a deep reinforcement learning agent.

This problem is even more crucial in environments such as autonomous driving, where we cannot risk many trials and errors due to the high cost of samples.

One of the reasons why the DQN agent suffers from such low sample efficiency could be the sampling method over the replay memory.

In many practical problems, the agent observes sparse and delayed reward signals.

There are two problems when we sample one-step transitions uniformly at random from the replay memory.

First, we have a very low chance of sampling the transitions with rewards for its sparsity.

The transitions with rewards should always be updated, otherwise the agent cannot figure out which action maximizes its expected return in such situations.

Second, there is no point in updating a one-step transition if the future transitions have not been updated yet.

Without the future reward signals propagated, the sampled transition will always be trained to return a zero value.

In this work, we propose Episodic Backward Update (EBU) to come up with solutions for such problems.

Our idea originates from a naive human strategy to solve such RL problems.

When we observe an event, we scan through our memory and seek for another event that has led to the former one.

Such episodic control method is how humans normally recognize the cause and effect relationship BID7 .

We can take a similar approach to train an RL agent.

We can solve the first problem above by sampling transitions in an episodic manner.

Then, we can be assured that at least one transition with non-zero reward is being updated.

We can solve the second problem by updating transitions in a backward way in which the transitions were made.

By then, we can perform an efficient reward propagation without any meaningless updates.

This method faithfully follows the principle of dynamic programing.

We evaluate our update algorithm on 2D MNIST Maze Environment and the Arcade Learning Environment.

We observe that our algorithm outperforms other baselines in many of the environments with a notable amount of performance boosts.

Reinforcement learning deals with environments where an agent can make a sequence of actions and receive corresponding reward signals, such as Markov decision processes (MDPs).

At time t, the agent encounters a state s t and takes an action a t ∈ A, observes the next state s t+1 and receives reward r t ∈ R. The agent's goal is to set up a policy π to take a sequence of actions so that the agent can maximize its expected return, which is the expected value of the discounted sum of rewards DISPLAYFORM0 Q-learning BID19 )

is one of the most widely used methods to solve such RL tasks.

The key idea of Q-learning is to estimate the state-action value function Q(s, a), generally called as the Qfunction.

The state-action value function may be characterized by the Bellman optimality equation DISPLAYFORM1 Given the state-action value function Q(s, a), the agent may perform the best action a * = argmax a Q(s, a) at each time step to maximize the expected return.

There are two major inefficiencies in the traditional Q-learning.

First, each experience is used only once to update the Q-network.

Secondly, learning from experiences in a chronologically forward order is much more inefficient than learning in a chronologically backward order.

Experience replay BID8 BID9 is proposed to overcome these inefficiencies.

After observing a transition (s t , a t , r t , s t+1 ), the agent stores the transition into its replay buffer.

In order to learn the Q-values, the agent samples transitions from the replay in a backward order.

In practice, the state space is extremely large, so it is impractical to tabularize Q-values of all stateaction pairs.

Deep Q-Network BID11 overcomes this issue by using deep neural networks to approximate the Q-function.

Deep Q-Network (DQN) takes a 2D representation of a state s t as an input.

Then the information of the state s t passes through a number of convolutional neural networks (CNNs) and fully connected networks.

Then it finally returns the Q-values of each action a t at state s t .

DQN adopts experience replay to use each transition in multiple updates.

Since DQN uses a function approximator, consecutive states output similar Q-values.

So when DQN updates transitions in a chronologically backward order, often errors cumulate and degrade the performance.

So DQN does not sample transitions in a backward order, but uniformly at random to train the network.

This process breaks down the correlations between consecutive transitions and reduces the variance of updates.

There have been a variety of methods proposed to improve the performance of DQN in terms of stability, sample efficiency and runtime.

Some methods propose new network architectures.

The dueling network architecture (Wang et al., 2015) contains two streams of separate Q-networks to estimate the value functions and the advantage functions.

Neural episodic control BID13 and model free episodic control BID3 use episodic memory modules to estimate the state-action values.

Some methods tackle the uniform random sampling replay strategy of DQN.

Prioritized experience replay assigns non-uniform probability to sample transitions, where greater probability is assigned for transitions with higher temporal difference error.

Inspired by Lin's backward use of replay memory, some methods try to aggregate TD values with Monte-Carlo returns.

Q(λ) and Retrace(λ) modify the target values to allow the on-policy samples to be used interchangeably for on-policy and off-policy learning, which ensures safe and efficient reward propagation.

Count-based exploration method combined with intrinsic motivation takes a mixture of one-step return and Monte-Carlo return to set up the target value.

Optimality Tightening BID5 Q(s t , a t ) ← r t + γ max a Q(s t+1 , a ) 5: end for plies constraints on the target using the values of several neighboring transitions.

Simply by adding a few penalty terms into the loss, it efficiently propagates reliable values to achieve faster convergence.

Our work lies on the same line of research.

Without a single change done on the network structure of the original DQN, we only modify the target value.

Instead of using a limited number of consecutive transitions, our method samples a whole episode from the replay memory and propagates values sequentially throughout the entire sampled episode in a backward way.

Our novel algorithm effectively reduces the errors generated from the consecutive updates of correlated states by a temporary backward Q-table with a diffusion coefficient.

We start with a simple motivating toy example to describe the effectiveness of episodic backward update.

Then we generalize the idea into deep learning architectures and propose the full algorithm.

Let us imagine a simple graph environment with a sparse reward FIG0 .

In this example, s 1 is the initial state and s 4 is the terminal state.

A reward of 1 is gained only when the agent moves to the terminal state and a reward of 0 is gained from any other transitions.

To make it simple, assume that we only have one episode stored in the experience memory: (s 1 → s 2 → s 1 → s 3 → s 4 ).

When sampling transitions uniformly at random as Nature DQN, the important transitions (s 1 → s 3 ) and (s 3 → s 4 ) may not be sampled for updates.

Even when those transitions are sampled, there is no guarantee that the update of the transition (s 3 → s 4 ) would be done before the update of (s 1 → s 3 ).

So by updating all transitions within the episode in a backward manner, we can speed up the reward propagation, and due to the recursive update, it is also computationally efficient.

We can calculate the probability of learning the optimal path (s 1 → s 3 → s 4 ) for the number of sample transitions trained.

With the simple episodic backward update stated in Algorithm 1 (which is a special case of Lin's algorithm BID8 with recency parameter λ = 0), the agent can come up with the optimal policy just after 4 updates of Q-values.

However, we see that the uniform sampling method requires more than 30 transitions to learn the optimal path FIG0 .Note that this method is different to the standard n-step Q-learning BID19 ).

DISPLAYFORM0 In n-step Q-learning, the number of future steps for target generation is fixed as n. However, our method takes T future values in consideration, which is the length of the sampled episode.

Also, n-step Q-learning takes max operator at the n-th step only, whereas we take max operator at every iterative backward steps which can propagate high values faster.

To avoid exponential decay of the Q-value, we set the learning rate α = 1 within the single episode update.

The fundamental idea of tabular version of backward update algorithm may be applied to its deep version with just a few modifications.

We use a function approximator to estimate the Q-values and generate a temporary Q-tableQ of the sampled episode for the recursive backward update.

The full algorithm is introduced in Algorithm 2.

The algorithm is almost the same as that of Nature DQN BID11 .

Our contributions are the episodic sampling method and the recursive backward With probability select a random action a t

Otherwise select a t = argmax a Q (s t , a; θ) DISPLAYFORM0 Execute action a t , observe reward r t and next state s t+1 9: DISPLAYFORM1 Sample a random episode E = {S, A, R, S } from D, set T = length(E)

Generate temporary target Q table,Q =Q S , ·; θ DISPLAYFORM0 Initialize target vector y = zeros(T ) 13: DISPLAYFORM1 DISPLAYFORM2 Perform a gradient descent step on (y − Q (S, A; θ)) 2 with respect to θ

Every C steps resetQ = Q 20:end for 21: end for target generation with a diffusion coefficient β (line number 10 to line number 17 of Algorithm 2), which prevents the errors from correlated states cumulating.

Our algorithm has its novelty starting from the sampling stage.

Instead of sampling transitions at uniformly random, we make use of all transitions within the sampled episode E = {S, A, R, S }.

Let the sampled episode start with a state S 1 and contain T transitions.

Then E can be denoted as a set of 1 × n vectors, i.e. S = {S 1 , S 2 , . . .

DISPLAYFORM0 The temporary target Q-tableQ, is initialized to store all the target Q-values of S for all valid actions.

Q is an |A| × T matrix which stores the target Q-values of all states S for all valid actions.

Therefore the j-th column ofQ is a column vector that containŝ Q S j+1 , a; θ − for all valid actions a from j = 1 to T .Our goal is to estimate the target vector y and train the network to minimize the loss between each Q (S j , A j ) and y j for all j from 1 to T .

After initialization of the temporary Q-table, we perform a recursive backward update.

Adopting the backward update idea, one elementQ [A k+1 , k] in the k-th column of theQ is replaced by the next transition's target y k+1 .

Then y k is estimated using the maximum of the newly modified k-th column ofQ. Repeating this procedure in a recursive manner until the start of the episode, we can successfully apply the backward update algorithm in a deep Q-network.

The process is described in detail with a supplementary diagram in Appendix D.When β = 1, the proposed algorithm is identical to the tabular backward algorithm stated in Algorithm 1.

But unlike the tabular situation, now we are using a function approximator and updating correlated states in a sequence.

As a result, we observe unreliable values with errors being propagated and compounded through recursive max operations.

We solve this problem by introducing the diffusion coefficient β.

By setting β ∈ (0, 1), we can take a weighted sum of the newly learnt value and the pre-existing value.

This process stabilizes the learning process by exponentially decreasing the error terms and preventing the compounded error from propagating.

Note that when β = 0, the algorithm is identical to episodic one-step DQN.We prove that episodic backward update with a diffusion coefficient β ∈ (0, 1) defines a contraction operator and converges to optimal Q-function in finite and deterministic MDPs.

Theorem 1.

Given a finite, deterministic, and tabular MDP M = (S, A, P, R), the episodic backward update algorithm in Algorithm 2 converges to the optimal Q function w.p.

1 as long as•

The step size satisfies the Robbins-Monro condition;• The sample trajectories are finite in lengths l: DISPLAYFORM1 • Every (state, action) pair is visited infinitely often.

We state the proof of Theorem 1 in Appendix E.We train the network to minimize the squared-loss between the Q-values of sampled states Q (S, A; θ) and the backward target y.

In general, the length of an episode is much longer than the minibatch size.

So we divide the loss vector y − Q (S, A; θ) into segments with size equal to the minibatch size.

At each step, the network is trained by a single segment.

A new episode is sampled only when all of the loss segments are used for training.

Our experiments are designed to verify the following two hypotheses: 1) EBU agent can propagate reward signals fast and efficiently in environments with sparse and delayed reward signals.

2) EBU algorithm is sample-efficient in complex domains and does not suffer from stability issues despite its sequential updates of correlated states.

To investigate these hypotheses, we performed experiments on 2D MNIST Maze Environment and on 49 games of the Arcade Learning Environment (Bellemare et al., 2013) .

We test our algorithm in 2D maze environment with sparse and delayed rewards.

Starting from the initial position, the agent has to navigate through the maze to discover the goal position.

The agent has 4 valid actions: up, down, left and right.

When the agent bumps into a wall, then the agent returns to its previous state.

To show effectiveness in complex domains, we use the MNIST dataset BID6 for state representation (illustrated in Figure 2 ).

When the agent arrives at each state, it receives the coordinates of the position in two MNIST images as the state representation.

We compare the performance of EBU to uniform one-step Q-learning and n-step Q-learning.

For n-step Q-learning, we set the value of n as the length of the episode.

We use 10 by 10 mazes with randomly placed walls.

The agent starts at (0,0) and has to reach the goal position at (9,9) as soon as possible.

Wall density indicates the probability of having a wall at each position.

We assign a reward of 1000 for reaching the goal and a reward of -1 for bumping into a wall.

For each wall density, we generate 50 random mazes with different wall locations.

We train a total of 50 independent agents, one agent for one maze over 200,000 steps each.

The MNIST images for state representation are randomly selected every time the agent visits each state.

The relative length is defined as l rel = l agent /l oracle , which is the ratio between the length of the agent's path l agent and the length of the ground truth shortest path l oracle .

FIG1 shows the median relative lengths of 50 agents over 200,000 training steps.

Since all three algorithms achieve median relative lengths of 1 at the end of training, we report the mean and the median relative lengths at 100,000 steps in TAB1 .

For this example, we set the diffusion coefficient β = 1.

The details of hyperparameters and the network structure are described in Appendix C.The result shows that EBU agent outperforms other baselines in most of the situations.

Uniform sampling DQN shows the worst performance in all configurations, implying the inefficiency of uniform sampling update in environments with sparse and delayed rewards.

As the wall density increases, valid paths to the goal become more complicated.

In other words, the oracle length l oracle increases, so it is important for the agent to make correct decisions at bottleneck positions.

N-step Q-learning shows the best performance with a low wall density, but as the wall density increases, EBU shows better performance than n-step Q. Especially when the wall density is 50%, EBU finds paths twice shorter than those of n-step Q. This performance gap originates from the difference between the target generation methods of the two algorithms.

EBU performs recursive max operators at each positions, so the optimal Q-values at bottlenecks are learned faster.

The Arcade Learning Environment (Bellemare et al., 2013 ) is one of the most popular RL benchmarks for its diverse set of challenging tasks.

The agent receives high-dimensional raw observations from exponentially large state space.

Even more, observations and objectives of the games are completely different over different games, so the strategies to achieve high score should also vary from game to game.

Therefore it is very hard to create a robust agent with a single set of networks and parameters that can learn to play all games.

We use the same set of 49 Atari 2600 games which was evaluated in Nature DQN paper BID11 .We compare our algorithm to four baselines: Nature DQN, Optimality Tightening BID5 , Prioritized Experience Replay and Retrace(λ) .

We train EBU and baselines agents for 10 million frames on 49 Atari games with the same network structure, hyperparameters and evaluation methods used by Nature DQN.

We divide the training steps into 40 epochs of 250,000 frames.

At the end of each epoch, we evaluate the agent for 30 episodes using -greedy policy with = 0.05.

Transitions of the Arcade Learning Environment are fully deterministic.

In order to give diversity in experience, both train and test episodes start with at most 30 no-op actions.

We train each game for 8 times with different random seeds.

For each agent with a different random seed, the best evaluation score during training is taken as its result.

Then we report the mean score of the 8 agents as the result of the game.

Detailed specifications for each baseline are described in Appendix C.We observe that the choice of β = 1 degrades the performance in most of the games.

Instead, we use β = 1 2 , which shows the best performance among { First, we show the improvements of EBU over Nature DQN for all 49 games in FIG2 .

To compare the performance of an agent over its baseline, we use the following measure (Wang et al., 2015)

.Score Agent − Score Baseline max{Score Human , Score Baseline } − Score Random .This measure shows how well the agent performs the task compared to its level of difficulty.

Out of the 49 games, our agent shows better performance in 32 games.

Not only that, for games such as 'Atlantis', 'Breakout' and 'Video Pinball', our agent shows significant amount of improvements.

In order to compare the overall performance of an algorithm, we use Eq.(4) to calculate the human normalized score (van Hasselt et al., 2015) .

DISPLAYFORM0 We report the mean and median of the human normalized scores of 49 games in TAB3 .

The result shows that our algorithm outperforms the baselines in both mean and median of the human normalized scores.

Furthermore, our method requires only about 37% of computation time used by Optimality Tightening 1 .

Since Optimality Tightening has to calculate the Q-values of neighboring states and compare them to generate the penalty term, it requires about 3 times more training time than Nature DQN.

Since EBU performs iterative episodic updates using the temporary Q-table that is shared by all transitions in the episode, its computational cost is almost the same as that of Nature DQN.

Scores for each game after 10 million frames of training are summarized in Appendix A.We show the performance of EBU and the baselines for 4 games 'Assault', 'Breakout', 'Gopher' and 'Video Pinball' in FIG4 .

EBU with a diffusion coefficient β = 0.5 shows competitive performances in all 4 games, reflecting that our algorithm does not suffer from the stability issue caused by the sequential update of correlated states.

Other baselines fail in some games, whereas our algorithm shows stable learning processes throughout all games.

Out of 49 games, our algorithm shows the worst performance in only 6 games.

Such stability leads to the best median and mean scores in total.

Note that naive backward algorithm with β = 1.0 fails in most games.

Each state is given as a grey scale 28 × 28 image.

We apply 2 convolutional neural networks (CNNs) and one fully connected layer to get the output Q-values for 4 actions: up, down, left and right.

The first CNN uses 64 channels with 4 × 4 kernels and stride of 3.

The next CNN uses 64 channels with 3 × 3 kernels and stride of 1.

Then the layer is fully connected into size of 512.

Then we fully connect the layer into size of the action space 4.

After each layer, we apply rectified linear unit.

We train the agent for a total of 200,000 steps.

The agent performs -greedy exploration.

starts from 1 and is annealed to 0 at 200,000 steps in a quadratic manner: = 1 (200,000) 2 (step −200, 000) 2 .

We use RMSProp optimizer with a learning rate of 0.001.

The online-network is updated every 50 steps, the target network is updated every 2000 steps.

The replay memory size is 30000 and we use minibatch size of 350.

We use a discount factor γ = 0.9 and a diffusion coefficient β = 1.0.

The agent plays the game until it reaches the goal or it stays in the maze for more than 1000 time steps.

Almost all specifications such as hyperparameters and network structures are identical for all baselines.

We use exactly the same network structure and hyperparameters of Nature DQN BID11 .

The raw observation is preprocessed into gray scale image of 84 × 84.

Then it passes through three convolutional layers: 32 channels with 8 × 8 kernels with stride of 4; 64 channels with 4 × 4 kernels with stride of 2; 64 channels with 3 × 3 kernels with stride of 1.

Then it is fully connected into size of 512.

Then it is again fully connected into the size of the action space.

We train agents for 10 million frames each, which is equivalent to 2.5 million steps with frame skip of 4.

The agent performs -greedy exploration.

starts from 1 and is linearly annealed to reach the final value 0.1 at 4 million frames of training.

To give randomness in experience, we select a number k from 1 to 30 uniform randomly at the start of each train and test episode.

We start the episode with k no-op actions.

The network is trained by RMSProp optimizer with a learning rate of 0.00025.

At each step, we update transitions in minibatch with size 32.

The replay buffer size is 1 million steps (4 million frames).

The target network is updated every 10,000 steps.

The discount factor is γ = 0.99.We divide the training process into 40 epochs of 250,000 frames each.

At the end of each epoch, the agent is tested for 30 episodes with = 0.05.

The agent plays the game until it runs out of lives or time (18,000 frames, 5 minutes in real time).Below are detailed specifications for each algorithm.

We set the diffusion coefficient β = 0.5.

To generate the lower and upper bounds, we use 4 future transitions and 4 past transitions.

As described in the paper , we use the rank-based DQN version with hyperparameters α = 0.5, β = 0.

Just as EBU, we sample a random episode and then generate the Retrace target for the transitions in the sampled episode.

First, we calculate the trace coefficients from s = 1 to s = T (terminal).

DISPLAYFORM0 Where µ is the behavior policy of the sampled transition and the evaluation policy π is the current policy.

Then we generate a loss vector for transitions in the sample episode from t = T to t = 1.

DISPLAYFORM1 APPENDIX D SUPPLEMENTARY FIGURE: BACKWARD UPDATE ALGORITHM DISPLAYFORM2 DISPLAYFORM3 ) ( ,) 0 DISPLAYFORM4 DISPLAYFORM5 ) ( ,) 0 DISPLAYFORM6 0 DISPLAYFORM7 DISPLAYFORM8 ) ( ,) 0 DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 ) 0 Note that is the target Q-value and +1 , : = 0.

DISPLAYFORM12 Line # 14~17, first iteration (k = T-1): Update ෩ and .

Let the T-th action in the replay memory be = (2) .

DISPLAYFORM13 DISPLAYFORM14 DISPLAYFORM15 ) ( ,) 0 DISPLAYFORM16 DISPLAYFORM17 DISPLAYFORM18 ) 0 Now, we will prove that the episodic backward update algorithm converges to the true action-value function Q * in the case of finite and deterministic environment.

DISPLAYFORM19 DISPLAYFORM20 In the episodic backward update algorithm, a single (state, action) pair can be updated through multiple episodes, where the evaluated targets of each episode can be different from each other.

Therefore, unlike the bellman operator, episodic backward operator depends on the exploration policy for the MDP.

Therefore, instead of expressing different policies in each state, we define a schedule to represent the frequency of every distinct episode (which terminates or continues indefinitely) starting from the target (state, action) pair.

Assume a MDP M = (S, A, P, R) , where R is a bounded function.

Then, for each state (s, a) ∈ S × A and j ∈ [1, ∞], we define j-length path set p s,a (j) and path set p(s, a) for (s, a) as DISPLAYFORM0 and p s,a = ∪ ∞ j=1 p s,a (j).

Also, we define a schedule set λ s,a for (state action) pair (s, a) as DISPLAYFORM1 Finally, to express the varying schedule in time at the RL scenario, we define a time schedule set λ for MDP M as DISPLAYFORM2 Since no element of the path can be the prefix of the others, the path set corresponds to the enumeration of all possible episodes starting from each (state, action) pair.

Therefore, if we utilize multiple episodes from any given policy, we can see the empirical frequency for each path in the path set belongs to the schedule set.

Finally, since the exploration policy can vary across time, we can group independent schedules into the time schedule set.

For a given time schedule and MDP, now we define the episodic backward operator.

= E s ∈S,P (s |s,a) DISPLAYFORM3 Where (p s,a ) i is the i-th path of the path set, and (s ij , a ij ) corresponds to the j-th (state, action) pair of the i-th path.

Episodic backward operator consists of two parts.

First, given the path that initiates from the target (state, action) pair, function T β,Q (ps,a)i computes the maximum return of the path via backward update.

Then, the return is averaged by every path in the path set.

Now, if the MDP M is deterministic, we can prove that the episodic backward operator is a contraction in the sup-norm, and the fixed point of the episodic backward operator is the optimal action-value function of the MDP regardless of the time schedule.

Theorem 2. (Contraction of episodic backward operator and the fixed point) Suppose M = (S, A, P, R) is a deterministic MDP.

Then, for any time schedule {λ s,a (t)} ∞ t=1,(s,a)∈S×A ∈ λ, H β t is a contraction in the sup-norm for any t, i.e DISPLAYFORM4 Furthermore, for any time schedule {λ s,a (t)} ∞ t=1,(s,a)∈S×A ∈ λ, the fixed point of H β t is the optimal Q function Q * .Proof.

First, we prove T β,Q (ps,a)i (j) is a contraction in the sup-norm for all j.

Since M is a deterministic MDP, we can reduce the return as DISPLAYFORM5 Since this occurs for any arbitrary path, the only remaining case is when DISPLAYFORM6 Now, let's speculate on the path s 0 , s 1 , s 2 , ...., s |(ps,a)i)| .

Let's first prove the contradiction when the length of the contradictory path is finite.

If Q * (s i1 , a i1 ) < γ −1 (Q * (s, a) − r(s, a)), then by the bellman equation, there exists action a = a i1 s.t Q * (s i1 , a) = γ −1 (Q * (s, a) − r(s, a)).

Then, we can find that T β,Q * (ps,a)1 (1) = γ −1 (Q * (s, a) − r(s, a)) so it contradicts the assumption.

Therefore, a i1should be the optimal action in s i1 .Repeating the procedure, we can find that a i1 , a i2 , ..., a |(ps,a)i)|−1 are optimal with respect to their corresponding states.

Finally, we can find that T β,Q * (ps,a)1 (|(p s,a ) i )|) = γ −1 (Q * (s, a) − r(s, a)) since all the actions satisfies the optimality condition of the inequality in equation 7.

Therefore, it is a contradiction to the assumption.

In the case of infinite path, we will prove that for any > 0, there is no path that satisfy Since the reward function is bounded, we can define r max as the supremum norm of the reward function.

Define q max = max s,a |Q(s, a)| and R max = max{r max , q max }.

We can assume R max > 0.

Then, let's set n = log γ (1−γ) Rmax + 1.

Since γ ∈ [0, 1), R max γ n 1−γ < .

Therefore, by applying the procedure on the finite path case for 1 ≤ j ≤ n , we can conclude that the assumption leads to a contradiction.

Since the previous n trajectories are optimal, the rest trajectories can only generate a return less than .Finally, we proved that max 1≤j≤|(ps,a)i| T β,Q * (ps,a)i (j) = Q * (s,a)−r(s,a) γ ∀1 ≤ i ≤ |p s,a | and therefore, every episodic backward operator has Q * as the fixed point.

Finally, we will show that the online episodic backward update algorithm converges to the optimal Q function Q * .Restatement of Theorem 1.

Given a finite, deterministic, and tabular MDP M = (S, A, P, R), the episodic backward update algorithm, given by the update rule Q t+1 (s t , a t ) = (1 − α t )Q t (s t , a t ) + α t r(s t , a t ) + γ |ps t ,a t | i=1 (λ (st,at) ) i (t) max 1≤j≤|(ps t ,a t )i| T β,Q (ps t ,a t )i (j) converges to the optimal Q function w.p.

1 as long as•

The step size satisfies the Robbins-Monro condition;• The sample trajectories are finite in lengths l: E[l]

< ∞;• Every (state, action) pair is visited infinitely often.

For the proof of Theorem 1, we follow the proof of BID10 .

Lemma 1.

The random process ∆ t taking values in R n and defined as ∆ t+1 (x) = (1 − α t (x))∆ t (x) + α t (x)F t (x)converges to zero w.p.1 under the following assumptions:• 0 ≤ α t ≤ 1, t α t (x) = ∞ and t α 2 t (x) < ∞; • E [F t (x)|F t ]

W ≤ γ ∆ t W , with γ < 1;• var [F t (x)|F t ] ≤ C 1 + ∆ t 2 W , for C > 0.By Lemma 1, we can prove that the online episodic backward update algorithm converges to the optimal Q * .Proof.

First, by assumption, the first condition of Lemma 1 is satisfied.

Also, we can see that by substituting ∆ t (s, a) = Q t (s, a) − Q * (s, a), and F t (s, a) = r(s, a) + γ Since the reward function is bounded, the third condition also holds as well.

Finally, by Lemma 1, Q t converges to Q * .Although the episodic backward operator can accommodate infinite paths, the operator can be practical when the maximum length of the episode is finite.

This assumption holds for many RL domains, such as ALE.

<|TLDR|>

@highlight

We propose Episodic Backward Update, a novel deep reinforcement learning algorithm which samples transitions episode by episode and updates values recursively in a backward manner to achieve fast and stable learning.

@highlight

Proposes a new DQN where the targets are computed on a full episode by a backward update (end to start) for faster propagation of rewards by the episode end.

@highlight

The authors propose to modify the DQN algorithm by applying the max Bellman operator recursively on a trajectory with some decay to prevent accumulating errors with the nested max.

@highlight

In deep-Q networks, update Q values starting from the end of the episode in order to facilitate quick propagation of rewards back along the episode.
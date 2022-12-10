We propose Bayesian Deep Q-Network  (BDQN), a  practical Thompson sampling based Reinforcement Learning (RL) Algorithm.

Thompson sampling allows for targeted exploration in high dimensions through posterior sampling but is usually computationally expensive.

We address this limitation by introducing uncertainty only at the output layer of the network through a Bayesian Linear Regression (BLR) model, which can be trained with fast closed-form updates and its samples can be drawn efficiently through the Gaussian distribution.

We apply our method to a wide range of Atari Arcade Learning Environments.

Since BDQN carries out more efficient exploration, it is able to reach higher rewards substantially faster than a key baseline, DDQN.

Designing algorithms that achieve an optimal trade-off between exploration and exploitation is one of the primary goal of reinforcement learning (RL).

However, targeted exploration in high dimensional spaces is a challenging problem in RL.

Recent advances in deep RL mostly deploy simple exploration strategies such as ε-greedy, where the agent chooses the optimistic action, the action with highest promising return, with probability (1 − ε), otherwise, uniformly at random picks one of the available actions.

Due to this uniform sampling, the ε-greedy method scales poorly with the dimensionality of state and action spaces.

Recent work has considered scaling exploration strategies to large domains BID12 .

Several of these papers have focused on employing optimism-under-uncertainty approaches, which essentially rely on computing confidence bounds over different actions, and acting optimistically with respect to that uncertainty.

An alternative to optimism-under-uncertainty (Brafman & Tennenholtz, 2003) is Thompson Sampling (TS) BID57 , one of the oldest heuristics for multi arm bandits.

TS is a Bayesian approach where one starts with a prior distribution over the belief and compute the posterior beliefs based on the collected data through the interaction with the environment and then maximizes the expected return under the sampled belief.

The TS based posterior sampling can provide more targeted exploration since it can trade off uncertainty with the expected return of actions.

In contrast, the ε-greedy strategy is indifferent to uncertainty of the actions and the expected rewards of sub-optimistic ones (set of actions excluding the optimistic action).There has been relatively little work on scaling Thompson Sampling to large state spaces.

The primary difficulty in implementing Thompson sampling is the difficulty of sampling from general posterior distributions.

Prior efforts in this space have generally required extremely expensive computations (e.g. BID21 BID52 )We derive a practical Thompson sampling framework, termed as Bayesian deep Q-networks (BDQN), where we approximate the posterior distribution on the set of Q-functions and sample from this approximated posterior.

BDQN is computationally efficient since it incorporates uncertainty only at the output layer, in the form of a Bayesian linear regression model.

Due to linearity and by choosing a Gaussian prior, we derive a closed-form analytical update to the approximated posterior distribution over Q functions.

We can also draw samples efficiently from the Gaussian distribution.

As addressed in BID32 , one of the major benefits of function approximation methods in deep RL is that the estimation of the Q-value, given a state-action pair, can generalize well to other state-action pairs, even if they are visited rarely.

We expect this to hold in BDQN as well, but additionally, we also expect the uncertainty of state-action pairs to generalize well.

We test BDQN on a wide range of Arcade Learning Environment Atari games BID13 BID15 ) against a strong baseline DDQN BID58 .

Aside from simplicity and popularity of DDQN, BDQN and DDQN share the same architecture, and follow same target objective.

These are the main reasons we choose DDQN for our comparisons.

In table.

1 we see significant gains for BDQN over DDQN.

BDQN is able to learn significantly faster and reach higher returns due to more efficient exploration.

The evidence of this is further seen from the fact that we are able to train BDQN with much higher learning rates compared to DDQN.

This suggests that BDQN is able to learn faster and reach better scores.

These promising results suggest that BDQN can further benefit from additional modifications that were done to DQN, e.g. Prioritized Experience Replay , Dueling approach , A3C BID33 , safe exploration BID30 , and etc.

This is because BDQN only changes that exploration strategy of DQN, and can easily accommodate additional improvements to DQN.

Step TAB3 : The first column presents the score ratio after last column steps.

The second column is the score ratio of BDQN after the number of steps in the last column compared to the score of DDQN † , the reported scores of DDQN in BID58 after running for 200M samples during evaluation time, and the third column is with respect to Human score reported at BID32 .

The complexity of the exploration-exploitation trade-off has been vastly investigated in RL literature BID26 BID14 BID51 BID10 BID18 .

BID6 addresses this question for multi-armed bandit problems where regret guarantees are provided.

BID23 investigates the regret analyses in MDPs where exploration happens through the optimal policy of optimistic model, a well-known Optimism in Face of Uncertainty (OFU) principle where a high probability regret upper bound is guaranteed.

BID8 deploys OFU in order to propose the high probability regret upper bound for Partially Observable MDPs (POMDPs) and finally, BID11 tackles a general case of partial monitoring games and provides minimax regret guarantee which is polynomial in certain dimensions of the problem.

In multi arm bandit, there is compelling empirical evidence that Thompson Sampling can provide better results than optimism-under-uncertainty approaches BID16 , while the state of the art performance bounds are preserved BID43 BID3 BID22 BID1 .

A natural adaptation of this algorithm to RL, posterior sampling RL (PSRL), first proposed by BID52 also shown to have good frequentist and Bayesian performance guarantees BID37 BID22 BID4 BID0 BID41 BID56 .

Even though the theoretical RL addresses the exploration and exploitation trade-offs, these problems are still prominent in empirical reinforcement learning research BID32 BID24 BID2 BID9 .

On the empirical side, the recent success in the video games has sparked a flurry of research interest.

Following the success of Deep RL on Atari games BID32 and the board game Go BID49 , many researchers have begun exploring practical applications of deep reinforcement learning (DRL).

Some investigated applications include, robotics BID29 , energy management BID35 , and self-driving cars BID48 .

Among the mentioned literature, the prominent exploration strategy for Deep RL agent has been ε-greedy.

Inevitably for PSRL, the act of posterior sampling for policy or value is computationally intractable with large systems, so PSRL can not be easily leveraged to high dimensional problems.

To remedy these failings consider the use of randomized value functions to approximate posterior samples for the value function in a computationally efficient manner.

They show that with a suitable linear value function approximation, using the approximated Bayesian linear regression for randomized least-squares value iteration method can remain statistically efficient BID38 but still is not scalable to large scale RL with deep neural networks.

To combat these shortcomings, BID39 suggests a bootstrapped-ensemble approach that trains several models in parallel to approximate uncertainty.

Other works suggest using a variational approximation to the Q-networks (Lipton et al., 2016b) or noisy network BID19 .

However, most of these approaches significantly increase the computational cost of DQN and neither approaches produced much beyond modest gains on Atari games.

Applying Bayesian regression in last layer of neural network was investigated in BID50 for object recognition, image caption generation, and etc.

where a significant advantage has been provided.

In this work we present another alternative approach that extends randomized least-squares value iteration method BID38 to deep neural networks: we approximate the posterior by a Bayesian linear regression only on the last layer of the neural network.

This approach has several benefits, e.g. simplicity, robustness, targeted exploration, and most importantly, we find that this method is much more effective than any of these predecessors in terms of sample complexity and final performance.

Concurrently, O'Donoghue et al. FORMULA0 studies how to construct the frequentist confidence of the regression on the feature space of the neural network for RL problems by learning the uncertainties through a shallow network while BID28 , similar to BDQN, suggests running linear regression on the representation layer of the deep network in order to learn the state-action value.

Drop-out, as another randomized exploration method is proposed by BID20 but BID39 argues about its estimated uncertainty and hardness in driving a suitable exploitation of it.

It is worth noting that most of proposed methods are based on ε-exploration.

In this section, we enumerate a few benefits of TS over ε-greedy strategies.

We show how TS strategies exploit the uncertainties and expected returns to design a randomized exploration while ε−greedy strategies disregard all these useful information for the exploration.

FIG1 (a) expresses the agent's estimated values and uncertainties for the available actions at a given state x.

While ε−greedy strategy mostly focus on the action 1, the optimistic action, the TS based strategy randomizes, mostly, over actions 1 through 4, utilizing their approximated returns and uncertainties, and with low frequency tries actions 5, 6.

Not only the ε−greedy strategy explores actions 5 and 6, where the RL agent is almost sure about low return of these actions, as frequent as other sub-optimistic actions, but also spends the network capacity to accurately estimate their values.

A commonly used technique in deep RL is a moving window of replay buffer to store the recent experiences.

The TS based agent, after a few tries of actions 5 and 6 builds a belief in low return of these actions given the current target values.

Since the replay buffer is bounded moving window, lack of samples of these actions pushes the posterior belief of these actions to the prior, over time, and the agent tries them again in order to update its belief FIG1 .

In general, TS based strategy advances the exploration-exploitation trade-off by making trade-off between the expected returns and the uncertainties, while ε−greedy strategy ignores all of this information.

Another superiority of TS over ε-greedy can be described using FIG1 .

Consider an episodic maze-inspired deterministic game, with episode length H of shortest pass from the start to the destination.

The agent is placed to the start point at the beginning of each episode where the goal state is to reach the destination and receive a reward of 1 otherwise reward is 0.

Consider an agent, which is given a hypothesis set of Q-functions where the true Q-function is within the set and is the most optimistic function in the set.

In this situation, TS randomizes over the Q-functions with high promising returns and relatively high uncertainty, including the true Q-function.

When it picks the true Q-function, it increases the posterior probability of this Q-function because it matches the likelihood.

When TS chooses other functions, they predict deterministically wrong values and the posterior update of those functions set to zero, therefore, the agent will not choose them again, i.e. TS finds the true Q-function very fast.

For ε-greedy agent, even though it chooses the true function at the beginning (it is the optimistic one), at each time step, it randomizes its action with the probability ε.

Therefore, it takes exponentially many trials in order to get to the target in this game.

An infinite horizon γ-discounted MDP M is a tuple X , A, T, R , with state space X , action space A, and the transition kernel T , accompanied with reward function of R where 0 < γ ≤ 1.

At each time step t, the environment is at a state x t , called current state, where the agent needs to make a decision a t under its policy.

Given the current state and action, the environment stochastically proceed to a successor state x t+1 under probability distribution T (X t+1 |x t , a t ) := P(X t+1 |x t , a t ) and provides a stochastic reward r t with mean of E[r|x = x t , a = a t ] = R(x t , a t ).

The agent objective is to optimize the overall expected discounted reward over its policy π := X → A, a stochastic mapping from states to actions, π(a|x) := P(a|x).

DISPLAYFORM0 The expectation in Eq. 1 is with respect to the randomness in the distribution of initial state, transition probabilities, stochastic rewards, and policy, under stationary distribution, where η * , π * are optimal return and optimal policy, respectively.

Let Q π (x, a) denote the average discounted reward under policy π starting off from state x and taking action a in the first place.

DISPLAYFORM1 For a given policy π and Markovian assumption of the model, we can rewrite the equation for the Q functions as follows: DISPLAYFORM2 To find the optimal policy, one can solve the Linear Programing problem in Eq. 1 or follow the corresponding Bellman equation Eq. 2 where both of optimization methods turn to DISPLAYFORM3 where Q * (x, a) = Q π * (x, a) and the optimal policy is a deterministic mapping from state to actions in A, i.e. x → arg max a Q * (x, a).

In RL, we do not know the transition kernel and the reward function in advance, therefore, we can not solve the posed Bellman equation directly.

In order to tackle this problem, BID27 BID5 studies the property of minimizing the Bellman residual of a given Q-function DISPLAYFORM4 Where the tuple (x, a, r, x , a ) consists of consecutive samples under behavioral policy π.

Furthermore, BID32 carries the same idea, and introduce Deep Q-Network (DQN) where the Q-functions are parameterized by a DNN.

To improve the quality of Q function, they use back propagation on loss L(Q) using TD update BID53 .

In the following we describe the setting used in DDQN.

In order to reduce the bias of the estimator, they introduce target network Q target and target value y = r + γQ target (x ,â) whereâ = arg max a Q(x , a ) with a new loss DISPLAYFORM5 Minimizing this regression loss, and respectably its estimation L(Q, Q target ), matches the Q to the target y.

Once in a while, the algorithm sets Q target network to Q network, peruses the regression with the new target value, and provides an biased estimator of the target.

We propose a Bayesian method to approximate the Q-function and match it to the target value.

We utilize the DQN architecture, remove its last layer, and build a Bayesian linear regression (BLR) BID42 on the feature representation layer, φ θ (x) ∈ R d , parametrized by θ.

We deploy BLR to efficiently approximate the distribution over the Q-values where the uncertainty over the values is captured.

A common assumption in DNN is that the feature representation is suitable for linear classification or regression (same assumption in DQN).The Q-functions can be approximated as a linear combination of features, i.e. for a given pair of state-action, Q(x, a) = φ θ (x) w a .

Therefore, by deploying BLR, we can approximate the generative model of the Q-function using its corresponding target value: y = r + γφ target w targetâ , where φ target θ (x) ∈ R d denotes the feature representation of target network, for any (x, a) as follows DISPLAYFORM0 where ∼ N (0, σ 2 ) is an iid noise.

Furthermore, we consider w a ∈ R d for a ∈ A are drawn approximately from a Gaussian prior N (0, σ 2 ).

Therefore, y|x, a, w a ∼ N (φ(x) w a , σ 2 ).

Moreover, the distribution of the target value y is P (y| a) = wa P (y|w a ) P (w a ) dw a .

Given a dataset D = {x τ , a τ , y τ } D τ =1 , we construct |A| disjoint datasets for each action, D a , where D = ∪ a∈A D a and D a is a set of tuples x τ , a τ , y τ with the action a τ = a and size D a .

We are interested in P(w a |D a ) and P(Q(x, a)|D a ), ∀x ∈ X .

We construct a matrix Φ a ∈ R d×Da , a concatenation of feature column vectors {φ(x i )} Da i=1 , and y a ∈ R Da , a concatenation of target values in set D a .

Therefore the posterior distribution is as follows: Select action a t = argmax a W φ θ (x t ) a 10: DISPLAYFORM1 Execute action a t in environment, observing reward r t and successor state x t+1 11:Store transition (x t , a t , r t , x t+1 ) in replay buffer 12:Sample a random minibatch of transitions (x τ , a τ , r τ , x τ +1 ) from replay buffer 13: DISPLAYFORM2 14: end for 23: end for 1 ∈ R d is a identity matrix, and Q(x, a)|D a = w a φ(x).

Since the prior and likelihood are conjugate of each other we have the posterior distribution over the discounted return approximated as DISPLAYFORM3 DISPLAYFORM4 The expression in Eqs. 5 gives the posterior distribution over weights w a and the function Q. As TS suggests, for the exploration, we exploit the expression in Eq. 5.

For all the actions, we set w target a as the mean of posterior distribution over w a .

For each action, we sample a wight vector w a in order to have samples of mean Q-value.

Then we act optimally with respect to the sampled means DISPLAYFORM5 Let DISPLAYFORM6 , and Cov = {Ξ a } |A| a=1 .

In BDQN, the agent interacts with the environment through applying the actions proposed by TS, i.e. a TS .

We utilize a notion of experience replay buffer where the agent stores its recent experiences.

The agent draws W ∼ N (W target , Cov) (abbreviation for sampling of vector w for each action separately) every T sample steps and act optimally with respect to the drawn weights.

During the inner loop of the algorithm, we draw a minibatch of data from replay buffer and use loss DISPLAYFORM7 where DISPLAYFORM8 and update the weights of network: DISPLAYFORM9 We update the target network every T target steps and set θ target to θ.

With the period of T

the agent updates its posterior distribution using larger minibatch of data drawn from replay buffer and sample W with respect to the updated posterior.

Algorithm 1 gives the full description of BDQN.

We apply BDQN on a variety of games in the OpenAiGym 1 BID15 .

As a baseline 2 , we run DDQN algorithm and evaluate BDQN on the measures of sample complexity and score.

Network architecture: The input to the network part of BDQN is 4 × 84 × 84 tensor with a rescaled and averaged over channels of the last four observations.

The first convolution layer has 32 filters of size 8 with a stride of 4.

The second convolution layer has 64 filters of size 4 with stride 2.

The last convolution layer has 64 filters of size 3 followed by a fully connected layers with size 512.

We add a BLR layer on top of this.

For BDQN, we set the values of W target to the mean of the posterior distribution over the weights of BLR with covariances Cov and draw W from this posterior.

For the fixed W and W target , we randomly initialize the parameters of network part of BDQN, θ, and train it using RMSProp, with learning rate of 0.0025, and a momentum of 0.95, inspired by BID32 where the discount factor is γ = 0.99, the number of steps between target updates T target = 10k steps, and weights W are re-sampled from their posterior distribution every T sample step.

We update the network part of BDQN every 4 steps by uniformly at random sampling a mini-batch of size 32 samples from the replay buffer.

We update the posterior distribution of the weight set W every T Bayes target using mini-batch of size B (if size of replay buffer is less than B at the current step, we choose the minimum of these two ), with entries sampled uniformly form replay buffer.

The experience replay contains the 1M most recent transitions.

Further hyper-parameters, are equivalent to ones in DQN setting.

Hyper-parameters tunning: For the BLR, we have noise variance σ , variance of prior over weights σ, sample size B, posterior update period T Bayes target , and the posterior sampling period T sample .

To optimize for this set of hyper-parameters we set up a very simple, fast, and cheap hyper-parameter tunning procedure which proves the robustness of BDQN.

To fine the first three, we set up a simple hyper-parameter search.

We used a pretrained DQN model for the game of Assault, and removed the last fully connected layer in order to have access to its already trained feature representation.

Then we tried combination of B = {T target , 10 · T target }, σ = {1, 0.1, 0.001}, and σ = {1, 10} and test for 1000 episode of the game.

We set these parameters to their best B = 10 · T target , σ = 0.001, σ = 1.The above hyper-parameter tuning is cheap and fast since it requires only a few times B number of forward passes.

For the remaining parameter, we ran BDQN ( with weights randomly initialized) on the same game, Assault, for 5M time steps, with a set of DISPLAYFORM0 100 } where BDQN performed better with choice of T Bayes target = 10 · T target .

For both choices of T sample , it performed almost equal where we choose the higher one.

We started off with the learning rate of 0.0025 and did not tune for that.

Thanks to the efficient TS exploration and closed form BLR, BDQN can learn a better policy in even shorter period of time.

In contrast, it is well known for DQN based methods that changing the learning rate causes a major degradation in the performance, Apx.

A. The proposed hyper-parameter search is very simple where the exhaustive hyper-parameter search is likely to provide even better performance.

In order to compare the fairness in sample usage, we argue in Apx.

A, that the network part of BDQN and its corresponding part in DDQN observe the same number of samples but the BLR part of BDQN uses 16 times less samples compared to its corresponding last layer in DDQN, Apx.

A. All the implementations are coded in MXNet framework BID17 and are available at ..... .

The results are provided in FIG2 and Table.

2.

Mostly the focus of the experiments are on sample complexity in Deep-RL, even though, BDQN provides much larger scores compared to base line.

For example, for the game Atlantis, DDQN † gives score of 64.67k after 200M samples during evaluation time, while BDQN reaches 3.24M after 40M samples.

As it is been shown in FIG2 , BDQN saturates for Atlantis after 20M samples.

We realized that BDQN reaches the internal OpenAIGym limit of max_episode, where relaxing it improves score after 15M steps to 62M .We can observe that BDQN can immediately learn significantly better policies due to its targeted exploration in much shorter period of time.

Since BDQN on game Atlantis promise a big jump around time step 20M , we ran it five more times in order to make sure it was not just a coincidence.

We did the same additional five experiments for the game Amidar as well.

We observed that the improvements are consistent among the different runs.

For the game Pong, we ran the experiment for a longer period but just plotted the beginning of it in order to observe the difference.

For some games we did not run the experiment to 100M samples since the reached their plateau.

This suggests that BDQN can benefit even more from further modifications to DQN such as e.g. Prioritized Experience Replay , Dueling approach , A3C BID33 , safe exploration BID30 , and etc.

We plan to explore the benefit of these modifications up to small changes in the future.

We also plan to combine strategies that incorporate uncertainty over model parameters with BDQN.

In RL, policy gradient BID54 BID25 BID46 is another approach which directly learn the policy.

In practical policy gradient, even though the optimal policy, given Markovian assumption, needs to be deterministic, the policy regularization is a dominant approach to make the policy stochastic and preserve the exploration thorough the stochasticity of the policy BID34 BID47 .

We plan to explore the advantage of TS based exploration instead of regularizing the policy and make it stochastic.

Learning rate: It is well known that DQN and DDQN are sensitive to the learning rate and change of learning rate can degrade the performance to even worse than random policy.

We tried the same learning rate as BDQN, 0.0025, for DDQN and observed that its performance drops.

FIG3 shows that the DDQN with higher learning rates learns as good as BDQN at the very beginning but it can not maintain the rate of improvement and degrade even worse than the original DDQN.

Computational and sample cost comparison:

For a given period of game time, the number of the backward pass in both BDQN and DQN are the same where for BDQN it is cheaper since it has one layer (the last layer) less than DQN.

In the sense of fairness in sample usage, for example in duration of 10 · T Bayes target = 100k, all the layers of both BDQN and DQN, except the last layer, see the same number of samples, but the last layer of BDQN sees 16 times fewer samples compared to the last layer of DQN.

The last layer of DQN for a duration of 100k, observes 25k = 100k/4 (4 is back prob period) mini batches of size 32, which is 16 · 100k, where the last layer of BDQN just observes samples size of B = 100k.

As it is mentioned in Alg.

1, to update the posterior distribution, BDQN draws B samples from the replay buffer and needs to compute the feature vector of them.

This step of BDQN gives a superiority to DQN in the sense of speed which is almost 70% faster than BDQN (DQN, on average, for the update does full forward and backward passes while BDQN does not do backward path on the last layer but needs an extra forward pass in order to compute the feature representation).

One can easily relax this limitation by parallelizing this step with the main body of BDQN or deploying on-line posterior update methods.

Thompson sampling frequency: The choice of TS update frequency can be crucial from domain to domain.

If one chooses T sample too short, then computed gradient for backpropagation of the feature representation is not going to be useful since the gradient get noisier and the loss function is changing too frequently.

On the other hand, the network tries to find a feature representation which is suitable for a wide range of different weights of the last layer, results in improper use of model capacity.

If the TS update frequency is too low, then it is far from being TS and losses randomized exploration property.

The current choice of T sample is suitable for a variety of Atari games since the length of each episode is in range of O(T sample ) and is infrequent enough to make the feature representation robust to big changes.

For the RL problems with shorter horizon we suggest to introduce two more parameters,T sample andW whereT sample , the period that ofW is sampled our of posterior, is much smaller than T sample andW is being used just for making TS actions while W is used for backpropagation of feature representation.

For game Assault, we tried usingT sample andW but did not observe much a difference, and set them to T sample and W .

But for RL setting with a shorter horizon, we suggest using them.

Further investigation in Atlantis: After removing the maximum episode length limit for the game Atlantis, BDQN gets the score of 62M.

This episode is long enough to fill half of the replay buffer and make the model perfect for the later part of the game but losing the crafted skill for the beginning of the game.

We observe in FIG4 that after losing the game in a long episode, the agent forgets a bit of its skill and loses few games but wraps up immediately and gets to score of 30M .

To overcome this issue, one can expand the replay buffer size, stochastically store samples in the reply buffer where the later samples get stored with lowers chance, or train new models for the later parts of the episode.

There are many possible cures for this interesting observation and while we are comparing against DDQN, we do not want to advance BDQN structure-wise.

@highlight

Using Bayesian regression to estimate the posterior over Q-functions and deploy Thompson Sampling as a targeted exploration strategy with efficient trade-off the exploration and exploitation

@highlight

The authors propose a new algorithm for exploration in Deep RL where they apply Bayesian linear regression with features from the last layer of a DQN network to estimate the Q function for each action.

@highlight

The authors describe how to use Bayesian neural networks with Thompson sampling for efficient exploration in q-learning and propose an approach that outperforms epsilon-greedy exploration approaches.
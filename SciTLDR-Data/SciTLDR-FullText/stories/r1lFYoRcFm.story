Learning from a scalar reward in continuous action space environments is difficult and often requires millions if not billions of interactions.

We introduce state aligned vector rewards, which are easily defined in metric state spaces and allow our deep reinforcement learning agent to tackle the curse of dimensionality.

Our agent learns to map from action distributions to state change distributions implicitly defined in a quantile function neural network.

We further introduce a new reinforcement learning technique inspired by quantile regression which does not limit agents to explicitly parameterized action distributions.

Our results in high dimensional state spaces show that training with vector rewards allows our agent to learn multiple times faster than an agent training with scalar rewards.

Reinforcement learning BID32 ) is a powerful paradigm in which an agent learns about an environment through interaction.

The common formulation consists of a Markov Decision Process (MDP) modeled as a 5-tuple (S, A, P, r, γ) where S is the (possibly infinite) set of states, A is the (possibly infinite) set of actions available to the agent, P : (S × A × S) → [0, 1] : P (s |s, a) is the transition probability of reaching state s ∈ S given state s ∈ S and action a ∈ A, r : (S × A) → R : r(s, a) is the reward received for taking action a in state s and γ is the reward discount factor.

The goal of the agent is to maximize the cumulative discounted reward R = ∞ t=0 γ t r(s t , a t ) by choosing actions a t according to some (possibly stochastic) policy π : (S × A) → [0, 1] : π(a t |s t ).

Sometimes it is further useful to make a distinction between the actual state space S and the correlated observation space O of the agent.

In this case π : (O × A) → [0, 1] : π(a t |o t ) with o t ∈ O. The use of deep neural networks allowed this formulation to scale to high dimensional visual inputs approaching continuity in state space BID20 while others extended deep reinforcement learning to continuous action spaces BID15 BID21 .

While neural networks are powerful function approximators, they require large amounts of training data to converge.

In the case of reinforcement learning this means interactions with the environment, a requirement easy to fulfill in simulation, yet impractical when the agent should interact with the real world.

This problem is aggravated by the weak training signal of classical reinforcement learning -a simple scalar reward.

While originally the dopamine activity in mammal brains was linked to general "rewarding" events, BID28 point out that the diversity of dopamine circuits in the mid brain is better modeled by viral vector strategies.

BID8 also show that human reinforcement learning incorporates effector specific value estimations to cope with the high dimensional action space.

Inspired by these biological insights, we improve the sample efficiency of a deep reinforcement learning algorithm in this work by modeling a d-dimensional vector reward.

A vector reward can in some domains easily be defined in alignment with the state space.

We say that two vector spaces are aligned if their dimensions correlate and show that if state and action space are not aligned, a mapping from action distribution to state change distribution can be learned.

As a motivating example, consider the agent in Figure 1 (a) trying to reach the goal (marked by the blue dot).

If we take p as the position vector of the agent relative to the goal, a sensible reward to guide the agent to the goal in this environment would be r = ||p|| − ||p + a|| where || · || can be any norm in the vector space of the environment.

For illustration purposes we'll focus on the L 1 norm in this example and throughout this paper.

During training the agent might try action a which moves Figure 1: (a) An agent freely moving in a 2D world might try to reach a goal at position (0, 0) by taking action a. A sensible reward in this environment is the change in absolute distance to the goal.

With a scalar reward this would be summarized as r = r x − r y , whereas a vector reward would keep the two reward dimensions distinguishable.

(b) In most cases action and state space are however not aligned, therefore a mapping from action to state change must be learned.it closer to the goal in x direction, but a bit further away from the goal in y direction.

The scalar reward would then just convey the information, that the action was rather positive (since the agent got closer to the goal) but miss out on the distinction that the action was good in x-direction but bad in y-direction.

To provide this distinction, a more informative reward would keep the dimensions separate and therefore be a vector itself: r = |p| − |p + a| where | · | denotes the element-wise absolute value here.

Note that this reward is dimension wise aligned with the position p, the state, of the agent.

Since we focus on reaching problems in this work, we'll use the terms "position" and "state" interchangeably.

The problem with such a state aligned vector reward is however that the action space is in most cases not state aligned.

To see this, consider the schematic robot arm in Figure 1 (b): The action dimensions a 1 and a 2 correspond to the torques of the robot arm and do not directly translate to a shift in x and y dimension, respectively.

To address this issue we use the method proposed by BID5 a) to train a deep neural network to approximate the quantile function, in our case of the position change, given the current observation and quantile target.

Additionally we give a parameterization of the action probability distribution as input to this position change prediction network (short PCPN).

We then train the agent, parameterized by another neural network which maps from observations to action probability distributions, through a new reinforcement learning method we call quantile regression reinforcement learning (short QRRL).

A schematic overview of our setup can be seen in FIG1 .To summarize, the contributions of this paper are the following:• We extend the reinforcement learning paradigm to allow for faster training based on more informative state aligned vector rewards.• We present an architecture that learns a probability distribution over possible state changes based on a probability distribution over possible actions.• We introduce a new reinforcement learning algorithm to train stochastic continuous action policies with arbitrary action probability distributions.

Quantile regression BID13 discusses approximation techniques for the inverse cumulative distribution function F −1 Y , i.e., the quantile function, of some probability distribution Y .

Recent work BID4 shows that a neural network can learn to approximate the quantile function by mapping a uniformly sampled quantile target τ ∼ U([0, 1]) to its corresponding quantile function value F −1 Y (τ ) ∈ R. Thereby the trained neural network implicitly models the full probability distribution Y .

More formally, let DISPLAYFORM0 of distributions U and Y , also characterized as the L p metric of quantile functions BID23 .

BID5 show that the quantile regression loss BID14 ) DISPLAYFORM1 minimizes the 1-Wasserstein distance of a scalar probability distribution Y to a uniform mixture of Diracs U .

Here, δ = y − u with y ∼ Y and u ∼ U is the quantile sample error.

generalized this result by showing that the expected quantile loss DISPLAYFORM2 Z of some distribution Z is equal to the quantile divergence DISPLAYFORM3 plus some constant not depending on the parameters θ.

Here, Q θ is the distribution implicitly defined byQ θ .

Therefore, training a neural networkQ θ (τ ) to minimize ρ τ (z −Q θ (τ )) with z sampled from the target probability distribution Z effectively minimizes the quantile divergence q(Z, Q θ ) and thereby models an approximate distribution Q θ of Z implicitly in the network parameters θ of the neural networkQ θ (τ ).By approximating the quantile function instead of a parameterized probability distribution, as common in many deep learning models BID11 BID15 BID9 ), we do not enforce any constraint on the probability distribution Z, e.g., Z can be multi-modal, not continuous and non-Gaussian.

This is crucial for our case as a position change distribution given a certain action distribution can have any shape.

In our setup, we train the PCPN with the quantile regression loss (1) to approximate a position change quantile function per position dimension.

Aside from a target quantile τ ∼ U([0, 1]) per position dimension, the network input consists of an observation o ∈ O and a multi-variate Gaussian action probability distribution A = N (µ, Σ) with diagonal covariance matrix Σ = Iσ, parameterized by the mean µ and variance σ vectors.

Therefore, the PCPN has the ability to implicitly learn the conditional position change distribution given an observation and an action distribution.

The dominant reinforcement learning algorithms are either value based methods, e.g., Q-learning BID37 BID20 , policy gradient based methods, e.g., REINFORCE BID38 BID33 , or a combination of both, e.g., actor-critic methods BID32 BID21 .

In this section we establish a new reinforcement learning objective based on quantile regression.

In contrast to Q-learning we allow for a continuous action space similar to policy based algorithms, but in contrast to policy based algorithms we do not limit ourselves to explicitly parameterized policies.

Note that the technique described in this section is generally applicable to reinforcement learning problems and we therefore use the terms "action" and "policy" in the common reinforcement learning meaning, which is distinct from the meaning of "action" and "policy" in the rest of the paper.

We later link in Section 4 the position change estimation of the PCPN to this new meaning of "action" to connect with the rest of the paper.

The main idea behind Quantile Regression Reinforcement Learning (or QRRL in short) is to model the policy implicitly in the network parameters, therefore allowing for complex stochastic policies which can find multi-modal stochastic optima in policy space.

For this we model for each action dimension the quantile functionQ ) and taking the network output as action.

Since the network approximates quantile functions, the network output of a uniformly at random sampled quantile target is a sample from the implicitly defined action distribution.

The question left to address is how to train the network, such that it (a) approximates the quantile functions of the action dimensions and (b) the implicitly defined policy maximizes the expected (discounted) reward R. DISPLAYFORM0 Here quantile regression comes in handy.

Informally put, quantile regression is linked to the Wasserstein metric which is also sometimes refered to as earth movers distance.

Imagine a pile of earth representing probability mass.

In reinforcement learning we essentially want to move probability mass towards actions that were good and away from actions that were bad, where "good" and "bad" are measured by discounted accumulative (bootstrapped) reward achieved.

Quantile regression can achieve this neatly by shaping the pile of earth according to an advantage estimation and the constraint of monotonicity (which is a core property of quantile functions).More formally, we are interested in the effect of the quantile regression loss (1) on action probabilities when the error δ is between two samples from the same network, one representing the action that was played in the environment (which resulted from some sample quantile τ ) and one representing the quantile function value at some τ where the network tries to approximate the quantile function of the optimal action distribution.

We focus in this analysis on the simple case of a single action dimension with scalar quantile target input and scalar quantile function output.

The generalization to multidimensional action-/quantile-functions with independent action dimensions follows trivially.

First, let us denote the action taken in the environment by a τ :=Q θ (τ, o) =Q θ (τ ), where we hide the dependence ofQ θ (τ, o) on the observation o hereafter for notation simplicity.

To train the quantile network we sample different τ and consider the loss DISPLAYFORM1 As we are interested in the effect of the loss on the probability of a τ consider a τ close to τ , i.e., τ = τ ± for some > 0 and < τ < 1 − .

For τ = τ − the loss reduces to DISPLAYFORM2 DISPLAYFORM3 where dτ 1 is an infinitely small but positive value Similarly we get for τδτ with dτ 2 an infinitely small but positive value.

Therefore, the quantile regression loss is positively correlated to the slope ofQ θ at τ .

The partial derivative of a quantile functionQ θ (τ ) with respect to the quantile τ is however also known as the sparsity function BID35 or quantile-density function BID26 and has the interesting property BID10 : DISPLAYFORM4 where p Q θ is the probability density function of distribution Q θ .

Hence, the quantile regression loss is inverse proportional to the probability of action a τ , which implies the following, given thatQ θ is the quantile function of distribution Q θ :1.

Minimizing the quantile loss (1) for a given action a τ increases the likelihood of action a τ .2.

Maximizing the quantile loss for a given action a τ decreases the likelihood of action a τ .This leads us to the QRRL actor objective DISPLAYFORM5 Q θ is monotonically increasing with τ where (R t,n − V ψ (o t )) is an advantage estimation BID21 with R t,n = γ n V ψ (o t+n ) + t+n t =t γ t −t r t being the n-step estimate of the discounted future reward and V ψ (·) being the output of a critic network with parameters ψ, which is trained to approximate R t,n .

Note that QRRL is a constraint optimization (sinceQ θ must be a proper quantile function, i.e., monotonically increasing).In our experiments we found it however sufficient to add this constraint as an additional loss term DISPLAYFORM6 We weight this additional loss term with a constant Lagrange multiplier λ mon in the full loss term DISPLAYFORM7 with DISPLAYFORM8 2 .

λ c is another constant Lagrange multiplier to weight the critic loss L c against the actor loss L a .

Although we focus in our experiments on an on-policy method with multiple actors, QRRL can easily be adapted to the off-policy setting.

In our State Aligned VEctor Reward (SAVER) agent, we use QRRL to train the agent network AN η through the PCPN.

For this we feed the action probability distribution output µ, σ of the agent network to a pretrained PCPN and train on the QRRL loss (2) with respect to the agent network parameters η, where we take the actual position change ∆p introduced by a sampled action a ∼ N (µ, Iσ) as QRRL action target a τ = ∆p and compare it to K sampled position change estimations ∆p(τ DISPLAYFORM0 .., K}. Note that we pretrain the PCPN in our setup with random observations and action probability distributions and freeze the PCPN weights during agent training.

Therefore, one could potentially train several agents to solve different tasks in the same environment using the same PCPN.Since the output of the PCPN can be seen as state aligned action, training on vector rewards is straight forward.

Instead of having a scalar critic estimating the value function V (·) ∈ R we estimate a value per action dimension, i.e., V (·) ∈ R d .

Similarly, the vector rewards can be summed to a vector n-step discounted reward estimation R t,n = γ n V (o t+n ) + t+n t =t γ t −t r t , which leads us to a vector advantage (R t,n − V (o t )) at timestep t.

Therefore we have an advantage estimation for each position change dimension individually and can apply loss (2) to each position change estimation dimension individually.

To test our ideas, we implement three simple experiments, two to test our approach in high dimensional metric spaces and one to show the applicability of our approach to a real world problem by modeling the robot arm shown in Figure 1(b) .

For reproducibility and to stimulate further research in this area, our code is publicly available.1 We compare our SAVER agent against an A2C agent, the synchronous variant of A3C BID21 .

We choose this baseline since it is most similar to our SAVER implementation and we believe that SAVER can benefit in the future from state-ofthe-art additions to A3C as presented by BID7 .

Here however, we want to focus on the benefit of using vector rewards instead of scalar rewards.

We implement a simple feed forward .

The x-axis shows the number of training steps in millions while the y-axis shows the average episode length over the last 100 episodes.

Training length was fixed to 2,560,000 steps.

Plotted is the average of 3 training runs with the shaded area indicating the standard deviation between runs.

The plots suggest that the higher dimensional the environment, the more apparent the gain of training on vector rewards is.neural architecture architecture and fix most hyperparameters to values that work well for SAVER and A2C.

Details can be found in Appendix A.In a first experiment, we let the agent move freely in all directions within a d-dimensional hypercube by choosing the action space to represent a set of (angle, step size) pairs for moving in the pairwise grouped environment dimensions.

Note that this setup is similar to the depiction in Figure 1 Step sizes are re-scaled and clipped to a maximal value of 0.1 and episodes are terminated after 1,000 steps if the agent didn't manage to get into close proximity of the goal beforehand.

We pretrain the PCPN with 100,000 batches of 128 transitions each by randomly sampling o ∼ U( DISPLAYFORM0 where actions between -1 and 1 correspond to step sizes between -0.1 and 0.1 and angles between 0 and π, respectively.

2 We performed a small hyperparameter search for the learning rate in the 8-dimensional hypercube and settled for a learning rate of 0.01 (chosen from {0.01, 0.003, 0.001}) for SAVER and a learning rate of 0.0003 (chosen from {0.001, 0.0003, 0.0001}) for A2C.

3 We measure the performance of the agents by the mean length of the last 100 episodes.

This mean episode length is plotted in FIG3 over the course of training for hypercubes of different dimensionality.

Average and standard deviation of three training runs is shown.

As can be seen from the plots, the higher the dimensionality d of the environment, the more apparent the advantage of training on vector rewards gets.

SAVER trains faster in high dimensional cubes and is even able to find the goal in a 16-dimensional cube given the step limit of 1,000 steps.

As a second experiment, we keep the environment specifications the same but change the action representation.

Here the agent's action consists of two parts: a softmax distribution from which the dimension to be manipulated is chosen (discrete action part) and a scalar Gaussian distribution defined by network outputs µ and σ from which the step size is sampled (continuous action part).

Note that this action composition leads to a not continuous position change distribution when regarding the position change dimensions isolated, since in each dimension the probability of a change equal to 0 is more likely then position changes close to 0.

We show with this experiment, that the quantile regression based PCPN can learn to implicitly model this complex position change distribution based on the composed action distribution input.

For this we pretrain the PCPN with 10,000 batches of 128 transitions each by sampling softmax logits and µ uniformly at random from U([−1, 1]) and Figure 1(b) .

The x-axis shows the number of training steps in millions while the y-axis shows the average episode length over the last 100 episodes.

Plotted is the average of 5 training runs with the shaded area indicating the standard deviation between runs.σ from U([0, 1]).

In this experiment we searched for an appropriat learning rate in the 4 dimensional hypercube and settled for 0.01 for SAVER (chosen from {0.01, 0.003, 0.001}) and 0.0001 for A2C (chosen from {0.001, 0.0003, 0.0001}).

The corresponding mean episode lengths over the course of agent training are plotted in Figure 4 .

Again we find that the higher dimensional the hypercube is, the more advantageous it is to train with vector rewards.

In our last experiment, we model the 2-joint robot arm depicted in Figure 1 2 ), respectively.

We choose the learning rate of SAVER as 0.003 and A2C as 0.0003 based on a search over {0.1, 0.01, 0.003, 0.001} and {0.01, 0.001, 0.0003, 0.0001}, respectively.

We pretrain the PCPN with 10,000 batches of 128 transitions each by sampling µ ∼ U( DISPLAYFORM1 The corresponding mean episode lengths over the course of training are plotted in Figure 5 .

Even in this low dimensional problem, SAVER trains slightly faster than A2C due to the more informative vector rewards.

Using a vector of rewards instead of a scalar reward is most common in the literature on MultiObjective Reinforcement Learning (MORL) BID16 and Multi-Objective Sequential Decision Making BID31 BID30 .

However, since most of this literature focuses on classical reinforcement learning and conflicting objectives, the methods discussed either reduce the multiple objectives to a single objective or learn different policies for each objective additionally to a superpolicy deciding on which policy to use when.

Recent work BID22 BID34 BID24 also applied these techniques to deep reinforcement learning agents.

In contrast, we directly use the vector reward as more informative training signal for a neural network which implicitly learns to trade off different objectives.

Other adjacent areas are multi-agent and multi-task deep reinforcement learning BID0 .

While former considers multiple agents with potentially different observations, later addresses how a single agent best solves multiple tasks.

In contrast, our work considers a single agent solving a single task by leveraging a multi-dimensional reward.

Klinkhammer (2018) discusses multiple problem aligned rewards for better learning, while BID3 discuss the effect of correlated rewards on learning performance.

BID3 also suggest multiple reward shapings of the same reward function for faster learning.

BID36 decompose the reward function into multiple rewards and train an architecture similar to ours with deep Q learning, assigning a Q-value output to each reward.

While all three approaches come to the same conclusion as we do, i.e., increased training performance, they do require hand engineered reward functions, reward shapings and/or reward decompositions.

In contrast, our approach is based on the fact that many state spaces are metric spaces and therefore allow for a straight forward vector reward interpretation.

This makes our approach easier to apply to a larger set of tasks.

While state proximity was already used by BID18 for faster backpropagation of rewards in tabular reinforcement learning, we are unaware of any deep learning algorithm capitalizing on state aligned vector rewards as we do.

Many recent approaches to deep reinforcement learning learn the environment dynamics to have a richer training signal BID6 , imagine action consequences BID29 , improve exploration BID27 or dream up solutions BID9 ).

An interesting line of research BID39 BID1 BID17 BID2 in this direction uses successor features to share knowledge between tasks in the same environment.

In contrast to most of these works, which often only predict one possible next state or successor feature, our PCPN incorporates the full probability distribution of possible state changes.

While BID9 also predict the full probability distribution of their next state representation by a Gaussian mixture model, our approach is more general in that it is also able to approximate non-Gaussian probability distributions.

BID5 were the first to use quantile regression in connection with deep reinforcement learning.

In their work, including their followup work BID4 , they focused on approximating the full probability distribution of the value function.

In contrast, with QRRL we explore possibilities of using quantile regression to approximate richer policies by not constraining the action distribution to an explicitly parameterized distribution. showed that quantile networks can also be used for generative modeling.

In general, we see quantile regression in combination with deep learning to have a lot of potential for future work.

In this work we present the idea of state aligned vector rewards for faster reinforcement learning.

While the idea is straight forward and simple, we are unaware of any work that addresses it so far.

Additionally, we also present a new reinforcement learning technique based on quantile regression in this work which we term QRRL.

QRRL allows for complex stochastic policies in continuous action spaces, not limiting agents to Gaussian actions.

Combining both, we show that the agent network in our SAVER agent can be trained through a quantile network pretrained in the environment.

We show that SAVER is capable of training orders of magnitudes faster in high dimensional metric spaces.

While d-dimensional metric spaces are mainly mathematical constructs for d > 3, we see a lot of potential in SAVER to be applied to problems in mathematics and related fields, including the field of deep (reinforcement) learning itself.

Paper under double-blind review.

As agent network, we used a simple layer-wise fully connected network with 3 hidden representations of size 256, 128 and 64, respectively.

From the last hidden representation we map with a linear layer to the action mean µ, while the action standard deviation σ is initialized to e −1 · [1, ..., 1]T and kept independent of the observation input.

For the A2C implementation, the last hidden representation of the agent network is also mapped through a linear layer to the scalar value estimate V (o).

For the PCPN we first map from µ, σ and o to a hidden representation of size 256.

From this hidden representation we map to d hidden representations, each of size 128, where d is the number of state dimensions.

We then multiply each of these d hidden representation with a 128-dimensional cosine embedding of corresponding quantile target τ j with j ∈ {1, ..., d}. Each of this new hidden representations is then fed through a fully connected layer with 64 neurons before being linearly projected to a scalar value representing the position change estimation of dimension j for quantile τ j .

We implement the QRRL critic as a separate network with 3 hidden representations of size 256, 128 and 64 and the same input as fed to the PCPN's first layer.

Besides the action distribution representation which is ajusted corresponding to the experiment, we keep this architecture and all of the hyperparameters (as listed in TAB0 ) fixed for all experiments.

The only hyperparameter we adjust is the learning rate, as described in the main text.

@highlight

We train with state aligned vector rewards an agent predicting state changes from action distributions, using a new reinforcement learning technique inspired by quantile regression.

@highlight

Presents algorithm that aims to speed up reinforcement learning in situations where the reward is aligned with the state space. 

@highlight

This paper addresses RL in the continuous action space, using a re-parametrised policy and a novel vector-based training objective.

@highlight

This work proposes to mix distributional RL with a net in charge of modeling the evolution of the world in terms of quantiles, claiming improvements in sample efficiency.
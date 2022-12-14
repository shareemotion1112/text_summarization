Exploration while learning representations is one of the main challenges Deep Reinforcement Learning (DRL) faces today.

As the learned representation is dependant in the observed data, the exploration strategy has a crucial role.

The popular DQN algorithm has improved significantly the capabilities of Reinforcement Learning (RL) algorithms to learn state representations from raw data, yet, it uses a naive exploration strategy which is statistically inefficient.

The Randomized Least Squares Value Iteration (RLSVI) algorithm (Osband et al., 2016), on the other hand, explores and generalizes efficiently via linearly parameterized value functions.

However, it is based on hand-designed state representation that requires prior engineering work for every environment.

In this paper, we propose a Deep Learning adaptation for RLSVI.

Rather than using hand-design state representation, we use a state representation that is being learned directly from the data by a DQN agent.

As the representation is being optimized during the learning process, a key component for the suggested method is a likelihood matching mechanism, which adapts to the changing representations.

We demonstrate the importance of the various properties of our algorithm on a toy problem and show that our method outperforms DQN in five Atari benchmarks, reaching competitive results with the Rainbow algorithm.

In Reinforcement Learning (RL), an agent seeks to maximize the cumulative rewards obtained from interactions with an unknown environment (Sutton et al., 1998) .

Since the agent can learn only by its interactions with the environment, it faces the exploration-exploitation dilemma: Should it take actions that will maximize the rewards based on its current knowledge or instead take actions to potentially improve its knowledge in the hope of achieving better future performance.

Thus, to find the optimal policy the agent needs to use an appropriate exploration strategy.

Classic RL algorithms were designed to face problems in the tabular settings where a table containing a value for each state-action pair can be stored in the computer's memory.

For more general settings, where generalization is required, a common practice is to use hand-designed state representation (or state-action), upon which a function approximation can be learned to represent the value for each state and action.

RL algorithms based on linear function approximation have demonstrated stability, data efficiency and enjoys convergence guarantees under mild assumptions (Tsitsiklis & Van Roy, 1997; Lagoudakis & Parr, 2003) .

They require that the desired learned function, e.g. Qfunction, will be a linear combination of the state representation.

This is, of course, a hard constraint as the representation is hand-designed, where the designer often does not know how the optimal value-function will look like.

Furthermore, hand-designed representation is environment-specific and requires re-designing for every new environment.

The DQN algorithm (Mnih et al., 2015) has changed RL.

Using Deep Neural Networks (DNN) as function approximators, the DQN algorithm enabled the learning of policies directly from raw highdimensional data and led to unprecedented achievements over a wide variety of domains (Mnih et al., 2015) .

Over the years, many improvements to DQN were presented, suggesting more fitting network architectures (Wang et al., 2015) , reducing overestimation (Van Hasselt et al., 2016; Anschel et al., 2017) or improving its data efficiency .

Despite its great success, DQN uses the overly simple -greedy strategy for exploration.

This strategy is one of the simplest exploration strategies that currently exist.

The agent takes random action with probability and takes the optimal action according to its current belief with probability 1 ??? .

This strategy is commonly used despite its simplicity and proven inefficiency (Osband et al., 2016) .

The main shortcoming of -greedy and similar strategies derives from the fact that they do not use observed data to improve exploration.

To explore, it takes a completely random action, regardless of the experience obtained by the agent.

Thompson Sampling (TS) (Thompson, 1933) , is one of the oldest heuristics to address the 'exploration/exploitation' trade-off in sequential decision-making problems.

Its variations were proposed in RL (Wyatt, 1998; Strens, 2000) and various bandits settings (Chapelle & Li, 2011; Scott, 2010) .

For Multi-Armed Bandit (MAB) problems, TS is very effective both in theory (Agrawal & Goyal, 2012; and practice (Chapelle & Li, 2011) .

Intuitively, TS randomly takes actions according to the probability it believes to be optimal.

In practice, a prior distribution is assumed over the model's parameters p(w), and a posterior distribution p(w|D) is computed using the Bayes theorem, where D is the observed data.

TS acts by sampling models from the posterior distribution, and plays the best action according to these samples.

Randomized Least Squares Value Iteration (Osband et al., 2016) is an RL algorithm which uses linear function approximation and is inspired by Thompson Sampling.

It explores by sampling plausible Q-functions from uncertainty sets and selecting the action that optimizes the sampled models.

This algorithm was proven to be efficient in tabular settings, with a bound on the expected regret that match the worst-case lower bound up to logarithmic factors.

More importantly, it demonstrates efficiency even when generalization is required.

Alas, as it assumes a linearly parametrized value function on a hand-designed state representation, the success of this algorithm crucially depends on the quality of the given state representation.

In this paper, we present a new DRL algorithm that combines the exploration mechanism of RLSVI with the representation learning mechanism of DQN; we call it the Deep Randomized Least Squares Value Iteration (DRLSVI) algorithm.

We use standard DQN to learn state representation and explores by using the last layer's activations of DQN as state representation for RLSVI.

To compensate for the constantly changing representation and the finite memory of DQN, we use a likelihood matching mechanism, which allows the transfer of information held by an old representation regarding past experience.

We evaluate our method on a toy-problem -the Augmented Chain environment -for a qualitative evaluation of our method on a small MDP with a known optimal value function.

Then, we compare our algorithm to the DQN and Rainbow algorithms on several Atari benchmarks.

We show that it outperforms DQN both in learning speed and performance.

Thompson Sampling in Multi-Armed Bandit problems: Thompson Sampling (TS) (Thompson, 1933) , is one of the oldest heuristics to address the 'exploration/exploitation' trade-off in sequential decision-making problems.

Chapelle & Li (2011) sparked much of the interest in Thompson Sampling in recent years.

They rewrote the TS algorithm for Bernoulli bandit and showed impressive empirical results on synthetic and real data sets that demonstrate the effectiveness of the TS algorithm.

Their results demonstrate why TS might be a better alternative to balance between exploration and exploitation in sequential decision-making problems than other popular alternatives like the Upper Confidence Bound algorithm (Auer et al., 2002) .

Agrawal & Goyal (2013) suggested a Thompson Sampling algorithm for the linear contextual bandit problem and supplied a high-probability regret bound for it.

They use Bayesian Linear Regression (BLR) with Gaussian likelihood and Gaussian prior to design their version of Thompson Sampling algorithm.

Riquelme et al. (2018) suggested performing a BLR on top of the representation of the last layer of a neural network.

The predicted value v i for each action a i is given by v i = ?? T i z x , where z x is the output of the last hidden layer of the network for context x. While linear methods directly try to regress values v on x, they independently trained a DNN to learn a representation z, and then used a BLR to regress v on z, obtaining uncertainty estimates on the ??'s, and making decisions accordingly via Thompson Sampling.

Moreover, the network is only being used to find good representation -z.

Since training the network and updating the BLR can be done independently, they train the network for a fixed number of iterations, then, perform a forward pass on all the training data to obtain the new z x , which is then fed to the BLR.

This procedure of evaluating the new representation for all the observed data is very costly, moreover, it requires infinite memory which obviously does not scale.

Zahavy & Mannor (2019) suggested matching the likelihood of the reward under old and new representation to avoid catastrophic forgetting when using such an algorithm with finite memory.

In the Reinforcement Learning settings, Strens (2000) suggested a method named "Posterior Sampling for Reinforcement Learning" (PSRL) which is an application of Thompson Sampling to Model-Based Reinforcement Learning.

PSRL estimates the posterior distribution over MDPs.

Each episode, the algorithm samples MDP from it and finds the optimal policy for this sampled MDP by dynamic programming.

Recent work (Osband et al., 2013; have shown a theoretical analysis of PSRL that guarantees strong expected performance over a wide range of environments.

The main problem with PSRL, like all modelbased approaches, is that it may be applied to relatively small environments.

The Randomized Least Squares Value Iteration (RLSVI) algorithm is an application of Thompson Sampling to Model-Free Reinforcement Learning.

It explores by sampling plausible Q-functions from uncertainty sets and selecting the action that optimizes the sampled models.

Thompson Sampling in DRL: Various approaches have been suggested to extend the idea behind RLSVI to DRL.

Bootstrapped DQN uses an ensemble of Q-networks, each trained with slightly different data samples.

To explore, Bootstrapped DQN randomly samples one of the networks and acts greedy with respect to it.

Recently, Osband et al. (2018) extended this idea by supplying each member of the ensemble with a different prior.

Fortunato et al. (2017) and Plappert et al. (2017) investigate a similar idea and propose to adaptively perturb the parameterspace, which can also be thought of as tracking approximate posterior over the network's parameters.

O'Donoghue et al. (2017) proposed TS in combination with uncertainty Bellman equation, which connects the uncertainty at any time-step to the expected uncertainties at subsequent timesteps.

Recently and most similar to our work, Azizzadenesheli et al. (2018) experimented with a Deep Learning extension to RLSVI.

They changed the network architecture to exclude the last layer weights, optimized the hyper parameters and used double-DQN.

In contrary, we don't change anything in the DQN agent.

We use the representation learned by DQN to perform RLSVI, however, the network structure, loss and hyper-parameters are the same.

Additionally, differently from our method, they don't compensate for the changing representation and solve BLR problem with the same arbitrary prior every time.

We consider the standard RL settings (Sutton et al., 1998) , in which an environment with discrete time steps is modeled by a Markov Decision Process (MDP).

An MDP is a tuple < S, A, P, R, ?? >, where S is a state space, A a finite action space, P : S ?? A ??? ??? ???(S), is a transition kernel, and R : S ?? A ??? ??? R a reward function.

At each step the agent receives an observation s t ??? S which represents the current physical state of the system, takes an action a t ??? A which is applied to the environment, receives a scalar reward r t = r(s t , a t ), and observes a new state s t+1 which the environment transitions to.

As mentioned above, the agent seeks an optimal policy ?? * : S ??? ??? ???(A), mapping an environment state to probabilities over the agent's executable actions.

?? ??? (0, 1) is the discount factor -a scalar representing the trade-off between immediate and delayed reward.

A brief survey of the DQN algorithm can be found in Appendix 1.

The Randomized Least Squares Value Iteration (RLSVI) algorithm is a TS-inspired exploration strategy for Model-Free Reinforcement Learning.

It combines TS-like exploration and linear function approximation, where the main novelty is in the manner in which it explores: Sampling value-functions rather than sampling actions.

The Q-function is assumed to be in the form Q(s, a) = ??(s, a)

T w, where ??(s, a) is a hand-designed state-action representation.

RLSVI operates similar to other linear function algorithms and minimizes the Bellman equation by solving a regression problem -Bayesian Linear Regression.

BLR obtains a posterior distribution over valuefunction instead of point estimates.

To explore, RLSVI samples plausible value functions from the posterior distribution and acts the greedy action according to the sampled value-function.

In the episodic settings where the representation is tabular, i.e., no generalization is needed, RLSVI guarantees near-optimal expected episodic regret.

Finally, the main benefit of this algorithm is that it displays impressive results even when generalization is required -despite the lack of theoretical guarantees.

A pseudo-code can be found in Appendix 1.

In this paper, we propose to use RLSVI as the exploration mechanism for DQN.

RLSVI capabilities are enhanced by using state representation that is learned directly from the data by a neural network rather than hand-designed one.

As the neural network gradually improves its representation of the states, a likelihood matching mechanism is applied to transfer information from old to new representations.

A DQN agent is trained in the standard fashion, i.e., the same architecture, hyper-parameters and loss function as the original DQN.

Two exceptions were made (1) The size of the last hidden layer is reduced to be d = 64.

(2) The Experience Replay buffer is divided evenly between actions and transitions are stored in a round-robin fashion.

I.e., whenever the buffer is full, a new transition < s t , a t , r t , s t+1 > is placed instead of the oldest transition with the same action a t .

Exploration is performed using RLSVI on top of the last hidden layer of the target network.

Given a state s t , the activations of the last hidden layer of the target network applied to this state are denoted as ??(s t ) = LastLayerActivations(Q ??target (s t )).

Several changes to the original RLSVI algorithm were made: First, rather than solving different regression problem for every time step, a different regression problem is being solved for every action.

As the last hidden layer's activations serves as state representation, the representation is time-homogeneous and shared among actions.

The regression targets y are DQN's targets which use the target network predictions.

Another change is that a slightly different formulation of Bayesian Linear Regression than RLSVI is being used.

Similar to RLSVI a Gaussian form for the likelihood is assumed:

2 ), however, like Riquelme et al. (2018) the noise parameter ?? 2 is formulated as a random variable, which is distributed according to the Inverse-Gamma distribution.

The prior for each regression problem is therefore in the form:

2 ) it is known that the posterior distribution can be calculated analytically as follows:

Formulating ?? 2 as a random variable allows adaptive exploration where the adaptation is derived directly from the observed data.

Lastly, while RLSVI's choice for the prior's parameters is somewhat arbitrary, in our algorithm the prior has a central role which we'll discuss further on.

Since RLSVI requires a fixed representation and the target network's weights are fixed, we use the last layer activations of the target network, denoted ??(s), as state-representation.

Every T target training time steps, the target network is updated with the weights of the Q-network.

In these T target time steps the Q-network is changing due to the optimization performed by the DQN algorithm.

Since the representation is changing, the posterior distribution that was approximated in the old representation can't be used.

A posterior distribution based on the new representation needs to be approximated.

Therefore, whenever the target network changes, new Bayesian linear regression problems are being solved using N BLR samples from the ER buffer.

Since the ER buffer is finite, some experience-tuples were used to approximate the posterior in the old representation and are no longer available.

Ignoring this lost experience can and will lead to degradation in performance derived by 'Catastrophic Forgetting' (Kirkpatrick et al., 2017) .

To compensate for the changing representation and the loss of old experience, we follow (Zahavy & Mannor, 2019) and match the likelihood of the Q-function in the old and new representation.

This approach assumes that the important information from old experiences is coded in the state representation.

Observe s t+1 , r t , Store Transition < s t , a t , s t+1 , r t > Train DQN using sampled mini-batch if t mod

Since the likelihood is Gaussian, to compensate for the changing representation, we will find moments that match the likelihood of the Q-function in the new representation to the old one and use them for our Gaussian prior belief.

Expectation prior: As DQN is trained to predict the Q-function, given the new last layer activations ??, a good prior for ?? 0 in the new representation will be the last layer weights of the DQN (Levine et al., 2017) .

Covariance prior: We use N SDP samples from the experience replay buffer.

We evaluate both old and new representation {??(

.

Our goal is to find a solution ?? ?? 0 that will match the covariance of the likelihood in the new representation to the old one:

Using the cyclic property of the trace, this is equivalent to finding Trace(??(

Adding the constraint that ?? ?? 0 should be Positive-Semi-Definite as it is a covariance matrix, we end up with the following Semi-Definite Program (SDP) (Vandenberghe & Boyd, 1996) :

In practice, we solve this SDP by using CVXPY (Diamond & Boyd, 2016) .

Approximate Sampling: To perform Thompson Sampling one needs to sample the posterior distribution before every decision.

This, regrettably, is computationally expensive and will slow down training significantly.

To speed up learning, we sample j weights for every action {w i,1 , ...,w i,j } every T Sample time steps.

Then, every step we sample an index i a ??? 1, .., j for every action, which is computationally cheap, and act greedy accordingly: a t = arg max ??w T ??,i?? ??(s t ).

Solving the SDP: Another bottleneck our algorithm faces is solving the SDP.

We refer the reader to Vandenberghe & Boyd (1996) for an excellent survey on the complexity of solving SDPs.

As the running time of an SDP solver mainly depends on the dimension of the representation d, the number of samples being used N SDP and the desired accuracy , we chose the last hidden layer size to be d = 64, used N SDP = 600 for every SDP and set = 1e ??? 5.

The running time for solving a single SDP took us 10-50 seconds using Intel's Xeon CPU E5-2686 v4 2.30 GHz.

We conduct a series of experiments that highlight the different aspects of our method.

We begin with a qualitative evaluation of our algorithm on a simple toy environment, then, move to report quantitative results on 5 different Atari games in the ALE.

The chain environment includes a chain of states S = 1, ..., n. In each step, the agent can transition left or right.

This standard-settings is augmented with additional k actions which transitions the agent to the same state (self-loop).

We name this variation "The Augmented Chain Environment".

All states have zero rewards except for the far-right n-state which gives a reward of 1.

Each episode is of length H = n ??? 1, and the agent will begin each episode at state 1.

The raw state-representation is a one-hot vector.

The Q-network is an MLP with 2 hidden layers.

Results are averaged across 5 different runs.

We report the cumulative episodic regret:

Here T is the number of played episodes, v * 0 (s 0 ) is the return of the optimal policy, and r t,h is the reward the agent received in episode t at time step h. An illustration for the augmented chain environment can be found in figure 1 .

The hyper-parameters that are being used in the following experiments can be found in Appendix 2.

We compared our algorithm to standard DQN where -greedy serves as the exploration strategy.

We experimented with various values, however, as the results for different values were similar, we display the result for a single value (Figure 2 (a) ).

We can see that -greedy (red) achieves a linear regret in T which is the lower bound for this type of problems, while our algorithm (blue) achieves much lower regret.

These results demonstrate that -greedy can be highly inefficient even in very simple scenarios.

In this experiment, we compared our algorithm with variants that do not model ?? 2 as a random variable.

We experimented with various constant ?? 2 values.

We can see that modeling ?? 2 as a random variable (red) leads to lower regret compared to constant ?? 2 variants (Figure 2 (b) ).

Note that choosing a small value for ?? 2 (blue) results in near-deterministic posterior function.

Therefore the results are very similar to the -greedy variant.

Intuitively, a deterministic posterior acts as a 0-greedy strategy.

On the other hand, choosing a high value for ?? 2 (purple) results in a very noisy sampling of the posterior approximation, therefore we get a policy which is relatively random concluding in a bad performance.

Choosing ?? 2 with the appropriate size for the given MDP (green, black) results in better performance, as indicated by the lower regret.

However, as ?? 2 is constant it doesn't adapt.

We can see that the regret at the beginning of the learning is better even compared to the adaptive-version.

However, as the uncertainty level decrease, the algorithm "over-explores" which results in inferior regret compared to the adaptive version.

We compared our method with a variant that matches only the expectation, similar to (Levine et al., 2017) , and a variant that does not match the likelihood at all, i.e., approximates the posterior with a fixed arbitrary prior.

The version that does not match the likelihood at all is close to BDQN (Azizzadenesheli et al., 2018) and can be thought of as our implementation for it.

Additionally, we report the results of a variant of the algorithm where the ER buffer is not bounded -this is possible due to the fact that the toy problem is very small and so choosing large enough buffer serves as infinite.

Results are shown in Figure 2 (c).

The superiority of our method (black) over the one-moment method (red) and no-prior at all (blue) support our claim that constructing priors by likelihood matching reduces the forgetting phenomenon.

Additionally, the fact that the unbounded memory algorithm (green) doesn't demonstrate any degradation in performance confirms that this phenomenon is caused since the ER buffer is bounded.

The previous experiment may suggest that catastrophic forgetting in DRLSVI can be avoided by simply increasing the buffer size.

In the following experiment, We examine the simple Chain environment (no self-loop actions; k = 0), with the following modification: we replaced the meaning of the actions in half of the states, i.e., to move right in the odd states, the agent needs to take the opposite action from the even states.

We compare our algorithm with variants that do not match the likelihood with different buffer sizes.

Figure 2 (c) shows the performance of each of the algorithms in this setup.

We can see that our algorithm (blue) doesn't suffer from degradation of performance.

The other algorithms, that don't match the likelihood, all suffer from degradation, where the only difference is the time in which the degradation starts.

These results demonstrate that without the likelihood matching mechanism, catastrophic forgetting will occur regardless of the buffer size.

It is interesting to observe how catastrophic forgetting happens: When the buffer reaches a point where it doesn't contain experience of acting the non-optimal actions, a quick degradation occurs.

Then, the algorithm initially succeeds to re-learn the optimal policy and the regret saturates.

This phenomenon is getting increasingly aggravated until the regret becomes linear.

These chain of events occurred in all the experiments without likelihood matching regardless of the buffer size.

We report the performance of our algorithm across 5 different Atari games.

We trained our algorithm for 10 million time steps and followed the standard evaluation: Every 250k training time steps we evaluated the model for 125k time steps.

Reported measurements are the average episode return during evaluation.

For evaluation, we used the learned Q-network with -greedy policy ( = 0.001), results are averaged across 5 different runs.

We use the original DQN's hyper-parameters.

Hyperparameters that are only relevant for our method are summarised in Appendix 2.

For comparison, we used the publically available learning curves for DQN 1 and Rainbow from the Dopamine framework (Castro et al., 2018) .

Rainbow (Hessel et al., 2018 ) is a complex agent comprised of multiple additions to the original DQN algorithm.

The averaged scores for the three methods are presented in Figure 3 .

The evaluation suggests that our method explores in a much faster rate than DQN, and is competitive with the Rainbow algorithm that combines multiple improvements to the DQN.

Note: Azizzadenesheli et al. (2018) didn't supply standard evaluation metrics and reported results for a single run only.

Additionally, they change the architecture of the Q-network to exclude last layer weights, so a direct comparison to our method is not feasible.

We, therefore, didn't compare our results with theirs.

A Deep Learning adaptation to RLSVI was presented which learn the state representation directly from the data.

We demonstrated the different properties of our method in experiments and showed the promise of our method.

We hope to further reduce the complexity and running time of our algorithm in future work.

The Deep Q-Networks (DQN) algorithm was the first algorithm that successfully combined Deep Learning architectures with Reinforcement Learning algorithms.

It operates in the standard RL setting (Sutton et al., 1998) where the state space is high dimensional.

It approximates the optimal Q-function using a Convolutional Neural Network (CNN).

The algorithm maintains two DNNs, the Q-network with weights ?? and the target network with weights ?? target .

The Q-Network gets a state s as input and produce |A| outputs, each one representing the Q-value of a different action a, Q(s, a).

The target network is an older version of the Q-network with fixed weights.

The target network is used to constructs the targets y that the Q-network is trained to predict.

The targets y are based on the Bellman equation.

The algorithm uses Stochastic Gradient Descent (SGD) to update the network's weights, by minimizing the mean squared error of the Bellman equation defined as E[||Q ?? (s t , a t ) ??? y t || 2 , where the target y t = r t if s t+1 is terminal, otherwise y t = r t + ?? max a Q ??target (s t+1 , a ).

The weights of the target network are set to ?? every fixed number of time steps, T target .

The tuples < s t , a t , r t , s t+1 > that are used to optimize the network's weights are first collected into an Experience Replay (ER) buffer (Lin, 1993) .

When performing an optimization step, a mini-batch of samples are randomly selected from the buffer and are used to calculate the gradients.

DQN is an off-policy algorithm which allows the agent to learn from experience collected by other means rather than its own experience.

To explore the environment it applies the -greedy strategy, i.e., with probability it takes random action and with probability 1??? it takes the greedy action with respect to the current estimate of Q.

Input: Q ?? (s, a), Q ??target (s, a), , ER buffer s 0 = EnvironmentReset() for t = 0, 1, ...

@highlight

A Deep Learning adaptation of Randomized Least Squares Value Iteration
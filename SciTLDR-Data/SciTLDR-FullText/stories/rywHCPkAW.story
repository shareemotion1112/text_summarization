We introduce NoisyNet, a deep reinforcement learning agent with parametric noise added to its weights, and show that the induced stochasticity of the agent’s policy can be used to aid efficient exploration.

The parameters of the noise are learned with gradient descent along with the remaining network weights.

NoisyNet is straightforward to implement and adds little computational overhead.

We find that replacing the conventional exploration heuristics for A3C, DQN and Dueling agents (entropy reward and epsilon-greedy respectively) with NoisyNet yields substantially higher scores for a wide range of Atari games, in some cases advancing the agent from sub to super-human performance.

Despite the wealth of research into efficient methods for exploration in Reinforcement Learning (RL) (Kearns & Singh, 2002; Jaksch et al., 2010) , most exploration heuristics rely on random perturbations of the agent's policy, such as -greedy BID20 or entropy regularisation BID25 , to induce novel behaviours.

However such local 'dithering' perturbations are unlikely to lead to the large-scale behavioural patterns needed for efficient exploration in many environments BID1 .Optimism in the face of uncertainty is a common exploration heuristic in reinforcement learning.

Various forms of this heuristic often come with theoretical guarantees on agent performance BID1 Lattimore et al., 2013; Jaksch et al., 2010; BID0 Kearns & Singh, 2002) .

However, these methods are often limited to small state-action spaces or to linear function approximations and are not easily applied with more complicated function approximators such as neural networks (except from work by BID12 b) but it doesn't come with convergence guarantees).

A more structured approach to exploration is to augment the environment's reward signal with an additional intrinsic motivation term BID19 ) that explicitly rewards novel discoveries.

Many such terms have been proposed, including learning progress (Oudeyer & Kaplan, 2007) , compression progress BID17 , variational information maximisation (Houthooft et al., 2016) and prediction gain BID3 .

One problem is that these methods separate the mechanism of generalisation from that of exploration; the metric for intrinsic reward, and-importantly-its weighting relative to the environment reward, must be chosen by the experimenter, rather than learned from interaction with the environment.

Without due care, the optimal policy can be altered or even completely obscured by the intrinsic rewards; furthermore, dithering perturbations are usually needed as well as intrinsic reward to ensure robust exploration (Ostrovski et al., 2017) .

Exploration in the policy space itself, for example, with evolutionary or black box algorithms (Moriarty et al., 1999; BID9 BID16 , usually requires many prolonged interactions with the environment.

Although these algorithms are quite generic and can apply to any type of parametric policies (including neural networks), they are usually not data efficient and require a simulator to allow many policy evaluations.

We propose a simple alternative approach, called NoisyNet, where learned perturbations of the network weights are used to drive exploration.

The key insight is that a single change to the weight vector can induce a consistent, and potentially very complex, state-dependent change in policy over multiple time steps -unlike dithering approaches where decorrelated (and, in the case of -greedy, state-independent) noise is added to the policy at every step.

The perturbations are sampled from a noise distribution.

The variance of the perturbation is a parameter that can be considered as the energy of the injected noise.

These variance parameters are learned using gradients from the reinforcement learning loss function, along side the other parameters of the agent.

The approach differs from parameter compression schemes such as variational inference (Hinton & Van Camp, 1993; BID7 BID14 BID8 BID11 and flat minima search (Hochreiter & Schmidhuber, 1997) since we do not maintain an explicit distribution over weights during training but simply inject noise in the parameters and tune its intensity automatically.

Consequently, it also differs from Thompson sampling BID22 Lipton et al., 2016) as the distribution on the parameters of our agents does not necessarily converge to an approximation of a posterior distribution.

At a high level our algorithm is a randomised value function, where the functional form is a neural network.

Randomised value functions provide a provably efficient means of exploration (Osband et al., 2014) .

Previous attempts to extend this approach to deep neural networks required many duplicates of sections of the network (Osband et al., 2016) .

By contrast in our NoisyNet approach while the number of parameters in the linear layers of the network is doubled, as the weights are a simple affine transform of the noise, the computational complexity is typically still dominated by the weight by activation multiplications, rather than the cost of generating the weights.

Additionally, it also applies to policy gradient methods such as A3C out of the box (Mnih et al., 2016) .

Most recently (and independently of our work) Plappert et al. (2017) presented a similar technique where constant Gaussian noise is added to the parameters of the network.

Our method thus differs by the ability of the network to adapt the noise injection with time and it is not restricted to Gaussian noise distributions.

We need to emphasise that the idea of injecting noise to improve the optimisation process has been thoroughly studied in the literature of supervised learning and optimisation under different names (e.g., Neural diffusion process (Mobahi, 2016) and graduated optimisation BID15 ).

These methods often rely on a noise of vanishing size that is non-trainable, as opposed to NoisyNet which tunes the amount of noise by gradient descent.

NoisyNet can also be adapted to any deep RL algorithm and we demonstrate this versatility by providing NoisyNet versions of DQN (Mnih et al., 2015) , Dueling BID24 and A3C (Mnih et al., 2016) algorithms.

Experiments on 57 Atari games show that NoisyNet-DQN and NoisyNetDueling achieve striking gains when compared to the baseline algorithms without significant extra computational cost, and with less hyper parameters to tune.

Also the noisy version of A3C provides some improvement over the baseline.

This section provides mathematical background for Markov Decision Processes (MDPs) and deep RL with Q-learning, dueling and actor-critic methods.

MDPs model stochastic, discrete-time and finite action space control problems BID5 BID6 Puterman, 1994 ).

An MDP is a tuple M = (X , A, R, P, γ) where X is the state space, A the action space, R the reward function, γ ∈]0, 1[ the discount factor and P a stochastic kernel modelling the one-step Markovian dynamics (P (y|x, a) is the probability of transitioning to state y by choosing action a in state x).

A stochastic policy π maps each state to a distribution over actions π(·|x) and gives the probability π(a|x) of choosing action a in state x. The quality of a policy π is assessed by the action-value function Q π defined as: DISPLAYFORM0 where E π is the expectation over the distribution of the admissible trajectories (x 0 , a 0 , x 1 , a 1 , . . . ) obtained by executing the policy π starting from x 0 = x and a 0 = a. Therefore, the quantity Q π (x, a) represents the expected γ-discounted cumulative reward collected by executing the policy π starting from x and a. A policy is optimal if no other policy yields a higher return.

The action-value function of the optimal policy is Q (x, a) = arg max π Q π (x, a).The value function V π for a policy is defined as DISPLAYFORM1 , and represents the expected γ-discounted return collected by executing the policy π starting from state x.

Deep Reinforcement Learning uses deep neural networks as function approximators for RL methods.

Deep Q-Networks (DQN) (Mnih et al., 2015) , Dueling architecture BID24 , Asynchronous Advantage Actor-Critic (A3C) (Mnih et al., 2016) , Trust Region Policy Optimisation BID18 , Deep Deterministic Policy Gradient (Lillicrap et al., 2015) and distributional RL (C51) BID4 are examples of such algorithms.

They frame the RL problem as the minimisation of a loss function L(θ), where θ represents the parameters of the network.

In our experiments we shall consider the DQN, Dueling and A3C algorithms.

DQN (Mnih et al., 2015) uses a neural network as an approximator for the action-value function of the optimal policy Q (x, a).

DQN's estimate of the optimal action-value function, Q(x, a), is found by minimising the following loss with respect to the neural network parameters θ: DISPLAYFORM0 where D is a distribution over transitions e = (x, a, r = R(x, a), y ∼ P (·|x, a)) drawn from a replay buffer of previously observed transitions.

Here θ − represents the parameters of a fixed and separate target network which is updated (θ − ← θ) regularly to stabilise the learning.

An -greedy policy is used to pick actions greedily according to the action-value function Q or, with probability , a random action is taken.

The Dueling DQN BID24 is an extension of the DQN architecture.

The main difference is in using Dueling network architecture as opposed to the Q network in DQN.

Dueling network estimates the action-value function using two parallel sub-networks, the value and advantage subnetwork, sharing a convolutional layer.

Let θ conv , θ V , and θ A be, respectively, the parameters of the convolutional encoder f , of the value network V , and of the advantage network A; and θ = {θ conv , θ V , θ A } is their concatenation.

The output of these two networks are combined as follows for every (x, a) ∈ X × A: DISPLAYFORM1 The Dueling algorithm then makes use of the double-DQN update rule to optimise θ: DISPLAYFORM2 DISPLAYFORM3 where the definition distribution D and the target network parameter set θ − is identical to DQN.In contrast to DQN and Dueling, A3C (Mnih et al., 2016 ) is a policy gradient algorithm.

A3C's network directly learns a policy π and a value function V of its policy.

The gradient of the loss on the A3C policy at step t for the roll-out DISPLAYFORM4 (6) H[π(·|x t ; θ)] denotes the entropy of the policy π and β is a hyper parameter that trades off between optimising the advantage function and the entropy of the policy.

The advantage function A(x t+i , a t+i ; θ) is the difference between observed returns and estimates of the return produced by A3C's value network: DISPLAYFORM5 , r t+j being the reward at step t + j and V (x; θ) being the agent's estimate of value function of state x.

The parameters of the value function are found to match on-policy returns; namely we have DISPLAYFORM6 where Q i is the return obtained by executing policy π starting in state x t+i .

In practice, and as in Mnih et al. FORMULA0 , we estimate DISPLAYFORM7 are rewards observed by the agent, and x t+k is the kth state observed when starting from observed state DISPLAYFORM8 where λ balances optimising the policy loss relative to the baseline value function loss.

NoisyNets are neural networks whose weights and biases are perturbed by a parametric function of the noise.

These parameters are adapted with gradient descent.

More precisely, let y = f θ (x) be a neural network parameterised by the vector of noisy parameters θ which takes the input x and outputs y. We represent the noisy parameters θ as θ def = µ + Σ ε, where ζ def = (µ, Σ) is a set of vectors of learnable parameters, ε is a vector of zero-mean noise with fixed statistics and represents element-wise multiplication.

The usual loss of the neural network is wrapped by expectation over the noise ε: DISPLAYFORM0 .

Optimisation now occurs with respect to the set of parameters ζ.

Consider a linear layer of a neural network with p inputs and q outputs, represented by DISPLAYFORM1 where x ∈ R p is the layer input, w ∈ R q×p the weight matrix, and b ∈ R q the bias.

The corresponding noisy linear layer is defined as: DISPLAYFORM2 where DISPLAYFORM3 and σ b ∈ R q are learnable whereas ε w ∈ R q×p and ε b ∈ R q are noise random variables (the specific choices of this distribution are described below).

We provide a graphical representation of a noisy linear layer in Fig. 4 (see Appendix B).We now turn to explicit instances of the noise distributions for linear layers in a noisy network.

We explore two options: Independent Gaussian noise, which uses an independent Gaussian noise entry per weight and Factorised Gaussian noise, which uses an independent noise per each output and another independent noise per each input.

The main reason to use factorised Gaussian noise is to reduce the compute time of random number generation in our algorithms.

This computational overhead is especially prohibitive in the case of single-thread agents such as DQN and Duelling.

For this reason we use factorised noise for DQN and Duelling and independent noise for the distributed A3C, for which the compute time is not a major concern.(a) Independent Gaussian noise: the noise applied to each weight and bias is independent, where each entry ε w i,j (respectively each entry ε b j ) of the random matrix ε w (respectively of the random vector ε b ) is drawn from a unit Gaussian distribution.

This means that for each noisy linear layer, there are pq + q noise variables (for p inputs to the layer and q outputs).(b) Factorised Gaussian noise: by factorising ε w i,j , we can use p unit Gaussian variables ε i for noise of the inputs and and q unit Gaussian variables ε j for noise of the outputs (thus p + q unit Gaussian variables in total).

Each ε w i,j and ε b j can then be written as: DISPLAYFORM4 DISPLAYFORM5 where f is a real-valued function.

In our experiments we used f (x) = sgn(x) |x|.

Note that for the bias Eq. FORMULA0 we could have set f (x) = x, but we decided to keep the same output noise for weights and biases.

Since the loss of a noisy network, DISPLAYFORM6 , is an expectation over the noise, the gradients are straightforward to obtain: DISPLAYFORM7 We use a Monte Carlo approximation to the above gradients, taking a single sample ξ at each step of optimisation: DISPLAYFORM8

We now turn to our application of noisy networks to exploration in deep reinforcement learning.

Noise drives exploration in many methods for reinforcement learning, providing a source of stochasticity external to the agent and the RL task at hand.

Either the scale of this noise is manually tuned across a wide range of tasks (as is the practice in general purpose agents such as DQN or A3C) or it can be manually scaled per task.

Here we propose automatically tuning the level of noise added to an agent for exploration, using the noisy networks training to drive down (or up) the level of noise injected into the parameters of a neural network, as needed.

A noisy network agent samples a new set of parameters after every step of optimisation.

Between optimisation steps, the agent acts according to a fixed set of parameters (weights and biases).

This ensures that the agent always acts according to parameters that are drawn from the current noise distribution.

Deep Q-Networks (DQN) and Dueling.

We apply the following modifications to both DQN and Dueling: first, ε-greedy is no longer used, but instead the policy greedily optimises the (randomised) action-value function.

Secondly, the fully connected layers of the value network are parameterised as a noisy network, where the parameters are drawn from the noisy network parameter distribution after every replay step.

We used factorised Gaussian noise as explained in (b) from Sec. 3.

For replay, the current noisy network parameter sample is held fixed across the batch.

Since DQN and Dueling take one step of optimisation for every action step, the noisy network parameters are re-sampled before every action.

We call the new adaptations of DQN and Dueling, NoisyNet-DQN and NoisyNet-Dueling, respectively.

We now provide the details of the loss function that our variant of DQN is minimising.

When replacing the linear layers by noisy layers in the network (respectively in the target network), the parameterised action-value function Q(x, a, ε; ζ) (respectively Q(x, a, ε ; ζ − )) can be seen as a random variable and the DQN loss becomes the NoisyNet-DQN loss: DISPLAYFORM0 where the outer expectation is with respect to distribution of the noise variables ε for the noisy value function Q(x, a, ε; ζ) and the noise variable ε for the noisy target value function Q(y, b, ε ; ζ − ).

Computing an unbiased estimate of the loss is straightforward as we only need to compute, for each transition in the replay buffer, one instance of the target network and one instance of the online network.

We generate these independent noises to avoid bias due to the correlation between the noise in the target network and the online network.

Concerning the action choice, we generate another independent sample ε for the online network and we act greedily with respect to the corresponding output action-value function.

Similarly the loss function for NoisyNet-Dueling is defined as: DISPLAYFORM1 Both algorithms are provided in Appendix C.1.Asynchronous Advantage Actor Critic (A3C).

A3C is modified in a similar fashion to DQN: firstly, the entropy bonus of the policy loss is removed.

Secondly, the fully connected layers of the policy network are parameterised as a noisy network.

We used independent Gaussian noise as explained in (a) from Sec. 3.

In A3C, there is no explicit exploratory action selection scheme (such as -greedy); and the chosen action is always drawn from the current policy.

For this reason, an entropy bonus of the policy loss is often added to discourage updates leading to deterministic policies.

However, when adding noisy weights to the network, sampling these parameters corresponds to choosing a different current policy which naturally favours exploration.

As a consequence of direct exploration in the policy space, the artificial entropy loss on the policy can thus be omitted.

New parameters of the policy network are sampled after each step of optimisation, and since A3C uses n step returns, optimisation occurs every n steps.

We call this modification of A3C, NoisyNet-A3C.Indeed, when replacing the linear layers by noisy linear layers (the parameters of the noisy network are now noted ζ), we obtain the following estimation of the return via a roll-out of size k: DISPLAYFORM2 As A3C is an on-policy algorithm the gradients are unbiased when noise of the network is consistent for the whole roll-out.

Consistency among action value functionsQ i is ensured by letting letting the noise be the same throughout each rollout, i.e., ∀i, ε i = ε.

Additional details are provided in the Appendix A and the algorithm is given in Appendix C.2.

In the case of an unfactorised noisy networks, the parameters µ and σ are initialised as follows.

Each element µ i,j is sampled from independent uniform distributions U[− DISPLAYFORM0 , where p is the number of inputs to the corresponding linear layer, and each element σ i,j is simply set to 0.017 for all parameters.

This particular initialisation was chosen because similar values worked well for the supervised learning tasks described in BID10 , where the initialisation of the variances of the posteriors and the variances of the prior are related.

We have not tuned for this parameter, but we believe different values on the same scale should provide similar results.

For factorised noisy networks, each element µ i,j was initialised by a sample from an independent uniform distributions U[−

We evaluated the performance of noisy network agents on 57 Atari games BID2 and compared to baselines that, without noisy networks, rely upon the original exploration methods (ε-greedy and entropy bonus).

We used the random start no-ops scheme for training and evaluation as described the original DQN paper (Mnih et al., 2015) .

The mode of evaluation is identical to those of Mnih et al. (2016) where randomised restarts of the games are used for evaluation after training has happened.

The raw average scores of the agents are evaluated during training, every 1M frames in the environment, by suspending We consider three baseline agents: DQN (Mnih et al., 2015) , duel clip variant of Dueling algorithm BID24 and A3C (Mnih et al., 2016) .

The DQN and A3C agents were training for 200M and 320M frames, respectively.

In each case, we used the neural network architecture from the corresponding original papers for both the baseline and NoisyNet variant.

For the NoisyNet variants we used the same hyper parameters as in the respective original paper for the baseline.

We compared absolute performance of agents using the human normalised score: DISPLAYFORM0 where human and random scores are the same as those in BID24 .

Note that the human normalised score is zero for a random agent and 100 for human level performance.

Per-game maximum scores are computed by taking the maximum raw scores of the agent and then averaging over three seeds.

However, for computing the human normalised scores in FIG2 , the raw scores are evaluated every 1M frames and averaged over three seeds.

The overall agent performance is measured by both mean and median of the human normalised score across all 57 Atari games.

The aggregated results across all 57 Atari games are reported in TAB1 , while the individual scores for each game are in FORMULA0 .

We report on the last column the percentage improvement on the baseline in terms of median human-normalised score.without noisy networks: DISPLAYFORM1 As before, the per-game score is computed by taking the maximum performance for each game and then averaging over three seeds.

The relative human normalised scores are shown in FIG1 .

As can be seen, the performance of NoisyNet agents (DQN, Dueling and A3C) is better for the majority of games relative to the corresponding baseline, and in some cases by a considerable margin.

Also as it is evident from the learning curves of FIG2 NoisyNet agents produce superior performance compared to their corresponding baselines throughout the learning process.

This improvement is especially significant in the case of NoisyNet-DQN and NoisyNet-Dueling.

Also in some games, NoisyNet agents provide an order of magnitude improvement on the performance of the vanilla agent; as can be seen in TAB0 in the Appendix E with detailed breakdown of individual game scores and the learning curves plots from Figs 6, 7 and 8, for DQN, Dueling and A3C, respectively.

We also ran some experiments evaluating the performance of NoisyNet-A3C with factorised noise.

We report the corresponding learning curves and the scores in Fig. 5 and TAB4 , respectively (see Appendix D).

This result shows that using factorised noise does not lead to any significant decrease in the performance of A3C.

On the contrary it seems that it has positive effects in terms of improving the median score as well as speeding up the learning process.

In this subsection, we try to provide some insight on how noisy networks affect the learning process and the exploratory behaviour of the agent.

In particular, we focus on analysing the evolution of the noise weights σ w and σ b throughout the learning process.

We first note that, as L(ζ) is a positive and continuous function of ζ, there always exists a deterministic optimiser for the loss L(ζ) (defined in Eq. FORMULA0 ).

Therefore, one may expect that, to obtain the deterministic optimal solution, the neural network may learn to discard the noise entries by eventually pushing σ w s and σ b towards 0.To test this hypothesis we track the changes in σ w s throughout the learning process.

Let σ w i denote the i th weight of a noisy layer.

We then defineΣ, the mean-absolute of the σ w i s of a noisy layer, as DISPLAYFORM0 Intuitively speakingΣ provides some measure of the stochasticity of the Noisy layers.

We report the learning curves of the average ofΣ across 3 seeds in FIG3 for a selection of Atari games in NoisyNet-DQN agent.

We observe thatΣ of the last layer of the network decreases as the learning proceeds in all cases, whereas in the case of the penultimate layer this only happens for 2 games out of 5 (Pong and Beam rider) and in the remaining 3 gamesΣ in fact increases.

This shows that in the case of NoisyNet-DQN the agent does not necessarily evolve towards a deterministic solution as one might have expected.

Another interesting observation is that the wayΣ evolves significantly differs from one game to another and in some cases from one seed to another seed, as it is evident from the error bars.

This suggests that NoisyNet produces a problem-specific exploration strategy as opposed to fixed exploration strategy used in standard DQN.

We have presented a general method for exploration in deep reinforcement learning that shows significant performance improvements across many Atari games in three different agent architectures.

In particular, we observe that in games such as Beam rider, Asteroids and Freeway that the standard DQN, Dueling and A3C perform poorly compared with the human player, NoisyNet-DQN, NoisyNet-Dueling and NoisyNet-A3C achieve super human performance, respectively.

Although the improvements in performance might also come from the optimisation aspect since the cost functions are modified, the uncertainty in the parameters of the networks introduced by NoisyNet is the only exploration mechanism of the method.

Having weights with greater uncertainty introduces more variability into the decisions made by the policy, which has potential for exploratory actions, but further analysis needs to be done in order to disentangle the exploration and optimisation effects.

Another advantage of NoisyNet is that the amount of noise injected in the network is tuned automatically by the RL algorithm.

This alleviates the need for any hyper parameter tuning (required with standard entropy bonus and -greedy types of exploration).

This is also in contrast to many other methods that add intrinsic motivation signals that may destabilise learning or change the optimal policy.

Another interesting feature of the NoisyNet approach is that the degree of exploration is contextual and varies from state to state based upon per-weight variances.

While more gradients are needed, the gradients on the mean and variance parameters are related to one another by a computationally efficient affine function, thus the computational overhead is marginal.

Automatic differentiation makes implementation of our method a straightforward adaptation of many existing methods.

A similar randomisation technique can also be applied to LSTM units BID10 and is easily extended to reinforcement learning, we leave this as future work.

Note NoisyNet exploration strategy is not restricted to the baselines considered in this paper.

In fact, this idea can be applied to any deep RL algorithms that can be trained with gradient descent, including DDPG (Lillicrap et al., 2015) , TRPO BID18 or distributional RL (C51) BID4 .

As such we believe this work is a step towards the goal of developing a universal exploration strategy.

In contrast with value-based algorithms, policy-based methods such as A3C (Mnih et al., 2016) parameterise the policy π(a|x; θ π ) directly and update the parameters θ π by performing a gradient ascent on the mean value-function E x∼D [V π(·|·;θπ) (x)] (also called the expected return) BID21 .

A3C uses a deep neural network with weights θ = θ π ∪θ V to parameterise the policy π and the value V .

The network has one softmax output for the policy-head π(·|·; θ π ) and one linear output for the value-head V (·; θ V ), with all non-output layers shared.

The parameters θ π (resp.

θ V ) are relative to the shared layers and the policy head (resp.

the value head).

A3C is an asynchronous and online algorithm that uses roll-outs of size k + 1 of the current policy to perform a policy improvement step.

For simplicity, here we present the A3C version with only one thread.

For a multi-thread implementation, refer to the pseudo-code C.2 or to the original A3C paper (Mnih et al., 2016) .

In order to train the policy-head, an approximation of the policy-gradient is computed for each state of the roll-out DISPLAYFORM0 whereQ i is an estimation of the returnQ i = DISPLAYFORM1 The gradients are then added to obtain the cumulative gradient of the roll-out: DISPLAYFORM2 A3C trains the value-head by minimising the error between the estimated return and the value DISPLAYFORM3 2 .

Therefore, the network parameters (θ π , θ V ) are updated after each roll-out as follows: DISPLAYFORM4 DISPLAYFORM5 where (α π , α V ) are hyper-parameters.

As mentioned previously, in the original A3C algorithm, it is recommended to add an entropy term β k i=0 ∇ θπ H(π(·|x t+i ; θ π )) to the policy update, where H(π(·|x t+i ; θ π )) = −β a∈A π(a|x t+i ; θ π ) log(π(a|x t+i ; θ π )).

Indeed, this term encourages exploration as it favours policies which are uniform over actions.

When replacing the linear layers in the value and policy heads by noisy layers (the parameters of the noisy network are now ζ π and ζ V ), we obtain the following estimation of the return via a roll-out of size k: DISPLAYFORM6 We would likeQ i to be a consistent estimate of the return of the current policy.

To do so, we should force ∀i, ε i = ε.

As A3C is an on-policy algorithm, this involves fixing the noise of the network for the whole roll-out so that the policy produced by the network is also fixed.

Hence, each update of the parameters (ζ π , ζ V ) is done after each roll-out with the noise of the whole network held fixed for the duration of the roll-out: DISPLAYFORM7 DISPLAYFORM8

In this Appendix we provide a graphical representation of noisy layer.

Figure 4: Graphical representation of a noisy linear layer.

The parameters µ w , µ b , σ w and σ b are the learnables of the network whereas ε w and ε b are noise variables which can be chosen in factorised or non-factorised fashion.

The noisy layer functions similarly to the standard fully connected linear layer.

The main difference is that in the noisy layer both the weights vector and the bias is perturbed by some parametric zero-mean noise, that is, the noisy weights and the noisy bias can be expressed as w = µ w + σ (18) .

In the case of A3C we inculde both factorised and non-factorised variant of the algorithm.

We report on the last column the percentage improvement on the baseline in terms of median human-normalised score.

@highlight

A deep reinforcement learning agent with parametric noise added to its weights can be used to aid efficient exploration.

@highlight

This paper introduces NoisyNets, neural networks whose parameters are perturbed by a parametric noise function, that obtain substantial performance improvement over baseline deep reinforcement learning algorithms.

@highlight

New exploration method for deep RL by injecting noise into deep networks' weights, with the noise taking various forms
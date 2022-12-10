The field of Deep Reinforcement Learning (DRL) has recently seen a surge in the popularity of maximum entropy reinforcement learning algorithms.

Their popularity stems from the intuitive interpretation of the maximum entropy objective and their superior sample efficiency on standard benchmarks.

In this paper, we seek to understand the primary contribution  of the entropy term to the performance of maximum entropy algorithms.

For the Mujoco benchmark, we demonstrate that the entropy term in Soft Actor Critic (SAC) principally addresses the bounded nature of the action spaces.

With this insight, we propose a simple normalization scheme which allows a streamlined algorithm without entropy maximization match the performance of SAC.

Our experimental results demonstrate a need to revisit the benefits of entropy regularization in DRL.

We also propose a simple non-uniform sampling method for selecting transitions from the replay buffer during training.

We further show that the streamlined algorithm with the simple non-uniform sampling scheme outperforms SAC and achieves state-of-the-art performance on challenging continuous control tasks.

Off-policy Deep Reinforcement Learning (RL) algorithms aim to improve sample efficiency by reusing past experience.

Recently a number of new off-policy Deep Reinforcement Learning algorithms have been proposed for control tasks with continuous state and action spaces, including Deep Deterministic Policy Gradient (DDPG) and Twin Delayed DDPG (TD3) (Lillicrap et al., 2015; Fujimoto et al., 2018) .

TD3, which introduced clipped double-Q learning, delayed policy updates and target policy smoothing, has been shown to be significantly more sample efficient than popular on-policy methods for a wide range of Mujoco benchmarks.

The field of Deep Reinforcement Learning (DRL) has also recently seen a surge in the popularity of maximum entropy RL algorithms.

Their popularity stems from the intuitive interpretation of the maximum entropy objective and their superior sample efficiency on standard benchmarks.

In particular, Soft Actor Critic (SAC), which combines off-policy learning with maximum-entropy RL, not only has many attractive theoretical properties, but can also give superior performance on a wide-range of Mujoco environments, including on the high-dimensional environment Humanoid for which both DDPG and TD3 perform poorly (Haarnoja et al., 2018a; b; Langlois et al., 2019) .

SAC has a similar structure to TD3, but also employs maximum entropy reinforcement learning.

In this paper, we first seek to understand the primary contribution of the entropy term to the performance of maximum entropy algorithms.

For the Mujoco benchmark, we demonstrate that when using the standard objective without entropy along with standard additive noise exploration, there is often insufficient exploration due to the bounded nature of the action spaces.

Specifically, the outputs of the policy network are often way outside the bounds of the action space, so that they need to be squashed to fit within the action space.

The squashing results in actions persistently taking on their maximal values, so that there is insufficient exploration.

In contrast, the entropy term in the SAC objective forces the outputs to have sensible values, so that even with squashing, exploration is maintained.

We conclude that the entropy term in the objective for Soft Actor Critic principally addresses the bounded nature of the action spaces in the Mujoco environments.

With this insight, we propose Streamlined Off Policy (SOP), a streamlined algorithm using the standard objective without the entropy term.

SOP employs a simple normalization scheme to address the bounded nature of the action spaces, allowing satisfactory exploration throughout training.

We also consider replacing the aforementioned normalization scheme with inverting gradients (IG) The contributions of this paper are thus threefold.

First, we uncover the primary contribution of the entropy term of maximum entropy RL algorithms when the environments have bounded action spaces.

Second, we propose a streamlined algorithm which do not employ entropy maximization but nevertheless matches the sampling efficiency and robustness performance of SAC for the Mujoco benchmarks.

And third, we combine our streamlined algorithms with a simple non-uniform sampling scheme to achieve state-of-the art performance for the Mujoco benchmarks.

We provide anonymized code for reproducibility 1 .

We represent an environment as a Markov Decision Process (MDP) which is defined by the tuple (S, A, r, p, γ), where S and A are continuous multi-dimensional state and action spaces, r(s, a) is a bounded reward function, p(s |s, a) is a transition function, and γ is the discount factor.

Let s(t) and a(t) respectively denote the state of the environment and the action chosen at time t. Let π = π(a|s), s ∈ S, a ∈ A denote the policy.

We further denote K for the dimension of the action space, and write a k for the kth component of an action a ∈ A, that is, a = (a 1 , . . .

, a K ).

The expected discounted return for policy π beginning in state s is given by:

γ t r(s(t), a(t))|s(0) = s]

Standard MDP and RL problem formulations seek to maximize V π (s) over policies π.

For finite state and action spaces, under suitable conditions for continuous state and action spaces, there exists an optimal policy that is deterministic (Puterman, 2014; Bertsekas & Tsitsiklis, 1996) .

In RL with unknown environment, exploration is required to learn a suitable policy.

In DRL with continuous action spaces, typically the policy is modeled by a parameterized policy network which takes as input a state s and outputs a value µ(s; θ), where θ represents the current parameters of the policy network (Schulman et al., 2015; Vuong et al., 2018; Lillicrap et al., 2015; Fujimoto et al., 2018) .

During training, typically additive random noise is added for exploration, so that the actual action taken when in state s takes the form a = µ(s; θ) + where is a K-dimensional Gaussian random vector with each component having zero mean and variance σ.

During testing, is set to zero.

Maximum entropy reinforcement learning takes a different approach than (1) by optimizing policies to maximize both the expected return and the expected entropy of the policy (Ziebart et al., 2008; Ziebart, 2010; Todorov, 2008; Rawlik et al., 2013; Levine & Koltun, 2013; Levine et al., 2016; Nachum et al., 2017; Haarnoja et al., 2017; 2018a; b) .

In particular, with maximization entropy RL, the objective is to maximize

where H(π(·|s)) is the entropy of the policy when in state s, and the temperature parameter λ determines the relative importance of the entropy term against the reward.

For entropy maximization DRL, when given state s the policy network will typically output a Kdimensional vector σ(s; θ) in addition to the vector µ(s; θ).

The action selected when in state s is then modeled as µ(s; θ) + where ∼ N (0, σ(s; θ)).

Maximum entropy RL has been touted to have a number of conceptual and practical advantages for DRL (Haarnoja et al., 2018a; b) .

For example, it has been argued that the policy is incentivized to explore more widely, while giving up on clearly unpromising avenues.

It has also been argued that the policy can capture multiple modes of near-optimal behavior, that is, in problem settings where multiple actions seem equally attractive, the policy will commit equal probability mass to those actions.

In this paper, we show for the Mujoco benchmarks that the standard additive noise exploration suffices and can achieve the same performance as maximum entropy RL.

3 THE SQUASHING EXPLORATION PROBLEM .

When selecting an action, the action needs to be selected within these bounds before the action can be taken.

DRL algorithms often handle this by squashing the action so that it fits within the bounds.

For example, if along any one dimension the value µ(s; θ) + exceeds a max , the action is set (clipped) to a max .

Alternatively, a smooth form of squashing can be employed.

For example, suppose a min k = −M and a max k = +M for some positive number M , then a smooth form of squashing could use a = M tanh(µ(s; θ) + ) in which tanh() is being applied to each component of the K-dimensional vector.

DDPG (Hou et al., 2017) and TD3 (Fujimoto et al., 2018) use clipping, and SAC (Haarnoja et al., 2018a; b) uses smooth squashing with the tanh() function.

For concreteness, henceforth we will assume that smooth squashing with the tanh() is employed.

We note that an environment may actually allow the agent to input actions that are outside the bounds.

In this case, the environment will typically first clip the actions internally before passing them on to the "actual" environment (Fujita & Maeda, 2018) .

We now make a simple but crucial observation: squashing actions to fit into a bounded action space can have a disastrous effect on additive-noise exploration strategies.

To see this, let the output of the policy network be µ(s) = (µ 1 (s), . . .

, µ K (s)).

Consider an action taken along one dimension k, and suppose µ k (s) >> 1 and | k | is relatively small compared to µ k (s).

Then the action a k = M tanh(µ k (s)+ k ) will be very close (essentially equal) to M .

If the condition µ k (s) >> 1 persists over many consecutive states, then a k will remain close to 1 for all these states, and consequently there will be essentially no exploration along the kth dimension.

We will refer to this problem as the squashing exploration problem.

A similar observation was made in Hausknecht & Stone (2015) .

We will argue that algorithms such as DDPG and TD3 based on the standard objective (1) with additive noise exploration can be greatly impaired by squashing exploration.

SAC is a maximum-entropy based off-policy DRL algorithm which provides good performance across all of the Mujuco benchmark environments.

To the best of our knowledge, it currently provides state of the art performance for the Mujoco benchmark.

In this section, we argue that the principal contribution of the entropy term in the SAC objective is to resolve the squashing exploration problem, thereby maintaining sufficient exploration when facing bounded action spaces.

To argue this, we consider two DRL algorithms: SAC with adaptive temperature (Haarnoja et al., 2018b) , and SAC with entropy removed altogether (temperature set to zero) but everything else the same.

We refer to them as SAC and as SAC without entropy.

For SAC without entropy, for exploration we use additive zero-mean Gaussian noise with σ fixed at 0.3.

Both algorithms use tanh squashing.

We compare these two algorithms on two Mujoco environments: Humanoid-v2 and Walker-v2.

Figure 1 shows the performance of the two algorithms with 10 seeds.

For Humanoid, SAC performs much better than SAC without entropy.

However, for Walker, SAC without entropy performs nearly as well as SAC, implying maximum entropy RL is not as critical for this environment.

To understand why entropy maximization is important for one environment but less so for another, we examine the actions selected when training these two algorithms.

Humanoid and Walker have action dimensions K = 17 and K = 6, respectively.

Here we show representative results for one dimension for both environments, and provide the full results in the Appendix.

The top and bottom rows of Figure 2 shows results for Humanoid and Walker, respectively.

The first column shows the µ k values for an interval of 1,000 consecutive time steps, namely, for time steps 599,000 to 600,000.

The second column shows the actual action values passed to the environment for these time steps.

The third and fourth columns show a concatenation of 10 such intervals of 1000 time steps, with each interval coming from a larger interval of 100,000 time steps.

The top and bottom rows of Figure 2 are strikingly different.

For Humanoid using SAC with entropy, the |µ k | values are small, mostly in the range [-1.5,1.5], and fluctuate significantly.

This allows the action values to also fluctuate significantly, providing exploration in the action space.

On the other hand, for SAC without entropy the |µ k | values are typically huge, most of which are well outside the interval [-10,10] .

This causes the actions a k to be persistently clustered at either M or -M , leading to essentially no exploration along that dimension.

As shown in the Appendix, this property (lack of exploration for SAC without entropy maximization) holds for all 17 action dimensions.

For Walker, we see that for both algorithms, the µ k values are sensible, mostly in the range [-1,1] and therefore the actions chosen by both algorithms exhibit exploration.

In conclusion, the principle benefit of maximum entropy RL in SAC for the Mujuco environments is that it resolves the squashing exploration problem.

For some environments (such as Walker), the outputs of the policy network take on sensible values, so that sufficient exploration is maintained and overall good performance is achieved without the need for entropy maximization.

For other environments (such as Humanoid), entropy maximization is needed to reduce the magnitudes of the outputs so that exploration is maintained and overall good performance is achieved.

Given the observations in the previous section, a natural question is: is it possible to design a streamlined off policy algorithm that does not employ entropy maximization but offers performance comparable to SAC (which has entropy maximization)?

As we observed in the previous section, without entropy maximization, in some environments the policy network output values |µ k |, k = 1, . . .

, K can become persistently huge, which leads to insufficient exploration due to the squashing.

A simple solution is to modify the outputs of the policy network by normalizing the output values when they collectively (across the action dimensions) become too large.

To this end, let µ = (µ 1 , . . .

, µ K ) be the output of the original policy network, and let G = k |µ k |/K. The G is simply the average of the magnitudes of the components of µ. The

otherwise, we leave µ unchanged.

With this simple normalization, we are assured that the average of the normalized magnitudes is never greater than one.

Henceforth we assume the policy network has been modified with the simple normalization scheme just described.

Our Streamlined Off Policy (SOP) algorithm is described in Algorithm 1.

The algorithm is essentially DDPG plus the normalization described above, plus clipped double Q-learning and target policy smoothing (Fujimoto et al., 2018) .

Another way of looking at it is as TD3 plus the normalization described above, minus the delayed policy updates and the target policy parameters.

SOP also uses tanh squashing instead of clipping, since tanh gives somewhat better performance in our experiments.

The SOP algorithm is "streamlined" as it has no entropy terms, temperature adaptation, target policy parameters or delayed policy updates.

In our experiments, we also consider TD3 plus the simple normalization, and also another streamlined algorithm in which we replace the simple normalization scheme described above with the inverting gradients (IG) scheme as described in Hausknecht & Stone (2015) .

The basic idea is: when gradients suggest increasing the action magnitudes, gradients will be downscaled if actions are within the boundaries, and inverted entirely if actions are outside the boundaries.

More implementation details can be found in the Appendix.

Algorithm 1 Streamlined Off-Policy 1: Input: initial policy parameters θ, Q-function parameters φ 1 , φ 2 , empty replay buffer D 2: Set target parameters equal to main parameters φ targ i ← φ i for i = 1, 2 3: repeat

Generate an episode using actions a = M tanh(µ θ (s) + ) where ∼ N (0, σ 1 ).

for j in range(however many updates) do

Randomly sample a batch of transitions, B = {(s, a, r, s)} from D

Compute targets for Q functions:

Update Q-functions by one step of gradient descent using

Update policy by one step of gradient ascent using

Update target networks with

Figure 3 compares SAC (with temperature adaptation (Haarnoja et al., 2018a; b) ) with SOP, TD3+ (that is, TD3 plus the simple normalization), and inverting gradients (IG) for five of the most chal-lenging Mujuco environments.

Using the same baseline code, we train with ten different random seeds for each of the two algorithms.

Each algorithm performs five evaluation rollouts every 5000 environment steps.

The solid curves correspond to the mean, and the shaded region to the standard deviation of the returns over the ten seeds.

Results show that SOP, SAC and IG have similar sample-efficiency performance and robustness across all environments.

TD3+ has slightly weaker asymptotic performance for Walker and Humanoid.

IG initially learns slowly for Humanoid with high variance across random seeds, but gives similar asymptotic performance.

This confirms that with a simple output normalization scheme in the policy network, the performance of SAC can be achieved without maximum entropy RL.

In the Appendix we provide an ablation study for SOP, which shows a major performance drop when removing either double Q-learning or normalization, whereas removing target policy smoothing (Fujimoto et al., 2018 ) results in only a small performance drop in some environments.

We now show how a small change in the sampling scheme for SOP can achieve state of the art performance for the Mujoco benchmark.

We call this sampling scheme Emphasizing Recent Experience (ERE).

ERE has 3 core features: (i) It is a general method applicable to any off-policy algorithm; (ii) It requires no special data structure, is very simple to implement, and has near-zero computational overhead; (iii) It only introduces one additional important hyper-parameter.

The basic idea is: during the parameter update phase, the first mini-batch is sampled from the entire buffer, then for each subsequent mini-batch we gradually reduce our range of sampling to sample more aggressively from more recent data.

Specifically, assume that in the current update phase we are to make 1000 mini-batch updates.

Let N be the max size of the buffer.

Then for the k th update, we sample uniformly from the most recent c k data points, where c k = N · η k and η ∈ (0, 1] is a hyper-parameter that determines how much emphasis we put on recent data.

η = 1 is uniform sampling.

When η < 1, c k decreases as we perform each update.

η can made to adapt to the learning speed of the agent so that we do not have to tune it for each environment.

The effect of such a sampling formulation is twofold.

The first is recent data have a higher chance of being sampled.

The second is that we do this in an ordered way: we first sample from all the data in the buffer, and gradually shrink the range of sampling to only sample from the most recent data.

This scheme reduces the chance of over-writing parameter changes made by new data with parameter changes made by old data (French, 1999; McClelland et al., 1995; McCloskey & Cohen, 1989; Ratcliff, 1990; Robins, 1995) .

This process allows us to quickly obtain new information from recent data, and better approximate the value functions near recently-visited states, while still maintaining an acceptable approximation near states visited in the more distant past.

What is the effect of replacing uniform sampling with ERE?

First note if we do uniform sampling on a fixed buffer, the expected number of times a data point is sampled is the same for all data points.

Now consider a scenario where we have a buffer of size 1000 (FIFO queue), we collect one data at a time, and then perform one update with mini-batch size of one.

If we start with an empty buffer and sample uniformly, as data fills the buffer, each data point gets less and less chance of being sampled.

Specifically, over a period of 1000 updates, the expected number of times the tth data is sampled is: 1/t + 1/(t + 1) + · · · + 1/T .

Figure 4f shows the expected number of times a data is sampled as a function of its position in the buffer.

We see that older data are expected to get sampled much more than newer data.

This is undesirable because when the agent is improving and exploring new areas of the state space; new data points may contain more interesting information than the old ones, which have already been updated many times.

When we apply the ERE scheme, we effectively skew the curve towards assigning higher expected number of samples for the newer data, allowing the newer data to be frequently sampled soon after being collected, which can accelerate the learning process.

In the Appendix, we provide further algorithmic detail and analysis on ERE, and compare ERE to two other sampling schemes: an exponential sampling scheme and Prioritized Experience Replay .

Figure 4 compares the performance of SOP, SOP+ERE, SAC and SAC+ERE.

With ERE, both SAC and SOP gain a significant performance improvement in all environments.

SOP+ERE learns faster than SAC and vanilla SOP in all Mujoco environments.

SOP+ERE also greatly improves overall performance for the two most challenging environments, Ant and Humanoid, and has the best performance for Humanoid.

In table 1, we show the mean test episode return and std across 10 random seeds at 1M timesteps for all environments.

The last column displays the percentage improvement of SOP+ERE over SAC, showing that SOP+ERE achieves state of the art performance.

In Ant and Humanoid, SOP+ERE improves performance by 21% and 24% over SAC at 1 million timesteps, respectively.

As for the std, SOP+ERE gives lower values, and for Humanoid a higher value.

ERE allows new data to be sampled many times soon after being collected.

In recent years, there has been significant progress in improving the sample efficiency of DRL for continuous robotic locomotion tasks with off-policy algorithms (Lillicrap et al., 2015; Fujimoto et al., 2018; Haarnoja et al., 2018a; b) .

There is also a significant body of research on maximum entropy RL methods (Ziebart et al., 2008; Ziebart, 2010; Todorov, 2008; Rawlik et al., 2013; Levine & Koltun, 2013; Levine et al., 2016; Nachum et al., 2017; Haarnoja et al., 2017; 2018a; b) .

Uniform sampling is the most common way to sample from a replay buffer.

One of the most wellknown alternatives is prioritized experience replay (PER) .

PER uses the absolute TD-error of a data point as the measure for priority, and data points with higher priority will have a higher chance of being sampled.

This method has been tested on DQN (Mnih et al., 2015) and double DQN (DDQN) (Van Hasselt et al., 2016 ) with significant improvement and applied successfully in other algorithms (Wang et al., 2015; Schulze & Schulze, 2018; Hessel et al., 2018; Hou et al., 2017) and can be implemented in a distributed manner (Horgan et al., 2018) .

There are other methods proposed to make better use of the replay buffer.

The ACER algorithm has an on-policy part and an off-policy part, with a hyper-parameter controlling the ratio of off-policy to on-policy updates (Wang et al., 2016) .

The RACER algorithm (Novati & Koumoutsakos, 2018) selectively removes data points from the buffer, based on the degree of "off-policyness", bringing improvement to DDPG (Lillicrap et al., 2015) , NAF (Gu et al., 2016) and PPO (Schulman et al., 2017) .

In De Bruin et al. (2015), replay buffers of different sizes were tested, showing large buffer with data diversity can lead to better performance.

Finally, with Hindsight Experience Replay (Andrychowicz et al., 2017) , priority can be given to trajectories with lower density estimation (Zhao & Tresp, 2019) to tackle multi-goal, sparse reward environments.

In this paper we first showed that the primary role of maximum entropy RL for the Mujoco benchmark is to maintain satisfactory exploration in the presence of bounded action spaces.

We then developed a new streamlined algorithm which does not employ entropy maximization but nevertheless matches the sampling efficiency and robustness performance of SAC for the Mujoco benchmarks.

Our experimental results demonstrate a need to revisit the benefits of entropy regularization in DRL.

Finally, we combined our streamlined algorithm with a simple non-uniform sampling scheme to achieve state-of-the art performance for the Mujoco benchmark.

In this ablation study we separately examine the importance of (i) the normalization at the output of the policy network; (ii) the double Q networks; (iii) and randomization used in the line 8 of the SOP algorithm (that is, target policy smoothing (Fujimoto et al., 2018) ).

Figure 5 shows the results for the five environments considered in this paper.

In Figure 5 , "no normalization" is SOP without the normalization of the outputs of the policy network; "single Q" is SOP with one Q-network instead of two; and "no smoothing" is SOP without the randomness in line 8 of the algorithm.

Figure 5 confirms that double Q-networks are critical for obtaining good performance (Van Hasselt et al., 2016; Fujimoto et al., 2018; Haarnoja et al., 2018a ).

Figure 5 also shows that output normalization is also critical.

Without output normalization, performance fluctuates wildly, and average performance can decrease dramatically, particularly for Humanoid and HalfCheetah.

Target policy smoothing improves performance by a relatively small amount.

Table 2 shows hyperparameters used for SOP, SOP+ERE and SOP+PER.

For adaptive SAC, we use our own PyTorch implementation for the comparisons.

Our implementation uses the same hyperparameters as used in the original paper (Haarnoja et al., 2018b) .

Our implementation of SOP variants and adaptive SAC share most of the code base.

For TD3, our implementation uses the same hyperparamters as used in the authors' implementation, which is different from the ones in the original paper (Fujimoto et al., 2018) .

They claimed that the new set of hyperparamters can improve performance for TD3.

We now discuss hyperparameter search for better clarity, fairness and reproducibility (Henderson et al., 2018; Duan et al., 2016; Islam et al., 2017) .

For the η value in the ERE scheme, in our early experiments we tried the values (0.993, 0.994, 0.995, 0.996, 0.997, 0.998) on the Ant and found 0.995 to work well.

This initial range of values was decided by computing the ERE sampling range for the oldest data.

We found that for smaller values, the range would simply be too small.

For the PER scheme, we did some informal preliminary search, then searched on Ant for β 1 in (0, 0.4, 0.6, 0.8), β 2 in (0, 0.4, 0.5, 0.6, 1), and learning rate in (1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3), we decided to search these values because the original paper used β 1 = 0.6, β 2 = 0.4 and with reduced learning rate.

For the exponential sampling scheme, we searched the λ value in (3e-7, 1e-6, 3e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4) in Ant, this search range was decided by plotting out the probabilities of sampling, and then pick a set of values that are not too extreme.

For σ in SOP, in some of our early experiments with SAC, we accidentally found that σ = 0.3 gives good performance for SAC without entropy and with Gaussian noise.

We searched values ( for HalfCheetah-v2) SOP gaussian noise std σ = σ 1 = σ 2 0.29 TD3 gaussian noise std for data collection σ 0.1 * action limit guassian noise std for target policy smoothingσ 0.2 TD3+ gaussian noise std for data collection σ 0.15 guassian noise std for target policy smoothingσ 0.2 ERE ERE initial η 0 0.995

Algorithm 2 SOP with Emphasizing Recent Experience 1: Input: initial policy parameters θ, Q-function parameters φ 1 , φ 2 , empty replay buffer D of size N , initial η 0 , recent and max performance improvement I recent = I max = 0.

2: Set target parameters equal to main parameters φ targ,i ← φ i for i = 1, 2 3: repeat 4:

Generate an episode using actions a = M tanh(µ θ (s) + ) where ∼ N (0, σ 1 ).

update I recent , I max with training episode returns, let K = length of episode 6:

for j in range(K) do 8:

Sample a batch of transitions, B = {(s, a, r, s)} from most recent c k data in D 10:

Compute targets for Q functions:

Update Q-functions by one step of gradient descent using

Update policy by one step of gradient ascent using

Update target networks with

In this section we discuss the details of the Inverting Gradient method.

Hausknecht & Stone (2015) discussed three different methods for bounded parameter space learning: Zeroing Gradients, Squashing Gradients and Inverting Gradients, they analyzed and tested the three methods and found that Inverting Gradients method can achieve much stronger performance than the other two.

In our implementation, we remove the tanh function from SOP and use Inverting Gradients instead to bound the actions.

Let p indicate the output of the last layer of the policy network.

During exploration p will be the mean of a normal distribution that we sample actions from, the IG approach can be summarized by the following equation (Hausknecht & Stone, 2015) :

Where ∇ p is the gradient of the policy loss w.r.t to p. During a policy network update, we first backpropagate the gradients from the outputs of the Q network to the output of the policy network for each data point in the batch, we then compute the ratio

for each p value (each action dimension), depending on the sign of the gradient.

We then backpropagate from the output of the policy network to parameters of the policy network, and we modify the gradients in the policy network according to the ratios we computed.

We made an efficient implementation and further discuss the computation efficiency of IG in the implementation details section.

We also investigate the effect of other interesting sampling schemes.

We also implement the proportional variant of Prioritized Experience Replay with SOP.

Since SOP has two Q-networks, we redefine the absolute TD error |δ| of a transition (s, a, r, s ) to be the average absolute TD error in the Q network update:

Within the sum, the first term y q (r,

is simply the target for the Q network, and the term Q θ,l (s, a) is the current estimate of the l th Q network.

For the i th data point, the definition of the priority value p i is p i = |δ i | + .

The probability of sampling a data point P (i) is computed as:

where β 1 is a hyperparameter that controls how much the priority value affects the sampling probability, which is denoted by α in , but to avoid confusion with the α in SAC, we denote it as β 1 .

The importance sampling (IS) weight w i for a data point is computed as:

where β 2 is denoted as β in .

Based on the SOP algorithm, we change the sampling method from uniform sampling to sampling using the probabilities P (i), and for the Q updates we apply the IS weight w i .

This gives SOP with Prioritized Experience Replay (SOP+PER).

We note that as compared with SOP+PER, ERE does not require a special data structure and has negligible extra cost, while PER uses a sum-tree structure with some additional computational cost.

We also tried several variants of SOP+PER, but preliminary results show that it is unclear whether there is improvement in performance, so we kept the algorithm simple.

The ERE scheme is similar to an exponential sampling scheme where we assign the probability of sampling according to the probability density function of an exponential distribution.

Essentially, in such a sampling scheme, the more recent data points get exponentially more probability of being sampled compared to older data.

For the i th most recent data point, the probability of sampling a data point P (i) is computed as:

We apply this sampling scheme to SOP and refer to this variant as SOP+EXP.

Figure 6 shows a performance comparison of SOP, SOP+ERE, SOP+EXP and SOP+PER.

Results show that the exponential sampling scheme gives a boost to the performance of SOP, and especially in the Humanoid environment, although not as good as ERE.

Surprisingly, SOP+PER does not give a significant performance boost to SOP (if any boost at all).

We also found that it is difficult to find hyperparameter settings for SOP+PER that work well for all environments.

Some of the other hyperparameter settings actually reduce performance.

It is unclear why PER does not work so well for SOP.

A similar result has been found in another recent paper (Fu et al., 2019) , showing that PER can significantly reduce performance on TD3.

Further research is needed to understand how PER can be successfully adapted to environments with continuous action spaces and dense reward structure.

Figure 7 shows, for fixed η, how η affects the data sampling process, under the ERE sampling scheme.

Recent data points have a much higher probability of being sampled compared to older data, and a smaller η value gives more emphasis to recent data.

Different η values are desirable depending on how fast the agent is learning and how fast the past experiences become obsolete.

So to make ERE work well in different environments with different reward scales and learning progress, we adapt η to the the speed of learning.

To this end, define performance to be the training episode return.

Define I recent to be how much performance improved from N/2 timesteps ago, and I max to be the maximum improvement throughout training, where N is the buffer size.

Let the hyperparameter η 0 be the initial η value.

We then adapt η according to the formula: η = η 0 · I recent /I max + 1 − (I recent /I max ).

Under such an adaptive scheme, when the agent learns quickly, the η value is low in order to learn quickly from new data.

When progress is slow, η is higher to make use of the stabilizing effect of uniform sampling from the whole buffer.

Figure 7b plots the expected number of times a data point in the buffer is sampled, with the data points ordered from most to least recent.

In this section we discuss some programming details.

These details are not necessary for understanding the algorithm, but they might help with reproducibility.

In the ERE scheme, the sampling range always starts with the entire buffer (1M data) and then gradually shrinks.

This is true even when the buffer is not full.

So even if there are not many data points in the buffer, we compute c k based as if there are 1M data points in the buffer.

One can also modify the design slightly to obtain a variant that uses the current amount of data points to compute c k .

In addition to the reported scheme, we also tried shrinking the sampling range linearly, but it gives less performance gain.

In our implementation we set the number of updates after an episode to be the same as the number of timesteps in that episode.

Since environments do not always end at 1000 timesteps, we can give a more general formula for c k .

Let K be the number of mini-batch updates, let N be the max size of the replay buffer, then:

With this formulation, the range of sampling shrinks in more or less the same way with varying number of mini-batch updates.

We always do uniform sampling in the first update, and we always have η

When η is small, c k can also become small for some of the mini-batches.

To prevent getting a minibatch with too many repeating data points, we set the minimum value for c k to 5000.

We did not find this value to be too important and did not find the need to tune it.

It also does not have any effect for any η ≥ 0.995 since the sampling range cannot be lower than 6000.

In the adaptive scheme with buffer of size 1M, the recent performance improvement is computed as the difference of the current episode return compared to the episode return 500,000 timesteps earlier.

Before we reach 500,000 timesteps, we simply use η 0 .

The exact way of computing performance improvement does not have a significant effect on performance as long as it is reasonable.

In this section we give analysis on the additional programming and computation complexity brought by ERE and PER.

In terms of programming complexity, ERE is a clear winner since it only requires a small adjustment to how we sample mini-batches.

It does not modify how the buffer stores the data, and does not require a special data structure to make it work efficiently.

Thus the implementation difficulty is minimal.

PER (proportional variant) requires a sum-tree data structure to make it run efficiently.

The implementation is not too complicated, but compared to ERE it is a lot more work.

The exponential sampling scheme is very easy to implement, although a naive implementation will incur a significant computation overhead when sampling from a large buffer.

To improve its computation efficiency, we instead uses an approximate sampling method.

We first sample data indexes from segments of size 100 from the replay buffer, and then for each segment sampled, we sample one data point uniformly from that segment.

In terms of computation complexity (not sample efficiency), and wall-clock time, ERE's extra computation is negligible.

In practice we observe no difference in computation time between SOP and SOP+ERE.

PER needs to update the priority of its data points constantly and compute sampling probabilities for all the data points.

The complexity for sampling and updates is O(log(N )), and the rank-based variant is similar .

Although this is not too bad, it does impose a significant overhead on SOP: SOP+PER runs twice as long as SOP.

Also note that this overhead grows linearly with the size of the mini-batch.

The overhead for the Mujoco environments is higher compared to Atari, possibly because the Mujoco environments have a smaller state space dimension while a larger batch size is used, making PER take up a larger portion of computation cost.

For the exponential sampling scheme, the extra computation is also close to negligible when using the approximate sampling method.

In terms of the proposed normalization scheme and the Inverting Gradients (IG) method, the normalization is very simple and can be easily implemented and has negligible computation overhead.

IG has a simple idea, but its implementation is slightly more complicated than the normalization scheme.

When implemented naively, IG can have a large computation overhead, but it can be largely avoided by making sure the gradient computation is still done in a batch-manner.

We have made a very efficient implementation and our code is publicly available so that interested reader can easily reproduce it.

In Figure 8 we show additional results on applying ERE to SOP+IG.

The result shows that after applying the ERE scheme, SOP and IG both get a performance boost.

The performance of the SOP+ERE and IG+ERE are similar.

In figure 9 , we show additional results comparing TD3 with TD3 plus our normalization scheme, which we refer as TD3+.

The results show that after applying our normalization scheme, TD3+ has a significant performance boost in Humanoid, while in other environments, both algorithms achieve similar performance.

To understand why entropy maximization is important for one environment but less so for another, we examine the actions selected when training SAC with and without entropy.

Humanoid and Walker2d have action dimensions K = 17 and K = 6, respectively.

In addition to the representative results shown for one dimension for both environments in Section 3.2, the results for all the dimensions are provided here in Figures 10 and 11 .

From Figure 10 , we see that for Humanoid using SAC (which uses entropy maximization), the |µ k | values are small and fluctuate significantly for all 17 dimensions.

On the other hand, for SAC without entropy the |µ k | values are typically huge, again for all 17 dimensions.

This causes the actions a k to be persistently clustered at either M or -M .

As for Walker, the |µ k | values are sensible for both algorithms for all 6 dimensions, as shown in figure 11 .

<|TLDR|>

@highlight

We propose a new DRL off-policy algorithm achieving state-of-the-art performance. 
Deep reinforcement learning algorithms that estimate state and state-action value functions have been shown to be effective in a variety of challenging domains, including learning control strategies from raw image pixels.

However, algorithms that estimate state and state-action value functions typically assume a fully observed state and must compensate for partial or non-Markovian observations by using finite-length frame-history observations or recurrent networks.

In this work, we propose a new deep reinforcement learning algorithm based on counterfactual regret minimization that iteratively updates an approximation to a cumulative clipped advantage function and is robust to partially observed state.

We demonstrate that on several partially observed reinforcement learning tasks, this new class of algorithms can substantially outperform strong baseline methods: on Pong with single-frame observations, and on the challenging Doom (ViZDoom) and Minecraft (Malmö) first-person navigation benchmarks.

Many reinforcement learning problems of practical interest have the property of partial observability, where observations of state are generally non-Markovian.

Despite the importance of partial observation in the real world, value function-based methods such as Q-learning (Mnih et al., 2013; BID6 generally assume a Markovian observation space.

On the other hand, Monte Carlo policy gradient methods do not assume Markovian observations, but many practical policy gradient methods such as A3C (Mnih et al., 2016) introduce the Markov assumption when using a critic or state-dependent baseline in order to improve sample efficiency.

Consider deep reinforcement learning methods that learn a state or state-action value function.

One common workaround for the problem of partial observation is to learn value functions on the space of finite-length frame-history observations, under the assumption that frame-histories of sufficient length will give the environment the approximate appearance of full observability.

When learning to play Atari 2600 games from images, deep Q-learning algorithms (Mnih et al., 2013; BID6 concatenate the last 4 observed frames of the video screen buffer as input to a state-action value convolutional network.

Not all non-Markovian tasks are amenable to finite-length frame-histories; recurrent value functions can incorporate longer and potentially infinite histories BID12 BID8 , but at the cost of solving a harder optimization problem.

Can we develop methods that learn a variant of the value function that is more robust to partial observability?Our contribution is a new model-free deep reinforcement learning algorithm based on the principle of regret minimization which does not require access to a Markovian state.

Our method learns a policy by estimating a cumulative clipped advantage function, which is an approximation to a type of regret that is central to two partial information game-solving algorithms from which we draw our primary inspiration: counterfactual regret minimization (CFR) BID35 and CFR+ BID28 .

Hence we call our algorithm "advantage-based regret minimization" (ARM).We evaluate our approach on three visual reinforcement learning domains: Pong with varying framehistory lengths BID2 , and the first-person games Doom BID16 and Minecraft BID15 .

Doom and Minecraft exhibit a first-person viewpoint in a 3-dimensional environment and should appear non-Markovian even with frame-history observations.

We find that our method offers substantial improvement over prior methods in these partially observ-able environments: on both Doom and Minecraft, our method can learn well-performing policies within about 1 million simulator steps using only visual input frame-history observations.

Deep reinforcement learning algorithms have been demonstrated to achieve excellent results on a range of complex tasks, including playing games (Mnih et al., 2015; BID20 and continuous control BID24 BID18 .

Prior deep reinforcement learning algorithms either learn state or state-action value functions (Mnih et al., 2013) , learn policies using policy gradients BID24 , or perform a combination of the two using actor-critic architectures (Mnih et al., 2016) .

Policy gradient methods typically do not need to assume a Markovian state, but tend to suffer from poor sample complexity, due to their inability to use off-policy data.

Methods based on learning Q-functions can use replay buffers to include off-policy data, accelerating learning BID18 .

However, learning Q-functions with Bellman error minimization typically requires a Markovian state space.

When learning from observations such as images, the inputs might not be Markovian.

Prior methods have proposed to mitigate this issue by using recurrent critics and Q-functions BID12 BID20 Mnih et al., 2016; BID13 , and learning Q-functions that depend on entire histories of observations.

Heuristics such as concatenation of short observation sequences have also been used (Mnih et al., 2015) .

However, all of these changes increase the size of the input space, increasing variance, and make the optimization problem more complex.

Our method instead learns cumulative advantage functions that depend only on the current state, but can still handle non-Markovian problems.

The form of our advantage function update resembles positive temporal difference methods BID21 BID30 .

Additionally, our update rule for a modified cumulative Q-function resembles the average Q-function BID1 used for variance reduction in Q-learning.

In both cases, the theoretical foundations of our method are based on cumulative regret minimization, and the motivation is substantively different.

Previous work by BID23 BID22 has connected regret minimization to reinforcement learning, imitation learning, and structured prediction, although not with counterfactual regret minimization.

Regression regret matching (Waugh et al., 2015) is based on a closely related idea, which is to directly approximate the regret with a linear regression model, however the use of a linear model is limited in representation compared to deep function approximation.

In this section, we provide background on CFR and CFR+, describe ARM in detail, and give some intuition for why ARM works.

In this section we review the algorithm of counterfactual regret minimization BID35 .

We closely follow the version of CFR as described in the Supplementary Material of , except that we try to use the notation of reinforcement learning where appropriate.

Consider the setting of an extensive game.

There are N players numbered i = 1, . . .

, N .

An additional player may be considered a "chance" player to simulate random events.

At each time step of the game, one player chooses an action a ∈ A i .

Define the following concepts and notation:• Sequences: A sequence specifically refers to a sequence of actions starting from an initial game state.

(It is assumed that a sequence of actions, including actions of the "chance" player, is sufficient for defining state within the extensive game.)

Let H be the space of all sequences, and let Z be the space of terminal sequences.• Information sets:

Let I be the space of information sets; that is, for each I ∈ I, I is a set of sequences h ∈ I which are indistinguishable to the current player.

Information sets are a represention of partial observability.

• Strategies: Let π i (a|I) be the strategy of the i-th player, where π i (a|I) is a probability distribution over action a conditioned on information set I. Let π = (π 1 , . . .

, π N ) denote the strategy profile for all players, and let π −i = (π 1 , . . .

, π i−1 , π i+1 , . . .

, π N ) denote the strategy profile for all players except the i-th player.• Sequence probabilities: Let ρ π (h) be the probability of reaching the sequence h when all players follow π.

Additionally, let ρ π (h, h ) be the probability of reaching h conditioned on h having already been reached.

Similarly, define ρ π i and ρ π −i to contain the contributions of respectively only the i-th player or of all players except the i-th.• Values:

Let u i (z) be the value of a terminal sequence z to the i-th player.

Let the expected value of a strategy profile π to the i-th player be DISPLAYFORM0 Define the counterfactual value Q CF π,i of all players following strategy π, except the i-th player plays to reach information set I and to then take action a: DISPLAYFORM1 The notation h h denotes that h is a prefix of h , while π|I → a denotes that action a is to be performed when I is observed.

The counterfactual value Q CF π,i (I, a) is a calculation that assumes the i-th player reaches any h ∈ I, and upon reaching any h ∈

I it always chooses a.

Consider a learning scenario where at the t-th iteration the players follow a strategy profile π t .

The i-th player's regret after T iterations is defined in terms of the i-th player's optimal strategy π * i : DISPLAYFORM2 The average regret is the average over learning iterations: (1/T )R T i .

Now define the counterfactual regret of the i-th player for taking action a at information set I: DISPLAYFORM3 The counterfactual regret (Equation (3)) can be shown to majorize the regret (Equation (2)) (Theorem 3, BID35 ).

CFR can then be described as a learning algorithm where the strategy is updated using regret matching BID11 applied to the counterfactual regret calculated in the most recent iteration: DISPLAYFORM4 If all players follow the CFR regret matching strategy (Equation FORMULA4 ), then at the T -th iteration the players' average regrets are bounded by O(T −1/2 ) (Theorem 4, BID35 ).

CFR+ BID28 consists of a modification to CFR, in which instead of calculating the full counterfactual regret as in (4), instead the counterfactual regret is recursively positively clipped to yield the clipped counterfactual regret: DISPLAYFORM0 Comparing Equation (4) with Equation (6), one can see that the only difference in CFR is that the previous iteration's counterfactual regret is positively clipped in the recursion.

The one-line change of CFR+ turns out to yield a large practical improvement in the performance of the algorithm , and there is also an associated regret bound for CFR+ that is as strong as the bound for CFR .

CFR and CFR+ are formulated for imperfect information extensive-form games, so they are naturally generalized to partially observed stochastic games since a stochastic game can always be represented in extensive form.

A 1-player partially observed stochastic game is simply a POMDP with observation space O BID19 .

By mapping information sets I ∈ I to observations o ∈ O, we may rewrite the counterfactual value as a kind of stationary observation-action value DISPLAYFORM0 π|o →a (o, a) that assumes the agent follows the policy π except on observing o, after which the action a is always performed BID3 .

We posit that the approximation Q (stat) DISPLAYFORM1 , where Q π is the usual action value function, is valid when observations are rarely seen more than once in a trajectory.

By approximating Q (stat) DISPLAYFORM2 , we get a recurrence in terms of more familiar value functions (compare Equations FORMULA5 and FORMULA9 ): DISPLAYFORM3 = max(0,Ā DISPLAYFORM4 is the cumulative clipped advantage function, and A πt (o, a) is the ordinary advantage function evaluated at policy π t .

Advantage-based regret minimization (ARM) is the resulting reinforcement learning algorithm that updates the policy to regret match on the cumulative clipped advantage function: DISPLAYFORM5 (10)Equations (9) and FORMULA1 suggest the outline of a batch-mode deep reinforcement learning algorithm.

At the t-th sampling iteration, a batch of data is collected by sampling trajectories using the current policy π t , followed by two processing steps: (a) fitĀ + t using Equation (9), then (b) set the next iteration's policy π t+1 using Equation (10).

To implement Equation (9) with deep function approximation, we define two value function approximations, DISPLAYFORM0 , as well as a target value function V (o k ; ϕ), where θ t , ω t , and ϕ are the learnable parameters.

The cumulative clipped advantage function is represented as A DISPLAYFORM1 Within each sampling iteration, the value functions are fitted using stochastic gradient descent by sampling minibatches and performing gradient steps.

The state-value function V πt (o k ; θ t ) is fit to minimize an n-step temporal difference loss with a moving target V (o k+n ; ϕ), essentially using the estimator of the deep deterministic policy gradient (DDPG) BID18 .

In the same minibatch,Q + t (o k , a k ; θ t ) is fit to a similar loss, but with an additional target reward bonus that incorporates the previous iteration's cumulative clipped advantage, max(0,Ā DISPLAYFORM2 terms of the n-step returns g DISPLAYFORM3 Altogether, each minibatch step of the optimization subproblem consists of the following three parameter updates in terms of the regression targets v(o k ; ϕ) andq DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 ϕ) update θ t with step size α and targets v(o k ) (Equation FORMULA1 ) update ω t with step size α and targetsq FORMULA1 ) update ϕ with moving average step size τ (Equation FORMULA1 DISPLAYFORM7 DISPLAYFORM8 The overall advantage-based regret minimization algorithm is summarized in Algorithm 1.We note that the mechanics of the ARM updates are similar to on-policy value function estimation, but ARM learns a modified on-policy Q-function from transitions with the added reward bonus FORMULA1 ).

This reward bonus can be thought of a kind of "optimism in the face of uncertainty." DISPLAYFORM9

In this section, we accentuate that ARM represents an inherently different update compared to existing policy gradient methods.

Recent work has shown that policy gradient methods and Q-learning methods are connected via entropy regularization (O'Donoghue et al., 2017; BID10 Nachum et al., 2017; BID26 BID0 .

One perspective is from the soft policy iteration framework for batch-mode reinforcement learning BID0 , where at each batch iteration the updated policy is obtained by minimizing the average KL-divergence between the policy class Π and a target policy f .

Below is the soft policy iteration update, where the subscript t refers to the batch iteration: DISPLAYFORM0 = arg min DISPLAYFORM1 Using the connection between policy gradient methods and Q-learning, we define the policy gradient target policy as the softmax distribution on the entropy regularized advantage function A β-soft : DISPLAYFORM2 We note that it is more conventional in the literature to use the soft Q-function Q β-soft (o, a) rather than the soft advantage function A β-soft (o, a), however since they differ only by a function of o then they both induce the same target softmax policy.

Now, parameterizing the policy π in terms of an explicit parameter θ, we obtain the expression for the existing policy gradient, where b(o) is a baseline function: DISPLAYFORM3 The classic policy gradient arises in the limit β → ∞.Note that an alternative choice of target policy f will lead to a different kind of policy gradient update.

A policy gradient algorithm based on ARM instead proposes the following target policy based on the regret-matching distribution: DISPLAYFORM4 Similarly, we can express the ARM-like policy gradient, where again b(o) is a baseline: DISPLAYFORM5 Comparing Equations FORMULA2 and FORMULA2 , we see that the ARM-like policy gradient (Equation FORMULA2 ) has a logarithmic dependence on the advantage-like functionĀ + , whereas the existing policy gradient (Equation FORMULA2 ) is only linearly dependent on the advantage function A β-soft .

This difference in logarithmic vs. linear dependence is responsible for a large part of the inherent distinction of ARM from existing policy gradient methods.

One consequence of the difference in logarithmic vs. linear dependence is that the ARM-like update should be less sensitive to large positive advantages that may result from overestimation compared to existing policy gradient methods.

We also see that for the existing policy gradient (Equation FORMULA2 ), the (1/β) log(π(a|o; θ)) term, which is derived from the policy entropy, is vanishing for large β (e.g. β = 100 is a common choice in practice).

On the other hand, for the ARM-like policy gradient (Equation FORMULA2 ), there is no similar vanishing effect, suggesting that ARM may perform a kind of entropy regularization by default.

In practice we cannot implement an ARM-like policy gradient exactly as in Equation FORMULA2 , as due to the positive clipping max(0,Ā + ) there can appear log(0).

However we believe this is not an intrinsic obstacle, leaving the issue of implementing an ARM-like policy gradient to future work.

In the previous Section 3.5, we showed that ARM and existing policy gradient methods can be distinguished by their choices of target policy and the nature of their dependence on their respective advantage-like functions.

In this section, we argue that the convergence results of CFR and CFR+ suggest that ARM, to the degree that it inherits the properties of CFR/CFR+, ought to benefit from greater partial observability compared to other methods.

We assume that regret bounds are a useful way to compare the convergence of different RL algorithms, due to the interpretation of regret as "area over the learning curve (and under the optimal expected value J * )."

Specifically, the regret bound of CFR and CFR+ is O(|O| √ T ) where |O| is the size of the observation space BID35 .

The policy gradient method with a suitable baseline has a learning rate η-dependent regret bound derived from the stochastic gradient method; assuming parameter norm bound B and gradient estimator second moments G 2 , by setting the learning rate η ∝ T −1/2 policy gradient achieves a regret bound of O( √ T ) with no explicit dependence on the observation space size |O| BID6 .We argue that possessing a regret bound proportional to the observation space size |O| is beneficial in highly partially observable domains.

Let us fix an underlying state space S. Compare two RL algorithms, where algorithm 1 (which is ARM-like) has a regret bound c 1 |O| √ T , whereas algorithm 2 (which is policy gradient-like) has a regret bound c 2 √ T ; here, c 1 and c 2 are constants.

Note that if c 1 |O| = c 2 or equivalently |O| = c 2 /c 1 , then the two RL algorithms possess the exact same regret bound.

If on the other hand |O| < c 2 /c 1 , then the regret bound of RL algorithm 1 is actually lower than that of RL algorithm 2.

Applying this intuition to CFR and hence ARM suggests that ARM can benefit from greater partial observability if the degree of partial observability is above a threshold.

For Q-learning per se, we are not aware of any known regret bound.

Szepesvári proved that the convergence rate of Q-learning in the L ∞ -norm, assuming a fixed exploration strategy, depends on a condition number C, which is the ratio of the minimum to maximum state-action occupation frequencies BID27 , and which describes how "balanced" the exploration strategy is.

If partial observability leads to imbalanced exploration due to confounding of states from perceptual aliasing (McCallum, 1997) , then Q-learning should be negatively affected.

We note that there remains a gap between ARM as implemented and the theory of CFR: the use of (a) function approximation and sampling over tabular enumeration; (b) the "ordinary" Q-function instead of the "stationary" Q-function; and (c) n-step bootstrapped values instead of full returns for value function estimation.

Waugh et al. (2015) address CFR with function approximation via a noisy version of a generalized Blackwell's condition BID5 .

Even the original implementation of CFR used sampling in place of enumeration BID35 .

We refer the reader to BID3 for a more in-depth discussion of the stationary Q-function.

Although only the full returns are guaranteed to be unbiased in non-Markovian settings, it is quite common for practical RL algorithms to trade off strict unbiasedness in favor of lower variance by using n-step returns or variations thereof BID25 BID9 .

Because we hypothesize that ARM should perform well in partially observable reinforcement learning environments, we conduct our experiments on visual domains that naturally provide partial observations of state.

All of our evaluations use feedforward convnets with frame-history observations.

We are interested in comparing ARM with methods that assume Markovian observations, namely double deep Q-learning , as well as methods that can handle non-Markovian observations, primarily TRPO BID24 , and to a lesser extent A3C (Mnih et al., 2016) whose critic assumes Markovian observations.

We are also interested in controlling for the advantage structure of ARM by comparing with other advantage-structured methods, which include dueling networks BID32 , as well as policy gradient methods that estimate an empirical advantage using a baseline state-value function or critic (e.g. TRPO, A3C).

Atari games consist of a small set of moving sprites with fixed shapes and palettes, and the motion of sprites can be highly deterministic, so that with only 4 recently observed frames as input one can predict hundreds of frames into the future on some games using only a feedforward model (Oh et al., 2015) .

To increase the partial observability of Atari games, one may artificially limit the amount of frame history fed as input to the networks BID12 .

As a proof of concept of ARM, we trained agents to play Pong via the Arcade Learning Environment BID2 when the frame-history length is varied between 4 (the default) and 1.

We found that the performance of double deep Q-learning degraded noticeably when the frame-history length was reduced from 4 to 1, whereas performance of ARM was not affected nearly as much.

Our results on Pong are summarized in FIG0 .

We evaluated ARM on the task of learning first-person navigation in the ViZDoom BID16 domain based on the game of Doom.

Doom is a substantially more complex domain than Atari, featuring an egocentric viewpoint, 3D perspective, and complex visuals.

We expect that Doom exhibits a substantial degree of partial observability and therefore serves as a more difficult evaluation of reinforcement learning algorithms' effectiveness on partially observable domains.

We performed our evaluation on two standard ViZDoom navigation benchmarks, "HealthGathering" and "MyWayHome." In "HealthGathering," the agent is placed in a toxic room and continually loses life points, but can navigate toward healthkit objects to prolong its life; the goal is to survive for as long as possible.

In "MyWayHome," the agent is randomly placed in a small maze and must find a target object that has a fixed visual appearance and is in a fixed location in the maze; the goal is to reach the target object before time runs out.

Figure 2 (top row) shows example observations from the two ViZDoom scenarios.

Unlike previous evaluations which augmented the raw pixel frames with extra information about the game state, e.g. elapsed time ticks or remaining health BID16 BID7 , in our evaluation we forced all networks to learn using only visual input.

Despite this restriction, ARM is still able to quickly learn policies with minimal tuning of hyperparameters and reach close to the maximum score in under 1 million steps.

On "HealthGathering," we observed that ARM very quickly learns a policy that can achieve close to the maximum episode return of 2100.

Double deep Q-learning learns a more consistent policy on "HealthGathering" compared to ARM and TRPO, but we believe this to be the result of evaluating double DQN's -greedy policy with small compared to the truly stochastic policies learned by ARM and TRPO.

On "MyWayHome," we observed that ARM generally learned a well-performing policy more quickly than other methods.

Additionally, we found that ARM is able to take advantage of an off-policy replay memory when learning on ViZDoom by storing the trajectories of previous sampling batches and applying an importance sampling correction to the n-step returns; please see Section 6.2 in the Appendix for details.

Our Doom results are in FIG4 .

We finally evaluated ARM on the task of learning first-person navigation in the Malmö domain based on the game of Minecraft BID15 .

Minecraft has similar visual complexity to Doom and should possess a comparable degree of partial observability, but Minecraft has the potential to be more difficult than Doom due to the diversity of possible Minecraft environments that can be generated.

Our evaluation on Minecraft is adapted from the teacher-student curriculum learning protocol (Matiisen et al., 2017) , which consists of 5 consecutive "levels" that successively increase the difficulty of completing the simple task of reaching a target block: the first level ("L1") consists of a single room; the intermediate levels ("L2"-"L4") consist of a corridor with lava-bridge and wall-gap obstacles; and the final level ("L5") consists of a 2 × 2 arrangement of rooms randomly separated by lava-bridge or wall-gap obstacles.

Figure 2 (bottom row) shows example observations from the five Minecraft levels.

We performed our Minecraft experiments using fixed curriculum learning schedules to evaluate the sample efficiency of different algorithms: the agent is initially placed in the first level ("L1"), and the agent is advanced to the next level whenever a preselected number of simulator steps have elapsed, until the agent reaches the last level ("L5").

We found that ARM and dueling double DQN both were able to learn on an aggressive "fast" schedule of only 62500 simulator steps between levels.

TRPO required a "slow" schedule of 93750 simulator steps between levels to reliably learn.

ARM was able to consistently learn a well performing policy on all of the levels, whereas double DQN learned more slowly on some of the intermediate levels.

ARM also more consistently reached a high score on the final, most difficult level ("L5").

Our Minecraft results are shown in

In this paper, we presented a novel deep reinforcement learning algorithm based on counterfactual regret minimization (CFR).

We call our method advantage-based regret minimization (ARM).

Similarly to prior methods that learn state or state-action value functions, our method learns a cumulative clipped advantage function of observation and action.

However, in contrast to these prior methods, ARM is well suited to partially observed or non-Markovian environments, making it an appealing choice in a number of difficult domains.

When compared to baseline methods, including deep Q-learning and TRPO, on non-Markovian tasks such as the challenging ViZDoom and Malmö firstperson navigation benchmarks, ARM achieves substantially better results.

This illustrates the value of ARM for partially observable problems.

In future work, we plan to further explore applications of ARM to more complex tasks, including continuous action spaces.6 APPENDIX 6.1 EXPERIMENTAL DETAILS 6.1.1 PONG (ARCADE LEARNING ENVIRONMENT)We use the preprocessing and convolutional network model of (Mnih et al., 2013) .

Specifically, we view every 4th emulator frame, convert the raw frames to grayscale, and perform downsampling to generate a single observed frame.

The input observation of the convnet is a concatenation of the most recent frames (either 4 frames or 1 frame).

The convnet consists of an 8 × 8 convolution with stride 4 and 16 filters followed by ReLU, a 4 × 4 convolution with stride 2 and 32 filters followed by ReLU, a linear map with 256 filters followed by ReLU, and a linear map with |A| filters where |A| is the action space cardinality (|A| = 6 for Pong).We used Adam with a constant learning rate of α = 10 −4 , a minibatch size of 32, and the moment decay rates set to their defaults β 1 = 0.9 and β 2 = 0.999.

Our results on each method are averaged across 3 random seeds.

We ran ARM with the hyperparameters: sampling batch size of 12500, 4000/3000 minibatches of Adam for the first/subsequent sampling iterations respectively, and target update step size τ = 0.01.

Double DQN uses the tuned hyperparameters .

Note that our choice of ARM hyperparameters yields an equivalent number of minibatch gradient updates per sample as used by DQN and double DQN, i.e. 1 minibatch gradient update per 4 simulator steps.

We used a convolutional network architecture similar to those of BID16 and BID7 ).

The Doom screen was rendered at a resolution of 160 × 120 and downsized to 84 × 84.

Only every 4th frame was rendered, and the input observation to the convnet is a concatenation of the last 4 rendered RGB frames for a total of 12 input channels.

The convnet contains 3 convolutions with 32 filters each: the first is size 8 × 8 with stride 4, the second is size 4 × 4 with stride 2, and the third is size 3 × 3 with stride 1.

The final convolution is followed by a linear map with 1024 filters.

A second linear map yields the output.

Hidden activations are gated by ReLUs.

For "HealthGathering" only, we scaled rewards by a factor of 0.01.

We did not scale rewards for "MyWayHome."

We used Adam with a constant learning rate of α = 10 −5 and a minibatch size of 32 to train all networks (except TRPO).

For "HealthGathering" we set β 1 = 0.95, whereas for "MyWayHome" we set β 1 = 0.9.

We set β 2 = 0.999 for both scenarios.

Our results on each method are averaged across 3 random seeds.

Double DQN and dueling double DQN: n = 5 step returns; update interval 30000; 1 minibatch gradient update per 4 simulator steps; replay memory uniform initialization size 50000; replay memory maximum size 240000; exploration period 240000; with final exploration rate = 0.01.

A3C: 16 workers; n = 20 steps for "HealthGathering" and n = 40 steps for "MyWayHome"; negentropy regularization β = 0.01; and gradient norm clip 5.TRPO: sampling batch size 12500; KL-divergence step size δ = 0.01; 10 conjugate gradient iterations; and Fisher information/Gauss-Newton damping coefficient λ = 0.1.

ARM: n = 5 step returns; sampling batch size 12500; 4000 Adam minibatches in the first sampling iteration, 3000 Adam minibatches in all subsequent sampling iterations; target update step size τ = 0.01.

Again, our choice of ARM hyperparameters yields an equivalent number of minibatch gradient updates per sample as used by DQN and double DQN.

For "HealthGathering" only, because ARM converges so quickly we annealed the Adam learning rate to α = 2.5 × 10 −6 after 500000 elapsed simulator steps.

Off-policy ARM: n = 5 step returns; sampling batch size 1563, replay cache sample size 25000; 400 Adam minibatches per sampling iteration; target update step size τ = 0.01; and importance sampling weight clip c = 1.

Our Minecraft tasks generally were the same as the ones used by Matiisen et al. (2017) , with a few differences.

Instead of using a continuous action space, we used a discrete action space with 4 move and turn actions.

To aid learning on the last level ("L5"), we removed the reward penalty upon episode timeout and we increased the timeout on "L5" from 45 seconds to 75 seconds due to the larger size of the environment.

We scaled rewards for all levels by 0.001.We use the same convolutional network architecture for Minecraft as we used for ViZDoom in Section 4.2.

The Minecraft screen was rendered at a resolution of 320 × 240 and downsized to 84 × 84.

Only every 5th frame was rendered, and the input observation of the convnet is a concatenation of the last 4 rendered RGB frames for a total of 12 input channels.

We used Adam with constant learning rate α = 10 −5 , moment decay rates β 1 = 0.9 and β 2 = 0.999, and minibatch size 32 to train all networks (except TRPO).

Our results on each method are averaged across 5 random seeds.

Double DQN and dueling double DQN: n = 5 step returns; update interval 12500; 1 minibatch gradient update per 4 simulator steps; replay memory uniform initialization size 12500; replay memory maximum size 62500; exploration period 62500; with final exploration rate = 0.01.TRPO: sampling batch size 6250; KL-divergence step size δ = 0.01; 10 conjugate gradient iterations; and Fisher information/Gauss-Newton damping coefficient λ = 0.1.

ARM: n = 5 step returns; sampling batch size 12500; 4000 Adam minibatches in the first sampling iteration, 3000 Adam minibatches in all subsequent sampling iterations; target update step size τ = 0.01.

Our current approach to running ARM with off-policy data consists of applying an importance sampling correction directly to the n-step returns.

Given the behavior policy µ under which the data was sampled, the current policy π t under which we want to perform estimation, and an importance sampling weight clip c for variance reduction, the corrected n-step return we use is: DISPLAYFORM0 where the truncated importance weight w µ πt (a|o) is defined (Ionides, 2008): DISPLAYFORM1 Our choice of c = 1 in our experiments was inspired by BID33 .

We found that c = 1 worked well but note other choices for c may also be reasonable.

When applying our importance sampling correction, we preserve all details of the ARM algorithm except for two aspects: the transition sampling strategy (a finite memory of previous batches are cached and uniformly sampled) and the regression targets for learning the value functions.

Specifically, the regression targets v(o k ; ϕ), q(o k , a k ; ϕ), andq + (o k , a k ; ϕ) (Equations FORMULA1 - FORMULA1 ) are modified to the following: DISPLAYFORM2 q µ πt (o k , a k ; ϕ) = r k + γw µ πt (a k |o k )g n−1 k+1 (µ π t ) + γ n V (o k+n ; ϕ)q + µ πt (o k , a k ; ϕ) = max(0,Q + t−1 (o k , a k ; ω t−1 ) − V πt−1 (o k ; θ t−1 )) + q µ πt (o k , a k ; ϕ).Note that the target value function V (o k+n ; ϕ) does not require an importance sampling correction because V already approximates the on-policy value function V πt (o k+n ; θ t ).

6.3.1 ATARI 2600 GAMES Although our primary interest is in partially observable reinforcement learning domains, we also want to check that ARM works in nearly fully observable and Markovian environments, such as Atari 2600 games.

We consider two baselines: double deep Q-learning, and double deep fitted Qiteration which is a batch counterpart to double DQN.

We find that double deep Q-learning is a strong baseline for learning to play Atari games, although ARM still successfully learns interesting policies.

One major benefit of Q-learning-based methods is the ability to utilize a large off-policy replay memory.

Our results on a suite of Atari games are in FIG3 .

We evaluated the effect of recurrent policy and value function estimation in the maze-like MyWayHome scenario of ViZDoom.

We found that recurrence has a small positive effect on the convergence of A2C (Mnih et al., 2016) , but was much less significant than the choice of algorithm.

Our hyperparameters were similar to those described for A3C in Section 6.1.2, except we used a learning rate 10 −4 and gradient norm clip 0.5.

For the recurrent policy and value function, we replaced the first fully connected operation with an LSTM featuring an equivalent number of hidden units (1024).

@highlight

Advantage-based regret minimization is a new deep reinforcement learning algorithm that is particularly effective on partially observable tasks, such as 1st person navigation in Doom and Minecraft.

@highlight

This paper introduces the concepts of counterfactual regret minimization in the field of Deep RL and an algorithm called ARM which can deal with partial observability better.

@highlight

The paper provides a game-theoretic inspired variant of policy-gradient algorithm based on the idea of counter-factual regret minimization and claims that the approach can deal with the partial observable domain better than standard methods.
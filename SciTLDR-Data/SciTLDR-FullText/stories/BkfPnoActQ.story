Despite significant advances in the field of deep Reinforcement Learning (RL), today's algorithms still fail to learn human-level policies consistently over a set of diverse tasks such as Atari 2600 games.

We identify three key challenges that any algorithm needs to master in order to perform well on all games:  processing diverse reward distributions, reasoning over long time horizons, and exploring efficiently.

In this paper, we propose an algorithm that addresses each of these challenges and is able to learn human-level policies on nearly all Atari games.

A new transformed Bellman operator allows our algorithm to process rewards of varying densities and scales; an auxiliary temporal consistency loss allows us to train stably using a discount factor of 0.999 (instead of 0.99) extending the effective planning horizon by an order of magnitude; and we ease the exploration problem by using human demonstrations that guide the agent towards rewarding states.

When tested on a set of 42 Atari games, our algorithm exceeds the performance  of an average human on 40 games using a common set of hyper parameters.

In recent years, significant advances in the field of deep Reinforcement Learning (RL) have led to artificial agents that are able to reach human-level control on a wide array of tasks such as some Atari 2600 games .

In many of the Atari games, these agents learn control policies that far exceed the capabilities of an average human player BID4 .

However, learning human-level policies consistently across the entire set of games remains an open problem.

We argue that an algorithm needs to overcome three key challenges in order to perform well on all Atari games.

The first challenge is processing diverse reward distributions.

An algorithm must learn stably regardless of reward density and scale.

BID12 showed that clipping rewards to the canonical interval [−1, 1] is one way to achieve stability.

However, this clipping operation may change the set of optimal policies.

For example, the agent no longer differentiates between striking a single pin or all ten pins in BOWLING.

Hence, optimizing the unaltered reward signal in a stable manner is crucial to achieving consistent performance across games.

The second challenge is reasoning over long time horizons, which means the algorithm should be able to choose actions in anticipation of rewards that might be far away.

For example, in MONTEZUMA'S REVENGE, individual rewards might be separated by several hundred time steps.

In the standard γ-discounted RL setting, this means the algorithm should be able to handle discount factors close to 1.

The third and final challenge is efficient exploration of the MDP.

An algorithm that explores efficiently is able to discover long trajectories with a high cumulative reward in a reasonable amount of time even if individual rewards are very sparse.

While each problem has been partially addressed in the literature, none of the existing deep RL algorithms have been able to address these three challenges at once.

In this paper, we propose a new Deep Q-Network (DQN) BID12 style algorithm that specifically addresses these three challenges.

In order to learn stably independent of the reward distribution, we use a transformed Bellman operator that reduces the variance of the action-value function.

Learning with the transformed operator allows us to process the unaltered environment rewards regardless of scale and density.

We prove that the optimal policy does not change in deterministic MDPs and show that under certain assumptions the operator is a contraction in stochastic MDPs (i.e., the algorithm converges to a fixed point) (see Sec. 3.2) .

Our algorithm learns stably even at high discount factors due to an auxiliary temporal consistency (TC) loss.

This loss prevents the network from prematurely generalizing to unseen states (Sec. 3.3) allowing us to use a discount factor as high as γ = 0.999 in practice.

This extends the effective planning horizon of our algorithm by one order of magnitude when compared to other deep RL approaches on Atari.

Finally, we improve the efficiency of DQN's default exploration scheme by combining the distributed experience replay approach of with the Deep Q-learning from Demonstrations (DQfD) algorithm of BID6 .

The resulting architecture is a distributed actor-learner system that combines offline expert demonstrations with online agent experiences (Sec. 3.4).We experimentally evaluate our algorithm on a set of 42 games for which we have demonstrations from an expert human player (see Table 6 ).

Using the same hyper parameters on all games, our algorithm exceeds the performance of an average human player on 40 games, the expert player on 34 games, and state-of-the-art agents on at least 28 games.

Furthermore, we significantly advance the state-of-the-art on sparse reward games.

Our algorithm completes the first level of MONTEZUMA'S REVENGE and it achieves a score of 3997 points on PITFALL! without compromising performance on dense reward games and while only using 5 demonstration trajectories.

Reinforcement Learning with Expert Demonstrations (RLED): RLED seeks to use expert demonstrations to guide the exploration process in difficult RL problems.

Some early works in this area BID0 BID16 used expert demonstrations to find a good initial policy before fine-tuning it with RL.

More recent approaches have explicitly combined expert demonstrations with RL data during the learning of the policy or action-value function BID2 BID10 BID14 .

In these works, expert demonstrations were used to build an imitation loss function (classification-based loss) or max-margin constraints.

While these algorithms worked reasonably well in small problems, they relied on handcrafted features to describe states and were not applied to large MDPs.

In contrast, approaches using deep neural networks allow RLED to be explored in more challenging RL tasks such as Atari or robotics.

In particular, our work builds upon DQfD BID6 , which used a separate replay buffer for expert demonstrations, and minimized the sum of a temporal difference loss and a supervised classification loss.

Another similar approach is Replay Buffer Spiking (RBS) BID11 , wherein the experience replay buffer is initialized with demonstration data, but this data is not kept for the full duration of the training.

In robotics tasks, similar techniques have been combined with other improvements to successfully solve difficult exploration problems BID13 BID20 .Deep Q-Networks (DQN): DQN BID12 used deep neural networks as function approximators to apply RL to Atari games.

Since that work, many extensions that significantly improve the algorithm's performance have been developed.

For example, DQN uses a replay buffer to store off-policy experiences and the algorithm learns by sampling batches uniformly from the replay buffer; instead of using uniform samples, BID17 proposed prioritized sampling where transitions are weighted by their absolute temporal difference error.

This concept was further improved by Ape-X DQN which decoupled the data collection and the learning processes by having many actors feed data to a central prioritized replay buffer that an independent learner can sample from.

BID3 observed that due to over-generalization in DQN, updates to the value of the current state also have an adverse effect on the values of the next state.

This can lead to unstable learning when the discount factor is high.

To counteract this effect, they constrained the TD update to be orthogonal to the direction of maximum change of the next state.

However, their approach only worked on toy domains such as Cart-Pole.

Finally, van Hasselt et al. (2016a) successfully extended DQN to process unclipped rewards with an algorithm called PopArt, which adaptively rescales the targets for the value network to have zero mean and unit variance.

In this section, we describe our algorithm, which consists of three components: (1) The transformed Bellman operator; (2) The temporal consistency (TC) loss; (3) Combining Ape-X DQN and DQfD.

Let X , A, r, p, γ be a finite, discrete-time MDP where X is the state space, A the action space, r the reward function which represents the one-step reward distribution r(x, a) of doing action a in state x, γ ∈ [0, 1] the discount factor and p a stochastic kernel modelling the one-step Markovian dynamics (p(x |x, a) is the probability of transitioning to state x by choosing action a in state x).

The quality of a policy π is determined by the action-value function DISPLAYFORM0 where E π is the expectation over the distribution of the admissible trajectories (x 0 , a 0 , x 1 , a 1 , . . . ) obtained by executing the policy π starting from state x and taking action a. The goal is to find a policy π * that maximizes the state-value V π (x) := max a∈A Q π (x, a) for all states x, i.e., find π * such that V π * (x) = sup π V π (x) for all x ∈ X .

While there may be several optimal policies, they all share a common optimal action-value function Q * BID15 .

Furthermore, acting greedily with respect to the optimal action-value function Q * yields an optimal policy.

In addition, Q * is the unique fixed point of the Bellman optimality operator T defined as DISPLAYFORM1 Because T is a γ-contraction, we can learn Q * using a fixed point iteration.

Starting with an arbitrary function Q (0) and then iterating Q (k) := T Q (k−1) for k ∈ N generates a sequence of functions that converges to Q * .DQN BID12 is an online-RL algorithm using a deep neural network f θ with parameters θ as a function approximator of the optimal action-value function Q * .

The algorithm starts with a random initialization of the network weights θ (0) and then iterates DISPLAYFORM2 where the expectation is taken with respect to a random sample of states and actions from an experience replay buffer and L is the Huber loss (Huber, 1964) defined as DISPLAYFORM3 In practice, the minimization problem in (1) is only approximately solved by performing a finite and fixed number of stochastic gradient descent (SGD) steps 1 and all expectations are approximated by sample averages.

BID12 have empirically observed that the errors induced by the limited network capacity, the approximate finite-time solution to (1), and the stochasticity of the optimization problem can cause the algorithm to diverge if the variance of the action-value function is too high.

In order to reduce the variance, they clip the reward distribution to the interval [−1, 1].

While this achieves the desired goal of stabilizing the algorithm, it significantly changes the set of optimal policies.

For example, consider a simplified version of BOWLING where an episode only consists of a single throw.

If the original reward is the number of hit pins and the rewards were clipped, any policy that hits at least a single pin would be optimal under the clipped reward function.

Instead of reducing the magnitude of the rewards, we propose to focus on the action-value function instead.

We use a function h : R → R that reduces the scale of the action-value function.

Our new operator T h is defined as

Proposition 3.1.

Let Q * be the fixed point of T and Q : X × A → R, then DISPLAYFORM0 (ii) If h is strictly monotonically increasing and the MDP is deterministic (i.e., p(·|x, a) and r(x, a) DISPLAYFORM1 Proposition 3.1 shows that in the basic cases when either h is linear or the MDP is deterministic, T h has the unique fixed point h • Q * .

We present a proof in Sec. B in the appendix.

Furthermore, the fixed point iteration T k h Q converges to h • Q * for all Q. We treat the case of stochastic MDPs in the appendix (see Proposition C.1).

The following proposition (see a proof in Sec. B in the appendix) shows that contracting h can achieve the desired goal of decreasing the scale and variance of the action-value function.

Proposition 3.2.

Let h : R → R be a contraction mapping with fixed point 0 (i.e., h(0) = 0), then DISPLAYFORM2 In our algorithm, we use h : z → sign(z)( |z| + 1 − 1) + εz with ε = 10 −2 where the additive regularization term εz ensures that h −1 is Lipschitz continuous (see Proposition C.1).

We chose this function because it is an invertible contraction mapping with fixed point 0 and has a closed-form inverse (see Proposition C.2).In practice, DQN minimizes the problem in (1) by sampling transitions of the form t = (x, a, r , x ) from a replay buffer where x ∈ X , a ∼ π(·|x), r ∼ r(x, a), and x ∼ p(x, a).

Let t 1 , ..., t N be N transitions from the buffer with normalized sampling weights w 1 , ..., w N , then for k ∈ N the loss function in (1) using the operator T h is approximated as DISPLAYFORM3 where a i := arg max a∈A f θ (k−1) (x i , a) for DQN and a i := arg max a∈A f θ (x i , a) for Double DQN (van Hasselt et al., 2016b) .

The stability of DQN, which minimizes the TD-loss L TD , is primarily determined by the target T h f θ (k−1) .

While the transformed Bellman operator provides an atemporal reduction of the target's scale and variance, instability can still occur as the discount factor γ approaches 1.

Increasing the discount factor decreases the temporal difference in value between non-rewarding states.

In particular, unwanted generalization of the neural network f θ to the next state x (due to the similarity of temporally adjacent target values) can result in catastrophic TD backups.

We resolve the problem by adding an auxiliary temporal consistency (TC) loss of the form DISPLAYFORM0 where k ∈ N is the current iteration.

The TC-loss penalizes weight updates that change the next action-value estimate f θ (x , a ).

This makes sure that the updated estimates adhere to the operator T h and thus are consistent over time.

In this section, we describe how we combine the transformed Bellman operator and the TC loss with the DQfD algorithm BID6 and distributed prioritized experience replay .

The resulting algorithm, which we call Ape-X DQfD following , is a distributed DQN algorithm with expert demonstrations that is robust to the reward distribution and can learn at discount factors an order of magnitude higher than what was possible before (i.e., γ = 0.999 instead of γ = 0.99).

Our algorithm consists of three components: (1) replay buffers; (2) actor processes; and (3) a learner process.

FIG0 shows how our architecture compares to the one used by .Replay buffers.

Following Hester et al. FORMULA2 , we maintain two replay buffers: an actor replay buffer and an expert replay buffer.

Both buffers store 1-step and 10-step transitions and are prioritized BID17 .

The transitions in the actor replay buffer come from actor processes that interact with the MDP.

In order to limit the memory consumption of the actor replay buffer, we regularly remove transitions in a FIFO-manner.

The expert replay buffer is filled once offline before training commences.

Actor processes. showed that we can significantly improve the performance of DQN with prioritized replay buffers by having many actor processes.

We follow their approach and use m = 128 actor processes.

Each actor i follows an ε i -greedy policy based on the current estimate of the action-value function.

The noise levels ε i are chosen as ε i := 0.1 αi+3(1−αi) where α i := i−1 m−1 .

Notably, this exploration is closer to the one used by BID6 and is much lower (i.e., less random exploration) than the schedule used by .Learner process.

The learner process samples experiences from the two replay buffers and minimizes a loss in order to approximate the optimal action-value function.

Following BID6 , we combine the TD-loss L TD with a supervised imitation loss.

Let t 1 , ..., t N be transitions of the form t i = (x i , a i , r i , x i , e i ) with normalized sampling weights w 1 , ..., w N where e i is 1 if the transition is part of the best (i.e., highest episode return) expert episode and 0 otherwise.

The imitation loss is a max-margin loss of the form DISPLAYFORM0 where λ ∈ R is the margin and δ a =ai is 1 if a = a i and 0 otherwise.

Combining the imitation loss with the TD loss and the TC loss yields the total loss formulation DISPLAYFORM1 Algo.

1, provided in the appendix, shows the entire learner procedure.

Note that while we only apply the imitation loss L IM on the best expert trajectory, we still use all expert trajectories for the other two losses.

Our learning algorithm differs from the one used by BID6 in three important ways.

First, we do not have a pre-training phase where we minimize L only using expert transitions.

We learn with a mix of actor and expert transitions from the beginning.

Second, we maintain a fixed ratio of actor and expert transitions.

For each SGD step, our training batch consists of 75% agent transitions and 25% expert transitions.

The ratio is constant throughout the entire learning process.

Finally, we only apply the imitation loss L IM to the best expert episode instead of all episodes.

avg.

human score−random score × 100 and then aggregate over all games (mean or median).

Because we only have demonstrations on 42 out of the 57 games, we report the performances on 42 games and also 57 games for baselines not using demonstrations.

Results of Ape-X DQN using our hyper parameters are marked with an asterisk*. The experiment Ape-X DQN* (u, 0.999) uses the exact same hyper parameters and network architecture used for the Ape-X DQfD experiments.

DISPLAYFORM2

We evaluate our approach on the same subset of 42 games from the Arcade Learning Environment (ALE) used by BID6 .

We report the performance using the no-op starts and the human starts test regimes BID12 .

Our hyper parameter setting deviates from the one used by in several ways (see Tab.

3 in the appendix).

As described in Sec. 3, we use a higher discount factor than Ape-X DQN ) (γ = 0.999 instead of γ = 0.99) and we do not clip the environment rewards to [−1, 1].

These changes are motivated by our goal of finding an algorithm that learns consistently on all games.

In order to distinguish the contribution of changed hyper parameters from the algorithmic contributions, we rerun Ape-X DQN using variations of our hyper parameters (column Ape-X DQfD in Tab.

3).

We use the naming strategy Ape-X DQN* FIG5 shows the results.

We can draw two conclusions regarding reward clipping in Ape-X DQN.

First, the overall performance as measured by the mean and median scores decreases when using the unclipped rewards.

This shows that simplifying the reward structure of the MDPs helps Ape-X DQN learn better policies.

However, when aiming at consistency (i.e., having one algorithm perform well on all games), reward clipping is the wrong thing to do.

When looking at our introductory example BOWLING FIG5 , we see that indeed reward clipping hurts performance and Ape-X DQN is able to learn a good policy only when it sees the true environment rewards.

As the following experiments show, using the transformed Bellman operator can help recover some of the performance losses incurred by using unclipped rewards.

We compare our approach to Ape-X DQN , on which our actor-learner architecture is based, DQfD BID6 , which introduced the expert replay buffer and the imitation loss, and Rainbow DQN , which combines all major DQN extensions from the literature into a single algorithm.

Note that the scores reported in were obtained by running 360 actors.

Due to resource constraints, we limit the number of actors to 128 for all Ape-X DQfD experiments.

Besides comparing our performance to other RL agents, we are also interested in comparing our scores to a human player.

Because our demonstrations were gathered from an expert player, the expert scores are mostly better than the level of human performance reported in the literature BID12 Wang et al., 2016) .

Hence, we treat the historical human scores as the performance of an average human and the scores of our expert as expert performance.

We first analyse the performance of Ape-X DQfD with the standard dueling DQN architecture (Wang et al., 2016) that is also used by the baselines.

We report the scores as Ape-X DQfD in TAB1 .

We designed the algorithm to achieve higher consistency over a broad range of games and the scores shown in TAB1 reflect that goal.

Whereas previous approaches outperformed an average human on at most 35 out of 42 games, Ape-X DQfD with the standard dueling architecture achieves a new state-of-the-art result of 39 out of 42 games.

This means we significantly improve the performance on the tails of the distribution of scores over the games.

When looking at this performance in the context of the median human-normalized scores reported in TAB2 , we see that we significantly increase the set of games where we learn good policies at the expense of achieving lower peak scores on some games.

One of the significant changes in our experimental setup is moving from a discount factor of γ = 0.99 to γ = 0.999.

BID9 argue that this increases the complexity of the learning problem and, thus, requires a bigger hypothesis space.

Hence, in addition to the standard architecture, we also evaluated a slightly wider (i.e., double the number of convolutional kernels) and deeper (one extra fully connected layer) network architecture (see FIG7 ).

With the deeper architecture, our algorithm outperforms an average human on 40 out of 42 games.

Furthermore, it is the first deep RL algorithm to learn non-trivial policies on all games including sparse reward games such as MONTEZUMA'S REVENGE, PRIVATE EYE, and PITFALL!. For example, we achieve 3997 points in PITFALL!, which is below the 6464 points of an average human but far above any baseline.

Finally, with a median human-normalized score of 702% and exceeding every baseline on at least 2 3 of the games, we demonstrate strong peak performance and consistency over the entire benchmark.

Although we use demonstration data, the goal of RLED algorithms is still to learn an optimal policy that maximizes the expected γ-discounted return.

While TAB1 shows that we exceed the best expert episode on 34 games using the deeper architecture, it is hard to grasp the qualitative differences between the expert policies and our algorithm's policies.

In order to qualitatively compare the agent and the expert, we provide videos in the appendix (see Sec. G) and we plot the cumulative episode return of the best expert and agent episodes in Fig. 2 .

We see that our algorithm ( ) finds more time-efficient policies than the expert ( ) in all cases.

This is a strong indicator that our algorithm does not do pure imitation but improves upon the demonstrated policies.

We evaluate the performance contributions of the three key ingredients of Ape-X DQfD (transformed Bellman operator, the TC-loss, and demonstration data) by performing an ablation study on a subset of 6 games.

We chose sparse-reward games (MONTEZUMA'S REVENGE, PRIVATE EYE), dense-reward games (MS.

PACMAN, SEAQUEST), and games where DQfD performs well (HERO, KANGAROO) (see Fig. 3 ).

): When using the standard Bellman operator T instead of the transformed one, Ape-X DQfD is stable but the performance is significantly worse.

Figure 2: The figure shows the cumulative undiscounted episode return over time and compares the best expert episode to the best Ape-X DQfD episode on three games.

On HERO, the algorithm exceeds the expert's performance, on MONTEZUMA'S REVENGE, it matches the expert's score but reaches it faster, and on MS.

PACMAN, the expert is still superior.

Figure 3 : Results of our ablation study using the standard network architecture.

The experiment without expert data ( ) was performed with the higher exploration schedule used in .

): In our setup, the TC loss is paramount to learning stably.

We see that without the TC loss the algorithm learns faster at the beginning of the training process.

However, at some point during training, the performance collapses and often the process dies with floating point exceptions.

Expert demonstrations ( and ): Unsurprisingly, removing demonstrations entirely ( ) severely degrades the algorithm's performance on sparse reward games.

However, in games that an ε-greedy policy can efficiently explore, such as SEAQUEST, the performance is on par or worse.

Hence, the bias induced by the expert data is beneficial in some games and harmful in others.

Just removing the imitation loss L IM ( ) does not have a significant effect on the algorithm's performance.

This stands in contrast to the original DQfD algorithm and is most likely because we only apply the loss on a single expert trajectory.

The problems of handling diverse reward distributions and network over-generalization in deep RL have been partially addressed in the literature (see Sec. 2).

Specifically, BID18 proposed PopArt and Durugkar & Stone (2017) used constrained TD updates.

We evaluate the performance of our algorithm when using alternative solutions and report the results in FIG3 .

We use the standard Bellman operator T in combination with PopArt, which adaptively normalizes the targets in (1) to have zero mean and unit variance.

While the modified algorithm manages to learn in some games, the overall performance is significantly worse than Ape-X DQfD. One possible limiting factor that makes PopArt a bad choice for our framework is that training batches contain highly rewarding states from the very beginning of training.

SGD updates performed before The figures show how our algorithm compares when we substitute the transformed Bellman operator to PopArt and when we substitute the TC loss to constrained TD updates.

Note that the scales differ from the ones in Fig. 3 because the experiments only ran for 40 hours.the moving statistics have adequately adapted the moments of the target distribution might result in catastrophic changes to the network's weights.

We replaced the TC-loss with the constrained TD update approach BID3 ) that removes the target network and constrains the gradient to prevent an SGD update from changing the predictions at the next state.

We did not find the approach to work in our case.

In this paper, we presented a deep Reinforcement Learning (RL) algorithm that achieves human-level performance on a wide variety of MDPs on the Atari 2600 benchmark.

It does so by addressing three challenges: handling diverse reward distributions, acting over longer time horizons, and efficiently exploring on sparse reward tasks.

We introduce novel approaches for each of these challenges: a transformed Bellman operator, a temporal consistency loss, and a distributed RLED framework for learning from human demonstrations and task reward.

Our algorithm exceeds the performance of an average human on 40 out of 42 Atari 2600 games.

Ape-X DQN Ape-X DQfD Why?End episodes on loss of life true false The demonstration trajectories recorded by the expert were recorded without the loss-of-life signal.n-step transition parameter n = 3 n = 1 and n = 10 n = 1 and n = 10 were the choices in the original DQfD paper.

Exploration of i-th actor where m is the total number of actors and αi := i−1 m−1.εi := 0.1 and Ape-X DQfD. In addition to highlighting the differences, we explain the reason behind the change.

DISPLAYFORM0

Proof of Proposition 3.1.

(i) is equivalent to linearly scaling the reward r by a constant α > 0, which implies the proposition.

For (ii) let Q * be the fixed point of T and note that h DISPLAYFORM0 where the last equality only holds if the MDP is deterministic.

DISPLAYFORM1

The following proposition shows that transformed Bellman operator is still a contraction for small γ if we assume a stochastic MDP and a more generic choice of h. However, the fixed point might not be h • Q * .Proposition C.1.

Let h be strictly monotonically increasing, Lipschitz continuous with Lipschitz constant L h , and have a Lipschitz continuous inverse DISPLAYFORM0 , T h is a contraction.

Proof.

Let Q, U : X × A → R be arbitrary.

It holds DISPLAYFORM0 where we used Jensen's inequality in (1) and the Lipschitz properties of h and h −1 in (1) and (2).For our algorithm, we use h : R → R, x → sign(x)( |x| + 1 − 1) + εx with ε = 10 −2 .

While Proposition C.2 shows that the transformed operator is a contraction, the discount factor γ we use in practice is higher than DISPLAYFORM1 .

We leave a deeper investigation of the contraction properties of T h in stochastic MDPs for future work.

Proposition C.2.

Let ε > 0 and h : R → R, x → sign(x)( |x| + 1 − 1) + εx.

It holds (i) h is strictly monotonically increasing.(ii) h is Lipschitz continuous with Lipschitz constant DISPLAYFORM2 (iv) h −1 is strictly monotonically increasing.

DISPLAYFORM3 We use the following Lemmas in order to prove Proposition C.2.Lemma C.1.

h : R → R, x → sign(x)( |x| + 1 − 1) + εx is differentiable everywhere with derivative DISPLAYFORM4 Proof of Lemma C.1.

For x > 0, h is differentiable as a composition of differentiable functions with DISPLAYFORM5 where with derivative DISPLAYFORM6 for all x ∈ R.Proof of Lemma C.2.

For x = 0, h −1 is differentiable as a composition of differentiable functions.

For x = 0, it holds DISPLAYFORM7 and similarly DISPLAYFORM8 , which concludes the proof.

Proof of Proposition C.2.

We prove all statements individually.

DISPLAYFORM9 + εx > 0 for all x ∈ R, which implies the proposition.(ii) Let x, y ∈ R with x < y, using the mean value theorem, we find DISPLAYFORM10 (iii) (i) Implies that h is invertible and simple substitution shows DISPLAYFORM11 > 0 for all x ∈ R, which implies the proposition.(v) Let x, y ∈ R with x < y, using the mean value theorem, we find DISPLAYFORM12 |x − y|.

The algorithm used by the learner to estimate the action-value function.

DISPLAYFORM0 for j = 1, ..., Tupdate do Tupdate is the target network update period (ti, wi) DISPLAYFORM1 Sample 75% agent and 25% expert transitions DISPLAYFORM2 Update the parameters using Adam DISPLAYFORM3 Compute the updated priorities based on the TD error UPDATEPRIORITIES FIG0 , ..., (tN , wN ) ) Send the updated priorities to the replay buffers end for end for

Ape-X DQfD Ape-X DQfD (deeper) Best expert trajectory Avg.

human Figure 5 : Training curves on all 42 games.

We report the performance using the standard network architecture (Wang et al., 2016) and the slightly deeper version (see FIG7 ).

The two network architectures that we used.

The upper one is the standard dueling architecture of Wang et al. (2016) and the lower one is a slightly wider and deeper version.

@highlight

Ape-X DQfD = Distributed (many actors + one learner + prioritized replay) DQN with demonstrations optimizing the unclipped 0.999-discounted return on Atari.

@highlight

The paper proposes three extensions (Bellman update, temporal consistency loss, and expert demonstration) to DQN to improve the learning performance on Atari games, achieving outperformance over the state-of-the-art results for Atari games. 

@highlight

This paper proposes a transformed Bellman operator that aims to solve sensitivity to unclipped reward, robustness to the value of the discount factor, and the exploration problem.
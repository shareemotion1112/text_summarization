The practical usage of reinforcement learning agents is often bottlenecked by the duration of training time.

To accelerate training, practitioners often turn to distributed reinforcement learning architectures to parallelize and accelerate the training process.

However, modern methods for scalable reinforcement learning (RL) often tradeoff between the throughput of samples that an RL agent can learn from (sample throughput) and the quality of learning from each sample (sample efficiency).

In these scalable RL architectures, as one increases sample throughput (i.e. increasing parallelization in IMPALA (Espeholt et al., 2018)), sample efficiency drops significantly.

To address this, we propose a new distributed reinforcement learning algorithm, IMPACT.

IMPACT extends PPO with three changes: a target network for stabilizing the surrogate objective, a circular buffer, and truncated importance sampling.

In discrete action-space environments, we show that IMPACT attains higher reward and, simultaneously, achieves up to 30% decrease in training wall-time than that of IMPALA.

For continuous control environments, IMPACT trains faster than existing scalable agents while preserving the sample efficiency of synchronous PPO.

Proximal Policy Optimization (Schulman et al., 2017 ) is one of the most sample-efficient on-policy algorithms.

However, it relies on a synchronous architecture for collecting experiences, which is closely tied to its trust region optimization objective.

Other architectures such as IMPALA can achieve much higher throughputs due to the asynchronous collection of samples from workers.

Yet, IMPALA suffers from reduced sample efficiency since it cannot safely take multiple SGD steps per batch as PPO can.

The new agent, Importance Weighted Asynchronous Architectures with Clipped Target Networks (IMPACT), mitigates this inherent mismatch.

Not only is the algorithm highly sample efficient, it can learn quickly, training 30 percent faster than IMPALA.

At the same time, we propose a novel method to stabilize agents in distributed asynchronous setups and, through our ablation studies, show how the agent can learn in both a time and sample efficient manner.

In our paper, we show that the algorithm IMPACT realizes greater gains by striking the balance between high sample throughput and sample efficiency.

In our experiments, we demonstrate in the experiments that IMPACT exceeds state-of-the-art agents in training time (with same hardware) while maintaining similar sample efficiency with PPO's.

The contributions of this paper are as follows:

1.

We show that when collecting experiences asynchronously, introducing a target network allows for a stabilized surrogate objective and multiple SGD steps per batch (Section 3.1).

2.

We show that using a circular buffer for storing asynchronously collected experiences allows for smooth trade-off between real-time performance and sample efficiency (Section 3.2).

3.

We show that IMPACT, when evaluated using identical hardware and neural network models, improves both in real-time and timestep efficiency over both synchronous PPO and IMPALA (Section 4).

into a large training batch and the learner performs minibatch SGD.

IMPALA workers asynchronously generate data.

IMPACT consists of a batch buffer that takes in worker experience and a target's evaluation on the experience.

The learner samples from the buffer.

Reinforcement Learning assumes a Markov Decision Process (MDP) setup defined by the tuple (S, A, p, γ, r) where S and A represent the state and action space, γ ∈ [0, 1] is the discount factor, and p : S × A × S → R and R : S × A → R are the transition dynamics and reward function that models an environment.

Let π(a t |s t ) : S × A → [0, 1] denote a stochastic policy mapping that returns an action distribution given state s t ∈ S. Rolling out policy π(a t |s t ) in the environment is equivalent to sampling a trajectory τ ∼ P(τ ), where τ := (s 0 , a 0 , ...., a T −1 , s T , a T ).

We can compactly define state and state-action marginals of the trajectory distribution p π (s t ) and p π (s t , a t ) induced by the policy π(a t |s t ).The goal for reinforcement learning aims to maximize the following objective:

When θ parameterizes π(a t |s t ), the policy is updated according to the Policy Gradient Theorem (Sutton et al., 2000) :

whereÂ π θ (s t , a t ) is an estimator of the advantage function.

The advantage estimator is usually defined as the 1-step TD error,Â π θ (s t , a t ) = r(s t , a t ) + γV (s t+1 ) −V (s t ), whereV (s t ) is an estimation of the value function.

Policy gradients, however, suffer from high variance and large update-step sizes, oftentimes leading to sudden drops in performance.

Per iteration, Proximal Policy Optimization (PPO) optimizes policy π θ from target π θold via the following objective function

where r t (θ) = π θ (at|st) π θ old (at|st) and is the clipping hyperparameter.

In addition, many PPO implementations use GAE-λ as a low bias, low variance advantage estimator forÂ t (Schulman et al., 2015b ).

PPO's surrogate objective contains the importance sampling ratio r t (θ), which can potentially explode if π θold is too far from π θ . (Han & Sung, 2017 ).

PPO's surrogate loss mitigates this with the clipping function, which ensures that the agent makes reasonable steps.

Alternatively, PPO can also be seen as an adaptive trust region introduced in TRPO (Schulman et al., 2015a) .

In Figure 1a , distributed PPO agents implement a synchronous data-gathering scheme.

Before data collection, workers are updated to π old and aggregate worker batches to training batch D train .

The learner performs many mini-batch gradient steps on D train .

Once the learner is done, learner weights are broadcast to all workers, who start sampling again.

In Figure 1b , IMPALA decouples acting and learning by having the learner threads send actions, observations, and values while the master thread computes and applies the gradients from a queue of learners experience (Espeholt et al., 2018) .

This maximizes GPU utilization and allows for increased sample throughput, leading to high training speeds on easier environments such as Pong.

As the number of learners grows, worker policies begin to diverge from the learner policy, resulting in stale policy gradients.

To correct this, the IMPALA paper utilizes V-trace to correct the distributional shift:

where, V φ is the value network, π θ is the policy network of the master thread, µ θ is the policy network of the learner thread, and c j = min c,

Input: Batch size M , number of workers W , circular buffer size N , replay coefficient K, target update frequency t target , weight broadcast frequency t frequency , learning rates α and β 1: Randomly initialize network weights (θ, w) 2: Initialize target network (θ , w ) ← (θ, w) 3: Create W workers and duplicate (θ, w) to each worker 4: Initialize circular buffer C(N, K) 5: for t = 1, .., T do Compute policy and value network gradients

Update policy and value network weights

If t ≡ 0 (mod t frequency ), broadcast weights to workers 13: end for Worker-i Input: Worker sample batch size S 1: repeat 2:

for t = 1, ..., S do Store (s t , a t , r t , s t+1 ) ran by θ i in batch B i

end for 6:

If broadcasted weights exist, set θ i ← θ 8: until learner finishes 3 IMPACT ALGORITHM Like IMPALA, IMPACT separates sampling workers from learner workers.

Algorithm 1 and Figure 1c describe the main training loop and architecture of IMPACT.

In the beginning, each worker copies weights from the master network.

Then, each worker uses their own policy to collect trajectories Since π worker may differ per worker, using this ratio results in trust region conflicts across multiple batches.

Since π learner is updated after each batch from the worker, only a single SGD step can be taken per batch.

The IMPACT objective allows for multiple SGD steps per async batch and has a stable trust region.

Figure 2: In asynchronous PPO, there are multiple candidate policies from which the trust region can be defined: (1) π workeri , the policy of the worker process that produced the batch of experiences, (2) π learner , the current policy of the learner process, and (3) π target , the policy of a target network.

Introducing the target network allows for both a stable trust region and multiple SGD steps per batch of experience collected asynchronously from workers, improving sample efficiency.

Since workers can generate experiences asynchronously from their copy of the master policy, this also allows for good real-time efficiency.

and sends the data (s t , a t , r t ) to the circular buffer.

Simultaneously, workers also asynchronously pull policy weights from the master learner.

In the meantime, the target network occasionally syncs with the master learner every t target iterations.

The master learner then repeatedly draws experience from the circular buffer.

Each sample is weighted by the importance ratio of πtarget .

The target network is used to provide a stable trust region ( Figure  2 ), allowing multiple steps per batch (i.e., like PPO) even in the asynchronous setting (i.e., with the IMPALA architecture).

In the next section, we describe the design of this improved objective.

PPO gathers experience from previous iteration's policy π θold , and the current policy trains by importance sampling off-policy experience with respect to π θ .

In the asynchronous setting, worker i's policy, denoted as π workeri , generates experience for the policy network π θ .

The probability that batch B comes from worker i can be parameterized as a categorical distribution i ∼ D(α 1 , ..., α n ).

We include this by adding an extra expectation to the importance-sampled policy gradient objective (IS-PG) (Jie & Abbeel, 2010) :

Since each worker contains a different policy, the agent introduces a target network for stability ( Figure 2 ).

Off-policy agents such as DDPG and DQN update target networks with a moving average.

For IMPACT, we periodically update the target network with the master network.

However, training with importance weighted ratio π θ πtarget can lead to numerical instability, as shown in Figure 3 .

To prevent this, we clip the importance sampling ratio from worker policy,π workeri , to target policy, π target :

where β = 1 ρ .

In the experiments, we set ρ as a hyperparameter with ρ ≥ 1 and β ≤ 1.

To see why clipping is necessary, when master network's action distribution changes significantly over few training iterations, worker i's policy, π workeri , samples data outside that of target policy, π target , leading to large likelihood ratios,

In (b), we show the target network update frequency is robust to a range of choices.

We try target network update frequency ttarget equal to the multiple (ranging from 1/16 and 16) of n = N · K, the product of the size of circular buffer and the replay times for each batch in the buffer.

large IS ratios to ρ.

Figure 10 in Appendix E provides additional intuition behind the target clipping objective.

We show that the target network clipping is a lower bound of the IS-PG objective.

For ρ > 1, the clipped target ratio is larger and serves to augment advantage estimatorÂ t .

This incentivizes the agent toward good actions while avoiding bad actions.

Thus, higher values of ρ encourages the agent to learn faster at the cost of instability.

We use GAE-λ with V-trace (Han & Sung, 2019) .

The V-trace GAE-λ modifies the advantage function by adding clipped importance sampling terms to the summation of TD errors:

where c i = min c, πtarget(aj |sj ) πworker i (aj |sj ) (we use the convention t−1 j=t c j = 1) and δ i V is the importance sampled 1-step TD error introduced in V-trace.

IMPACT uses a circular buffer (Figure 4 ) to emulate the mini-batch SGD used by standard PPO.

The circular buffer stores N batches that can be traversed at max K times.

Upon being traversed K times, a batch is discarded and replaced by a new worker batch.

For motivation, the circular buffer and the target network are analogous to mini-batching from π old experience in PPO.

When target network's update frequency n = N K, the circular buffer is equivalent to distributed PPO's training batch when the learner samples N minibatches for K SGD iterations.

This is in contrast to standard replay buffers, such as in ACER and APE-X, where transitions (s t , a t , r t , s t+1 ) are either uniformly sampled or sampled based on priority, and, when the buffer is full, the oldest transitions are discarded (Wang et al., 2016; Horgan et al., 2018) .

We investigate the performance of the clipped-target objective relative to prior work, which includes PPO and IS-PG based objectives.

Specifically, we consider the following ratios below:

For all three experiments, we truncate all three ratios with PPO's clipping function: c(R) = clip(R, 1− , 1+ ) and train in an asynchronous setting.

Figure 4 (a) reveals two important takeaways: first, R 1 suffers from sudden drops in performance midway through training.

Next, R 2 trains stably but does not achieve good performance.

We theorize that R 1 fails due to the target and worker network mismatch.

During periods of training where the master learner undergoes drastic changes, worker action outputs vastly differ from the learner outputs, resulting in small action probabilities.

This creates large ratios in training and destabilizes training.

We hypothesize that R 2 fails due to different workers pushing and pulling the learner in multiple directions.

The learner moves forward with the most recent worker's suggestions without developing a proper trust region, resulting in many worker's suggestions conflicting with each other.

The loss function, R 3 shows that clipping is necessary and can help facilitate training.

By clipping the target-worker ratio, we make sure that the ratio does not explode and destabilize training.

Furthermore, we prevent workers from making mutually destructive suggestions by having a target network provide singular guidance.

In Section 3.2, an analogy was drawn between PPO's mini-batching mechanism and the circular buffer.

Our primary benchmark for target update frequency is n = N · K, where N is circular buffer size and K is maximum replay coefficient.

This is the case when PPO is equivalent to IMPACT.

In Figure 4 (b), we test the frequency of updates with varying orders of magnitudes of n. In general, we find that agent performance is robust to vastly differing frequencies.

However, when n = 1 ∼ 4, the agent does not learn.

Based on empirical results, we theorize that the agent is able to train as long as a stable trust region can be formed.

On the other hand, if update frequency is too low, the agent is stranded for many iterations in the same trust region, which impairs learning speed.

Counter to intuition, the tradeoff between time and sample efficiency when K increases is not necessarily true.

In Figure 4b and 4c, we show that IMPACT realizes greater gains by striking the balance between high sample throughput and sample efficiency.

When K = 2, IMPACT performs the best in both time and sample efficiency.

Our results reveal that wall-clock time and sample efficiency can be optimized based on tuning values of K in the circular buffer.

We investigate how IMPACT attains greater performance in wall clock-time and sample efficiency compared with PPO and IMPALA across six different continuous control and discrete action tasks.

We tested the agent on three continuous environments ( Figure 5 ): HalfCheetah, Hopper, and Humanoid on 16 CPUs and 1 GPU.

The policy networks consist of two fully-connected layers of 256 units with nonlinear activation tanh.

The critic network shares the same architecture as the policy network.

For consistentency, same network architectures were employed across PPO, IMPALA, and IMPACT.

For the discrete environments (Figure 6 ), Pong, SpaceInvaders, and Breakout were chosen as common benchmarks used in popular distributed RL libraries (Caspi et al., 2017; Liang et al., 2018) .

Additional experiments for discrete environments are in the Appendix.

These experiments were ran on 32 CPUs and 1 GPU.

The policy network consists of three 4x4 and one 11x11 conv layer, with nonlinear activation ReLU.

The critic network shares weights with the policy network.

The input of the network is a stack of four 42x42 down-sampled images of the Atari environment.

The hyper-parameters for continuous and discrete environments are listed in the Appendix B table 1 and 2 respectively.

Figures 5 and 6 show the total average return on evaluation rollouts for IMPACT, IMPALA and PPO.

We train each algorithm with three different random seeds on each environment for a total time of three hours.

According to the experiments, IMPACT is able to train much faster than PPO and IMPALA in both discrete and continuous domains, while preserving same or better sample efficiency than PPO.

Our results reveal that continuous control tasks for IMPACT are sensitive to the tuple (N, K) for the circular buffer.

N = 16 and K = 20 is a robust choice for continuous control.

Although higher K inhibits workers' sample throughput, increased sample efficiency from replaying experiences results in an overall reduction in training wall-clock time and higher reward.

For discrete tasks, N = 1 and K = 2 works best.

Empirically, agents learn faster from new experience than replaying old experience, showing how exploration is crucial to achieving high asymptotic performance in discrete enviornments.

Figure 7: Performance of IMPACT with respect to the number of workers in both continuous and discrete control tasks Figure 7 shows how IMPACT's performance scales relative to the number of workers.

More workers means increased sample throughput, which in turn increases training throughput (the rate that learner consumes batches).

With the learner consuming more worker data per second, IMPACT can attain better performance in less time.

However, as number of workers increases, observed increases in performance begin to decline.

Distributed RL architectures are often used to accelerate training.

Gorila (Nair et al., 2015) and A3C (Mnih et al., 2016) use workers to compute gradients to be sent to the learner.

A2C (Mnih et al., 2016) and IMPALA (Espeholt et al., 2018) send experience tuples to the learner.

Distributed replay buffers, introduced in ACER (Wang et al., 2016) and Ape-X (Horgan et al., 2018) , collect worker-collected experience and define an overarching heuristic for learner batch selection.

IMPACT is the first to fully incorporate the sample-efficiency benefits of PPO in an asynchronous setting.

Surreal PPO (Fan et al., 2018 ) also studies training with PPO in the asynchronous setting, but do not consider adaptation of the surrogate objective nor IS-correction.

Their use of a target network for broadcasting weights to workers is also entirely different from IMPACT's.

Consequently, IMPACT is able to achieve better results in both real-time and sample efficiency.

Off-policy methods, including DDPG (Lillicrap et al., 2015) and QProp, utilize target networks to stabilize learning the Q function (Lillicrap et al., 2015; Gu et al., 2016) .

This use of a target network is related but different from IMPACT, which uses the network to define a stable trust region for the PPO surrogate objective.

In conclusion, we introduce IMPACT, which extends PPO with a stabilized surrogate objective for asynchronous optimization, enabling greater real-time performance without sacrificing timestep efficiency.

We show the importance of the IMPACT objective to stable training, and show it can outperform tuned PPO and IMPALA baselines in both real-time and timestep metrics.

Time ( In Figure 9 , we gradually add components to IMPALA until the agent is equivalent to IMPACT's.

Starting from IMPALA, we gradually add PPO's objective function, circular replay buffer, and target-worker clipping.

In particular, IMPALA with PPO's objective function and circular replay buffer is equivalent to an asynchronous-variant of PPO (APPO).

APPO fails to perform as well as synchronous distributed PPO, since PPO is an on-policy algorithm.

In Figure 6 , IMPALA performs substantially worse than other agents in continuous environments.

We postulate that IMPALA suffers from low asymptotic performance here since its objective is an importance-sampled version of the Vanilla Policy Gradient (VPG) objective, which is known to suffer from high variance and large update-step sizes.

We found that for VPG, higher learning rates encourage faster learning in the beginning but performance drops to negative return later in training.

In Appendix B, for IMPALA, we heavily tuned on the learning rate, finding that small learning rates stabilize learning at the cost of low asymptotic performance.

Prior work also reveals the agents that use VPG fail to attain good performance in non-trivial continuous tasks (Achiam, 2018) .

Our results with IMPALA reaches similar performance compared to other VPG-based algorithms.

The closest neighbor to IMPALA, A3C uses workers to compute gradients from the VPG objective to send to the learner thread.

A3C performs well in InvertedPendulum yet flounders in continuous environments (Tassa et al., 2018) .

The following ratios represent the objective functions for different ablation studies.

In the plots (Figure 10 ), we set the advantage function to be one, i.e.

Â t = 1.

• IS ratio: According to Figure 10 , IS ratio is large when π workeri assigns low probability.

IMPACT target -clip is a lower bound of the PPO -clip.

In an distributed asynchronous setting, the trust region suffers from larger variance stemming from off-policy data.

IMPACT target -clip ratio mitigates this by encouraging conservative and reasonable policy-gradient steps.

<|TLDR|>

@highlight

IMPACT helps RL agents train faster by decreasing training wall-clock time and increasing sample efficiency simultaneously.
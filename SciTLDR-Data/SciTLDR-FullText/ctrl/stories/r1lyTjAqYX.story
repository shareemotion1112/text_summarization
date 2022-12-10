Building on the recent successes of distributed training of RL agents, in this paper we investigate the training of RNN-based RL agents from distributed prioritized experience replay.

We study the effects of parameter lag resulting in representational drift and recurrent state staleness and empirically derive an improved training strategy.

Using a single network architecture and fixed set of hyper-parameters, the resulting agent, Recurrent Replay Distributed DQN, quadruples the previous state of the art on Atari-57, and matches the state of the art on DMLab-30.

It is the first agent to exceed human-level performance in 52 of the 57 Atari games.

Reinforcement Learning (RL) has seen a rejuvenation of research interest recently due to repeated successes in solving challenging problems such as reaching human-level play on Atari 2600 games BID15 , beating the world champion in the game of Go BID21 , and playing competitive 5-player DOTA BID18 .

The earliest of these successes leveraged experience replay for data efficiency and stacked a fixed number of consecutive frames to overcome the partial observability in Atari 2600 games.

However, with progress towards increasingly difficult, partially observable domains, the need for more advanced memory-based representations increases, necessitating more principled solutions such as recurrent neural networks (RNNs).

The use of LSTMs BID8 within RL has been widely adopted to overcome partial observability BID5 BID16 BID3 BID4 .In this paper we investigate the training of RNNs with experience replay.

We have three primary contributions.

First, we demonstrate the effect of experience replay on parameter lag, leading to representational drift and recurrent state staleness.

This is potentially exacerbated in the distributed training setting, and ultimately results in diminished training stability and performance.

Second, we perform an empirical study into the effects of several approaches to RNN training with experience replay, mitigating the aforementioned effects.

Third, we present an agent that integrates these findings to achieve significant advances in the state of the art on Atari-57 BID1 and matches the state of the art on DMLab-30 BID0 .

To the best of our knowledge, our agent, Recurrent Replay Distributed DQN (R2D2), is the first to achieve this using a single network architecture and fixed set of hyper-parameters.

Our work is set within the Reinforcement Learning (RL) framework BID23 , in which an agent interacts with an environment to maximize the sum of discounted, γ ∈ [0, 1), rewards.

We model the environment as a Partially Observable Markov Decision Process (POMDP) given by the tuple (S, A, T , R, Ω, O) BID17 BID10 BID12 .

The underlying Markov Decision Process (MDP) is defined by (S, A, T , R), where S is the set of states, A the set of actions, T a transition function mapping state-actions to probability distributions over next states, and R : S × A → R is the reward function.

Finally, Ω gives the set of observations 2.2 DISTRIBUTED REINFORCEMENT LEARNING Recent advances in reinforcement learning have achieved significantly improved performance by leveraging distributed training architectures which separate learning from acting, collecting data from many actors running in parallel on separate environment instances BID3 BID4 BID18 a; BID11 .Distributed replay allows the Ape-X agent to decouple learning from acting, with actors feeding experience into the distributed replay buffer and the learner receiving (randomized) training batches from it.

In addition to distributed replay with prioritized sampling , Ape-X uses n-step return targets BID22 , the double Q-learning algorithm , the dueling DQN network architecture BID26 and 4-framestacking.

Ape-X achieved state-of-the-art performance on Atari-57, significantly out-performing the best single-actor algorithms.

It has also been used in continuous control domains and again showed state-of-the-art results, further demonstrating the performance benefits of distributed training in RL.IMPALA BID3 ) is a distributed reinforcement learning architecture which uses a first-in-first-out queue with a novel off-policy correction algorithm called V-trace, to learn sequentially from the stream of experience generated by a large number of independent actors.

IMPALA stores sequences of transitions along with an initial recurrent state in the experience queue, and since experience is trained on exactly once, this data generally stays very close to the learner parameters.

BID3 showed that IMPALA could achieve strong performance in the Atari-57 and DMLab-30 benchmark suites, and furthermore was able to use a single large network to learn all tasks in a benchmark simultaneously while maintaining human-level performance.

We propose a new agent, the Recurrent Replay Distributed DQN (R2D2), and use it to study the interplay between recurrent state, experience replay, and distributed training.

R2D2 is most similar to Ape-X, built upon prioritized distributed replay and n-step double Q-learning (with n = 5), generating experience by a large number of actors (typically 256) and learning from batches of replayed experience by a single learner.

Like Ape-X, we use the dueling network architecture of BID26 , but provide an LSTM layer after the convolutional stack, similarly to BID4 .Instead of regular (s, a, r, s ) transition tuples, we store fixed-length (m = 80) sequences of (s, a, r) in replay, with adjacent sequences overlapping each other by 40 time steps, and never crossing episode boundaries.

When training, we unroll both online and target networks BID15 on the same sequence of states to generate value estimates and targets.

We leave details of our exact treatment of recurrent states in replay for the next sections.

Like Ape-X, we use 4-frame-stacks and the full 18-action set when training on Atari.

On DMLab, we use single RGB frames as observations, and the same action set discretization as BID7 .

Following the modified Ape-X version in BID19 , we do not clip rewards, but instead use an invertible value function rescaling of the form h(x) = sign(x)( |x| + 1 − 1) + x which results in the following n-step targets for the Q-value function: DISPLAYFORM0 Here, θ − denotes the target network parameters which are copied from the online network parameters θ every 2500 learner steps.

Our replay prioritization differs from that of Ape-X in that we use a mixture of max and mean absolute n-step TD-errors δ i over the sequence: p = η max i δ i + (1 − η)δ.

We set η and the priority exponent to 0.9.

This more aggressive scheme is motivated by our observation that averaging over long sequences tends to wash out large errors, thereby compressing the range of priorities and limiting the ability of prioritization to pick out useful experience.

Finally, compared to Ape-X, we used the slightly higher discount of γ = 0.997, and disabled the loss-of-life-as-episode-end heuristic that has been used in Atari agents in some of the work since BID15 .

A full list of hyper-parameters is provided in the Appendix.

We train the R2D2 agent with a single GPU-based learner, performing approximately 5 network updates per second (each update on a mini-batch of 64 length-80 sequences), and each actor performing ∼ 260 environment steps per second on Atari (∼ 130 per second on DMLab).

In order to achieve good performance in a partially observed environment, an RL agent requires a state representation that encodes information about its state-action trajectory in addition to its current observation.

The most common way to achieve this is by using an RNN, typically an LSTM BID8 , as part of the agent's state encoding.

To train an RNN from replay and enable it to learn meaningful long-term dependencies, whole state-action trajectories need to be stored in replay and used for training the network.

BID5 compared two strategies of training an LSTM from replayed experience:• Using a zero start state to initialize the network at the beginning of sampled sequences.• Replaying whole episode trajectories.

The zero start state strategy's appeal lies in its simplicity, and it allows independent decorrelated sampling of relatively short sequences, which is important for robust optimization of a neural network.

On the other hand, it forces the RNN to learn to recover meaningful predictions from an atypical initial recurrent state ('initial recurrent state mismatch'), which may limit its ability to fully rely on its recurrent state and learn to exploit long temporal correlations.

The second strategy on the other hand avoids the problem of finding a suitable initial state, but creates a number of practical, computational, and algorithmic issues due to varying and potentially environment-dependent sequence length, and higher variance of network updates because of the highly correlated nature of states in a trajectory when compared to training on randomly sampled batches of experience tuples.

BID5 observed little difference between the two strategies for empirical agent performance on a set of Atari games, and therefore opted for the simpler zero start state strategy.

One possible explanation for this is that in some cases, an RNN tends to converge to a more 'typical' state if allowed a certain number of 'burn-in' steps, and so recovers from a bad initial recurrent state on a sufficiently long sequence.

We also hypothesize that while the zero start state strategy may suffice in the mostly fully observable Atari domain, it prevents a recurrent network from learning actual long-term dependencies in more memory-critical domains (e.g. on DMLab).To fix these issues, we propose and evaluate two strategies for training a recurrent neural network from randomly sampled replay sequences, that can be used individually or in combination: DISPLAYFORM0 Figure 1: Top row shows Q-value discrepancy ∆Q as a measure for recurrent state staleness.

(a) Diagram of how ∆Q is computed, with green box indicating a whole sequence sampled from replay.

For simplicity, l = 0 (no burn-in).

(b) ∆Q measured at first state and last state of replay sequences, for agents training on a selection of DMLab levels (indicated by initials) with different training strategies.

Bars are averages over seeds and through time indicated by bold line on x-axis in bottom row.

(c) Learning curves on the same levels, varying the training strategy, and averaged over 3 seeds.• Stored state: Storing the recurrent state in replay and using it to initialize the network at training time.

This partially remedies the weakness of the zero start state strategy, however it may suffer from the effect of 'representational drift' leading to 'recurrent state staleness', as the stored recurrent state generated by a sufficiently old network could differ significantly from a typical state produced by a more recent version.• Burn-in: Allow the network a 'burn-in period' by using a portion of the replay sequence only for unrolling the network and producing a start state, and update the network only on the remaining part of the sequence.

We hypothesize that this allows the network to partially recover from a poor start state (zero, or stored but stale) and find itself in a better initial state before being required to produce accurate outputs.

In all our experiments we will be using the proposed agent architecture from Section 2.3 with replay sequences of length m = 80, with an optional burn-in prefix of l = 40 or 20 steps.

Our aim is to assess the negative effects of representational drift and recurrent state staleness on network training and how they are mitigated by the different training strategies.

For that, we will compare the Qvalues produced by the network on sampled replay sequences when unrolled using one of these strategies and the Q-values produced when using the true stored recurrent states at each step (see Figure 1a , showing different sources for the hidden state).More formally, let o t , . . .

, o t+m and h t , . . .

, h t+m denote the replay sequence of observations and stored recurrent states, and denote by h t+1 = h(o t , h t ; θ) and q(h t ; θ) the recurrent state and Qvalue vector output by the recurrent neural network with parameter vector θ, respectively.

We writê h t for the hidden state, used during training and initialized under one of the above strategies (either DISPLAYFORM1 is computed by unrolling the network with parametersθ on the sequence o t , . . .

, o t+l+m−1 .

We estimate the impact of representational drift and recurrent state staleness by their effect on the Q-value estimates, by measuring Q-value discrepancy DISPLAYFORM2 for the first (i = l) and last (i = l + m − 1) states of the non-burn-in part of the replay sequence (see Figure 1a for an illustration).

The normalization by the maximal Q-value helps comparability between different environments and training stages, as the Q-value range of an agent can vary dras-tically between these.

Note that we are not directly comparing the Q-values produced at acting and training time, q(h t ; θ) and q(ĥ t ;θ), as these can naturally be expected to be distinct as the agent is being trained.

Instead we focus on the difference that results from applying the same network (parameterized byθ) to the distinct recurrent states.

In Figure 1b , we are comparing agents trained with the different strategies on several DMLab environments in terms of this proposed metric.

It can be seen that the zero start state heuristic results in a significantly more severe effect of recurrent state staleness on the outputs of the network.

As hypothesized above, this effect is greatly reduced for the last sequence states compared to the first ones, after the RNN has had time to recover from the atypical start state, but the effect of staleness is still substantially worse here for the zero state than the stored state strategy.

Another potential downside of the pure zero state heuristic is that it prevents the agent from strongly relying on its recurrent state and exploit long-term temporal dependencies, see Section 5.We observe that the burn-in strategy on its own partially mitigates the staleness problem on the initial part of replayed sequences, while not showing a significant effect on the Q-value discrepancy for later sequence states.

Empirically, this translates into noticeable performance improvements, as can be seen in Figure 1c .

This itself is noteworthy, as the only difference between the pure zero state and the burn-in strategy lies in the fact that the latter unrolls the network over a prefix of states on which the network does not receive updates.

In informal experiments (not shown here) we observed that this is not due to the different unroll lengths themselves (i.e., the zero state strategy without burn-in, on sequences of length l + m, performed worse overall).

We hypothesize that the beneficial effect of burn-in lies in the fact that it prevents 'destructive updates' to the RNN parameters resulting from highly inaccurate initial outputs on the first few time steps after a zero state initialization.

The stored state strategy, on the other hand, proves to be overall much more effective at mitigating state staleness in terms of the Q-value discrepancy, which also leads to clearer and more consistent improvements in empirical performance.

Finally, the combination of both methods consistently yields the smallest discrepancy on the last sequence states and the most robust performance gains.

We conclude the section with the observation that both stored state and burn-in strategy provide substantial advantages over the naive zero state training strategy, in terms of (indirect) measures of the effect of representation drift and recurrent state staleness, and empirical performance.

Since they combine beneficially, we use both of these strategies (with burn-in length of l = 40) in the empirical evaluation of our proposed agent in Section 4.

Additional results on the effects of distributed training on representation drift and Q-value discrepancy are given in the Appendix.

In this section we evaluate the empirical performance of R2D2 on two challenging benchmark suites for deep reinforcement learning: Atari-57 BID1 and DMLab-30 BID0 .

One of the fundamental contributions of Deep Q-Networks (DQN) BID15 was to set as standard practice the use of a single network architecture and set of hyper-parameters across the entire suite of 57 Atari games.

Unfortunately, expanding past Atari this standard has not been maintained and, to the best of our knowledge, at present there is no algorithm applied to both Atari-57 and DMLab-30 under this standard.

In particular, we will compare performance with Ape-X and IMPALA for which hyper-parameters are tuned separately for each benchmark.

For R2D2, we use a single neural network architecture and a single set of hyper-parameters across all experiments.

This demonstrates greater robustness and generality than has been previously observed in deep RL.

It is also in pursuit of this generality, that we decided to disable the (Atari-specific) heuristic of treating life losses as episode ends, and did not apply reward clipping.

Despite this, we observe state-of-the-art performance in both Atari and DMLab, validating the intuitions derived from our empirical study.

A more detailed ablation study of the effects of these modifications is presented in the Appendix.

The Atari-57 benchmark is built upon the Arcade Learning Environment (ALE) BID1 , and consists of 57 classic Atari 2600 video games.

Initial human-level performance was .

Right: Example individual learning curves of R2D2, averaged over 3 seeds, and Ape-X, single seed.achieved by DQN BID15 , and since then RL agents have improved significantly through both algorithmic and architectural advances.

Currently, state of the art for a single actor is achieved by the recent distributional reinforcement learning algorithms IQN and Rainbow BID6 , and for multi-actor results, Ape-X .Figure 2 (left) shows the median human-normalized scores across all games for R2D2 and related methods (see Appendix for full Atari-57 scores and learning curves).

R2D2 achieves an order of magnitude higher performance than all single-actor agents and quadruples the previous state-ofthe-art performance of Ape-X using fewer actors (256 instead of 360), resulting in higher sampleand time-efficiency.

Table 1 lists mean and median human-normalized scores for R2D2 and other algorithms, highlighting these improvements.

In addition to achieving state-of-the-art results on the entire task suite, R2D2 also achieves the highest ever reported agent scores on a large fraction of the individual Atari games, in many cases 'solving' the respective games by achieving the highest attainable score.

In FIG0 (right) we highlight some of these individual learning curves of R2D2.

As an example, notice the performance on MS.PACMAN is even greater than that of the agent reported in BID25 , which was engineered specifically for this game.

Furthermore, we notice that Ape-X achieves super-human performance for the same number of games as Rainbow (49) , and that its improvements came from improving already strong scores.

R2D2 on the other hand is super-human on 52 out of 57 games.

Of those remaining, we anecdotally observed that three (SKIING, SOLARIS, and PRIVATE EYE) can reach super-human performance with higher discount rates and faster target network updates.

The other two (MONTEZUMA'S REVENGE and PITFALL) are known hard exploration problems, and solving these with a general-purpose algorithm will likely require new algorithmic insights.

DMLab-30 is a suite of 30 problems set in a 3D first-person game engine, testing for a wide range of different challenges BID0 .

While Atari can largely be approached with only frame-stacking, DMLab-30 requires long-term memory to achieve reasonable performance.

Perhaps because of this, and the difficulty of integrating recurrent state with experience replay, topperforming agents have, to date, always come in the form of actor-critic algorithms trained in (near) on-policy settings.

For the first time we show state-of-the-art performance on DMLab-30 using a value-function-based agent.

We stress that, different from the state-of-the-art IMPALA architecture BID3 , the R2D2 agent uses the same set of hyper-parameters here as on Atari.

Here we are mainly interested in comparing to the IMPALA 'experts', not to its multi-task variant.

Since the original IMPALA experts were trained on a smaller amount of data (approximately 333M Figure 3 : DMLab-30 comparison of R2D2 and R2D2+ with our re-run of IMPALA shallow and deep in terms of mean-capped human-normalized score BID3 .

DMLab-30 Human-Normalized Score Median Mean Median Mean-Capped Ape-X 434.1% 1695.6% --Reactor BID4 187.0% ---IMPALA, deep BID3 191 Table 1 :

Comparison of Atari-57 and DMLab-30 results.

R2D2 average final score over 3 seeds (1 seed for feed-forward variant), IMPALA final score over 1 seed, Ape-X best training score with 1 seed.

Our re-run of IMPALA uses the same improved action set from BID7 as R2D2, and is trained for a comparable number of environment frames (10B frames; the original IMPALA experts in BID3 were only trained for approximately 333M frames).

R2D2+ refers to the adapted R2D2 variant matching deep IMPALA's 15-layer ResNet architecture and asymmetric reward clipping, as well as using a shorter target update period of 400.environment frames) and since R2D2 uses the improved action set introduced in BID7 , we decided to re-run the IMPALA agent with improved action set and for a comparable training time (10B environment frames) for a fairer comparison, resulting in substantially improved scores for the IMPALA agent compared to the original in BID3 , see Table 1 .Figure 3 compares R2D2 with IMPALA.

We note that R2D2 exceeds the performance of the (shallow) IMPALA version, despite using the exact same set of hyper-parameters and architecture as the variant trained on Atari, and in particular not using the 'optimistic asymmetric reward clipping' used by all IMPALA agents 1 .To demonstrate the potential of our agent, we also devise a somewhat adapted R2D2 version for DMLab only (R2D2+) by adding asymmetric reward clipping, using the 15-layer ResNet from IMPALA (deep), and reducing the target update frequency from 2500 to 400 for better sample efficiency.

To fit the larger model in GPU memory, we reduced the batch size from 64 to 32 in these runs only.

We observe that this modified version yields further substantial improvements over standard R2D2 and matches deep IMPALA in terms of sample efficiency as well as asymptotic performance.

Both our re-run of deep IMPALA and R2D2+ are setting new state-of-the-art scores on the DMLab-30 benchmark.

Atari-57 is a class of environments which are almost fully observable (given 4-frame-stack observations), and agents trained on it are not necessarily expected to strongly benefit from a memoryaugmented representation.

The main algorithmic difference between R2D2 and its predecessor, Ape-X, is the use of a recurrent neural network, and it is therefore surprising by how large a margin R2D2 surpasses the previous state of the art on Atari.

In this section we analyze the role of the LSTM network and other algorithmic choices for the high performance of the R2D2 agent.

Since the performance of asynchronous or distributed RL agents can depend on subtle implementational details and even factors such as precise hardware setup, it is impractical to perform a direct comparison to the Ape-X agent as reported in .

Instead, here we verify that the LSTM and its training strategy play a crucial role for the success of R2D2 by a comparison of the R2D2 agent with a purely feed-forward variant, all other parameters held fixed.

Similarly, we consider the performance of R2D2 using reward clipping without value rescaling (Clipped) and using a smaller discount factor of γ = 0.99 (Discount).

The ablation results in Figure 4 show very clearly that the LSTM component is crucial for boosting the agent's peak performance as well as learning speed, explaining much of the performance difference to Ape-X. Other design choices have more mixed effects, improving in some games and hurting performance in others.

Full ablation results (in particular, an ablation over the full Atari-57 suite of the feed-forward agent variant, as well as an ablation of the use of the life-loss-as-episode-termination heuristic) are presented in the Appendix.

In our next experiment we test to what extent the R2D2 agent relies on its memory, and how this is impacted by the different training strategies.

For this we select the Atari game MS.PACMAN, on which R2D2 shows state-of-the-art performance despite the game being virtually fully observable, and the DMLab task EMSTM WATERMAZE, which strongly requires the use of memory.

We train two agents on each game, using the zero and stored state strategies, respectively.

We then evaluate these agents by restricting their policy to a fixed history length: at time step t, their policy uses an LSTM unrolled over time steps o t−k+1 , . . .

, o t , with the hidden state h t−k replaced by zero instead of the actual hidden state (note this is only done for evaluation, not at training time of the agents).In Figure 5 (left) we decrease the history length k from ∞ (full history) down to 0 and show the degradation of agent performance (measured as mean score over 10 episodes) as a function of k. We additionally show the difference of max-Q-values and the percentage of correct greedy actions (where the unconstrained variant is taken as ground truth).We observe that restricting the agent's memory gradually decreases its performance, indicating its nontrivial use of memory on both domains.

Crucially, while the agent trained with stored state shows higher performance when using the full history, its performance decays much more rapidly than for the agent trained with zero start states.

This is evidence that the zero start state strategy, used in past RNN-based agents with replay, limits the agent's ability to learn to make use of its memory.

While this doesn't necessarily translate into a performance difference (like in MS.PACMAN), it does so whenever the task requires an effective use of memory (like EMSTM WATERMAZE).

This advantage of the stored state compared to the zero state strategy may explain the large performance difference between R2D2 and its close cousin Reactor BID4 , which trains its LSTM policy from replay with the zero state strategy.

Finally, the right and middle columns of Figure 5 show a monotonic decrease of the quality of Qvalues and the resulting greedy policy as the available history length k is decreased to 0, providing a simple causal link between the constraint and the empirical agent performance.

For a qualitative comparison of different behaviours learned by R2D2 and its feed-forward variant, we provide several agent videos at https://bit.ly/r2d2600.

Here we take a step back from evaluating performance and discuss our empirical findings in a broader context.

There are two surprising findings in our results.

First, although zero state initialization was often used in previous works BID5 BID4 , we have found that it leads to misestimated action-values, especially in the early states of replayed sequences.

Moreover, without burn-in, updates through BPTT to these early time steps with poorly estimated outputs seem to give rise to destructive updates and hinder the network's ability to recover from sub-optimal initial recurrent states.

This suggests that either the context-dependent recurrent state should be stored along with the trajectory in replay, or an initial part of replayed sequences should be reserved for burn-in, to allow the RNN to rely on its recurrent state and exploit long-term temporal dependencies, and the two techniques can also be combined beneficially.

We have also observed that the underlying problems of representational drift and recurrent state staleness are potentially exacerbated in the distributed setting (see Appendix), highlighting the importance of robustness to these effects through an adequate training strategy of the RNN.Second, we found that the impact of RNN training goes beyond providing the agent with memory.

Instead, RNN training also serves a role not previously studied in RL, potentially by enabling better representation learning, and thereby improves performance even on domains that are fully observable and do not obviously require memory (cf.

BREAKOUT results in the feed-forward ablation).Finally, taking a broader view on our empirical results, we note that scaling up of RL agents through parallelization and distributed training allows them to benefit from huge experience throughput and achieve ever-increasing results over broad simulated task suites such as Atari-57 and DMLab-30.

Impressive as these results are in terms of raw performance, they come at the price of high sample complexity, consuming billions of simulated time steps in hours or days of wall-clock time.

One widely open avenue for future work lies in improving the sample efficiency of these agents, to allow applications to domains that do not easily allow fast simulation at similar scales.

Another remaining challenge, very apparent in our results on Atari-57, is exploration: Save for the hardest-exploration games from Atari-57, R2D2 surpasses human-level performance on this task suite significantly, essentially 'solving' many of the games therein.

Figure 6 : Left: Parameter lag experienced with distributed prioritized replay with (top) 256 and (bottom) 64 actors on four DMLab levels: explore obstructed goals large (eogl), explore object rewards many (eorm), lasertag three opponents small (lots), rooms watermaze (rw).

Center: initialstate and Right: final-state Q-value discrepancy for the same set of experiments.

In this section, we investigate the effects of distributed training of an agent using a recurrent neural network, where a large number of actors feed their experience into a replay buffer for a single learner.

On the one hand, the distributed setting typically presents a less severe problem of representational drift than the single-actor case, such as the one studied in BID5 .

This is because in relative terms, the large amount of generated experience is replayed less frequently (on average, an experience sample is replayed less than once in the Ape-X agent, compared to eight times in DQN), and so distributed agent training tends to give rise to a smaller degree of 'parameter lag' (the mean age, in parameter updates, of the network parameters used to generate an experience, at the time it is being replayed).On the other hand, the distributed setting allows for easy scaling of computational resources according to hardware or time constraints.

An ideal distributed agent should therefore be robust to changes in, e.g., the number of actors, without careful parameter re-tuning.

As we have seen in the previous section, RNN training from replay is sensitive to the issue of representational drift, the severity of which can depend on exactly these parameters.

To investigate these effects, we train the R2D2 agent with a substantially smaller number of actors.

This has a direct (inversely proportional) effect on the parameter lag (see Figure 6 (left)).

Specifically, in our experiments, as the number of actors is changed from 256 to 64, the mean parameter lag goes from 1500 to approximately 5500 parameter updates, which in turn impacts the magnitude of representation drift and recurrent state staleness, as measured by ∆Q in Section 3.The right two columns in Figure 6 show an overall increase of the average ∆Q for the smaller number of actors, both for first and last states of replayed sequences.

This supports the above intuitions and highlights the increased importance of an improved training strategy (compared to the zero state strategy) in the distributed training setting, if a certain level of empirical agent performance is to be maintained across ranges of extrinsic and potentially hardware dependent parameters. . 'reset' refers to the agent variant using life losses as full episode terminations (preventing value function bootstrapping across life loss events, as well as resetting the LSTM state), whereas 'roll' only prevents value function bootstrapping, but unrolls the LSTM for the duration of a full episode (potentially spanning multiple life losses).

In this section we give additional experimental results supporting our empirical study in the main text.

FIG2 gives a more in-depth view of the ablation results from Figure 4 .

We see that, with the exception of the feed-forward ablation, there are always games in which the ablated choice performs better.

Our choice of architecture and configuration optimizes for overall performance and general (cross-domain) applicability, but for individual games there are different configurations that would yield improved performance.

Additionally, in FIG4 we compare R2D2 with variants using the life loss signal as episode termination.

Both ablation variants interrupt value function bootstrapping past the life loss events, but differ in that one ('reset') also resets the LSTM state at these events, whereas the other ('roll') only resets the LSTM state at actual episode boundaries, like regular R2D2.

Despite the fact that the life loss heuristic is generally helpful to speed up learning in Atari, we did not use it in our main R2D2 agent for the sake of generality of the algorithm.

Atari-57 -Human-normalized Median

Ape-X R2D2, FF Rainbow Reactor Figure 9 : Comparing sample efficiency between state-of-the-art agents on Atari-57.

We observe a general trend of increasing final performance being negatively correlated with sample efficiency, which holds for all four algorithms compared.

In Figure 9 we compare the sample efficiency of R2D2 with recent state-of-the-art agents on Atari-57 in terms of human-normalized median score.

As expected, the more distributed agents have worse sample efficiency early on, but also much improved long-term performance.

This is an interesting correlation on its own, but we add that R2D2 appears to achieve a qualitatively different performance curve than any of the other algorithms.

Note that, while Ape-X has a larger number of actors than R2D2 (360 compared to 256), its learner processes approximately 20 batches of size 512 per second, whereas R2D2 performs updates on batches of 64 × 80 observations (batch size × sequence length), at a rate of approximately 5 per second.

This results in a reduced 'replay ratio' (effective number of times each experienced observation is being replayed): On average, Ape-X replays each observation approximately 1.3 times, whereas this number is only about 0.8 for R2D2, which explains the initial sample efficiency advantage of Ape-X.

R2D2 uses the same 3-layer convolutional network as DQN BID15 , followed by an LSTM with 512 hidden units, which feeds into the advantage and value heads of a dueling network BID26 , each with a hidden layer of size 512.

Additionally, the LSTM receives as input the reward and one-hot action vector from the previous time step.

On the four language tasks in the DMLab suite, we are using the same additional language-LSTM with 64 hidden units as IMPALA BID3 Target network update interval 2500 updates Value function rescaling h(x) = sign(x)( |x| + 1 − 1) + x, = 10 −3 Table 2 : Hyper-parameters values used in R2D2.

All missing parameters follow the ones in Ape-X .As is usual for agent training on Atari since BID15 , we cap all (training and evaluation) episodes at 30 minutes (108, 000 environment frames Table 3 : Performance of R2D2 and R2D2+, averaged over 3 seeds, compared to our own singleseed re-run of IMPALA (shallow/deep) with improved action-set and trained on the same amount of data (10B environment frames).

Compared to standard R2D2, the R2D2+ variant uses a shorter target network update frequency (400 compared to 2500), as well as the substantially larger 15-layer ResNet and the custom 'optimistic asymmetric reward clipping' from BID3

<|TLDR|>

@highlight

Investigation on combining recurrent neural networks and experience replay leading to state-of-the-art agent on both Atari-57 and DMLab-30 using single set of hyper-parameters.
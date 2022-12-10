Deep learning has achieved remarkable successes in solving challenging reinforcement learning (RL) problems when dense reward function is provided.

However, in sparse reward environment it still often suffers from the need to carefully shape reward function to guide policy optimization.

This limits the applicability of RL in the real world since both reinforcement learning and domain-specific knowledge are required.

It is therefore of great practical importance to develop algorithms which can learn from a binary signal indicating successful task completion or other unshaped, sparse reward signals.

We propose a novel method called competitive experience replay, which efficiently supplements a sparse reward by placing learning in the context of an exploration competition between a pair of agents.

Our method complements the recently proposed hindsight experience replay (HER) by inducing an automatic exploratory curriculum.

We evaluate our approach on the tasks of reaching various goal locations in an ant maze and manipulating objects with a robotic arm.

Each task provides only binary rewards indicating whether or not the goal is achieved.

Our method asymmetrically augments these sparse rewards for a pair of agents each learning the same task, creating a competitive game designed to drive exploration.

Extensive experiments demonstrate that this method leads to faster converge and improved task performance.

Recent progress in deep reinforcement learning has achieved very impressive results in domains ranging from playing games BID28 BID39 BID31 , to high dimensional continuous control , and robotics BID21 BID2 .Despite these successes, in robotics control and many other areas, deep reinforcement learning still suffers from the need to engineer a proper reward function to guide policy optimization (see e.g. BID36 BID29 .

In robotic control as stacking bricks, reward function need to be sharped to very complex which consists of multiple terms BID36 .

It is extremely hard and not applicable to engineer such reward function for each task in real world since both reinforcement learning expertise and domain-specific knowledge are required.

Learning to perform well in environments with sparse rewards remains a major challenge.

Therefore, it is of great practical importance to develop algorithms which can learn from binary signal indicating successful task completion or other unshaped reward signal.

In environments where dense reward function is not available, only a small fraction of the agents' experiences will be useful to compute gradient to optimize policy, leading to substantial high sample complexity.

Providing agents with useful signals to pursue in sparse reward environments becomes crucial in these scenarios.

In the domain of goal-directed RL, the recently proposed hindsight experience replay (HER) BID0 addresses the challenge of learning from sparse rewards by re-labelling visited states as goal states during training.

However, this technique continues to suffer from sample inefficiency, ostensibly due to difficulties related to exploration.

In this work, we address these limitations by introducing a method called Competitive Experience Replay (CER).

This technique attempts to emphasize exploration by introducing a competition between two agents attempting to learn the same task.

Intuitively, agent A (the agent ultimately used for evaluation) receives a penalty for visiting states that the competitor agent (B) also visits; and B is rewarded for visiting states found by A. Our approach maintains the reward from the original task such that exploration is biased towards the behaviors best suited to accomplishing the task goals.

We show that this competition between agents can automatically generate a curriculum of exploration and shape otherwise sparse reward.

We jointly train both agents' policies by adopting methods from multi-agent RL.

In addition, we propose two versions of CER, independent CER, and interact CER, which differ in the state initialization of agent B: whether it is sampled from the initial state distribution or sampled from off-policy samples of agent A, respectively.

Whereas HER re-labels samples based on an agent's individual rollout, our method re-labels samples based on intra-agent behavior; as such, the two methods do not interfere with each other algorithmically and are easily combined during training.

We evaluate our method both with and without HER on a variety of reinforcement learning tasks, including navigating an ant agent to reach a goal position and manipulating objects with a robotic arm.

For each such task the default reward is sparse, corresponding to a binary indicator of goal completion.

Ablation studies show that our method is important for achieving a high success rate and often demonstrates faster convergence.

Interestingly, we find that CER and HER are complementary methods and employ both to reach peak efficiency and performance.

Furthermore, we observe that, when combined with HER, CER outperforms curiosity-driven exploration.

Here, we provide an introduction to the relevant concepts for reinforcement learning with sparse reward (Section 2.1), Deep Deterministic Policy Gradient, the backbone algorithm we build off of, (Section 2.2), and Hindsight Experience Replay (Section 2.3).

Reinforcement learning considers the problem of finding an optimal policy for an agent that interacts with an uncertain environment and collects reward per action.

The goal of the agent is to maximize its cumulative reward.

Formally, this problem can be viewed as a Markov decision process over the environment states s ∈ S and agent actions a ∈ A, with the (unknown) environment dynamics defined by the transition probability T (s |s, a) and reward function r(s t , a t ), which yields a reward immediately following the action a t performed in state s t .We consider goal-conditioned reinforcement learning from sparse rewards.

This constitutes a modification to the reward function such that it depends on a goal g ∈ G, such that r g : S × A × G → R. Every episode starts with sampling a state-goal pair from some distribution p(s 0 , g).

Unlike the state, the goal stays fixed for the whole episode.

At every time step, an action is chosen according to some policy π, which is expressed as a function of the state and the goal, π : S × G → A. For generality, our only restriction on G is that it is a subset of S. In other words, the goal describes a target state and the task of the agent is to reach that state.

Therefore, we apply the following sparse reward function: DISPLAYFORM0 where g is a goal, |s t − g| is a distance measure, and δ is a predefined threshold that controls when the goal is considered completed.

Following policy gradient methods, we model the policy as a conditional probability distribution over states, DISPLAYFORM1 where [s, g] denotes concatenation of state s and goal g, and θ are the learnable parameters.

Our objective is to optimize θ with respect to the expected cumulative reward, given by: DISPLAYFORM2 where ρ π (s) = ∞ t=1 γ t−1 Pr(s t = s) is the normalized discounted state visitation distribution with discount factor γ ∈ [0, 1).

To simplify the notation, we denote E s∼ρπ,a∼π(a|s,g),g∼G [·] by simply E π [·] in the rest of paper.

According to the policy gradient theorem BID43 , the gradient of J(θ) can be written as DISPLAYFORM3 where DISPLAYFORM4 )|s 1 = s, a 1 = a , called the critic, denotes the expected return under policy π after taking an action a in state s, with goal g.

Here, we introduce Deep Deterministic Policy Gradient (DDPG) BID24 , a model-free RL algorithm for continuous action spaces that serves as our backbone algorithm.

Our proposed modifications need not be restricted to DDPG; however, we leave experimentation with other continuous control algorithms to future work.

In DDPG, we maintain a deterministic target policy µ(s, g) and a critic Q(s, a, g), both implemented as deep neural networks (note: we modify the standard notation to accommodate goal-conditioned tasks).

To train these networks, episodes are generated by sampling actions from the policy plus some noise, a ∼ µ(s, g) + N (0, 1).

The transition tuple associated with each action (s t , a t , g t , r t , s t+1 ) is stored in the so-called replay buffer.

During training, transition tuples are sampled from the buffer to perform mini-batch gradient descent on the loss L which encourages the approximated Q-function to satisfy the Bellman equation DISPLAYFORM0 2 , where y t = r t +γQ(s t+1 , µ(s t+1 , g t ), g t ).

Similarly, the actor can be updated by training with mini-batch gradient descent on the loss J(θ) = −E s Q(s, µ(s, g), g) through the deterministic policy gradient algorithm (Silver et al., 2014) , DISPLAYFORM1 To make training more stable, the targets y t are typically computed using a separate target network, whose weights are periodically updated to the current weights of the main network BID24 BID27 .

Despite numerous advances in the application of deep learning to RL challenges, learning in the presence of sparse rewards remains a major challenge.

Intuitively, these algorithms depend on sufficient variability within the encountered rewards and, in many cases, random exploration is unlikely to uncover this variability if goals are difficult to reach.

Recently, BID0 proposed Hindsight Experience Replay (HER) as a technique to address this challenge.

The key insight of HER is that failed rollouts (where no task reward was obtained) can be treated as successful by assuming that a visited state was the actual goal.

Basically, HER amounts to a relabelling strategy.

For every episode the agent experiences, it gets stored in the replay buffer twice: once with the original goal pursued in the episode and once with the goal replaced with a future state achieved in the episode, as if the agent were instructed to reach this state from the beginning.

Formally, HER randomly samples a mini-batch of episodes in buffer, for each episode DISPLAYFORM0 ), for each state s t , where 1 ≤ t ≤ T − 1 in an episode, we randomly choose s k where t + 1 ≤ k ≤ T and relabel transition (s t , a t , g t , r t , s t+1 ) to (s t , a t , s k , r t , s t+1 ) and recalculate reward r t , DISPLAYFORM1 3 METHODIn this section, we present Competitive Experience Replay (CER) for policy gradient methods (Section 3.1) and describe the application of multi-agent DDPG to enable this technique (Section 3.2).

While the re-labelling strategy introduced by HER provides useful rewards for training a goal-conditioned policy, it assumes that learning from arbitrary goals will generalize to the actual task goals.

As such, exploration remains a fundamental challenge for goal-directed RL with sparse reward.

We propose a relabelling strategy designed to overcome this challenge.

Our method is partly inspired by the success of self-play in learning to play competitive games, where sparse rewards (i.e. win or lose) are common.

Rather than train a single agent, we train a pair of agents on the same task and apply an asymmetric reward relabelling strategy to induce a competition designed to encourage exploration.

We call this strategy Competitive Experience Replay (CER).To implement CER, we learn a policy for each agent, π A and π B , as well as a multi-agent critic (see below), taking advantage of methods for decentralized execution and centralized training.

During decentralized execution, π A and π B collect DDPG episode rollouts in parallel.

Each agent effectively plays a singleplayer game; however, to take advantage of multi-agent training methods, we arbitrarily pair the rollout from π A with that from π B and store them as a single, multi-agent rollout in the replay buffer D. When training on a mini-batch of off policy samples, we first randomly sample a mini-batch of episodes in D and then randomly sample transitions in each episode.

We denote the resulting mini-batch of transitions as DISPLAYFORM0 , where m is the size of the mini-batch.

Reward re-labelling in CER attempts to create an implicit exploration curriculum by punishing agent A for visiting states that agent B also visited, and, for those same states, rewarding agent B. For each A state s Conversely, each time such a nearby state pair is found, we increment the associated reward for agent B, r j B , by +1.

Each transition for agent A can therefore be penalized only once, whereas no restriction is placed on the extra reward given to a transition for agent B. Following training both agents with the re-labelled rewards, we retain the policy π A for evaluation.

Additional implementation details are provided in the appendix (Section D).We focus on two variations of CER that satisfy the multi-agent self-play requirements: first, the policy π B receives its initial state from the task's initial state distribution; second, although more restricted to re-settable environments, π B receives its initial state from a random off-policy sample of π A .

We refer to the above methods as independent-CER and interact-CER, respectively, in the following sections.

Importantly, CER re-labels rewards based on intra-agent behavior, whereas HER re-labels rewards based on each individual agent's behavior.

As a result, the two methods can be easily combined.

In fact, as our experiments demonstrate, CER and HER are complementary and likely reflect distinct challenges that are both addressed through reward re-labelling.

We extend multi-agent DDPG (MADDPG), proposed by BID26 , for training using CER.

MAD-DPG attempts to learn a different policy per agent and a single, centralized critic that has access to the combined states, actions, and goals of all agents.

More precisely, consider a game with N agents with policies parameterized by θ θ θ = {θ 1 , . . .

, θ N }, and let π π π = {π 1 , . . .

, π N } be the set of all agent policies.

g = [g 1 , . . . , g N ] represents the concatenation of each agent's goal, s = [s 1 , . . .

, s N ] the concatenated states, and a = [a 1 , . . . , a N ] the concatenated actions.

With this notation, we can write the gradient of the expected return for agent i, J(θ i ) = E[R i ] as: DISPLAYFORM0 With deterministic policies µ µ µ = {µ 1 , . . .

, µ N }, the gradient becomes: DISPLAYFORM1 The centralized action-value function Q µ µ µ i , which estimates the expected return for agent i, is updated as: DISPLAYFORM2 where µ µ µ = {µ µ µ θ 1 , ..., µ µ µ θ N } is the set of target policies with delayed parameters θ i .

In practice people usually soft update it as θ i ← τ θ i + (1 − τ )θ i , where τ is a Polyak coefficient.

During training, we collect paired rollouts as described above, apply any re-labelling strategies (such as CER or HER) and use the MADDPG algorithm to train both agent policies and the centralized critic, concatenating states, actions, and goals where appropriate.

Putting everything together, we summarize the full method in Algorithm 1.

We start by asking whether CER improves performance and sample efficiency in a sparse reward task.

To this end, we constructed two different mazes, an easier 'U' shaped maze and a more difficult 'S' shaped maze FIG1 ).

The goal of the ant agent is to reach the target mark by a red sphere, whose location is randomly sampled for each new episode.

At each step, the agent obtains a reward of 0 if the goal has been achieved and −1 otherwise.

Additional details of the ant maze environments are found in Appendix B. An advantage of this environment is that it can be reset to any given state, facilitating comparison between our two proposed variants of CER (int-CER requires the environment to have this feature).We compare agents trained with HER, both variants of CER, and both variants of CER with HER.

Since each uses DDPG as a backbone algorithm, we also include results from a policy trained using DDPG alone.

To confirm that any difficulties are not simply due to DDPG, we include results from a policy trained using Proximal Policy Optimization (PPO) BID44 .

The results for each maze are shown in FIG1 .

DDPG and PPO each performs quite poorly by itself, likely due to the sparsity of the reward in this task set up.

Adding CER to DDPG partially overcomes this limitation in terms of final success rate, and, notably, reaches this stronger result with many fewer examples.

A similar result is seen when adding HER to DDPG.

Importantly, adding both HER and CER offers the best improvement of final success rate without requiring any observable increase in the number of episodes, as compared to each of the other baselines.

These results support our hypothesis that existing state-of-the-art methods do not sufficiently address the exploration challenge intrinsic to sparse reward environments.

Furthermore, these results show that CER improves both the quality and efficiency of learning in such challenging settings, especially when combined with HER.

These results also show that int-CER tends to outperform ind-CER.

As such, int-CER is considered preferable but has more restrictive technical requirements.

To examine the efficacy of our method on a broader range of tasks, we evaluate the change in performance when ind-CER is added to HER on the challenging multi-goal sparse reward environments introduced in .

(Note: we would prefer to examine int-CER but are prevented by technical issues related to the environment.)

Results for each of the 12 tasks we trained on are illustrated in FIG3 .

Our method, when used on top of HER, improves performance wherever it is not already saturated.

This is especially true on harder tasks, where HER alone achieves only modest success (e.g., HandManipulateEggFull and handManipulatePenRotate).

These results further support the conclusion that existing methods often fail to achieve sufficient exploration.

Our method, which provides a targeted solution to this challenge, naturally complements HER.

CER is designed to encourage exploration, but, unlike other methods, uses the behavior of a competitor agent to automatically determine the criteria for exploratory reward.

Numerous methods have been proposed for improving exploration; for example, count-based exploration BID3 BID44 , VIME BID19 , bootstrap DQN BID32 , goal exploration process BID11 , parameter space noise BID34 , dynamics curiosity and boredom (Schmidhuber, 1991) , and EX2 BID12 .

One recent and popular method, intrinsic curiosity module (ICM) BID33 BID6 , augments task reward with curiosity-driven reward.

Much like CER and HER, ICM provides a method to relabel the reward associated with each transition; this reward comes from the error in a jointly trained forward prediction model.

We compare how CER and ICM affect task performance and how they interact with HER across 4 tasks where we were able to implement ICM successfully.

FIG2 shows the results on several robotic control and maze tasks.

We observe CER to consistently outperform ICM when each is implemented in isolation.

In addition, we observe HER to benefit more from the addition of CER than of ICM.

Compared to ICM, CER also tends to improve the speed of learning.

These results suggest that the underlying problem is not a lack of diversity of states being visited but the size of the state space that the agent must explore (as also discussed in BID0 ).

One interpretation is that, since the two CER agents are both learning the task, the dynamics of their competition encourage more task-relevant exploration.

From this perspective, CER provides an automatic curriculum for exploration such that visited states are utilized more efficiently.

Figure 4 illustrates the success rates of agents A and B as well as the 'effect ratio,' which is the fraction of mini-batch samples whose reward is changed by CER during training, calculated as φ = N M where N is the number of samples changed by CER and M is the number of samples in mini-batch.

We observe a strong correspondence between the success rate and effect ratio, likely reflecting the influence of CER on the learning dynamics.

While a deeper analysis would be required to concretely understand the interplay of these two terms, we point out that CER re-labels a substantial fraction of rewards in the mini-batch.

Interestingly, even the relatively small effect ratio observed in the first few epochs is enough support rapid learning.

We speculate that the sampling strategy used in int-CER provides a more targeted re-labelling, leading to the more rapid increase in success rate for agent A. We observe that agent B reaches a lower level of performance.

This likely results from resetting the parameters of agent B periodically during early training, which we observe to improve the ultimate performance of agent A (see Section D for details).

It is also possible that the reward structure of CER asymmetrically benefits agent A with respect to the underlying task.

To gain additional insight into how each method influences the behaviors learned across training, we visualize the frequency with which each location in the 'U' maze is visited after the 5th and 35th training epoch ( Figure 5) .

Comparing DDPG and HER shows that the latter clearly helps move the state distribution towards the goal, especially during the early parts of training.

When CER is added, states near the goal create a clear mode in the visitation distribution.

This is never apparent with DDPG alone and only obvious for HER and ICM+HER later in training.

CER also appears to focus the visitation of the agent, as it less frequently gets caught along the outer walls.

These disparities are emphasized in FIG5 , where we show the difference in the visitation profiles.

The left two plots compare CER+HER vs. HER.

The right two plots compare Agent A vs Agent B from CER+HER.

Interestingly, the Agents A and B exhibit fairly similar aggregate visitation profiles with the exception that Agent A reaches the goal more often later during training.

These visitation profiles underscore both the quantitative and qualitative improvements associated with CER.

Self-play has a long history in this domain of research (Samuel, 1959; BID45 .

BID40 use self-play with deep reinforcement learning techniques to master the game of Go; and self-play has even been applied in Dota 5v5 (OpenAI, 2018).

Curriculum learning is widely used for training neural networks (see e.g., BID4 BID14 BID8 BID30 .

A general framework for automatic task selection is Powerplay BID41 BID41 , which proposes an asymptotically optimal way of selecting tasks from a set of tasks with program search, and use replay buffer to avoid catastrophic forgetting. propose to automatically adjust task difficulty during training.

BID11 study intrinsically motivated goal for robotic learning.

Recently, BID42 suggest to use self-play between two agents where reward depends on the time of the other agent to complete to enable implicit curriculum.

Our work is similar to theirs, but we propose to use sample-based competitive experience replay, which is not only more readily scalable to high-dimension control but also integrates easily with Hindsight Experience Replay BID0 .

The method of initializing based on a state not sampled from the initial state distribution has been explored in other works.

For example, BID20 propose to create a backwards curriculum for continuous control tasks through learning a dynamics model.

BID38 and Salimans & Chen (2018) propose to train policies on Pommerman BID37 and the Atari game 'Montezumas Revenge' by starting each episode from a different point along a demonstration.

Recently, BID13 and propose a learned backtracking model to generate traces that lead to high value states in order to obtain higher sample efficiency.

Experience replay has been introduced in BID25 and later was a crucial ingredient in learning to master Atari games BID27 .

BID47 propose truncation with bias correction to reduce variance from using off-policy data in buffer and achieves good performance on continuous and discrete tasks.

Different approaches have been proposed to incorporate model-free learning with experience replay BID15 BID9 .

Schaul et al. (2015) improve experience replay by assigning priorities to transitions in the buffer to efficiently utilize samples.

BID18 further improve experience replay by proposing a distributed RL system in which experiences are shared between parallel workers and accumulated into a central replay memory and prioritized replay is used to update the policy based on the diverse accumulated experiences.

We introduce Competitive Experience Replay, a new and general method for encouraging exploration through implicit curriculum learning in sparse reward settings.

We demonstrate an empirical advantage of our technique when combined with existing methods in several challenging RL tasks.

In future work, we aim to investigate richer ways to re-label rewards based on intra-agent samples to further harness multi-agent competition, it's interesting to investigate counterfactual inference to promote efficient re-label off-policy samples.

We hope that this will facilitate the application of our method to more open-end environments with even more challenging task structures.

In addition, future work will explore integrating our method into approaches more closely related to model-based learning, where adequate exposure to the dynamics of the environment is often crucial.

As BID0 and BID1 observe (and we also observe), performance depends on the batch size.

We leverage this observation to tune the relative strengths of Agents A and B by separately manipulating the batch sizes used for updating each.

For simplicity, we control the batch size by changing the number of MPI workers devoted to a particular update.

Each MPI worker computes the gradients for a batch size of 256; averaging the gradients from each worker results in an effective batch size of N * 256.

For our single-agent baselines, we choose N = 30 workers, and, when using CER, a default of N = 15 for each A and B In the following, AxBy denotes, for agent A, N = x and, for agent B, N = y. These results suggest that, while a sufficiently large batch size is important for achieving the best performance, the optimal configuration occurs when the batch sizes used for the two agents are balanced.

Interestingly, we observe that batch size imbalance adversely effects both agents trained during CER.

In the code for HER BID0 , the authors use MPI to increase the batch size.

MPI is used here to run rollouts in parallel and average gradients over all MPI workers.

We found that MPI is crucial for good performance, since training for longer time with a smaller batch size gives sub-optimal performance.

This is consistent with the authors' findings in their code that having a much larger batch size helps a lot.

For each experiment, we provide the per-worker batch sizes in Table 1 ; note that the effective batch size is multiplied by the number of MPI workers N .In our implementation, we do N rollouts in parallel for each agent and have separate optimizers for each.

We found that periodically resetting the parameters of agent B in early stage of training helps agent A more consistently reach a high level of performance.

This resetting helps to strike an optimal balance between the influences of HER and CER in training agent A. We also add L2 regularization, following the practice of BID0 .For the experiments with neural networks, all parameters are randomly initialized from N (0, 0.2).

We use networks with three layers of hidden layer size 256 and Adam (Kingma & Ba, 2014) for optimization.

Presented results are averaged over 5 random seeds.

We summarize the hyperparameters in

<|TLDR|>

@highlight

a novel method to learn with sparse reward using adversarial reward re-labeling

@highlight

Proposes to use a competitive multi-agent setting for encouraging exploration and shows that CER + HER > HER ~ CER

@highlight

Propose a new method for learning from sparse rewards in model-free reinforcement learning settings and densifying reward

@highlight

To address the sparse reward problems and encourage exploration in RL algorithms, the authors propose a relabeling strategy called Competitive Experience Reply (CER).
Deep reinforcement learning algorithms require large amounts of experience to learn an individual task.

While in principle meta-reinforcement learning (meta-RL) algorithms enable agents to learn new skills from small amounts of experience, several major challenges preclude their practicality.

Current methods rely heavily on on-policy experience, limiting their sample efficiency.

They also lack mechanisms to reason about task uncertainty when adapting to new tasks, limiting their effectiveness in sparse reward problems.

In this paper, we address these challenges by developing an off-policy meta-RL algorithm that disentangles task inference and control.

In our approach, we perform online probabilistic filtering of latent task variables to infer how to solve a new task from small amounts of experience.

This probabilistic interpretation enables posterior sampling for structured and efficient exploration.

We demonstrate how to integrate these task variables with off-policy RL algorithms to achieve both meta-training and adaptation efficiency.

Our method outperforms prior algorithms in sample efficiency by 20-100X as well as in asymptotic performance on several meta-RL benchmarks.

Learning large repertoires of behaviors with conventional RL methods quickly becomes prohibitive as learning each task often requires millions of interactions with the environment.

Fortunately, many of the problems we would like our autonomous agents to solve share common structure.

For example screwing a cap on a bottle and turning a doorknob both involve grasping an object in the hand and rotating the wrist.

Exploiting this structure to learn new tasks more quickly remains an open and pressing topic.

While meta-learned policies adapt to new tasks with only a few trials, during training they require massive amounts of data drawn from a large set of distinct tasks, exacerbating the problem of sample efficiency that plagues RL algorithms.

Most current meta-RL methods require on-policy data during both meta-training and adaptation BID3 ; BID26 ; BID2 ; BID13 ; BID16 ; , rendering them exceedingly inefficient during meta-training.

However, making use of off-policy data for meta-RL poses new challenges.

Meta-learning typically operates on the principle that meta-training time should match meta-test time.

This makes it inherently difficult to meta-train a policy to adapt from off-policy data, which is systematically different from the data the policy would see when it explores (on-policy) in a new task at meta-test time.

To achieve both adaptation and meta-training data efficiency, our approach integrates online inference of probabilistic context variables with existing off-policy RL algorithms.

During meta-training, we learn a probabilistic encoder that accumulates the necessary statistics from past experience that enable the policy to perform the task.

At meta-test time, our method adapts quickly by sampling context variables ("task hypotheses"), acting according to that task, and then updating its belief about the task by updating the posterior over the context variables.

Our approach integrates easily with existing off-policy RL algorithms, enabling good sample efficiency during meta-training.

The primary contribution of our work is an off-policy meta-RL algorithm Probabilistic Embeddings for Actor-critic RL (PEARL) that achieves excellent sample efficiency during meta-training, enables fast adaptation by accumulating experience online, and performs structured exploration by reasoning about uncertainty over tasks.

We demonstrate 20-100X improvement in meta-training sample efficiency on six continuous control meta-learning environments, and demonstrate how our model structured exploration to adapt rapidly to new tasks with sparse rewards.

Our work builds on the meta-learning framework BID18 ; BID1 ; BID24 in the context of reinforcement learning.

Recurrent BID2 BID26 and recursive BID13 meta-RL methods adapt to new tasks by aggregating experience into a latent representation on which the policy is conditioned.

We model latent task variables as probabilistic and use a simpler aggregation function inspired by BID20 .

Prior work has explored training recurrent Q-functions with off-policy Q-learning methods BID10 ; BID8 .

We find the straightforward application of these methods to meta-RL difficult, and explore how to effectively make use of off-policy data during meta-training.

Gradient-based meta-RL methods focus on on-policy learning, using policy gradients BID3 The actor and critic are meta-learned jointly with the inference network, which is optimized with gradients from the critic as well as from an information bottleneck on Z. De-coupling the data sampling strategies for context and RL batches is critical for off-policy learning.

BID23 ; , or hyperparameters BID28 .

We instead focus on meta-learning from off-policy data, which is non-trivial to do with these prior methods.

Prior work has applied probabilistic models to meta-learning.

For supervised learning, Rusu et al. FORMULA0 ; BID5 BID4 adapt model predictions using probabilistic latent task variables inferred via amortized approximate inference.

In RL, BID9 also conditions the policy on inferred task variables, but the aim is to compose skills via the learned embedding space, while we focus on adapting to new tasks.

While we infer task variables and explore via posterior sampling, BID6 adapts via gradient descent and explores via sampling from the prior.

We assume a distribution of tasks p(T ), where each task is a Markov decision process (MDP).

Formally, a task T = {p(s 0 ), p(s t+1 |s t , a t ), r(s t , a t )} consists of an initial state distribution p(s 0 ), transition distribution p(s t+1 |s t , a t ), reward function r(s t , a t ).

We assume that the transition and reward functions are unknown, but can be sampled by taking actions in the environment.

Given a set of training tasks sampled from p(T ), the meta-training process learns a policy that adapts to the task at hand by conditioning on the history of past transitions, which we refer to as context C. Let c l n = (s n , a n , r n , s n ) be one transition in the task l so that c l 1:N comprises the experience collected so far.

At test-time, the policy must adapt to a new set of tasks from p(T ).

We capture knowledge about how the current task should be performed in a latent probabilistic context variable Z, on which we condition the policy as π θ (a|s, z).

Meta-training consists of leveraging data from a variety of training tasks to learn to infer Z from a recent history of experience in the new task, as well as optimizing the policy to solve the task given samples from the posterior over Z.To enable adaptation, the latent context Z must encode salient information about the task.

We adopt an amortized variational inference approach Kingma & Welling (2014); Rezende et al. FORMULA0 ; BID0 to learn to infer Z. We train an inference network q φ (z|c) that estimates the posterior p(z|c).

While there are several choices for the objective to optimize q φ (z|c) including learning predictive models of rewards and dynamics or maximizing returns through the policy, we choose to optimize it to predict the task state-action value function.

Modeling the objective as a pseudo-likelihood, the resulting variational lower bound training objective is: DISPLAYFORM0 where p(·) is a unit Gaussian prior over Z and R(T , z) is the Bellman error for a state-action value function conditioned on z. While the parameters of q φ are optimized during meta-training, at meta-test time the latent context for a new task is simply inferred from gathered experience.

The inference network q φ (z|c) should be expressive enough to capture minimal sufficient statistics of task-relevant information, without modeling irrelevant dependencies.

We note that an encoding of a fully observed MDP should be permutation invariant: if we would like to infer what the task is, identify the MDP model, or train a value function, it is enough to have access to a collection of transitions {s i , a i , s i , r i }, without regard for the order in which these transitions were observed.

We therefore choose a permutation-invariant representation q φ (z|c 1:N ) factorized as DISPLAYFORM1 To keep the method tractable, we use Gaussian factors Ψ(z|c n ) = N (f µ (c n ), f σ (c n )), which result in a Gaussian posterior, see FIG0 (left).For fast adaptation at meta-test time, it is critical for the agent to be able to explore and determine the task efficiently.

In prior work, posterior sampling for exploration BID22 ; BID14 models a distribution over MDPs and executes the optimal policy for an MDP sampled from the posterior for the duration of an episode.

Acting optimally according to a random MDP allows for temporally extended exploration, meaning that the agent can act to test hypotheses even when the results of actions are not immediately informative of the task.

PEARL meta-learns a prior over Z that captures the distribution over tasks.

Sampling z's (initially from the prior and then the updated posterior) and holding them constant across an episode results in temporally extended exploration strategies which become closer to the optimal behavior for the task as the belief narrows.

A primary goal of our work is to enable efficient off-policy meta-reinforcement learning.

However, designing off-policy meta-RL algorithms is non-trivial partly because modern meta-learning is predicated on the assumption that the distribution of data used for adaptation will match across metatraining and meta-test.

In RL, this implies that since at meta-test time on-policy data will be used to adapt, on-policy data should be used during meta-training as well.

Furthermore, meta-RL requires the policy to reason about distributions to learn effective stochastic exploration strategies.

This problem inherently cannot be solved by off-policy RL methods that minimize temporal-difference error, as they do not have the ability to directly optimize for distributions of states visited.

In contrast, policy gradient methods have direct control over the actions taken by the policy.

Our main insight in designing an off-policy meta-RL method with the probabilistic model in Section 3.1 is that the data used to train the probabilistic encoder need not be the same as the data used to train the policy.

The policy can treat the context z as part of the state in an off-policy RL loop, while the stochasticity of the exploration process is provided by the uncertainty in the encoder q(z|c).

The actor and critic are always trained with off-policy data sampled from the entire replay buffer B. We define a sampler S c to sample context batches for training the encoder -we find the sampling from a pool of recently collected data works best.

We summarize our training procedure in FIG0 (right).We build our algorithm on top of the soft actor-critic algorithm (SAC) BID7 , an off-policy actor-critic method based on the maximum entropy RL objective which augments the traditional sum of discounted returns with the entropy of the policy.

We optimize the parameters of the inference network q(z|c) jointly with the parameters of the actor π θ (a|s, z) and critic Q θ (s, a, z), using the reparameterization trick to compute gradients for parameters of q φ (z|c) through sampled z's.

We train the inference network using gradients from the Bellman update for the critic, given by the following loss function DISPLAYFORM0 whereV is a target network andz indicates that gradients are not being computed through it.

Sample Efficiency and Performance We evaluate PEARL on six continuous control environments simulated via MuJoCo BID25 .

These locomotion task distributions require adaptation across dynamics (Walker-2D-Params) or across reward functions (the rest of the domains), and were introduced by BID3 and BID16 .

We compare to existing policy gradient meta-RL methods ProMP BID16 , MAML- TRPO Finn et al. (2017) , and RL Dashed lines correspond to the maximum return achieved by each baseline after 1e8 steps.

By leveraging off-policy data during meta-training, PEARL is 20 − 100x more sample efficient than the baselines, and achieves state-of-the-art final performance.et al. FORMULA0 with PPO Schulman et al. (2017) .

We attempted to adapt recurrent DDPG BID10 to our setting, however we were unable to optimize it.

To evaluate on the meta-testing tasks, we perform online adaptation at the trajectory level, where the first trajectory is collected with context variable z sampled from the prior p(z) and subsequent trajectories are collected with z ∼ q(z|c).

Here we report performance after two trajectories.

PEARL is able to start adapting to the task after collecting on average only 5 trajectories.

We compare to MAESN BID6 ).Our approach uses 20-100x fewer samples during meta-training than previous policy gradient approaches while often also improving final asymptotic performance, FIG1 .Posterior Sampling For Exploration Posterior sampling in our model enables effective exploration strategies in sparse reward MDPs.

We demonstrate this behavior with a 2-D navigation task in which a point robot must navigate to different locations on a semi-circle.

A shaped reward is given only when the agent is within a certain radius of the goal (we experiment with radius 0.2 and 0.8).

We sample training and testing sets of tasks, each consisting of 100 randomly sampled goals.

To mitigate the difficulty of meta-training with sparse rewards, we assume access to the dense reward during meta-training, as in BID6 , but this burden could also be mitigated with task-agnostic exploration strategies.

In this setting, we compare to MAESN BID6 ) and demonstrate we are able to adapt to the new sparse goal in fewer trajectories, while also requiring far fewer samples for metatraining to solve the task, FIG2 .

<|TLDR|>

@highlight

Sample efficient meta-RL by combining variational inference of probabilistic task variables with off-policy RL 

@highlight

This paper proposes using off-policy RL during the meta-training time to greatly improve sample efficiency of Meta-RL methods.
Meta-Reinforcement learning approaches aim to develop learning procedures that can adapt quickly to a distribution of tasks with the help of a few examples.

Developing efficient exploration strategies capable of finding the most useful samples becomes critical in such settings.

Existing approaches to finding efficient exploration strategies add auxiliary objectives to promote exploration by the pre-update policy, however, this makes the adaptation using a few gradient steps difficult as the pre-update (exploration) and post-update (exploitation) policies are quite different.

Instead, we propose to explicitly model a separate exploration policy for the task distribution.

Having two different policies gives more flexibility in training the exploration policy and also makes adaptation to any specific task easier.

We show that using self-supervised or supervised learning objectives for adaptation stabilizes the training process and also demonstrate the superior performance of our model compared to prior works in this domain.

Reinforcement learning (RL) approaches have seen many successes in recent years, from mastering the complex game of Go BID10 to even discovering molecules BID8 .

However, a common limitation of these methods is their propensity to overfitting on a single task and inability to adapt to even slightly perturbed configuration BID12 .

On the other hand, humans have this astonishing ability to learn new tasks in a matter of minutes by using their prior knowledge and understanding of the underlying task mechanics.

Drawing inspiration from human behaviors, researchers have proposed to incorporate multiple inductive biases and heuristics to help the models learn quickly and generalize to unseen scenarios.

However, despite a lot of effort it has been difficult to approach human levels of data efficiency and generalization.

Meta-RL tries to address these shortcomings by learning these inductive biases and heuristics from the data itself.

These inductive biases or heuristics can be induced in the model in various ways like optimization, policy initialization, loss function, exploration strategies, etc.

Recently, a class of policy initialization based meta-learning approaches have gained attention like Model Agnostic MetaLearning (MAML) BID1 .

MAML finds a good initialization for a policy that can be adapted to a new task by fine-tuning with policy gradient updates from a few samples of that task.

Given the objective of meta RL algorithms to adapt to a new task from a few examples, efficient exploration strategies are crucial for quickly finding the optimal policy in a new environment.

Some recent works BID3 have tried to address this problem by using latent variables to model the distribution of exploration behaviors.

Another set of approaches BID11 BID9 focus on improving the credit assignment of the meta learning objective to the pre-update trajectory distribution.

However, all these prior works use one or few policy gradient updates to transition from preto post-update policy.

This limits the applicability of these methods to cases where the post-update (exploitation) policy is similar to the pre-update (exploration) policy and can be obtained with only a few updates.

Also, for cases where pre-and post-update policies are expected to exhibit different behaviors, large gradient updates may result in training instabilities and lack of convergence.

To address this problem, we propose to explicitly model a separate exploration policy for the distribution of tasks.

The exploration policy is trained to find trajectories that can lead to fast adaptation of the exploitation policy on the given task.

This formulation provides much more flexibility in training the exploration policy.

In the process, we also establish that, in order to adapt as quickly as possible to the new task, it is often more useful to use self-supervised or supervised learning approaches, where possible, to get more effective updates.

Unlike RL which tries to find an optimal policy for a single task, meta-RL aims to learn a policy that can generalize to a distribution of tasks.

Each task T sampled from the distribution ρ(T ) corresponds to a different Markov Decision Process (MDP).

These MDPs have similar state and action space but might differ in the reward function or the environment dynamics.

The goal of meta RL is to quickly adapt the policy to any task T ∼ ρ(T ) with the help of few examples from that task.

BID1 introduced MAML -a gradient-based meta-RL algorithm that tries to find a good initialization for a policy which can be adapted to any task T ∼ ρ(T ) by fine-tuning with one or more gradient updates using the sampled trajectories of that task.

MAML maximizes the following objective function:

where U is the update function that performs one policy gradient ascent step to maximize the expected reward R(τ ) obtained on the trajectories τ sampled from task T .

BID9 showed that the gradient of the objective function J(θ) can be written as: DISPLAYFORM0 where, ∇ θ J post (τ , τ ) optimizes θ to increase the likelihood of the trajectories τ that lead to higher returns given some trajectories τ .

In other words, this term does not optimize θ to yield trajectories τ that lead to good adaptation steps.

That is infact, done by the second term ∇ θ J pre (τ , τ ).

It optimizes for the pre-update trajectory distribution, P T (τ |θ), i.e, increases the likelihood of trajectories τ that lead to good adaptation steps.

During optimization, MAML only considers J post (τ , τ ) and ignores J pre (τ , τ ).

Thus MAML finds a policy that adapts quickly to a task given relevant experiences, however, the policy is not optimized to gather useful experiences from the environment that can lead to fast adaptation.

ProMP BID9 analyzes this issue with MAML and incorporates ∇ θ J pre (τ , τ ) term in the update as well.

They propose to use DICE BID2 to allow causal credit assignment on the pre-update trajectory distribution, however, the gradients computed by DICE suffer from high variance estimates.

To remedy this, they proposed a low variance (and slightly biased) approximation of the DICE based loss that leads to stable updates.

The pre-update and post-update policies are often expected to exhibit very different behaviors, i.e, exploration and exploitation behaviors respectively.

In such cases, transitioning a single policy from pure exploration phase to pure exploitation phase via policy gradient updates will require multiple steps.

Unfortunately, this significantly increases the computational and memory complexities of the algorithm.

Furthermore, it may not even be possible to achieve this transition via few gradient updates.

This raises an important question: DO WE REALLY NEED TO USE THE PRE-

Using separate policies for pre-update and post-update sampling: The straightforward solution to the above problem is to use a separate exploration policy µ φ responsible for collecting trajectories for the inner loop updates to get θ .

Following that, the post-update policy π θ can be used to collect trajectories for performing the outer loop updates.

Unfortunately, this is not as simple as it sounds.

To understand this, let's look at the inner loop updates: DISPLAYFORM0 When using the exploration policy for sampling, we need to perform importance sampling.

The update thus becomes: DISPLAYFORM1 where P T (τ |θ) and Q T (τ |φ) represent the trajectory distribution sampled by π θ and µ φ respectively.

Note that the above update is an off-policy update which results in high variance estimates when the two trajectory distributions are quite different from each other.

This makes it infeasible to use the importance sampling update in the current form.

In fact, this is a more general problem that arises even in the on-policy regime.

The policy gradient updates in the inner loop results in both ∇ θ J pre and ∇ θ J post terms being high variance.

This stems from the mis-alignment of the outer gradients (∇ θ J outer ) and the inner gradient,hessian Using a self-supervised/supervised objective for the inner loop update step: The instability in the inner loop updates arises due to the high variance nature of the policy gradient update.

Note that the objective of inner loop update is to provide some task specific information to the agent with the help of which it can adapt its behavior in the new environment.

We believe that this could be achieved using some form of self-supervised or supervised learning objective in place of policy gradient in the inner loop to ensure that the updates are more stable.

We propose to use a network for predicting some task (or MDP) specific property like reward function, expected return or value.

During the inner loop update, the network updates its parameters by minimizing its prediction error on the given task.

Unlike prior meta-RL works where the task adaptation in the inner loop is done by policy gradient updates, here, we update some parameters shared with the exploitation policy using a supervised loss objective function resulting in stability during the adaptation phase.

However, note that the variance and usefulness of the update depends heavily on the choice of the self-supervision/supervision objective.

We delve into this in more detail in the appendix.

DISPLAYFORM2

Our proposed model comprises of three modules, the exploration policy µ φ (s), the exploitation policy π θ,z (s), and the self-supervision network M β,z (s, a).

Note that M β,z and π θ,z share a set of parameters z while containing their own set of parameters β and θ respectively.

The agent first collects a set of trajectories τ using its exploration policy µ φ for each task T ∼ ρ(T ).

It then updates the shared parameter z by minimizing the regression DISPLAYFORM0 where, M (s, a) is the target which can be any of the task specific quantities as mentioned above.

We further modify the above equation by multiplying the DICE operator to simplify the gradient computation w.r.t φ.

This eliminates the need to apply the policy gradient trick to expand the above expression for gradient computation.

The update then becomes: DISPLAYFORM1 where ⊥ is the stop gradient operator.

After obtaining the updated parameters z , the agent samples trajectories τ using its updated exploitation policy π θ,z .

Note that our model enables the agent to learn a generic exploitation policy π θ,z for the task distribution which can then be adapted to any specific task by updating z to z as shown above.

Effectively, z encodes the necessary information regarding the task that helps an agent in adapting its behavior to maximize its expected return.

The collected trajectories are then used to perform a policy gradient update for all the parameters z, θ, φ and β using the following objective: DISPLAYFORM2 The gradients of J(z , θ) w.r.t.

φ are shown in Eq. 6 (see appendix).

Although the gradients are unbiased, they still have very high variance.

To solve this problem, we draw inspiration from BID7 and replace the return R µ t (see Eq. 7 in appendix) with an advantage estimate A µ t (see 8 in appendix).

Due to space constraints we describe these formulations in more detail in appendix.

We have evaluated our proposed model on the environments used by BID9 .

Specifically, we have used HalfCheetahFwdBack, HalfCheetahVel and Walker2DFwdBack environments for the dense reward tasks and a 2D point environment proposed in BID9 for sparse rewards.

The details of the network architecture and the hyperparameters used for learning have been mentioned in the appendix.

We would like to state that we have not performed much hyperparameter tuning due to computational constraints and we expect the results of our method to show further improvements with further tuning.

Also, we restrict ourselves to a single adaptation step in all environments for the baselines as well as our method, but it can be easily extended to multiple gradient steps as well by conditioning the exploration policy on z.

The results of the baselines for the benchmark environments have been borrowed directly from the the official ProMP website 1 .

For the point environments, we have used the publicly available official implementation 2 .

We also compare our method with 3 baseline approaches: MAML, EMAML and ProMP on the benchmark continuous control tasks.

The performance plots for all four algorithms are shown in FIG0 .

In all the environments, our proposed method outperforms others in terms of asymptotic performance.

The training is also more stable for our method and leads to lower variance plots.

Our algorithm particularly shines in 2DPointEnvCorner FIG1 where the reward is sparse.

In this environment, the agent needs to perform efficient exploration and use the sparse reward trajectories to perform stable updates both of which are salient aspects of our algorithm.

Although ProMP manages to reach similar peak performance to our method in 2DPointEnvCorner, the training itself is pretty unstable indicating the inherent fragility of their updates.

Further, we show that our method leads to good exploration behavior in a sparse reward point environment where the agent is allowed to sample only two trajectories in order to perform the updates illustrating the strength of our procedure.

We also show that the separation of exploration and exploitation policies in this scenario allows us to train the exploration policy using an independent objective providing better performance in certain situations.

Ablations: We perform several ablation experiments to analyze the impact of different components of our algorithm on 2D point navigation task.

FIG1 shows the performance plots for 5 different variants: VPG-Inner loop : The supervised loss in the inner loop is replaced with the vanilla policy gradient loss as in MAML while using the exploration policy to sample the pre-update trajectories.

As expected, this model performs poorly due to the high variance off-policy updates in the inner loop.

Reward Self-Supervision : A reward based self-supervised objective is used instead of return based self-supervision.

This variant performs reasonably well but struggles to reach peak performance since the task is sparse reward.

Vanilla DICE : We directly use the dice gradients to perform updates on φ instead of using the low variance gradient estimator.

The high variance dice gradients lead to unstable training as can be seen from the plots.

E-MAML Based : Used an E-MAML BID11 type objective to compute the gradients w.r.t φ instead of using DICE.

Although this variant manages to reach peak performance, it is unstable due to the lack of causal credit assignment.

Ours : Used the low variance estimate of the dice gradients to compute updates for φ along with return based self-supervision for inner loop updates.

Our model reaches peak performance and exhibits stable training due to low variance updates.

Unlike conventional meta-RL approaches, we proposed to explicitly model a separate exploration policy for the task distribution.

Having two different policies gives more flexibility in training the exploration policy and also makes adaptation to any specific task easier.

Hence, as future work, we would like to explore the use of separate exploration and exploitation policies in other meta-learning approaches as well.

We showed that, through various experiments on both sparse and dense reward tasks, our model outperforms previous works while also being very stable during training.

This validates that using self-supervised techniques increases the stability of these updates thus allowing us to use a separate exploration policy to collect the initial trajectories.

Further, we also show that the variance reduction techniques used in the objective of exploration policy also have a huge impact on the performance.

However, we would like to note that the idea of using a separate exploration and exploitation policy is much more general and doesn't need to be restricted to MAML.

that to compute M β,z (s t , a t ) = w T β m β (s t , a t ).

Using the successor representations can effectively be seen as using a more accurate/powerful baseline than directly predicting the N-step returns using the (s t , a t )pair.

We perform some additional experiments on another toy environment to illustrate the exploration behavior shown by our model and demonstrate the benefits of using different exploration and exploitation policies.

FIG2 an environment where the agent is initialized at the center of the semi-circle.

Each task in this environment corresponds to reaching a goal location (red dot) randomly sampled from the semi circle (green dots).

This is also a sparse reward task where the agent receives a reward only if it is sufficiently close to the goal location.

However, unlike the previous environments, we only allow the agent to sample 2 pre-update trajectories per task in order to identify the goal location.

Thus the agent has to explore efficiently at each exploration step in order to perform reasonably at the task.

FIG2 the trajectories taken by our exploration agent (orange and blue) and the exploitation/trained agent (green).

Clearly, our agent has learnt to explore the environment.

However, we know that a policy going around the periphery of the semi-circle would be a more useful exploration policy.

In this environment we know that this exploration behavior can be reached by simply maximizing the environment rewards collected by the exploration policy.

FIG3 shows this experiment where the exploration policy is trained using environment reward maximization while everything else is kept unchanged.

We call this variant Ours-EnvReward.

We also show the trajectories traversed by promp in FIG4 It is clear that it struggles to learn different exploration and exploitation behaviors.

FIG5 shows the performance of our two variants along with the baselines.

This experiment shows that decoupling the exploration and exploitation policies also allows us, the designers more flexibility at training them, i.e, it allows us to add any domain knowledge we might have regarding the exploration or the exploitation policies to further improve the performance.

We also experiment with Walker2DRandParams and HopperRandParams.

The different tasks in these environments arise from variations in the dynamics of the agent.

The results are shown in FIG7 We observe that in both these environments we match the performance of the baselines but don't really perform much better.

This could be because we still use the n-step return as the self-supervision objective in our experiments.

We expect the results to get better if we test with next-state prediction etc as self-superivision objectives.

We leave that for future work.

For all the experiments, we treat the shared parameter z as a learnable latent embedding with fixed initial values of 0 as proposed in BID13 ,i.e, we don't perform any outer-loop updates on z..

The exploitation policy π θ,z (s) and the self-supervision network M β,z (s, a) concatenates z with their respective inputs.

All the three networks (π, µ, M ) have the same architecture (except inputs and output sizes) as that of the policy network in BID9 for all experiments.

We also stick to the same values of hyper-parameters such as inner loop learning rate, gamma, tau and number of outer loop updates.

We keep a constant embedding size of 32 and a constant N=15 (for computing the N-step returns) across all experiments and runs.

We use the Adam BID5 optimizer with a learning rate of 7e − 4 for all parameters except φ, which uses a learning rate of 7e − 5.

Also, we restrict ourselves to a single adaptation step in all environments, but it can be easily extended to multiple gradient steps as well by conditioning the exploration policy on the latent parameters z. We have provided a version of our code in the supplementary material.

We will soon open source a cleaned version of this online.

<|TLDR|>

@highlight

We propose to use a separate exploration policy to collect the pre-adaptation trajectories in MAML. We also show that using a self-supervised objective in the inner loop leads to more stable training and much better performance.
Humans are experts at high-fidelity imitation -- closely mimicking a demonstration, often in one attempt.

Humans use this ability to quickly solve a  task instance, and to bootstrap learning of new tasks.

Achieving these abilities in autonomous agents is an open problem.

In this paper, we introduce an off-policy RL algorithm (MetaMimic) to narrow this gap.

MetaMimic can learn both (i) policies for high-fidelity one-shot imitation of diverse novel skills, and (ii) policies that enable the agent to solve tasks more efficiently than the demonstrators.

MetaMimic relies on the principle of storing all experiences in a memory and replaying these to learn massive deep neural network policies by off-policy RL.

This paper introduces, to the best of our knowledge, the largest existing neural networks for deep RL and shows that larger networks with normalization are needed to achieve one-shot high-fidelity imitation on a challenging manipulation task.

The results also show that both types of policy can be learned from vision, in spite of the task rewards being sparse, and without access to demonstrator actions.

One-shot imitation is a powerful way to show agents how to solve a task.

For instance, one or a few demonstrations are typically enough to teach people how to solve a new manufacturing task.

In this paper, we introduce an AI agent that when provided with a novel demonstration is able to (i) mimic the demonstration with high-fidelity, or (ii) forego high-fidelity imitation to solve the intended task more efficiently.

Both types of imitation can be useful in different domains.

Motor control is a notoriously difficult problem, and we are often deceived by how simple a manipulation task might appear to be.

Tying shoe-laces, a behaviour many of us learn by imitation, might appear to be simple.

Yet, tying shoe-laces is something most 6 year olds struggle with, long after object recognition, walking, speech, often translation, and sometimes even reading comprehension.

This long process of learning that eventually results in our ability to rapidly imitate many behaviours provides inspiration for the work in this paper.

We refer to high-fidelity imitation as the act of closely mimicking a demonstration trajectory, even when some actions may be accidental or irrelevant to the task.

This is sometimes called over-imitation BID28 .

It is known that humans over-imitate more than other primates BID18 and that this may be useful for rapidly acquiring new skills BID24 .

For AI agents however, learning to closely imitate even one single demonstration from raw sensory input can be difficult.

Many recent works focus on using expensive reinforcement learning (RL) methods to solve this problem BID46 BID27 BID37 BID3 .

In contrast, high-fidelity imitation in humans is often cheap: in one-shot we can closely mimic a demonstration.

Inspired by this, we introduce a meta-learning approach (MetaMimic - FIG0 ) to learn high-fidelity one-shot imitation policies by off-policy RL.

These policies, when deployed, require a single demonstration as input in order to mimic the new skill being demonstrated.

AI agents could acquire a large and diverse set of skills by high-fidelity imitation with RL.

However, representing many behaviours requires the adoption of a model with very high capacity, such as a very large deep neural network.

Unfortunately, showing that RL methods can be used to train massive deep neural networks has been an open question because of the variance inherent to these methods.

Indeed, traditional deep RL neural networks tend to be small, to the point that researchers have recently questioned their contribution BID42 .

In this paper, we show that it is possible to train massive high-fidelity imitation policy π(ot,gt) with off-policy RL.

This policy, represented with a massive deep neural network, enables the robot arm to mimic any demonstration in one-shot.

In addition to producing an imitation policy that generalizes well, MetaMimic populates its replay memory with all its rich experiences, including not only the demonstration videos, but also its past observations, actions and rewards.

By harnessing these augmented experiences, a task policy π(ot) can be trained to solve difficult sparse-reward control tasks.deep networks by off-policy RL to represent many behaviours.

Moreover, we show that bigger networks generalize better.

These results therefore provide important evidence that RL is indeed a scalable and viable framework for the design of AI agents.

Specifically this paper makes the following contributions 1 :• It introduces the MetaMimic algorithm and shows that it is capable of one-shot high-fidelity imitation from video in a complex manipulation domain.• It shows that MetaMimic can harness video demonstrations and enrich them with actions and rewards so as to learn uncoditional policies capable of solving manipulation tasks more efficiently than teleoperating humans.

By retaining and taking advantage of all its experiences, MetaMimic also substantially outperforms the state-of-the-art D4PG RL agent, when D4PG uses only the current task experiences.• The experiments provide ablations showing that larger networks (to the best of our knowledge, the largest networks ever used in deep RL) lead to improved generalization in high-fidelity imitation.

The ablations also highlight the important value of instance normalization.• The experiments show that increasing the number of demonstrations during training leads to better generalization on one-shot high-fidelity imitation tasks.

MetaMimic is an algorithm to learn both one-shot high-fidelity imitation policies and unconditional task policies that outperform demonstrators.

Component 1 takes as input a dataset of demonstrations, and produces (i) a set of rich experiences and (ii) a one-shot high-fidelity imitation policy.

Component 2 takes as input a set of rich experiences and produces an unconditional task policy.

Component 1 uses a dataset of demonstrations to define a set of imitation tasks and using RL it trains a conditional policy to perform well across this set.

Here each demonstration is a sequence of observations, without corresponding actions.

This component can use any RL algorithm, and is applicable whenever the agent can be run in the environment and the environment's initial conditions can be precisely set.

In practice we make use of D4PG , an efficient off-policy RL algorithm, for training the agent's policy from demonstration data.

In Section 2.1 we give a detailed description of our approach for learning one-shot high-fidelity imitation policies.

Here we also describe the neural network architectures used in our approach to imitation; our results will show that as the number of imitation tasks increases it becomes necessary to train large-scale neural network policies to generalize well.

Furthermore, the process of training the imitation policies results in a memory of experiences which includes both actions and rewards.

As shown in Section 2.2 we can replay these experiences to learn unconditional policies capable of solving new tasks and outperforming human demonstrators.

Algorithms for the imitation learner, the task actor and the task learner are provided in Appendix A.

We consider a single stochastic task and define a demonstration of this task as a sequence of DISPLAYFORM0 While there is only one task, there is ample diversity in the environment's initial conditions and in the demonstrations.

Each observation d t can be either an image or a combination of both images and proprioceptive information.

We let π θ represent a deterministic parameterized imitation policy that produces actions a t = π θ (o t ,d) by conditioning on the current observations and a given demonstration.

We also let E denote the environment renderer, which accepts an arbitrary policy and produces a sequence of observations o = {o 1 ,...,o T } = E(π).

We can think of this last step as producing a rollout trajectory given the policy, as illustrated on the right side of FIG1 .The goal of high-fidelity imitation is to estimate the parameters θ of a policy that maximizes the expected imitation return, e.g. similarity between the observations o and sampled demonstrations d ∼ p d .

That is, DISPLAYFORM1 where sim is a similarity measure (reward), which will be discussed in greater detail at the end of this section, and γ is the discount factor.

In general it is not possible to differentiate through the environment E. We thus choose to optimize this objective with RL, while sampling trajectories by acting on the environment.

Finally, we refer to this process as one-shot imitation because although we make use of multiple demonstrations at training time to learn the imitation policy, at test time we are able to follow a single novel demonstration d using the learned conditional policy π θ * (o t ,d ).We adopt the recently proposed D4PG algorithm as a subroutine for training the imitation policy.

This is a distributed off-policy RL algorithm that interacts with the environment using independent actors (see FIG1 , each of which inserts trajectory data into a replay table.

A learner process in parallel samples (possibly prioritized) experiences from this replay dataset and optimizes the policy in order to maximize the expected return.

MetaMimic first builds on this earlier work by making a very specific choice of reward and policy.

At the beginning of every episode a single demonstration d is sampled, and the initial conditions of the environment are set to those of the demonstration, i.e. o 1 = d 1 .

The actor then interacts with the environment by producing actions a t = π θ (o t ,d).

While this policy representation is popular in the feature-based supervised one-shot imitation literature, in our case the observations and demonstrations are sequences of high-dimensional sensory inputs and hence this approach becomes computationally prohibitive.

To overcome this challenge, we simplify the model to only consider local context DISPLAYFORM2 In this formulation, the future demonstration state can be interpreted as a goal state and the approach may be thought of as goal-conditional imitation with a time-varying goal.

At every timestep, we compute the reward r t = r(o t+1 ,d).

While in general this reward can depend on the entire trajectory-or on small subsequences such as (d t ,d t+1 )-in practice we will restrict Figure 3 : Network architecture.

Since high-fidelity imitation is a fine-grained perception task, we found it necessary to use large convolutional neural networks.

Our best network consists of a residual network BID14 with twenty convolutional layers, instance normalization (Ulyanov et al.) between convolutional layers, layer normalization BID4 between fully connected layers, and exponential linear units BID8 .

We use a similar network architecture for the imitation policy and task policy, however the task policy does not receive a goal gt.it to depend only on the goal state d t+1 .

Ideally the reward function can be learned during training BID13 BID33 .

However, in this work we experiment with a simple reward function based on the Euclidean distance 2 over observations: DISPLAYFORM3 where o has no information about objects in the environment, so it may fail to encourage the imitator to interact with the objects; where as o image t contains information about the body and objects, but is insufficient to uniquely describe either.

In practice, we found a combination of both to work best.

Again we note that the next demonstration state d t+1 can be interpreted as a goal state.

In this perspective the goals are set by the demonstration, and the agent is rewarded by the degree to which it reaches those goals.

Because the imitation goals are explicitly given to the policy, the imitation policy is able to imitate many diverse demonstrations, and even generalize to unseen demonstrations as described in Section 2.2.High-fidelity imitation is a fine-grained perception task and hence the choice of policy architecture is critical.

In particular, the policy must be able to closely to mimic not only one but many possible ways of accomplishing the same stochastic task under different environment configurations.

This representational demand motivates the introduction of high-capacity deep neural networks.

We found the architecture, shown in Figure 3 , with residual connections, 20 convolution layers with 512 channels for a total of 22 million parameters, and instance normalization to drastically improve performance, as shown in Figure 6 of the Experiments section.

Following the recommendations of BID42 , we compare the performance of this model with a smaller (15 convolution layers with 32 channels and 1.6 million) network proposed recently by BID10 and find size to matter.

We note however that the IMPALA network of BID10 is large in comparison to previous networks used in important RL and control milestones, including AlphaGo BID48 , Atari DQN BID30 , dexterous in-hand manipulation BID35 , QT-Opt for vision-based robotic manipulation BID22 , and DOTA among others.

MetaMimic fills its replay memory with a rich set of experiences, which can also be used to learn a task more quickly.

In order to train a task policy using RL we need to both explore, i.e. to find a sequence of actions that leads to high reward; and to learn, i.e. to harness reward signals to improve the generation of actions so as to generalize well.

Unfortunately, stumbling on a rewarding sequence of actions for many control tasks is unlikely, especially when reward is only provided after a long sequence of actions.

A powerful way of using demonstrations for exploration is to inject the demonstration data directly into an off-policy experience-replay memory BID52 BID32 .

However, these methods require access to privileged information about the demonstration -the sequences of actions and rewards -which is often not available.

Our method takes a different approach.

Figure 4: One-shot high-fidelity imitation.

Given novel, diverse, test-set demonstration videos (three examples are shown above), the imitation policy is able to closely mimic the human demonstrator.

In particular, it successfully maps image observations to arm velocities while managing the complex interaction forces among the arm, blocks and ground.

While our high-fidelity imitation policy attempts to imitate the demonstration from observations only, it generates its own observations, actions and rewards.

These experiences are often rewarding enough to help with exploration.

Therefore, instead of injecting demonstration trajectories, we place all experiences generated by our imitation policy in the experience-replay memory, as illustrated in FIG0 .

The key design principle behind our approach is that RL agents should store all their experiences and take advantage of them for solving new problems.

More precisely, as the imitation policy interacts with the environment we also assume the existence of a task reward (r task t ).

Given these rewards we can introduce an additional, off-policy task learner which optimizes an unconditional task policy π ω (o t ).

This policy can be learned from transitions (o t ,a t ,o t+1 ,r task t ) generated asynchronously by the imitation actors following policy π θ (o t ,d t+1 ).

This learning process is made possible by the fact that the task learner is simply optimizing the cumulative reward in an off-policy fashion.

It should also be noted that this does not require privileged information about demonstrations because the sampled transitions are generated by the imitation actor's process of learning to imitate.

Due to the existence of demonstrations, the imitation trajectories are likely to lie in areas of high reward and as a result these samples can help circumvent the exploration problem.

However, they are also likely to be very off-policy initially during learning.

As a result, we augment these trajectories with samples generated asynchronously by additional task actors using the unconditional task policy π ω (o t ).

The task learner then trains its policy by sampling from both imitation and task trajectories.

For more algorithmic details see Appendix A. As the imitation policy improves, rewarding experiences are added to the replay memory and the task learner draws on these rewarding sequences to circumvent the exploration problem through off-policy learning.

We will show this helps accelerate learning of the task policy, and that it works as well as methods that have direct access to expert actions and expert rewards.

In this section, we analyze the performance of our imitation and task policies.

We chose to evaluate our methods in a particularly challenging environment: a robotic block stacking setup with both sparse Figure 5 : More demonstrations improve generalization when using the imitation policy.

The goal of one-shot high-fidelity imitation is generalization to novel demonstrations.

We analyze how generalization performance increases as we increase the number of demonstrations in the training set: 10 (-), 50 (-), 100 (-), 500 (-).

With 10 demonstrations, the policy is able to closely mimic the training set, but generalizes poorly.

With 500 demonstrations, the policy has more difficultly closely mimicking all trajectories, but has similar imitation reward on train and validation sets.

The figure also shows that higher imitation reward when following the imitation policy (BOTTOM ROW) results in higher task reward (TOP ROW).

Here we normalize the task reward by dividing it by the average of demonstration cumulative reward.rewards and diverse initial conditions, learned from visual observations.

In this space, our goal is to learn a policy performing high-fidelity imitation from human demonstration, while generalizing to new initial conditions.

Our environment consists of a Kinova Jaco arm with six arm joints and three actuated fingers, simulated in MuJoCo.

In the block stacking task BID32 , the robot interacts with two blocks on a tabletop.

The task reward is a sparse piecewise constant function as described in BID55 In this environment, we collected demonstrations using a SpaceNavigator 3D motion controller, which allows human operators to control the robot arm with a position controller.

We collected 500 episodes of demonstrations as imitation targets.

Another 500 episodes were gathered for validation purposes by a different human demonstrator.

Note that the images shown in this paper have been rendered with the path-tracer Mitsuba for illustration purposes.

Our agent does not however require such high-quality video input-the environment output generated by MuJoCo o image t which our agent observes is lower in resolution and quality.

We use D4PG (see Appendix C.1) to train the imitation policy in a one-shot manner (sec. 2.1) on the block stacking task.

The policy observes the visual input o image t , as well a demonstration sequence d randomly sampled from the set of 500 expert episodes.

In fig. 4 we show that our policy can closely mimic novel, diverse, test-set demonstration videos.

Recall these test demonstrations are provided by Figure 6 : Larger networks and normalization improve rewards in high-fidelity imitation.

We compare the ResNet model used by the IMPALA agent (15 conv layers, 32 channels) (-) with the much larger networks used by MetaMimic inspired by ResNet34 (20 conv layers, 512 channels) with (-) and without (-) instance normalization.

We use three metrics for this comparison: task reward (LEFT), imitation reward when tracking pixels (MIDDLE) and imitation reward when tracking arm joint positions and velocities (RIGHT).

We find large neural networks, and normalization significantly improve performance for high-fidelity imitation.

To our knowledge, MetaMimic uses the largest neural network trained end-to-end using reinforcement learning.

a different expert and require generalization to a distinct stacking style.

The test demonstrations are so different from the training ones that the average cumulative reward of the test demonstrations is lower than that of training by as much as 70.

(The average episodic reward for the training demonstration set is 355 and that for the test set is 285.)

On these novel demonstrations we are able to achieve 52% of the reward of the demonstration without any task reward, solely by doing high-fidelity imitation.

It is important to note that we achieve this while placing significantly less assumptions on the environment and demonstration than comparable methods.

Unlike supervised methods (e.g. BID32 ) we can imitate without actions at training time.

And while proprioceptive features are used as part of computing the imitation reward, they are not observed by the policy, which means MetaMimic's imitation policy can mimic block stacking demonstrations purely from video at test time.

And finally, as opposed to pure reinforcement-learning approaches ((Barth-Maron et al., 2018)) we do not train on a task reward.

Generalization: To analyze how well our learned policy generalizes to novel demonstrations, we run the policy conditioned on demonstrations from the validation set.

As fig. 5 shows, validation rewards track the training curves fairly well.

We also notice that policies trained on a small number of demonstrations achieve high imitation reward in training, but low reward on validation, while policies trained on a bigger set generalize much better.

While we do not use a task reward in training, we use it to measure the performance on the stacking task.

We see the same behavior as for the imitation reward: Policies trained on 50 or more demonstrations generalize very well.

On the training set, performance varies from 67% of the average demonstration reward to 81% of the average demonstration reward.

Network architecture: Most reinforcement learning methods use comparatively small networks.

However, high-fidelity imitation of this stochastic task requires the coordination of fine-grained perception and accurate motor control.

These problems can strongly benefit from large architectures used for other difficult vision tasks.

And due the fact that we are training with a dense reward, the training signal is rich enough to properly train even large models.

In fig. 3 we demonstrate that indeed a large ResNet34-style network BID14 clearly outperforms the network from IMPALA BID10 .

Additionally, we show that using instance normalization (Ulyanov et al.) improves performance even more.

We chose instance normalization instead of batch norm BID21 to avoid distribution drift between training and running the policy BID20 .

To our knowledge, this is the largest neural network trained end-to-end using reinforcement learning.

In the previous section we have shown that we are able to learn a high-fidelity imitation policy which mimics a novel demonstration in a single shot.

It does however require a demonstration sequence as an input at test time.

The task policy (see sec. 2.2), not conditioned on a demonstration sequence, is trained concurrently to the imitation policy, and learns from the imitation experiences along with its own experiences.

In fig. 7 we show a qualitative comparison of the task policy and a corresponding demonstration sequence.

This is achieved by starting the task policy at the same initial state as the demonstration Figure 7 : Efficient task policy.

The task policy (which does not condition on a demonstration) is able to outperform the demonstration videos.

We test this by initializing the environment to the same initial state as a demonstration from the training-set.

The task policy is able to stack within 50 frames, while the demonstration stacks in 200 frames.

The task policy has no incentive to behave like the demonstration, so it often lays its arm down after stacking.

MetaMimic (-) to demonstrations (-), D4PG (-), and two methods that use demonstrations with access to additional information: D4PGfD (-), D4PG with a demonstration curriculum (--), and MetaMimic with a curriculum (--).

D4PG is not able to reach the performance of the demonstrations.

The methods with access to additional information are able to quickly outperform the demonstrators.

MetaMimic matches their performance even without access to expert rewards or actions.sequence.

The task policy is not merely imitating the demonstration sequence, but has learned to perform the same task in much shorter time.

For our task, the policy is able to outperform the demonstrations it learned from by 50% in terms of task reward (see FIG4 ).

Pure RL approaches are not able to reach this performance; the D4PG baseline scored significantly below the demonstration reward.

A really powerful technique in RL is to use demonstrations as a curriculum.

To do that, one starts the episode during training such that the initial state is set to a random state along a random demonstration trajectory.

This technique enables the agent to see the later and often rewarding part of a demonstration episode often.

This approach has been shown to be very beneficial to RL as well as imitation learning BID43 BID19 BID39 .

It, however, requires an environment which can be reset to any demonstration state, which is often only possible in simulation.

We compare D4PG and our method both with and without a demonstration curriculum.

As shown in previous results, using this curriculum for training significantly improves convergence for both methods.

Our method without a demonstration curriculum performs as well as D4PG with the curriculum.

When trained with the curriculum, our method significantly outperform all other methods; see FIG4 .Last but not least, we compare our task policy against D4PGfD BID53 ) (see sec. 4 for more details).

D4PGfD differs from our approach in that it requires demonstration actions.

While it takes time for our imitation policy to take off and help with training of the task policy, D4PGfD can help with exploration right away.

It therefore is more efficient.

Despite having no access to actions, however, our task policy catches up with D4PGfD quickly and reaches the same performance as the policy trained with D4PGfD. This speaks to the efficiency of our imitation policy.

See FIG4 for more details.

General imitation learning: Many prior works focus on using imitation learning to directly learn a task policy.

There are two main approaches: Behavior cloning (BC) which attempts to learn a task policy by supervised learning BID38 , and inverse RL (IRL) which attempts to learn a reward function from a set of demonstrations, and then uses RL to learn a task policy that maximizes that learned reward BID34 BID0 BID56 .While BC has problems with accumulating errors over long sequences, it has been used successfully both on its own BID40 and as an auxiliary loss in combination with RL BID41 BID32 .

IRL methods do not necessarily require expert actions.

Generative Adversarial Imitation Learning (GAIL) BID16 ) is one example.

GAIL constructs a reward function that measures the similarity between expert-generated observations and observations generated by the current policy.

GAIL has been successfully applied in a number of different environments BID16 BID25 BID55 .

While these methods work quite well, they focus on learning task policies, and not one-shot imitation.

One-shot imitation learning: Our approach is a form of one-shot imitation.

A few recent works have explored one-shot task-based imitation learning BID12 BID9 , i.e. given a single demonstration, generalize to a new task instance with no additional environment interactions.

These methods do not focus on high-fidelity imitation and therefore may not faithfully execute the same plan as the demonstrator at test time.

Imitation by tracking: Our method learns from demonstrations using a tracking reward BID2 .

This method has seen increased popularity in games BID3 and control BID46 BID27 BID37 .

All these methods use tracking to imitate a single demonstration trajectory.

Imitation by tracking has several advantages.

For example it does not require access to expert actions at training time, can track long demonstrations, and is amenable to third person imitation BID45 .

To our knowledge, MetaMimic is the first to train a single policy to closely track hundreds of demonstration trajectories, as well as generalize to novel demonstrations.

Inverse dynamics models: Our method is closely related to recent work on learned inverse dynamics models BID36 BID31 .

These works train inverse dynamics models without expert demonstrations by self-supervision.

However since these methods are based on random exploration they rely on high level control policies, structured exploration, and short horizon tasks.

BID50 also train an inverse dynamics model to learn an unconditional policy.

Their method, however, uses supervised learning, and does not outperform BC.Multi-task off-policy reinforcement learning: Our approach is related to recent work that learns a family of policies, with a shared pool of experiences BID49 BID7 BID44 .

This allows for sparse reward tasks to be solved faster, when paired with related dense reward tasks.

BID7 and BID44 require the practitioner to design a family of tasks and reward functions related to the task of interest.

In this paper, we circumvent the need of auxiliary task design via imitation.fD-style methods: When demonstration actions are available, one can embed the expert demonstrations into the replay memory BID53 .

Through off-policy learning, the demonstrations could lead to better exploration.

This is similar to our approach as detailed in sec 2.2.

We, however, eliminate the need for expert actions through high-fidelity imitation.

For a tabular comparison of the different imitation techniques, please refer to Table 2 in the Appendix.

In this paper, we introduced MetaMimic, a method to 1) train a high-fidelity one-shot imitation policy, and to 2) efficiently train a task policy.

MetaMimic employs the largest neural network trained via RL, and works from vision, without the need of expert actions.

The one-shot imitation policy can generalize to unseen trajectories and can mimic them closely.

Bootstrapping on imitation experiences, the task policy can quickly outperform the demonstrator, and is competitive with methods that receive privileged information.

The framework presented in this paper can be extended in a number of ways.

First, it would be exciting to combine this work with existing methods for learning third-person imitation rewards BID45 BID3 .

This would bring us a step closer to how humans imitate: By watching other agents act in the environment.

Second, it would be exciting to extend MetaMimic to imitate demonstrations of a variety of tasks.

This may allow it to generalize to demonstrations of unseen tasks.

To improve the ease of application of MetaMimic to robotic tasks, it would be desirable to address the question of how to relax the initialization constraints for high-fidelity imitation; specifically not having to set the initial agent observation to be close to the initial demonstration observation.

Given:• We use D4PG (Barth-Maron et al., 2018) as our main training algorithm.

Briefly, D4PG is a distributed off-policy reinforcement learning algorithm for continuous control problems.

In a nutshell, D4PG uses Q-learning for policy evaluation and Deterministic Policy Gradients (DPG) BID47 for policy optimization.

An important characteristic of D4PG is that it maintains a replay memory M (possibility prioritized ) that stores SARS tuples which allows for off-policy learning.

D4PG also adopts target networks for increased training stability.

In addition to these principles, D4PG utilized distributed training, distributional value functions, and multi-step returns to further increase efficiency and stability.

In this section, we explain the different ingredients of D4PG.D4PG maintains an online value network Q(o,a|θ) and an online policy network π(o|φ).

The target networks are of the same structures as the value and policy network, but are parameterized by different parameters θ and φ which are periodically updated to the current parameters of the online networks.

Given the Q function, we can update the policy using DPG: J (φ) = E ot∼M ∇ φ Q(o t ,π(o t |φ)) .Instead of using a scalar Q function, D4PG adopts a distributional value function such that Q(o t ,a|θ) = E Z(o t ,a|θ) where Z is a random variable such that Z = z i w.p.

p i ∝ exp(ω(o t ,a|θ)).The z i 's take on V bins discrete values that ranges uniformly between V min and V max such that z i = V min +i Vmax−Vmin V bins for i ∈ {0,···,V bins −1}.To construct a bootstrap target, D4PG uses N-step returns.

Given a sampled tuple from the replay memory: o t ,a t ,{r t ,r t+1 ,···,r t+N −1 },o t+N , we construct a new random variable Z such that Z = z i + N −1 n=0 γ n r t+n w.p.

p i ∝ exp(ω(o t ,a|θ )).

Notice, Z no longer has the same support.

We therefore adopt the same projection Φ employed by BID6 .

The training loss for the value fuction L(θ) = E ot,at,{rt,···,r t+N −1 },o t+N ∼M H(Φ(Z ),Z(o t ,a t |θ)) , where H is the cross entropy.

Last but not least, D4PG is also distributed following .

Since all learning processes only rely on the replay memory, we can easily decouple the 'actors' from the 'learners'.

D4PG therefore uses a large number of independent actor processes which act in the environment and write data to a central replay memory process.

The learners could then draw samples from the replay memory for learning.

The learner also serves as a parameter server to the actors which periodically update their policy parameters from the learner.

For more details see Algorithms 1-4.

When using a demonstration curriculum, we randomly sample the initial state from the first 300 steps of a random demonstration.

For the Jaco arm experiments, we consider the vectors between the hand and the target block for the environment and the demonstration and compute the L2 distance between these.

The episode terminates if the distance exceeds a threshold (0.01).

@highlight

We present MetaMimic, an algorithm that takes as input a demonstration dataset and outputs (i) a one-shot high-fidelity imitation policy (ii) an unconditional task policy.

@highlight

The paper looks at the problem of one-shot imitation with high accuracy of imitation, extending DDPGfD to use only state trajectories.

@highlight

This paper proposes an approach for one-shot imitation with high accuracy, and addresses the common problem of exploration in imitation learning.

@highlight

Presents an RL method for learning from video demonstration without access to expert actions
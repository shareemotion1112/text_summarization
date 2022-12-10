Learning to control an environment without hand-crafted rewards or expert data remains challenging and is at the frontier of reinforcement learning research.

We present an unsupervised learning algorithm to train agents to achieve perceptually-specified goals using only a stream of observations and actions.

Our agent simultaneously learns a goal-conditioned policy and a goal achievement reward function that measures how similar a state is to the goal state.

This dual optimization leads to a co-operative game, giving rise to a learned reward function that reflects similarity in controllable aspects of the environment instead of distance in the space of observations.

We demonstrate the efficacy of our agent to learn, in an unsupervised manner, to reach a diverse set of goals on three domains -- Atari, the DeepMind Control Suite and DeepMind Lab.

Currently, the best performing methods on many reinforcement learning benchmark problems combine model-free reinforcement learning methods with policies represented using deep neural networks BID18 BID8 .

Despite reaching or surpassing human-level performance on many challenging tasks, deep model-free reinforcement learning methods that learn purely from the reward signal learn in a way that differs greatly from the manner in which humans learn.

In the case of learning to play a video game, a human player not only acquires a strategy for achieving a high score, but also gains a degree of mastery of the environment in the process.

Notably, a human player quickly learns which aspects of the environment are under their control as well as how to control them, as evidenced by their ability to rapidly adapt to novel reward functions BID22 .Focusing learning on mastery of the environment instead of optimizing a single scalar reward function has many potential benefits.

One benefit is that learning is possible even in the absence of an extrinsic reward signal or with an extrinsic reward signal that is very sparse.

Another benefit is that an agent that has fully mastered its environment should be able to reach arbitrary achievable goals, which would allow it to generalize to tasks on which it wasn't explicitly trained.

Building reinforcement learning agents that aim for environment mastery instead of or in addition to learning about a scalar reward signal is currently an open challenge.

One way to represent such knowledge about an environment is using an environment model.

Modelbased reinforcement learning methods aim to learn accurate environment models and use them either for planning or for training a policy.

While learning accurate environment models of some visually rich environments is now possible BID33 BID7 BID15 using learned models in model-based reinforcement learning has proved to be challenging and model-free approaches still dominate common benchmarks.

We present a new model-free agent architecture of Discriminative Embedding Reward Networks, or DISCERN for short.

DISCERN learns to control an environment in an unsupervised way by learning purely from the stream of observations and actions.

The aim of our agent is to learn a goal-conditioned policy π θ (a|s; s g ) BID19 BID37 which can reach any goal state s g that is reachable from the current state s.

We show how to learn a goal achievement reward function r(s; s g ) that measures how similar state s is to state s g using a mutual information objective at the same time as learning π θ (a|s; s g ).

The resulting learned reward function r(s; s g ) measures similarity in the space of controllable aspects of the environment instead of in the space of raw observations.

Crucially, the DISCERN architecture is able to deal with goal states that are not perfectly reachable, for example, due to the presence of distractor objects that are not under the agent's control.

In such cases the goal-conditioned policy learned by DISCERN tends to seek states where the controllable elements match those in the goal state as closely as possible.

We demonstrate the effectiveness of our approach on three domains -Atari games, continuous control tasks from the DeepMind Control Suite, and DeepMind Lab.

We show that our agent learns to successfully achieve a wide variety of visually-specified goals, discovering underlying degrees of controllability of an environment in a purely unsupervised manner and without access to an extrinsic reward signal.

In the standard reinforcement learning setup an agent interacts with an environment over discrete time steps.

At each time step t the agent observes the current state s t and selects an action a t according to a policy π(a t |s t ).

The agent then receives a reward r t = r(s t , a t ) and transitions to the next state s t+1 .

The aim of learning is to maximize the expected discounted return R = ∞ t=0 γ t r t of policy π where γ ∈ [0, 1) is a discount factor.

In this work we focus on learning only from the stream of actions and observations in order to forego the need for an extrinsic reward function.

Motivated by the idea that an agent capable of reaching any reachable goal state s g from the current state s has complete mastery of its environment, we pose the problem of learning in the absence of rewards as one of learning a goal-conditioned policy π θ (a|s; s g ) with parameters θ.

More specifically, we assume that the agent interacts with an environment defined by a transition distribution p(s t+1 |s t , a t ).

We define a goal-reaching problem as follows.

At the beginning of each episode, the agent receives a goal s g sampled from a distribution over possible goals p goal .

For example, p goal could be the uniform distribution over all previously visited states.

The agent then acts for T steps according to the goal-conditioned policy π θ (a|s; s g ) receiving a reward of 0 for each of the first T − 1 actions and a reward of r(s T ; s g ) after the last action, where r(s; s g ) ∈ [0, 1] for all s and s g 1 .

The goal achievement reward function r(s; s g ) measures the degree to which being in state s achieves goal s g .

The episode terminates upon the agent receiving the reward r(s T ; s g ) and a new episode begins.

It is straightforward to train π θ (a|s; s g ) in a tabular environment using the indicator reward r(s; s g ) = 1{s = s g }.

We are, however, interested in environments with continuous highdimensional observation spaces.

While there is extensive prior work on learning goal-conditioned policies BID19 BID37 BID0 BID16 BID34 , the reward function is often hand-crafted, limiting generality of the approaches.

In the few cases where the reward is learned, the learning objective is typically tied to a pre-specified notion of visual similarity.

Learning to achieve goals based purely on visual similarity is unlikely to work in complex, real world environments due to the possible variations in appearance of objects, or goal-irrelevant perceptual context.

We now turn to the problem of learning a goal achievement reward function r φ (s; s g ) with parameters φ for high-dimensional state spaces.

We aim to simultaneously learn a goal-conditioned policy π θ and a goal achievement reward function r φ by maximizing the mutual information between the goal state s g and the achieved state s T as shown in (1).

DISPLAYFORM0 Note that we are slightly overloading notation by treating s g as a random variable distributed according to p goal .

Similarly, s T is a random variable distributed according to the state distribution induced by running π θ for T steps for goal states sampled from p goal .The prior work of BID13 showed how to learn a set of abstract options by optimizing a similar objective, namely the mutual information between an abstract option and the achieved state.

Following their approach, we simplify (1) in two ways.

First, we rewrite the expectation in terms of the goal distribution p goal and the goal conditioned policy π θ .

Second, we lower bound the expectation term by replacing p(s g |s T ) with a variational distribution q φ (s g |s T ) with parameters φ following BID3 , leading to DISPLAYFORM1 Finally, we discard the entropy term H(s g ) from (2) because it does not depend on either the policy parameters θ or the variational distribution parameters φ, giving our overall objective DISPLAYFORM2 This objective may seem difficult to work with because the variational distribution q φ is a distribution over possible goals s g , which in our case are high-dimensional observations, such as images.

We sidestep the difficulty of directly modelling the density of high-dimensional observations by restricting the set of possible goals to be a finite subset of previously encountered states that evolves over time BID25 .

Restricting the support of q φ to a finite set of goals turns the problem of learning q φ into a problem of modelling the conditional distribution of possible intended goals given an achieved state, which obviates the requirement of modelling arbitrary statistical dependencies in the observations.

Optimization: The expectation in the DISCERN objective is with respect to the distribution of trajectories generated by the goal-conditioned policy π θ acting in the environment against goals drawn from the goal distribution p goal .

We can therefore optimize this objective with respect to policy parameters θ by repeatedly generating trajectories and performing reinforcement learning updates on π θ with a reward of log q φ (s g |s T ) given at time T and 0 for other time steps.

Optimizing the objective with respect to the variational distribution parameters φ is also straightforward since it is equivalent to a maximum likelihood classification objective.

As will be discussed in the next section, we found that using a reward that is a non-linear transformation mapping log q φ (s g |s T ) to [0, 1] worked better in practice.

Nevertheless, since the reward for the goal conditioned-policy is a function of log q φ (s g |s T ), training the variational distribution function q φ amounts to learning a reward function.

Communication Game Interpretation: Dual optimization of the DISCERN objective has an appealing interpretation as a cooperative communication game between two players -an imitator that corresponds to the goal-conditioned policy and a teacher that corresponds to the variational distribution.

At the beginning of each round or episode of the game the imitator is provided with a goal state.

The aim of the imitator is to communicate the goal state to the teacher by taking T actions in the environment.

After the imitator takes T actions, the teacher has to guess which state from a set of possible goals was given to the imitator purely from observing the final state s T reached by the imitator.

The teacher does this by assigning a probability to each candidate goal state that it was the goal given to the imitator at the start of the episode, i.e. it produces a distribution p(s g |s T ).

The objective of both players is for the teacher to guess the goal given to the imitator correctly as measured by the log probability assigned by the teacher to the correct goal.

We now describe the DISCERN algorithm -a practical instantiation of the approach for jointly learning π θ (a|s; s g ) and r(s; s g ) outlined in the previous section.

Goal distribution: We adopt a non-parametric approach to the problem of proposing goals, whereby we maintain a fixed size buffer G of past observations from which we sample goals during training.

We update G by replacing the contents of an existing buffer slot with an observation from the agent's recent experience according to some substitution strategy; in this work we considered two such strategies, detailed in Appendix A3.

This means that the space of goals available for training drifts as a function of the agent's experience, and states which may not have been reachable under a poorly trained policy become reachable and available for substitution into the goal buffer, leading to a naturally induced curriculum.

In this work, we sample training goals for our agent uniformly at random from the goal buffer, leaving the incorporation of more explicitly instantiated curricula to future work.

We train a goal achievement reward function r(s; s g ) used to compute rewards for the goal-conditioned policy based on a learned measure of state similarity.

We parameterize r(s; s g ) as the positive part of the cosine similarity between s and s g in a learned embedding space, although shaping functions other than rectification could be explored.

The state embedding in which we measure cosine similarity is the composition of a feature transformation h(·) and a learned L 2 -normalized mapping ξ φ (·).

In our implementation, where states and goals are represented as 2-D RGB images, we take h(·) to be the final layer features of the convolutional network learned by the policy in order to avoid learning a second convolutional network.

We find this works well provided that while training r, we treat h(·) as fixed and do not adapt the convolutional network's parameters with respect to the reward learner's loss.

This has the effect of regularizing the reward learner by limiting its adaptive capacity while avoiding the need to introduce a hyperparameter weighing the two losses against one another.

We train ξ φ (·) according to a goal-discrimination objective suggested by (3).

However, rather than using the set of all goals in the buffer G as the set of possible classes in the goal discriminator, we sample a small subset for each trajectory.

Specifically, the set of possible classes includes the goal g for the trajectory and DISPLAYFORM0 we maximize the log likelihood given by DISPLAYFORM1 where β is an inverse temperature hyperparameter which we fix to K + 1 in all experiments.

Note that (5) is a maximum log likelihood training objective for a softmax nearest neighbour classifier in a learned embedding space, making it similar to a matching network BID47 .

Intuitively, updating the embedding ξ φ using the objective in (5) aims to increase the cosine similarity between e(s T ) and e(g) and to decrease the cosine similarity between e(s T ) and the decoy embeddings e(d), . . .

, e(d K ).

Subsampling the set of possible classes as we do is a known method for approximate maximum likelihood training of a softmax classifier with many classes BID6 .We use max(0, g ) as the reward for reaching state s T when given goal g. We found that this reward function is better behaved than the reward logq(s g = g|s T ; d 1 , . . .

d K , π θ ) suggested by the DISCERN objective in Section 3 since it is scaled to lie in [0, 1].

The reward we use is also less noisy since, unlike logq, it does not depend on the decoy states.

Goal-conditioned policy: The goal-conditioned policy π θ (a|s; s g ) is trained to optimize the goal achievement reward r(s; s g ).

In this paper, π θ (a|s; s g ) is an -greedy policy of a goal-conditioned action-value function Q with parameters θ.

Q is trained using Q-learning and minibatch experience replay; specifically, we use the variant of Q(λ) due to Peng (see Chapter 7, BID40 ).

We use a form of goal relabelling BID19 or hindsight experience replay BID0 BID32 as a source successfully achieved goals as well as to regularize the embedding e(·).

Specifically, for the purposes of parameter updates (in both the policy and the reward learner) we substitute, with probability p HER the goal with an observation selected from the final H steps of the trajectory, and consider the agent to have received a reward of 1.

The motivation, in the case of the policy, is similar to that of previous work, i.e. that being in state s t should correspond to having achieved the goal of reaching s t .

When employed in the reward learner, it amounts to encouraging temporally consistent state embeddings BID29 BID38 , i.e. encouraging observations which are nearby in time to have similar embeddings.

Pseudocode for the DISCERN algorithm, decomposed into an experience-gathering (possibly distributed) actor process and a centralized learner process, is given in Algorithm 1.

The problem of reinforcement learning in the context of multiple goals dates at least to BID19 , where the problem was examined in the context of grid worlds where the state space is DISPLAYFORM0 /* See Appendix A3 */ end with probability p HER , Sample s HER uniformly from {s T −H , . . .

, s T } and set g ← s HER , r T ← 1 otherwise Compute g using (4) r T ← max(0, g ) Send (s 1:T , a 1:T , r 1:T , g) to the learner.

Poll the learner periodically for updated values of θ,φ.

Reset the environment if the episode has terminated.

until termination procedure LEARNER Input :Batch size B, number of decoys K, initial policy parameters θ, initial goal embedding parameters φ repeat Assemble batch of experience B = {(s DISPLAYFORM1 Use an off-policy reinforcement learning algorithm to update θ based on B Update φ to maximize DISPLAYFORM2 small and enumerable.

BID41 proposed generalized value functions (GVFs) as a way of representing knowledge about sub-goals, or as a basis for sub-policies or options.

Universal Value Function Approximators (UVFAs) BID37 extend this idea by using a function approximator to parameterize a joint function of states and goal representations, allowing compact representation of an entire class of conditional value functions and generalization across classes of related goals.

While the above works assume a goal achievement reward to be available a priori, our work includes an approach to learning a reward function for goal achievement jointly with the policy.

Several recent works have examined reward learning for goal achievement in the context of the Generative Adversarial Networks (GAN) paradigm BID12 .

The SPIRAL BID11 algorithm trains a goal conditioned policy with a reward function parameterized by a Wasserstein GAN BID1 discriminator.

Similarly, AGILE BID2 learns an instruction-conditional policy where goals in a grid-world are specified in terms of predicates which should be satisfied, and a reward function is learned using a discriminator trained to distinguish states achieved by the policy from a dataset of instruction, goal state pairs.

Reward learning has also been used in the context of imitation.

BID17 derives an adversarial network algorithm for imitation, while time-contrastive networks BID38 leverage pre-trained ImageNet classifier representations to learn a reward function for robotics skills from video demonstrations, including robotic imitation of human poses.

Universal Planning Networks (UPNs) BID39 ) learn a state representation by training a differentiable planner to imitate expert trajectories.

Experiments showed that once a UPN is trained the state representation it learned can be used to construct a reward function for visually specified goals.

Bridging goal-conditioned policy learning and imitation learning, BID34 learns a goal-conditioned policy and a dynamics model with supervised learning without expert trajectories, and present zero-shot imitation of trajectories from a sequence of images of a desired task.

A closely related body of work to that of goal-conditioned reinforcement learning is that of unsupervised option or skill discovery.

BID26 proposes a method based on an eigendecomposition of differences in features between successive states, further explored and extended in BID27 .

Variational Intrinsic Control (VIC) BID13 leverages the same lower bound on the mutual information as the present work in an unsupervised control setting, in the space of abstract options rather than explicit perceptual goals.

VIC aims to jointly maximize the entropy of the set of options while making the options maximally distinguishable from their final states according to a parametric predictor.

Recently, BID9 showed that a special case of the VIC objective can scale to significantly more complex tasks and provide a useful basis for low-level control in a hierarchical reinforcement learning context.

Other work has explored learning policies in tandem with a task policy, where the task or environment rewards are assumed to be sparse. propose a framework in which low-level skills are discovered in a pre-training phase of a hierarchial system based on simple-to-design proxy rewards, while BID36 explore a suite of auxiliary tasks through simultaneous off-policy learning.

Several authors have explored a pre-training stage, sometimes paired with fine-tuning, based on unsupervised representation learning.

BID35 and BID23 employ a two-stage framework wherein unsupervised representation learning is used to learn a model of the observations from which to sample goals for control in simple simulated environments.

BID31 propose a similar approach in the context of model-free Q-learning applied to 3-dimensional simulations and robots.

Goals for training the policy are sampled from the model's prior, and a reward function is derived from the latent codes.

This contrasts with our non-parametric approach to selecting goals, as well as our method for learning the goal space online and jointly with the policy.

An important component of our method is a form of goal relabelling, introduced to the reinforcement learning literature as hindsight experience replay by BID0 , based on the intuition that any trajectory constitutes a valid trajectory which achieves the goal specified by its own terminal observation.

Earlier, BID32 employed a related scheme in the context of supervised learning of motor programs, where a program encoder is trained on pairs of trajectory realizations and programs obtained by expanding outwards from a pre-specified prototypical motor program through the addition of noise.

BID45 expands upon hindsight replay and the all-goal update strategy proposed by BID19 , generalizing the latter to non-tabular environments and exploring related strategies for skill discovery, unsupervised pre-training and auxiliary tasks.

BID24 propose a hierarchical Q-learning system which employs hindsight replay both conventionally in the lower-level controller and at higher levels in the hierarchy.

BID31 also employ a generalized goal relabeling scheme whereby the policy is trained based on a trajectory's achievement not just of its own terminal observation, but a variety of retrospectively considered possible goals.

We evaluate, both qualitatively and quantitatively, the ability of DISCERN to achieve visuallyspecified goals in three diverse domains -the Arcade Learning Environment BID5 , continuous control tasks in the DeepMind Control Suite BID42 , and DeepMind Lab, a 3D first person environment BID4 .

Experimental details including architecture details, details of distributed training, and hyperparameters can be found in the Appendix.

We compared DISCERN to several baseline methods for learning goal-conditioned policies:Conditioned Autoencoder (AE): In order to specifically interrogate the role of the discriminative reward learning criterion, we replace the discriminative criterion for embedding learning with an L 2 reconstruction loss on h t ; that is, in addition to ξ φ (·), we learn an inverse mapping ξ −1 φ (·) with a separate set of parameters, and train both with the criterion h t − ξ DISPLAYFORM0 Conditioned WGAN Discriminator: We compare to an adversarial reward on the domains considered according to the protocol of BID11 , who successfully used a WGAN discriminator as a reward for training agents to perform inverse graphics tasks.

The discriminator takes two pairs of images -(1) a real pair of goal images (s g , s g ) and (2) a fake pair consisting of the terminal state of the agent and the goal frame (s t , s g ).

The output of the discriminator is used as the reward function for the policy.

Unlike our DISCERN implementation and the conditioned autoencoder baseline, we train the WGAN discriminator as a separate convolutional network directly from pixels, as in previous work.

Pixel distance reward (L2): Finally, we directly compare to a reward based on L 2 distance in pixel space, equal to exp − s t − s g 2 /σ pixel where σ pixel is a hyperparameter which we tuned on a per-environment basis.

All the baselines use the same goal-conditioned policy architecture as DISCERN.

The baselines also used hindsight experience replay in the same way as DISCERN.

They can therefore be seen as ablations of DISCERN's goal-achievement reward learning mechanism.

The suite of 57 Atari games provided by the Arcade Learning Environment (Bellemare et al., 2013) is a widely used benchmark in the deep reinforcement learning literature.

We compare DISCERN to other methods on the task of achieving visually specified goals on the games of Seaquest and Montezuma's Revenge.

The relative simplicity of these domains makes it possible to handcraft a detector in order to localize the controllable aspects of the environment, namely the submarine in Seaquest and Panama Joe, the character controlled by the player in Montezuma's Revenge.

We evaluated the methods by running the learned goal policies on a fixed set of goals and measured the percentage of goals it was able to reach successfully.

We evaluated both DISCERN and the baselines with two different goal buffer substitution strategies, uniform and diverse, which are described in the Appendix.

A goal was deemed to be successfully achieved if the position of the avatar in the last frame was within 10% of the playable area of the position of the avatar in the goal for each controllable dimension.

The controllable dimensions in Atari were considered to be the x-and y-coordinates of the avatar.

The results are displayed in FIG0 .

DISCERN learned to achieve a large fraction of goals in both Seaquest and Montezuma's Revenge while none of the baselines learned to reliably achieve goals in either game.

We hypothesize that the baselines failed to learn to control the avatars because their objectives are too closely tied to visual similarity.

FIG0 shows examples of goal achievement on Seaquest and Montezuma's Revenge.

In Seaquest, DISCERN learned to match the position of the submarine in the goal image while ignoring the position of the fish, since the fish are not directly controllable.

We have provided videos of the goal-conditioned policies learned by DISCERN on Seaquest and Montezuma's Revenge at the following anonymous URL https://sites.google.com/view/discern-anonymous/home.

The DeepMind Control Suite BID42 ) is a suite of continuous control tasks built on the MuJoCo physics engine BID44 .

While most frequently used to evaluate agents which receive the underlying state variables as observations, we train our agents on pixel renderings of the scene using the default environment-specified camera, and do not directly observe the state variables.

Agents acting greedily with respect to a state-action value function require the ability to easily maximize Q over the candidate actions.

For ease of implementation, as well as comparison to other considered environments, we discretize the space of continuous actions to no more than 11 unique actions per environment (see Appendix A4.1).The availability of an underlying representation of the physical state, while not used by the learner, provides a useful basis for comparison of achieved states to goals.

We mask out state variables relating to entities in the scene not under the control of the agent; for example, the position of the target in the reacher or manipulator domains.

DISCERN is compared to the baselines on a fixed set of 100 goals with 20 trials for each goal.

The goals are generated by acting randomly for 25 environment steps after initialization.

In the case of R a n d o m a g e n t P i x e l d i s t a n c e Figure 2: Average achieved frames for point mass (task easy), reacher (task hard), manipulator (task bring ball), pendulum (task swingup), finger (task spin) and ball in cup (task catch) environments.

The goal is shown in the top row and the achieved frame is shown in the bottom row.cartpole, we draw the goals from a random policy acting in the environment set to the balance task, where the pole is initialized upwards, in order to generate a more diverse set of goals against which to measure.

FIG1 compares learning progress of 5 independent seeds for the "uniform" goal replacement strategy (see Appendix A5 for results with "diverse" goal replacement) for 6 domains.

We adopt the same definition of achievement as in Section 6.1.

Figure 2 summarizes averaged goal achievement frames on these domains except for the cartpole domain for policies learned by DISCERN.

Performance on cartpole is discussed in more detail in FIG4 of the Appendix.

The results show that in aggregate, DISCERN outperforms baselines in terms of goal achievement on several, but not all, of the considered Control Suite domains.

In order to obtain a more nuanced understanding of DISCERN's behaviour when compared with the baselines, we also examined achievement in terms of the individual dimensions of the controllable state.

Figure 4 shows goal achievement separately for each dimension of the underlying state on four domains.

The perdimension results show that on difficult goal-achievement tasks such as those posed in cartpole (where most proposed goal states are unstable due to the effect of gravity) and finger (where a free-spinning piece is only indirectly controllable) DISCERN learns to reliably match the major dimensions of controllability such as the cart position and finger pose while ignoring the other Actor steps Figure 4 : Per-dimension quantitative evaluation of goal achievement on continuous control domains using the "uniform" goal substitution scheme (Appendix A3).

Each subplot corresponds to a domain, with each group of colored rows representing a method.

Each individual row represents a dimension of the controllable state (such as a joint angle).

The color of each cell indicates the fraction of goal states for which the method was able to match the ground truth value for that dimension to within 10% of the possible range.

The position along the x-axis indicates the point in training in millions of frames.

For example, on the reacher domain DISCERN learns to match both dimensions of the controllable state, but on the cartpole domain it learns to match the first dimension (cart position) but not the second dimension (pole angle).dimensions, whereas none of the baselines learned to reliably match any of the controllable state dimensions on the difficult tasks cartpole and finger.

We omitted the manipulator domain from these figures as none of the methods under consideration achieved non-negligible goal achievement performance on this domain, however a video showing the policy learned by DISCERN on this domain can be found at https://sites.google.com/view/discern-anonymous/home.

The policy learned on the manipulator domain shows that DISCERN was able to discover several major dimensions of controllability even on such a challenging task, as further evidenced by the per-dimension analysis on the manipulator domain in Figure 8 in the Appendix.

DeepMind Lab BID4 ) is a platform for 3D first person reinforcement learning environments.

We trained DISCERN on the watermaze level and found that it learned to approximately achieve the same wall and horizon position as in the goal image.

While the agent did not learn to achieve the position and viewpoint shown in a goal image as one may have expected, it is encouraging that our approach learns a reasonable space of goals on a first-person 3D domain in addition to domains with third-person viewpoints like Atari and the DM Control Suite.

We have presented a system that can learn to achieve goals, specified in the form of observations from the environment, in a purely unsupervised fashion, i.e. without any extrinsic rewards or expert demonstrations.

Integral to this system is a powerful and principled discriminative reward learning objective, which we have demonstrated can recover the dominant underlying degrees of controllability in a variety of visual domains.

In this work, we have adopted a fixed episode length of T in the interest of simplicity and computational efficiency.

This implicitly assumes not only that all sampled goals are approximately achievable in T steps, but that the policy need not be concerned with finishing in less than the allotted number of steps.

Both of these limitations could be addressed by considering schemes for early termination based on the embedding, though care must be taken not to deleteriously impact training by terminating episodes too early based on a poorly trained reward embedding.

Relatedly, our goal selection strategy is agnostic to both the state of the environment at the commencement of the goal episode and the current skill profile of the policy, utilizing at most the content of the goal itself to drive the evolution of the goal buffer G. We view it as highly encouraging that learning proceeds using such a naive goal selection strategy, however more sophisticated strategies, such as tracking and sampling from the frontier of currently achievable goals BID16 , may yield substantial improvements.

DISCERN's ability to automatically discover controllable aspects of the observation space is a highly desirable property in the pursuit of robust low-level control.

A natural next step is the incorporation of DISCERN into a deep hierarchical reinforcement learning setup BID46 BID24 BID30 where a meta-policy for proposing goals is learned after or in tandem with a low-level controller, i.e. by optimizing an extrinsic reward signal.

We employ a distributed reinforcement learning architecture inspired by the IMPALA reinforcement learning architecture BID8 , with a centralized GPU learner batching parameter updates on experience collected by a large number of CPU-based parallel actors.

While BID8 learns a stochastic policy through the use of an actor-critic architecture, we instead learn a goal-conditioned state-action value function with Q-learning.

Each actor acts -greedily with respect to a local copy of the Q network, and sends observations s t , actions a t , rewards r t and discounts γ t for a trajectory to the learner.

Following BID18 , we use a different value of for each actor, as this has been shown to improve exploration.

The learner batches re-evaluation of the convolutional network and LSTM according to the action trajectories supplied and performs parameter updates, periodically broadcasting updated model parameters to the actors.

As Q-learning is an off-policy algorithm, the experience traces sent to the learner can be used in the usual n-step Q-learning update without the need for an off-policy correction as in BID8 .

We also maintain actor-local replay buffers of previous actor trajectories and use them to perform both standard experience replay BID25 and our variant of hindsight experience replay BID0 .

Our network architectures closely resemble those in BID8 , with policy and value heads replaced with a Q-function.

We apply the same convolutional network to both s t and s g and concatenate the final layer outputs.

Note that the convolutional network outputs for s g need only be computed once per episode.

We include a periodic representation (sin(2πt/T ), cos(2πt/T )) of the current time step, with period equal to the goal length achievement period T , as an extra input to the network.

The periodic representation is processed by a single hidden layer of rectified linear units and is concatenated with the visual representations fed to the LSTM.

While not strictly necessary, we find that this allows the agent to become better at achieving goal states which may be unmaintainable due to their instability in the environment dynamics.

The output of the LSTM is the input to a dueling action-value output network BID48 .

In all of our experiments, both branches of the dueling network are linear mappings.

That is, given LSTM outputs ψ t , we compute the action values for the current time step t as DISPLAYFORM0

We experimented with two strategies for updating the goal buffer.

In the first strategy, which we call uniform, the current observation replaces a uniformly selected entry in the goal buffer with probability p replace .

The second strategy, which we refer to as diverse goal sampling attempts to maintain a goal buffer that more closely approximates the uniform distribution over all observation.

In the diverse goal strategy, we consider the current observation for addition to the goal buffer with probability p replace at each step during acting.

If the current observation s is considered for addition to the goal buffer, then we select a random removal candidate s r by sampling uniformly from the goal buffer and replace it with s if s r is closer to the rest of the goal buffer than s. If s is closer to the rest of the goal buffer than s r then we still replace s r with s with probability p add−non−diverse .

We used L 2 distance in pixel space for the diverse sampling strategy and found it to greatly increase the coverage of states in the goal buffer, especially early during training.

This bears some relationship to Determinantal Point Processes BID20 , and goal-selection strategies with a more explicit theoretical foundation are a promising future direction.

The following hyper-parameters were used in all of the experiments described in Section 6.

All weight matrices are initialized using a standard truncated normal initializer, with the standard deviation inversely proportional to the square root of the fan-in.

We maintain a goal buffer of size 1024 and use p replace = 10 −3 .

We also use p add−non−diverse = 10 −3 .

For the teacher, we choose ξ φ (·) to be an L 2 -normalized single layer of 32 tanh units, trained in all experiments with 4 decoys (and thus, according to our heuristic, β equal to 5).

For hindsight experience replay, a highsight goal is substituted 25% of the time.

These goals are chosen uniformly at random from the last 3 frames of the trajectory.

Trajectories were set to be 50 steps long for Atari and DeepMind Lab and 100 for the DeepMind control suite.

It is important to note that the environment was not reset after each trajectory, but rather the each new trajectory begins where the previous one ended.

We train the agent and teacher jointly with RMSProp BID43 ) with a learning rate of 10 −4 .

We follow the preprocessing protocol of BID28 , resizing to 84 × 84 pixels and scaling 8-bit pixel values to lie in the range [0, 1].

While originally designed for Atari, we apply this preprocessing pipeline across all environments used in this paper.

In the point mass domain we use a control step equal to 5 times the task-specified default, i.e. the agent acts on every fifth environment step BID28 .

In all other Control Suite domains, we use the default.

We use the "easy" version of the task where actuator semantics are fixed across environment episodes.

Discrete action spaces admit function approximators which simultaneously compute the action values for all possible actions, as popularized in BID28 .

The action with maximal Q-value can thus be identified in time proportional to the cardinality of the action space.

An enumeration of possible actions is no longer possible in the continuous setting.

While approaches exist to enable continuous maximization in closed form BID14 , they come at the cost of greatly restricting the functional form of Q.For ease of implementation, as well as comparison to other considered environments, we instead discretize the space of continuous actions.

For all Control Suite environments considered except manipulator, we discretize an A-dimensional continuous action space into 3A discrete actions, consisting of the Cartesian product over action dimensions with values in {−1, 0, 1}. In the case of manipulator, we adopt a "diagonal" discretization where each action consists of setting one actuator to ±1, and all other actuators to 0, with an additional action consisting of every actuator being set to 0.

This is a reasonable choice for manipulator because any position can be achieved by a concatenation of actuator actions, which may not be true of more complex Control Suite environments such as humanoid, where the agent's body is subject to gravity and successful trajectories may require multi-joint actuation in a single control time step.

The subset of the Control Suite considered in this work was chosen primarily such that the discretized action space would be of a reasonable size.

We leave extensions to continuous domains to future work.

We ran two additional baselines on Seaquest and Montezuma's Revenge, ablating our use of hindsight experience replay in opposite ways.

One involved training the goal-conditioned policy only in hindsight, without any learned goal achievement reward, i.e. p HER = 1.

This approach achieved 12% of goals on Seaquest and 11.4% of goals on Montezuma's Revenge, making it comparable to a uniform random policy.

This result underscores the importance of learning a goal achievement reward.

The second baseline consisted of DISCERN learning a goal achievement reward without hindsight experience replay, i.e. p HER = 0.

This also performed poorly, achieving 11.4% of goals on Seaquest and 8% of goals on Montezuma's Revenge.

Taken together, these preliminary results suggest that the combination of hindsight experience replay and a learned goal achievement reward is important.

For the sake of completeness, FIG3 reports goal achievement curves on Control Suite domains using the "diverse" goal selection scheme.

Figure 4 for a description of the visualization.

DISCERN learns to reliably control more dimensions of the underlying state than any of the baselines.

@highlight

Unsupervised reinforcement learning method for learning a policy to robustly achieve perceptually specified goals.
Representation learning is a central challenge across a range of machine learning areas.

In reinforcement learning, effective and functional representations have the potential to tremendously accelerate learning progress and solve more challenging problems.

Most prior work on representation learning has focused on generative approaches, learning representations that capture all the underlying factors of variation in the observation space in a more disentangled or well-ordered manner.

In this paper, we instead aim to learn functionally salient representations: representations that are not necessarily complete in terms of capturing all factors of variation in the observation space, but rather aim to capture those factors of variation that are important for decision making -- that are "actionable".

These representations are aware of the dynamics of the environment, and capture only the elements of the observation that are necessary for decision making rather than all factors of variation, eliminating the need for explicit reconstruction.

We show how these learned representations can be useful to improve exploration for sparse reward problems, to enable long horizon hierarchical reinforcement learning, and as a state representation for learning policies for downstream tasks.

We evaluate our method on a number of simulated environments, and compare it to prior methods for representation learning, exploration, and hierarchical reinforcement learning.

Representation learning refers to a transformation of an observation, such as a camera image or state observation, into a form that is easier to manipulate to deduce a desired output or perform a downstream task, such as prediction or control.

In reinforcement learning (RL) in particular, effective representations are ones that enable generalizable controllers to be learned quickly for challenging and temporally extended tasks.

While end-to-end representation learning with full supervision has proven effective in many scenarios, from supervised image recognition BID21 to vision-based robotic control , devising representation learning methods that can use unlabeled data or experience effectively remains an open problem.

Much of the prior work on representation learning in RL has focused on generative approaches.

Learning these models is often challenging because of the need to model the interactions of all elements of the state.

We instead aim to learn functionally salient representations: representations that are not necessarily complete in capturing all factors of variation in the observation space, but rather aim to capture factors of variation that are relevant for decision making -that are actionable.

How can we learn a representation that is aware of the dynamical structure of the environment?

We propose that a basic understanding of the world can be obtained from a goal-conditioned policy, a policy that can knows how to reach arbitrary goal states from a given state.

Learning how to execute shortest paths between all pairs of states suggests a deep understanding of the environment dynamics, and we hypothesize that a representation incorporating the knowledge of a goal-conditioned policy can be readily used to accomplish more complex tasks.

However, such a policy does not provide a readily usable state representation, and it remains to choose how an effective state representation should be extracted.

We want to extract those factors of the state observation that are critical for deciding which action to take.

We can do this by comparing which actions a goal-conditioned policy takes for two different goal states.

Intuitively, if two goal states require different actions, then they are functionally different and vice-versa.

This principle is illustrated in the diagram in Figure 1 .

Based on this principle, we propose actionable representations for control (ARC), representations in which Euclidean distances between states correspond to expected differences between actions taken to reach them.

Such representations emphasize factors in the state that induce significant differences in the corresponding actions, and de-emphasize those features that are irrelevant for control.

Figure 1: Actionable representations: 3 houses A, B, C can only be reached by indicated roads.

The actions taken to reach A, B, C are shown by arrows.

Although A, B are very close in space, they are functionally different.

The car has to take a completely different road to reach A, compared to B and C. Representations z A , z B , z C learn these functional differences to differentiate A from B and C, while keeping B and C close.

While learning a goal-conditioned policy to extract such a representation might itself represent a daunting task, it is worth noting that such a policy can be learned without any knowledge of downstream tasks, simply through unsupervised exploration of the environment.

It is reasonable to postulate that, without active exploration, no representation learning method can possibly acquire a dynamics-aware representation, since understanding the dynamics requires experiencing transitions and interactions, rather than just observations of valid states.

As we demonstrate in our experiments, representations extracted from goal-conditioned policies can be used to better learn more challenging tasks than simple goal reaching, which cannot be easily contextualized by goal states.

The process of learning goal-conditioned policies can also be made recursive, so that the actionable representations learned from one goalconditioned policy can be used to quickly learn a better one.

Actionable representations for control are useful for a number of downstream tasks: as representations for task-specific policies, as representations for hierarchical RL, and to construct well-shaped reward functions.

We show that ARCs enable these applications better than representations that are learned using unsupervised generative models, predictive models, and other prior representation learning methods.

We analyze structure of the learned representation, and compare the performance of ARC with a number of prior methods on downstream tasks in simulated robotic domains such as wheeled locomotion, legged locomotion, and robotic manipulation.

Goal-conditioned reinforcement learning.

In RL, the goal is to learn a policy π θ (a t |s t ) that maximizes the expected return R t = E π θ [ t r t ].

Typically, RL learns a single task that optimizes for a particular reward function.

If we instead would like to train a policy that can accomplish a variety of tasks, we might instead train a policy that is conditioned on another input -a goal.

When the different tasks directly correspond to different states, this amounts to conditioning the policy π on both the current and goal state.

The policy π θ (a t |s t , g) is trained to reach goals from the state space g ∼ S, by optimizing E g∼S [E π θ (a|s,g) (R g ))], where R g is a reward for reaching the goal g.

Maximum entropy RL.

Maximum entropy RL algorithms modify the RL objective, and instead learns a policy to maximize the reward as well as the entropy of the policy BID15 BID36 , according to π = arg max π E π [r(s, a)] + H(π).

In contrast to standard RL, where optimal policies in fully observed environments are deterministic, the solution in maximum entropy RL is a stochastic policy, where the entropy reflects the sensitivity of the rewards to the action: when the choice of action has minimal effect on future rewards, actions are more random, and when the choice of action is critical, the actions are more deterministic.

In this way, the action distributions for a maximum entropy policy carry more information about the dynamics of the task.

For a pair of states s 1 , s 2 , the divergence between the goal-conditioned action distributions they induce defines the actionable distance D Act , which in turn is used to learn representation φ.

In this work, we extract a representation that can distinguish states based on actions required to reach them, which we term an actionable representation for control (ARC).

In order to learn state representations φ that can capture the elements of the state which are important for decision making, we first consider defining actionable distances D Act (s 1 , s 2 ) between states.

Actionable distances are distances between states that capture the differences between the actions required to reach the different states, thereby implicitly capturing dynamics.

If actions required for reaching state s 1 are very different from the actions needed for reaching state s 2 , then these states are functionally different, and should have large actionable distances.

This subsequently allows us to extract a feature representation(φ(s)) of state, which captures elements that are important for decision making.

To formally define actionable distances, we build on the framework of goal-conditioned RL.

We assume that we have already trained a maximum entropy goal-conditioned policy π θ (a|s, g) that can start at an arbitrary state s 0 ∈ S in the environment, and reach a goal state s g ∈ S. Although this is a significant assumption, we will discuss later how this is in fact reasonable in many settings.

We can extract actionable distances by examining how varying the goal state affects action distributions for goal-conditioned policies.

Formally, consider two different goal states s 1 and s 2 .

At an intermediate state s, the goal-conditioned policy induces different action distributions π θ (a|s, s 1 ) and π θ (a|s, s 2 ) to reach s 1 and s 2 respectively.

If these distributions are similar over many intermediate states s, this suggests that these states are functionally similar, while if these distributions are different, then the states must be functionally different.

This motivates a definition for actionable distances D Act as DISPLAYFORM0 The distance consists of the expected divergence over all initial states s (refer to Section B for how we do this practically).

If we focus on a subset of states, the distance may not capture action differences induced elsewhere, and can miss functional differences between states.

Since maximum entropy policies learn unique optimal stochastic policies, the actionable distance is well-defined and unambiguous.

Furthermore, because max-ent policies capture sensitivity of the value function, goals are similar under ARC if they require the same action and they are equally "easy" to reach.

We can use D Act to extract an actionable representation of state.

To learn this representation φ(s), we optimize φ such that Euclidean distance between states in representation space corresponds to actionable distances D Act between them.

This optimization yields good representations of state because it emphasizes the functionally relevant elements of state, which significantly affect the actionable distance, while suppressing less functionally relevant elements of state.

The problem is: DISPLAYFORM1 This objective yields representations where Euclidean distances are meaningful.

This is not necessarily true in the state space or in generative representations (Section 6.4).

These representations are meaningful for several reasons.

First, since we are leveraging a goal-conditioned policy, they are aware of dynamics and are able to capture local connectivity of the environment.

Secondly, the representation is optimized so that it captures only the functionally relevant elements of state.

Requirement for Goal Conditioned Policy: A natural question to ask is whether needing a goalconditioned policy is too strong of a prerequisite.

However, it is worth noting that the GCP can be trained with existing RL methods (TRPO) using a sparse task-agnostic reward (Section 6.2, Appendix A.1) -obtaining such a policy is not especially difficult, and existing methods are quite capable of doing so BID29 ).

Furthermore, it is likely not possible to acquire a functionalityaware state representation without some sort of active environment interaction, since dynamics can only be understood by observing outcomes of actions, rather than individual states.

Importantly, we discuss in the following section how ARCs help us solve tasks beyond what a simple goalconditioned policy can achieve.

A natural question that emerges when learning representations from a goal-conditioned policy pertains to what such a representation enables over the goal-conditioned policy itself.

Although goalconditioned policies enable reaching between arbitrary states, they suffer from fundamental limitations: they do not generalize very well to new states, and they are limited to solving only goalreaching tasks.

We show in our empirical evaluation that the ARC representation expands meaningfully over these limitations of a goal-conditioned policy -to new tasks and to new regions of the environment.

In this section, we detail how ARCs can be used to generalize beyond a goalconditioned policy to help solve tasks that cannot be expressed as goal reaching (Section 4.1), tasks involving larger regions of state space (Section 4.2), and temporally extended tasks which involve sequences of goals (Section 4.3).

Goal-conditioned policies are trained with only a goal-reaching reward, and so are unaware of reward structures used for other tasks in the environment which do not involve simple goal-reaching.

Tasks which cannot be expressed as simply reaching a goal are abundant in real life scenarios such as navigation under non-uniform preferences or manipulation with costs on quality of motion, and for such tasks, using the ARC representation as input for a policy or value function can make the learning problem easier.

We can learn a policy for a downstream task, of the form π θ (a|φ(s)), using the representation φ(s) instead of state s.

The implicit understanding of the environment dynamics in the learned representation prioritizes the parts of the state that are most important for learning, and enables quicker learning for these tasks as we see in Section 6.6.

We can use ARC to construct better-shaped reward functions.

It is common in continuous control to define rewards in terms of some distance to a desired state, oftentimes using Euclidean distance BID33 BID26 .

However, Euclidean distance in state space is not necessarily a meaningful metric of functional proximity.

ARC provides a better metric, since it directly accounts for reachability.

We can use the actionable representation to define better-shaped reward functions for downstream tasks.

We define a shaping of this form to be the negative Euclidean distance between two states in ARC space: −||φ(s 1 ) − φ(s 2 )|| 2 : for example, on a goal-reaching task r(s) = r sparse (s, s g ) − ||φ(s) − φ(s g )|| 2 .

This allows us to explore and learn policies even in the presence of sparse reward functions.

One may wonder whether, instead of using ARCs for reward shaping, we might directly use the goalconditioned policy to reach a particular goal.

As we will illustrate in Section 6.5, the representation typically generalizes better than the goal-conditioned policy.

Goal-conditioned policies typically can be trained on small regions of the state space, but don't extrapolate well to new parts of the state space.

We observe that ARC exhibits better generalization, and can provide effective reward shaping for goals that are very difficult to reach with the goal-conditioned policy.

Goal-conditioned policies can serve as low-level controllers for hierarchical tasks which require synthesizing a particular sequence of behaviours, and thus not expressible as a single goal-reaching objective.

One approach to solving such tasks learns a high-level controller π meta (g|s) via RL that produces desired goal states for a goal-conditioned policy to reach sequentially BID27 .

The high-level controller suggests a goal, which the goal conditioned policy attempts to reach for several time-steps, following which the high-level controller picks a new goal.

For many tasks, naively training such a high-level controller which outputs goals directly in state space is unlikely to perform well, since such a controller must disentangle the relevant attributes in the goal for long horizon reasoning.

We consider two schemes to use ARCs for hierarchical RL -learning a high level policy which commands directly in ARC space or commands in a clustered latent space.

HRL directly in ARC space: ARC representations provide a better goal space for high-level controllers, since they de-emphasize components of the goal space irrelevant for determining the optimal action.

In this scheme, the high-level controller π meta (z|s) observes the current state and generates a distribution over points in the latent space.

At every meta-step, a sample z h is taken from π meta (z|s) which represents the high level command.

z h is then translated into a goal g h via a decoder which is trained to reconstruct states from their corresponding goals.

This goal g h can then be used to command the goal conditioned policy for several time steps, before resampling again from π meta .

A high-level controller producing outputs in ARC space does not need to rediscover saliency in the goal space, which makes the search problem less noisy and more accurate.

We show in Section 4.3 that using ARC as a hierarchical goal space enables significant improvement for waypoint navigation tasks.

Clustering in ARC space: Since ARC captures the topology of the environment, clusters in ARC space often correspond to semantically meaningful state abstractions.

We utilize these clusters, with the intuition that a meta-controller searching in "cluster space" should learn faster than directly outputting states.

In this scheme, we first build a discrete number of clusters by clustering the points that the goal conditioned policy is trained on using the k-means algorithm within the ARC representation space.

We then train a high-level controller π meta (c|s) which observes a state s and generates a distribution over discrete clusters c. At every meta-step, a cluster sample c h is taken from π meta (c|s).

A goal in state space g h is then chosen uniformly at random from points within the cluster c h and used to command the GCP for several time steps before the next cluster is sampled from π meta .

We train a meta-policy to output clusters, instead of states: π meta (cluster|s).

We see that for hierarchical tasks with less granular reward functions such as room navigation, performing RL in "cluster space" induced by ARC outperform cluster spaces induced by other representations, since the distance metric is much more meaningful in ARC space.

The capability to learn effective representations is a major advantage of deep neural network models.

These representations can be acquired implicitly, through end-to-end training BID13 , or explicitly, by formulating and optimizing a representation learning objective.

A classic approach to representation learning is generative modeling, where a latent variable model is trained to model the data distribution, and the latent variables are then used as a representation BID32 BID10 BID19 BID12 BID9 BID14 BID16 .

In the context of control and sequence models, generative models have also been proposed to model transitions BID38 BID1 BID41 BID22 .

While generative models are general and principled, they must not only explain the entirety of the input observation, but must also generate it.

Several methods perform representation learning without generation, often based on contrastive losses BID34 BID37 BID3 BID8 BID39 .

While these methods avoid generation, they either still require modeling of the entire input, or utilize heuristics that encode user-defined information.

In contrast, ARCs are directly trained to focus on decision-relevant features of input, providing a broadly applicable objective that is still selective about which aspects of input to represent.

In the context of RL and control, representation learning methods have been used for many downstream applications BID24 , including representing value functions BID2 and building models BID38 BID1 BID41 .

Our approach is complementary: it can also be applied to these applications.

Several works have sought to learn representations that are specifically suited for physical dynamical systems BID18 and that use interaction to build up dynamics-aware features BID4 LaversanneFinot et al., 2018) .

In contrast to BID18 , our method does not attempt to encode all physically-relevant features of state, only those relevant for choosing actions.

In contrast to BID4 ; BID23 , our approach does not try to determine which features of the state can be independently controlled, but rather which features are relevant for choosing controls.

BID35 also consider learning representations through goal-directed behaviour, but receives supervision through demonstrations instead of active observation.

Related methods learn features that are predictive of actions based on pairs of sequential states (so-called inverse models) BID0 BID31 BID40 .

More recent work such as BID7 ) perform a large scale study of these types of methods in the context of exploration.

Unlike ARC, which is learned from a policy performing long-horizon control, inverse models are not obliged to represent all relevant features for multi-step control, and suffer from greedy reasoning.

The aim of our experimental evaluation is to study the following research questions:1.

Can we learn ARCs for multiple continuous control environments?

What are the properties of these learned representations?

2.

Can ARCs be used as feature representations for learning policies quickly on new tasks?

3.

Can reward shaping with ARCs enable faster learning?

4.

Do ARCs provide an effective mechanism for hierarchical RL?Full experimental and hyperparemeter tuning details are presented in the appendix.

We study six simulated environments as illustrated in FIG2 : 2D navigation tasks in two settings, wheeled locomotion tasks in two settings, legged locomotion, and object pushing with a robotic gripper.

The 2D navigation domains consist of either a room with a central divider wall or four rooms.

Wheeled locomotion involves a two-wheeled differential drive robot, either in free space or with four rooms.

For legged locomotion, we use a quadrupedal ant robot, where the state space consists of all joint angles, along with the Cartesian position of the center of mass (CoM).

The manipulation task uses a simulated Sawyer arm to push an object, where the state consists of endeffector and object positions.

Further details are presented in Appendix C.These environments present interesting representation learning challenges.

In 2D navigation, the walls impose structure similar to those in Figure 1 : geometrically proximate locations on either side of a wall are far apart in terms of reachability.

The locomotion environments present an additional challenge: an effective representation must account for the fact that the internal joints of each robot (legs or wheel orientation) are less salient for long-horizon tasks than CoM. The original state representation does not reflect this structure: joint angles expressed in radians carry as much weight as CoM positions in meters.

In the object manipulation task, a key representational challenge is to distinguish between pushing the block and simply moving the arm in free space.

We first learn a stochastic goal-conditioned policy parametrized by a neural network which outputs actions given the current state and the desired goal.

This goal-conditioned policy is trained using a sparse reward using entropy-regularized Trust Region Policy Optimization (TRPO) (Schulman et al., 2015) .

For a discussion of the assumption about the existence of a goal-conditioned policy, please refer to Section 3.

Exact details about the training procedure, the reward function, and hyperparameters are presented in Appendix A.1.

To train the ARC representation, we collect a dataset of 500 trajectories with horizon 100 from the goal-conditioned policy, where each trajectory has an arbitrary start state and intended goal state.

We optimize Eqn 2 as a supervised learning problem using this dataset to train the representation, computing the relevant expectations by uniform sampling from states in the dataset.

A detailed outline of the training procedure, along with hyperparameter and architecture choices, is presented in Appendix A.2.

We compare ARC to other representation learning methods used in previous works for control: variational autoencoders (Kingma & Welling, 2013) (VAE), variational autoencoders trained for feature slowness BID18 (slowness), features extracted from a predictive model BID30 (predictive model), features extracted from inverse models BID0 BID6 , and a naïve baseline that uses the full state space as the representation (state).

Details of the exact objectives used to train these methods is provided in Appendix B.For each downstream task in Section 4, we also compare with alternative approaches for solving the task not involving representation learning.

For reward shaping, we compare with VIME BID17 , an exploration method based on novelty bonuses.

For hierarchical RL, we compare with option critic BID20 and an on-policy adaptation of HIRO BID27 .

We also compare to model-based reinforcement learning with MPC BID28 , a method which explicitly learns and uses environment dynamics, as compared to the implicit dynamics learnt by ARC.

Because sample complexity of model-based and model-free methods differ, all results with model-based reinforcement learning indicate final performance.

To ensure a fair comparison between the methods, we provide the same information and trajectory data that ARC receives to all of the representation learning methods.

Each representation is trained on the same dataset of trajectories collected from the goal-conditioned policy, ensuring that each comparisons receives data from the full state distribution and meaningful transitions.

We analyze the structure of ARC space for the tasks described in Section 6.1, to identify which factors of state ARC chooses to emphasize, and how system dynamics affect the representation.

In the 2D navigation tasks, we visualize the original state and learned representations in FIG3 .

In both environments, ARC reflects the dynamics: points close by in Euclidean distance in the original state space are distant in representation space when they are functionally distinct.

For instance, there is a clear separation in the latent space where the wall should be, and points on opposite sides of the wall are much further apart in ARC space ( FIG3 ) than in the original environment and in the VAE representation.

In the room navigation task, the passages between rooms are clear bottlenecks, and the ARC representation separates the rooms according to these bottlenecks. , and less for secondary elements (shown in purple).

ARC exhibits this property, with a spread orange region -robot CoM or object position, and a suppressed purple region -joint angles and other secondary elements.

The VAE and naive state representations do not capture this saliency, containing spread purple regions.

The representations learned in more complex domains, such as wheeled or legged locomotion and block manipulation, also show meaningful patterns.

We aim to understand which elements of state are being emphasized by the representation, by analyzing how distances in the latent space change as we perturb various elements of state.

FIG4 .

For each environment, we determine two factors in the state: one which we consider salient for decision making (in orange), and one which is secondary (in purple).

We expect a good representation to have a larger variation in distance as we perturb the important factor than when we perturb the secondary factor.

In the legged locomotion environment, the CoM is the important factor and the joint angles are secondary.

As we perturb the CoM, the representation should vary significantly, while the effect should be muted as we perturb the joints.

For the wheeled environment, position of the car should cause large variations while the orientation should be secondary.

For the object pushing, we expect block position to be salient and end-effector position to be secondary.

Since distances in the high-dimensional representation space are hard to visualize, we project [ARC, VAE, State] representations of perturbed states into 2 dimensions FIG4 using multi-dimensional scaling (MDS) BID5 , which projects points while preserving Euclidean distances.

FIG4 we see that ARC captures the factors of interest; as the important factor is perturbed the representation changes significantly (spread out orange points), while when the secondary factor is perturbed the representation changes minimally (close together purple points).

This implies that for Ant, ARC captures CoM while suppressing joint angles; for wheeled, ARC captures position while suppressing orientation; for block pushing, ARC captures block position, suppressing arm movement.

Both VAE representations and original state space are unable to capture this.

As desribed in Section 4.2, distances in ARC space can be used for reward shaping to solve tasks that present a large exploration challenge with sparse reward functions.

We investigate this on two challenging exploration tasks for wheeled locomotion and legged locomotion (seen in FIG5 .

We acquire an ARC from a goal-conditioned policy in the region S where the CoM is within a 2m square.

The learned representation is then used to guide learning via reward shaping for learning a goal-conditioned policy on a larger region S , where the CoM is within a square of 8m.

The task is to reach arbitrary goals in S , but with only a sparse goal completion reward, so exploration is challenging.

We shape the reward with a term corresponding to distance between the representation of the current and desired state: r(s, g) = r sparse − φ(s) − φ(g) 2 .

To ensure fairness, all comparisons initialize from the goal-conditioned policy on small region S and train on the same data.

Further details on the experimental setup for this domain can be found in Appendix A.3.

As shown in FIG5 ARC demonstrates faster learning speed and better asymptotic performance over all compared methods, when all are initialized from the goal conditioned policy trained on the small region.

This can be attributed to the fact that, unlike the other representation learning algorithms, the ARC representation explicitly optimizes for functional distances in latent space, which generalizes well to a larger domain since the functionality in the new space is preserved.

The performance of ARC is similar to a hand-designed reward shaping corresponding to distance in COM space, corroborating FIG4 that ARC considers CoM to be the most salient feature.

We notice that representations which are dynamics-aware (ARC, predictive models, inverse models) outperform VIME, which uses a novelty-based exploration strategy without considering environment dynamics, indicating that effectively incorporating dynamics information into representations can help tackle exploration challenges in large environments.

We consider using the ARC representation as a feature space for learning policies for tasks that cannot be expressed with a goal-reaching objective.

We consider a quadruped ant robot task which requires the agent to reach a target (shown in green in FIG6 while avoiding a dangerous region (shown in red in FIG6 .

Instead of learning a policy from state π(a|s), we learn a policy using a representation φ as features π(a|φ(s)).

It is important to note that this task cannot be solved directly by a goal-conditioned policy (GCP), and a GCP attempting to reach the specified goal will walk through the dangerous region and receive a reward of -760.

The reward function for this task and other experimental details are noted in Appendix A.4.

Although all the methods ultimately learn to solve the task, policies using ARC features learn at a significantly faster rate FIG6 ).

Policies using ARC features solve the task by Iteration 100, by which point all other methods can only solve with 5% success.

We attribute the rapid learning progress to the ability of ARC to emphasize elements of the state that are important for multi-timestep control, rather than greedy features discovered by reconstruction or one-step prediction.

Features which emphasize elements important for control make learning easier because they reduce redundancy and noise in the input, and allows the RL algorithm to effectively assign credit.

We further note that other representation learning methods learn only as fast as the original state representation, and model-based MPC controllers BID28 also perform suboptimally.

It is important to note that the same representation can be used to quickly train many different tasks, amortizing the cost of training a GCP.6.7 BUILDING HIERARCHY FROM ACTIONABLE REPRESENTATIONS Figure 9 : Waypoint and multi-room HRL tasksWe consider using ARC representations to control high-level controllers for learning temporally extended navigation tasks in room and waypoint navigation settings, as described in Section 4.

In the multi-room environments, the agent must navigate through a sequence of 50 rooms in order, receiving a sparse reward when it enters the correct room.

In waypoint navigation, the ant must reach a sequence of waypoints in order with a similar sparse reward.

These tasks are illustrated in Fig 9, and are described in detail in Appendix A.5.We evaluate the two schemes for hierarchical reasoning with ARCs detailed in Section 4.3: commanding directly in representation space or through a k-means clustering of the representation space.

We train a high-level controller π h with TRPO which outputs as actions either a direct point in the latent space z h or a cluster index c h , from which a goal g h is decoded and passed to the goal-conditioned policy to follow for 50 timesteps.

Exact specifications and details are in Appendix A.5 and A.6.

Using a hierarchical meta-policy with ARCs performs significantly better than those using alternative representations which do not properly capture abstraction and environment dynamics FIG7 .

For multi-rooms, ARC clusters very clearly capture different rooms FIG3 , so commanding in cluster space reduces redundancy in action space, allowing for effective exploration.

ARC likely works better than commanding goals in spaces learned by other representation learning algorithms, because the learned ARC space is more structured for high-level control, which makes search and clustering simpler.

Semantically similar states like two points in the same room end up in the same ARC cluster, thus simplifying the high-level planning process for the meta-controller.

As compared to learning from scratch via TRPO and standard HRL methods such as option critic BID20 and an on-policy adaptation of HIRO BID27 , commanding in representation space enables more effective search and high-level control.

The failure of TRPO and option-critic, algorithms not using a goal-conditioned policy, emphasizes the task difficulty and indicates that a goal-conditioned policy trained on simple reaching tasks can be re-used to solve long-horizon problems.

Commanding in ARC space is better than in state space using HIRO because state space has redundancies which makes search challenging.

In this work, we introduce actionable representations for control (ARC), which capture representations of state important for decision making.

We build on the framework of goal-conditioned RL to extract state representations that emphasize features of state that are functionally relevant.

The learned state representations are implicitly aware of the dynamics, and capture meaningful distances in representation space.

ARCs are useful for tasks such as learning policies, HRL and exploration.

While ARC are learned by first training a goal-conditioned policy, learning this policy using offpolicy data is a promising direction for future work.

Interleaving the process of representation learning and learning of the goal-conditioned policy promises to scale ARC to more general tasks.

A EXPERIMENTAL DETAILS

We train a stochastic goal-conditioned policy π(·|s, g) using TRPO with an entropy regularization term, where the goal space G coincides with the state space S. In every episode, a starting state and a goal state s, g ∈ S are sampled from a uniform distribution on states, with a sparse reward given of the form below, where is task-specific, and listed in the table below.

DISPLAYFORM0 For the Sawyer environment, although this sparse reward formulation can learn a goal-conditioned policy, it is highly sample inefficient, so in practice we use a shaped reward as detailed in Appendix C. For all of the other environments, in the free space and rooms environments, the goal-conditioned policy is trained using a sparse reward.

The goal-conditioned policy is parameterized as DISPLAYFORM1 is a fully-connected neural network which takes in the state and the desired goal state as a concatenated vector, and has three hidden layers containing 150, 100, and 50 units respectively.

Σ is a learned diagonal covariance matrix, and is initially set to Σ = I. After training a goal-conditioned policy π on the specified region of interest, we collect 500 trajectories each of length 100 timesteps, where each trajectory starts at an arbitrary start state, going towards an arbitrary goal state, selected exactly as the goal-conditioned policy was trained in Appendix A.1.

This dataset was chosen to be large enough so that the collected dataset has full coverage of the entire state space.

Each of the representation learning methods evaluated is trained on this dataset, which means that each learning algorithm receives data from the full state space, and witnesses meaningful transitions between states (s t , s t+1 ).We evaluate ARCs against representations minimizing reconstruction error (VAE, slowness) and representations performing one-step prediction (predictive model, inverse dynamics).

For each representation, each component is parametrized by a neural network with ReLU activations and linear outputs, and the objective function is optimized using Adam with a learning rate of 10 −3 , holding out 20% of the trajectories as a validation set.

We perform coarse hyperparameter sweeps over various hyperparameters for all of the methods, including the dimensionality of the latent state, the size of the neural networks, and the parameters which weigh the various terms in the objectives.

The exact objective functions for each representation are detailed further in Appendix B.

We test the reward shaping capabilities of the learned representations with a set of navigation tasks on the Wheeled and Ant tasks.

A goal-conditioned policy π is trained on a n × n meter square of free space, and representations are learned (as specified above) on trajectories collected in this small region.

We then attempt to generalize to an m × m meter square (where m >> n), and consider the set of tasks of reaching an arbitrary goal in the larger region: a start state and goal state are chosen uniformly at random every episode.

The environment setup is identical to that in Appendix A.1, although with a larger region, and policy training is done the same with two distinctions.

Instead of training with the sparse reward r sparse (s, g), we train on a "shaped" surrogate reward DISPLAYFORM0 where α weights between the euclidean distance and the sparse reward terms.

Second, the policy is initialized to the parameters of the original goal-conditioned policy π which was previously trained on the small region to help exploration.

As a heuristic for the best possible shaping term, we compare with a "hand-specified" In addition to reward shaping with the various representations, we also compare to a dedicated exploration algorithm, VIME BID17 , which also uses TRPO as a base algorithm.

Understanding that different representation learning methods may learn representations with varying scales, we performed a hyperparameter sweep on α for all the representation methods.

For VIME, we performed a hyperparameter sweep on η.

The parameters used for TRPO are exactly those in Appendix A.1, albeit for 3000 iterations.

We test the ability of the representation to be used as features for a policy learning some downstream task within the Ant environment.

The downstream task is a "reach-while-avoid" task, in which the Ant requires the quadruped robot to start at the point (−1.5, −1.5) and reach the point (1.5, 1.5) while avoiding a circular region centered at the origin with radius 1 (all units in meters).

Letting d goal (s) be the distance of the agent to (1.5, 1.5) and d origin (s) to be the distance of the agent to the origin, the reward function for the task is DISPLAYFORM0 For any given representation φ(s), we train a policy which uses the feature representation as input as follows.

We use TRPO to train a stochastic policy π(a|φ(s)), which is of the form N (µ θ (φ(s)), Σ θ ).

The mean is a fully connected neural network which takes in the representation, and has two layers of size 50 each, and Σ is a learned diagonal covariance matrix initially set to Σ = I. Note that gradients do not flow through the representation, so only the policy is adapted and the representation is fixed for the entirety of the experiment.

We provide comparisons on using the learned representation to direct a goal-conditioned policy for long-horizon sequential tasks.

In particular, we consider a waypoint reaching task for the Ant, in which the agent must navigate to a sequence of 50 target locations in order: {(x 1 , y 1 ), (x 2 , y 2 ), . . . (x 50 , y 50 )}.

The agent receives as input the state of the ant and the checkpoint number that it is currently trying to reach (encoded as a one-hot vector).

When the agent gets within 0.5m of the checkpoint, it receives +1 reward, and the checkpoint is moved to the next point, making this a highly sparse reward.

Target locations are sampled uniformly at random from a 8 × 8 meter region, but are fixed for the entirety of the experiment.

We consider learning a high-level policy π h (z h |s) which outputs goals in latent space, which are then executed by a goal-conditioned policy as described in Appendix A.1.

Specifically, when the high-level policy outputs a goal in latent space z h , we use a reconstruction network ψ, which is described below, to receive a goal state g h = ψ(z h ).

The goal-conditioned policy executes for 50 timesteps according to π(a|s, g h ).

The high-level policy is trained with TRPO with the reward being equal to the sum of the rewards obtained by running the goal-conditioned policy for every meta-step.

We parametrize the high-level policy π h (z h |s) as having a Gaussian distribution in the latent space, with the mean being specified as a MLP with two layers of 50 units and Tanh activations, and the covariance as a learned diagonal matrix independent of state.

To allow the latent representation z to provide commands for the goal-conditioned policy, we separately train a reconstruction network ψ which minimizes the loss function DISPLAYFORM0 For any latent z, we can now use ψ(z) as an input into the goal-conditioned policy.

Note that an alternative method of providing commands in latent space is to train a new goal-conditioned policy π φ , which is trained to minimize the loss E s,g [D KL (π φ (·|s, φ(g)) π(·|s, g))], however to maintain abstraction between the representation and the goal-conditioned policy, we choose the former approach.

We provide comparisons on using the learned representation to direct a goal-conditioned policy in cluster space, as described in Section 4.

We consider navigation through a sequence of rooms in order in the rooms and wheeled rooms environment, as visualized in FIG2 .

A sequence of 50 checkpoints are sampled uniformly from the four rooms with the extra constraint that the same room is never repeated two checkpoints in a row (that is, each checkpoint is chosen to be any of the four rooms), and held fixed for the entirety of the experiment.

The agent is tasked with going through these rooms in order, receiving a +1 reward every time it enters the appropriate room.

The policy receives as input the state of the agent, and which number checkpoint the agent is currently trying to reach (encoded as a 50-dimensional one-hot vector).After having learned a representation φ using some set of trajectory data, as described in Appendix A.2, we run k-means clustering on states in the trajectory data to cluster latent states in the representation into k components.

We then consider learning a high-level policy π h (c h |s) which outputs a cluster between {1 . . .

k}.

Given a cluster number c h from the high-level policy, the low-level policy samples a latent state z h uniformly from the cluster, and then proceeds to command a learnt goal-conditioned policy exactly as described in Appendix A.5.Specifically, we learn a high-level policy of the form π h (c h |s) ∼ Categorical(p θ (s)) using TRPO where the probabilities for each cluster are specified by a neural network π θ which has two layers of 50 units each, with Tanh activations, and a final Softmax activation to normalize outputs into the probability simplex.

We performed hyperparameter sweeps over k -the number of clusters -for each representation method.

We provide the loss functions that are used to train each of the representations evaluated in our work.

All representations are trained on a dataset of trajectories D = {τ i } n i=1 .

We use the notation s ∼ D to denote sampling a state uniformly at random from a trajectory uniformly at random from the dataset.

We use the notation s t , s t+1 ∼ D to denote sampling a state and the state right after it according to the same uniform sampling scheme.• ARC -After precomputing D act :a matrix of actionable distances, we train a neural network φ to minimize DISPLAYFORM0 Here µ φ , σ φ , ψ θ are all neural networks, and β is a tunable hyperparameter.

The loglikelihood term is equivalent to minimizing mean squared error.• Slowness BID18 DISPLAYFORM1 Here µ φ , σ φ , ψ θ are all neural networks, and α, β are tunable hyperparameters.

The loglikelihood terms are equivalent to minimizing mean squared error.• Predictive Model BID30 DISPLAYFORM2 where φ is the learnt representation,f a model in representation space, and ψ a reconstruction network retrieving are all neural networks.

DISPLAYFORM3 Here, φ is the learnt representation, f is a learnt model in the representation space, and g is a learnt inverse dynamics model in the representation space.

β is a hyperparameter which controls how forward prediction error is balanced with inverse prediction error.

• 2D Navigation This environment consists of an agent navigating to points in an environment, either with a wall as in FIG2 or with four rooms, as in FIG2 .

The state space is 2-dimensional, consisting of the Cartesian coordinates of the agent.

The agent has acceleration control, so the action space is 2-dimensional.

Downstream tasks for this environment include reaching target locations in the environment and navigating through a sequence of 50 rooms.• Wheeled Navigation This environment consists of a car navigating to locations in an empty region, or with four rooms, as illustrated in FIG2 .

The state space is 6-dimensional, consisting of the Cartesian coordinates, heading, forward velocity, and angular velocity of the car.

The agent controls the velocity of both of its wheels, resulting in a 2-dimensional action space.

Goal-conditioned policies are trained within a 3 × 3 meter square.

Downstream tasks for wheeled navigation include reaching target locations in the environment, navigating through sequences of rooms, and navigating through sequences of waypoints.• Ant This task requires a quadrupedal ant robot navigating in free space.

The state space is 15-dimensional, consisting of the Cartesian coordinates of the ant, body orientation as a quaternion, and all the joint angles of the ant.

The agent must use torque control to control it's joints, resulting in an 8-dimensional action space.

Goal conditioned policies are trained within a 2 × 2 meter square.

Downstream tasks for the ant include reaching target locations in the environment, navigating through sequences of waypoints, and reaching target locations while avoiding other locations.• Sawyer This environment involves a Sawyer manipulator and a freely moving block on a table-top.

The state space is 6-dimensional, consisting of the Cartesian coordinates of the end-effector of the Sawyer, and the Cartesian coordinates of the block.

The Sawyer is controlled via end-effector position control with a 3-dimensional action space.

Because training a goal-conditioned policy takes an inordinate number of samples for the Sawyer environment, we instead use the following shaped reward to train the GCVF where h(s) is the position of the hand and o(s) is the position of the object r shaped (s, g) = r sparse (s, g) − h(s) − o(s) − 2 o(s) − o(g)

We perform hyperparameter tuning on three ends: one to discover appropriate parameters for each representation for each environment which are then held constant for the experimental analysis, then on the downstream applications, to choose a scaling factor for reward shaping (see Appendix A.3), and to choose the number of clusters for the hierarchical RL experiments in cluster space (see Appendix A.6).To discover appropriate parameters for each representation for the legged and wheeled locomotion environments, we evaluate representations on the downstream reward-shaping task, performing a hyperparameter sweep on latent dimension and the parameters which weigh the various terms in the representation learning objectives.

We keep the network architecture fixed for each representation and each task.

We emphasize carefully here that the ARC representation requires no parameters to tune beyond the size of the latent dimension, and we perform a hyperparameter sweep on the penalty terms to ensure that other methods aren't improperly penalized.

On the size of the latent dimension, we sweep over {2, 3, 4} for wheeled locomotion and {3, 5, 7, 9, 11} for the ant.

For the relative weighting for the penalty terms for the comparison representations (defined by β in Appendix B), we evaluate possible values β ∈ {4 −2 , 4 −1 , 1, 4 1 , 4 2 }.

These representations are then fixed and used for all the downstream applications.

For reward shaping, we tune the relative scales between the sparse reward and the shaping term, (denoted by α in Appendix A.3) over possible values α ∈ {1, 4 1 , 4 2 , 4 3 , 4 4 } for each representation on both the legged and wheeled locomotion environments.

Tuning for α is required because the representations may have different latent dimensions and different scales, and chose to perform this hyperparameter sweep instead of adding a term to the representation learning objectives to ensure uniformity in scale.

For performing k-means clustering on the HRL cluster experiments, we sweep over possible values k ∈ {4, 5, 6, 7, 8} for each representation on the room navigation tasks for 2D and wheeled navigation, but however found that most representations were robust to choice of the number of clusters.

<|TLDR|>

@highlight

Learning state representations which capture factors necessary for control

@highlight

An approach to representation learning in the context of reinforcement learning that distinguishes two stages functionally in terms of the actions that are needed to reach them.

@highlight

The paper presents a method to learn representations where proximity in euclidean distance represents states that are achieved by similar policies.
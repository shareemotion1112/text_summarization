Solving long-horizon sequential decision making tasks in environments with sparse rewards is a longstanding problem in reinforcement learning (RL) research.

Hierarchical Reinforcement Learning (HRL) has held the promise to enhance the capabilities of RL agents via operation on different levels of temporal abstraction.

Despite the success of recent works in dealing with inherent nonstationarity and sample complexity, it remains difficult to generalize to unseen environments and to transfer different layers of the policy to other agents.

In this paper, we propose a novel HRL architecture, Hierarchical Decompositional Reinforcement Learning (HiDe), which allows decomposition of the hierarchical layers into independent subtasks, yet allows for joint training of all layers in end-to-end manner.

The main insight is to combine a control policy on a lower level with an image-based planning policy on a higher level.

We evaluate our method on various complex continuous control tasks for navigation, demonstrating that generalization across environments and transfer of higher level policies can be achieved.

See videos https://sites.google.com/view/hide-rl

Reinforcement learning (RL) has been succesfully applied to sequential-decision making tasks, such as learning how to play video games in Atari (Mnih et al., 2013) , mastering the game of Go or continuous control in robotics (Lillicrap et al., 2015; Levine et al., 2015; .

However, despite the success of RL agents in learning control policies for myopic tasks, such as reaching a nearby target, they lack the ability to effectively reason over extended horizons.

In this paper, we consider continuous control tasks that require planning over long horizons in navigation environments with sparse rewards.

The task becomes particularly challenging with sparse and delayed rewards since an agent needs to infer which actions caused the reward in a domain where most samples give no signal at all.

Common techniques to mitigate the issue of sparse rewards include learning from demonstrations (Schaal, 1999; Peng et al., 2018) or using enhanced exploration strategies (Bellemare et al., 2016; Pathak et al., 2017; Andrychowicz et al., 2017) .

Hierarchical Reinforcement Learning (HRL) has been proposed in part to solve such tasks.

Typically, a sequential decision making task is split into several simpler subtasks of different temporal and functional abstraction levels (Sutton et al., 1999; Andre & Russell, 2002) .

Although the hierarchies would ideally be learned in parallel, most methods resort to curriculum learning (Frans et al., 2017; Florensa et al., 2017; Bacon et al., 2016; Vezhnevets et al., 2017) .

Recent goal-conditioned hierarchical architectures have successfully trained policies jointly via off-policy learning (Levy et al., 2019; Nachum et al., 2018; .

However, these methods often do not generalize to unseen environments as we show in Section 5.1.

We argue that this is due to a lack of true separation of planning and low-level control across the hierarchy.

In this paper, we consider two main problems, namely functional decomposition of HRL architectures in navigation-based domains and generalization of RL agents to unseen environments (figure 1).

To address these issues, we propose a novel multi-level HRL architecture that enables both functional decomposition and temporal abstraction.

We introduce a 3-level hierarchy that decouples the major roles in a complex navigation task, namely planning and low-level control.

The benefit of a modular design is twofold.

First, layers have access to only task-relevant information for a predefined task, which significantly improves the generalization ability of the overall policy.

Hence, this enables policies learned on a single task to solve randomly configured environments.

Second, Figure 1 : Navigation environments.

The red sphere indicates the goal an agent needs to reach, with the starting point at the opposite end of the maze.

The agent is trained on environment a).

To test generalization, we use the environments with b) reversed starting and goal positions, c) mirrored maze with reversed starting and goal positions and d) randomly generated mazes.

the planning and control layers are modular and thus allow for composition of cross-agent architectures.

We empirically show that the planning layer of the hierarchy can be transferred successfully to new agents.

During training we provide global environment information only to the planning layer, whereas the full internal state of the agent is only accessible by the control layer.

The actions of the top and middle layers are in the form of displacement in space.

Similarly, the goals of the middle and lowest layers are relative to the current position.

This prevents the policies from overfitting to the global position in an environment and hence encourages generalization to new environments.

In our framework (see figure 2), the planner (i.e., the highest level policy π 2 ) learns to find a trajectory leading the agent to the goal.

Specifically, we learn a value map of the environment by means of a value propagation network (Nardelli et al., 2019) .

To prevent the policy from issuing too ambitious subgoals, an attention network estimates the range of the lower level policy π 0 (i.e., the agent).

This attention mask also ensures that the planning considers the agent performance.

The action of π 2 is the position which maximizes the masked value map, which serves as goal input to the policy π 1 .

The middle layer implements an interface between the upper planner and lower control layer, which refines the coarser subgoals into shorter and reachable targets for the agent.

The middle layer is crucial in functionally decoupling the abstract task of planning (π 2 ) from agent specific continuous control.

The lowest layer learns a control policy π 0 to steer the agent to intermediate goals.

While the policies are functionally decoupled, they are trained together and must learn to cooperate.

In this work, we focus on solving long-horizon tasks with sparse rewards in complex continuous navigation domains.

We first show in a maze environment that generalization causes challenges for state-of-the-art approaches.

We then demonstrate that training with the same environment configuration (i.e., fixed start and goal positions) can generalize to randomly configured environments.

Lastly, we show the benefits of functional decomposition via transfer of individual layers between different agents.

In particular, we train our method with a simple 2DoF ball agent in a maze environment to learn the planning layer which is later used to steer a more complex agent.

The results indicate that the proposed decomposition of policy layers is effective and can generalize to unseen environments.

In summary our main contributions include:

• A novel multi-layer HRL architecture that allows functional decomposition and temporal abstraction for navigation tasks.

• This architecture enables generalization beyond training conditions and environments.

• Functional decomposition that allows transfer of individual layers across different agents.

Learning hierarchical policies has seen lasting interest (Sutton et al., 1999; Schmidhuber, 1991; Dietterich, 1999; Parr & Russell, 1998; McGovern & Barto, 2001; Dayan & Hinton, 2000) , but many approaches are limited to discrete domains or induce priors.

More recent works (Bacon et al., 2016; Vezhnevets et al., 2017; Tirumala et al., 2019; Nachum et al., 2018; Levy et al., 2019) have demonstrated HRL architectures in continuous domains.

Vezhnevets et al. (2017) introduce FeUdal Networks (FUN), which was inspired by feudal reinforcement learn-

Figure 2: Our 3-layer HRL architecture.

The planning layer π 2 receives a birds eye view of the environment and the agent's position s xy and sets an intermediate target position g 2 .

The interface layer π 2 splits this subgoal into reachable targets g 1 .

A goal-conditioned control policy π 0 learns the required motor skills to reach the target g 1 given the agent's joint information s joints .

ing (Dayan & Hinton, 2000) .

In FUN, a hierarchic decomposition is achieved via a learned state representation in latent space.

While being able to operate in continuous state space, the approach is limited to discrete action spaces.

Tirumala et al. (2019) introduce hierarchical structure into KLdivergence regularized RL using latent variables and induces semantically meaningful representations.

The separation of concerns between high-level and low-level policy is guided by information asymmetry theory.

Transfer of resulting structure can solve or speed up training of new tasks or different agents.

Nachum et al. (2018) present HIRO, an off-policy HRL method with two levels of hierarchy.

The non-stationary signal of the upper policy is mitigated via off-policy corrections, while the lower control policy benefits from densely shaped rewards.

Nachum et al. (2019) propose an extension of HIRO, which we call HIRO-LR, by learning a representation space from environment images, replacing the state and subgoal space with neural representations.

While HIRO-LR can generalize to a flipped environment, it needs to be retrained, as only the learned space representation generalizes.

Contrarily, HiDe generalizes without retraining.

Levy et al. (2019) introduce Hierarchical Actor-Critic (HAC), an approach that can jointly learn multiple policies in parallel.

The policies are trained in sparse reward environments via different hindsight techniques.

HAC, HIRO and HIRO-LR consist of a set of nested policies where the goal of a policy is provided by the top layer.

In this setting the goal and state space of the lower policy is identical to the action space of the upper policy.

This necessitates sharing of the state space across layers.

Overcoming this limitation, we introduce a modular design to decouple the functionality of individual layers.

This allows us to define different state, action and goal spaces for each layer.

Global information about the environment is only available to the planning layer, while lower levels only receive information that is specific to the respective layer.

Furthermore, HAC and HIRO have a state space that includes the agent's position and the goal position, while (Nachum et al., 2019) and our method both have access to global information in the form of a top-down view image.

In model-based reinforcement learning much attention has been given to learning of a dynamics model of the environment and subsequent planning (Sutton, 1990; Sutton et al., 2012; Wang et al., 2019) .

Eysenbach et al. (2019) propose a planning method that performs a graph search over the replay buffer.

However, they require to spawn the agent at different locations in the environment and let it learn a distance function in order to build the search graph.

Unlike model-based RL, we do not learn state transitions explicitly.

Instead, we learn a spatial value map from collected rewards.

Recently, differentiable planning modules that can be trained via model-free reinforcement learning have been proposed (Tamar et al., 2016; Oh et al., 2017; Nardelli et al., 2019; Srinivas et al., 2018) .

Tamar et al. (2016) establish a connection between convolutional neural networks and Value Iteration (Bertsekas, 2000) .

They propose Value Iteration Networks (VIN), an approach where modelfree RL policies are additionally conditioned on a fully differrentiable planning module.

MVProp (Nardelli et al., 2019) extends this work by making it more parameter-efficient and generalizable.

The planning layer in our approach is based on MVProp, however contrary to prior work we do not rely on a fixed neighborhood mask to sequentially provide actions in its vicinity in order to reach a goal.

Instead we propose to learn an attention mask which is used to generate intermediate goals for the underlying layers.

Gupta et al. (2017) learn a map of indoor spaces and planning on it using a multi-scale VIN.

In their setting, the policy is learned from expert actions using supervised learning.

Moreover, the robot operates on discrete set of high level macro actions.

Srinivas et al. (2018) propose Universal Planning Networks (UPN), which learn how to plan an optimal action trajectory via a latent space representation.

In contrast to our approach, the method relies on expert demonstrations and transfer to harder tasks can only be achieved after retraining.

We model a Markov Decision Process (MDP) augmented with a set of goals G. We define the MDP as a tuple M = {S, A, G, R, T , ρ 0 , γ}, where S and A are set of states and actions, respectively, R t = r(s t , a t , g t ) a reward function, γ a discount factor ∈ [0, 1], T = p(s t+1 |s t , a t ) the transition dynamics of the environment and ρ 0 = p(s 1 ) the initial state distribution, with s t ∈ S and a t ∈ A. Each episode is initialized with a goal g ∈ G and an initial state is sampled from ρ 0 .

We aim to find a policy π : S × G → A, which maximizes the expected return.

We train our policies by using an actor-critic framework where the goal augmented action-value function is defined as:

The Q-function (critic) and the policy π (actor) are approximated by using neural networks with parameters θ Q and θ π .

The objective for θ Q minimizes the loss:

, where

The policy parameters θ π are trained to maximize the Q-value:

To address the issue of sparse rewards, we utilize Hindsight Experience Replay (HER) (Andrychowicz et al., 2017), a technique to improve sample-efficiency in training goal-conditioned environments.

The insight is that the desired goals of transitions stored in the replay buffer can be relabeled as states that were achieved in hindsight.

Such data augmentation allows learning from failed episodes, which may generalize enough to solve the intended goal.

In HAC, Levy et al. (2019) apply two hindsight techniques to address the challenges introduced by the non-stationary nature of hierarchical policies and the environments with sparse rewards.

In order to train a policy π i , optimal behavior of the lower-level policy is simulated by hindsight action transitions.

More specifically, the action a i is replaced with a state s i−1 that is actually achieved by the lower-level policy π i−1 .

Identically to HER, hindsight goal transitions replace the subgoal g i−1 with an achieved state s i−1 , which consequently assigns a reward to the lower-level policy π i−1 for achieving the virtual subgoal.

Additionally, a third technique called subgoal testing is proposed.

The incentive of subgoal testing is to help a higher-level policy understand the current capability of a lower-level policy and to learn Q-values for subgoal actions that are out of reach.

We find both techniques effective and apply them to our model during training.

Tamar et al. (2016) propose differentiable value iteration networks (VIN) for path planning and navigation problems.

Nardelli et al. (2019) propose value propagation networks (MVProp) with better sample efficiency and generalization behavior.

MVProp creates reward-and propagation maps covering the environment.

The reward map highlights the goal location and the propagation map determines the propagation factor of values through a particular location.

The reward map is an imager i,j of the same size as the environment image I, wherer i,j = 0 if the pixel (i, j) overlaps with the goal position and −1 otherwise.

The value map V is calculated by unrolling max-pooling operations in a neighborhood N for k steps as follows:

Figure 3: Planner layer π 2 (s xy , G, I).

Given the top-view environment image I and goal G on the map, the maximum value propagation network (MVProp) calculates a value map V .

By using the agent's current position s xy , we estimate an attention mask M restricting the global value map V to a local and reachable subgoal mapV .

The policy π 2 selects the coordinates with maximum value and assigns the lower policy π 1 with a sugboal that is relative to the agent's current position.

The action (i.e., the target position) is selected to be the pixels (i , j ) maximizing the value in a predefined 3x3 neighborhood N (i 0 , j 0 ) of the agent's current position (i 0 , j 0 ):

Note that the window N (i 0 , j 0 ) is determined by the discrete, pixel-wise actions.

We introduce a novel hierarchical architecture, HiDe, allowing for an explicit functional decomposition across layers.

Similar to HAC (Levy et al., 2019) , our method achieves temporal abstractions via nested policies.

Moreover, our architecture enables functional decomposition explicitly.

This is achieved by nesting i) an abstract planning layer, followed ii) by a local planer to iii) guide a control component.

Crucially, only the top layer receives global information and is responsible for planning a trajectory towards a goal.

The lowest layer learns a control policy for agent locomotion.

The middle layer converts the planning layer output into subgoals for the control layer.

Achieving functional decoupling across layers crucially depends on reducing the state in each layer to the information that is relevant to its specific task.

This design significantly improves generalization (see Section 5).

The highest layer of a hierarchical architecture is expected to learn high-level actions over a longer horizon, which define a coarse trajectory in navigation-based tasks.

In the related work (Levy et al., 2019; Nachum et al., 2018; , the planning layer, learning an implicit value function, shares the same architecture as lower layers.

Since the task is learned for a specific environment, limits to generalization are inherent to this design choice.

In contrast, we introduce a planning specific layer consisting of several components to learn the map and to find a feasible path to the goal.

The planning layer is illustrated in figure 3 .

We utilize a value propagation network (MVProp) (Nardelli et al., 2019) to learn an explicit value map which projects the collected rewards onto the environment image.

Given a top-down image of the environment, a convolutional network determines the per pixel flow probability p i,j .

For example, the probability value of a pixel corresponding to a wall should be 0 and that for free passages 1 respectively.

Nardelli et al. (2019) use a predefined 3 × 3 neighborhood of the agent's current position and pass the location of the maximum value in this neighbourhood as goal position to the agent (equation 5).

We augment a MVProp network with an attention model which learns to define the neighborhood dynamically and adaptively.

Given the value map V and the agent's current position s xy , we estimate how far the agent can go, modeled by a 2D Gaussian.

More specifically, we predict a full covariance matrix Σ with the agent's global position s xy as mean.

We later build a 2D mask M of the same size as the environment image I by using the likelihood function:

Figure 4: A visual comparison of (left) our dynamic attention window with a (right) fixed neighborhood.

The green dot corresponds to the selected subgoal in this case.

Notice how our window is shaped so that it avoids the wall and induces a further subgoal.

Intuitively, the mask defines the density for the agent's success rate.

Our planner policy selects an action (i.e., subgoal) that maximizes the masked value map as follows:

wherev i,j corresponds to the value at pixel (i, j) on the masked value mapV .

Note that the subgoal selected by the planning layer g 2 is relative to the agent's current position s xy , which improves generalization performance of our model.

The benefits of having an attention model are twofold.

First, the planning layer considers the agent dynamics in assigning subgoals which may lead to fine-or coarse-grained subgoals depending on the underlying agent's performance.

Second, the Gaussian window allows us to define a dynamic set of actions for the planner policy π 2 , which is essential to find a trajectory of subgoals on the map.

While the action space includes all pixels of the value map V , it is limited to the subset of only reachable pixels by the Gaussian mask M .

Qualitatively we find this leads to better obstacle avoidance behaviour such as the corners and walls shown in figure 4.

Since our planner layer operates in a discrete action space (i.e., pixels), the resolution of the projected maze image defines the minimum amount of displacement for the agent, affecting maneuverability.

This could be tackled by using a soft-argmax (Chapelle & Wu, 2010) to select the subgoal pixel, allowing to choose real-valued actions and providing in-variance to image resolution.

In our experiments we see no difference in terms of the final performance.

However, since the former setting allows for the use of DQN (Mnih et al., 2013) instead of DDPG (Silver et al., 2014) , we prefer the discrete action space for simplicity and faster convergence.

The middle layer in our hierarchy interfaces the high-level planning with low-level control by introducing an additional level of temporal abstraction.

The planner's longer-term goals are further split into a number of shorter-term targets.

Such refinement policy provides the lower-level control layer with reachable targets, which in return yields easier rewards and hence accelerated learning.

The interface layer policy is the only layer that is not directly interacting with the environment.

More specifically, the policy π 1 only receives the subgoal g 2 from the upper layer π 2 and chooses an action (i.e. subgoal g 1 ) for the lower-level locomotion layer π 0 .

Note that all the goal, state and action spaces of the policy π 1 are in 2D space.

Contrary to Levy et al. (2019) , we use subgoals that are relative to the agent's position s xy .

This helps to generalize and learn better.

The lowest layer learns a goal-conditioned control policy.

Due to our explicit functional decomposition, it is the only layer with access to the agent's internal state s joints including joint positions and velocities.

Whereas the higher layers only have access to the agent's position.

In a navigation task, the agent has to learn locomotion to reach the goal position.

Similar to HAC, we use hindsight goal transition techniques so that the control policy receives rewards even in failure cases.

All policies in our hierarchy are jointly-trained.

We use the DDPG algorithm (Lillicrap et al., 2015) with the goal-augmented actor-critic framework (equation 2-3) for the control and interface layers, and DQN (Mnih et al., 2013) for the planning layer (see section 4.1).

We evaluate our method on a series of simulated continuous control tasks in navigation-based environments 1 .

All environments are simulated in the MuJoCo physics engine (Todorov et al., 2012) .

Experiment and implementation details are provided in the Appendix B. First, in section 5.1, we compare to various baseline methods.

In section 5.2, we move to a new maze with a more complex design in order to show our model's generalization capabilities.

Section 5.3 demonstrates that our approach indeed leads to functional decomposition by composing new agents via combining the planning layer of one agent with the locomotion layer of another.

Finally, in section 5.4 we provide an ablation study for our design choices.

We introduce the following task configurations: Maze Forward: the training environment in all experiments.

The task is to reach a goal from a fixed pre-determined start position.

Maze Backward: the training maze layout with swapped start and goal positions.

Maze Flipped: a mirrored version of the training environment.

Maze Random: a set of randomly generated mazes with random start and goal positions.

In our experiments, we always train in the Maze Forward environment.

The reward signal during training is constantly -1, unless the agent reaches the given goal (except for HIRO and HIRO-LR, see section 5.1).

We test the agents on the above tasks with fixed starting and fixed goal position.

For more details about the environments, we refer to Appendix A. We intend to answer the following two questions: 1) Can our method generalize to unseen test environments?

2) Is it possible to transfer the planning layer policies between agents?

We compare our method to state-of-the-art approaches including HIRO (Nachum et al., 2019) , HIRO-LR (Nachum et al., 2019) , HAC (Levy et al., 2019 ) and a modified version of HAC called Rel-HAC in a simple Maze Forward environment as shown in figure 6 .

For a fair comparison, we made a number of improvements to the HAC and HIRO implementations.

For HAC, we introduced target networks and used the hindsight experience replay technique with the future strategy (Andrychowicz et al., 2017) .

In our experiments we observed that oscillations around the goal kept HIRO agents from finishing the task, which was solved via doubling the distance-threshold of success.

HIRO-LR is the closest to our method, as it also receives a top-down view image of the environment.

Note that both HIRO and HIRO-LR have access to dense negative distance reward, which is an advantage over HAC and HiDe that only receive a reward when finishing the task.

We train a modified HAC model, dubbed RelHAC, to asses our planning layer.

RelHAC has the same lowest and middle layers as HiDe, whereas the top layer has the same structure as the middle layer, therefore missing an effective planner.

Preliminary experiments using fixed start and fixed goal positions during training for HAC, HIRO and HIRO-LR yielded 0 success rates in all cases.

Therefore, the baseline models are trained by using fixed start and random goal positions, allowing it to receive a reward signal without having to reach the intended goal at the other end of the maze.

Contrarily, HiDe is trained with fixed start and fixed goal positions, whereas HiDe-R represents HiDe under the same conditions as the baseline methods.

All models learned this task successfully as shown in figure 5 and table 1 (Forward column).

HIRO demonstrates slightly better convergence and final performance, which can be attributed to the fact that it is trained with dense rewards.

RelHAC performs worse than HAC due to the pruned state space of each layer and due to the lack of an effective planner.

HIRO-LR takes longer to converge because it has to learn a latent goal space representation.

Table 1 summarizes the models' generalization abilities to the unseen Maze Backward and Maze Flipped environments (see figure 6 ).

While HIRO, HIRO-LR and HAC manage to solve the training environment (Maze Forward) with success rates between 99% and 82%, they suffer from overfiting to the training environment, indicated by the 0% success rates in the unseen test environments.

Contrarily, our method is able to achieve 54% and 69% success rates in this generalization task.

As expected, training our model with random goal positions (i.e., HiDe-R) yields a more robust model outperforming vanilla HiDe.

In subsequent experiments, we only report the results for our method, as our experiments have shown that the baseline methods cannot solve the training task for more complex environments.

In this experiment, we train an ant and a ball agent (see Appendix A.1) in the Maze Forward task with a more complex environment layout (cf. figure 1), while we keep both the start and goal positions intact.

We then evaluate this model in 4 different tasks (see section 5).

Table 2 reports success rates of both agents in this complex task.

Our model successfully transfers its navigation skills to unseen environments.

The performance for the Maze Backward and Maze Flipped tasks is similar to the results shown in section 5.1 despite the increased difficulty.

Since the randomly generated mazes are typically easier, our model shows similar or better performance.

To demonstrate that the layers in our architecture indeed learn separate sub-tasks we transfer individual layers across different agents.

We first train an agent without our planning layer, i.e., with RelHAC.

We then replace the top layer of this agent with the planning layer from the models trained in section 5.2.

Additionally, we train a humanoid agent and show as a proof of concept that transfer to a very complex agent can be achieved.

We carry out two sets of experiments.

First, we transfer the ant model's planning layer to the simpler 2 DoF ball agent.

As indicated in Table 3 , the performance of the ball with the ant's planning layer matches the results in Table 2 .

The ball agent's success rate increases for random (from 96% to 100%) and forward (96% to 97%) maze tasks whereas it decreases slightly in the backward (from 100% to 90%) and flipped (from 99% to 88%) configurations.

HiDe-A 0 ± 0 0 ± 0 0 ± 0 HiDe-AR 95 ± 1 52 ± 33 34 ± 45 Table 4 : Success rates in the simple maze.

HiDe-A is our method with absolute subgoals.

HiDe-AR has absolute goals and samples random goals during training.

HiDe-A 0 ± 0 0 ± 0 0 ± 0 0 ± 0 HiDe-AR 0 ± 0 0 ± 0 0 ± 0 0 ± 0 HiDe-NI 10 ± 5 46 ± 16 0 ± 0 3 ± 4 Table 5 : Success rates of achieving a goal in the complex maze environment.

HiDe-A and HiDe-AR as in Table 4 .

HiDe-NI is our method without the inferface layer.

Second, we attach the ball agent's planning layer to the more complex ant agent.

Our new compositional agent performs marginally better or worse in the Flipped, Random and Backward tasks.

Please note that this experiment is an example of a case where the environment is first learned with a fast and easy-to-train agent (i.e., ball) and then utilized by a more complex agent.

We hereby show that transfer of layers between agents is possible and therefore find our hypothesis to be valid.

Moreover, an estimate indicates that the training is roughly 3 -4 times faster, since the complex agent does not have to learn the planning layer.

To demonstrate our method's transfer capabilities, we train a humanoid agent (17 DoF) in an empty environment with shaped rewards.

We then use the planning and interface layer from a ball agent and connect it as is with the locomotion layer of the trained humanoid 2 .

To support the claim that our architectural design choices lead to better generalization and functional decomposition, we compare empirical results for different variants of our method with an ant agent.

First, we compare the performance of relative and the absolute positions for both experiment 1 and experiment 2.

For this reason, we train HiDe-A and HiDe-AR, the corresponding variants of HiDe and HiDe-R that use absolute positions.

Unlike for relative positions, the policy needs to learn all values within the range of the environment dimensions.

Second, we compare HiDe against a variant of HiDe without the interface layer called HiDe-NI.

The results for experiment 1 are in Table 4 .

HiDe-A does not manage to solve the task at all, similar to HAC and HIRO without random goal sampling.

HiDe-AR succeeds in solving the Forward task.

However, it generalizes worse than both Hide and HiDe-R in the Backward and Flipped task.

Both HiDe-A and HiDe-AR fail to solve the complex maze for experiment 2 as shown in the Table 5 .

These results indicate that 1) relative positions improve performance and are an important aspect of our method to achieve generalization to other environments and 2) random goal position sampling can help agents, but may not be available depending on the environment.

As seen in Table 5 , the variant of HiDe without interface layer (HiDe-NI) performs worse than HiDe (cf.

Table 2 ) in all experiments.

Thus, the interface layer is an important part of our architecture.

We also run an ablation study for HiDe with a fixed window size.

More specifically, we train and evaluate an ant agent on window sizes 3×3, 5×5, and 9×9.

The results are included in Tables 12,13 , and 14.

The learned attention window (HiDe) achieves better or comparable performance.

In all cases, HiDe generalizes better in the Backward complex maze.

Moreover, the learned attention eliminates the need for tuning the window size hyperparameter per agent and environment.

In this paper, we introduce a novel HRL architecture that can solve complex navigation tasks in 3D-based maze environments.

The architecture consists of a planning layer which learns an explicit value map and is connected with a subgoal refinement layer and a low-level control layer.

The framework can be trained end-to-end.

While training with a fixed starting and goal position, our method is able to generalize to previously unseen settings and environments.

Furthermore, we demonstrate that transfer of planners between different agents can be achieved, enabling us to transfer a planner trained with a simplistic agent such as a ball to a more complex agent such as an ant or humanoid.

In future work, we want to consider integration of a more general planner that is not restricted to navigation-based environments.

We build on the Mujoco (Todorov et al., 2012) environments used in Nachum et al. (2018) .

All environments use dt = 0.02.

Each episode in experiment 1 is terminated after 500 steps and after 800 steps in the rest of the experiments or after the goal in reached.

All rewards are sparse as in Levy et al. (2019) , i.e., 0 for reaching the goal and −1 otherwise.

We consider goal reached if |s − g| max < 1.

Since HIRO sets the goals in the far distance to encourage the lower layer to move to the goal faster, it can't stay exactly at the target position.

Moreover, they do not terminate the episode after the goal is reached.

Thus for HIRO, we consider a goal reached if |s − g| 2 < 2.5.

Our Ant agent is equivalent to the one in Levy et al. (2019) .

In other words, the Ant from Rllab (Duan et al., 2016) with gear power of 16 instead of 150 and 10 frame skip instead of 5.

Our Ball agent is the PointMass agent from DM Control Suite (Tassa et al., 2018) .

We made the change the joints so that the ball rolls instead of sliding.

Furthermore, we resize the motor gear and the ball itself to match the maze size.

All mazes are modelled by immovable blocks of size 4 × 4 × 4.

Nachum et al. (2018) uses blocks of 8 × 8 × 8.

The environment shapes are clearly depicted in figure 1.

For the randomly generated maze, we sample each block with probability being empty p = 0.8,start and goal positions are also sampled randomly at uniform.

Mazes where start and goal positions are adjacent or where goal is not reachable are discarded.

For the evaluation, we generated 500 of such environments and reused them (one per episode) for all experiments.

Our PyTorch (Paszke et al., 2017) implementation will be available at the project website.

For both HIRO and HAC we used the original authors implementation 45 .

In HIRO, we set the goal success radius for evaluation as described above.

We ran the hiro xy variant, which uses only position coordinates for subgoal instead of all joint positions, to have a fair comparison with our method.

To improve the performance of HAC in experiment one, we modified their Hindsight Experience Replay (Andrychowicz et al., 2017) implementation so that they use FUTURE strategy.

More importantly, we also added target networks to both the actor and critic to improve the performance.

For evaluation, we trained 5 seeds each for 2.5M steps on the Forward environment with continuous evaluation (every 100 episodes for 100 episodes).

After training, we selected the best checkpoint based on the continuous evaluation of each seed.

Then, we tested the learned policies for 500 episodes and reported the average success rate.

Although the agent and goal positions are fixed, the initial joint positions and velocities are sampled from uniform distribution as standard in OpenAI Gym environments.

Therefore, the tables in the paper contain means and standard deviation across 5 seeds.

B.3 NETWORK STRUCTURE B.3.1 PLANNING LAYER Input images for the planning layer were binnarized in the following way: each pixel corresponds to one block (0 if it was a wall or 1 if it was a corridor).

In our planning layer, we process the input image of size 32x32 (20x20 for experiment 1) via two convolutional layers with 3 × 3 kernels.

Both layers have only 1 input and output channel and are padded so that the output size is the same as the input size.

We propagate the value through the value map as in Nardelli et al. (2019) K = 35 times using a 3 × 3 max pooling layer.

Finally, the value map and agent position image (a black image with a dot at agent position) is processed by 3 convolutions with 32 output channels and 3 × 3 filter window interleaved by 2 × 2 max pool with ReLU activation functions and zero padding.

The final result is flatten and processed by two fully connected layers with 64 neurons each producing three outputs: σ 1 , σ 2 , ρ with softplus, softplus and tanh activation functions respectively.

The final covariance matrix Σ is given by

, so that the matrix is always symmetric and positive definite.

For numerical reasons, we multiply by binnarized kernel mask instead of the actual Gaussian densities.

We set values higher than mean to 1 and others to zeros.

In practice, we use this line: kernel = t.where(kernel >= kernel.mean(dim=[1,2], keepdim=True), t.ones_like(kernel), t.zeros_like(kernel))

We use the same network architecture for the middle and lower layer as proposed by Levy et al. (2019) , i.e. we use 3 times fully connected layer with ReLU activation function.

The locomation layer is activated with tanh, which is then scaled to the action range.

• Discount γ = 0.98 for all agents.

• Adam optimizer.

Learning rate 0.001 for all actors and critics.

• Soft updates using moving average; τ = 0.05 for all controllers.

• Replay buffer size was designed to store 500 episodes, similarly as in Levy et al. (2019) • We performed 40 actor and critic learning updates after each epoch on each layer, after the replay buffer contained at least 256 transitions.

• Batch size 1024.

• No gradient clipping

• Rewards 0 and -1 without any normalization.

• Subgoal testing (Levy et al., 2019) only for the middle layer.

• Maximum subgoal horizon H = 10 for all 3 layers algorithms and H = 25 for ablations without the inferace layer.

See psuedocode 1.

• Observations also were not normalized.

• 2 HER transitions per transition using the FUTURE strategy (Andrychowicz et al., 2017) .

• Exploration noise: 0.05, 0.01 and 0.1 for the planning, middle and locomotion layer respectively.

In this section, we present all results collected for this paper including individual runs.

@highlight

Learning Functionally Decomposed Hierarchies for Continuous Navigation Tasks
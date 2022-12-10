Imitation learning provides an appealing framework for autonomous control: in many tasks, demonstrations of preferred behavior can be readily obtained from human experts, removing the need for costly and potentially dangerous online data collection in the real world.

However, policies learned with imitation learning have limited flexibility to accommodate varied goals at test time.

Model-based reinforcement learning (MBRL) offers considerably more flexibility, since a predictive model learned from data can be used to achieve various goals at test time.

However, MBRL suffers from two shortcomings.

First, the model does not help to choose desired or safe outcomes -- its dynamics estimate only what is possible, not what is preferred.

Second, MBRL typically requires additional online data collection to ensure that the model is accurate in those situations that are actually encountered when attempting to achieve test time goals.

Collecting this data with a partially trained model can be dangerous and time-consuming.

In this paper, we aim to combine the benefits of imitation learning and MBRL, and propose imitative models: probabilistic predictive models able to plan expert-like trajectories to achieve arbitrary goals.

We find this method substantially outperforms both direct imitation and MBRL in a simulated autonomous driving task, and can be learned efficiently from a fixed set of expert demonstrations without additional online data collection.

We also show our model can flexibly incorporate user-supplied costs at test-time, can plan to sequences of goals, and can even perform well with imprecise goals, including goals on the wrong side of the road.

Reinforcement learning (RL) algorithms offer the promise of automatically learning behaviors from raw sensory inputs with minimal engineering.

However, RL generally requires online learning: the agent must collect more data with its latest strategy, use this data to update a model, and repeat.

While this is natural in some settings, deploying a partially-trained policy on a real-world autonomous system, such as a car or robot, can be dangerous.

In these settings the behavior must be learned offline, usually with expert demonstrations.

How can we incorporate such demonstrations into a flexible robotic system, like an autonomous car?

One option is imitation learning (IL), which can learn policies that stay near the expert's distribution.

Another option is model-based RL (MBRL) BID8 BID2 , which can use the data to fit a dynamics model, and can in principle be used with planning algorithms to achieve any user-specified goal at test time.

However, in practice, model-based and model-free RL algorithms are vulnerable to distributional drift BID32 BID24 : when acting according to the learned model or policy, the agent visits states different from those seen during training, and in those it is unlikely to determine an effective course of action.

This is especially problematic when the data intentionally excludes adverse events, such as crashes.

A model ignorant to the possibility of a crash cannot know how to prevent it.

Therefore, MBRL algorithms usually require online collection and training BID6 BID12 .

Imitation learning algorithms use expert demonstration data and, despite similar drift shortcomings BID26 , can sometimes learn effective policies without additional online data collection BID35 .

However, standard IL offers little task flexibility since it only predicts low-level behavior.

While several works augmented IL with goal conditioning BID4 BID1 , these goals must be specified in advance during training, and are typically simple (e.g., turning left or right).Figure 1: We apply our approach to navigation in CARLA BID5 .

Columns 1,2: Images depicting the current scene.

The overhead image depicts a 50 m 2 area.

Column 3: LIDAR input and goals are provided to our deep imitative trajectory model, and plans to the goals are computed under the model's likelihood objective, and colored according to their ranking under the objective, with red indicating the best plan.

The red square indicates the chosen high-level goal, and the yellow cross indicates a point along our plan used as a setpoint for a PID controller.

The LIDAR map is 100 m 2 , and each goal is ≥20 m away from the vehicle.

Column 4: Our model can incorporate arbitrary test-time costs, and use them to adjust its planning objective and plan ranking.

Figure 2: A brief taxonomy of learning-based control methods.

In our scenario, we avoid online data collection, specifically from the policy we seek to imitate.

We structure our imitation learner with a model to make it flexible to new tasks at test time.

We compare against other offline approaches (front face).The goal in our work is to devise a new algorithm that combines the advantages of IL and MBRL, affording both the flexibility to achieve new user-specified goals at test time and the ability to learn entirely from offline data.

By learning a deep probabilistic predictive model from expert-provided data, we capture the distribution of expert behaviors without using manually designed reward functions.

To plan to a goal, our method infers the most probable expert state trajectory, conditioned on the current position and reaching the goal.

By incorporating a model-based representation, our method can easily plan to previously unseen user-specified goals while respecting rules of the road, and can be flexibly repurposed to perform a wide range of test-time tasks without any additional training.

Inference with this model resembles trajectory optimization in model-based reinforcement learning, and learning this model resembles imitation learning.

Our method's relationship to other work is illustrated in Fig. 2 .

We demonstrate our method on a simulated autonomous driving task (see FIG0 .

A high-level route planner provides navigational goals, which our model uses to automatically generate plans that obey the rules of the road, inferred entirely from data.

In contrast to IL, our method produces an interpretable distribution over trajectories and can follow a variety of goals without additional training.

In contrast to MBRL, our method generates human-like behaviors without additional data collection or learning.

In our experiments, our approach substantially outperforms both MBRL and IL: it can efficiently learn near-perfect driving through the static-world CARLA simulator from just 7,000 trajectories obtained from 19 hours of driving.

We also show that our model can flexibly incorporate and achieve goals not seen during training, and is robust to errors in the high-level navigation system, even when the high-level goals are on the wrong side of the road.

Videos of our results are available.

To learn robot dynamics that are not only possible, but preferred, we construct a model of expert behavior.

We fit a probabilistic model of trajectories, q, to samples of expert trajectories drawn from an unknown distribution p. A probabilistic model is necessary because expert behavior is often stochastic and multimodal: e.g., choosing to turn either left or right at an intersection are both common decisions.

Because an expert's behavior depends on their perception, we condition our model, q, on observations φ.

In our application, φ includes LIDAR features χ ∈ R H×W ×C and a small window of previous positions s −τ :0 = {s −τ , . . .

, s 0 }, such that φ = {χ, s −τ :0 }.By training q(s 1:T |φ) to forecast expert trajectories with high likelihood, we model the sceneconditioned expert dynamics, which can score trajectories by how likely they are to come from the expert.

At test time, q(s 1:T |φ) serves as a learned prior over the set of undirected expert trajectories.

To execute samples from this distribution is to imitate an expert driver in an undirected fashion.

We first describe how we use the generic form of this model to plan, and then discuss our particular implementation in Section 2.2.

Besides simply imitating the expert demonstrations, we wish to direct our agent to desired goals at test time, and have the agent reason automatically about the mid-level details necessary to achieve these goals.

In general, we can define a driving task by a set of goal variables G. We will instantiate examples of G concretely after the generic goal planning derivation.

The probability of a plan conditioned on the goal G is given as posterior distribution p(s 1:T |G, φ).

Planning a trajectory under this posterior corresponds to MAP inference with prior q(s 1:T |φ) and likelihood p(G|s 1:T , φ).

We briefly derive the MAP inference result starting from the posterior maximization objective, which uses the learned Imitative Model to generate plans that achieve abstract goals: (1) DISPLAYFORM0 Waypoint planning: One example of a concrete inference task is to plan towards a specific goal location, or waypoint.

We can achieve this task by using a tightly-distributed goal likelihood function centered at the user's desired final state.

This effectively treats a desired goal location, g T , as if it were a noisy observation of a future state, with likelihood p(G|s 1:T , φ) = N (g T |s T , I).

The resulting inference corresponds to planning the trajectory s 1:T to a likely point under the distribution N (g T |s T , I).

We can also plan to successive states with DISPLAYFORM1 ) if the user (or program) wishes to specify the desired end velocity or acceleration when reached the final goal g T location FIG3 .

Alternatively, a route planner may propose a set of waypoints with the intention that the robot should reach any one of them.

This is possible using a Gaussian mixture likelihood and can be useful if some of those waypoints along a route are inadvertently located at obstacles or potholes (Fig. 4) .Waypoint planning leverages the advantage of conditional imitation learning: a user or program can communicate where they desire the agent to go without knowing the best and safest actions.

The planning-as-inference procedure produces paths similar to how an expert would acted to reach the given goal.

In contrast to black-box, model-free conditional imitation learning that regresses controls, our method produces an explicit plan, accompanied by an explicit score of the plan's quality.

This provides both interpretability and an estimate of the feasibility of the plan.

Costed planning: If the user desires more control over the plan, our model has the additional flexibility to accept arbitrary user-specified costs c at test time.

For example, we may have updated knowledge of new hazards at test time, such as a given map of potholes (Fig. 4) or a predicted cost map.

Given costs c(s i |φ), this can be treated by including an optimality variable C in G, where BID33 BID11 .

The goal log-likelihood is log p({g T , C = 1}|s 1:T , φ) = log N (g T |s T , I) + Figure 4: Imitative planning to goals subject to a cost at test time.

The cost bumps corresponds to simulated "potholes," which the imitative planner is tasked with avoiding.

The imitative planner generates and prefers routes that curve around the potholes, stay on the road, and respect intersections.

Demonstrations of this behavior were never observed by our model.

DISPLAYFORM2

The primary structural requirement of an Imitative Model is the ability to compute q(s 1:T |φ).

The ability to also compute gradients ∇ s 1:T q(s 1:T |φ) enables gradient-based optimization for planning.

Finally, the quality and efficiency of learning are important.

One deep generative model for Imitation Learning is the Reparameterized Pushforward Policy (R2P2) BID23 ).

R2P2's use of pushforward distributions BID16 , employed in other invertible generative models BID21 BID3 allows it to efficiently minimize both false positives and false negatives (type I and type II errors) BID18 .

Optimization of KL(p, q), which penalizes mode loss (false negatives), is straightforward with R2P2, as it can evaluate q(s 1:T |φ).

Here, p is the sampleable, but unknown, distribution of expert behavior.

Reducing false positives corresponds to minimizing KL(q, p), which penalizes q heavily for generating bad DISPLAYFORM0 DISPLAYFORM1 end while 6: return s 1:T samples under p. As p is unknown, R2P2 first uses a spatial cost modelp to approximate p, which we can also use as c in our planner.

The learning objective is KL(p, q) + βKL(q,p).Figure 5: Architecture of m t and σ t , modified from BID23 with permission.

In R2P2, q(s 1:T |φ) is induced by an invertible, differentiable function: f (z; φ) : R 2T → R 2T , which warps latent samples from a base distribution z ∼ q 0 = N (0, I 2T ×2T ) to the output space over s 1:T .

f embeds the evolution of learned discrete-time stochastic dynamics; each state is given by: DISPLAYFORM2 The m t ∈ R 2 and σ t ∈ R 2×2 are computed by expressive, nonlinear neural networks that observe previous states and LIDAR input.

The resulting trajectory distribution is complex and multimodal.

We modified the RNN method described by BID23 and used LIDAR features χ = R 200×200×2 , with χ ij representing a 2-bin histogram of points below and above the ground in 0.5 m 2 cells (Fig 5) .

We used T = 40 trajectories at 5Hz (8 seconds of prediction or planning), τ = 19.

At test time, we use three layers of spatial abstractions to plan to a faraway destination, common to model-based (not end-to-end) autonomous vehicle setups: coarse route planning over a road map, path planning within the observable space, and feedback control to follow the planned path BID19 BID29 .

For instance, a route planner based on a conventional GPSbased navigation system might output waypoints at a resolution of 20 meters -roughly indicating the direction of travel, but not accounting for the rules of the road or obstacles.

The waypoints are treated as goals and passed to the Imitative Planner (Algorithm 1), which then generates a path chosen according to the optimization in Eq. 1.

These plans are fed to a low-level controller (we use a PID-controller) that follows the plan.

In Fig. 6 we illustrate how we use our model in our application.

Figure 6: Illustration of our method applied to autonomous driving.

Our method trains an Imitative Model from a dataset of expert examples.

After training, the model is repurposed as an Imitative Planner.

At test time, a route planner provides waypoints to the Imitative Planner, which computes expert-like paths to each goal.

The best plan chosen according to the planning objective, and provided to a low-level PID-controller in order to produce steering and throttle actions.

Previous work has explored conditional IL for autonomous driving.

Two model-free approaches were proposed by BID1 , to map images to actions.

The first uses three network "heads", each head only trained on an expert's left/straight/right turn maneuvers.

The robot is directed by a route planner that chooses the desired head.

Their second method input the goal location into the network, however, this did not perform as well.

While model-free conditional IL can be effective given a discrete set of user directives, our model-based conditional IL has several advantages.

Our model has flexibility to handle more complex directives post training, e.g. avoiding hazardous potholes (Fig. 4) or other costs, the ability to rank plans and goals by its objective, and interpretability: it can generate entire planned and unplanned (undirected) trajectories.

Work by BID12 also uses multi-headed model-free conditional imitation learning to "warm start" a DDPG driving algorithm BID13 .

While warm starting hastens DDPG training, any subsequent DDPG post fine-tuning is inherently trial-and-error based, without guarantees of safety, and may crash during this learning phase.

By contrast, our method never executes unlikely transitions w.r.t.

expert behavior at training time nor at test time.

Our method can also stop the car if no plan reaches a minimum threshold, indicating none are likely safe to execute.

While our target setting is offline data collection, online imitation learning is an active area of research in the case of hybrid IL-RL BID25 BID31 and "safe" IL BID30 BID17 BID34 .

Although our work does not consider multiagent environments, several methods predict the behavior of other vehicles or pedestrians.

Typically this involves recurrent neural networks combined with Gaussian density layers or generative models based on some context inputs such as LIDAR, images, or known positions of external agents BID28 BID37 BID7 BID14 .

However, none of these methods can evaluate the likelihood of trajectories or repurpose their model to perform other inference tasks.

Other methods include inverse reinforcement learning to fit a probabilistic reward model to human demonstrations using the principle of maximum entropy BID36 BID27 BID22 .

We evaluate our method using the CARLA urban driving simulator BID5 .

Each test episode begins with the vehicle randomly positioned on a road in the Town01 or Town02 maps.

The task is to drive to a goal location, chosen to be the furthest road location from the vehicle's initial position.

As shown in Fig. 6 , we use three layers of spatial abstractions to plan to the goal location, common to model-based (not end-to-end) autonomous vehicle setups: coarse route planning over a road map, path planning within the observable space, and feedback control to follow the planned path BID19 BID29 .

First, we compute a route to the goal location using A * given knowledge of the road graph.

Second, we set waypoints along the route no closer than 20 m of the vehicle at any time to direct the vehicle.

Finally, we use a PID-controller to compute the vehicle steering value.

The PID-controller was tuned to steer the vehicle towards a setpoint (target) 5 meters away along the planned path.

We consider four metrics for this task: 1) Success rate in driving to the goal location without any collisions.

2) Proportion of time spent driving in the correct lane.

3) Frequency of crashes into obstacles.

4) Passenger comfort, by comparing the distribution of accelerations (and higher-order terms) between each method.

To contrast the benefits of our method against existing approaches, we compare against several baselines that all receive the same inputs and training data as our method.

Since our approach bridges model-free IL and MBRL, we include an IL baseline algorithm, and a MBRL baseline algorithm.

PID control: The PID baseline uses the PID-controller to follow the high-level waypoints along the route.

This corresponds to removing the middle layer of autonomous vehicle decision abstraction, which serves as a baseline for the other methods.

The PID controller is effective when the setpoint is several meters away, but fails when the setpoint is further away (i.e. at 20 m), causing the vehicle to cut corners at intersections.

We designed an IL baseline to control the vehicle.

A common straightforward approach to IL is behavior-cloning: learning to predict the actions taken by a demon-strator BID20 BID0 BID15 BID1 .

Our setting is that of goal-conditioned IL: in order to achieve different behaviors, the imitator is tasked with generating controls after observing a target high-level waypoint and φ.

We designed two baselines: one with the branched architecture of BID1 , where actions are predicted based on left/straight/right "commands" derived from the waypoints, and other that predicts the setpoint for the PID-controller.

Each receives the same φ and is trained with the same set of trajectories as our main method.

We found the latter method very effective for stable control on straightaways.

When the model encounters corners, however, prediction is more difficult, as in order to successfully avoid the curbs, the model must implicitly plan a safe path.

In the latter method, we used a network architecture nearly identical to our approach's..

To compare against a purely model-based reinforcement learning algorithm, we propose a model-predictive control baseline.

This baseline first learns a forwards dynamics model f : (s t−3 , s t−2 , s t−1 , s t , a t ) → s t+1 given observed expert data (a t are recorded vehicle actions).

We use an MLP with two hidden layers, each 100 units.

Note that our forwards dynamics model does not imitate the expert preferred actions, but only models what is physically possible.

Together with the same LIDAR map χ our method uses to locate obstacles, this baseline uses its dynamics model to plan a reachability tree BID9 through the free-space to the waypoint while avoiding obstacles.

We plan forwards over 20 time steps using a breadth-first search search over CARLA steering angle {−0.3, −0.1, 0., 0.1, 0.3}, noting valid steering angles are normalized to [−1, 1], with constant throttle at 0.5, noting the valid throttle range is [0, 1].

Our search expands each state node by the available actions and retains the 50 closest nodes to the waypoint.

The planned trajectory efficiently reaches the waypoint, and can successfully plan around perceived obstacles to avoid getting stuck.

To convert the LIDAR images into obstacle maps, we expanded all obstacles by the approximate radius of the car, 1.5 meters.

Performance results that compare our methods against baselines according to multiple metrics are includes in TAB1 .

With the exception of the success rate metric, lower numbers are better.

We define success rate as the proportion of episodes where the vehicles navigated across the road map to a goal location on the other side without any collisions.

In our experiments we do not include any other drivers or pedestrians, so a collision is w.r.t.

a stationary obstacle.

Collision impulse (in N · s) is the average cumulative collision intensities over episodes.

"

Wrong lane" and "Off road" percentage of the vehicle invading other lanes or offroad (averaged over time and episodes).

While safety metrics are arguably the most important metric, passenger comfort is also relevant.

Passenger comfort can be ambiguous to define, so we simply record the second to sixth derivatives of the position vector with respect to time, respectively termed acceleration, jerk, snap, crackle, and pop.

In TAB1 we note the 99th percentile of each statistic given all data collected per path planning method.

Generally speaking, lower numbers correspond to a smoother driving experience.

The poor performance of the PID baseline indicates that the high-level waypoints do not communicate sufficient information about the correct driving direction.

Imitation learning achieves better levels of comfort than MBRL, but exhibits substantially worse generalization from the training data, since it does not reason about the sequential structure in the task.

Model-based RL succeeds on most of the trials in the training environment, but exhibits worse generalization.

Notably, it also scores much worse than IL in terms of staying in the right lane and maintaining comfort, which is consistent with our hypothesis: it is able to achieve the desired goals, but does not capture the behaviors in the data.

Our method performs the best under all metrics, far exceeding the success and comfort metrics of imitation learning, and far exceeding the lane-obeyance and comfort metrics of MBRL.

To further illustrate the capability of our method to incorporate test-time costs, we designed a pothole collision experiment.

We simulated 2m-wide potholes in the environment by randomly inserting them in the cost map offset from each waypoint, distributed N (µ = [−15m, 2m], Σ = diag([1, 0.01])), (i.e. the mean is centered on the right side of the lane 15m before each waypoint).

We ran our method that incorporates a test-time cost map of the simulated potholes, and compared to our method that did not incorporate the cost map (and thus had no incentive to avoid potholes).

In addition to the other metrics, we recorded the number of collisions with potholes.

In TAB2 , we see that our method with cost incorporated achieved nearly perfect pothole avoidance, while still avoiding collisions with the environment.

To do so, it drove closer to the centerline, and occasionally dipped into the opposite lane.

Our model internalized obstacle avoidance by staying on the road, and demonstrated its flexibility to obstacles not observed during training.

FIG4 shows an example of this behavior.

As another test of our model's capability to stay in the distribution of demonstrated behavior, we designed a "decoy waypoints" experiment, in which half of the waypoints are highly perturbed versions of the other half, serving as distractions for our planner.

The planner is tasked with planning to all of the waypoints under the Gaussian mixture likelihood.

The perturbation distribution is N (0, σ = 8m): each waypoint is perturbed with a standard deviation of 8 meters.

We observed the imitative model to be surprisingly robust to decoy waypoints.

Examples of this robustness are shown in Fig. 8 .

One failure mode of this approach is when decoy waypoints lie on a valid off-route path at intersections, which temporarily confuses the planner about the best route.

In TAB3 , we report the success rate and the mean number of planning rounds for successful and failed episodes.

These numbers indicate our method can execute dozens to hundreds of planning rounds without decoy waypoints derailing it.

We also designed an experiment to test our method under systemic bias in the route planner.

Our method is provided waypoints on the wrong side of the road.

We model this by increasing the goal likelihood observation noise .

After tuning the noise, we found our method to still be very effective at navigating, and report results in TAB3 .

This further illustrates our method's tendency to stay near the distribution of expert behavior, as our expert never drove on the wrong side of the road.

Our method with waypoints on wrong side, Town01 10 / 10 0.338% 0.002% Our method with waypoints on wrong side, Town02 7 / 10 3.159% 0.044%

We proposed a method that combines elements of imitation learning and model-based reinforcement learning (MBRL).

Our method first learns what preferred behavior is by fitting a probabilistic model to the distribution of expert demonstrations at training time, and then plans paths to achieve userspecified goals at test time while maintaining high probability under this distribution.

We demonstrated several advantages and applications of our algorithm in autonomous driving scenarios.

In the context of MBRL, our method mitigates the distributional drift issue by explicitly preferring plans that stay close to the expert demonstration data.

This implicitly allows our method to enforce basic safety properties: in contrast to MBRL, which requires negative examples to understand the potential for adverse outcomes (e.g., crashes), our method automatically avoids such outcomes specifically because they do not occur (or rarely occur) in the training data.

In the context of imitation learning, our method provides a flexible, safe way to generalize to new goals by planning, compared to prior work on black-box, model-free conditional imitation learning.

Our algorithm produces an explicit plan within the distribution of preferred behavior accompanied with a score: the former offers interpretability, and the latter provides an estimate of the feasibility of the plan.

We believe our method is broadly applicable in settings where expert demonstrations are available, flexibility to new situations is demanded, and safety is critical.

Figure 8 : Tolerating bad waypoints.

The planner prefers waypoints in the distribution of expert behavior: on the road at a reasonable distance.

Columns 1,2: Planning with 1 /2 decoy waypoints.

Columns 3,4: Planning with all waypoints on the wrong side of the road.

<|TLDR|>

@highlight

Hybrid Vision-Driven Imitation Learning and Model-Based Reinforcement Learning for Planning, Forecasting, and Control
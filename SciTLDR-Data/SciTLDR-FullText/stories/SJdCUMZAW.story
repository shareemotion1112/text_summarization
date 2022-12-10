Grasping an object and precisely stacking it on another is a difficult task for traditional robotic control or hand-engineered approaches.

Here we examine the problem in simulation and provide techniques aimed at solving it via deep reinforcement learning.

We introduce two straightforward extensions to the Deep Deterministic Policy Gradient algorithm (DDPG), which make it significantly more data-efficient and scalable.

Our results show that by making extensive use of off-policy data and replay, it is possible to find high-performance control policies.

Further, our results hint that it may soon be feasible to train successful stacking policies by collecting interactions on real robots.

Dexterous manipulation is a fundamental challenge in robotics.

Researchers have long sought a way to enable robots to robustly and flexibly interact with fixed and free objects of different shapes, materials, and surface properties in the context of a broad range of tasks and environmental conditions.

Such flexibility is very difficult to achieve with manually designed controllers.

The recent resurgence of neural networks and "deep learning" has inspired hope that these methods will be as effective in the control domain as they are for perception.

Indeed, recent work has used neural networks to learn solutions to a variety of control problems BID31 BID6 BID30 BID10 BID16 .While the flexibility and generality of learning approaches is promising for robotics, these methods typically require a large amount of data that grows with the complexity of the task.

What is feasible on a simulated system, where hundreds of millions of control steps are possible BID22 BID31 , does not necessarily transfer to real robot applications due to unrealistic learning times.

One solution to this problem is to restrict the generality of the controller by incorporating task specific knowledge, e.g. in the form of dynamic movement primitives BID29 , or in the form of strong teaching signals, e.g. kinesthetic teaching of trajectories BID23 .

Recent works have had success learning flexible neural network policies directly on real robots (e.g. BID4 BID38 ), but tasks as complex as precise grasping-and-stacking remain daunting.

In this paper we investigate in simulation the possibility of learning precise manipulation skills endto-end with a general purpose model-free deep reinforcement learning algorithm.

We assess the feasibility of performing analogous experiments on real robotics hardware and provide guidance with respect to the choice of learning algorithm, experimental setup, and the performance that we can hope to achieve.

We consider the task of picking up a Lego brick from the table and stacking it onto a second nearby brick using a robotic arm and gripper.

This task involves contact-rich interactions between the robotic arm and two freely moving objects.

It also requires mastering several sub-skills (reaching, grasping, lifting, and stacking) .

Each of these sub-skills is challenging in its own right as they require both precision (for instance, successful stacking requires accurate alignment of the two bricks) and as well as robust generalization over a large state space (e.g. different initial positions of the bricks and the initial configuration of the arm).

Finally, there exist non-trivial and long-ranging dependencies between the solutions for different sub-tasks: for instance, the ability to successfully stack the brick depends critically on having picked up the brick in a sensible way beforehand.

This paper makes several contributions: 1.

We build on the Deep Deterministic Policy Gradient (DDPG; ), a general purpose model-free reinforcement learning algorithm for continuous actions, and extend it in two ways: firstly, we improve the data efficiency of the algorithm by scheduling updates of the network parameters independently of interactions with the environment.

Secondly, we overcome the computational and experimental bottlenecks of single-machine single-robot learning by introducing a distributed version of DDPG which allows data collection and network training to be spread out over multiple computers and robots.

2.

We show how to use these straightforward algorithmic developments to solve a complex, multi-stage manipulation problem.

We further propose two broadly applicable strategies that allow us to reliably find solutions to complex tasks and further reduce the amount of environmental interaction.

The first of these strategies is a recipe for designing effective shaping rewards for compositional tasks, while the second biases the distribution of initial states to achieve an effect akin a form of apprenticeship learning.

In combination these contributions allow us to reliably learn robust policies for the full stacking task from scratch in less than 10 million environment transitions.

This corresponds to less than 10 hours of interaction time on 16 robots.

In addition, we show that when states from demonstration trajectories are used as the start states for learning trials the full task can be learned with 1 million transitions (i.e. less than 1 hour of interaction on 16 robots).

To our knowledge our results provide the first demonstration of end-to-end learning for a complex manipulation problem involving multiple freely moving objects.

They are also suggest that it may be possible to learn such non-trivial manipulation skills directly on real robots.

Reinforcement learning (RL) approaches solve tasks through repeated interactions with the environment guided by a reward signal of success or failure BID33 .

A distinction is often made between value-based and policy search methods.

The latter have been routinely applied in robotics, in part because they straightforwardly handle continuous and high-dimensional action spaces BID2 , and applications include manipulation BID25 BID36 BID4 BID38 BID7 , locomotion e.g. BID15 BID20 , and a range of other challenges such as helicopter flight BID0 ).

However, policy search methods can scale poorly with the number of parameters that need to be estimated, requiring the need for restricted policy classes, that in turn might not be powerful enough for solving complex tasks.

One exception are guided policy search methods (GPS) BID38 .

These employ a teacher algorithm to locally optimize trajectories which are then summarized by a neural network policy.

They gain data-efficiency by employing aggressive local policy updates and extensive training of their neural network policy.

The teacher can use model-based or model-free BID38 trajectory optimization.

The former can struggle with strong discontinuities in the dynamics, and both rely on access to a well defined and fully observed state space.

Alternatively, model-free value function approaches enable effective reuse of data and do not require full access to the state space or to a model of the environment.

The use of rich function approximators such as neural networks in value function methods dates back many years, e.g. BID37 BID34 BID11 BID9 , and recent success with deep learning has driven the development of new end-to-end training methods for challenging control problems BID21 BID5 c; .

Closely related to the ideas followed in this paper, BID4 demonstrates that value-based methods using neural network approximators can be used for relatively simple robotic manipulation tasks in the real world BID6 ).

This work also followed a recent trend towards the use of experimental rigs that allow parallelized data collection, e.g. BID26 , via the use of multiple robots from which experience is gathered simultaneously BID4 BID38 .Finally, the use of demonstration data has played an important role in robot learning, both as a means to obtain suitable cost functions BID1 BID13 BID3 BID7 but also to bootstrap and thus speed up learning.

For the latter, kinesthetic teaching is widely used BID25 BID38 , though the need for a human operator to be able to guide the robot through the full movement can be limiting.

In this section we explain the learning problem and summarize the DDPG algorithm.

We explain its relationship to other Q-function based RL algorithms in the Appendix.

The RL problem consists of an agent interacting with an environment in a sequential manner to maximize the expected sum of rewards.

At time t the agent observes the state x t of the system and produces a control u t = π(x t ; θ) according to policy π with parameters θ.

This leads the environment to transition to a new state x t+1 according to the dynamics x t+1 ∼ p(·|x t , u t ), and the agent receives a reward r t = r(x t , u t ).

The goal is to maximize the expected sum of discounted rewards J(θ) = E τ ∼ρ θ t γ t−1 r(x t , u t ) , where ρ θ is the distribution over trajectories τ = (x 0 , u 0 , x 1 , u 1 , . . . ) induced by the current policy: DISPLAYFORM0 DPG BID32 ) is a policy gradient algorithm for continuous action spaces that improves the deterministic policy function π via backpropagation of the action-value gradient from a learned approximation to the Q-function.

Specifically, DPG maintains a parametric approximation Q(x t , u t ; φ) to the action value function Q π (x t , u t ) associated with π and φ is chosen to minimize DISPLAYFORM1 where y t = r(x t , u t ) + γQ(x t+1 , π(x t+1 )).ρ is usually close to the marginal transition distribution induced by π but often not identical.

For instance, during learning u t may be chosen to be a noisy version of π(x t ; θ), e.g. u t = π(x t ; θ) + where ∼ N (0, σ 2 ) andρ is then the transition distribution induced by this noisy policy.

The policy parameters θ are then updated according to DISPLAYFORM2 DDPG incorporates experience replay and target networks to the original DPG algorithm: Experience is collected into a buffer and updates to θ and φ (eqs. 1, 2) are computed using mini-batch updates with samples from this buffer.

A second set of "target-networks" is maintained with parameters θ and φ .

These are used to compute y t in eqn.(1) and their parameters are slowly updated towards the current parameters θ, φ.

Both measures significantly improve the stability of DDPG.The use of a Q-function facilitates off-policy learning.

This decouples the collection of experience data from the updates of the policy and value networks which allows us to make many parameter update steps per step in the environment, ensuring that the networks are well fit to the data that is currently available.

The full task that we consider in this paper is to use the arm to pick up one Lego brick from the table and stack it onto the remaining brick.

This "composite" task can be decomposed into several subtasks, including grasping and stacking.

We consider the full task as well as the two sub-tasks in isolation:Starting state Reward Grasp Both bricks on In every episode the arm starts in a random configuration with an appropriate positioning of gripper and brick.

We implement the experiments in a physically plausible simulation in MuJoCo BID35 with the simulated arm being closely matched to a real-world Jaco arm setup in our lab.

Episodes are terminated after 150 steps of 50ms of physical simulation time.

The agent thus has 7.5 seconds to perform the task.

Unless otherwise noted we give a reward of one upon successful completion of the task and zero otherwise.

The observation contains information about the angles and angular velocities of the 6 joints of the arm and 3 fingers of the gripper, as well as the position and orientation of the two bricks and relative distances of the two bricks to the pinch position of the gripper (roughly the position where the fingertips would meet if the fingers are closed).

The 9-dimensional continuous action directly sets the velocities of the arm and finger joints.

In experiments not reported in this paper we have tried using observations with only the raw state of the brick and the arm configuration (i.e. without the vector between the end-effector and brick) This increased the number of environment interactions needed roughly by a factor of two to three.

For each experimental condition we optimize the learning rate and train and measure the performance of 10 agents with different random initial network parameters.

After every 30 training episodes the agent is evaluated for 10 episodes.

We used the mean performance at each evaluation phase as the performance measure presented in all plots.

In the plots the line shows the mean performance across agents and the shaded regions correspond to the range between the worst and best performing one In all plots the x-axis represents the number of environment transitions seen so far at an evaluation point (in millions) and the y-axis represent episode return.

A video of the full setup and examples of policies solving the component and full tasks can be found here: https://www.youtube.com/watch?v=7vmXOGwLq24.

In this section we study two methods for extending the DDPG algorithm and find that they can have significant effect on data and computation efficiency, in some cases making the difference between finding a solution to a task or not.

Multiple mini-batch replay steps Deep neural networks can require many steps of gradient descent to converge.

In a supervised learning setting this affects purely computation time.

In reinforcement learning, however, neural network training is interleaved with the acquisition of interaction experience giving rise to a complex interaction.

To gain a better understanding of this effect we modified the original DDPG algorithm as described in to perform a fixed but configurable number of mini-batch updates per step in the environment.

In one update was performed after each new interaction step.

We refer to DDPG with a configurable number of update steps as DPG-R and tested the impact of this modification on the two primitive tasks Grasp and StackInHand.

The results are shown in FIG1 .

The number of update steps has a dramatic effect on the amount of experience data required.

After one million interactions the original version of DDPG with a single update step (blue traces) appears to have made no progress towards a successful policy for stacking, and only a small number of controllers have learned to grasp.

Increasing the number of updates per interaction to 5 greatly improves the results (green traces), and with 40 updates (purple) the first successful policies for stacking and grasping are obtained after 200,000 and 300,000 interactions respectively (corresponding to 1,300 and 2,000 episodes).

Notably, for both tasks we continue to see a reduction in total environment interaction up to 40 update steps, the maximum used in the experiment.

One possible explanation for this effect is the interaction alluded to above: insufficient training may lead to a form of underfitting of the policy.

Since the policy is then used for exploration this affects the quality of the data collected in the next iteration which in turn has an effect on training in future iterations leading to overall slow learning.

We have observed in various experiments (not shown) that other aspects of the network architecture (layer sizes, non-linearities) can similarly affect learning speed.

Finally, it is important to note that one cannot replicate the effect of multiple replay steps simply by increasing the learning rate.

In practice we find that attempts to do so make training unstable.

Asynchronous DPG Increasing the number of update steps relative to the number of environment interactions greatly improves the data efficiency but also dramatically increases compute time.

When the overall run time is dominated by the network updates it may scale linearly with the number of replay steps.

In this setting experiments can quickly become impractical and parallelizing computation can provide a solution.

Similarly, in a robotics setup the overall run time is typically dominated by the collection of interactions.

In this case it is desirable to be able to collect experience from multiple robots simultaneously (e.g. as in BID38 BID4 ).We therefore develop an asynchronous version of DPG that allows parallelization of training and environment interaction by combining multiple instances of an DPG-R actor and critic that each share their network parameters and can be configured to either share or have independent experience replay buffers.

This is inspired by the A3C algorithm proposed in BID22 , and also analogous to BID4 BID38 : We employ asynchronous updates whereby each worker has its own copy of the parameters and uses it for computing gradients which are then applied to a shared parameter instance without any synchronization.

We use the Adam optimizer BID14 with local non-shared first-order statistics and a single shared instance of second-order statistics.

The pseudo code of the asynchronous DPG-R is shown in algorithm box 1.

Initialize global shared critic and actor network parameters: θ Q and θ µ Pseudo code for each learner thread: Initialize critic network Q(s, a|θ Q ) and policy network µ(s|θ µ ) with weights θ Q and θ µ .Initialize target network Q and µ with weights: DISPLAYFORM0 Initialize replay buffer R for episode = 1, M do Receive initial observation state s1 for t = 1, T do Select action at = µ(st|θ µ ) + Nt according to the current policy and exploration noise Perform action at, observe reward rt and new state st+1 Store transition (st, at, rt, st+1) in R for update = 1, R do Sample a random minibatch of N transitions (si, ai, ri, si+1) from R Set yi = ri + γQ (si+1, µ (si+1|θ µ )|θ Q ) Perform asynchronous update of the shared critic parameters by minimizing the loss: DISPLAYFORM1 2 ) Perform asynchronous update of the shared policy parameters using the sampled gradient: DISPLAYFORM2 Copy the shared parameters to the local ones: DISPLAYFORM3 Every S update steps, update the target networks: Figure 2 (right) compares the performance of ADPG-R for different number of update steps and 16 workers (all workers performing both data collection and computing updates).

Similar to FIG1 (left) we find that increasing the ratio of update steps per environment steps improves data efficiency, although the effect appears to be somewhat less pronounced than for DPG-R. FIG2 (left) directly compares the single-worker and asynchronous version of DPG-R. In both cases we choose the best performing number of replay steps and learning rate.

As we can see, the use of multiple workers does not affect overall data efficiency for StackInHand but it reduced roughly in half for Grasp, with the note that the single worker still hasn't quite converged.

DISPLAYFORM4 for end for end forFigure 3 (right) plots the same data but as a function of environment steps per worker.

This measure corresponds to the optimal wall clock efficiency that we can achieve, under the assumption that communication time between workers is negligible compared to environment interaction and gradient computation (this usually holds up to a certain degree of parallelization).

The theoretical wall clock time for 16 workers is about 16x lower for StackInHand and roughly 8x lower for Grasp.

Overall these results show that distributing neural network training and data collection across multiple computers and robots can be an extremely effective way of reducing the overall run time of experiments and thus making it feasible to run more challenging experiments.

We make extensive use of asynchronous DPG for remaining the experiments.

The reward function in the previous section was "sparse" or "pure" reward where a reward of 1 was given for states that correspond to successful task completion (brick lifted above 3cm for grasp; for stack) and 0 otherwise.

For this reward to be useful it is necessary that the agent enters the goal region at least some of the time.

While possible for each of the two subtasks in isolation, this is highly unlikely for the full task: without further guidance naïve random exploration is very unlikely to lead to a successful grasp-and -stack as we experimentally verify in FIG3 .One solution are informative shaping rewards that provide a learning signal even for simple exploration strategies, e.g. by embedding information about the value function in the reward function.

This is a convenient way of embedding prior knowledge about the solution and is a widely and successfully used approach for simple problems.

For complex sequential or compositional tasks such as the one we are interested in here, however, a suitable reward function is often non-obvious and may require considerable effort and experimentation.

In this section we propose and analyze several reward functions for the full Stack task, and provide a general recipe that can be applied to other tasks with compositional structure.

Shaping rewards are often defined using a distance from or progress towards a goal state.

Analogously our composite (shaping) reward functions return an increasing reward as the agent completes components of the full task.

They are either piece-wise constant or smoothly varying across different regions of the state space that correspond to completed subtasks.

In the case of Stack we use the following reward components (see the Appendix): These reward components can be combined in different ways.

We consider three different composite rewards in additional to the original sparse task reward: Grasp shaping: Grasp brick 1 and Stack brick 1, i.e. the agent receives a reward of 0.25 when brick 1 has been grasped and a reward of 1.0 after completion of the full task.

Reach and grasp shaping: Reach brick 1, Grasp brick 1 and Stack brick 1, i.e. the agent receives a reward of 0.125 when close to brick 1, a reward of 0.25 when brick 1 has been grasped, and a reward of 1.0 after completion of the full task.

Full composite shaping: the sparse reward components as before in combination with the distancebased smoothly varying components A full description of the reward functions is provided in the Appendix.

The actual reward functions given above are specific to the stacking task.

But the general principle, a piecewise-constant sequence of rewards that increases as components of the tasks are completed, augmented with simple smoothly varying rewards that guide towards completion of individual subtasks should be widely applicable.

It is important to note that the above reward functions do not describe all aspects of the task solution: we do not tell the agent how to grasp or stack but merely to bring the arm into a position where grasping (stacking) can be discovered from exploration and the sparse reward component.

This eases the burden on the designer and is less likely to change the optimal solution in unwanted ways.

In the previous section we described a strategy for designing effective compositional reward functions that alleviate the burden of exploration.

However, designing such rewards can still be error prone and we did indeed encounter several unexpected failure cases as shown in the supplemental video (https://www.youtube.com/watch?v=7vmXOGwLq24) and detailed in the Appendix.

Furthermore, suitable rewards may rely on privileged information not easily available in a real robotics setup.

In this section we describe a second, complementary strategy for embedding prior knowledge into the training process and improving exploration.

Specifically we propose to let the distribution of states at which the learning agent is initialized at the beginning of an episode reflect the compositional nature of the task: e.g., instead of initializing the agent at the beginning of the full task with both bricks on the table, we can initialize the agent occasionally with the brick already in its hand and thus prepared for stacking in the same way as when learning the subtask StackInHand in section 5.More generally, we can initialize episodes with states taken from anywhere along or close to successful trajectories.

Suitable states can be either manually defined (as in section 5), or they can be obtained from a human demonstrator or a previously trained agent that can partially solve the task.

This can be seen as a form of apprenticeship learning in which we provide teacher information by influencing the state visitation distribution.

Unlike many other forms of imitation or apprenticeship learning, however, this approach requires neither complete trajectories nor demonstrator actions. .

On all plots, x-axis is millions of transitions of total experience and y-axis is mean episode return.

Policies with mean return over 100 robustly perform the full Stack from different starting states.

Without reward shaping and basic start states only (a, blue) there is no learning progress.

Instructive start states allow learning even with very uninformative sparse rewards indicating only overall task success (a,red).We perform experiments with two methods for generating the starting states.

The first one uses the manually defined initial states from section 5 (both bricks located on the table or in states where the first brick is already in the gripper as if the agent just performed a successful grasp).

The second method initializes the learning agent at start states sampled randomly from successful demonstration trajectories (derived from agents previously trained end-to-end on the compositional reward).The results of these experiments are shown in FIG3 .

Green traces show results for the four reward functions from section 6 in combination with the manually defined start states (from section 5).

While there is still no learning for the sparse reward case, results obtained with all other reward functions are improved.

In particular, even for the second simplest reward function (Grasp shaping) we obtain some controllers that can solve the full task.

Learning with the full composite shaping reward is faster and more robust than without the use of instructive states.

The leftmost plot of FIG3 (red trace) shows results for the case where the episode is initialized anywhere along trajectories from a pre-trained controller (which was obtained using full composite shaping; rightmost blue curve).

We use this start state distribution in combination with the basic sparse reward for the overall case (Stack without shaping).

Episodes were configured to be 50 steps, which we found to be better suited to this setup with assisted exploration.

During testing we still used episodes with 150 steps as before (so that the traces are comparable).

We can see a large improvement in performance in comparison to the two-state method variant even in the absence of any shaping rewards.

We can learn a robust policy for all seeds within a total of 1 million environment transitions -less than 1 hour of interaction time on 16 simulated robots.

These results suggest that an appropriate start state distribution not only speeds up learning, it also allows simpler reward functions to be used.

In our final experiment we found that the simplest reward function (i.e. only indicating overall experimental success) was sufficient to solve the task.

In this case the robustness of trained policies to starting state variation is also encouraging.

Over 1000 test trials we obtain 99.2% success for Grasp, 98.2% for StackInHand, and 95.5% for the full Stack task.

We have introduced two extensions to the DDPG algorithm which make it a practical method for learning robust policies for complex continuous control tasks.

We have shown that by decoupling the frequency of network updates from the environment interaction we can dramatically improve data-efficiency.

Parallelizing data acquisition and learning substantially reduces wall clock time.

In addition, we presented two methods that help to guide the learning process towards good solutions and thus reduce the pressure on exploration strategies and speed up learning.

In combination these contributions allow us to solve a challenging manipulation problem end-to-end, suggesting that many hard control problems lie within the reach of modern learning methods.

It is of course challenging to judge the transfer of results in simulation to the real world.

We have taken care to design a physically realistic simulation, and in initial experiments, which we have performed both in simulation and on the physical robot, we generally find a good correspondence of performance and learning speed between simulation and real world.

This makes us optimistic that performance numbers may also hold when going to the real world.

A second limitation of our simulated setup is that it currently uses information about the state of the environment would require additional instrumentation of the experimental setup, e.g. to determine the position of the two bricks in the work space.

These are issues that need to be addressed with care as experiments move to robotics hardware in the lab.

Nevertheless, the algorithms and techniques presented here offer important guidance for the application of deep reinforcement learning methods to dexterous manipulation on a real robot.

9 DDPG AND OTHER ALGORITHMS DDPG bears a relation to several other recent model free RL algorithms: The NAF algorithm BID6 which has recently been applied to a real-world robotics problem BID4 can be viewed as a DDPG variant where the Q-function is quadratic in the action so that the optimal action can be easily recovered directly from the Q-function, making a separate representation of the policy unnecessary.

DDPG and especially NAF are the continuous action counterparts of DQN BID21 , a Q-learning algorithm that recently re-popularized the use of experience replay and target networks to stabilize learning with powerful function approximators such as neural networks.

DDPG, NAF, and DQN all interleave mini-batch updates of the Q-function (and the policy for DDPG) with data collection via interaction with the environment.

These mini-batch based updates set DDPG and DQN apart from the otherwise closely related NFQ and NFQCA algorithms for discrete and continuous actions respectively.

NFQ BID28 and NFQCA BID8 employ the same basic update as DDPG and DQN, however, they are batch algorithms that perform updates less frequently and fully re-fit the Q-function and the policy network after every episode with several hundred iterations of gradient descent with Rprop BID27 and using full-batch updates with the entire replay buffer.

The aggressive training makes NFQCA data efficient, but the full batch updates can become impractical with large networks, large observation spaces, or when the number of training episodes is large.

Finally, DPG can be seen as the deterministic limit of a particular instance of the stochastic value gradients (SVG) family BID10 , which also computes policy gradient via back-propagation of value gradients, but optimizes stochastic policies.

Target networks DQN DDPG, NAF Full-batch learning with Rprop Parameter resetting NFQ NFQCA

In this section we provide further details regarding the composite reward functions described in the main text.

For our experiments we derived these from the state vector of the simulation, but they could also be obtained through instrumentation in hardware.

The reward functions are defined in terms of the following quantities:• b(1) z : height of brick 1 above table • s B1 {x,y,z} : x,y,z positions of site located roughly in the center of brick 1 • s B2 {x,y,z} : x,y,z positions of site located just above brick 2, at the position where s B1 will be located when brick 1 is stacked on top of brick 2.• s P {x,y,z} : x,y,z positions of the pinch site of the hand -roughly the position where the fingertips would meet if the fingers are closed..

Using the above we can define the following conditions for the successful completion of subtasks:Reach Brick 1 The pinch site of the fingers is within a virtual box around the first brick position.

DISPLAYFORM0 where ∆ reach {x,y,z} denote the half-lengths of the sides of the virtual box for reaching.

Grasp Brick 1 Brick 1 is located above the table surface by a threshold, θ, that is possible only if the arm is the brick has been lifted.

grasp =b (1) z > θ Stack Brick 1 is stacked on brick 2.

This is expressed as a box constraint on the displacement between brick 1 and brick 2 measured in the coordinate system of brick 2.

stack =(|C (2) DISPLAYFORM1 where ∆ stack {x,y,z} denote the half-lengths of the sides of the virtual box for stacking, and C (2) is the rotation matrix that projects a vector into the coordinate system of brick 2.

This projection into the coordinate system of brick 2 is necessary since brick 2 is allowed to move freely.

It ensures that the box constraint is considered relative to the pose of brick 2.

While this criterion for a successful stack is quite complicated to express in terms of sites, it could be easily implemented in hardware e.g. via a contact sensor attached to brick 2.

The full composite reward also includes two distance based shaping components that guide the hand to the brick 1 and then brick 1 to brick 2.

These could be approximate and would be relatively simple to implement with a hardware visual system that can only roughly identify the centroid of an object.

The shaping components of the reward are given as follows:Reaching to brick 1 : DISPLAYFORM0 Reaching to brick 2 for stacking r S2 (s B1 , s B2 ) = 1 − tanh 2 (w 2 s B1 − s

2 ).

Using the above components the reward functions we implement the composite reward functions described in the main text: Stack, Grasp shaping, Reach and grasp shaping, and Full composite shaping can be expressed as in equations (3, 4, 5, 6) below.

These make use of the predicates above to determine whether which subtasks have been completed and return a reward accordingly.

@highlight

Data-efficient deep reinforcement learning can be used to learning precise stacking policies.
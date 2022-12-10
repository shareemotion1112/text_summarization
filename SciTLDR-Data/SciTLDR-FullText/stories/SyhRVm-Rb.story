Reinforcement learning (RL) is a powerful technique to train an agent to perform a task.

However, an agent that is trained using RL is only capable of achieving the single task that is specified via its reward function.

Such an approach does not scale well to settings in which an agent needs to perform a diverse set of tasks, such as navigating to varying positions in a room or moving objects to varying locations.

Instead, we propose a method that allows an agent to automatically discover the range of tasks that it is capable of performing in its environment.

We use a generator network to propose tasks for the agent to try to achieve, each task being specified as reaching a certain parametrized subset of the state-space.

The generator network is optimized using adversarial training to produce tasks that are always at the appropriate level of difficulty for the agent.

Our method thus automatically produces a curriculum of tasks for the agent to learn.

We show that, by using this framework, an agent can efficiently and automatically learn to perform a wide set of tasks without requiring any prior knowledge of its environment (Videos and code available at: https://sites.google.com/view/goalgeneration4rl).

Our method can also learn to achieve tasks with sparse rewards, which pose significant challenges for traditional RL methods.

Reinforcement learning (RL) can be used to train an agent to perform a task by optimizing a reward function.

Recently, a number of impressive results have been demonstrated by training agents using RL: such agents have been trained to defeat a champion Go player BID16 , to outperform humans in 49 Atari games (Guo et al., 2016; Mnih et al., 2015) , and to perform a variety of difficult robotics tasks (Lillicrap et al., 2015; BID18 .

In each of the above cases, the agent is trained to optimize a single reward function in order to learn to perform a single task.

However, there are many real-world environments in which a robot will need to be able to perform not a single task but a diverse set of tasks, such as navigating to varying positions in a room or moving objects to varying locations.

We consider the problem of maximizing the average success rate of our agent over all possible goals, where success is defined as the probability of successfully reaching each goal by the current policy.

In order to efficiently maximize this objective, the algorithm must intelligently choose which goals to focus on at every training stage: goals should be at the appropriate level of difficulty for the current policy.

To do so, our algorithm allows an agent to generate its own reward functions, defined with respect to target subsets of the state space, called goals.

We generate such goals using a Goal Generative Adversarial Network (Goal GAN), a variation of to the GANs introduced by Goodfellow et al. (2014) .

A goal discriminator is trained to evaluate whether a goal is at the appropriate level of difficulty for the current policy, and a goal generator is trained to generate goals that meet this criteria.

We show that such a framework allows an agent to quickly learn a policy that reaches all feasible goals in its environment, with no prior knowledge about the environment or the tasks being performed.

Our method automatically creates a curriculum, in which, at each step, the generator generates goals that are only slightly more difficult than the goals that the agent already knows how to achieve.

In summary, our main contribution is a method for automatic curriculum generation that considerably improves the sample efficiency of learning to reach all feasible goals in the environment.

Learning to reach multiple goals is useful for multi-task settings such as navigation or manipulation, in which we want the agent to perform a wide range of tasks.

Our method also naturally handles sparse reward functions, without needing to manually modify the reward function for every task, based on prior task knowledge.

Instead, our method dynamically modifies the probability distribution from which goals are sampled to ensure that the generated goals are always at the appropriate difficulty level, until the agent learns to reach all goals within the feasible goal space.

The problem that we are exploring has been referred to as "multi-task policy search" BID11 or "contextual policy search," in which the task is viewed as the context for the policy BID10 BID13 .

Unlike the work of BID11 , our work uses a curriculum to perform efficient multi-task learning, even in sparse reward settings.

In contrast to BID13 , which trains from a small number of discrete contexts / tasks, our method generates a training curriculum directly in continuous task space.

Intrinsic Motivation:

Intrinsic motivation involves learning with an intrinsically specified objective (Schmidhuber, 1991; BID4 .

Intrinsic motivation has also been studied extensively in the developmental robotics community, such as SAGG-RIAC BID4 BID5 , which has a similar goal of learning to explore a parameterized task space.

However, our experiments with SAGG-RIAC demonstrate that this approach do not explore the space as efficiently as ours.

A related concept is that of competence-based intrinsic motivation BID3 , which uses a selector to select from a discrete set of experts.

Recently there have been other formulations of intrinsic motivation, relating to optimizing surprise BID19 BID0 or surrogates of state-visitation counts BID7 BID19 .

All these approaches improve learning in sparse tasks where naive exploration performs poorly.

However, these formulations do not have a notion of which states are hard for the learner, and the intrinsic motivation is independent of the current performance of the agent.

In contrast, our formulation of intrinsic motivation directly relates to our policy performance: the agent is motivated to train on tasks that push the boundaries of its capabilities.

Curriculum Learning: The increasing interest on training single agents to perform multiple tasks is leading to new developments on how to optimally present the tasks to the agent during learning.

The idea of using a curriculum has been explored in many prior works on supervised learning BID9 BID22 BID8 .

However, these curricula are usually hand-designed, using the expertise of the system designer.

Another lines of work take into explicit consideration which examples are hard for the current learner (Kumar et al., 2010; Jiang et al., 2015) ; or use learning progress to build an automatic curriculum (Graves et al., 2017) , however both approaches have mainly been applied for supervised tasks.

Most curriculum learning in RL still relies on fixed pre-specified sequences of tasks (Karpathy & Van De Panne, 2012) .

Other recent work has proposed using a given baseline performance for several tasks to gauge which tasks are the hardest and require more training BID15 , but the framework can only handle a finite set of tasks and cannot handle sparse rewards.

Our method trains a policy that generalizes to a set of continuously parameterized tasks and is shown to perform well even under sparse rewards by not allocating training effort to tasks that are too hard for the current performance of the agent.

Finally, an interesting self-play strategy has been proposed that is concurrent to our work BID17 ; however, they view their approach as simply providing an exploration bonus for a single target task; in contrast, we focus on the problem of efficiently optimizing a policy across a range of goals, as we explain below.

In the traditional RL framework, at each timestep t, the agent in state s t ∈ S ⊆ R n takes an action a t ∈ A ⊆ R m , according to some policy π(a t |s t ) that maps from the current state s t to a probability distribution over actions.

Taking this action causes the agent to enter into a new state s t+1 according to a transition distribution p(s t+1 |s t , a t ), and receive a reward r t = r(s t , a t , s t+1 ).

The objective of the agent is to find the policy π that maximizes the expected return, defined as the sum of rewards R = T t=0 r t , where T is a maximal time given to perform the task.

The learned policy corresponds to maximizing the expected return for a single reward function.

In our framework, instead of learning to optimize a single reward function, we consider a range of reward functions r g indexed or parametrized by a goal g ∈ G. Each goal g corresponds to a set of states S g ⊂ S such that goal g is considered to be achieved when the agent is in any state s t ∈ S g .

Then the objective is to learn a policy that, given any goal g ∈ G, acts optimally with respect to r g .

We define a very simple reward function that measures whether the agent has reached the goal: DISPLAYFORM0 where 1 is the indicator function.

In our case, we use S g = {s t : d(f (s t ), g) ≤ }, where f (·) is a function that projects a state into goal space G, d(·, ·) is a distance metric in goal space, and is the acceptable tolerance that determines when the goal is reached.

However, our method can handle generic binary rewards (as in Eq. FORMULA0 ) and does not require a distance metric for learning.

Furthermore, we define our MDP such that each episode terminates when s t ∈ S g .

Thus, the return DISPLAYFORM1 r g t is a binary random variable whose value indicates whether the agent has reached the set S g in at most T time-steps.

Hence, the return of a trajectory s 0 , s 1 , . . . can be expressed as DISPLAYFORM2 , policies are also conditioned on the current goal g (as in Schaul et al. (2015) ), written as π(a t | s t , g).

The expected return obtained when we take actions sampled from the policy can then be expressed as the probability of succeeding on each goal within T timesteps, as shown in Eq. (2).

DISPLAYFORM3 The sparse indicator reward function of Eq. FORMULA0 is not only simple but also represents a property of many real-world goal problems: in many settings, it may be difficult to tell whether the agent is getting closer to achieving a goal, but easy to tell when a goal has been achieved.

For example, for a robot moving in a maze, taking actions that maximally reduce the straight-line distance from the start to the goal is usually not a feasible approach for reaching the goal, due to the presence of obstacles along the path.

In theory, one could hand-engineer a meaningful distance function for each task that could be used to create a dense reward function.

Instead, we use the indicator function of Eq.(1), which simply captures our objective by measuring whether the agent has reached the goal state.

We show that our method is able to learn even with such sparse rewards.

We desire to find a policy π(a t | s t , g) that achieves a high reward for many goals g. We assume that there is a test distribution of goals p g (g) that we would like to perform well on.

For simplicity, we assume that the test distribution samples goals uniformly from the set of goals G, although in practice any distribution can be used.

The overall objective is then to find a policy π * such that DISPLAYFORM0 Recall from Eq. (2) that R g (π) is the probability of success for each goal g. Thus the objective of Eq. (3) measures the average probability of success over all goals sampled from p g (g).

We refer to the objective in Eq. (3) as the coverage objective.

Similar to previous work (Schaul et al., 2015; Kupcsik et al., 2013; BID13 BID11 we need a continuous goal-space representation such that a goal-conditioned policy can efficiently generalize over the goals.

In particular, we assume that:1.

A policy trained on a sufficient number of goals in some area of the goal-space will learn to interpolate to other goals within that area.

2.

A policy trained on some set of goals will provide a good initialization for learning to extrapolate to close-by goals, meaning that the policy can occasionally reach them but maybe not consistently so.

Furthermore, we assume that if a goal is reachable, there exists a policy that does so reliably.

This is a reasonable assumption for any practical robotics problem, and it will be key for our method, as it strives to train on every goal until it is consistently reached.

Our approach can be broken down into three parts: First, we label a set of goals based on whether they are at the appropriate level of difficulty for the current policy.

Second, using these labeled goals, we construct and train a generator to output new goals that are at the appropriate level of difficulty.

Finally, we use these new goals to efficiently train the policy, improving its coverage objective.

We iterate through each of these steps until the policy converges.

As shown in our experiments, sampling goals from p g (g) directly, and training our policy on each sampled goal may not be the most sample efficient way to optimize the coverage objective of Eq. (3).

Instead, we modify the distribution from which we sample goals during training: we wish to find the set of goals g in the set DISPLAYFORM0 The justification for this is as follows: due to the sparsity of the reward function, for most goals g, the current policy π i (at iteration i) obtains no reward.

Instead, we wish to train our policy on goals g for which π i is able to receive some minimum expected return R g (π i ) > R min such that the agent receives enough reward signal for learning.

On the other hand, if we only sample from goals for which R g (π i ) > R min , we might sample repeatedly from a small set of already mastered goals.

To force our policy to train on goals that still need improvement, we train on the set of goals g for which R g (π i ) ≤ R max , where R max is a hyperparameter setting a maximum level of performance above which we prefer to concentrate on new goals.

Thus, training our policy on goals in G i allows us to efficiently maximize the coverage objective of Eq. (3).

Note that from Eq. (2), R min and R max can be interpreted as a minimum and maximum probability of reaching a goal over T timesteps.

Given a set of goals sampled from some distribution p data (g), we wish to estimate a label y g ∈ {0, 1} for each goal g that indicates whether g ∈ G i .

These labels are obtained based on the policy performance during the policy update step (Sec. 4.3); see Appendix C for details on this procedure.

In the next section we describe how we can generate more goals that also belong to G i , in addition to the goals that we have labeled.

In order to sample new goals g uniformly from G i , we introduce an adversarial training procedure called "goal GAN", which is a modification of the procedure used for training Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) .

The modification allows us to train the generative model both with positive examples from the distribution we want to approximate and negative examples sampled from a distribution that does not share support with the desired one.

This improves the accuracy of the generative model despite being trained with very few positive samples.

Our choice of GANs for goal generation was motivated both from this potential to train from negative examples as well as their ability to generate very high dimensional samples such as images (Goodfellow et al., 2014) which is important for scaling up our approach to goal generation in high-dimensional goal spaces.

Other generative models like Stochastic Neural Networks BID20 don't accept negative examples and don't have the potential to scale up to higher dimensions.

In our particular application, we use a "goal generator" neural network G(z) to generate goals g from a noise vector z. We train the goal generator to uniformly output goals in G i using a second "goal discriminator" network D(g).

The latter is trained to distinguish goals that are in G i from goals that are not in G i .

We optimize our G(z) and D(g) in a manner similar to that of the LeastSquares GAN (LSGAN) (Mao et al., 2016 ), which we modify by introducing the binary label y g to indicate whether g ∈ G i (allowing us to train from "negative examples" when y g = 0): DISPLAYFORM0 We directly use the original hyperparameters reported in Mao et al. (2016) in all our experiments (a = -1, b = 1, and c = 0).

The LSGAN approach gives us a considerable improvement in training stability over vanilla GAN, and it has a comparable performance to WGAN BID2 .

However, unlike in the original LSGAN paper (Mao et al., 2016) , we have three terms in our value function V (D) rather than the original two.

For goals g for which y g = 1, the second term disappears and we are left with only the first and third terms, which are identical to that of the original LSGAN framework.

Viewed in this manner, the discriminator is trained to discriminate between goals from p data (g) with a label y g = 1 and the generated goals G(z).

Looking at the second term, our discriminator is also trained with "negative examples," i.e. goals with a label y g = 0 which our generator should not generate.

The generator is trained to "fool" the discriminator, i.e. to output goals that match the distribution of goals in p data (g) for which y g = 1.

Algorithm 1: Generative Goal Learning Input : DISPLAYFORM0 Our full algorithm for training a policy π(a t | s t , g) to maximize the coverage objective in Eq. FORMULA4 is shown in Algorithm 1.

At each iteration i, we generate a set of goals by first using sample noise to obtain a noise vector z from p z (·) and then passing this noise to the generator G.We use these goals to train our policy using RL, with the reward function given by Eq. (1) (update policy).

The training can be done with any RL algorithm; in our case we use TRPO (Schulman et al., 2015a) with GAE BID14 .

Our policy's emperical performance on these goals (evaluate policy) is used to determine each goal's label y g (label goals), as described in Section 4.1.

Next, we use these labels to train our goal generator and our goal discriminator (train GAN), as described in Section 4.2.

The generated goals from the previous iteration are used to compute the Monte Carlo estimate of the expectations with respect to the distribution p data (g) in Eq. (4).

By training on goals within G i produced by the goal generator, our method efficiently finds a policy that optimizes the coverage objective.

For details on how we initialize the goal GAN (initialize GAN), and how we use a replay buffer to prevent "catastrophic forgetting" (update replay), see Appendix A.The algorithm described above naturally creates a curriculum for our policy.

The goal generator generates goals in G i , for which our policy obtains an intermediate level of return, and thus such goals are at the appropriate level of difficulty for our current policy π i .

As our policy improves, the generator learns to generate goals in order of increasing difficulty.

Hence, our method can be viewed as a way to automatically generate a curriculum of goals.

However, the curriculum occurs as a by-product via our optimization, without requiring any prior knowledge of the environment or the tasks that the agent must perform.

In this section we provide the experimental results to answer the following questions:• Does our automatic curriculum yield faster maximization of the coverage objective?• Does our Goal GAN dynamically shift to sample goals of the appropriate difficulty?• Does it scale to a higher-dimensional state-space with a low-dimensional space of feasible goals?To answer the first two questions, we demonstrate our method in two challenging robotic locomotion tasks, where the goals are the (x, y) position of the Center of Mass (CoM) of a dynamically complex quadruped agent.

In the first experiment the agent has no constraints and in the second one the agent is inside a U-maze.

To answer the third question, we study how our method scales with the dimension of the state-space in an environment where the feasible region is kept of approximately constant volume in an embedding space that grows in dimension.

We compare our Goal GAN method against four baselines.

Uniform Sampling is a method that does not use a curriculum at all, training at every iteration on goals uniformly sampled from the goal-space.

To demonstrate that a straight-forward distance reward can be prone to local minima, Uniform Sampling with L2 loss samples goals in the same fashion as the first baseline, but instead of the indicator reward that our method uses, it receives the negative L2 distance to the goal as a reward at every step.

We have also adapted two methods from the literature to our setting: Asymmetric Selfplay BID17 and SAGG-RIAC BID6 .

Finally, we provide an ablation and an oracle for our method to better understand the importance of sampling "good" goals.

The ablation GAN fit all consists on not training the GAN only on the "good" goals but rather on every goal attempted in the previous iteration.

Given the noise injected at the output of the GAN this generates a gradually expanding set of goals -similar to any hand-designed curriculum.

The oracle consists in sampling goals uniformly from the feasible state-space, but only keeping them if they satisfy the criterion defined in Section 4.1.

This Rejection Sampling method is orders of magnitude more expensive in terms of labeling, but it serves to estimate an upper-bound for our method in terms of performance.

We test our method in two challenging environments of a complex robotic agent navigating either a free space (Free Ant) or a U-shaped maze (Maze Ant).

The latter is depicted in Fig. 1 , where the orange quadruped is the Ant, and a possible goal to reach is drawn in red.

describe the task of trying to reach the other end of the U-turn and they show that standard RL methods are unable to solve it.

We further extend the task to ask to be able to reach any given point within the maze, or the [−5, 5] 2 square for Free Ant.

The reward is still a sparse indicator function which takes the value 1 only when the (x, y) CoM of the Ant is within = 0.5 of the goal.

Therefore the goal space is 2 dimensional, the statespace is 41 dimensional and the action space is 8 dimensional (see details in Appendix B.1).We first explore whether, by training on goals that are generated by our Goal GAN, we are able to improve our policy's training efficiency, compared to the baselines described above.

In Figs. 2a- FIG0 we see that our method leads to faster training compared to the baselines.

The Uniform Sampling baseline does very poorly because too many samples are wasted attempting to train on goals that are infeasible or not reachable by the current policy -hence not receiving any learning signal.

If an L2 loss is added to try to guide the learning, the agent falls into a poor local optima of not moving to avoid further negative rewards.

The two other baselines that we compare against perform better, but still do not surpass the performance of our method.

In particular, Asymmetric Self-play needs to train the goal-generating policy (Alice) at every outer iteration, with an amount of rollouts equivalent to the ones used to train the goal-reaching policy.

This additional burden is not represented in the plots, being therefore at least half as sample-efficient as the plots indicate.

SAGG-RIAC maintains an ever-growing partition of the goal-space that becomes more and more biased towards areas that already have more sub-regions, leading to reduced exploration and slowing down the expansion of the policy's capabilities.

Details of our adaptation of these two methods to our problem, as well as further study of their failure cases, is provided in the Appendices F.1 and F.2.To better understand the efficiency of our method, we analyze the goals generated by our automatic curriculum.

In these Ant navigation experiments, the goal space is two dimensional, allowing us to study the shift in the probability distribution generated by the Goal GAN (Fig. 3) along with the improvement of the policy coverage (Fig. 4) .

We have indicated the difficulty of reaching the generated goals in Fig. 3 .

It can be observed in these figures that the location of the generated goals shifts to different parts of the maze, concentrating on the area where the current policy is receiving some learning signal but needs more improvement.

The percentage of generated goals that are at the appropriate level of difficulty ("good goals") stays around 20% even as the policy improves.

The goals in these figures include a mix of newly generated goals from the Goal GAN as well as goals from previous iterations that we use to prevent our policy from "forgetting" (Appendix A.1).

Overall it is clear that our Goal GAN dynamically shift to sample goals of the appropriate difficulty.

See Appendix D for additional experiments.

It is interesting to analyze the importance of generating "good goals" for efficient learning.

This is done in FIG0 , where we first show an ablation of our method GAN fit all, that disregards the labels.

This method performs worse than ours, because the expansion of the goals is not related to the current performance of the policy.

Finally, we study the Rejection Sampling oracle.

As explained in Section 4.1, we wish to sample from the set of "good" goals G i , which we approximate by fitting a Goal GAN to the distribution of good goals observed in the previous policy optimization step.

We evaluate now how much this approximation affects learning by comparing the learning performance of our Goal GAN to a policy trained on goals sampled uniformly from G i by using rejection sampling.

This method is orders of magnitude more sample inefficient, but gives us an upper bound on the performance of our method.

Figs. 2c-2d demonstrate that our performance is quite close to the performance of this much less efficient baseline.

In most real-world RL problems, the set of feasible states is a lower-dimensional subset of the full state space, defined by the constraints of the environment.

For example, the kinematic constraints of a robot limit the set of feasible states that the robot can reach.

Therefore, uniformly sampling goals from the full state-space would yield very few achievable goals.

In this section we use an N-dimensional Point Mass to explore this issue and demonstrate the performance of our method as the embedding dimension increases.

In our experiments, the full state-space of the N -dimensional Point Mass is the hypercube [−5, 5] N .

However, the Point Mass can only move within a small subset of this state space.

In the two-dimensional case, the set of feasible states corresponds to the [−5, 5] × [−1, 1] rectangle, making up 20% of the full space.

For N > 2, the feasible space is the Cartesian product of this 2D strip with [− , ] N −2 , where = 0.3.

In this higher-dimensional environment, our agent receives a reward of 1 when it moves within DISPLAYFORM0 of the goal state, to account for the increase in average L2 distance between points in higher dimensions.

The ratio of the volume of the embedded space to the volume of the full state space decreases as N increases, down to 0.00023:1 for 6 dimensions.

FIG2 shows the performance of our method compared to the other methods, as the number of dimensions increases.

The uniform sampling baseline has very poor performance as the number of dimensions increases because the fraction of feasible states within the full state space decreases as the dimension increases.

Thus, sampling uniformly results in sampling an increasing percentage of unfeasible states, leading to poor learning signal.

In contrast, the performance of our method does not decay as much as the state space dimension increases, because our Goal GAN always generates goals within the feasible portion of the state space (and at the appropriate level of difficulty).

The GAN fit all variation of our method suffers from the increase in dimension because it is not encouraged to track the narrow feasible region.

Finally, the oracle and the baseline with an L2 distance reward have perfect performance, which is expected in this simple task where the optimal policy is just to go in a straight line towards the goal.

Even without this prior knowledge, the Goal GAN discovers the feasible subset of the goal space.

We propose a new paradigm in RL where the objective is to train a single policy to succeed on a variety of goals, under sparse rewards.

To solve this problem we develop a method for automatic curriculum generation that dynamically adapts to the current performance of the agent.

The curriculum is obtained without any prior knowledge of the environment or of the tasks being performed.

We use generative adversarial training to automatically generate goals for our policy that are always at the appropriate level of difficulty (i.e. not too hard and not too easy).

In the future we want to combine our goal-proposing strategy with recent multi-goal approaches like HER BID1 ) that could greatly benefit from better ways to select the next goal to train on.

Another promising line of research is to build hierarchy on top of the multi-task policy that we obtain with our method by training a higher-level policy that outputs the goal for the lower level multi-task policy (like in Heess et al. (2016 ) or in Florensa et al. (2017a ).

The hierarchy could also be introduced by replacing our current feed-forward neural network policy by an architecture that learns to build implicit plans (Mnih et al., 2016; BID18 , or by leveraging expert demonstrations to extract sub-goals BID23 , although none of these approaches tackles yet the multi-task learning problem formulated in this work.

In addition to training our policy on the goals that were generated in the current iteration, we also save a list ("regularized replay buffer") of goals that were generated during previous iterations (update replay).

These goals are also used to train our policy, so that our policy does not forget how to achieve goals that it has previously learned.

When we generate goals for our policy to train on, we sample two thirds of the goals from the Goal GAN and we sample the one third of the goals uniformly from the replay buffer.

To prevent the replay buffer from concentrating in a small portion of goal space, we only insert new goals that are further away than from the goals already in the buffer, where we chose the goal-space metric and to be the same as the ones introduced in Section 3.1.

In order to begin our training procedure, we need to initialize our goal generator to produce an initial set of goals (initialize GAN).

If we initialize the goal generator randomly (or if we initialize it to sample uniformly from the goal space), it is likely that, for most (or all) of the sampled goals, our initial policy would receives no reward due to the sparsity of the reward function.

Thus we might have that all of our initial goals g haveR g (π 0 ) < R min , leading to very slow training.

To avoid this problem, we initialize our goal generator to output a set of goals that our initial policy is likely to be able to achieve withR g (π i ) ≥ R min .

To accomplish this, we run our initial policy π 0 (a t | s t , g) with goals sampled uniformly from the goal space.

We then observe the set of states S v that are visited by our initial policy.

These are states that can be easily achieved with the initial policy, π 0 , so the goals corresponding to such states will likely be contained within S I 0 .

We then train the goal generator to produce goals that match the state-visitation distribution p v (g), defined as the uniform distribution over the set f (S v ).

We can achieve this through traditional GAN training, with p data (g) = p v (g).

This initialization of the generator allows us to bootstrap the Goal GAN training process, and our policy is able to quickly improve its performance.

The ant is a quadruped with 8 actuated joints, 2 for each leg.

The environment is implemented in Mujoco BID21 .

Besides the coordinates of the center of mass, the joint angles and joint velocities are also included in the observation of the agent.

The high degrees of freedom make navigation a quite complex task requiring motor coordination.

More details can be found in , and the only difference is that in our goal-oriented version of the Ant we append the observation with the goal, the vector from the CoM to the goal and the distance to the goal.

For the Free Ant experiments the objective is to reach any point in the square [−5m, 5m] 2 on command.

The maximum time-steps given to reach the current goal are 500.

The agent is constrained to move within the maze environment, which has dimensions of 6m x 6m.

The full state-space has an area of size 10 m x 10 m, within which the maze is centered.

To compute the coverage objective, goals are sampled from within the maze according to a uniform grid on the maze interior.

The maximum time-steps given to reach the current goal are 500.

For the N-dim point mass of Section 5.2, in each episode (rollout) the point-mass has 400 timesteps to reach the goal, where each timestep is 0.02 seconds.

The agent can accelerate in up to a rate of 5 m/s 2 in each dimension (N = 2 for the maze).

The observations of the agent are 2N dimensional, including position and velocity of the point-mass.

After the generator generates goals, we add noise to each dimension of the goal sampled from a normal distribution with zero mean and unit variance.

At each step of the algorithm, we train the policy for 5 iterations, each of which consists of 100 episodes.

After 5 policy iterations, we then train the GAN for 200 iterations, each of which consists of 1 iteration of training the discriminator and 1 iteration of training the generator.

The generator receives as input 4 dimensional noise sampled from the standard normal distribution.

The goal generator consists of two hidden layers with 128 nodes, and the goal discriminator consists of two hidden layers with 256 nodes, with relu nonlinearities.

The policy is defined by a neural network which receives as input the goal appended to the agent observations described above.

The inputs are sent to two hidden layers of size 32 with tanh nonlinearities.

The final hidden layer is followed by a linear N -dimensional output, corresponding to accelerations in the N dimensions.

For policy optimization, we use a discount factor of 0.998 and a GAE lambda of 0.995.

The policy is trained with TRPO with Generalized Advantage Estimation implemented in rllab (Schulman et al., 2015a; b; .

Every "update policy" consists of 5 iterations of this algorithm.

To label a given goal (Section 4.1), we could empirically estimate the expected return for this goal R g (π i ) by performing rollouts of our current policy π i .

The label for this goal is then set to DISPLAYFORM0 Nevertheless, having to execute additional rollouts just for labeling is not sample efficient.

Therefore, we instead use the rollouts that were used for the most recent policy update.

This is an approximation as the rollouts where performed under π i−1 , but as we show in Figs. 7a-7b, this small "delay" does not affect learning significantly.

Indeed, using the true label (estimated with three new rollouts from π i ) yields the Goal GAN true label curves that are only slightly better than what our method does.

In the same plots we also study another definition of "good" goals that has been previously used in the literature: learning progress BID6 Graves et al., 2017) .

Given that we work in a continuous goal-space, estimating the learning progress of a single goal requires estimating the performance of the policy on that goal before the policy update and after the policy update (potentially being able to replace one of these estimations with the rollouts from the policy optimization, but not both).

Therefore the method does require more samples, but we deemed interesting to compare how well the metric to automatically build a curriculum.

We see in the Figs. 7a-7b that the two metrics yield a very similar learning, at least in the case of Ant navigation tasks with sparse rewards.

Similar to the experiments in Figures 3 and 4 , here we show the goals that were generated for the Free Ant experiment in which a robotic quadruped must learn to move to all points in free space.

FIG5 show the results.

As shown, our method produces a growing circle around the origin; as the policy learns to move the ant to nearby points, the generator learns to generate goals at increasingly distant positions.

Figure 9: Visualization of the policy performance for different parts of the state space (same policy training as in FIG5 ).

For illustration purposes, the feasible state-space is divided into a grid, and a goal location is selected from the center of each grid cell.

Each grid cell is colored according to the expected return achieved on this goal: Red indicates 100% success; blue indicates 0% success.

In this section we show that our Goal GAN method is efficient at tracking clearly multi-modal distributions of good goals.

To this end, we introduce a new maze environment with multiple paths, as can be seen in FIG6 .

To keep the experiment simple we replace the Ant agent by a point-mass environment (in orange), which actions are directly the velocity vector (2 dim).

As in the other experiments, our aim is to learn a policy that can reach any feasible goal corresponding to -balls in state space like the one depicted in red.

Similar to the experiments in Figures 3 and 4 , here we show the goals that were generated for the Mutli-path point-mass maze experiment.

FIG0 show the results.

It can be observed that our method produces a multi-modal distribution over goals, tracking all the areas where goals are at the appropriate level of difficulty.

Note that the samples from the regularized replay buffer are responsible for the trailing spread of "High Reward" goals and the Goal GAN is responsible for the more concentrated nodes, as can be seen in Fig. 13 .

A clear benefit of using our Goal GAN as a generative model is that no prior knowledge about the distribution to fit is required (like the number of modes).

Finally, note that the fact of having several possible paths to reach a specific goal does not hinder the learning of our algorithm that consistently reaches full coverage in this problem as seen in Fig. 14.

FIG5 ).

For illustration purposes, the feasible state-space is divided into a grid, and a goal location is selected from the center of each grid cell.

Each grid cell is colored according to the expected return achieved on this goal: Red indicates 100% success; blue indicates 0% success.

Although not specifically designed for the problem presented in this paper, it is straight forward to apply the method proposed by BID17 to our problem.

An interesting study of its limitations in a similar setting can be found in FIG0 ).

In our implementation of this method, we use TRPO as the "Low-Level Goal-Directed Exploration with Evolving Context".

We therefore implement the method as batch: at every iteration, we sample N new new goals {y i } i=0...Nnew , then we collect rollouts of t max steps trying to reach them, and perform the optimization of the parameters using all the collected data.

The detailed algorithm is given in the following pseudo-code.

while number steps in(paths) < batch size do Reset s 0 ← s rest ; y g ← Uniform(goals); y f , Γ yg , path ← collect rollout(π θi (·, y g ), s reset ); paths.append(path); UpdateRegions(R, y f , 0) ; UpdateRegions(R, y g , Γ yg ); end π θi+1 ← train π θi with TRPO on collected paths; end UpdateRegions(R, y f , Γ y f ) is exactly the Algorithm 2 described in the original paper, and Selfgenerate is the "Active Goal Self-Generation (high-level)" also described in the paper (Section 2.4.4 and Algorithm 1), but it's repeated N new times to produce a batch of N new goals jointly.

As for the competence Γ yg , we use the same formula as in their section 2.4.1 (use highest competence if reached close enough to the goal) and C(y g , y f ) is computed with their equation (7) .

The collect rollout function resets the state s 0 = s reset and then applies actions following the goal-conditioned policy π θ (·, y g ) until it reaches the goal or the maximum number of steps t max has been taken.

The final state, transformed in goal space, y f is returned.

As hyperparameters, we have used the recommended ones in the paper, when available: p 1 = 0.7, p 2 = 0.2, p 3 = 0.1.

For the rest, the best performance in an hyperparameter sweep yields: ζ = 100, g max = 100.

The noise for mode(3) is chosen to be Gaussian with variance 0.1, the same as the tolerance threshold max and the competence threshold C .As other details, in our tasks there are no constraints to penalize for, so ρ = ∅. Also, there are no sub-goals.

The reset value r is 1 as we reset to s start after every reaching attempt.

The number of explorative movements q ∈ N has a less clear equivalence as we use a policy gradient update with a stochastic policy π θ instead of a SSA-type algorithm.

@highlight

We efficiently solve multi-task problems with an automatic curriculum generation algorithm based on a generative model that tracks the learning agent's performance.
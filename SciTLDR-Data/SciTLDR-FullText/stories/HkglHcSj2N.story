Designing rewards for Reinforcement Learning (RL) is challenging because it needs to convey the desired task, be efficient to optimize, and be easy to compute.

The latter is particularly problematic when applying RL to robotics, where detecting whether the desired configuration is reached might require considerable supervision and instrumentation.

Furthermore, we are often interested in being able to reach a wide range of configurations, hence setting up a different reward every time might be unpractical.

Methods like Hindsight Experience Replay (HER) have recently shown promise to learn policies able to reach many goals, without the need of a reward.

Unfortunately, without tricks like resetting to points along the trajectory, HER might take a very long time to discover how to reach certain areas of the state-space.

In this work we investigate different approaches to incorporate demonstrations to drastically speed up the convergence to a policy able to reach any goal, also surpassing the performance of an agent trained with other Imitation Learning algorithms.

Furthermore, our method can be used when only trajectories without expert actions are available, which can leverage kinestetic or third person demonstration.

Reinforcement Learning (RL) has shown impressive results in a plethora of simulated tasks, ranging from attaining super-human performance in video-games BID18 BID35 and board-games (Silver et al., 2017) , to learning complex locomotion behaviors BID34 BID4 .

Nevertheless, these successes are shyly echoed in real world robotics (Riedmiller et BID36 .

This is due to the difficulty of setting up the same learning environment that is enjoyed in simulation.

One of the critical assumptions that are hard to obtain in the real world are the access to a reward function.

Self-supervised methods have the power to overcome this limitation.

A very versatile and reusable form of self-supervision for robotics is to learn how to reach any previously observed state upon demand.

This problem can be formulated as training a goal-conditioned policy BID14 BID27 that seeks to obtain the indicator reward of having the observation exactly match the goal.

Such a reward does not require any additional instrumentation of the environment beyond the sensors the robot already has.

But in practice, this reward is never observed because in continuous spaces like the ones in robotics, the exact same observation is never observed twice.

Luckily, if we are using an off-policy RL algorithm BID17 BID11 , we can "relabel" a collected trajectory by replacing its goal by a state actually visited during that trajectory, therefore observing the indicator reward as often as we wish.

This method was introduced as Hindsight Experience Replay BID0 or HER.In theory these approaches could learn how to reach any goal, but the breadth-first nature of the algorithm makes that some areas of the space take a long time to be learned BID7 .

This is specially challenging when there are bottlenecks between different areas of the statespace, and random motion might not traverse them easily BID5 .

Some practical examples of this are pick-and-place, or navigating narrow corridors between rooms, as illustrated in Fig. 5 in appendix depicting the diverse set of environments we work with.

In both cases a specific state needs to be reached (grasp the object, or enter the corridor) before a whole new area of the space is discovered (placing the object, or visiting the next room).

This problem could be addressed by engineering a reward that guides the agent towards the bottlenecks, but this defeats the purpose of trying to learn without direct reward supervision.

In this work we study how to leverage a few demonstrations that traverse those bottlenecks to boost the learning of goal-reaching policies.

Learning from Demonstrations, or Imitation Learning (IL), is a well-studied field in robotics BID15 BID25 BID2 .

In many cases it is easier to obtain a few demonstrations from an expert than to provide a good reward that describes the task.

Most of the previous work on IL is centered around trajectory following, or doing a single task.

Furthermore it is limited by the performance of the demonstrations, or relies on engineered rewards to improve upon them.

In this work we study how IL methods can be extended to the goal-conditioned setting, and show that combined with techniques like HER it can outperform the demonstrator without the need of any additional reward.

We also investigate how the different methods degrade when the trajectories of the expert become less optimal, or less abundant.

Finally, the method we develop is able to leverage demonstrations that do not include the expert actions.

This is very convenient in practical robotics where demonstrations might have been given by a motion planner, by kinestetic demonstrations (moving the agent externally, and not by actually actuating it), or even by another agent.

To our knowledge, this is the first framework that can boost goal-conditioned policy learning with only state demonstrations.

We define a discrete-time finite-horizon discounted Markov decision process (MDP) by a tuple M = (S, A, P, r, ρ 0 , γ, H), where S is a state set, A is an action set, P : S × A × S → R + is a transition probability distribution, γ ∈ [0, 1] is a discount factor, and H is the horizon.

Our objective is to find a stochastic policy π θ that maximizes the expected discounted reward within the DISPLAYFORM0 We denote by τ = (s 0 , a 0 , ..., ) the entire state-action trajectory, where s 0 ∼ ρ 0 (s 0 ), a t ∼ π θ (a t |s t ), and s t+1 ∼ P(s t+1 |s t , a t ).

In the goal-conditioned setting that we use here, the policy and the reward are also conditioned on a "goal" g ∈ S. The reward is r(s t , a t , s t+1 , g) = 1 s t+1 == g , and hence the return is the γ h , where h is the number of time-steps to the goal.

Given that the transition probability is not affected by the goal, g can be "relabeled" in hindsight, so a transition (s t , a t , s t+1 , g, r = 0) can be treated as (s t , a t , s t+1 , g = s t+1 , r = 1).

Finally, we also assume access to D trajectories (s DISPLAYFORM1 that were collected by an expert attempting to reach a goal g j sampled uniformly among the feasible goals.

Those trajectories must be approximately geodesics, meaning that the actions are taken such that the goal is reached as fast as possible.

In this section we describe the different algorithms we compare to pure Hindsight Experience Replay BID0 .

See the Appendix to prior work on adding a Be- havioral Cloning loss to the policy update as in BID19 .

Here we propose a novel expert relabeling technique, we formulate for the first time a goal-conditioned GAIL algorithm, and propose a method to train it with state-only demonstrations.

The expert trajectories are collected by asking the expert to reach a specific goal g j .

But they are also valid trajectories to reach any other state visited within the demonstration!

This is the key motivating insight to propose a new type of relabeling: if we have the transitions (s DISPLAYFORM0 t+k ) as also coming from the expert!

This can be understood as a type of data augmentation leveraging the assumption that the tasks we work on are quasi-static.

It will be particularly effective when not many demonstrations are available.

In FIG2 we compare the final performance of two agents for Four Rooms environment, one trained with pure Behavioral Cloning, and the other one also using expert relabeling.

The compounding error in Behavioral Cloning might make the policy deviate arbitrarily from the demonstrations, and it requires too many demonstrations when the state dimension increases.

The first problem is less severe in our goalconditioned case because in fact we do want to visit and be able to purposefully reach all states, even the ones that the expert did not visited.

But the second drawback will become pressing when attempting to scale this method to practical robotics tasks where the observations might be high-dimensional sensory input like images.

Both problems can be mitigated by using other Imitation Learning algorithms that can leverage additional rollouts collected by the learning agent in a self-supervised manner, like GAIL BID13 .

In this section we extend the formulation of GAIL to tackle goal-conditioned tasks, and then we detail how it can be combined with HER BID0 , which allows to outperform the demonstrator and generalize to all goals.

We call this algorithm goal-GAIL.First of all, the discriminator needs to also be conditioned on the goal D ψ (a, s, g).

Once the discriminator is fitted, we can run our favorite RL algorithm on the DISPLAYFORM0 In our case we used the offpolicy algorithm DDPG BID17 to allow for the relabeling techniques outlined above.

In the goalconditioned case we also supplement with the indicator reward r DISPLAYFORM1 h .

This combination is slightly tricky because now the fitted Q φ does not have the same clear interpretation it has when only one of the two rewards is used BID6 .

Nevertheless, both rewards are pushing the policy towards the goals, so it shouldn't be too conflicting.

Furthermore, to avoid any drop in final performance, the weight of the reward coming from GAIL (δ GAIL ) can be annealed.

See Appendix for details.

Both Behavioral Cloning and GAIL use state-action pairs from the expert.

This limits the use of the methods, combined or not with HER, to setups where the exact same agent was actuated to reach different goals.

Nevertheless, much more data could be cheaply available if the action was not required.

For example, kinestetic demonstration or third-person imitation (Stadie et al., 2017) .

The main insight we have here is that we can replace the action in the GAIL formulation by the next state s , and in most environments this should be as informative as having access to the action directly.

Intuitively, given a desired goal g, it should be possible to determine if a transition s → s is taking the agent in the right direction.

The loss function to train a discriminator able to tell apart the current agent and demonstrations (always transitioning towards the goal) is simply: DISPLAYFORM0

We are interested in answering the following questions:1.

Can the use of demonstrations accelerate the learning of goal-conditioned tasks without reward?

2.

Is the Expert Relabeling an efficient way of doing dataaugmentation on the demonstrations?

We evaluate these questions in two different simulated robotic goal-conditioned tasks that are detailed in the next subsection.

All the results use 20 demonstrations.

All curves have 5 random seeds and the shaded area is one standard deviation

Experiments are conducted in two continuous environments in MuJoCo BID33 .

The performance metric we use in all our experiments is the percentage of goals in the feasible goal space the agent is able to reach.

A point mass is placed in an environment with four rooms connected through small openings.

The action space is continuous and specifies the desired change in state space which corresponds to the goal space.

Pick and Place: A fetch robot needs to pick a block and place it in a desired point in space as described in BID19 .

The control is four-dimensional, corresponding to a change in position of the end-effector and a change in gripper opening.

The goal space is the position of the block.

In goal-conditioned tasks, HER BID0 should eventually converge to a policy able to reach any desired goal.

Nevertheless, this might take a long time, specially in environments where there are bottlenecks that need to be traversed before accessing a whole new area of the goal space.

In this section we show how the methods introduced in the previous section can leverage a few demonstrations to improve the convergence speed of HER.

This was already studied for the case of Behavioral Cloning by BID19 , and in this work we show we also get a benefit when using GAIL as the Imitation Learning algorithm.

In both environments, we observe that running GAIL with relabeling (GAIL+HER) considerably outperforms running each of them in isolation.

HER alone has a very slow convergence, although as expected it ends up reaching the same final performance if run long enough.

On the other hand GAIL by itself learns fast at the beginning, but its final performance is capped.

This is because despite collecting more samples on the environment, those come with no reward of any kind indicating what is the task to perform (reach the given goals).

Therefore, once it has extracted all the information it can from the demonstrations it cannot keep learning and generalize to goals further from the demonstrations.

This is not an issue anymore when combined with HER, as our results show.

Here we show that the Expert Relabeling technique introduced in Section 3.1 is beneficial in the goal-conditioned imitation learning framework.

As shown in FIG4 , our expert relabeling technique brings considerable performance boosts for both Behavioral Cloning methods and goal-GAIL in both environments.

We also perform a further analysis of expert relabeling in the four-rooms environment.

We see in FIG2 that without the expert relabeling, the agent fails to learn how to reach many intermediate states visited in the middle of a demonstration.

Behavioral Cloning and standard GAIL rely on the stateaction (s, a) tuples from the expert.

Nevertheless there are many cases in robotics where we only have access to observation-only demonstrations.

In this section we want to emphasize that all the results obtained with our goal-GAIL method and reported in FIG3 and FIG4 do not require actions that the expert took.

Surprisingly, in the four rooms environment, despite the more restricted information goal-GAIL has access to, it outperforms BC combined with HER.

This might be due to the superior imitation learning performance of GAIL, and also to the fact that these tasks might be possible to solve by only matching the state-distribution of the expert.

With GAIL conditioned only on current state but not action (as also done in other non-goal-conditioned works BID8 ), we observe that the discriminator learns a very well shaped reward that encourages the agent to go towards the goal, as pictured in Fig. 6 in appendix.

See the Appendix for more details.

In the above sections we assumed access to optimal experts.

Nevertheless, in practical applications the experts might have a more erratic behavior.

In this section we study how the different methods perform with a sub-optimal expert.

To do so we collect trajectories attempting goals g by modifying our optimal expert π * (a|s, g) in two ways: We add noise α to the optimal actions and make it be -greedy.

The sub-optimal expert is then a = 1[ DISPLAYFORM0 ) and u is a uniformly sampled random action.

In FIG5 we observe that approaches that copy the action of the expert, like Behavioral Cloning, greatly suffer under a sub-optimal expert.

On the other hand, discriminator-based methods are able to leverage noisier experts.

A possible explanation is that a discriminator approach can give a positive signal as long as the transition is "in the right direction", without trying to exactly enforce a single action.

Under this lens, having some noise in the expert might actually improve the performance of these adversarial approaches, as it has been observed in many generative models literature (Goodfellow et al.) .

Hindsight relabeling can be used to learn useful behaviors without any reward supervision for goal-conditioned tasks, but they are inefficient when the state-space is large or includes exploration bottlenecks.

In this work we show how only a few demonstrations can be leveraged to improve the convergence speed of these methods.

We introduce a novel algorithm, goal-GAIL, that converges faster than HER and to a better final performance than a naive goal-conditioned GAIL.

We also study the effect of doing expert relabeling as a type of data augmentation on the provided demonstrations, and demonstrate it improves the performance of our goal-GAIL as well as goal-conditioned Behavioral Cloning.

We emphasize that our goal-GAIL method only needs state demonstrations, without using expert actions like other Behavioral Cloning methods.

Finally, we show that goal-GAIL is robust to sub-optimalities in the expert behavior.

Imitation Learning can be seen as an alternative to reward crafting to train desired behaviors.

There are many ways to leverage demonstrations, from Behavioral Cloning BID22 ) that directly maximizes the likelihood of the expert actions under the training agent policy, to Inverse Reinforcement Learning that extracts a reward function from those demonstrations and then trains a policy to maximize it BID38 BID3 BID8 .

Another formulation close to the later introduced by BID13 is Generative Adversarial Imitation Learning (GAIL), explained in details in the next section.

Originally, the algorithms used to optimize the policy were on-policy methods like Trust Region Policy Optimization BID29 , but recently there has been a wake of works leveraging the efficiency of off-policy algorithms without loss in stability BID1 BID26 BID28 BID16 .

This is a key capability that we are going to exploit later on.

Unfortunately most work in the field cannot outperform the expert, unless another reward is available during training BID34 BID9 BID32 , which might defeat the purpose of using demonstrations in the first place.

Furthermore, most tasks tackled with these methods consist on tracking expert state trajectories BID37 BID21 , but can't adapt to unseen situations.

In this work we are interested in goal-conditioned tasks, where the objective is to be able to reach any state upon demand.

This kind of multi-task learning are pervasive in robotics, but challenging if no reward-shaping is applied.

Relabeling methods like Hindsight Experience Replay BID0 unlock the learning even in the sparse reward case BID6 .

Nevertheless, the inherent breath-first nature of the algorithm might still make very inefficient learning to learn complex policies.

To overcome the exploration issue we investigate the effect of leveraging a few demonstrations.

The closest prior work is by BID19 , where a Behavioral Cloning loss is used with a Q-filter.

We found that a simple annealing of the Behavioral Cloning loss BID23 works better.

Furthermore, we also introduce a new relabeling technique of the expert trajectories that is particularly useful when only few demonstrations are available.

We also experiment with Goal-conditioned GAIL, leveraging the recently shown compatibility with off-policy algorithms.

For a more comprehensive review of related work, please see Appendix.

The most direct way to leverage demonstrations DISPLAYFORM0 is to construct a data-set D of all state-action-goal tuples (s j t , a j t , g j ), and run a supervised regression algorithm.

In the goal-conditioned case and assuming a deterministic policy π θ (s, g), the loss is: DISPLAYFORM1 This loss and its gradient are computed without any additional environments samples from the trained policy π θ .

This makes it particularly convenient to combine a gradient descend step based on this loss with other policy updates.

In particular we can use a standard offpolicy Reinforcement Learning algorithm like DDPG BID17 , where we fit the Q φ (a, s, g), and then estimate the gradient of the expected return as: DISPLAYFORM2 ).

The improvement guarantees with respect to the task reward are lost when we combine the BC and the deterministic policy gradient updates, but this can be side-stepped by either applying a Q-filter: 1 Q(s t , a t , g) > Q(s t , π(s t , g), g) to the BC loss as proposed in BID19 , or by annealing it as we do in our experiments, which allows the agent to eventually outperform the expert.

All possible variants we study are detailed in Algorithm 1 as presented in appendix.

In particular, α = 0 falls back to pure Behavioral Cloning, β = 0 removes the BC component, p = 0 doesn't relabel agent trajectories, δ GAIL = 0 removes the discriminator output from the reward, and EX-PERT RELABEL indicates whether the here explained expert relabeling should be performed.

In the two environments, i.e. Four Rooms environment and Fetch Pick & Place, the task horizons are set to 300 and 100 respectively.

The discount factors are γ = 1 − 1 H .

In all experiments, the Q function, policy and discriminator are paramaterized by fully connected neural networks with two hidden layers of size 256.

DDPG is used for policy optimization and hindsight probability is set to p = 0.8.

The initial value of the behavior cloning loss weight β is set to 0.1 and is annealed by 0.9 per 250 rollouts collected.

The initial value of the discriminator reward weight δ GAIL is set to 0.1.

We found empirically that there is no need to anneal δ GAIL .t , g h ))27: Anneal δ GAIL and β 28: end while For experiments with sub-optimal expert in section 4.5, is set to 0.4 and 0.5, and σ α is set to 1.5 and 0.3 respectively for Four Rooms environment and Fetch Pick & Place.

We trained the discriminator in three settings:• current state and goal: (s, g)• current state, next state and goal: (s, s , g)• current state, action and goal: (s, a, g)We compare the three different setups in Fig. 7 and 8.

@highlight

We tackle goal-conditioned tasks by combining Hindsight Experience Replay and Imitation Learning algorithms, showing faster convergence than the first and higher final performance than the second.
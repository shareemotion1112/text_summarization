Many real world tasks exhibit rich structure that is repeated across different parts of the state space or in time.

In this work we study the possibility of leveraging such repeated structure to speed up and regularize learning.

We start from the KL regularized expected reward objective which introduces an additional component, a default policy.

Instead of relying on a fixed default policy, we learn it from data.

But crucially, we restrict the amount of information the default policy receives, forcing it to learn reusable behaviors that help the policy learn faster.

We formalize this strategy and discuss connections to information bottleneck approaches and to the variational EM algorithm.

We present empirical results in both discrete and continuous action domains and demonstrate that, for certain tasks, learning a default policy alongside the policy can significantly speed up and improve learning.

Please watch the video demonstrating learned experts and default policies on several continuous control tasks ( https://youtu.be/U2qA3llzus8 ).

For many interesting reinforcement learning tasks, good policies exhibit similar behaviors in different contexts, behaviors that need to be modified only slightly or occasionally to account for the specific task at hand or to respond to information becoming available.

For example, a simulated humanoid in navigational tasks is usually required to walk -independently of the specific goal it is aiming for.

Similarly, an agent in a simulated maze tends to primarily move forward with occasional left/right turns at intersections.

This intuition has been explored across multiple fields, from cognitive science (e.g. BID22 to neuroscience and machine learning.

For instance, the idea of bounded rationality (e.g. BID46 ) emphasizes the cost of information processing and the presence of internal computational constraints.

This implies that the behavior of an agent minimizes the need to process information, and more generally trades off task reward with computational effort, resulting in structured repetitive patterns.

Computationally, these ideas can be modeled using tools from information and probability theory (e.g. BID50 BID32 BID47 BID40 BID33 BID49 , for instance, via constraints on the channel capacity between past states and future actions in a Markov decision process.

In this paper we explore this idea, starting from the KL regularized expected reward objective (e.g. BID51 BID52 BID19 BID36 BID23 BID48 , which encourages an agent to trade off expected reward against deviations from a prior or default distribution over trajectories.

We explore how this can be used to inject subjective knowledge into the learning problem by using an informative default policy that is learned alongside the agent policy This default policy encodes default behaviours that should be executed in multiple contexts in absence of addi-tional task information and the objective forces the learned policy to be structured in alignment with the default policy.

To render this approach effective, we introduce an information asymmetry between the default and agent policies, preventing the default policy from accessing certain information in the state.

This prevents the default policy from collapsing to the agent's policy.

Instead, the default policy is forced to generalize across a subset of states, implementing a form of default behavior that is valid in the absence of the missing information, and thereby exerting pressure that encourages sharing of behavior across different parts of the state space.

FIG0 illustrates the proposed setup, with asymmetry imposed by hiding parts of the state from the default policy.

We investigate the proposed approach empirically on a variety of challenging problems including both continuous action problems such as controlling simulated high-dimensional physical embodied agents, as well as discrete action visual mazes.

We find that even when the agent and default policies are learned at the same time, significant speed-ups can be achieved on a range of tasks.

We consider several variations of the formulation, and discuss its connection to several ideas in the wider literature, including information bottleneck, and variational formulations of the EM algorithm for learning generative models.

Throughout this paper we use s t and a t to denote the state and action at time step t, and rps, aq the instantaneous reward for the agent if it executes action a in state s.

We denote the history up to time t by x t " ps 1 , a 1 , . . .

, s t q, and the whole trajectory by τ " ps 1 , a 1 , s 2 , . .

.q.

Our starting point is the KL regularized expected reward objective Lpπ, π 0 q " E πτ "ř t γ t rps t , a t q´αγ t KL " πpa t |x t q}π 0 pa t |x t q ‰‰ ,where π is the agent policy (parameterized by θ and to be learned), π 0 the default policy, and E πτ r¨s is taken with respect to the distribution π τ over trajectories defined by the agent policy and system dynamics: π τ pτ q " pps 1 q ś t πpa t |x t qpps t`1 |s t , a t q. Note that our policies are history-dependent.

KLrπpa t |x t q}π 0 pa t |x t qs is the Kullback-Leibler (KL) divergence between the agent policy π and a default or prior policy π 0 given history x t .

The discount factor is γ P r0, 1s and α is a hyperparameter scaling the relative contributions of both terms.

Intuitively, this objective expresses the desire to maximize the reward while also staying close to a reference behaviour defined by π 0 .

As discussed later, besides being a convenient way to express a regularized RL problem, it also has deep connections to probabilistic inference.

One particular instantiation of eq. FORMULA0 is when π 0 is the uniform distribution (assuming a compact action space).

In this case one recovers, up to a constant, the entropy regularized objective (e.g. BID55 BID8 BID11 BID44 BID13 : DISPLAYFORM0 This objective has been motivated in various ways: it prevents the policy from collapsing to a deterministic solution thus improving exploration, it encourages learning of multiple solutions to a task which can facilitate transfer, and it provides robustness to perturbations and model mismatch.

One approximation of the entropy regularized objective is for the history dependent entropy to be used as an additional (auxiliary) loss to the RL loss; this approach is widely used in the literature (e.g. BID53 .

While the motivations for considering the entropy regularized objective are intuitive and reasonable, the choice of regularizing towards an uniform policy is less obvious, particularly in cases with large or high dimensional action spaces.

In this work we explore whether regularization towards more sophisticated default policies can be advantageous.

Both objectives (1) and (2) can be generalized beyond the typical Markov assumption in MDPs.

In particular, additional correlations among actions can be introduced, e.g. using latent variables BID13 .

This can be useful when, as discussed below, either π 0 or π are not given full access to the state, rendering the setup partially observed.

In the following we will not explore such extensions, though note that we do work with policies πpa t |x t q and π 0 pa t |x t q that depend on history x t .

Many works that consider the KL regularized objective either employ a simple or fixed default policy or directly work with the entropy formulation (e.g. BID40 BID8 BID11 BID13 .

In contrast, here we will be studying the possibility of learning the default policy itself, and the form of the subjective knowledge that this introduces to the learning system.

Our guiding intuition, as described earlier, is the notion of a default behaviour that is executed in the absence of additional goal-directed information.

Instances which we explore in this paper include a locomotive body navigating to a goal location where the locomotion pattern depends largely on the body configuration and less so on the goal, and a 3D visual maze environment with discrete actions, where the typical action includes forward motion, regardless of the specific task at hand.

To express the notion of a default behavior, which we also refer to as "goal-agnostic" (although the term should be understood very broadly), we consider the case where the default policy π 0 is a function (parameterized by φ) of a subset of the interaction history up to time t, i.e. π 0 pa t |x t q " π 0 pa t |x D t q, where x D t is a subset of the full history x t and is the goal-agnostic information that we allow the default policy to depend on.

We denote by x G t the other (goal-directed) information in x t and assume that the full history is the disjoint union of both.

The objective (1) specializes to: DISPLAYFORM0 To By hiding information from the default policy, the system forces the default policy to learn the average behaviour over histories x t with the same value of x D t .

If x D t hides goal-directed information, the default policy will learn behaviour that is generally useful regardless of the current goal.

We can make this precise by noting that optimizing the objective (1) with respect to π 0 amounts to supervised learning of π 0 on trajectories generated by π τ , i.e. this is a distillation process from π τ to π 0 BID17 BID41 BID34 BID48 .

In the nonparametric case, the optimal default policy π 0 can be derived as: DISPLAYFORM1 where π τ pxtq is the probability of seeing historyxt at time stept under the policy π, and the indicator DISPLAYFORM2 q is 1 if the goal-agnostic information of the two histories matches and 0 otherwise.

It is also worth considering the effect of the objective eq. (3) on the learned policy π.

Since π 0 is learned alongside π and not specified in advance, this objective does not favor any particular behavior a priori.

Instead it will encourage a solution in which similar behavior will be executed in different parts of the state space that are similar as determined by x D t , since the policy π is regularized towards the default policy π 0 .

More generally, during optimization of π the default policy effectively acts like a shaping reward while the entropy contained in the KL discourages deterministic solutions.

Reinforcement learning objectives with information theoretic constraints have been considered by multiple authors BID50 BID47 BID49 .

Such constraints can be motivated by the internal computational limitations of the agent, which limit the rate with which information can be extracted from states (or observations) and translated into actions.

Such capacity constraints can be expressed via an information theoretic regularization term that is added to the expected reward.

Specializing to our scenario, where the "information flow" to be controlled is between the goal-directed history information x G t and action a t (so that the agent prefers default, goal-agnostic, behaviour), consider the objective: DISPLAYFORM0 qs is positive (see BID1 .

Re-introducing this into (5) we find that the KL regularized objective in eq. (3) can be seen as a lower bound to eq. (5), where the agent has a capacity constraint on the channel between goal-directed history information and (future) actions.

See section A in the appendix for a generalization including latent variables.

In this light, we can see our work as a particular implementation of the information bottleneck principle, where we penalize the dependence on the information that is hidden from the default policy.

The above setup also bears significant similarity to the training of variational autoencoders BID20 BID37 and, more generally the variational EM framework for learning latent variable models BID6 BID30 .

The setup is as follows.

Given observations X " tx 1 , . . .

x N u the goal is to maximize the log marginal likelihood log p θ pX q " ř i log p θ px i q where p θ pxq " ş p θ px, zqdz.

This marginal likelihood can be bounded from below by ř i E q φ pz|xiq rlog p θ px i |zq´log q φ pz|xiq p θ pzq s with q φ pz|x i q being a learned approximation to the true posterior p θ pz|x i q. This lower bound exhibits a similar information asymmetry between q and p as the one introduced between π and π 0 in the objective in eq. (3).

In particular, in the multi-task case discussed in section 3 with one task per episode, x i can be seen to take the role of the task, log ppx i |zq that of the task reward, qpz|x i q that of task conditional policy, and ppzq the default policy.

Therefore maximizing eq. (3) can then be thought of as learning a generative model of behaviors that can explain the solution to different tasks.

In practice the objective in eq. 3 can be optimized in different ways.

A simple approach is to perform alternating gradient ascent in π 0 and π.

Optimizing L with respect to π 0 amounts to supervised learning with π as the data distribution (distilling π into π 0 ).

Optimizing π given π 0 requires solving a regularized expected reward problem which can be achieved with a variety of algorithms BID44 BID48 BID11 BID13 BID12 .The specific algorithm choice in our experiments depends on the type of environment.

For the continuous control domains we use SVG(0) BID15 with experience replay and a modification for the KL regularized setting BID13 BID12 .

The SVG(0) algorithm learns stochastic policies by backpropagation from the action-value function.

We estimate the action value function using K-step returns and the Retrace operator for low-variance off-policy correction (see BID27 ; as well as BID13 ; BID39 ).

For discrete action spaces we use a batched actor-critic algorithm (see BID7 ).

The algorithm employs a learned state-value function and obtains value estimates for updating the value function and advantages for computing the policy gradient using K-step returns in combination with the V-trace operator for off-policy correction.

All algorithms are implemented in batched distributed fashion with a single learner and multiple actors.

In algorithm 1 we provide pseudo-code for actor-critic version of the algorithm with K-step returns.

Details of the off-policy versions of the algorithms for continuous and discrete action spaces can be found in the appendix (section D).

There are several well established connections between certain formulations of the reinforcement learning literature and concepts from the probabilistic modeling literature.

The formalisms are often closely related although derived from different intuitions, and with different intentions.

for t = 0, K, 2K, . . .

T do rollout partial trajectory: τ t:t`K " ps t , a t , r t . . .

r t`K q compute KL: DISPLAYFORM0 Default policy loss: DISPLAYFORM1 Maximum entropy reinforcement learning, stochastic optimal control, and related approaches build on the observation that some formulation of the reinforcement learning problem can be interpreted as exact or approximate variational inference in a probabilistic graphical model in which the reward function takes the role of log-likelihood (e.g. BID55 BID19 BID52 .

While the exact formulation and algorithms vary, they result in an entropy or KL regularized expected reward objective.

These algorithms were originally situated primarily in the robotics and control literature but there has been a recent surge in interest in deep reinforcement learning community (e.g. BID8 BID44 BID28 BID11 BID13 BID12 .Related but often seen as distinct is the familiy of expectation maximization policy search algorithms (e.g. BID35 BID36 BID23 BID26 BID4 BID0 .

These cast policy search as an alternating optimization problem similar to the EM algorithm for learning probabilistic models.

They differ in the specific implementation of the equivalents of the E and M steps; intuitively the default policy is repeatedly replaced by a new version of the policy.

The DISTRAL algorithm BID48 as well as the present paper can be seen as taking an intermediate position: unlike in the class of RL-as-inference algorithms the default policy is not fixed but learned, but unlike in the classical EM policy search the final result of the optimization remains regularized since the default policy is constrained relative to the policy.

As explained above this can be seen as analogous to the relative roles of learned model and observation specific posterior in fitting a generative model.

Similar to DISTRAL, Divide and Conquer BID9 learns an ensemble of policies, each specializing to a particular context, which are regularized towards one another via a symmetric KL penalty, with the behavior of the ensemble distilled to a single fixed policy.

In concurrent work BID10 propose an information bottleneck architecture for policies with latent variables that leads to a KL-regularized formulation similar to the one described in Appendix A.2.

The information bottleneck is implemented in latent space and the default policy is obtained by marginalization with a goal-agnostic prior.

An important feature of EM policy search and other policy gradient algorithms is the presence of a KL constraint that limits the relative change of the policy to some older version across iterations to control for the rate of change in the policy (e.g. BID43 BID15 BID45 BID16 BID29 .

The constraint can be implemented in different ways, and collectively the algorithms are often classified as "trust region" methods.

Note that for a KL regularized objective to be a trust region BID31 , additional assumptions need to hold.

In principle, as an optimization technique, the critical points of the KL regularized objective for some function f pθq have to be, provably, the same as for the non-regularized objective.

This is not trivial to show unless the trust region for step k is around θ k .

In our case, there is no such guarantee even if we remove the asymmetry in information between default policy and policy or make the default policy be an old copy of the policy.

Other related works motivated from an optimization perspective include Deep Mutual Learning BID54 applied in supervised learning, where KL-regularization is used with a learned prior that receives the same amount of information as the trained model.

introduces EWC to address catastrophic forgetting, where a second order Taylor expansion of the KL, in a KL-regularized objective, forces the main policy to stay close to solutions of previously encountered tasks.

also relies on a KL-regularized objective to ensure policies explored in a curriculum stay close to each other.

Conceptually distinct but formally closely related to maximum entropy and KL-regularized formulations are computational models of bounded rationality (e.g. BID50 BID32 BID47 BID40 BID49 which introduce information constraints to account for the agent's internal computational constraints on its ability to process information.

As discussed in section 4 the present formulation can be seen as a more general formulation of the idea.

In our experiments, we study the effect of using a learned default policy to regularize the behavior of our agents, across a wide range of environments spanning sparse and dense reward tasks.

In particular, we evaluate the impact of conditioning the default policy on various information sets x D on the learning dynamics, and evaluate the potential of pretrained default policies for transfer learning.

In these experiments, we consider two streams of information which are fed to our agents: task specific information (task) and proprioception (proprio), corresponding to walker (body) specific observations (joint angles etc.).

Walls task with humanoid, where the goal is avoid walls while running through a terrain.

We consider three walkers: jumping ball with 3 degrees of freedom (DoF) and 3 actuators; quadruped with 12 DoF and 8 actuators; humanoid with 28 DoF and 21 actuators.

The task is specified to the agent either via an additional feature vector (referred to as feature-tasks) or in the form of visual input (vision-task).

The tasks differ in the type of reward: in sparse reward tasks a non-zero reward is only given when a (sub-)goal is achieved (e.g. the target was reached); in dense reward tasks smoothly varying shaping reward is provided (e.g. negative distance to the target).

We consider the following tasks.

Walking task, a dense-reward task based on features.

The walker needs to move in one of four randomly sampled directions, with a fixed speed; the direction being resampled half-way through the episode.

Walls task, a dense-reward vision-task.

Here the walker has to traverse a corridor while avoiding walls.

Go to one of K targets task, a sparse-reward feature-based task.

The walker has to go to one of K randomly sampled targets.

For K=1, the target can either reappear within the episode (referred to as the moving target task) or the episode can end upon reaching the target.

Move one box to one of K targets, a sparse-reward feature-based-task.

The walker has to move a box to one of K targets, and optionally, go on to one of the remaining targets.

The latter is referred to as the move one box to one of K targets and go to another target).

Foraging in the maze task, a sparse-reward vision-task.

The walker collects apples in a maze.

FIG2 shows visualizations of the walkers and some of the tasks.

Refer to appendix C for more details.

Experimental Setup As baseline, we consider policies trained with standard entropy regularization.

When considering the full training objective of eq. 1, the default policy network shares the same structure as the agent's policy.

In both cases, hyper-parameters are optimized on a per-task basis.

We employ a distributed actor-learner architecture BID7 : actors execute recent copies of the policy and send data to a replay buffer of fixed size; while the learner samples short trajectory windows from the replay and computes updates to the policy, value, and default policy.

We experimented with a number of actors in t32, 64, 128, 256u (depending on the task) and a single learner.

Results with a single actor are presented in appendix B. Unless otherwise mentioned, we plot average episodic return as a function of the number of environment transitions processed by the learner 1 .

Each experiment is run with five random seeds.

For more details, see appendix D.2We consider three information sets passed to the default policy: proprioceptive, receiving only proprioceptive information; task-subset, receiving proprioceptive and a subset of task-specific information; full-information, receiving the same information as the policy.

Results for the sparse-reward tasks with complex walkers.

Left: go to moving target task with humanoid.

Center: foraging in the maze results with quadruped.

Right: moving one box to one of two targets and go to another target task with quadruped.

The legends denote additional to the proprioception, information passed to the default policy (except baseline, where we do not use default policy).The main finding of our experiments is that the default policy with limited task information provides considerable speed-up in terms of learner steps for the sparse-reward tasks with complex walkers (quadruped, humanoid).

The results on these tasks are presented in FIG3 .

More cases are covered in the appendix E.Overall, the proprioceptive default policy is very effective and gives the biggest gains in the majority of tasks.

Providing additional information to the default policy, leads to an improvement only in a small number of cases (figure 3, right and appendix E.3).

In these cases, the additional information (e.g. box position), adds useful inductive bias for the policy learning.

For the dense-reward tasks or for a simple walker body adding the default policy has limited or no effect (see appendix E.1, E.2).

We hypothesize that the absence of gain is due to the relative simplicity of the regular policy learning versus the KL-regularized setup.

In the case of dense-reward tasks the agent has a strong reward signal.

For simple walkers, the action space is too simple to require sophisticated exploration provided by the default policy.

Finally, with full information in the default policy, the optimal default policy would exactly copy the agent policy, which would not provide additional learning signal beyond the regular policy learning.

In all these cases, the default policy will not be forced to generalize across different contexts and hence not provide a meaningful regularization signal.

We analyze the agent behavior on the go to moving target task with a quadruped walker.

We illustrate the agent trajectory for this task in FIG4 , left.

The red dot corresponds to the agent starting position.

The green stars on the left and central figures correspond to the locations of the targets with Center: KL divergence from the agent policy to the proprioceptive default policy plotted over time for the same trajectory.

Right: Performance of the transfer on move one box to one of 3 targets task with quadruped.

The legend whether the default policy is learned or is transferred.

Furthermore, it specifies the task from which the default policy is transferred as well as additional information other than the proprioceptive information that the default policy is conditioned on, if any.blue numbers indicating the order of achieving the targets.

The yellow dots on the left and central curves indicate the segment (of 40 time steps) near the target.

In FIG4 , center, we show the KL divergence, KLrπ}π 0 s, from the agent policy to the proprioceptive default policy.

We observe that for the segments which are close to the target (yellow dots near green star), the value of the KL divergence is high.

In these segments the walker has to stop and turn in order to go to another target.

It represents a deviation from the standard, walking behavior, and we can observe it as spikes in the KL.

Furthermore, for the segments between the targets, e.g. 4 -> 5, the KL is much lower.

We additionally explore the possibility of reusing pretrained default policies to regularize learning on new tasks.

Our transfer task is moving one box to one of 2 targets and going to another target task with the quadruped.

We consider different default policies: GTT proprio: proprioceptive information only trained on going to moving target task (GTT); MB proprio: proprioceptive information only trained on moving one box to one target task (MB); MB box: similar MB proprio, but with box position information as additional input.

The results are given in figure 4, right.

We observe a significant improvement in learning speed transferring the pretrained default policies to the new task.

Performance improves as the trajectory distribution modeled by the default policy is closer to the one appropriate for the transfer task (compare GTT proprio with MB proprio; and MB proprio with MB box).

Ablative Analysis To gain deeper insights into our method, we compare different forms of regularization of the standard RL objective: entropy bonus -adding an entropy term Hpπp¨|s t qq to the per-timestep actor loss; entropy regularization -optimizing the objective (2); KL bonus -adding the KL-divergence term KL " πpa t |s t q}π 0 pa t |s t q ‰ from the agent policy to the default one to the per-timestep actor loss; KL-regularization -optimizing the objective (1); KL-regularization to the old policy -optimization of the objective 1 where regularization is done wrt.

an older version of the main policy (updated every 100 steps).

The default policy receives only proprioceptive information in these experiments.

The task is go to moving target.

As can be seen in FIG5 left, all three KL-based variants improve performance over the baseline, but regularizing against the information restricted default policy outperforms regularization against an old version of the policy.

FIG5 center, demonstrates that the benefit of the default policy depends on the reward structure.

When replacing the sparse reward with a dense shaping reward, proportional to the inverse distance from the walker to the target, our method and the baseline perform similarly, which is consistent with dense-reward results.

Finally, we assess the benefit of the KL-regularized objective 1 when used with an idealized default policy.

We repeat the go-to-target experiment with a pretrained default policy on the same task.

FIG5 right, shows a significant difference between the baseline and different regularization variants: using the pretrained default policy, learning the default policy alongside the main policy or using a pretrained expert (default policy with access to the full state).

This suggests that large gains may be achievable in situations when a good default policy is known a priori.

We performed the same analysis for the dense reward but we did not notice any gain.

The speed-up from regularizing to the pretrained expert is significant, however it corresponds to regularizing against an existing solution and can thus primarily be used as a method to speed-up the experiment cycles, as it was demonstrated in kickstarting framework BID42 ).Finally, we study impact of the direction of the KL in objective 1 on the learning dynamics.

Motivated by the work in policy distillation BID41 we flip the KL and use KL " π 0 pa t |s t q}πpa t |s t q ‰ instead of the described before KL " πpa t |s t q}π 0 pa t |s t q ‰ .

The experiments showed that there was no significant difference between these regularization schemes, which suggests that the idea of learned default policy can be viewed from student-teacher perspective, where default policy plays the role of the teacher.

This teacher can be used in a new task.

For the details, please refer to the appendix E.6.

We also evaluate our method on the DMLab-30 set of environments.

DMLab BID3 provides a suite of rich, first-person environments with tasks ranging from complex navigation and laser-tag to language-instructed goal finding.

Recent works on multitask training BID7 in this domain have used a form of batched-A2C with the V-trace algorithm to maximize an approximation of the entropy regularized objective described earlier, where the default policy is a uniform distribution over the actions.

Typically, the agent receives visual information at each step, along with an instruction channel used in a subset of tasks.

The agent receives no task identifier.

We adopt the architecture employed in previous work BID7 in which frames, past actions and rewards are passed successively through a deep residual network and LSTM, finally predicting a policy and value function.

All our experiments are tuned with population-based training BID18 .

Further details are provided in appendix D.1.DMLab exposes a large action space, specifically the cartesian product of atomic actions along seven axes.

However, commonly a human-engineered restricted subset of these actions is used at training and test time, simplifying the exploration problem for the agent.

For example, the used action space has a forward bias, with more actions resulting in the agent moving forward rather than backwards.

This helps with exploration in navigation tasks, where even a random walk can get the agent to move away from the starting position.

The uniform default policy is used on top of this human engineered small action space, where its semantics are clear.

In this work, we instead consider a much larger combinatorial space of actions.

We show that a pure uniform default policy is in fact unhelpful when human knowledge is removed from defining the right subset of actions to be uniform over, and the agent under-performs.

Learning the default policy, even in the extreme case when the default policy is not conditioned on any state information, helps recovering which actions are worth exploring and leads to the emergence of a useful action space without any hand engineering.

FIG6 shows the results of our experiments.

We consider a flat action space of 648 actions, each moving the agent in different spatial dimensions.

We run the agent from BID7 as baseline which is equivalent to considering the default policy to be a uniform distribution over the 648 actions, and three variants of our approach, where the default policy is actually learnt.

FORMULA0 ) that uses uniform distribution over actions as a default policy and three different possible default policies.

Center, the entropy for the vector default policy over learning.

Right, marginalized distribution over few actions of interest for the vector default policy.

For feed forward default policy, while the agent is recurrent, the default policy is not.

That is the policy π is conditioned on the full trace of observed states s 1 , a 1 , ..s t , while the default policy π 0 is conditioned only on the current frame a t´1 , s t .

Given that most of the 30 tasks considered require memory in order to be solvable, the default policy has to generalize over important task details.

LSTM default policy on the other hand, while being recurrent as the agent, it observes only the previous action a t´1 and does not receive any other state information.

In this instance, the default policy can only model the most likely actions given recent behaviour a 1 , ..a t´1 in absence of any visual stimuli.

For example, if previous actions are moving forward, the default policy might predict moving forward as the next action too.

This is because the agent usually moves consistently in any given direction in order to navigate efficiently.

Finally, the vector default policy refers to a default policy that is independent of actions and states (i.e. average behaviour over all possible histories of states and actions).Using any of the default policies outperforms the baseline, with LSTM default policy slightly underperforming compared with the others.

The vector default policy performs surprisingly well, highlighting that for DMLab defining a meaningful action space is extremely important for solving the task.

Our approach can provide a mechanism for identifying this action space without requiring human expert knowledge on the tasks.

Note in middle plot, FIG6 , that the entropy of the default policy over learning frames goes down, indicating that the default policy becomes peaky and is quite different from the uniform distribution which the baseline assumes.

Note that when running the same experiments with the original human-engineered smaller action space, no gains are observed.

This is similar to the continuous control setup, corresponding to changing the walker to a simple one and hence converting the task into a denser reward one.

Additionally, in figure 6 right, for the vector default policy, we show the probability of a few actions of interest by marginalizing over all other actions.

We notice that the agent has a tendency of moving forward 70%, while moving backwards is quite unlikely 10%.

The default policy discovers one element of the human defined action space, namely forward-bias which is quite useful for exploring the map.

The uniform bias would put same weight for moving forward as for moving backwards, making exploration harder.

We also note that the agent has a tendency to turn right and look right.

Given that each episode involves navigating a new sampled map, such a bias provides a meaningful exploration boost, as it suggest a following the wall strategy, where at any new intersection the agent always picks the same turning direction (e.g. right) to avoid moving in circles.

But as expected, since neither looking up or looking down provides any advantage, these actions are equally probable.

In this work we studied the influence of learning the default policy in the KL-regularized RL objective.

Specifically we looked at the scenario where we enforce information asymmetry between the default policy and the main one.

In the continuous control, we showed empirically that in the case of sparse-reward tasks with complex walkers, there is a significant speed-up of learning compared to the baseline.

In addition, we found that there was no significant gain in dense-reward tasks and/or with simple walkers.

Moreover, we demonstrated that significant gains can be achieved in the discrete action spaces.

We provided evidence that these gains are mostly due to the information asymmetry between the agent and the default policy.

Best results are obtained when the default policy sees only a subset of information, allowing it to learn task-agnostic behaviour.

Furthermore, these default polices can be reused to significantly speed-up learning on new tasks.

In this appendix we derive the connection between KL-regularized RL and information bottleneck in detail.

For simplicity we assume that x D t is empty, consider dependence only on current state s t and do not use subscript by t in detailed derivations for notational convenience.

We also apologize for some notational inconsistencies, and will fix them in a later draft.

DISPLAYFORM0 The simple formulation of the information bottleneck corresponds to maximizing reward while minimizing the per-timestep information between actions and state (or a subset of state, like the goal): DISPLAYFORM1 Upper-bounding the mutual information term: DISPLAYFORM2 DISPLAYFORM3 Thus DISPLAYFORM4 i.e. the problem turns into one of KL-regularized RL.

For policies with latent variables such as πpa|sq " ş πpa|zqπpz|sqdz we obtain:MIrA; Ss " ż πpa, sq log πpa|sqdads´ż πpaq log πpaqda (16) ď π ż πpa, sq log πpa|sqdads´ż πpaq logπ 0 paqdaas before.

We choose π 0 paq " ş πpa|zqπ 0 pzqdz, then: DISPLAYFORM0 and thusMIrA; Ss ď ż πpa, sq log qpa|sqdads´ż πpaq logπ 0 paqda (24) ď ż πpa, sq log πpa|sqdads´ż πpa, sq log πpa|sqdads´ż πpz, sq log π 0 pzq DISPLAYFORM1 Therefore: DISPLAYFORM2 Thus, the KL regularized objective discussed above can be seen as implementing an information bottleneck.

Different forms of the default policy correspond to restricting the information flow between different components of the interaction history (past states or observations), and to different approximations to the resulting mutual information penalties.

This perspective suggests two different interpretations of the KL regularized objective discussed above: We can see the role of the default policy implementing a way of restricting information flow between (past) states and (future) actions.

An alternative view, more consistent with the analogy between RL and probabilistic modeling invoked above is that of learning a "default" behavior that is independent of some aspect of the state. (Although the information theoretic view has recently gained more hold in the probabilistic modeling literature, too (e.g. BID1 ).

We use a distributed off-policy setup similar to BID38 .

There is one learner and multiple actors.

These are essentially the instantiations of the main agent used for different purposes.

Each actor is the main agent version which receives the copy of parameters from the learner and unrolls the trajectories in the environment, saving it to the replay buffer of fixed size 1e6.

The learner is the agent version which samples a batch of short trajectories windows (window size is defined by unroll length) from the replay buffer, calculates the gradients and updates the parameters.

The updated parameters are then communicated to the actors.

Such a setup speeds-up learning significantly and makes the final performance of the policy better.

We compare the performance of on go to moving target task with 1 and 32 actors.

From figure 7, we see that the effect of the default policy does not disappear when the number of actor decreases to 1, but the learning becomes much slower, noisier and weaker.

Walkers visualization is provided in figure 8 .

Below we give a detaatiled description of each continuous control task we studied.

Walking task.

Type.

Dense-reward feature-based-task.

Description.

Each half of the episode, a random direction among 4 (left, right, forward and backwards) is sampled.

Task information is specified via a one-hot encoding of the required direction.

The walker is required to move in this direction with the target speed v t and receives the reward r. Reward.

r " exp´| vcur´vt| 2 .

Technical details.

Target speed, v t " 3.

The episode length is 10 seconds.

For the humanoid task we use the absolute head height termination criteria: h ă 0.95.

Type.

Dense-reward vision-task.

Description.

Walker is required to run through a terrain and avoid the walls.

The task-specific information is a vision input.

It receives the reward r defined as a difference between the current walker speed v cur and the target speed v t along the direction of the track.

Reward.

r " exp´| Technical details.

Target speed, v t " 3.

The episode length is 45 seconds.

For the humanoid task we use the absolute head height termination criteria: h ă 0.9.Go to one of K single targets.

Type.

Sparse-reward feature-based-task.

Description.

On an infinite floor, there is a finite area of size 8x8 with K randomly placed targets.

The walker is also randomly placed in a finite area.

The walker's initial position is also randomly placed on the finite area.

The walker is required to one of the K targets, specified via command vector.

Once it achieves the target, the episode terminates and the walker receives the reward r. Reward.

r " 60.

Technical details.

The episode length is 20 seconds.

Go to one moving target.

Type.

Sparse-reward feature-based-task.

Description.

Similar to the previous one, but there is only one target and once the walker achieves it, the target reappears in a new random place.

The walker receives r for 10 consecutive steps staying on the target before the target reappears in a new random position.

Reward.

r " 1.

Technical details.

The episode length is 25 seconds.

Move one box to one of the K targets.

Type.

Sparse-reward feature-based-task.

Description.

There is a finite floor of size 3x3 padded with walls with K randomly placed targets and one box.

The walker is required to move this box to one of the specified targets.

Once the box is placed on the target, the episode terminates and the walker receives the reward r. Reward.

r " 60.

Technical details.

The episode length is 30 seconds.

Control timestep is 0.05 for quadruped and 0.025 for jumping ball.

Move one box to one of the K targets and go to another.

Type.

Sparse-reward feature-based-task.

Description.

Similar to the previous one, but the walker is also required to go to another target (which is different from the one where it must place the box on).

The walker receives the a r task for each task solved, and a r end if it solves both tasks.

The other parameters are the same.

Reward.

r task " 10, r end " 50.

Technical details.

Same as in the previous task.

Foraging in the maze.

Type.

Sparse-reward vision-task.

Description.

There is a maze with 8 apples which walker must collect.

For each apple, it receives reward r.

The episode terminates once the walker collects all the apples or the time is elapsed.

Reward.

r " 1.

Technical details.

The episode length is 90 seconds.

Control timestep is 0.025 for jumping ball, and 0.05 for quadruped.

Our agents run in off-policy regime sampling the trajectories from the replay buffer.

In practice, it means that the trajectories are coming from the behavior (replay buffer) policy π b , and thus, the correction must be applied (specified below).

Below we provide architecture details, baselines, hyperparmaeters as well as algorithm details for discrete and continuous control cases.

In discrete experiments, we use V-trace off-policy correction as in BID7 .

We reuse all the hyperparameters for DMLab from the mentionned paper.

At the top of that, we add default policy network and optimize the corresponding α parameter using population-base training.

The difference with the setup in BID7 is that they use the human prior over actions (table D.2 in the mentionned paper), which results in 9-dimensional action space.

In our work, we take the rough DMLab action space, consisting of all possible rotations, and moving forward/backward, and "fire" actions.

It results in the action space of dimension 648.

It make the learning much more challenging, as it has to explore in much larger space.

The agent network (see FIG0 is divided into actor and critic networks without any parameter sharing.

In the case of feature-based-task, the task-specific information is encoded by one layer MLP with ELU activations.

For the vision-task, we use a 3-layer ResNet BID14 .

The encoded task information is then concatenated with the proprioceptive information and passed to the agent network.

The actor network encodes a Gaussian policy, N pμ,σq, by employing a two-layer MLP, with mean µ and log variance log σ as outputs and applying the following processing procedures: DISPLAYFORM0 where f is a sigmoid function: DISPLAYFORM1 The critic network is a two-layer MLP and a linear readout.

The default policy network has the same structure as actor network, but receives a concatenation of the proprioceptive information with only a subset (potentially, empty) of a task-specific information.

There is no parameter sharing between the agent and the default policy.

ELU is used as activation everywhere.

The exact actor, critic and default policy network architectures are described below.

We tried to use LSTM for the default policy network instead of MLP, but did not see a difference.

We use separate optimizers and learning rates β π , β Q , β π 0 for the actor, critic and default policy networks correspondingly.

For each network (which we call online), we also define the target network, similar to the target Q-networks BID24 .

The target networks are updated are updated in a slower rate than the online ones by copying their parameters.

We assume that the trajectories are coming from the replay buffer B. To correct for being off-policy, we make use of the Retrace operator (see BID27 ).

This operator is applied to the Q function essentially introducing the importance weights.

We will note RQ the action for this operator.

Algorithm 2 is an off-policy version with retraced Q function of the initial algorithm 1.We use the same update period for actor and critic networks, P a and a different period for the default network P d .

The baseline is the agent network (see FIG0 ) without the default policy with an entropy bonus λ.

All the hyperparameters of the baseline are tuned for each task.

For each best baseline hyperparameters configuration, we tune the default policy parameters.

When we use the default policy, we do not have the entropy bonus.

Instead, we have a regularisation parameter α.

The other parameteres which we consider are: batch size, unroll length.

Below we provide the hyperparameters for each of the task.

The following default hyperparameters are used unless some particular one is specified.sampling actions from it and backpropagating through Q. In this algorithm, we learn a value function V using V-trace BID7 ) and the policy is updated using an off-policy corrected policy gradient with empirical returns.

The results for the dense-reward tasks are given in FIG9 .

We observe little difference of using default policy comparing to the baseline.

In the walls task, we also consider the default policy with global information, such as the orientation, position and speed in the global coordinates.

We do not observe a significant difference between using the default policy and the baseline.

The reason for this, we believe, is that the agent is being trained very quickly by seeing a strong reward signal, so the default policy cannot catch it up.

The results for the sparse reward tasks with jumping ball are given in FIG9 .

We see little difference of using default policy comparing to the baseline.

Our hypothesis consists in the fact that since the default policy affects the policy by regularizing the state-conditional action distribution (policy), for too simple actions space such is given here (3 actions), this effect is not strong enough.

Center: moving one box to one target.

Right: foraging in the maze.

The legends denote additional to the proprioception, information passed to the default policy (except baseline, where we do not use default policy).

In this section, we provide more results for the sparse reward tasks.

In FIG0 the results for going to one of K targets task with quadruped are presented.

The proprioceptive default policy gives significant gains comparing to others.

What interesting is that when the number of targets K increases, the baseline performance drops dramatically, whereas the proprioceptive default policy solve the task reliably.

Our hypothesis is that the default policy learns quickly the default walking behavior which becomes very helpful for the agent to explore the floor and search for the target.

FIG0 : Results for go to one of K targets tasks with quadruped.

Left: go to 1 target.

Center: go to one of 2 targets.

Right: go to one of 3 targets.

The legends denote additional to the proprioception, information passed to the default policy (except baseline, where we do not use default policy).We also provide the results for move box to one of K targets task, where K " 1, 2, 3, and move box to one of two targets task with go to another.

The results are given in figure 12.

Similar effect occurs here.

Starting from left, first: move one box to one of 2 targets with go to another.

Second: move one box to 1 target.

Third: move one box to one of 2 targets.

Forth: move one box to one of 3 targets.

The legends denote additional to the proprioception, information passed to the default policy (except baseline, where we do not use default policy).

In this section, we provide additional transfer experiment results for the range of the tasks.

They are given in FIG0 .

In the first two cases we see that proprioceptive default policy from the go to target task gives a significant boost to the performance comparing to the learning from scratch.

We also observe, that for the box pushing tasks, the default policy with the box position significantly speeds up learning comparing to other cases.

We believe it happens because this default policy learns the best default behavior for these tasks possible: going to the box and push it.

For the most complicated task, move one box to one of two targets and go to another one, 13, right, the box default policy makes a big difference: it makes the policy avoid being stuck in go to target behavior (line with reward of 10).Additional results for the transfer experiments are given in FIG0 .

We observe the same effect happening: whereas the baseline performance drops significantly, the agent with default policy stays E.5 ABLATION WALLS QUADRUPED Ablations for the walls quadruped are given in figure 14.

Center: move one box to one of two targets.

Right: move one box to one of two targets and go to another one.

The legend whether the default policy is learned or is transferred.

Furthermore, it specifies the task from which the default policy is transferred as well as additional information other than the proprioceptive information that the default policy is conditioned on, if any.

Published as a conference paper at ICLR 2019

The results for having the different order of the default policy in the KL-term (KLrπ||π 0 s or KLrπ 0 ||πs) for go to moving target task with quadruped walker are shown in FIG0 .

We use this term either in per time step actor loss (auxiliary loss) or as a regularizer by optimizing the objective 1 (with different order of KL).

We do not observe significant difference.

<|TLDR|>

@highlight

Limiting state information for the default policy can improvement performance, in a KL-regularized RL framework where both agent and default policy are optimized together